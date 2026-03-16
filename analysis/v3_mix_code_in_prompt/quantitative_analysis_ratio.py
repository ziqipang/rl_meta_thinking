#!/usr/bin/env python3
"""Quantitative analysis of ratio experiments: comparing all 4 code-in-prompt models.

Models:
  - NB-Only RL:  trained on non-bijective only (1:0 bij:nb)
  - Both RL:     trained on bij + nb at 1:1 ratio
  - Bij3:NB1 RL: trained on bij + nb at 3:1 ratio
  - Bij1:NB3 RL: trained on bij + nb at 1:3 ratio
"""

import json
import os
import re
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

BASE = str(Path(__file__).resolve().parent / "rollouts")
MODELS = ["nb_only_model", "bij1_nb3_model", "both_model", "bij3_nb1_model"]
MODEL_LABELS = {
    "nb_only_model": "NB-Only (0:1)",
    "bij1_nb3_model": "Bij1:NB3",
    "both_model": "Both (1:1)",
    "bij3_nb1_model": "Bij3:NB1",
}

EVAL_SETS = [
    "nonbijective_test",
    "bijective_test",
    "bijective_inverse",
    "nb_more_candidates",
    "nb_long_outputs",
    "nb_long_more_cand",
    "lenchange",
]

NB_EVAL_SETS = [
    "nonbijective_test",
    "nb_more_candidates",
    "nb_long_outputs",
    "nb_long_more_cand",
    "lenchange",
]

OOD_EVAL_SETS = [
    "bijective_test",
    "bijective_inverse",
    "nb_long_outputs",
    "nb_long_more_cand",
    "nb_more_candidates",
    "lenchange",
]

# ─── Import the actual training scorers to match wandb test_acc ──────────────
from verl.utils.reward_score import codeio_multi as _codeio_multi
from verl.utils.reward_score import codeio as _codeio

EVAL_DATA_SOURCES = {
    "nonbijective_test": "codeio-nb-multi-preimage-incomplete-depth1",
    "bijective_test": "codeio-multi-preimage-incomplete-depth1",
    "bijective_inverse": "codeio-backward-incomplete-depth1",
    "nb_long_outputs": "codeio-nb-multi-preimage-eval-long-outputs-v2-code-depth1",
    "nb_more_candidates": "codeio-nb-multi-preimage-eval-more-candidates-v2-code-depth1",
    "nb_long_more_cand": "codeio-nb-multi-preimage-eval-long-outputs-more-candidates-v2-code-depth1",
    "lenchange": "codeio-nb-multi-preimage-lenchange-eval-depth1",
}

# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_parquet(model, eval_set):
    path = os.path.join(BASE, model, f"{eval_set}.parquet")
    if not os.path.exists(path):
        return None
    return pd.read_parquet(path)


def extract_answer_text(response_str):
    m = re.search(r"<answer>(.*?)</answer>", response_str, flags=re.DOTALL | re.IGNORECASE)
    if not m:
        return None
    inner = m.group(1).replace('"', '')
    inner = re.sub(r"\s+", "", inner)
    return inner


def parse_candidates(answer_text):
    if not answer_text:
        return []
    return re.findall(r"\d+", answer_text)


def get_func_name(prompt, row=None):
    if row is not None:
        try:
            gt = get_gt(row)
            ref_code = gt.get("ref_code", "")
            m = re.search(r'def\s+(func_\d+|nb_\d+|lenop_\d+)\s*\(', ref_code)
            if m:
                return m.group(1)
        except Exception:
            pass
    if isinstance(prompt, list):
        text = " ".join(m.get("content", "") for m in prompt if isinstance(m, dict))
    elif isinstance(prompt, str):
        text = prompt
    else:
        return None
    m = re.search(r'return\s+(func_\d+)\s*\(', text)
    if m:
        return m.group(1)
    m2 = re.search(r'def\s+(func_\d+|nb_\d+|lenop_\d+)\s*\(', text)
    if m2:
        return m2.group(1)
    m3 = re.search(r'\b(func_\d+)\b', text)
    return m3.group(1) if m3 else None


def get_gt(row):
    rm = row["reward_model"]
    gt = rm["ground_truth"]
    if isinstance(gt, str):
        return json.loads(gt)
    return gt


def get_gt_str(row):
    rm = row["reward_model"]
    gt = rm["ground_truth"]
    if isinstance(gt, str):
        return gt
    return json.dumps(gt)


def get_num_answers(gt):
    gt_info = gt.get("gt_info")
    if isinstance(gt_info, dict):
        n = gt_info.get("num_answers")
        if n and int(n) > 0:
            return int(n)
    n = gt.get("num_answers")
    if n:
        return int(n)
    return 2


def score_response(response_str, gt_str, eval_set):
    data_source = EVAL_DATA_SOURCES[eval_set]
    if "multi-" in data_source:
        return _codeio_multi.compute_score(response_str, gt_str, data_source)
    else:
        return _codeio.compute_score(response_str, gt_str, data_source)


def score_row(row, eval_set):
    gt_str = get_gt_str(row)
    responses = row["responses"]
    return [score_response(r, gt_str, eval_set) for r in responses]


# ─── Reasoning pattern detectors ─────────────────────────────────────────────

def has_case_analysis(text):
    patterns = [
        r"case\s*[12]", r"Case\s*[12]",
        r"two\s+(possible\s+)?cases", r"Two\s+(possible\s+)?cases",
        r"consider\s+(the\s+)?(two|both|each)\s+case",
        r"there\s+are\s+two\s+(possible\s+)?(cases|scenarios|solutions)",
    ]
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def has_enumeration(text):
    patterns = [
        r"let'?s?\s+try", r"trying\s+(input|value|string|x\s*=)",
        r"test\s+input", r"try\s+(another|different|other)",
        r"trial\s+and\s+error", r"brute\s*force",
        r"let me try", r"check\s+if\s+\d+\s+works",
        r"if\s+x\s*=\s*\d+", r"suppose\s+x\s*=",
    ]
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def has_systematic_inversion(text):
    patterns = [
        r"invert", r"inverse", r"reverse\s+(the\s+)?operation",
        r"undo", r"work\s+backwards?", r"backward",
        r"starting\s+from\s+the\s+output", r"given.*output.*find.*input",
        r"reverse\s+engineer", r"step\s+by\s+step.*reverse",
        r"apply.*in\s+reverse", r"going\s+backwards?",
    ]
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def has_hypothesis_verification(text):
    patterns = [
        r"let'?s?\s+verify", r"let'?s?\s+check", r"verify\s+this",
        r"check.*correct", r"confirm", r"to\s+verify",
        r"plug\s+(it\s+)?back", r"substitut",
    ]
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def has_code_tracing(text):
    patterns = [
        r"step\s*\d+\s*:", r"line\s+by\s+line",
        r"executing.*function", r"trace\s+through",
        r"running.*code", r"state\s*=", r"result\s*=",
        r"after\s+(this\s+)?(line|step|operation)",
        r"the\s+function\s+(does|performs|computes|applies)",
        r"first,?\s+the\s+function", r"looking\s+at\s+the\s+code",
        r"according\s+to\s+the\s+code",
    ]
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def has_none_recognition(text):
    patterns = [
        r"\bNone\b", r"no\s+other\s+(valid\s+)?input",
        r"only\s+one\s+(valid\s+)?input", r"unique\s+preimage",
        r"bijective", r"one-to-one", r"exactly\s+one",
        r"no\s+second\s+(answer|input|preimage)",
    ]
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def classify_error_type(response_str, gt, eval_set):
    answer_text = extract_answer_text(response_str)
    if answer_text is None:
        return "format_error"
    candidates = parse_candidates(answer_text)
    if not candidates:
        return "empty_answer"

    has_inv = has_systematic_inversion(response_str)
    has_enum = has_enumeration(response_str)

    if len(response_str) > 30000:
        return "truncated"

    if has_inv and not has_enum:
        return "inversion_error"
    elif has_enum and not has_inv:
        return "enumeration_failure"
    elif has_inv and has_enum:
        return "mixed_strategy_error"
    else:
        return "other_error"


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 110)
    print("RATIO EXPERIMENT ANALYSIS: Effect of Bijective:Non-Bijective Training Ratio")
    print("=" * 110)
    print()
    print("Models compared (ordered by increasing bijective fraction):")
    for m in MODELS:
        print(f"  {MODEL_LABELS[m]}")
    print()

    # Check available data
    available = {}
    for model in MODELS:
        for eval_set in EVAL_SETS:
            df = load_parquet(model, eval_set)
            available[(model, eval_set)] = df is not None
            if df is not None:
                print(f"  Found: {model}/{eval_set} ({len(df)} rows)")

    print()

    # ── SECTION 1: Overall accuracy ──────────────────────────────────────────
    print("=" * 110)
    print("SECTION 1: Overall Accuracy (codeio_multi scorer — matches wandb test_acc)")
    print("=" * 110)
    print()

    all_row_scores = {}

    header = f"{'Eval Set':<22}"
    for m in MODELS:
        header += f" {MODEL_LABELS[m]:>14}"
    print(header)
    print("-" * len(header))

    for eval_set in EVAL_SETS:
        row_str = f"{eval_set:<22}"
        for model in MODELS:
            df = load_parquet(model, eval_set)
            if df is None:
                row_str += f" {'N/A':>14}"
                continue

            row_tuples = []
            for idx, row in df.iterrows():
                try:
                    s = score_row(row, eval_set)
                except Exception:
                    s = [(0.0, 0.0)] * len(row["responses"])
                row_tuples.append(s)

            all_row_scores[(model, eval_set)] = row_tuples
            flat_accs = [ac for row_s in row_tuples for sc, ac in row_s if sc >= 0]
            mean_acc = np.mean(flat_accs) if flat_accs else 0.0
            row_str += f" {mean_acc:>14.4f}"

        print(row_str)

    print()

    # Also print Score (includes format bonus for legacy evals)
    print("Score (includes 10% format bonus for legacy OOD evals):")
    header = f"{'Eval Set':<22}"
    for m in MODELS:
        header += f" {MODEL_LABELS[m]:>14}"
    print(header)
    print("-" * len(header))

    for eval_set in EVAL_SETS:
        row_str = f"{eval_set:<22}"
        for model in MODELS:
            if (model, eval_set) not in all_row_scores:
                row_str += f" {'N/A':>14}"
                continue
            row_tuples = all_row_scores[(model, eval_set)]
            flat_scores = [sc for row_s in row_tuples for sc, ac in row_s if sc >= 0]
            mean_score = np.mean(flat_scores) if flat_scores else 0.0
            row_str += f" {mean_score:>14.4f}"
        print(row_str)

    print()

    # ── SECTION 2: Per-prompt consistency ────────────────────────────────────
    print("=" * 110)
    print("SECTION 2: Per-Prompt Consistency (fraction of prompts where all 4 responses correct)")
    print("=" * 110)
    print()

    for eval_set in EVAL_SETS:
        print(f"--- {eval_set} ---")
        header = f"  {'Metric':<30}"
        for m in MODELS:
            header += f" {MODEL_LABELS[m]:>14}"
        print(header)

        # Compute per-prompt stats
        model_stats = {}
        for model in MODELS:
            df = load_parquet(model, eval_set)
            if df is None:
                model_stats[model] = None
                continue

            all_wrong = 0
            all_correct = 0
            n = len(df)

            for idx, row in df.iterrows():
                try:
                    tuples = score_row(row, eval_set)
                except Exception:
                    tuples = [(0.0, 0.0)] * len(row["responses"])

                accs = [ac for sc, ac in tuples if sc >= 0]
                if all(a == 0 for a in accs):
                    all_wrong += 1
                if all(a > 0 for a in accs):
                    all_correct += 1

            model_stats[model] = {
                "n": n,
                "all_wrong": all_wrong / n if n > 0 else 0,
                "all_correct": all_correct / n if n > 0 else 0,
            }

        row_aw = f"  {'All 4 wrong':<30}"
        row_ac = f"  {'All 4 correct':<30}"
        for model in MODELS:
            if model_stats[model] is None:
                row_aw += f" {'N/A':>14}"
                row_ac += f" {'N/A':>14}"
            else:
                row_aw += f" {model_stats[model]['all_wrong']:>14.4f}"
                row_ac += f" {model_stats[model]['all_correct']:>14.4f}"
        print(row_aw)
        print(row_ac)
        print()

    # ── SECTION 3: Function-level breakdown ──────────────────────────────────
    print("=" * 110)
    print("SECTION 3: Function-Level Breakdown (Acc)")
    print("=" * 110)
    print()

    for eval_set in EVAL_SETS:
        print(f"\n--- {eval_set} ---")

        # Collect all functions across models
        all_funcs = set()
        func_data = {}  # (model, func) -> (n, mean_acc)

        for model in MODELS:
            df = load_parquet(model, eval_set)
            if df is None:
                continue

            func_tuples = defaultdict(list)
            for idx, row in df.iterrows():
                func = get_func_name(row["prompt"], row=row)
                if not func:
                    func = "unknown"
                try:
                    s = score_row(row, eval_set)
                except Exception:
                    s = [(0.0, 0.0)] * len(row["responses"])
                func_tuples[func].append(s)

            for func, sl in func_tuples.items():
                all_funcs.add(func)
                n = len(sl)
                flat_ac = [ac for row_s in sl for sc, ac in row_s if sc >= 0]
                ma = np.mean(flat_ac) if flat_ac else 0.0
                func_data[(model, func)] = (n, ma)

        sorted_funcs = sorted(all_funcs, key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else 999)

        header = f"  {'Function':<12} {'N':>6}"
        for m in MODELS:
            header += f" {MODEL_LABELS[m]:>14}"
        print(header)
        print(f"  {'-' * (18 + 14 * len(MODELS))}")

        for func in sorted_funcs:
            n_val = max((func_data.get((m, func), (0, 0))[0] for m in MODELS), default=0)
            row_str = f"  {func:<12} {n_val:>6}"
            for model in MODELS:
                if (model, func) in func_data:
                    _, ma = func_data[(model, func)]
                    row_str += f" {ma:>14.4f}"
                else:
                    row_str += f" {'N/A':>14}"
            print(row_str)
        print()

    # ── SECTION 4: Reasoning strategy prevalence ─────────────────────────────
    print("=" * 110)
    print("SECTION 4: Reasoning Strategy Prevalence & Accuracy Correlation")
    print("=" * 110)
    print()

    strategies = [
        ("Enumeration/Trial", has_enumeration),
        ("Systematic Inversion", has_systematic_inversion),
        ("Code Tracing", has_code_tracing),
        ("Case Analysis", has_case_analysis),
        ("Verification", has_hypothesis_verification),
    ]

    for eval_set in EVAL_SETS:
        print(f"--- {eval_set} ---")
        for strat_name, strat_fn in strategies:
            row_str = f"  {strat_name:<25}"
            for model in MODELS:
                df = load_parquet(model, eval_set)
                if df is None:
                    row_str += f"  {'N/A':>20}"
                    continue

                with_accs = []
                total = 0
                for idx, row in df.iterrows():
                    try:
                        s = score_row(row, eval_set)
                    except Exception:
                        s = [(0.0, 0.0)] * len(row["responses"])
                    for resp, (sc, ac) in zip(row["responses"], s):
                        if sc < 0:
                            continue
                        total += 1
                        if strat_fn(resp):
                            with_accs.append(ac)

                rate = len(with_accs) / total if total > 0 else 0
                acc_w = np.mean(with_accs) if with_accs else 0.0
                row_str += f"  {rate:>5.1%}(acc={acc_w:.3f})"
            print(row_str)
        print()

    # ── SECTION 5: Bijective test - None recognition ─────────────────────────
    print("=" * 110)
    print("SECTION 5: Bijective Test - Single Preimage Recognition")
    print("=" * 110)
    print()

    for model in MODELS:
        df = load_parquet(model, "bijective_test")
        if df is None:
            continue

        total = 0
        single_answer = 0
        none_recognized = 0

        for idx, row in df.iterrows():
            for resp in row["responses"]:
                total += 1
                answer_text = extract_answer_text(resp)
                candidates = parse_candidates(answer_text)
                if len(candidates) <= 1:
                    single_answer += 1
                if has_none_recognition(resp):
                    none_recognized += 1

        print(f"Model: {MODEL_LABELS[model]} (N={total} responses)")
        print(f"  Single answer (0-1 cand):   {single_answer:>6} ({single_answer/total:.4f})")
        print(f"  Recognizes uniqueness/None: {none_recognized:>6} ({none_recognized/total:.4f})")
        print()

    # ── SECTION 6: Error categorization ──────────────────────────────────────
    print("=" * 110)
    print("SECTION 6: Error Categorization (Incorrect Responses)")
    print("=" * 110)
    print()

    for eval_set in EVAL_SETS:
        print(f"--- {eval_set} ---")
        for model in MODELS:
            df = load_parquet(model, eval_set)
            if df is None:
                continue

            error_cats = defaultdict(int)
            total_wrong = 0
            total_right = 0

            for idx, row in df.iterrows():
                gt = get_gt(row)
                try:
                    tuples = score_row(row, eval_set)
                except Exception:
                    tuples = [(0.0, 0.0)] * len(row["responses"])

                for resp, (sc, ac) in zip(row["responses"], tuples):
                    if sc < 0:
                        continue
                    if ac > 0:
                        total_right += 1
                    else:
                        total_wrong += 1
                        err_type = classify_error_type(resp, gt, eval_set)
                        error_cats[err_type] += 1

            print(f"  {MODEL_LABELS[model]}: correct={total_right}, wrong={total_wrong}")
            if total_wrong > 0:
                for cat in sorted(error_cats.keys()):
                    cnt = error_cats[cat]
                    print(f"    {cat:<25} {cnt:>6} ({cnt/total_wrong:.4f})")
        print()

    # ── SECTION 7: Response length ───────────────────────────────────────────
    print("=" * 110)
    print("SECTION 7: Response Length Statistics (characters)")
    print("=" * 110)
    print()

    header = f"{'Eval Set':<22}"
    for m in MODELS:
        header += f" {MODEL_LABELS[m]:>14}"
    print(header)
    print("-" * len(header))

    for eval_set in EVAL_SETS:
        row_str = f"{eval_set:<22}"
        for model in MODELS:
            df = load_parquet(model, eval_set)
            if df is None:
                row_str += f" {'N/A':>14}"
                continue
            lengths = [len(r) for _, row in df.iterrows() for r in row["responses"]]
            row_str += f" {np.mean(lengths):>14.0f}"
        print(row_str)

    print()

    # ── SECTION 8: Strategy co-occurrence on key OOD evals ───────────────────
    print("=" * 110)
    print("SECTION 8: Strategy Co-occurrence on Key OOD Evals")
    print("=" * 110)
    print()

    for eval_set in OOD_EVAL_SETS:
        print(f"--- {eval_set} ---")
        header = f"  {'Combo':<25}"
        for m in MODELS:
            header += f" {MODEL_LABELS[m]:>20}"
        print(header)

        # Collect combos for all models
        model_combos = {}
        for model in MODELS:
            df = load_parquet(model, eval_set)
            if df is None:
                model_combos[model] = None
                continue

            combos = defaultdict(lambda: {"count": 0, "correct": 0})
            total = 0

            for idx, row in df.iterrows():
                try:
                    tuples = score_row(row, eval_set)
                except Exception:
                    tuples = [(0.0, 0.0)] * len(row["responses"])

                for resp, (sc, ac) in zip(row["responses"], tuples):
                    if sc < 0:
                        continue
                    total += 1
                    inv = has_systematic_inversion(resp)
                    enum = has_enumeration(resp)
                    key = f"{'inv' if inv else 'no_inv'}+{'enum' if enum else 'no_enum'}"
                    combos[key]["count"] += 1
                    if ac > 0:
                        combos[key]["correct"] += 1

            model_combos[model] = (combos, total)

        all_keys = set()
        for model in MODELS:
            if model_combos[model] is not None:
                all_keys.update(model_combos[model][0].keys())

        for key in sorted(all_keys):
            row_str = f"  {key:<25}"
            for model in MODELS:
                if model_combos[model] is None:
                    row_str += f" {'N/A':>20}"
                else:
                    combos, total = model_combos[model]
                    c = combos[key]["count"]
                    corr = combos[key]["correct"]
                    acc = corr / c if c > 0 else 0
                    pct = c / total if total > 0 else 0
                    row_str += f" {pct:>5.1%} acc={acc:.3f}"
                    # pad to 20 chars
            print(row_str)
        print()

    # ── SECTION 9: Differential analysis (pairwise) ──────────────────────────
    print("=" * 110)
    print("SECTION 9: Differential Analysis — Bij3:NB1 vs Bij1:NB3 (direct ratio comparison)")
    print("=" * 110)
    print()

    for eval_set in EVAL_SETS:
        df_31 = load_parquet("bij3_nb1_model", eval_set)
        df_13 = load_parquet("bij1_nb3_model", eval_set)

        if df_31 is None or df_13 is None:
            continue

        if len(df_31) != len(df_13):
            print(f"{eval_set}: different lengths, skipping pairwise")
            continue

        bij3_wins = 0
        bij1_wins = 0
        both_right = 0
        both_wrong = 0
        n = len(df_31)

        for idx in range(n):
            row_31 = df_31.iloc[idx]
            row_13 = df_13.iloc[idx]
            try:
                s31 = score_row(row_31, eval_set)
                s13 = score_row(row_13, eval_set)
            except Exception:
                continue

            r31 = any(ac > 0 for sc, ac in s31 if sc >= 0)
            r13 = any(ac > 0 for sc, ac in s13 if sc >= 0)

            if r31 and r13:
                both_right += 1
            elif r31:
                bij3_wins += 1
            elif r13:
                bij1_wins += 1
            else:
                both_wrong += 1

        print(f"{eval_set} (N={n}):")
        print(f"  Both correct:    {both_right:>6} ({both_right/n:.4f})")
        print(f"  Bij3:NB1 only:   {bij3_wins:>6} ({bij3_wins/n:.4f})")
        print(f"  Bij1:NB3 only:   {bij1_wins:>6} ({bij1_wins/n:.4f})")
        print(f"  Both wrong:      {both_wrong:>6} ({both_wrong/n:.4f})")
        print()

    # Also do Bij3:NB1 vs Both(1:1)
    print("--- Bij3:NB1 vs Both(1:1) ---")
    for eval_set in EVAL_SETS:
        df_31 = load_parquet("bij3_nb1_model", eval_set)
        df_11 = load_parquet("both_model", eval_set)

        if df_31 is None or df_11 is None:
            continue

        if len(df_31) != len(df_11):
            continue

        bij3_wins = 0
        both_wins = 0
        both_right = 0
        both_wrong = 0
        n = len(df_31)

        for idx in range(n):
            row_31 = df_31.iloc[idx]
            row_11 = df_11.iloc[idx]
            try:
                s31 = score_row(row_31, eval_set)
                s11 = score_row(row_11, eval_set)
            except Exception:
                continue

            r31 = any(ac > 0 for sc, ac in s31 if sc >= 0)
            r11 = any(ac > 0 for sc, ac in s11 if sc >= 0)

            if r31 and r11:
                both_right += 1
            elif r31:
                bij3_wins += 1
            elif r11:
                both_wins += 1
            else:
                both_wrong += 1

        print(f"  {eval_set}: Both-correct={both_right/n:.4f} Bij3-only={bij3_wins/n:.4f} Both(1:1)-only={both_wins/n:.4f} Both-wrong={both_wrong/n:.4f}")

    print()

    print("=" * 110)
    print("QUANTITATIVE ANALYSIS COMPLETE")
    print("=" * 110)


if __name__ == "__main__":
    main()
