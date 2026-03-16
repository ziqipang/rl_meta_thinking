#!/usr/bin/env python3
"""Quantitative analysis of code-in-prompt rollouts: nb_only vs both_funcs."""

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
MODELS = ["nb_only_model", "both_model"]
MODEL_LABELS = {
    "nb_only_model": "NB-Only RL",
    "both_model": "Both RL",
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

# Eval sets that use nonbijective scoring
NB_EVAL_SETS = [
    "nonbijective_test",
    "nb_more_candidates",
    "nb_long_outputs",
    "nb_long_more_cand",
    "lenchange",
]

# Eval sets that use bijective scoring
BIJ_EVAL_SETS = ["bijective_test"]

# Eval sets that use single-answer inverse scoring
INVERSE_EVAL_SETS = ["bijective_inverse"]

# OOD eval sets (main story)
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

# Data source strings used in training (determines which scorer is used)
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
    # First try extracting from ground_truth ref_code (most reliable for code-in-prompt)
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
    """Return ground_truth as a JSON string (what the scorer expects)."""
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
    """Score a single response using the exact training scorer.

    Returns (score, acc) tuple matching wandb test_score/test_acc.
    """
    data_source = EVAL_DATA_SOURCES[eval_set]
    if "multi-" in data_source:
        return _codeio_multi.compute_score(response_str, gt_str, data_source)
    else:
        return _codeio.compute_score(response_str, gt_str, data_source)


def score_row(row, eval_set):
    """Score all responses for a row. Returns list of (score, acc) tuples."""
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
        r"two\s+different\s+(cases|scenarios|possibilities)",
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


def has_none_recognition(text):
    patterns = [
        r"\bNone\b", r"no\s+other\s+(valid\s+)?input",
        r"only\s+one\s+(valid\s+)?input", r"unique\s+preimage",
        r"bijective", r"one-to-one", r"exactly\s+one",
        r"no\s+second\s+(answer|input|preimage)",
    ]
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def has_code_tracing(text):
    """Detect if model traces through the function code step-by-step."""
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
    print("=" * 90)
    print("CODE-IN-PROMPT QUANTITATIVE ANALYSIS: NB-Only RL vs Both RL")
    print("=" * 90)
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

    # ── SECTION 1: Overall accuracy (using exact training scorer) ───────────
    print("=" * 90)
    print("SECTION 1: Overall Accuracy (codeio_multi scorer — matches wandb test_acc)")
    print("=" * 90)
    print()

    # all_row_scores[(model, eval_set)] = list of lists of (score, acc) tuples
    all_row_scores = {}

    header = f"{'Eval Set':<22}"
    for m in MODELS:
        header += f" {MODEL_LABELS[m]+' Score':>14} {MODEL_LABELS[m]+' Acc':>14}"
    print(header)
    print("-" * len(header))

    for eval_set in EVAL_SETS:
        row_str = f"{eval_set:<22}"
        for model in MODELS:
            df = load_parquet(model, eval_set)
            if df is None:
                row_str += f" {'N/A':>14} {'N/A':>14}"
                continue

            row_tuples = []
            for idx, row in df.iterrows():
                try:
                    s = score_row(row, eval_set)  # list of (score, acc) tuples
                except Exception:
                    s = [(0.0, 0.0)] * len(row["responses"])
                row_tuples.append(s)

            all_row_scores[(model, eval_set)] = row_tuples

            # Mean score and mean acc (matching wandb test_score and test_acc)
            flat_scores = [sc for row_s in row_tuples for sc, ac in row_s if sc >= 0]
            flat_accs = [ac for row_s in row_tuples for sc, ac in row_s if sc >= 0]
            mean_score = np.mean(flat_scores) if flat_scores else 0.0
            mean_acc = np.mean(flat_accs) if flat_accs else 0.0
            row_str += f" {mean_score:>14.4f} {mean_acc:>14.4f}"

        print(row_str)

    print()

    # ── SECTION 2: Function-level breakdown for OOD evals ────────────────────
    print("=" * 90)
    print("SECTION 2: Function-Level Breakdown")
    print("=" * 90)
    print()

    for eval_set in EVAL_SETS:
        print(f"\n--- {eval_set} ---")
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

            print(f"  Model: {MODEL_LABELS[model]}")
            print(f"  {'Function':<12} {'N':>6} {'MeanScore':>10} {'MeanAcc':>10}")
            print(f"  {'-'*42}")

            sorted_funcs = sorted(func_tuples.keys(), key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else 999)
            for func in sorted_funcs:
                sl = func_tuples[func]
                n = len(sl)
                flat_sc = [sc for row_s in sl for sc, ac in row_s if sc >= 0]
                flat_ac = [ac for row_s in sl for sc, ac in row_s if sc >= 0]
                ms = np.mean(flat_sc) if flat_sc else 0.0
                ma = np.mean(flat_ac) if flat_ac else 0.0
                print(f"  {func:<12} {n:>6} {ms:>10.4f} {ma:>10.4f}")
            print()

    # ── SECTION 3: Reasoning patterns ────────────────────────────────────────
    print("=" * 90)
    print("SECTION 3: Reasoning Strategy Prevalence & Correlation with Accuracy")
    print("=" * 90)
    print()

    strategies = [
        ("Case Analysis", has_case_analysis),
        ("Enumeration/Trial", has_enumeration),
        ("Systematic Inversion", has_systematic_inversion),
        ("Hypothesis Verification", has_hypothesis_verification),
        ("Code Tracing", has_code_tracing),
    ]

    for eval_set in EVAL_SETS:
        print(f"--- {eval_set} ---")
        for model in MODELS:
            df = load_parquet(model, eval_set)
            if df is None:
                continue

            print(f"  Model: {MODEL_LABELS[model]}")
            print(f"  {'Strategy':<25} {'Rate':>8} {'Acc_with':>10} {'Acc_without':>12} {'Lift':>8}")
            print(f"  {'-'*65}")

            for strat_name, strat_fn in strategies:
                with_accs = []
                without_accs = []
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
                        else:
                            without_accs.append(ac)

                rate = len(with_accs) / total if total > 0 else 0
                acc_w = np.mean(with_accs) if with_accs else 0.0
                acc_wo = np.mean(without_accs) if without_accs else 0.0
                lift = acc_w - acc_wo

                print(f"  {strat_name:<25} {rate:>8.4f} {acc_w:>10.4f} {acc_wo:>12.4f} {lift:>+8.4f}")

            print()

    # ── SECTION 4: Bijective test - None recognition ─────────────────────────
    print("=" * 90)
    print("SECTION 4: Bijective Test - Single Preimage Recognition")
    print("=" * 90)
    print()

    for model in MODELS:
        df = load_parquet(model, "bijective_test")
        if df is None:
            continue

        total = 0
        none_recognized = 0
        single_answer = 0
        two_answers = 0
        more_answers = 0
        correct_with_none = 0
        correct_with_two = 0

        for idx, row in df.iterrows():
            gt_str = get_gt_str(row)
            for resp in row["responses"]:
                total += 1
                answer_text = extract_answer_text(resp)
                candidates = parse_candidates(answer_text)
                n_cand = len(candidates)

                recognizes_none = has_none_recognition(resp)
                if recognizes_none:
                    none_recognized += 1

                if n_cand <= 1:
                    single_answer += 1
                elif n_cand == 2:
                    two_answers += 1
                else:
                    more_answers += 1

                try:
                    sc, ac = score_response(resp, gt_str, "bijective_test")
                except Exception:
                    sc, ac = 0.0, 0.0

                if ac > 0:
                    if n_cand <= 1 or recognizes_none:
                        correct_with_none += 1
                    else:
                        correct_with_two += 1

        print(f"Model: {MODEL_LABELS[model]} (N={total} responses)")
        print(f"  Single answer (0-1 candidates):  {single_answer:>6} ({single_answer/total:.4f})")
        print(f"  Two answers:                     {two_answers:>6} ({two_answers/total:.4f})")
        print(f"  3+ answers:                      {more_answers:>6} ({more_answers/total:.4f})")
        print(f"  Recognizes uniqueness/None:       {none_recognized:>6} ({none_recognized/total:.4f})")
        print(f"  Correct & recognizes uniqueness:  {correct_with_none:>6}")
        print(f"  Correct & gives 2 answers:        {correct_with_two:>6}")
        print()

    # ── SECTION 5: Error categorization ──────────────────────────────────────
    print("=" * 90)
    print("SECTION 5: Error Categorization (Incorrect Responses)")
    print("=" * 90)
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

            print(f"  Model: {MODEL_LABELS[model]} (correct={total_right}, wrong={total_wrong})")
            if total_wrong > 0:
                for cat in sorted(error_cats.keys()):
                    cnt = error_cats[cat]
                    print(f"    {cat:<25} {cnt:>6} ({cnt/total_wrong:.4f})")
            print()

    # ── SECTION 6: Response length statistics ────────────────────────────────
    print("=" * 90)
    print("SECTION 6: Response Length Statistics (characters)")
    print("=" * 90)
    print()

    header = f"{'Eval Set':<22}"
    for m in MODELS:
        header += f" {MODEL_LABELS[m]+' Mean':>14} {MODEL_LABELS[m]+' Med':>12}"
    print(header)
    print("-" * len(header))

    for eval_set in EVAL_SETS:
        row_str = f"{eval_set:<22}"
        for model in MODELS:
            df = load_parquet(model, eval_set)
            if df is None:
                row_str += f" {'N/A':>14} {'N/A':>12}"
                continue
            lengths = [len(r) for _, row in df.iterrows() for r in row["responses"]]
            lengths = np.array(lengths)
            row_str += f" {np.mean(lengths):>14.0f} {np.median(lengths):>12.0f}"
        print(row_str)

    print()

    # Response length by correctness
    print("Response length by correctness:")
    for eval_set in EVAL_SETS:
        print(f"  --- {eval_set} ---")
        for model in MODELS:
            df = load_parquet(model, eval_set)
            if df is None:
                continue

            correct_lens = []
            wrong_lens = []
            for idx, row in df.iterrows():
                try:
                    tuples = score_row(row, eval_set)
                except Exception:
                    tuples = [(0.0, 0.0)] * len(row["responses"])
                for resp, (sc, ac) in zip(row["responses"], tuples):
                    if sc < 0:
                        continue
                    if ac > 0:
                        correct_lens.append(len(resp))
                    else:
                        wrong_lens.append(len(resp))

            c_mean = np.mean(correct_lens) if correct_lens else 0
            w_mean = np.mean(wrong_lens) if wrong_lens else 0
            ratio = w_mean / c_mean if c_mean > 0 else 0
            print(f"    {MODEL_LABELS[model]}: correct={c_mean:.0f} wrong={w_mean:.0f} ratio={ratio:.2f}x (n_correct={len(correct_lens)}, n_wrong={len(wrong_lens)})")
        print()

    # ── SECTION 7: Differential analysis ─────────────────────────────────────
    print("=" * 90)
    print("SECTION 7: Differential Analysis (Both RL vs NB-Only RL)")
    print("=" * 90)
    print()

    for eval_set in EVAL_SETS:
        df_both = load_parquet("both_model", eval_set)
        df_nb = load_parquet("nb_only_model", eval_set)

        if df_both is None or df_nb is None:
            continue

        if len(df_both) != len(df_nb):
            print(f"{eval_set}: different lengths ({len(df_both)} vs {len(df_nb)}), skipping pairwise")
            continue

        both_wins = 0
        nb_wins = 0
        both_right = 0
        both_wrong = 0
        n = len(df_both)

        for idx in range(n):
            row_b = df_both.iloc[idx]
            row_n = df_nb.iloc[idx]
            try:
                sb = score_row(row_b, eval_set)
                sn = score_row(row_n, eval_set)
            except Exception:
                continue

            b_correct = any(ac > 0 for sc, ac in sb if sc >= 0)
            n_correct = any(ac > 0 for sc, ac in sn if sc >= 0)

            if b_correct and n_correct:
                both_right += 1
            elif b_correct:
                both_wins += 1
            elif n_correct:
                nb_wins += 1
            else:
                both_wrong += 1

        print(f"{eval_set} (N={n}):")
        print(f"  Both correct:        {both_right:>6} ({both_right/n:.4f})")
        print(f"  Both-RL only:        {both_wins:>6} ({both_wins/n:.4f})")
        print(f"  NB-Only only:        {nb_wins:>6} ({nb_wins/n:.4f})")
        print(f"  Both wrong:          {both_wrong:>6} ({both_wrong/n:.4f})")
        print()

    # ── SECTION 8: Strategy co-occurrence on key OOD evals ───────────────────
    print("=" * 90)
    print("SECTION 8: Strategy Co-occurrence on OOD Evals")
    print("=" * 90)
    print()

    for eval_set in OOD_EVAL_SETS:
        print(f"--- {eval_set} ---")
        for model in MODELS:
            df = load_parquet(model, eval_set)
            if df is None:
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

            print(f"  Model: {MODEL_LABELS[model]} (total={total})")
            for key in sorted(combos.keys()):
                c = combos[key]["count"]
                corr = combos[key]["correct"]
                acc = corr / c if c > 0 else 0
                print(f"    {key:<25} {c:>6} ({c/total:.4f})  acc={acc:.4f}")
            print()

    print("=" * 90)
    print("QUANTITATIVE ANALYSIS COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    main()
