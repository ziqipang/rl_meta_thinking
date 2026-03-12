#!/usr/bin/env python3
"""Quantitative analysis of v3_mix rollouts: SFT baseline vs nb_only vs both_funcs."""

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
MODELS = ["sft_baseline", "nb_only_model", "both_model"]
MODEL_LABELS = {
    "sft_baseline": "SFT Base",
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

# ─── Import scoring from reward module ────────────────────────────────────────
_CODEIO_FUNC_ENV = None
try:
    from projects.datasets import create_string_data as _csd
    _CODEIO_FUNC_ENV = {
        name: getattr(_csd, name)
        for name in dir(_csd)
        if re.fullmatch(r"(func_\d{2}|nb_\d{2})", name)
    }
    _num_bijective = int(getattr(_csd, "NUM_BIJECTIVE_FUNCS", 24))
    _num_nb_v2 = int(getattr(_csd, "NUM_NB_V2_FUNCS", 25))
    for _i in range(_num_nb_v2):
        _nb_name = f"nb_{_i:02d}"
        _func_name = f"func_{(_num_bijective + _i):02d}"
        if hasattr(_csd, _nb_name):
            _CODEIO_FUNC_ENV[_func_name] = getattr(_csd, _nb_name)
    for _helper in [
        "_digits", "_undigits", "scr_reverse", "scr_rotate1",
        "scr_swappairs", "scr_oddeven", "scr_multperm3",
        "mix_A_neighbor_sum_affine", "mix_B_triple_weight_affine",
        "mix_C_runsum_affine", "mix_D_nonlinear_mod5_embed",
    ]:
        if hasattr(_csd, _helper):
            _CODEIO_FUNC_ENV[_helper] = getattr(_csd, _helper)
except Exception:
    pass


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


def get_func_name(prompt):
    if isinstance(prompt, list):
        text = " ".join(m.get("content", "") for m in prompt if isinstance(m, dict))
    elif isinstance(prompt, str):
        text = prompt
    else:
        return None
    m = re.search(r'return\s+(func_\d+)\s*\(', text)
    if m:
        return m.group(1)
    m2 = re.search(r'\b(func_\d+)\b', text)
    return m2.group(1) if m2 else None


def get_gt(row):
    rm = row["reward_model"]
    gt = rm["ground_truth"]
    if isinstance(gt, str):
        return json.loads(gt)
    return gt


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


def _is_close(a, b):
    if isinstance(a, str) and isinstance(b, str):
        return a.strip() == b.strip()
    try:
        return abs(float(a) - float(b)) < 1e-6
    except (ValueError, TypeError):
        return str(a).strip() == str(b).strip()


def build_ref_fn(ref_code, funcname="main_solution"):
    ns = {"__builtins__": __builtins__}
    if _CODEIO_FUNC_ENV:
        ns.update(_CODEIO_FUNC_ENV)
    exec(ref_code, ns, ns)
    return ns[funcname]


def score_nb_preimage(candidates, gt):
    ref_output = gt["ref_output"]
    ref_code = gt["ref_code"]
    funcname = gt.get("funcname", "main_solution")
    num_answers = get_num_answers(gt)
    if not candidates:
        return 0.0
    try:
        ref_fn = build_ref_fn(ref_code, funcname)
    except Exception:
        return -1.0
    seen = set()
    unique = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    correct = 0
    for cand in unique[:max(5, num_answers)]:
        try:
            out = ref_fn(str(cand))
            if _is_close(out, ref_output):
                correct += 1
        except Exception:
            continue
    return min(correct, num_answers) / float(num_answers)


def score_bij_preimage(candidates, gt):
    ref_output = gt.get("ref_output")
    ref_code = gt.get("ref_code")
    funcname = gt.get("funcname", "main_solution")
    num_answers = get_num_answers(gt)
    if not candidates:
        return 0.0
    try:
        ref_fn = build_ref_fn(ref_code, funcname)
    except Exception:
        return -1.0
    seen = set()
    unique = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    correct = 0
    for cand in unique[:max(5, num_answers)]:
        try:
            out = ref_fn(str(cand))
            if _is_close(out, ref_output):
                correct += 1
        except Exception:
            continue
    return min(correct, num_answers) / float(num_answers)


def score_inverse_single(response_str, gt):
    answer_text = extract_answer_text(response_str)
    if not answer_text:
        return 0.0
    ref_input = gt.get("ref_input")
    if isinstance(ref_input, dict):
        ref_input = str(ref_input.get("x", ""))
    else:
        ref_input = str(ref_input)
    candidates = parse_candidates(answer_text)
    for c in candidates:
        if _is_close(str(c), ref_input):
            return 1.0
    return 0.0


def score_response(response_str, gt, eval_set):
    if eval_set in INVERSE_EVAL_SETS:
        return score_inverse_single(response_str, gt)

    answer_text = extract_answer_text(response_str)
    candidates = parse_candidates(answer_text)

    scoring_mode = None
    gt_info = gt.get("gt_info")
    if isinstance(gt_info, dict):
        scoring_mode = gt_info.get("multi_response_scoring")

    if scoring_mode == "bijective_inverse":
        return score_bij_preimage(candidates, gt)
    elif scoring_mode == "nonbijective_preimage":
        return score_nb_preimage(candidates, gt)
    else:
        # Default: use exec-based scoring
        return score_nb_preimage(candidates, gt)


def score_row(row, eval_set):
    gt = get_gt(row)
    responses = row["responses"]
    return [score_response(r, gt, eval_set) for r in responses]


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
    """Detect brute-force enumeration / trial-and-error."""
    patterns = [
        r"let'?s?\s+try", r"trying\s+(input|value|string|x\s*=)",
        r"test\s+input", r"try\s+(another|different|other)",
        r"trial\s+and\s+error", r"brute\s*force",
        r"let me try", r"check\s+if\s+\d+\s+works",
        r"if\s+x\s*=\s*\d+", r"suppose\s+x\s*=",
    ]
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def has_systematic_inversion(text):
    """Detect systematic/algebraic inversion reasoning."""
    patterns = [
        r"invert", r"inverse", r"reverse\s+(the\s+)?operation",
        r"undo", r"work\s+backwards?", r"backward",
        r"starting\s+from\s+the\s+output", r"given.*output.*find.*input",
        r"reverse\s+engineer", r"step\s+by\s+step.*reverse",
        r"apply.*in\s+reverse", r"going\s+backwards?",
    ]
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def has_hypothesis_verification(text):
    """Detect hypothesis-verification approach."""
    patterns = [
        r"let'?s?\s+verify", r"let'?s?\s+check", r"verify\s+this",
        r"check.*correct", r"confirm", r"to\s+verify",
        r"plug\s+(it\s+)?back", r"substitut",
    ]
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def has_none_recognition(text):
    """Detect if model recognizes 'None' as valid answer (for bijective with num_answers=2)."""
    patterns = [
        r"\bNone\b", r"no\s+other\s+(valid\s+)?input",
        r"only\s+one\s+(valid\s+)?input", r"unique\s+preimage",
        r"bijective", r"one-to-one", r"exactly\s+one",
        r"no\s+second\s+(answer|input|preimage)",
    ]
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def classify_error_type(response_str, gt, eval_set):
    """Classify error type for incorrect responses."""
    answer_text = extract_answer_text(response_str)

    if answer_text is None:
        return "format_error"  # No <answer> tag

    candidates = parse_candidates(answer_text)
    if not candidates:
        return "empty_answer"  # <answer> exists but no numbers

    # Check reasoning patterns
    has_inv = has_systematic_inversion(response_str)
    has_enum = has_enumeration(response_str)
    has_verify = has_hypothesis_verification(response_str)

    # Check if response is truncated (hit max length)
    if len(response_str) > 30000:  # Very long, likely truncated
        return "truncated"

    if has_inv and not has_enum:
        return "inversion_error"  # Used correct strategy but made computation error
    elif has_enum and not has_inv:
        return "enumeration_failure"  # Tried guessing, failed
    elif has_inv and has_enum:
        return "mixed_strategy_error"
    else:
        return "other_error"


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 90)
    print("v3_mix QUANTITATIVE ANALYSIS: SFT Baseline vs NB-Only RL vs Both RL")
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

    # ── SECTION 1: Overall accuracy ──────────────────────────────────────────
    print("=" * 90)
    print("SECTION 1: Overall Accuracy (Pass@1 and Mean Score)")
    print("=" * 90)
    print()

    all_row_scores = {}

    header = f"{'Eval Set':<22}"
    for m in MODELS:
        header += f" {MODEL_LABELS[m]+' P@1':>14} {MODEL_LABELS[m]+' Mean':>14}"
    print(header)
    print("-" * len(header))

    for eval_set in EVAL_SETS:
        row_str = f"{eval_set:<22}"
        for model in MODELS:
            df = load_parquet(model, eval_set)
            if df is None:
                row_str += f" {'N/A':>14} {'N/A':>14}"
                continue

            scores = []
            for idx, row in df.iterrows():
                try:
                    s = score_row(row, eval_set)
                except Exception:
                    s = [0.0] * len(row["responses"])
                scores.append(s)

            all_row_scores[(model, eval_set)] = scores

            pass1 = np.mean([1.0 if any(s > 0 for s in ss) else 0.0 for ss in scores])
            flat = [s for ss in scores for s in ss if s >= 0]
            mean_s = np.mean(flat) if flat else 0.0
            row_str += f" {pass1:>14.4f} {mean_s:>14.4f}"

        print(row_str)

    print()

    # Per-response strict accuracy
    print("Per-response strict accuracy (score == 1.0):")
    header2 = f"{'Eval Set':<22}"
    for m in MODELS:
        header2 += f" {MODEL_LABELS[m]:>14}"
    print(header2)
    print("-" * len(header2))

    for eval_set in EVAL_SETS:
        row_str = f"{eval_set:<22}"
        for model in MODELS:
            scores = all_row_scores.get((model, eval_set))
            if scores is None:
                row_str += f" {'N/A':>14}"
                continue
            flat = [s for ss in scores for s in ss if s >= 0]
            acc = np.mean([1.0 if abs(s - 1.0) < 1e-6 else 0.0 for s in flat]) if flat else 0.0
            row_str += f" {acc:>14.4f}"
        print(row_str)

    print()

    # ── SECTION 2: Function-level breakdown ──────────────────────────────────
    print("=" * 90)
    print("SECTION 2: Function-Level Breakdown (nonbijective_test)")
    print("=" * 90)
    print()

    for model in MODELS:
        df = load_parquet(model, "nonbijective_test")
        if df is None:
            continue

        func_scores = defaultdict(list)
        for idx, row in df.iterrows():
            func = get_func_name(row["prompt"])
            if not func:
                func = "unknown"
            try:
                s = score_row(row, "nonbijective_test")
            except Exception:
                s = [0.0] * len(row["responses"])
            func_scores[func].append(s)

        print(f"Model: {MODEL_LABELS[model]}")
        print(f"{'Function':<12} {'N':>6} {'Pass@1':>8} {'Mean':>8} {'StrictAcc':>10}")
        print("-" * 48)

        sorted_funcs = sorted(func_scores.keys(), key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else 999)
        for func in sorted_funcs:
            sl = func_scores[func]
            n = len(sl)
            p1 = np.mean([1.0 if any(s > 0 for s in ss) else 0.0 for ss in sl])
            flat = [s for ss in sl for s in ss if s >= 0]
            ms = np.mean(flat) if flat else 0.0
            sa = np.mean([1.0 if abs(s-1.0) < 1e-6 else 0.0 for s in flat]) if flat else 0.0
            print(f"{func:<12} {n:>6} {p1:>8.4f} {ms:>8.4f} {sa:>10.4f}")
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
    ]

    for eval_set in ["nonbijective_test", "bijective_test", "bijective_inverse"]:
        print(f"--- {eval_set} ---")
        for model in MODELS:
            df = load_parquet(model, eval_set)
            if df is None:
                continue

            print(f"  Model: {MODEL_LABELS[model]}")
            print(f"  {'Strategy':<25} {'Rate':>8} {'Acc_with':>10} {'Acc_without':>12} {'Lift':>8}")
            print(f"  {'-'*65}")

            for strat_name, strat_fn in strategies:
                with_scores = []
                without_scores = []
                total = 0

                for idx, row in df.iterrows():
                    try:
                        s = score_row(row, eval_set)
                    except Exception:
                        s = [0.0] * len(row["responses"])

                    for resp, sc in zip(row["responses"], s):
                        if sc < 0:
                            continue
                        total += 1
                        if strat_fn(resp):
                            with_scores.append(sc)
                        else:
                            without_scores.append(sc)

                rate = len(with_scores) / total if total > 0 else 0
                acc_w = np.mean([1.0 if s > 0 else 0.0 for s in with_scores]) if with_scores else 0.0
                acc_wo = np.mean([1.0 if s > 0 else 0.0 for s in without_scores]) if without_scores else 0.0
                lift = acc_w - acc_wo

                print(f"  {strat_name:<25} {rate:>8.4f} {acc_w:>10.4f} {acc_wo:>12.4f} {lift:>+8.4f}")

            print()

    # ── SECTION 4: Bijective test - None recognition ─────────────────────────
    print("=" * 90)
    print("SECTION 4: Bijective Test - Single Preimage Recognition")
    print("=" * 90)
    print()
    print("Bijective functions have exactly 1 preimage, but model is asked for 2 answers.")
    print("Ideal behavior: output 1 correct answer + 'None' for the second.")
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
            gt = get_gt(row)
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

                # Check correctness
                try:
                    sc = score_response(resp, gt, "bijective_test")
                except Exception:
                    sc = 0.0

                if sc > 0:
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

    for eval_set in ["nonbijective_test", "bijective_test", "bijective_inverse"]:
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
                    scores = score_row(row, eval_set)
                except Exception:
                    scores = [0.0] * len(row["responses"])

                for resp, sc in zip(row["responses"], scores):
                    if sc < 0:
                        continue
                    if sc > 0:
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

            b_correct = any(s > 0 for s in sb if s >= 0)
            n_correct = any(s > 0 for s in sn if s >= 0)

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

    print("=" * 90)
    print("QUANTITATIVE ANALYSIS COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    main()
