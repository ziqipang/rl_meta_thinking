#!/usr/bin/env python3
"""Comprehensive quantitative analysis of rollout data."""

import json
import os
import re
import sys
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE = "/home/ubuntu/rl_meta_thinking/analysis/rollouts"
MODELS = ["both_model", "nb_only_model"]
EVAL_SETS = [
    "nonbijective_test",
    "bijective_test",
    "nb_more_candidates",
    "nb_long_outputs",
    "lenchange",
    "inverse_test",
]

# ─── Helpers ───────────────────────────────────────────────────────────────────

def load_parquet(model, eval_set):
    path = os.path.join(BASE, model, f"{eval_set}.parquet")
    if not os.path.exists(path):
        return None
    return pd.read_parquet(path)


def extract_answer_text(response_str):
    """Extract text inside <answer>...</answer>."""
    m = re.search(r"<answer>(.*?)</answer>", response_str, flags=re.DOTALL | re.IGNORECASE)
    if not m:
        return None
    inner = m.group(1).replace('"', '')
    inner = re.sub(r"\s+", "", inner)
    return inner


def parse_candidates(answer_text):
    """Parse digit-string candidates from answer text."""
    if not answer_text:
        return []
    return re.findall(r"\d+", answer_text)


def get_func_name(prompt_text):
    """Extract nb_XX or func_XX from prompt."""
    m = re.search(r'(nb_\d+|func_\d+)', prompt_text)
    return m.group(1) if m else None


def get_gt(row):
    """Parse ground_truth from reward_model column."""
    return json.loads(row["reward_model"]["ground_truth"])


def get_scoring_mode(gt):
    gt_info = gt.get("gt_info")
    if isinstance(gt_info, dict):
        return gt_info.get("multi_response_scoring", None)
    return None


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


def build_ref_fn(ref_code, funcname):
    """Build callable from ref_code string."""
    ns = {}
    exec(ref_code, ns)
    return ns[funcname]


def score_nonbijective(candidates, gt):
    """Score for nonbijective_preimage: fraction of required distinct correct answers."""
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
    for cand in unique:
        try:
            out = ref_fn(str(cand))
            if _is_close(out, ref_output):
                correct += 1
                if correct >= num_answers:
                    break
        except Exception:
            continue
    submitted = len(candidates)
    if num_answers <= 0 or submitted <= 0 or correct <= 0:
        return 0.0
    return min(correct, num_answers) / max(num_answers, submitted)


def score_bijective(candidates, gt):
    """Score for bijective_inverse: need to find the unique preimage."""
    ref_input_obj = gt.get("ref_input")
    if not isinstance(ref_input_obj, dict) or "x" not in ref_input_obj:
        return -1.0
    ref_input = str(ref_input_obj["x"])
    if not candidates:
        return 0.0
    seen = set()
    unique = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    correct = 1 if any(_is_close(str(c), ref_input) for c in unique) else 0
    submitted = len(candidates)
    if submitted <= 0 or correct <= 0:
        return 0.0
    return min(correct, 1) / max(1, submitted)


def score_inverse(response_str, gt):
    """Score for standard inverse (single answer)."""
    answer_text = extract_answer_text(response_str)
    if not answer_text:
        return 0.0
    ref_input = str(gt["ref_input"]["x"]) if isinstance(gt.get("ref_input"), dict) else str(gt.get("ref_input", ""))
    # For single-answer inverse, the answer should match ref_input
    candidates = parse_candidates(answer_text)
    if not candidates:
        return 0.0
    # Check if any candidate matches
    for c in candidates:
        if _is_close(str(c), ref_input):
            return 1.0
    return 0.0


def _is_close(a, b):
    """Compare two values for closeness."""
    if isinstance(a, str) and isinstance(b, str):
        return a.strip() == b.strip()
    try:
        return abs(float(a) - float(b)) < 1e-6
    except (ValueError, TypeError):
        return str(a).strip() == str(b).strip()


def score_row(row, eval_set):
    """Score all responses for a single row. Returns list of scores (one per response)."""
    gt = get_gt(row)
    scoring_mode = get_scoring_mode(gt)
    responses = row["responses"]
    scores = []

    for resp in responses:
        if eval_set == "inverse_test":
            s = score_inverse(resp, gt)
        else:
            answer_text = extract_answer_text(resp)
            candidates = parse_candidates(answer_text)
            if scoring_mode == "bijective_inverse":
                s = score_bijective(candidates, gt)
            elif scoring_mode == "nonbijective_preimage":
                s = score_nonbijective(candidates, gt)
            else:
                # Legacy or lenchange (no multi_response_scoring in gt_info)
                # Use nonbijective scoring as default for NB sets
                if eval_set in ("nonbijective_test", "nb_more_candidates", "nb_long_outputs", "lenchange"):
                    s = score_nonbijective(candidates, gt)
                else:
                    s = score_nonbijective(candidates, gt)
        scores.append(s)
    return scores


def has_case_analysis(text):
    """Detect if response uses case analysis (case 1/2, consider...cases, etc.)."""
    patterns = [
        r"case\s*[12]",
        r"Case\s*[12]",
        r"two\s+(possible\s+)?cases",
        r"Two\s+(possible\s+)?cases",
        r"consider\s+(the\s+)?(two|both|each)\s+case",
        r"there\s+are\s+two\s+(possible\s+)?(cases|scenarios|solutions)",
        r"two\s+different\s+(cases|scenarios|possibilities)",
        r"state.*=.*or.*state.*=",
        r"\\text\{state\}.*=.*\\text\{state\}.*=",
    ]
    for p in patterns:
        if re.search(p, text, re.IGNORECASE):
            return True
    return False


def has_trial_and_error(text):
    """Detect if response uses trial and error approach."""
    patterns = [
        r"let'?s?\s+try",
        r"trying\s+(input|value|string)",
        r"test\s+input",
        r"try\s+(another|different|other)",
        r"trial\s+and\s+error",
        r"brute\s*force",
        r"trying\s+\w+\s*=",
        r"let me try",
    ]
    for p in patterns:
        if re.search(p, text, re.IGNORECASE):
            return True
    return False


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: Pass@1 for every (model, eval_set)
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 80)
print("SECTION 1: Pass@1 Scores for Every (Model, Eval Set) Combination")
print("=" * 80)
print()
print("Pass@1 = fraction of problems where at least one response scores > 0")
print("Mean score = average score across all (problem, response) pairs")
print()

results_pass1 = {}
results_mean = {}
all_row_scores = {}  # (model, eval_set) -> list of list of scores

header = f"{'Eval Set':<25} {'both_model Pass@1':>20} {'both_model Mean':>18} {'nb_only Pass@1':>18} {'nb_only Mean':>15}"
print(header)
print("-" * len(header))

for eval_set in EVAL_SETS:
    row_parts = [f"{eval_set:<25}"]
    for model in MODELS:
        df = load_parquet(model, eval_set)
        if df is None:
            row_parts.append(f"{'N/A':>20}" if model == "both_model" else f"{'N/A':>18}")
            row_parts.append(f"{'N/A':>18}" if model == "both_model" else f"{'N/A':>15}")
            results_pass1[(model, eval_set)] = None
            results_mean[(model, eval_set)] = None
            continue

        all_scores = []
        for idx, row in df.iterrows():
            try:
                scores = score_row(row, eval_set)
            except Exception as e:
                scores = [0.0] * len(row["responses"])
            all_scores.append(scores)

        all_row_scores[(model, eval_set)] = all_scores

        # Pass@1: fraction of problems with at least one response scoring > 0
        pass1 = np.mean([1.0 if any(s > 0 for s in scores) else 0.0 for scores in all_scores])
        # Mean score across all responses
        flat_scores = [s for scores in all_scores for s in scores if s >= 0]
        mean_score = np.mean(flat_scores) if flat_scores else 0.0

        results_pass1[(model, eval_set)] = pass1
        results_mean[(model, eval_set)] = mean_score

        if model == "both_model":
            row_parts.append(f"{pass1:>20.4f}")
            row_parts.append(f"{mean_score:>18.4f}")
        else:
            row_parts.append(f"{pass1:>18.4f}")
            row_parts.append(f"{mean_score:>15.4f}")

    print("".join(row_parts))

print()

# Also compute per-response pass rates (pass@1 with single response = avg response score > 0)
print("Per-response accuracy (fraction of individual responses with score > 0):")
header2 = f"{'Eval Set':<25} {'both_model':>15} {'nb_only_model':>15}"
print(header2)
print("-" * len(header2))
for eval_set in EVAL_SETS:
    parts = [f"{eval_set:<25}"]
    for model in MODELS:
        scores_list = all_row_scores.get((model, eval_set))
        if scores_list is None:
            parts.append(f"{'N/A':>15}")
            continue
        flat = [s for ss in scores_list for s in ss if s >= 0]
        acc = np.mean([1.0 if s > 0 else 0.0 for s in flat]) if flat else 0.0
        parts.append(f"{acc:>15.4f}")
    print("".join(parts))

print()
# Also show "strict" pass@1 (score == 1.0)
print("Strict accuracy (fraction of individual responses with score == 1.0):")
header3 = f"{'Eval Set':<25} {'both_model':>15} {'nb_only_model':>15}"
print(header3)
print("-" * len(header3))
for eval_set in EVAL_SETS:
    parts = [f"{eval_set:<25}"]
    for model in MODELS:
        scores_list = all_row_scores.get((model, eval_set))
        if scores_list is None:
            parts.append(f"{'N/A':>15}")
            continue
        flat = [s for ss in scores_list for s in ss if s >= 0]
        acc = np.mean([1.0 if abs(s - 1.0) < 1e-6 else 0.0 for s in flat]) if flat else 0.0
        parts.append(f"{acc:>15.4f}")
    print("".join(parts))

print()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: Function-level breakdown on nonbijective_test
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 80)
print("SECTION 2: Function-Level Breakdown on nonbijective_test")
print("=" * 80)
print()

for model in MODELS:
    df = load_parquet(model, "nonbijective_test")
    if df is None:
        continue

    func_scores = defaultdict(list)
    for idx, row in df.iterrows():
        prompt_text = row["prompt"][0]["content"]
        func = get_func_name(prompt_text)
        if func is None:
            func = "unknown"
        try:
            scores = score_row(row, "nonbijective_test")
        except Exception:
            scores = [0.0] * len(row["responses"])
        func_scores[func].append(scores)

    print(f"Model: {model}")
    print(f"{'Function':<15} {'N_problems':>12} {'Pass@1':>10} {'Mean Score':>12} {'Strict Acc':>12}")
    print("-" * 63)

    sorted_funcs = sorted(func_scores.keys(), key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else 999)
    for func in sorted_funcs:
        scores_list = func_scores[func]
        n = len(scores_list)
        pass1 = np.mean([1.0 if any(s > 0 for s in ss) else 0.0 for ss in scores_list])
        flat = [s for ss in scores_list for s in ss if s >= 0]
        mean_s = np.mean(flat) if flat else 0.0
        strict = np.mean([1.0 if abs(s - 1.0) < 1e-6 else 0.0 for s in flat]) if flat else 0.0
        print(f"{func:<15} {n:>12} {pass1:>10.4f} {mean_s:>12.4f} {strict:>12.4f}")

    print()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: Case analysis rates and correlation with correctness
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 80)
print("SECTION 3: Case Analysis Rates & Correlation with Correctness")
print("=" * 80)
print()

NB_EVAL_SETS = ["nonbijective_test", "nb_more_candidates", "nb_long_outputs", "lenchange"]

for model in MODELS:
    print(f"Model: {model}")
    print(f"{'Eval Set':<25} {'Case Rate':>10} {'Acc w/ Case':>12} {'Acc w/o Case':>13} {'Lift':>8}")
    print("-" * 70)

    for eval_set in NB_EVAL_SETS:
        df = load_parquet(model, eval_set)
        if df is None:
            print(f"{eval_set:<25} {'N/A':>10}")
            continue

        case_scores = []
        no_case_scores = []
        total_responses = 0

        for idx, row in df.iterrows():
            try:
                scores = score_row(row, eval_set)
            except Exception:
                scores = [0.0] * len(row["responses"])

            for resp, s in zip(row["responses"], scores):
                if s < 0:
                    continue
                total_responses += 1
                if has_case_analysis(resp):
                    case_scores.append(s)
                else:
                    no_case_scores.append(s)

        case_rate = len(case_scores) / total_responses if total_responses > 0 else 0
        acc_case = np.mean([1.0 if s > 0 else 0.0 for s in case_scores]) if case_scores else 0.0
        acc_no_case = np.mean([1.0 if s > 0 else 0.0 for s in no_case_scores]) if no_case_scores else 0.0
        lift = acc_case - acc_no_case

        print(f"{eval_set:<25} {case_rate:>10.4f} {acc_case:>12.4f} {acc_no_case:>13.4f} {lift:>+8.4f}")

    print()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: Trial-and-error rates and correlation with correctness
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 80)
print("SECTION 4: Trial-and-Error Rates & Correlation with Correctness")
print("=" * 80)
print()

for model in MODELS:
    print(f"Model: {model}")
    print(f"{'Eval Set':<25} {'T&E Rate':>10} {'Acc w/ T&E':>12} {'Acc w/o T&E':>13} {'Lift':>8}")
    print("-" * 70)

    for eval_set in NB_EVAL_SETS:
        df = load_parquet(model, eval_set)
        if df is None:
            print(f"{eval_set:<25} {'N/A':>10}")
            continue

        te_scores = []
        no_te_scores = []
        total_responses = 0

        for idx, row in df.iterrows():
            try:
                scores = score_row(row, eval_set)
            except Exception:
                scores = [0.0] * len(row["responses"])

            for resp, s in zip(row["responses"], scores):
                if s < 0:
                    continue
                total_responses += 1
                if has_trial_and_error(resp):
                    te_scores.append(s)
                else:
                    no_te_scores.append(s)

        te_rate = len(te_scores) / total_responses if total_responses > 0 else 0
        acc_te = np.mean([1.0 if s > 0 else 0.0 for s in te_scores]) if te_scores else 0.0
        acc_no_te = np.mean([1.0 if s > 0 else 0.0 for s in no_te_scores]) if no_te_scores else 0.0
        lift = acc_te - acc_no_te

        print(f"{eval_set:<25} {te_rate:>10.4f} {acc_te:>12.4f} {acc_no_te:>13.4f} {lift:>+8.4f}")

    print()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: Response length statistics
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 80)
print("SECTION 5: Response Length Statistics (chars) on NB Test Sets")
print("=" * 80)
print()

for model in MODELS:
    print(f"Model: {model}")
    print(f"{'Eval Set':<25} {'Mean':>10} {'Median':>10} {'Std':>10} {'Min':>8} {'Max':>8}")
    print("-" * 73)

    for eval_set in NB_EVAL_SETS + ["bijective_test"]:
        df = load_parquet(model, eval_set)
        if df is None:
            print(f"{eval_set:<25} {'N/A':>10}")
            continue

        lengths = []
        for idx, row in df.iterrows():
            for resp in row["responses"]:
                lengths.append(len(resp))

        lengths = np.array(lengths)
        print(f"{eval_set:<25} {np.mean(lengths):>10.0f} {np.median(lengths):>10.0f} {np.std(lengths):>10.0f} {np.min(lengths):>8} {np.max(lengths):>8}")

    print()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: Answer format analysis on bijective_test
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 80)
print("SECTION 6: Answer Format Analysis on bijective_test")
print("=" * 80)
print()
print("Bijective problems have exactly 1 preimage. Model is asked for 2 answers.")
print("We check: how many distinct candidates does the model output?")
print()

for model in MODELS:
    df = load_parquet(model, "bijective_test")
    if df is None:
        continue

    count_by_n = defaultdict(int)
    total = 0
    correct_by_n = defaultdict(int)

    for idx, row in df.iterrows():
        for resp in row["responses"]:
            answer_text = extract_answer_text(resp)
            candidates = parse_candidates(answer_text)
            n_cand = len(candidates)
            unique_cand = len(set(candidates))

            total += 1
            count_by_n[n_cand] += 1

            gt = get_gt(row)
            ref_input = str(gt["ref_input"]["x"])
            if any(_is_close(str(c), ref_input) for c in candidates):
                correct_by_n[n_cand] += 1

    print(f"Model: {model}")
    print(f"{'N candidates':<15} {'Count':>8} {'Fraction':>10} {'Has Correct':>13} {'Acc':>8}")
    print("-" * 56)
    for n in sorted(count_by_n.keys()):
        cnt = count_by_n[n]
        frac = cnt / total if total > 0 else 0
        corr = correct_by_n.get(n, 0)
        acc = corr / cnt if cnt > 0 else 0
        print(f"{n:<15} {cnt:>8} {frac:>10.4f} {corr:>13} {acc:>8.4f}")
    print(f"{'Total':<15} {total:>8}")
    print()

    # Also check: does the model output exactly 1 answer (recognizing bijection)?
    single_answer_responses = 0
    single_answer_correct = 0
    for idx, row in df.iterrows():
        for resp in row["responses"]:
            answer_text = extract_answer_text(resp)
            candidates = parse_candidates(answer_text)
            if len(candidates) == 1:
                single_answer_responses += 1
                gt = get_gt(row)
                ref_input = str(gt["ref_input"]["x"])
                if _is_close(str(candidates[0]), ref_input):
                    single_answer_correct += 1

    print(f"  Single-answer rate: {single_answer_responses}/{total} = {single_answer_responses/total:.4f}")
    print(f"  Single-answer accuracy: {single_answer_correct}/{single_answer_responses:.0f} = {single_answer_correct/max(1,single_answer_responses):.4f}")
    print()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7: Error categorization on NB test (both_model)
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 80)
print("SECTION 7: Error Categorization on nonbijective_test")
print("=" * 80)
print()
print("For each problem with 4 responses, categorize into:")
print("  - All correct (all scores > 0)")
print("  - Some correct (at least one > 0, at least one == 0)")
print("  - All wrong (all scores == 0)")
print()

for model in MODELS:
    df = load_parquet(model, "nonbijective_test")
    if df is None:
        continue

    scores_data = all_row_scores.get((model, "nonbijective_test"))
    if not scores_data:
        continue

    all_correct = 0
    some_correct = 0
    all_wrong = 0
    n = len(scores_data)

    for scores in scores_data:
        valid = [s for s in scores if s >= 0]
        n_correct = sum(1 for s in valid if s > 0)
        n_wrong = sum(1 for s in valid if s == 0)
        if n_correct == len(valid) and len(valid) > 0:
            all_correct += 1
        elif n_correct > 0:
            some_correct += 1
        else:
            all_wrong += 1

    print(f"Model: {model} (N={n} problems)")
    print(f"  All correct:  {all_correct:>6} ({all_correct/n:.4f})")
    print(f"  Some correct: {some_correct:>6} ({some_correct/n:.4f})")
    print(f"  All wrong:    {all_wrong:>6} ({all_wrong/n:.4f})")
    print()

    # Breakdown by number of correct responses
    print(f"  Distribution of correct-response count per problem:")
    count_dist = defaultdict(int)
    for scores in scores_data:
        valid = [s for s in scores if s >= 0]
        n_correct = sum(1 for s in valid if s > 0)
        count_dist[n_correct] += 1
    for k in sorted(count_dist.keys()):
        print(f"    {k} of 4 correct: {count_dist[k]:>6} ({count_dist[k]/n:.4f})")
    print()

# Also do score distribution
print("Score value distribution (across all responses, nonbijective_test):")
for model in MODELS:
    scores_data = all_row_scores.get((model, "nonbijective_test"))
    if not scores_data:
        continue
    flat = [s for ss in scores_data for s in ss if s >= 0]
    vals, counts = np.unique(np.round(flat, 4), return_counts=True)
    print(f"\n  Model: {model}")
    for v, c in zip(vals, counts):
        print(f"    score={v:.4f}: {c:>6} ({c/len(flat):.4f})")

print()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8: Differential analysis on nonbijective_test
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 80)
print("SECTION 8: Differential Analysis (both_model vs nb_only_model)")
print("=" * 80)
print()

for eval_set in ["nonbijective_test", "nb_more_candidates", "nb_long_outputs"]:
    df_both = load_parquet("both_model", eval_set)
    df_nb = load_parquet("nb_only_model", eval_set)

    if df_both is None or df_nb is None:
        print(f"{eval_set}: one or both models missing")
        continue

    if len(df_both) != len(df_nb):
        print(f"{eval_set}: different lengths ({len(df_both)} vs {len(df_nb)}), skipping")
        continue

    both_wins = 0
    nb_wins = 0
    both_right = 0
    both_wrong = 0
    n = len(df_both)

    for idx in range(n):
        row_both = df_both.iloc[idx]
        row_nb = df_nb.iloc[idx]

        try:
            scores_both = score_row(row_both, eval_set)
            scores_nb = score_row(row_nb, eval_set)
        except Exception:
            continue

        both_correct = any(s > 0 for s in scores_both if s >= 0)
        nb_correct = any(s > 0 for s in scores_nb if s >= 0)

        if both_correct and nb_correct:
            both_right += 1
        elif both_correct and not nb_correct:
            both_wins += 1
        elif not both_correct and nb_correct:
            nb_wins += 1
        else:
            both_wrong += 1

    print(f"Eval set: {eval_set} (N={n} problems)")
    print(f"  Both correct:             {both_right:>6} ({both_right/n:.4f})")
    print(f"  both_model only correct:  {both_wins:>6} ({both_wins/n:.4f})")
    print(f"  nb_only_model only corr:  {nb_wins:>6} ({nb_wins/n:.4f})")
    print(f"  Both wrong:               {both_wrong:>6} ({both_wrong/n:.4f})")
    print()

    # Mean score comparison
    both_means = []
    nb_means = []
    for idx in range(n):
        row_both = df_both.iloc[idx]
        row_nb = df_nb.iloc[idx]
        try:
            sb = score_row(row_both, eval_set)
            sn = score_row(row_nb, eval_set)
        except Exception:
            continue
        valid_b = [s for s in sb if s >= 0]
        valid_n = [s for s in sn if s >= 0]
        if valid_b:
            both_means.append(np.mean(valid_b))
        if valid_n:
            nb_means.append(np.mean(valid_n))

    print(f"  Mean per-problem score: both_model={np.mean(both_means):.4f}, nb_only={np.mean(nb_means):.4f}")
    print()

print()
print("=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
