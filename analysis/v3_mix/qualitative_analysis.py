#!/usr/bin/env python3
"""
Qualitative analysis of v3_mix rollouts: deep-dive into reasoning patterns.

This script samples representative cases from model rollouts and produces
structured categorizations of reasoning strategies, error modes, and
the key differences between models trained with vs without bijective co-training.

Research hypothesis: Co-training on bijective inversion prevents "reasoning hacking"
because mutate-and-enumerate fails when there's exactly one valid preimage.
"""

import json
import os
import re
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

BASE = str(Path(__file__).resolve().parent / "rollouts")
MODELS = ["sft_baseline", "nb_only_model", "both_model"]
MODEL_LABELS = {
    "sft_baseline": "SFT Baseline",
    "nb_only_model": "NB-Only RL",
    "both_model": "Both RL",
}

# Import function environment for execution
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


# ─── Utilities ────────────────────────────────────────────────────────────────

def load_parquet(model, eval_set):
    path = os.path.join(BASE, model, f"{eval_set}.parquet")
    if not os.path.exists(path):
        return None
    return pd.read_parquet(path)


def extract_answer_text(text):
    m = re.search(r"<answer>(.*?)</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    if not m:
        return None
    inner = m.group(1).replace('"', '')
    return re.sub(r"\s+", "", inner)


def extract_thinking(text):
    """Extract content between <think> and </think>."""
    m = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # Sometimes the response starts with thinking directly
    m2 = re.search(r"^(.*?)</?answer>", text, flags=re.DOTALL)
    if m2:
        return m2.group(1).strip()
    return text.strip()


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


def build_ref_fn(ref_code, funcname="main_solution"):
    ns = {"__builtins__": __builtins__}
    if _CODEIO_FUNC_ENV:
        ns.update(_CODEIO_FUNC_ENV)
    exec(ref_code, ns, ns)
    return ns[funcname]


def _is_close(a, b):
    if isinstance(a, str) and isinstance(b, str):
        return a.strip() == b.strip()
    try:
        return abs(float(a) - float(b)) < 1e-6
    except (ValueError, TypeError):
        return str(a).strip() == str(b).strip()


def exec_check(candidate, gt):
    """Execute ref_code on candidate and check if output matches."""
    ref_code = gt.get("ref_code")
    ref_output = gt.get("ref_output")
    funcname = gt.get("funcname", "main_solution")
    if not ref_code:
        return None, None
    try:
        ref_fn = build_ref_fn(ref_code, funcname)
        result = ref_fn(str(candidate))
        matches = _is_close(result, ref_output)
        return result, matches
    except Exception as e:
        return None, False


# ─── Reasoning strategy classification ────────────────────────────────────────

@dataclass
class ReasoningProfile:
    """Detailed reasoning profile for a single response."""
    uses_inversion: bool = False       # Systematic step-by-step inversion
    uses_enumeration: bool = False     # Trial-and-error / brute force guessing
    uses_case_analysis: bool = False   # Explicit case branching
    uses_verification: bool = False    # Hypothesis verification / plugging back
    uses_pattern_matching: bool = False  # Pattern recognition without explicit computation
    uses_digit_manipulation: bool = False  # Direct digit-level operations
    recognizes_uniqueness: bool = False  # Recognizes only one valid preimage
    mentions_none: bool = False        # Explicitly says None for second answer
    has_computation_error: bool = False  # Shows work but gets wrong answer
    is_truncated: bool = False          # Response was cut off
    reasoning_length: int = 0           # Length of thinking section
    n_candidates_tried: int = 0         # Number of candidates tried in reasoning
    n_candidates_submitted: int = 0     # Number in final answer
    correct: bool = False
    score: float = 0.0


def classify_response(response_text, gt, eval_set) -> ReasoningProfile:
    """Classify a single response's reasoning strategy."""
    profile = ReasoningProfile()

    thinking = extract_thinking(response_text)
    profile.reasoning_length = len(thinking)
    profile.is_truncated = len(response_text) > 30000

    answer_text = extract_answer_text(response_text)
    candidates = parse_candidates(answer_text)
    profile.n_candidates_submitted = len(candidates)

    # Count candidates explored in reasoning (not just final answer)
    tried_in_thinking = re.findall(r"(?:x\s*=|input\s*=|try|trying)\s*[\"']?(\d+)[\"']?", thinking, re.IGNORECASE)
    profile.n_candidates_tried = len(set(tried_in_thinking))

    # Strategy detection
    inv_patterns = [
        r"invert", r"inverse", r"reverse\s+(the\s+)?operation",
        r"undo", r"work\s+backwards?", r"reverse\s+engineer",
        r"apply.*in\s+reverse", r"going\s+backwards?",
        r"starting\s+from\s+the\s+output",
        r"step\s+by\s+step.*reverse", r"reverse\s+each\s+step",
    ]
    profile.uses_inversion = any(re.search(p, thinking, re.IGNORECASE) for p in inv_patterns)

    enum_patterns = [
        r"let'?s?\s+try", r"trying\s+(input|value|string|x)",
        r"try\s+(another|different|other|next)", r"trial\s+and\s+error",
        r"brute\s*force", r"let me try", r"check\s+if\s+\d+\s+works",
        r"if\s+x\s*=\s*\d+", r"suppose\s+x\s*=",
        r"testing\s+\d+", r"check.*x\s*=\s*\d+",
    ]
    profile.uses_enumeration = any(re.search(p, thinking, re.IGNORECASE) for p in enum_patterns)

    case_patterns = [
        r"case\s*[12]", r"Case\s*[12]",
        r"two\s+(possible\s+)?cases", r"consider\s+(the\s+)?(two|both|each)\s+case",
        r"there\s+are\s+two\s+(possible\s+)?(cases|scenarios|solutions)",
    ]
    profile.uses_case_analysis = any(re.search(p, thinking, re.IGNORECASE) for p in case_patterns)

    verify_patterns = [
        r"let'?s?\s+verify", r"let'?s?\s+check", r"verify\s+this",
        r"check.*correct", r"to\s+verify", r"plug\s+(it\s+)?back",
        r"substitut", r"confirm",
    ]
    profile.uses_verification = any(re.search(p, thinking, re.IGNORECASE) for p in verify_patterns)

    pattern_patterns = [
        r"pattern", r"notice\s+that", r"observe\s+that",
        r"the\s+function.*maps", r"looking\s+at\s+the\s+mapping",
    ]
    profile.uses_pattern_matching = any(re.search(p, thinking, re.IGNORECASE) for p in pattern_patterns)

    digit_patterns = [
        r"digit", r"each\s+character", r"position\s+\d+",
        r"swap", r"rotate", r"shift", r"modular",
        r"mod\s+\d+", r"remainder",
    ]
    profile.uses_digit_manipulation = any(re.search(p, thinking, re.IGNORECASE) for p in digit_patterns)

    unique_patterns = [
        r"\bNone\b", r"no\s+other\s+(valid\s+)?input",
        r"only\s+one\s+(valid\s+)?input", r"unique",
        r"bijective", r"one-to-one", r"exactly\s+one",
        r"no\s+second", r"no\s+other\s+(possible|valid)",
    ]
    profile.recognizes_uniqueness = any(re.search(p, thinking, re.IGNORECASE) for p in unique_patterns)

    profile.mentions_none = bool(re.search(r"\bNone\b", response_text))

    # Computation error detection: model shows work but answer is wrong
    # This is determined after scoring

    return profile


def score_for_analysis(response_text, gt, eval_set):
    """Score a response, returning float score."""
    answer_text = extract_answer_text(response_text)
    candidates = parse_candidates(answer_text)

    gt_info = gt.get("gt_info")
    scoring_mode = gt_info.get("multi_response_scoring") if isinstance(gt_info, dict) else None

    if eval_set in ("bijective_inverse",):
        # Single-answer inverse
        if not answer_text:
            return 0.0
        ref_input = gt.get("ref_input")
        if isinstance(ref_input, dict):
            ref_input = str(ref_input.get("x", ""))
        else:
            ref_input = str(ref_input)
        for c in candidates:
            if _is_close(str(c), ref_input):
                return 1.0
        return 0.0

    ref_code = gt.get("ref_code", "")
    ref_output = gt.get("ref_output")
    funcname = gt.get("funcname", "main_solution")
    num_answers = 2
    if isinstance(gt_info, dict):
        na = gt_info.get("num_answers")
        if na:
            num_answers = int(na)
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


# ─── Sampling & analysis ─────────────────────────────────────────────────────

def analyze_eval_set(eval_set, sample_n=50, seed=42):
    """Analyze reasoning patterns across models for a given eval set."""
    print(f"\n{'='*90}")
    print(f"QUALITATIVE ANALYSIS: {eval_set}")
    print(f"{'='*90}\n")

    dfs = {}
    for model in MODELS:
        df = load_parquet(model, eval_set)
        if df is not None:
            dfs[model] = df

    if not dfs:
        print("No data available.")
        return

    # Use the first available model to determine problem indices
    ref_model = list(dfs.keys())[0]
    ref_df = dfs[ref_model]
    n_problems = len(ref_df)
    np.random.seed(seed)
    sample_indices = np.random.choice(n_problems, min(sample_n, n_problems), replace=False)
    sample_indices.sort()

    # Classify all responses for sampled problems
    model_profiles = {}  # model -> list of (idx, response_idx, profile)

    for model in dfs:
        df = dfs[model]
        profiles = []
        for idx in sample_indices:
            if idx >= len(df):
                continue
            row = df.iloc[idx]
            gt = get_gt(row)
            for resp_idx, resp in enumerate(row["responses"]):
                profile = classify_response(resp, gt, eval_set)
                try:
                    profile.score = score_for_analysis(resp, gt, eval_set)
                    profile.correct = profile.score > 0
                except Exception:
                    profile.score = 0.0
                    profile.correct = False

                # Detect computation error: uses inversion but gets wrong answer
                if profile.uses_inversion and not profile.correct and not profile.uses_enumeration:
                    profile.has_computation_error = True

                profiles.append((int(idx), resp_idx, profile))

        model_profiles[model] = profiles

    # ── Aggregate statistics ──────────────────────────────────────────────
    print("--- Reasoning Strategy Distribution ---\n")

    for model in dfs:
        profiles = model_profiles[model]
        n_total = len(profiles)
        if n_total == 0:
            continue

        counts = {
            "Systematic Inversion": sum(1 for _, _, p in profiles if p.uses_inversion),
            "Enumeration/Trial": sum(1 for _, _, p in profiles if p.uses_enumeration),
            "Case Analysis": sum(1 for _, _, p in profiles if p.uses_case_analysis),
            "Verification": sum(1 for _, _, p in profiles if p.uses_verification),
            "Pattern Matching": sum(1 for _, _, p in profiles if p.uses_pattern_matching),
            "Digit Manipulation": sum(1 for _, _, p in profiles if p.uses_digit_manipulation),
            "Recognizes Uniqueness": sum(1 for _, _, p in profiles if p.recognizes_uniqueness),
            "Mentions None": sum(1 for _, _, p in profiles if p.mentions_none),
        }

        correct_total = sum(1 for _, _, p in profiles if p.correct)

        print(f"  Model: {MODEL_LABELS[model]} ({n_total} responses, {correct_total} correct [{correct_total/n_total:.1%}])")
        for strat, cnt in counts.items():
            # Accuracy when strategy is used vs not
            with_strat = [p for _, _, p in profiles if getattr(p, {
                "Systematic Inversion": "uses_inversion",
                "Enumeration/Trial": "uses_enumeration",
                "Case Analysis": "uses_case_analysis",
                "Verification": "uses_verification",
                "Pattern Matching": "uses_pattern_matching",
                "Digit Manipulation": "uses_digit_manipulation",
                "Recognizes Uniqueness": "recognizes_uniqueness",
                "Mentions None": "mentions_none",
            }[strat])]
            acc_with = np.mean([1.0 if p.correct else 0.0 for p in with_strat]) if with_strat else 0.0
            print(f"    {strat:<25} {cnt:>4}/{n_total} ({cnt/n_total:.1%})  acc={acc_with:.1%}")
        print()

    # ── Strategy co-occurrence ────────────────────────────────────────────
    print("--- Strategy Co-occurrence (Inversion & Enumeration) ---\n")

    for model in dfs:
        profiles = model_profiles[model]
        n_total = len(profiles)
        if n_total == 0:
            continue

        both = sum(1 for _, _, p in profiles if p.uses_inversion and p.uses_enumeration)
        inv_only = sum(1 for _, _, p in profiles if p.uses_inversion and not p.uses_enumeration)
        enum_only = sum(1 for _, _, p in profiles if not p.uses_inversion and p.uses_enumeration)
        neither = sum(1 for _, _, p in profiles if not p.uses_inversion and not p.uses_enumeration)

        acc_both = np.mean([1.0 if p.correct else 0.0 for _, _, p in profiles if p.uses_inversion and p.uses_enumeration]) if both else 0
        acc_inv = np.mean([1.0 if p.correct else 0.0 for _, _, p in profiles if p.uses_inversion and not p.uses_enumeration]) if inv_only else 0
        acc_enum = np.mean([1.0 if p.correct else 0.0 for _, _, p in profiles if not p.uses_inversion and p.uses_enumeration]) if enum_only else 0
        acc_neither = np.mean([1.0 if p.correct else 0.0 for _, _, p in profiles if not p.uses_inversion and not p.uses_enumeration]) if neither else 0

        print(f"  {MODEL_LABELS[model]}:")
        print(f"    Inv+Enum:  {both:>4} ({both/n_total:.1%}) acc={acc_both:.1%}")
        print(f"    Inv only:  {inv_only:>4} ({inv_only/n_total:.1%}) acc={acc_inv:.1%}")
        print(f"    Enum only: {enum_only:>4} ({enum_only/n_total:.1%}) acc={acc_enum:.1%}")
        print(f"    Neither:   {neither:>4} ({neither/n_total:.1%}) acc={acc_neither:.1%}")
        print()

    # ── Error mode breakdown ──────────────────────────────────────────────
    print("--- Error Mode Breakdown (incorrect responses only) ---\n")

    for model in dfs:
        profiles = model_profiles[model]
        wrong = [(i, ri, p) for i, ri, p in profiles if not p.correct and p.score >= 0]
        if not wrong:
            continue

        n_wrong = len(wrong)
        comp_errors = sum(1 for _, _, p in wrong if p.has_computation_error)
        format_errors = sum(1 for _, _, p in wrong if p.n_candidates_submitted == 0)
        truncated = sum(1 for _, _, p in wrong if p.is_truncated)
        enum_fails = sum(1 for _, _, p in wrong if p.uses_enumeration and not p.uses_inversion)
        inv_fails = sum(1 for _, _, p in wrong if p.uses_inversion and not p.uses_enumeration and not p.has_computation_error)

        print(f"  {MODEL_LABELS[model]} ({n_wrong} wrong responses):")
        print(f"    Computation error (good strategy, bad math): {comp_errors:>4} ({comp_errors/n_wrong:.1%})")
        print(f"    Enumeration failure (tried guessing, failed): {enum_fails:>4} ({enum_fails/n_wrong:.1%})")
        print(f"    Inversion failure (unclear error):            {inv_fails:>4} ({inv_fails/n_wrong:.1%})")
        print(f"    Format error (no candidates):                 {format_errors:>4} ({format_errors/n_wrong:.1%})")
        print(f"    Truncated (hit max length):                   {truncated:>4} ({truncated/n_wrong:.1%})")
        print()

    # ── Response length analysis ──────────────────────────────────────────
    print("--- Response Length by Correctness ---\n")

    for model in dfs:
        profiles = model_profiles[model]
        correct_lens = [p.reasoning_length for _, _, p in profiles if p.correct]
        wrong_lens = [p.reasoning_length for _, _, p in profiles if not p.correct]

        if correct_lens and wrong_lens:
            print(f"  {MODEL_LABELS[model]}:")
            print(f"    Correct: mean={np.mean(correct_lens):.0f} chars, median={np.median(correct_lens):.0f}")
            print(f"    Wrong:   mean={np.mean(wrong_lens):.0f} chars, median={np.median(wrong_lens):.0f}")
            print(f"    Ratio:   {np.mean(wrong_lens)/np.mean(correct_lens):.2f}x")
            print()

    # ── Representative examples ───────────────────────────────────────────
    print_representative_examples(dfs, model_profiles, eval_set, sample_indices)


def print_representative_examples(dfs, model_profiles, eval_set, sample_indices, max_examples=3):
    """Print representative examples showing key differences between models."""
    print(f"\n{'─'*90}")
    print("REPRESENTATIVE EXAMPLES")
    print(f"{'─'*90}\n")

    # Find problems where models diverge
    models_present = [m for m in MODELS if m in dfs]
    if len(models_present) < 2:
        print("Need at least 2 models for comparative examples.")
        return

    # Build per-problem score maps
    problem_scores = defaultdict(dict)  # idx -> model -> [scores]
    for model in models_present:
        for idx, resp_idx, profile in model_profiles[model]:
            if idx not in problem_scores:
                problem_scores[idx] = {}
            if model not in problem_scores[idx]:
                problem_scores[idx][model] = []
            problem_scores[idx][model].append(profile)

    # Category 1: both_model correct, nb_only wrong
    if "both_model" in models_present and "nb_only_model" in models_present:
        print("=== Category: Both-RL correct, NB-Only wrong ===\n")
        count = 0
        for idx in sorted(problem_scores.keys()):
            both_profiles = problem_scores[idx].get("both_model", [])
            nb_profiles = problem_scores[idx].get("nb_only_model", [])
            if not both_profiles or not nb_profiles:
                continue

            both_any_correct = any(p.correct for p in both_profiles)
            nb_any_correct = any(p.correct for p in nb_profiles)

            if both_any_correct and not nb_any_correct and count < max_examples:
                row = dfs["both_model"].iloc[idx]
                func = get_func_name(row["prompt"])
                gt = get_gt(row)
                ref_output = gt.get("ref_output", "?")
                print(f"  Problem idx={idx}, func={func}, ref_output={ref_output}")

                # Show first correct both_model response (abbreviated)
                for resp_idx, p in enumerate(both_profiles):
                    if p.correct:
                        thinking = extract_thinking(row["responses"][resp_idx])
                        answer = extract_answer_text(row["responses"][resp_idx])
                        strategies = []
                        if p.uses_inversion: strategies.append("inversion")
                        if p.uses_enumeration: strategies.append("enumeration")
                        if p.uses_verification: strategies.append("verification")
                        if p.uses_digit_manipulation: strategies.append("digit-manip")
                        print(f"    Both-RL (correct, score={p.score:.2f}): strategies={strategies}")
                        print(f"      Thinking (first 500 chars): {thinking[:500]}...")
                        print(f"      Answer: {answer}")
                        break

                # Show first nb_only response (abbreviated)
                nb_row = dfs["nb_only_model"].iloc[idx]
                for resp_idx, p in enumerate(nb_profiles):
                    thinking = extract_thinking(nb_row["responses"][resp_idx])
                    answer = extract_answer_text(nb_row["responses"][resp_idx])
                    strategies = []
                    if p.uses_inversion: strategies.append("inversion")
                    if p.uses_enumeration: strategies.append("enumeration")
                    if p.uses_verification: strategies.append("verification")
                    if p.uses_digit_manipulation: strategies.append("digit-manip")
                    print(f"    NB-Only (wrong, score={p.score:.2f}): strategies={strategies}")
                    print(f"      Thinking (first 500 chars): {thinking[:500]}...")
                    print(f"      Answer: {answer}")
                    break

                print()
                count += 1

        if count == 0:
            print("  (No divergent cases found in sample)\n")

    # Category 2: NB-only correct, both_model wrong
    if "both_model" in models_present and "nb_only_model" in models_present:
        print("=== Category: NB-Only correct, Both-RL wrong ===\n")
        count = 0
        for idx in sorted(problem_scores.keys()):
            both_profiles = problem_scores[idx].get("both_model", [])
            nb_profiles = problem_scores[idx].get("nb_only_model", [])
            if not both_profiles or not nb_profiles:
                continue

            both_any_correct = any(p.correct for p in both_profiles)
            nb_any_correct = any(p.correct for p in nb_profiles)

            if not both_any_correct and nb_any_correct and count < max_examples:
                row = dfs["nb_only_model"].iloc[idx]
                func = get_func_name(row["prompt"])
                gt = get_gt(row)
                ref_output = gt.get("ref_output", "?")
                print(f"  Problem idx={idx}, func={func}, ref_output={ref_output}")

                for resp_idx, p in enumerate(nb_profiles):
                    if p.correct:
                        thinking = extract_thinking(row["responses"][resp_idx])
                        answer = extract_answer_text(row["responses"][resp_idx])
                        strategies = []
                        if p.uses_inversion: strategies.append("inversion")
                        if p.uses_enumeration: strategies.append("enumeration")
                        print(f"    NB-Only (correct, score={p.score:.2f}): strategies={strategies}")
                        print(f"      Thinking (first 500 chars): {thinking[:500]}...")
                        print(f"      Answer: {answer}")
                        break

                both_row = dfs["both_model"].iloc[idx]
                for resp_idx, p in enumerate(both_profiles):
                    thinking = extract_thinking(both_row["responses"][resp_idx])
                    answer = extract_answer_text(both_row["responses"][resp_idx])
                    strategies = []
                    if p.uses_inversion: strategies.append("inversion")
                    if p.uses_enumeration: strategies.append("enumeration")
                    print(f"    Both-RL (wrong, score={p.score:.2f}): strategies={strategies}")
                    print(f"      Thinking (first 500 chars): {thinking[:500]}...")
                    print(f"      Answer: {answer}")
                    break

                print()
                count += 1

        if count == 0:
            print("  (No divergent cases found in sample)\n")

    # Category 3: Interesting enumeration examples
    print("=== Category: Heavy Enumeration Examples ===\n")
    for model in models_present:
        count = 0
        for idx, resp_idx, p in model_profiles[model]:
            if p.uses_enumeration and p.n_candidates_tried >= 3 and count < 2:
                row = dfs[model].iloc[idx]
                func = get_func_name(row["prompt"])
                thinking = extract_thinking(row["responses"][resp_idx])
                answer = extract_answer_text(row["responses"][resp_idx])
                print(f"  {MODEL_LABELS[model]}, idx={idx}, func={func}, correct={p.correct}, score={p.score:.2f}")
                print(f"    Candidates tried: {p.n_candidates_tried}, submitted: {p.n_candidates_submitted}")
                print(f"    Thinking (first 600 chars): {thinking[:600]}...")
                print(f"    Answer: {answer}")
                print()
                count += 1


def main():
    print("=" * 90)
    print("v3_mix QUALITATIVE ANALYSIS")
    print("Deep-dive into reasoning patterns across SFT, NB-Only RL, and Both RL models")
    print("=" * 90)

    # Analyze each eval set
    for eval_set in ["nonbijective_test", "bijective_test", "bijective_inverse"]:
        analyze_eval_set(eval_set, sample_n=100, seed=42)

    # Additional focused analysis on bijective test
    print("\n" + "=" * 90)
    print("FOCUSED ANALYSIS: Bijective Test - Uniqueness Recognition")
    print("=" * 90 + "\n")

    for model in MODELS:
        df = load_parquet(model, "bijective_test")
        if df is None:
            continue

        n_total = 0
        n_correct = 0
        n_recognizes_unique = 0
        n_correct_and_unique = 0
        n_says_none = 0
        n_gives_two_diff = 0
        n_gives_two_same = 0

        for idx, row in df.iterrows():
            gt = get_gt(row)
            for resp in row["responses"]:
                n_total += 1
                profile = classify_response(resp, gt, "bijective_test")
                try:
                    profile.score = score_for_analysis(resp, gt, "bijective_test")
                    profile.correct = profile.score > 0
                except Exception:
                    pass

                if profile.correct:
                    n_correct += 1
                if profile.recognizes_uniqueness:
                    n_recognizes_unique += 1
                if profile.correct and profile.recognizes_uniqueness:
                    n_correct_and_unique += 1
                if profile.mentions_none:
                    n_says_none += 1

                answer_text = extract_answer_text(resp)
                candidates = parse_candidates(answer_text)
                unique_cands = list(set(candidates))
                if len(unique_cands) >= 2:
                    n_gives_two_diff += 1
                elif len(candidates) >= 2 and len(unique_cands) == 1:
                    n_gives_two_same += 1

        print(f"Model: {MODEL_LABELS[model]} (N={n_total} responses)")
        print(f"  Correct:                 {n_correct:>6} ({n_correct/n_total:.1%})")
        print(f"  Recognizes uniqueness:   {n_recognizes_unique:>6} ({n_recognizes_unique/n_total:.1%})")
        print(f"  Correct + unique:        {n_correct_and_unique:>6} ({n_correct_and_unique/n_total:.1%})")
        print(f"  Says 'None':             {n_says_none:>6} ({n_says_none/n_total:.1%})")
        print(f"  Two DIFFERENT candidates:{n_gives_two_diff:>6} ({n_gives_two_diff/n_total:.1%})")
        print(f"  Two SAME candidates:     {n_gives_two_same:>6} ({n_gives_two_same/n_total:.1%})")
        print()

    print("=" * 90)
    print("QUALITATIVE ANALYSIS COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    main()
