#!/usr/bin/env python3
"""Analyze whether Bij1:NB3 performance drop is from strategy choice or execution quality.

Key question: When models use the SAME strategy, does Bij1:NB3 execute it worse?
If yes → the bijective training corrupted execution, not just strategy selection.
If no → the drop is purely from adopting the wrong strategy mix.
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


def load_parquet(model, eval_set):
    path = os.path.join(BASE, model, f"{eval_set}.parquet")
    if not os.path.exists(path):
        return None
    return pd.read_parquet(path)


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


def get_func_name(row):
    try:
        gt = get_gt(row)
        ref_code = gt.get("ref_code", "")
        m = re.search(r'def\s+(func_\d+|nb_\d+|lenop_\d+)\s*\(', ref_code)
        if m:
            return m.group(1)
    except Exception:
        pass
    return "unknown"


def score_response(response_str, gt_str, eval_set):
    data_source = EVAL_DATA_SOURCES[eval_set]
    if "multi-" in data_source:
        return _codeio_multi.compute_score(response_str, gt_str, data_source)
    else:
        return _codeio.compute_score(response_str, gt_str, data_source)


# ─── Strategy detectors ──────────────────────────────────────────────────────

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


def has_case_analysis(text):
    patterns = [
        r"case\s*[12]", r"Case\s*[12]",
        r"two\s+(possible\s+)?cases", r"Two\s+(possible\s+)?cases",
        r"consider\s+(the\s+)?(two|both|each)\s+case",
        r"there\s+are\s+two\s+(possible\s+)?(cases|scenarios|solutions)",
    ]
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def has_verification(text):
    patterns = [
        r"let'?s?\s+verify", r"let'?s?\s+check", r"verify\s+this",
        r"check.*correct", r"confirm", r"to\s+verify",
        r"plug\s+(it\s+)?back", r"substitut",
    ]
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def classify_strategy(text):
    """Classify into primary strategy bucket."""
    inv = has_systematic_inversion(text)
    enum = has_enumeration(text)
    if inv and enum:
        return "inv+enum"
    elif inv:
        return "inv_only"
    elif enum:
        return "enum_only"
    else:
        return "other"


# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 110)
    print("STRATEGY-CONDITIONAL EXECUTION ANALYSIS")
    print("=" * 110)
    print()

    # ── ANALYSIS 1: Strategy-conditional accuracy ────────────────────────────
    # For each strategy bucket, compare accuracy across models.
    # If Bij1:NB3 has lower accuracy WITHIN the same strategy → execution problem.
    print("=" * 110)
    print("ANALYSIS 1: Strategy-Conditional Accuracy")
    print("  For each strategy, how accurately does each model execute it?")
    print("=" * 110)
    print()

    strategy_names = ["enum_only", "inv_only", "inv+enum", "other"]

    for eval_set in EVAL_SETS:
        print(f"--- {eval_set} ---")
        header = f"  {'Strategy':<15}"
        for m in MODELS:
            header += f" {MODEL_LABELS[m]:>20}"
        print(header)
        print(f"  {'-' * (15 + 20 * len(MODELS))}")

        for strat in strategy_names:
            row_str = f"  {strat:<15}"
            for model in MODELS:
                df = load_parquet(model, eval_set)
                if df is None:
                    row_str += f" {'N/A':>20}"
                    continue

                accs = []
                total = 0
                for idx, row in df.iterrows():
                    gt_str = get_gt_str(row)
                    for resp in row["responses"]:
                        s = classify_strategy(resp)
                        if s != strat:
                            continue
                        total += 1
                        try:
                            sc, ac = score_response(resp, gt_str, eval_set)
                        except Exception:
                            ac = 0.0
                        accs.append(ac)

                if accs:
                    mean_acc = np.mean(accs)
                    row_str += f" {mean_acc:.3f} (n={len(accs):>4})"
                else:
                    row_str += f" {'---':>20}"
            print(row_str)
        print()

    # ── ANALYSIS 2: Matched-prompt strategy-conditional accuracy ─────────────
    # For the SAME prompt, compare accuracy when models happen to use the same strategy
    print("=" * 110)
    print("ANALYSIS 2: Matched-Prompt Analysis (Same prompt, same strategy → compare accuracy)")
    print("=" * 110)
    print()

    # Compare Bij1:NB3 vs NB-Only and Bij1:NB3 vs Both(1:1)
    comparisons = [
        ("nb_only_model", "bij1_nb3_model", "NB-Only vs Bij1:NB3"),
        ("both_model", "bij1_nb3_model", "Both(1:1) vs Bij1:NB3"),
        ("nb_only_model", "both_model", "NB-Only vs Both(1:1)"),
        ("both_model", "bij3_nb1_model", "Both(1:1) vs Bij3:NB1"),
    ]

    for eval_set in ["nonbijective_test", "nb_long_outputs", "nb_long_more_cand", "lenchange",
                      "bijective_inverse", "bijective_test"]:
        print(f"--- {eval_set} ---")
        for model_a, model_b, label in comparisons:
            df_a = load_parquet(model_a, eval_set)
            df_b = load_parquet(model_b, eval_set)
            if df_a is None or df_b is None or len(df_a) != len(df_b):
                continue

            # For each strategy, track accuracy when both models use that strategy on same prompt
            strat_matched = defaultdict(lambda: {"a_accs": [], "b_accs": []})

            for idx in range(len(df_a)):
                row_a = df_a.iloc[idx]
                row_b = df_b.iloc[idx]
                gt_str_a = get_gt_str(row_a)
                gt_str_b = get_gt_str(row_b)

                # Compare response-by-response (same index)
                n_resp = min(len(row_a["responses"]), len(row_b["responses"]))
                for ri in range(n_resp):
                    resp_a = row_a["responses"][ri]
                    resp_b = row_b["responses"][ri]
                    strat_a = classify_strategy(resp_a)
                    strat_b = classify_strategy(resp_b)

                    if strat_a == strat_b:
                        try:
                            _, ac_a = score_response(resp_a, gt_str_a, eval_set)
                        except:
                            ac_a = 0.0
                        try:
                            _, ac_b = score_response(resp_b, gt_str_b, eval_set)
                        except:
                            ac_b = 0.0
                        strat_matched[strat_a]["a_accs"].append(ac_a)
                        strat_matched[strat_a]["b_accs"].append(ac_b)

            print(f"  {label}:")
            for strat in strategy_names:
                d = strat_matched[strat]
                if not d["a_accs"]:
                    continue
                a_mean = np.mean(d["a_accs"])
                b_mean = np.mean(d["b_accs"])
                n = len(d["a_accs"])
                diff = b_mean - a_mean
                print(f"    {strat:<15} n={n:>5}  {MODEL_LABELS[model_a]}={a_mean:.4f}  "
                      f"{MODEL_LABELS[model_b]}={b_mean:.4f}  diff={diff:>+.4f}")
            print()

    # ── ANALYSIS 3: Strategy TRANSITION analysis ─────────────────────────────
    # For same prompt: what strategy does each model use?
    print("=" * 110)
    print("ANALYSIS 3: Strategy Transition Matrix (Same prompt, how does strategy change?)")
    print("=" * 110)
    print()

    for eval_set in ["nonbijective_test", "nb_long_outputs", "lenchange",
                      "bijective_inverse", "bijective_test"]:
        print(f"--- {eval_set} ---")
        # Compare NB-Only → Bij1:NB3 and NB-Only → Both(1:1) transitions
        for model_a, model_b, label in [
            ("nb_only_model", "bij1_nb3_model", "NB-Only → Bij1:NB3"),
            ("nb_only_model", "both_model", "NB-Only → Both(1:1)"),
            ("nb_only_model", "bij3_nb1_model", "NB-Only → Bij3:NB1"),
        ]:
            df_a = load_parquet(model_a, eval_set)
            df_b = load_parquet(model_b, eval_set)
            if df_a is None or df_b is None or len(df_a) != len(df_b):
                continue

            # Track: for prompts where model_a uses enum, what does model_b do?
            transition = defaultdict(lambda: {"count": 0, "acc_a": [], "acc_b": []})

            for idx in range(len(df_a)):
                row_a = df_a.iloc[idx]
                row_b = df_b.iloc[idx]
                gt_str_a = get_gt_str(row_a)
                gt_str_b = get_gt_str(row_b)

                n_resp = min(len(row_a["responses"]), len(row_b["responses"]))
                for ri in range(n_resp):
                    resp_a = row_a["responses"][ri]
                    resp_b = row_b["responses"][ri]
                    strat_a = classify_strategy(resp_a)
                    strat_b = classify_strategy(resp_b)
                    key = f"{strat_a} → {strat_b}"
                    transition[key]["count"] += 1
                    try:
                        _, ac_a = score_response(resp_a, gt_str_a, eval_set)
                    except:
                        ac_a = 0.0
                    try:
                        _, ac_b = score_response(resp_b, gt_str_b, eval_set)
                    except:
                        ac_b = 0.0
                    transition[key]["acc_a"].append(ac_a)
                    transition[key]["acc_b"].append(ac_b)

            total = sum(v["count"] for v in transition.values())
            print(f"  {label} (total={total}):")
            for key in sorted(transition.keys(), key=lambda k: -transition[k]["count"]):
                v = transition[key]
                pct = v["count"] / total
                if pct < 0.01:
                    continue
                acc_a = np.mean(v["acc_a"])
                acc_b = np.mean(v["acc_b"])
                diff = acc_b - acc_a
                arrow = "↑" if diff > 0.02 else ("↓" if diff < -0.02 else "≈")
                print(f"    {key:<30} {v['count']:>5} ({pct:>5.1%})  "
                      f"acc: {acc_a:.3f} → {acc_b:.3f} ({diff:>+.3f}) {arrow}")
            print()

    # ── ANALYSIS 4: Fine-grained strategy features ───────────────────────────
    print("=" * 110)
    print("ANALYSIS 4: Detailed Strategy Features per Model")
    print("  (verification, code-tracing, case-analysis rates and their accuracy)")
    print("=" * 110)
    print()

    features = [
        ("Enumeration", has_enumeration),
        ("Inversion", has_systematic_inversion),
        ("Code Tracing", has_code_tracing),
        ("Case Analysis", has_case_analysis),
        ("Verification", has_verification),
    ]

    for eval_set in ["nonbijective_test", "nb_long_outputs", "lenchange", "bijective_inverse"]:
        print(f"--- {eval_set} ---")
        for model in MODELS:
            df = load_parquet(model, eval_set)
            if df is None:
                continue

            # For responses using inversion, what fraction also verify?
            inv_responses = {"total": 0, "correct": 0, "with_verify": 0, "with_verify_correct": 0,
                             "with_trace": 0, "with_trace_correct": 0, "lengths": []}
            enum_responses = {"total": 0, "correct": 0, "with_verify": 0, "with_verify_correct": 0,
                              "lengths": []}

            for idx, row in df.iterrows():
                gt_str = get_gt_str(row)
                for resp in row["responses"]:
                    try:
                        sc, ac = score_response(resp, gt_str, eval_set)
                    except:
                        sc, ac = 0.0, 0.0

                    inv = has_systematic_inversion(resp)
                    enum = has_enumeration(resp)
                    verify = has_verification(resp)
                    trace = has_code_tracing(resp)

                    if inv:
                        inv_responses["total"] += 1
                        if ac > 0: inv_responses["correct"] += 1
                        inv_responses["lengths"].append(len(resp))
                        if verify:
                            inv_responses["with_verify"] += 1
                            if ac > 0: inv_responses["with_verify_correct"] += 1
                        if trace:
                            inv_responses["with_trace"] += 1
                            if ac > 0: inv_responses["with_trace_correct"] += 1

                    if enum:
                        enum_responses["total"] += 1
                        if ac > 0: enum_responses["correct"] += 1
                        enum_responses["lengths"].append(len(resp))
                        if verify:
                            enum_responses["with_verify"] += 1
                            if ac > 0: enum_responses["with_verify_correct"] += 1

            print(f"  {MODEL_LABELS[model]}:")
            if inv_responses["total"] > 0:
                t = inv_responses["total"]
                c = inv_responses["correct"]
                vt = inv_responses["with_verify"]
                vc = inv_responses["with_verify_correct"]
                tt = inv_responses["with_trace"]
                tc = inv_responses["with_trace_correct"]
                avg_len = np.mean(inv_responses["lengths"])
                print(f"    INVERSION: n={t}, acc={c/t:.3f}, avg_len={avg_len:.0f}")
                if vt > 0:
                    print(f"      +verify: n={vt} ({vt/t:.1%}), acc={vc/vt:.3f}")
                if tt > 0:
                    print(f"      +trace:  n={tt} ({tt/t:.1%}), acc={tc/tt:.3f}")
                inv_no_verify = t - vt
                inv_no_verify_correct = c - vc
                if inv_no_verify > 0:
                    print(f"      -verify: n={inv_no_verify} ({inv_no_verify/t:.1%}), acc={inv_no_verify_correct/inv_no_verify:.3f}")
            if enum_responses["total"] > 0:
                t = enum_responses["total"]
                c = enum_responses["correct"]
                vt = enum_responses["with_verify"]
                vc = enum_responses["with_verify_correct"]
                avg_len = np.mean(enum_responses["lengths"])
                print(f"    ENUMERATION: n={t}, acc={c/t:.3f}, avg_len={avg_len:.0f}")
                if vt > 0:
                    print(f"      +verify: n={vt} ({vt/t:.1%}), acc={vc/vt:.3f}")
            print()

    # ── ANALYSIS 5: Per-function strategy shift and accuracy ─────────────────
    print("=" * 110)
    print("ANALYSIS 5: Per-Function Strategy Shift (NB test + lenchange)")
    print("  For each function: what strategy does each model predominantly use, and how accurate?")
    print("=" * 110)
    print()

    for eval_set in ["nonbijective_test", "lenchange"]:
        print(f"--- {eval_set} ---")

        # Collect per-function, per-model data
        func_model_data = defaultdict(lambda: defaultdict(lambda: {
            "enum_n": 0, "enum_correct": 0,
            "inv_n": 0, "inv_correct": 0,
            "other_n": 0, "other_correct": 0,
            "total": 0, "total_correct": 0,
        }))

        all_funcs = set()
        for model in MODELS:
            df = load_parquet(model, eval_set)
            if df is None:
                continue
            for idx, row in df.iterrows():
                func = get_func_name(row)
                all_funcs.add(func)
                gt_str = get_gt_str(row)
                for resp in row["responses"]:
                    try:
                        sc, ac = score_response(resp, gt_str, eval_set)
                    except:
                        ac = 0.0
                    d = func_model_data[func][model]
                    d["total"] += 1
                    if ac > 0: d["total_correct"] += 1

                    if has_enumeration(resp):
                        d["enum_n"] += 1
                        if ac > 0: d["enum_correct"] += 1
                    elif has_systematic_inversion(resp):
                        d["inv_n"] += 1
                        if ac > 0: d["inv_correct"] += 1
                    else:
                        d["other_n"] += 1
                        if ac > 0: d["other_correct"] += 1

        sorted_funcs = sorted(all_funcs, key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else 999)

        for func in sorted_funcs:
            print(f"  {func}:")
            for model in MODELS:
                d = func_model_data[func][model]
                if d["total"] == 0:
                    continue
                t = d["total"]
                acc = d["total_correct"] / t
                enum_pct = d["enum_n"] / t
                inv_pct = d["inv_n"] / t
                enum_acc = d["enum_correct"] / d["enum_n"] if d["enum_n"] > 0 else 0
                inv_acc = d["inv_correct"] / d["inv_n"] if d["inv_n"] > 0 else 0
                other_acc = d["other_correct"] / d["other_n"] if d["other_n"] > 0 else 0
                print(f"    {MODEL_LABELS[model]:<16} acc={acc:.3f}  "
                      f"enum={enum_pct:.0%}(acc={enum_acc:.3f})  "
                      f"inv={inv_pct:.0%}(acc={inv_acc:.3f})  "
                      f"other={1-enum_pct-inv_pct:.0%}(acc={other_acc:.3f})")
            print()

    print("=" * 110)
    print("ANALYSIS COMPLETE")
    print("=" * 110)


if __name__ == "__main__":
    main()
