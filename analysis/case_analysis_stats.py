"""Analyze case analysis usage in model rollouts."""

import re
import json
import pandas as pd
import numpy as np
from scipy import stats
from collections import defaultdict

# ---------- helpers ----------

def has_case_analysis(text: str) -> bool:
    """Check if a response contains case analysis patterns."""
    t = text.lower()
    patterns = [
        r'case\s+[12]',                       # "Case 1", "Case 2"
        r'state\s*=\s*\d+\s+or\s+state\s*=',  # "state = X or state = Y"
        r'state\s+could\s+be',
        r'two\s+possible',
        r'two\s+cases',
        r'either\s+.{1,60}\s+or\s+',          # "either ... or"
        r'state\s+can\s+be',
    ]
    for p in patterns:
        if re.search(p, t):
            return True
    return False


def extract_func_name(prompt_content: str) -> str:
    """Extract nb_XX function name from prompt."""
    m = re.search(r'def (nb_\d+)', prompt_content)
    return m.group(1) if m else 'unknown'


def compute_correctness(response: str, ground_truth_str: str, data_source: str) -> float:
    """Compute accuracy score for a single response."""
    from verl.utils.reward_score import codeio_multi
    try:
        score, acc = codeio_multi.compute_score(response, ground_truth_str, task=data_source)
        return acc
    except Exception:
        return 0.0


def analyze_dataset(path: str, label: str, compute_rewards: bool = True):
    """Analyze a single parquet file."""
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"  {path}")
    print(f"{'='*70}")

    df = pd.read_parquet(path)

    total_responses = 0
    case_analysis_count = 0

    # Per-function tracking
    func_total = defaultdict(int)
    func_case = defaultdict(int)

    # For correctness analysis
    correct_with_case = 0
    correct_without_case = 0
    incorrect_with_case = 0
    incorrect_without_case = 0

    for idx, row in df.iterrows():
        prompt_text = row['prompt'][0]['content']
        func_name = extract_func_name(prompt_text)
        gt_str = row['reward_model']['ground_truth']
        ds = row['data_source']

        for resp in row['responses']:
            total_responses += 1
            func_total[func_name] += 1
            ca = has_case_analysis(resp)
            if ca:
                case_analysis_count += 1
                func_case[func_name] += 1

            if compute_rewards:
                acc = compute_correctness(resp, gt_str, ds)
                is_correct = acc > 0.99
                if ca and is_correct:
                    correct_with_case += 1
                elif ca and not is_correct:
                    incorrect_with_case += 1
                elif not ca and is_correct:
                    correct_without_case += 1
                else:
                    incorrect_without_case += 1

    pct = 100.0 * case_analysis_count / total_responses if total_responses else 0
    print(f"\nTotal responses: {total_responses}")
    print(f"With case analysis: {case_analysis_count} ({pct:.1f}%)")
    print(f"Without case analysis: {total_responses - case_analysis_count} ({100-pct:.1f}%)")

    if compute_rewards:
        print(f"\n--- Correctness Breakdown ---")
        total_case = correct_with_case + incorrect_with_case
        total_nocase = correct_without_case + incorrect_without_case
        if total_case > 0:
            print(f"  WITH case analysis:    {correct_with_case}/{total_case} correct ({100*correct_with_case/total_case:.1f}%)")
        else:
            print(f"  WITH case analysis:    0/0 correct (N/A)")
        if total_nocase > 0:
            print(f"  WITHOUT case analysis: {correct_without_case}/{total_nocase} correct ({100*correct_without_case/total_nocase:.1f}%)")
        else:
            print(f"  WITHOUT case analysis: 0/0 correct (N/A)")

        total_correct = correct_with_case + correct_without_case
        total_incorrect = incorrect_with_case + incorrect_without_case
        if total_correct > 0:
            print(f"\n  Of CORRECT responses:   {correct_with_case}/{total_correct} use case analysis ({100*correct_with_case/total_correct:.1f}%)")
        if total_incorrect > 0:
            print(f"  Of INCORRECT responses: {incorrect_with_case}/{total_incorrect} use case analysis ({100*incorrect_with_case/total_incorrect:.1f}%)")

    # Per-function breakdown
    print(f"\n--- Per-Function Breakdown ---")
    for fn in sorted(func_total.keys()):
        t = func_total[fn]
        c = func_case[fn]
        print(f"  {fn}: {c}/{t} ({100*c/t:.1f}%)")

    return {
        'total': total_responses,
        'case_analysis': case_analysis_count,
        'correct_with_case': correct_with_case,
        'correct_without_case': correct_without_case,
        'incorrect_with_case': incorrect_with_case,
        'incorrect_without_case': incorrect_without_case,
    }


def main():
    import sys
    sys.path.insert(0, '/home/ubuntu/rl_meta_thinking')

    # ---------- Main test sets ----------
    both_main = analyze_dataset(
        'analysis/rollouts/both_model/nonbijective_test.parquet',
        'BOTH MODEL - nonbijective_test',
        compute_rewards=True,
    )

    nb_main = analyze_dataset(
        'analysis/rollouts/nb_only_model/nonbijective_test.parquet',
        'NB-ONLY MODEL - nonbijective_test',
        compute_rewards=True,
    )

    # ---------- Statistical test ----------
    print(f"\n{'='*70}")
    print(f"  STATISTICAL COMPARISON (both_model vs nb_only_model on nonbijective_test)")
    print(f"{'='*70}")

    table = np.array([
        [both_main['case_analysis'], both_main['total'] - both_main['case_analysis']],
        [nb_main['case_analysis'], nb_main['total'] - nb_main['case_analysis']],
    ])
    print(f"\nContingency table (case_analysis vs no_case_analysis):")
    print(f"  both_model:    {table[0]}")
    print(f"  nb_only_model: {table[1]}")

    chi2, p_chi, dof, expected = stats.chi2_contingency(table)
    print(f"\nChi-squared test: chi2={chi2:.4f}, p={p_chi:.6f}, dof={dof}")

    odds_ratio_result = stats.fisher_exact(table)
    print(f"Fisher's exact test: OR={odds_ratio_result[0]:.4f}, p={odds_ratio_result[1]:.6f}")

    # ---------- OOD sets ----------
    ood_files = [
        ('analysis/rollouts/both_model/nb_more_candidates.parquet', 'BOTH MODEL - nb_more_candidates (OOD)'),
        ('analysis/rollouts/nb_only_model/nb_more_candidates.parquet', 'NB-ONLY MODEL - nb_more_candidates (OOD)'),
        ('analysis/rollouts/both_model/nb_long_outputs.parquet', 'BOTH MODEL - nb_long_outputs (OOD)'),
        ('analysis/rollouts/nb_only_model/nb_long_outputs.parquet', 'NB-ONLY MODEL - nb_long_outputs (OOD)'),
    ]

    ood_results = {}
    for path, label in ood_files:
        ood_results[label] = analyze_dataset(path, label, compute_rewards=True)

    # ---------- Summary ----------
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")

    all_results = {
        'both_model/nonbijective_test': both_main,
        'nb_only_model/nonbijective_test': nb_main,
        **ood_results,
    }

    print(f"\n{'Dataset':<50} {'Case %':>8} {'Correct w/ case':>18} {'Correct w/o case':>18}")
    print("-" * 96)
    for name, r in all_results.items():
        ca_pct = 100 * r['case_analysis'] / r['total'] if r['total'] else 0
        cw = r['correct_with_case']
        iw = r['incorrect_with_case']
        tw = cw + iw
        cwo = r['correct_without_case']
        iwo = r['incorrect_without_case']
        two = cwo + iwo
        cw_pct = f"{100*cw/tw:.1f}%" if tw else "N/A"
        cwo_pct = f"{100*cwo/two:.1f}%" if two else "N/A"
        print(f"{name:<50} {ca_pct:>7.1f}% {cw}/{tw} ({cw_pct:>5}){'':<3} {cwo}/{two} ({cwo_pct:>5})")


if __name__ == '__main__':
    main()
