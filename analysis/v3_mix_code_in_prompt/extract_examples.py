import pandas as pd
import json
import re
import sys
sys.path.insert(0, '.')
from verl.utils.reward_score import codeio_multi, codeio

def score_response(resp, gt_str, data_source):
    if 'multi-' in data_source:
        return codeio_multi.compute_score(resp, gt_str, data_source)
    else:
        return codeio.compute_score(resp, gt_str, data_source)

def extract_answer_text(resp):
    m = re.search(r"<answer>(.*?)</answer>", resp, flags=re.DOTALL|re.IGNORECASE)
    return m.group(1).strip() if m else None

def get_gt(row):
    gt = row['reward_model']['ground_truth']
    return json.loads(gt) if isinstance(gt, str) else gt

def get_gt_str(row):
    gt = row['reward_model']['ground_truth']
    return gt if isinstance(gt, str) else json.dumps(gt)

base = 'analysis/v3_mix_code_in_prompt/rollouts'

eval_configs = {
    'bijective_inverse': 'codeio-backward-incomplete-depth1',
    'bijective_test': 'codeio-multi-preimage-incomplete-depth1',
    'nonbijective_test': 'codeio-nb-multi-preimage-incomplete-depth1',
    'lenchange': 'codeio-nb-multi-preimage-lenchange-eval-depth1',
    'nb_long_outputs': 'codeio-nb-multi-preimage-eval-long-outputs-v2-code-depth1',
}

# ============================================================
# EXAMPLE SET 1: Bijective inverse — Both correct, NB wrong
# Find a case where Both model uses clean inversion vs NB-Only enumeration
# ============================================================
print("=" * 80)
print("EXAMPLE SET 1: Bijective inverse — Both correct via inversion, NB-Only wrong")
print("=" * 80)

ev = 'bijective_inverse'
ds = eval_configs[ev]
df_both = pd.read_parquet(f'{base}/both_model/{ev}.parquet')
df_nb = pd.read_parquet(f'{base}/nb_only_model/{ev}.parquet')

found = 0
for idx in range(len(df_both)):
    if found >= 1:
        break
    row_b = df_both.iloc[idx]
    row_n = df_nb.iloc[idx]
    gt = get_gt(row_b)
    gt_str_b = get_gt_str(row_b)
    gt_str_n = get_gt_str(row_n)

    # Find a case where Both gets it right and NB gets it wrong
    b_scores = [score_response(r, gt_str_b, ds) for r in row_b['responses']]
    n_scores = [score_response(r, gt_str_n, ds) for r in row_n['responses']]

    b_any_correct = any(ac > 0 for sc, ac in b_scores)
    n_any_correct = any(ac > 0 for sc, ac in n_scores)

    if b_any_correct and not n_any_correct:
        # Find the best Both response and worst NB response
        best_b_idx = max(range(len(b_scores)), key=lambda i: b_scores[i][1])
        worst_n_idx = 0  # just take first

        resp_b = row_b['responses'][best_b_idx]
        resp_n = row_n['responses'][worst_n_idx]

        # Extract function name from ref_code
        func_m = re.search(r'def\s+(func_\d+)\s*\(', gt.get('ref_code', ''))
        func_name = func_m.group(1) if func_m else 'unknown'

        # Get prompt (user message only, trimmed)
        prompt = row_b['prompt']
        if isinstance(prompt, list):
            user_msg = [m['content'] for m in prompt if m.get('role') == 'user'][0]
        else:
            user_msg = str(prompt)

        print(f"\nFunction: {func_name}")
        print(f"Ground truth input: {gt.get('ref_input')}")
        print(f"Given output: {gt.get('ref_output')}")
        print(f"Both score: {b_scores[best_b_idx]}, NB score: {n_scores[worst_n_idx]}")
        print(f"\n--- PROMPT (first 800 chars) ---")
        print(user_msg[:800])
        print(f"\n--- BOTH MODEL RESPONSE (first 1500 chars) ---")
        print(resp_b[:1500])
        ans_b = extract_answer_text(resp_b)
        print(f"\n[Answer]: {ans_b}")
        print(f"\n--- NB-ONLY MODEL RESPONSE (first 1500 chars) ---")
        print(resp_n[:1500])
        ans_n = extract_answer_text(resp_n)
        print(f"\n[Answer]: {ans_n}")
        found += 1

# ============================================================
# EXAMPLE SET 2: Bijective test — Both recognizes uniqueness
# ============================================================
print("\n" + "=" * 80)
print("EXAMPLE SET 2: Bijective test — Both recognizes single preimage, NB gives two")
print("=" * 80)

ev = 'bijective_test'
ds = eval_configs[ev]
df_both = pd.read_parquet(f'{base}/both_model/{ev}.parquet')
df_nb = pd.read_parquet(f'{base}/nb_only_model/{ev}.parquet')

found = 0
for idx in range(len(df_both)):
    if found >= 1:
        break
    row_b = df_both.iloc[idx]
    row_n = df_nb.iloc[idx]
    gt = get_gt(row_b)
    gt_str_b = get_gt_str(row_b)
    gt_str_n = get_gt_str(row_n)

    b_scores = [score_response(r, gt_str_b, ds) for r in row_b['responses']]
    n_scores = [score_response(r, gt_str_n, ds) for r in row_n['responses']]

    # Find case where Both gives single answer (acc=1.0) and NB gives two (acc<=0.5)
    b_perfect = [i for i, (sc, ac) in enumerate(b_scores) if ac == 1.0]
    n_partial = [i for i, (sc, ac) in enumerate(n_scores) if 0 < ac <= 0.5]

    if b_perfect and n_partial:
        resp_b = row_b['responses'][b_perfect[0]]
        resp_n = row_n['responses'][n_partial[0]]

        # Check Both response actually submits 1 candidate
        ans_b = extract_answer_text(resp_b)
        cands_b = re.findall(r'\d+', ans_b) if ans_b else []
        ans_n = extract_answer_text(resp_n)
        cands_n = re.findall(r'\d+', ans_n) if ans_n else []

        if len(cands_b) == 1 and len(cands_n) == 2:
            func_m = re.search(r'def\s+(func_\d+)\s*\(', gt.get('ref_code', ''))
            func_name = func_m.group(1) if func_m else 'unknown'

            prompt = row_b['prompt']
            if isinstance(prompt, list):
                user_msg = [m['content'] for m in prompt if m.get('role') == 'user'][0]
            else:
                user_msg = str(prompt)

            print(f"\nFunction: {func_name}")
            print(f"Ground truth: ref_input={gt.get('ref_input')}, ref_output={gt.get('ref_output')}")
            print(f"Both score: {b_scores[b_perfect[0]]}, NB score: {n_scores[n_partial[0]]}")
            print(f"\n--- PROMPT (first 800 chars) ---")
            print(user_msg[:800])
            print(f"\n--- BOTH MODEL RESPONSE (first 2000 chars) ---")
            print(resp_b[:2000])
            print(f"\n[Answer]: {ans_b}  (candidates: {cands_b})")
            print(f"\n--- NB-ONLY MODEL RESPONSE (first 2000 chars) ---")
            print(resp_n[:2000])
            print(f"\n[Answer]: {ans_n}  (candidates: {cands_n})")
            found += 1

# ============================================================
# EXAMPLE SET 3: Lenchange — Both inverts, NB enumerates
# ============================================================
print("\n" + "=" * 80)
print("EXAMPLE SET 3: Lenchange — Both inverts successfully, NB-Only enumerates and fails")
print("=" * 80)

ev = 'lenchange'
ds = eval_configs[ev]
df_both = pd.read_parquet(f'{base}/both_model/{ev}.parquet')
df_nb = pd.read_parquet(f'{base}/nb_only_model/{ev}.parquet')

found = 0
for idx in range(len(df_both)):
    if found >= 1:
        break
    row_b = df_both.iloc[idx]
    row_n = df_nb.iloc[idx]
    gt = get_gt(row_b)
    gt_str_b = get_gt_str(row_b)
    gt_str_n = get_gt_str(row_n)

    b_scores = [score_response(r, gt_str_b, ds) for r in row_b['responses']]
    n_scores = [score_response(r, gt_str_n, ds) for r in row_n['responses']]

    b_best_ac = max(ac for sc, ac in b_scores)
    n_best_ac = max(ac for sc, ac in n_scores)

    if b_best_ac > 0.3 and n_best_ac == 0:
        best_b_idx = max(range(len(b_scores)), key=lambda i: b_scores[i][1])
        resp_b = row_b['responses'][best_b_idx]
        resp_n = row_n['responses'][0]

        # Check strategies
        has_inv = bool(re.search(r'(invert|inverse|reverse.*operation|undo|work.*backwards?|backward)', resp_b, re.I))
        has_enum = bool(re.search(r"(let'?s?\s+try|trying|trial|brute|check\s+if)", resp_n, re.I))

        if has_inv and has_enum:
            func_m = re.search(r'def\s+(lenop_\d+)\s*\(', gt.get('ref_code', ''))
            func_name = func_m.group(1) if func_m else 'unknown'

            prompt = row_b['prompt']
            if isinstance(prompt, list):
                user_msg = [m['content'] for m in prompt if m.get('role') == 'user'][0]
            else:
                user_msg = str(prompt)

            print(f"\nFunction: {func_name}, num_answers={gt.get('num_answers')}")
            print(f"Given output: {gt.get('ref_output')}")
            print(f"Both score: {b_scores[best_b_idx]}, NB score: {n_scores[0]}")
            print(f"Both response length: {len(resp_b)}, NB response length: {len(resp_n)}")
            print(f"\n--- PROMPT (first 1000 chars) ---")
            print(user_msg[:1000])
            print(f"\n--- BOTH MODEL RESPONSE (first 2500 chars) ---")
            print(resp_b[:2500])
            ans_b = extract_answer_text(resp_b)
            print(f"\n[Answer]: {ans_b}")
            print(f"\n--- NB-ONLY MODEL RESPONSE (first 2500 chars) ---")
            print(resp_n[:2500])
            ans_n = extract_answer_text(resp_n)
            print(f"\n[Answer]: {ans_n}")
            found += 1

# ============================================================
# EXAMPLE SET 4: NB-Only wins — func_23 (SPN) verification catches error
# ============================================================
print("\n" + "=" * 80)
print("EXAMPLE SET 4: func_23 (SPN) — NB-Only verification catches Both's inversion error")
print("=" * 80)

ev = 'bijective_inverse'
ds = eval_configs[ev]
df_both = pd.read_parquet(f'{base}/both_model/{ev}.parquet')
df_nb = pd.read_parquet(f'{base}/nb_only_model/{ev}.parquet')

found = 0
for idx in range(len(df_both)):
    if found >= 1:
        break
    row_b = df_both.iloc[idx]
    row_n = df_nb.iloc[idx]
    gt = get_gt(row_b)

    func_m = re.search(r'def\s+(func_\d+)\s*\(', gt.get('ref_code', ''))
    func_name = func_m.group(1) if func_m else ''
    if func_name != 'func_23':
        continue

    gt_str_b = get_gt_str(row_b)
    gt_str_n = get_gt_str(row_n)

    b_scores = [score_response(r, gt_str_b, ds) for r in row_b['responses']]
    n_scores = [score_response(r, gt_str_n, ds) for r in row_n['responses']]

    b_any = any(ac > 0 for sc, ac in b_scores)
    n_any = any(ac > 0 for sc, ac in n_scores)

    if n_any and not b_any:
        best_n_idx = max(range(len(n_scores)), key=lambda i: n_scores[i][1])
        resp_b = row_b['responses'][0]
        resp_n = row_n['responses'][best_n_idx]

        prompt = row_b['prompt']
        if isinstance(prompt, list):
            user_msg = [m['content'] for m in prompt if m.get('role') == 'user'][0]
        else:
            user_msg = str(prompt)

        print(f"\nFunction: {func_name}")
        print(f"Ground truth input: {gt.get('ref_input')}")
        print(f"Given output: {gt.get('ref_output')}")
        print(f"Both scores: {b_scores}, NB scores: {n_scores}")
        print(f"\n--- PROMPT (first 1000 chars) ---")
        print(user_msg[:1000])
        print(f"\n--- BOTH MODEL RESPONSE (first 2000 chars) ---")
        print(resp_b[:2000])
        ans_b = extract_answer_text(resp_b)
        print(f"\n[Answer]: {ans_b}")
        print(f"\n--- NB-ONLY MODEL RESPONSE (first 2500 chars) ---")
        print(resp_n[:2500])
        ans_n = extract_answer_text(resp_n)
        print(f"\n[Answer]: {ans_n}")
        found += 1

# ============================================================
# EXAMPLE SET 5: NB test — Both models succeed, different strategies
# ============================================================
print("\n" + "=" * 80)
print("EXAMPLE SET 5: NB test — Both models correct, different strategies")
print("=" * 80)

ev = 'nonbijective_test'
ds = eval_configs[ev]
df_both = pd.read_parquet(f'{base}/both_model/{ev}.parquet')
df_nb = pd.read_parquet(f'{base}/nb_only_model/{ev}.parquet')

found = 0
for idx in range(len(df_both)):
    if found >= 1:
        break
    row_b = df_both.iloc[idx]
    row_n = df_nb.iloc[idx]
    gt = get_gt(row_b)
    gt_str_b = get_gt_str(row_b)
    gt_str_n = get_gt_str(row_n)

    b_scores = [score_response(r, gt_str_b, ds) for r in row_b['responses']]
    n_scores = [score_response(r, gt_str_n, ds) for r in row_n['responses']]

    # Both models get it right
    b_perfect = [i for i, (sc, ac) in enumerate(b_scores) if ac == 1.0]
    n_perfect = [i for i, (sc, ac) in enumerate(n_scores) if ac == 1.0]

    if b_perfect and n_perfect:
        resp_b = row_b['responses'][b_perfect[0]]
        resp_n = row_n['responses'][n_perfect[0]]

        # Check different strategies
        b_has_enum = bool(re.search(r"(let'?s?\s+try|trying|trial|brute)", resp_b, re.I))
        n_has_enum = bool(re.search(r"(let'?s?\s+try|trying|trial|brute)", resp_n, re.I))
        b_has_case = bool(re.search(r"(case\s*[12]|two\s+cases|two\s+possible)", resp_b, re.I))

        if not b_has_enum and n_has_enum and b_has_case:
            func_m = re.search(r'def\s+(func_\d+|nb_\d+)\s*\(', gt.get('ref_code', ''))
            func_name = func_m.group(1) if func_m else 'unknown'

            prompt = row_b['prompt']
            if isinstance(prompt, list):
                user_msg = [m['content'] for m in prompt if m.get('role') == 'user'][0]
            else:
                user_msg = str(prompt)

            print(f"\nFunction: {func_name}")
            print(f"num_answers: {gt.get('num_answers')}")
            print(f"ref_output: {gt.get('ref_output')}")
            print(f"Both score: {b_scores[b_perfect[0]]}, NB score: {n_scores[n_perfect[0]]}")
            print(f"Both response length: {len(resp_b)}, NB response length: {len(resp_n)}")
            print(f"\n--- PROMPT (first 800 chars) ---")
            print(user_msg[:800])
            print(f"\n--- BOTH MODEL RESPONSE (first 2000 chars) ---")
            print(resp_b[:2000])
            ans_b = extract_answer_text(resp_b)
            print(f"\n[Answer]: {ans_b}")
            print(f"\n--- NB-ONLY MODEL RESPONSE (first 2000 chars) ---")
            print(resp_n[:2000])
            ans_n = extract_answer_text(resp_n)
            print(f"\n[Answer]: {ans_n}")
            found += 1
