"""
Tests for the constrained multi-response dataset generator and reward function.

Covers:
1. Constraint generation logic
2. Constraint-to-prompt text formatting
3. Constrained input generation
4. Constrained preimage search
5. End-to-end sample generation
6. Reward function with constraint checking
7. Reward routing
"""

import json
import random
import sys
from pathlib import Path

# Ensure repo root is on path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from projects.datasets.create_multi_response_with_none_constrained import (
    generate_random_constraints,
    constraints_to_prompt_text,
    constraints_to_json_serializable,
    constraints_from_json,
    input_satisfies_constraints,
    _random_constrained_digit_string,
    find_output_with_k_constrained_preimages,
    _generate_constrained_nonbijective_sample,
)
from projects.datasets import create_string_data as base
from verl.utils.reward_score.codeio_multi_constrained import (
    _parse_digit_constraints,
    _input_satisfies_constraints as reward_satisfies_constraints,
    _compute_score_constrained_preimage,
    compute_score,
)
from verl.utils.reward_score import _default_compute_score


# ===================================================================
# Test 1: Constraint generation
# ===================================================================

def test_generate_random_constraints():
    random.seed(0)
    for _ in range(50):
        num_pos = random.randint(2, 3)
        available = list(range(5))
        constraints = generate_random_constraints(
            num_positions=num_pos,
            available_positions=available,
            min_excluded=3,
            max_excluded=6,
        )
        assert len(constraints) == num_pos, f"Expected {num_pos} positions, got {len(constraints)}"
        for pos, excluded in constraints.items():
            assert pos in available, f"Position {pos} not in available {available}"
            assert 3 <= len(excluded) <= 6, f"Expected 3-6 excluded, got {len(excluded)}"
            assert len(set(excluded)) == len(excluded), "Excluded digits should be unique"
            assert all(0 <= d <= 9 for d in excluded), "Excluded digits must be 0-9"
    print("[PASS] test_generate_random_constraints")


# ===================================================================
# Test 2: Constraint text formatting
# ===================================================================

def test_constraints_to_prompt_text():
    constraints = {0: [3, 5, 7], 2: [0, 1, 2, 8]}
    text = constraints_to_prompt_text(constraints)
    assert "1st digit" in text
    assert "3rd digit" in text
    assert "3, 5, 7" in text
    assert "0, 1, 2, 8" in text
    # Should have two lines
    lines = [l for l in text.strip().split("\n") if l.strip()]
    assert len(lines) == 2
    print("[PASS] test_constraints_to_prompt_text")


# ===================================================================
# Test 3: JSON round-trip
# ===================================================================

def test_constraints_json_roundtrip():
    constraints = {0: [1, 2, 3], 4: [5, 6, 7, 8, 9]}
    json_obj = constraints_to_json_serializable(constraints)
    assert isinstance(json_obj, dict)
    assert all(isinstance(k, str) for k in json_obj.keys())
    # Verify JSON serializable
    serialized = json.dumps(json_obj)
    deserialized = json.loads(serialized)
    recovered = constraints_from_json(deserialized)
    assert recovered == constraints
    print("[PASS] test_constraints_json_roundtrip")


# ===================================================================
# Test 4: input_satisfies_constraints
# ===================================================================

def test_input_satisfies_constraints():
    constraints = {0: [3, 5, 7], 2: [0, 1, 2]}

    # Valid: 1st digit=1 (ok), 3rd digit=5 (ok)
    assert input_satisfies_constraints("195", constraints) is True
    # Invalid: 1st digit=3 (excluded)
    assert input_satisfies_constraints("395", constraints) is False
    # Invalid: 3rd digit=0 (excluded)
    assert input_satisfies_constraints("190", constraints) is False
    # Invalid: too short (position 2 doesn't exist)
    assert input_satisfies_constraints("19", constraints) is False
    # Valid: longer input, constrained positions ok
    assert input_satisfies_constraints("1953", constraints) is True
    # Invalid: 1st digit=5 (excluded)
    assert input_satisfies_constraints("5953", constraints) is False
    print("[PASS] test_input_satisfies_constraints")


# ===================================================================
# Test 5: Constrained digit string generation
# ===================================================================

def test_random_constrained_digit_string():
    random.seed(42)
    constraints = {0: [0, 1, 2, 3, 4], 1: [5, 6, 7, 8, 9]}

    for _ in range(200):
        s = _random_constrained_digit_string(min_len=3, max_len=5, constraints=constraints)
        assert s is not None, "Should be able to generate constrained string"
        assert 3 <= len(s) <= 5
        assert int(s[0]) not in constraints[0], f"1st digit {s[0]} should not be in {constraints[0]}"
        assert int(s[1]) not in constraints[1], f"2nd digit {s[1]} should not be in {constraints[1]}"
        assert input_satisfies_constraints(s, constraints)

    # Impossible constraint: all digits excluded for position 0
    impossible = {0: list(range(10))}
    s = _random_constrained_digit_string(min_len=2, max_len=4, constraints=impossible)
    assert s is None, "Should return None for impossible constraints"
    print("[PASS] test_random_constrained_digit_string")


# ===================================================================
# Test 6: Constrained preimage search
# ===================================================================

def test_find_output_with_k_constrained_preimages():
    random.seed(42)
    all_set, train_set, _ = base._build_function_sets(
        function_kind="nonbijective_v2", nb_v2_split="balanced",
    )
    # Pick a known lossy function for reliable collision finding
    func_name = "nb_00"
    func = train_set[func_name]

    constraints = {0: [0, 1, 2], 1: [7, 8, 9]}  # mild constraints

    result = find_output_with_k_constrained_preimages(
        func, k=2, attempts=5000, min_len=3, max_len=6, constraints=constraints,
    )
    assert result is not None, "Should find constrained preimages for a lossy function"
    output, preimages = result
    assert len(preimages) >= 2
    for p in preimages:
        assert input_satisfies_constraints(p, constraints), f"Preimage {p} violates constraints"
        assert func(p) == output, f"Preimage {p} doesn't produce expected output"
    print("[PASS] test_find_output_with_k_constrained_preimages")


# ===================================================================
# Test 7: End-to-end sample generation
# ===================================================================

def test_generate_constrained_sample():
    random.seed(42)
    _, train_set, _ = base._build_function_sets(
        function_kind="nonbijective_v2", nb_v2_split="balanced",
    )

    # Try multiple times since generation is stochastic
    sample = None
    for _ in range(100):
        sample = _generate_constrained_nonbijective_sample(
            custom_functions=train_set,
            depth=1,
            stage=1,
            num_answers=2,
            min_len=3,
            max_len=6,
            search_attempts=4000,
            estimate_attempts=0,
            estimate_max_unique=50,
        )
        if sample is not None:
            break

    assert sample is not None, "Should generate at least one constrained sample"

    # Verify data_source
    assert "-with-constraints" in sample["data_source"]
    assert "codeio-nb-multi-preimage" in sample["data_source"]

    # Verify prompt has constraints
    prompt_content = sample["prompt"][0]["content"]
    assert "cannot be" in prompt_content
    assert "Additionally" in prompt_content

    # Verify ground_truth has constraints
    gt = json.loads(sample["reward_model"]["ground_truth"])
    assert "digit_constraints" in gt
    constraints = constraints_from_json(gt["digit_constraints"])
    assert len(constraints) >= 2

    # Verify known_preimages satisfy constraints
    for preimage in gt["known_preimages"]:
        assert input_satisfies_constraints(preimage, constraints), \
            f"Known preimage {preimage} violates constraints {constraints}"

    # Verify extra_info has constraints
    assert "digit_constraints" in sample["extra_info"]
    assert sample["extra_info"]["digit_constraints"] == gt["digit_constraints"]

    # Verify gt_info scoring mode
    assert gt["gt_info"]["multi_response_scoring"] == "nonbijective_preimage_constrained"

    print("[PASS] test_generate_constrained_sample")
    return sample


# ===================================================================
# Test 8: Reward function — correct answers satisfying constraints
# ===================================================================

def test_reward_correct_with_constraints():
    random.seed(42)
    _, train_set, _ = base._build_function_sets(
        function_kind="nonbijective_v2", nb_v2_split="balanced",
    )
    sample = None
    for _ in range(100):
        sample = _generate_constrained_nonbijective_sample(
            custom_functions=train_set, depth=1, stage=1,
            num_answers=2, min_len=3, max_len=6,
            search_attempts=4000, estimate_attempts=0, estimate_max_unique=50,
        )
        if sample is not None:
            break
    assert sample is not None

    gt_str = sample["reward_model"]["ground_truth"]
    gt = json.loads(gt_str)
    preimages = gt["known_preimages"]

    # Build a correct solution string
    answer_str = ",".join(preimages[:2])
    solution_str = f"<think>reasoning</think><answer> {answer_str} </answer>"

    score, acc = compute_score(solution_str, gt_str)
    assert score > 0, f"Expected positive score for correct answers, got {score}"
    assert acc > 0, f"Expected positive accuracy, got {acc}"
    print(f"[PASS] test_reward_correct_with_constraints (score={score}, acc={acc})")


# ===================================================================
# Test 9: Reward function — constraint-violating answers get rejected
# ===================================================================

def test_reward_constraint_violation():
    # Build a ground_truth with known constraints
    constraints = {0: [1, 2, 3, 4, 5, 6, 7, 8, 9]}  # only digit 0 allowed at pos 0

    # nb_00: cyclic floor-average
    all_set, _, _ = base._build_function_sets(
        function_kind="nonbijective_v2", nb_v2_split="balanced",
    )
    func = all_set["nb_00"]

    # Find valid constrained preimages
    random.seed(123)
    valid_input = None
    for _ in range(1000):
        x = _random_constrained_digit_string(min_len=3, max_len=3, constraints=constraints)
        if x is not None:
            try:
                y = func(x)
                if isinstance(y, str):
                    valid_input = x
                    break
            except Exception:
                pass
    assert valid_input is not None, "Should find at least one valid constrained input"
    output = func(valid_input)

    # Build ref_code
    ref_code = base.generate_full_code("nb_00(x)", stage=1, custom_functions=all_set)

    gt = {
        "ref_input": {"x": valid_input},
        "ref_output": output,
        "ref_code": ref_code,
        "funcname": "main_solution",
        "num_answers": 1,
        "gt_info": {"num_answers": 1, "allow_fewer_answers": True, "multi_response_scoring": "nonbijective_preimage_constrained"},
        "known_preimages": [valid_input],
        "preimage_lower_bound": 1,
        "digit_constraints": constraints_to_json_serializable(constraints),
    }
    gt_str = json.dumps(gt)

    # Answer that violates constraint (1st digit not 0)
    violating = "1" + valid_input[1:]
    # Check it would produce correct output (may or may not)
    solution_str = f"<think>test</think><answer> {violating} </answer>"
    score, acc = compute_score(solution_str, gt_str)
    # Even if functionally correct, constraint violation should mean no credit
    # (unless the violating input also happens to produce the right output AND satisfy constraints)
    if int(violating[0]) in constraints[0]:
        assert score == 0.0, f"Constraint-violating answer should score 0, got {score}"
        print("[PASS] test_reward_constraint_violation (violating answer correctly scored 0)")
    else:
        print("[PASS] test_reward_constraint_violation (answer didn't violate, adjusted test)")

    # Now test with a definitely valid answer
    solution_str_valid = f"<think>test</think><answer> {valid_input} </answer>"
    score_valid, acc_valid = compute_score(solution_str_valid, gt_str)
    assert score_valid > 0, f"Valid constrained answer should score > 0, got {score_valid}"
    print(f"[PASS] test_reward_constraint_violation valid case (score={score_valid})")


# ===================================================================
# Test 10: Reward routing in _default_compute_score
# ===================================================================

def test_reward_routing():
    # Verify routing dispatches constrained data_source correctly
    random.seed(42)
    _, train_set, _ = base._build_function_sets(
        function_kind="nonbijective_v2", nb_v2_split="balanced",
    )
    sample = None
    for _ in range(100):
        sample = _generate_constrained_nonbijective_sample(
            custom_functions=train_set, depth=1, stage=1,
            num_answers=2, min_len=3, max_len=6,
            search_attempts=4000, estimate_attempts=0, estimate_max_unique=50,
        )
        if sample is not None:
            break
    assert sample is not None

    gt_str = sample["reward_model"]["ground_truth"]
    gt = json.loads(gt_str)
    preimages = gt["known_preimages"]
    answer_str = ",".join(preimages[:2])
    solution_str = f"<think>reasoning</think><answer> {answer_str} </answer>"

    data_source = sample["data_source"]
    assert "-with-constraints" in data_source

    # Call through the routing function
    result = _default_compute_score(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=gt_str,
    )
    if isinstance(result, tuple):
        score, acc = result
    else:
        score = float(result)
        acc = score
    assert score > 0, f"Routed score should be positive for correct answers, got {score}"
    print(f"[PASS] test_reward_routing (score={score})")


# ===================================================================
# Test 11: No-answer and missing-tag edge cases
# ===================================================================

def test_reward_edge_cases():
    gt = {
        "ref_input": {"x": "012"},
        "ref_output": "000",
        "ref_code": "def main_solution(x):\n    return '000'",
        "funcname": "main_solution",
        "num_answers": 2,
        "gt_info": {"num_answers": 2, "allow_fewer_answers": True, "multi_response_scoring": "nonbijective_preimage_constrained"},
        "known_preimages": ["012", "345"],
        "preimage_lower_bound": 2,
        "digit_constraints": {"0": [9]},
    }
    gt_str = json.dumps(gt)

    # No <answer> tag
    score, acc = compute_score("just some text without answer tags", gt_str)
    assert score == -1, f"Missing <answer> should return -1, got {score}"

    # Empty answer
    score, acc = compute_score("<think>hmm</think><answer></answer>", gt_str)
    assert score == 0.0 or score == -1, f"Empty answer should return 0 or -1, got {score}"

    print("[PASS] test_reward_edge_cases")


# ===================================================================
# Test 12: Verify _parse_digit_constraints in reward module
# ===================================================================

def test_parse_digit_constraints():
    gt_with = {"digit_constraints": {"0": [1, 2, 3], "2": [5, 6]}}
    parsed = _parse_digit_constraints(gt_with)
    assert parsed == {0: [1, 2, 3], 2: [5, 6]}

    gt_without = {"ref_output": "test"}
    parsed = _parse_digit_constraints(gt_without)
    assert parsed is None

    gt_empty = {"digit_constraints": {}}
    parsed = _parse_digit_constraints(gt_empty)
    assert parsed is None

    print("[PASS] test_parse_digit_constraints")


# ===================================================================
# Main
# ===================================================================

if __name__ == "__main__":
    test_generate_random_constraints()
    test_constraints_to_prompt_text()
    test_constraints_json_roundtrip()
    test_input_satisfies_constraints()
    test_random_constrained_digit_string()
    test_find_output_with_k_constrained_preimages()
    test_generate_constrained_sample()
    test_reward_correct_with_constraints()
    test_reward_constraint_violation()
    test_reward_routing()
    test_reward_edge_cases()
    test_parse_digit_constraints()
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
