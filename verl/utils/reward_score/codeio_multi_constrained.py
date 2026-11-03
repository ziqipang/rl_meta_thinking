"""
Reward scorer for constrained multi-preimage tasks.

Extends the standard ``codeio_multi`` scorer to additionally verify that
each submitted answer satisfies digit-position constraints stored in the
``ground_truth.digit_constraints`` field.

A candidate answer must:
1. Produce the correct output when fed through the reference function.
2. Satisfy ALL digit constraints (i.e. the digit at each constrained
   position must NOT be in the excluded set).

Only answers meeting both criteria count as correct.
"""

import json
import re
from typing import Iterable

from . import codeio as _codeio


do_print = False
original_print = print


def print(*args, **kwargs):  # type: ignore
    if do_print:
        original_print(*args, **kwargs)


# ---------------------------------------------------------------------------
# Helpers (shared with codeio_multi.py)
# ---------------------------------------------------------------------------

def _dedupe_keep_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _parse_multi_preimage_answer(answer_text: str) -> list[str]:
    return re.findall(r"\d+", answer_text)


def _compute_optional_shortfall_accuracy(
    *,
    correct_unique: int,
    submitted_count: int,
    required_count: int,
) -> float:
    if required_count <= 0 or submitted_count <= 0 or correct_unique <= 0:
        return 0.0
    return min(correct_unique, required_count) / float(max(required_count, submitted_count))


def _get_num_answers(ground_truth: dict[str, object]) -> int:
    gt_info = ground_truth.get("gt_info")
    if isinstance(gt_info, dict):
        try:
            n = int(gt_info.get("num_answers"))  # type: ignore[arg-type]
            if n > 0:
                return n
        except Exception:
            pass
    try:
        n = int(ground_truth.get("num_answers", 2))
        if n > 0:
            return n
    except Exception:
        pass
    return 2


# ---------------------------------------------------------------------------
# Constraint checking
# ---------------------------------------------------------------------------

def _parse_digit_constraints(ground_truth: dict[str, object]) -> dict[int, list[int]] | None:
    """Extract digit constraints from ground_truth.

    Returns ``{position (int): [excluded_digits (int), ...]}`` or ``None``
    if no constraints are present.
    """
    raw = ground_truth.get("digit_constraints")
    if not isinstance(raw, dict) or not raw:
        return None
    try:
        return {int(pos): [int(d) for d in excluded] for pos, excluded in raw.items()}
    except (ValueError, TypeError):
        return None


def _input_satisfies_constraints(x: str, constraints: dict[int, list[int]]) -> bool:
    """Check whether digit string ``x`` satisfies all constraints.

    If the input is shorter than a constrained position the constraint
    is considered violated.
    """
    for pos, excluded in constraints.items():
        if pos >= len(x):
            return False
        digit_val = int(x[pos])
        if digit_val in excluded:
            return False
    return True


# ---------------------------------------------------------------------------
# Main scoring function
# ---------------------------------------------------------------------------

def _compute_score_constrained_preimage(
    solution_candidates: list[str],
    ground_truth: dict[str, object],
) -> tuple[float, float]:
    ref_output = ground_truth["ref_output"]
    ref_code = ground_truth["ref_code"]
    funcname = str(ground_truth.get("funcname", "main_solution"))
    num_answers = _get_num_answers(ground_truth)
    constraints = _parse_digit_constraints(ground_truth)

    if num_answers <= 0:
        return -1, 0
    if not solution_candidates:
        return 0.0, 0.0

    try:
        ref_fn = _codeio._build_codeio_ref_fn(str(ref_code), funcname)
        if not callable(ref_fn):
            print("[Error] Reference function is not callable.")
            return -1, 0
    except Exception as e:
        print(f"[Error] Failed to build reference function: {e}")
        return -1, 0

    unique_candidates = _dedupe_keep_order(solution_candidates)
    correct_response_num = 0
    for cand in unique_candidates:
        # Check digit constraints first (cheap)
        if constraints is not None and not _input_satisfies_constraints(cand, constraints):
            print(f"[Constraint Violation] Candidate '{cand}' violates digit constraints.")
            continue

        # Check functional correctness
        try:
            execed_output = ref_fn(str(cand))
        except Exception:
            continue
        if _codeio.is_close(execed_output, ref_output):
            correct_response_num += 1
            if correct_response_num >= num_answers:
                break

    acc = _compute_optional_shortfall_accuracy(
        correct_unique=correct_response_num,
        submitted_count=len(solution_candidates),
        required_count=num_answers,
    )
    return acc, acc


def compute_score(solution_str: str, ground_truth: str, task: str = "") -> tuple[float, float]:
    ground_truth_obj = json.loads(ground_truth)
    if not isinstance(ground_truth_obj, dict):
        return -1, 0

    global do_print
    do_print = False

    if "<answer>" not in solution_str.lower():
        return -1, 0

    extracted = _codeio.extract_answer(solution_str)
    if not isinstance(extracted, dict) or "output" not in extracted:
        return -1, 0

    answer_text = str(extracted["output"])
    candidates = _parse_multi_preimage_answer(answer_text)
    return _compute_score_constrained_preimage(candidates, ground_truth_obj)
