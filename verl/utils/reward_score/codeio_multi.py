import json
import re
from typing import Iterable

from . import codeio as _codeio


do_print = False
original_print = print
# Legacy datasets use a format bonus plus anti-stuffing accuracy. Newer
# multi-response datasets opt into alternate scoring via
# `ground_truth.gt_info.multi_response_scoring`.
_FORMAT_WEIGHT = 0.1


def print(*args, **kwargs):  # type: ignore
    if do_print:
        original_print(*args, **kwargs)


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
    """Parse the `<answer>...</answer>` content into digit-string candidates.

    The prompt asks for comma-separated digit strings, but we accept any text
    that contains digit substrings inside the `<answer>` block.
    """
    # Extract contiguous digit spans (preserves leading zeros).
    return re.findall(r"\d+", answer_text)


def _get_num_answers_from_ground_truth(ground_truth: dict[str, object]) -> int:
    """Read `num_answers` with backward-compatible fallbacks.

    Newer datasets may store this under `gt_info.num_answers`.
    Older datasets store it at the top-level as `num_answers`.
    """
    # Preferred: nested metadata (`gt_info`) for eval variants.
    gt_info = ground_truth.get("gt_info")
    if isinstance(gt_info, dict):
        try:
            n = int(gt_info.get("num_answers"))  # type: ignore[arg-type]
            if n > 0:
                return n
        except Exception:
            pass

    # Back-compat: top-level.
    try:
        n = int(ground_truth.get("num_answers", 2))
        if n > 0:
            return n
    except Exception:
        pass

    return 2


def _get_scoring_mode_from_ground_truth(ground_truth: dict[str, object]) -> str | None:
    gt_info = ground_truth.get("gt_info")
    if not isinstance(gt_info, dict):
        return None

    mode = gt_info.get("multi_response_scoring")
    if isinstance(mode, str) and mode:
        return mode
    return None


def _compute_score_multi_preimage_legacy(
    solution_candidates: list[str],
    ground_truth: dict[str, object],
) -> tuple[float, float]:
    if _codeio._CODEIO_FUNC_ENV is None:
        print("[Error] codeio-multi requires `projects.datasets.create_string_data` to be importable.")
        return -1, 0

    ref_output = ground_truth["ref_output"]
    ref_code = ground_truth["ref_code"]
    funcname = str(ground_truth.get("funcname", "main_solution"))

    num_answers = _get_num_answers_from_ground_truth(ground_truth)

    unique_candidates = _dedupe_keep_order(solution_candidates)
    format_ok = len(solution_candidates) == num_answers and len(unique_candidates) == num_answers
    format_score = 1.0 if format_ok else 0.0

    try:
        ref_fn = _codeio._build_codeio_ref_fn(str(ref_code), funcname)
        if not callable(ref_fn):
            print("[Error] Reference function is not callable.")
            return -1, 0
    except Exception as e:
        print(f"[Error] Failed to build reference function: {e}")
        return -1, 0

    missing = max(0, num_answers - len(unique_candidates))

    max_eval_unique = 64
    num_unevaluated = max(0, len(unique_candidates) - max_eval_unique)
    eval_candidates = unique_candidates[:max_eval_unique]

    wrong = 0
    wrong_budget = num_answers - missing - num_unevaluated
    if wrong_budget <= 0:
        acc = 0.0
    else:
        for cand in eval_candidates:
            if wrong >= wrong_budget:
                break
            try:
                execed_output = ref_fn(str(cand))
                ok = bool(_codeio.is_close(execed_output, ref_output))
            except Exception:
                ok = False
            if not ok:
                wrong += 1

        effective_wrong = wrong + missing + num_unevaluated
        acc = 0.0 if effective_wrong >= num_answers else (num_answers - effective_wrong) / float(num_answers)

    score = float(_FORMAT_WEIGHT * format_score + (1.0 - _FORMAT_WEIGHT) * acc)
    return score, acc


def _compute_score_bijective_inverse(
    solution_candidates: list[str],
    ground_truth: dict[str, object],
) -> tuple[float, float]:
    if not solution_candidates:
        return 0.0, 0.0

    ref_input_obj = ground_truth.get("ref_input")
    if not isinstance(ref_input_obj, dict) or "x" not in ref_input_obj:
        print("[Error] Missing ref_input.x for bijective multi-response scoring.")
        return -1, 0

    ref_input = str(ref_input_obj["x"])
    correct_or_not = 1 if any(_codeio.is_close(str(candidate), ref_input) for candidate in solution_candidates) else 0
    acc = correct_or_not / float(len(solution_candidates))
    return acc, acc


def _compute_score_nonbijective_preimage(
    solution_candidates: list[str],
    ground_truth: dict[str, object],
) -> tuple[float, float]:
    if _codeio._CODEIO_FUNC_ENV is None:
        print("[Error] codeio-multi requires `projects.datasets.create_string_data` to be importable.")
        return -1, 0

    ref_output = ground_truth["ref_output"]
    ref_code = ground_truth["ref_code"]
    funcname = str(ground_truth.get("funcname", "main_solution"))
    num_answers = _get_num_answers_from_ground_truth(ground_truth)

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
        try:
            execed_output = ref_fn(str(cand))
        except Exception:
            continue
        if _codeio.is_close(execed_output, ref_output):
            correct_response_num += 1
            if correct_response_num >= num_answers:
                break

    acc = min(correct_response_num, num_answers) / float(num_answers)
    return acc, acc


def compute_score_multi_preimage(solution_candidates: list[str], ground_truth: dict[str, object]) -> tuple[float, float]:
    scoring_mode = _get_scoring_mode_from_ground_truth(ground_truth)
    if scoring_mode == "bijective_inverse":
        return _compute_score_bijective_inverse(solution_candidates, ground_truth)
    if scoring_mode == "nonbijective_preimage":
        return _compute_score_nonbijective_preimage(solution_candidates, ground_truth)
    return _compute_score_multi_preimage_legacy(solution_candidates, ground_truth)


def compute_score(solution_str: str, ground_truth: str, task: str = "codeio-nb-multi-preimage") -> tuple[float, float]:
    if task.startswith("codeio-nb-multi-"):
        task = task[len("codeio-nb-multi-") :]
    elif task.startswith("codeio-multi-"):
        task = task[len("codeio-multi-") :]

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

    if not task.startswith("preimage"):
        raise NotImplementedError(f"Task {task} not implemented in codeio_multi.")

    answer_text = str(extracted["output"])
    candidates = _parse_multi_preimage_answer(answer_text)
    return compute_score_multi_preimage(candidates, ground_truth_obj)
