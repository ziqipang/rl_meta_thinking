import json
import re
from typing import Iterable

from . import codeio as _codeio


do_print = False
original_print = print
# Reward shaping:
# - `format_score` is 1.0 iff the model outputs exactly `num_answers` distinct
#   digit-string candidates (parsed from the <answer>...</answer> block).
# - `acc` is an "anti-stuffing" accuracy: we score against a worst-case
#   selection of `num_answers` candidates by matching wrong candidates first.
# The returned `score` blends the two to keep rewards in [0, 1].
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


def compute_score_multi_preimage(solution_candidates: list[str], ground_truth: dict[str, object]) -> tuple[float, float]:
    if _codeio._CODEIO_FUNC_ENV is None:
        print("[Error] codeio-multi requires `projects.datasets.create_string_data` to be importable.")
        return -1, 0

    ref_output = ground_truth["ref_output"]
    ref_code = ground_truth["ref_code"]
    funcname = str(ground_truth.get("funcname", "main_solution"))

    try:
        num_answers = int(ground_truth.get("num_answers", 2))
    except Exception:
        num_answers = 2
    if num_answers <= 0:
        print("[Error] Invalid num_answers in ground_truth.")
        return -1, 0

    # NOTE: `num_answers` is produced by `projects/datasets/create_string_data_multi.py`
    # and stored in the sample's ground_truth JSON. We intentionally do *not*
    # hard-code it here.
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

    # Anti-stuffing accuracy: "match wrong ones first" (order-independent).
    #
    # Intuition: We assume an adversary selects `num_answers` candidates from the
    # model's set and tries to minimize correctness by picking wrong ones first.
    #
    # If there are >= num_answers wrong candidates (or missing distinct answers),
    # accuracy is forced to 0.
    #
    # We treat missing distinct answers as wrong, so duplicates don't help.
    missing = max(0, num_answers - len(unique_candidates))

    # Defensive cap: if the model outputs a huge number of candidates, treat the
    # tail as wrong (conservative) to avoid slow reward computation and to
    # discourage stuffing.
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

    # Blend format + accuracy into a single bounded training reward.
    score = float(_FORMAT_WEIGHT * format_score + (1.0 - _FORMAT_WEIGHT) * acc)
    return score, acc


def compute_score(solution_str: str, ground_truth: str, task: str = "codeio-nb-multi-preimage") -> tuple[float, float]:
    if task.startswith("codeio-nb-multi-"):
        task = task[len("codeio-nb-multi-") :]
    elif task.startswith("codeio-multi-"):
        task = task[len("codeio-multi-") :]

    ground_truth_obj = json.loads(ground_truth)

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
