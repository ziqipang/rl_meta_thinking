import json
import re
from typing import Iterable

from . import codeio as _codeio


do_print = False
original_print = print


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

    max_candidates = 5
    unique_candidates = _dedupe_keep_order(solution_candidates)[:max_candidates]

    try:
        ref_fn = _codeio._build_codeio_ref_fn(str(ref_code), funcname)
        if not callable(ref_fn):
            print("[Error] Reference function is not callable.")
            return -1, 0
    except Exception as e:
        print(f"[Error] Failed to build reference function: {e}")
        return -1, 0

    correct: list[str] = []
    for cand in unique_candidates:
        try:
            execed_output = ref_fn(str(cand))
        except Exception:
            continue
        if _codeio.is_close(execed_output, ref_output):
            correct.append(cand)

    # Partial credit: each distinct correct preimage contributes 1/num_answers.
    score = min(len(correct), num_answers) / float(num_answers)
    acc = score
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
