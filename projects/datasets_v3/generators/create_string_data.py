"""
Self-contained dataset generator for the v3 atomic digit-string functions.

This mirrors `projects/datasets/create_string_data.py`, but keeps all v3-specific
logic local to `projects/datasets_v3/generators/`.
"""

from __future__ import annotations

from argparse import ArgumentParser
import inspect
import json
import os
from pathlib import Path
import random
import string
import sys
from typing import Callable

try:
    from tqdm.auto import tqdm  # type: ignore
except Exception:  # pragma: no cover
    class _NoOpTqdm:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def update(self, n: int = 1) -> None:
            pass

        def close(self) -> None:
            pass

    def tqdm(*args, **kwargs):  # type: ignore
        return _NoOpTqdm()

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from projects.datasets_v3 import atomic_functions as atomic_lib  # type: ignore


FORWARD_PROMPT = (
    "You are given a code:\n\n{code}\n\n"
    "Can you predict the output of `main_solution(\"{input}\")` without writing any code? "
    "Please reason between the XML tags of <think> </think> and put the answer, which is ONLY the result string between <answer> </answer> tags. The final response should look like <think> ... </think><answer> ... </answer>"
)

BACKWARD_PROMPT = (
    "You are given a code:\n\n{code}\n\n"
    "You are told that `main_solution(x)` returns the string \"{output}\".\n"
    "What is the unique digit-string input `x`?\n\n"
    "Please reason between the XML tags of <think> </think>, and put the answer, which is ONLY the result string between <answer> </answer> tags. The final response should look like <think> ... </think><answer> ... </answer>"
)

PREIMAGE_PROMPT = (
    "You are given a code:\n\n{code}\n\n"
    "You are told that `main_solution(x)` returns the string \"{output}\".\n"
    "Find ONE possible digit-string input `x` such that `main_solution(x)` returns exactly that string.\n\n"
    "Please reason between the XML tags of <think> </think>, and put the answer, which is ONLY the result string between <answer> </answer> tags. The final response should look like <think> ... </think><answer> ... </answer>"
)


BIJECTIVE_FUNCTION_NAMES = [func.__name__ for func in atomic_lib.ALL_BIJECTIVE]
NONBIJECTIVE_FUNCTION_NAMES = [func.__name__ for func in atomic_lib.ALL_NONBIJECTIVE]

DEFAULT_BIJECTIVE_TRAIN_FUNCTION_NAMES = [f"func_{i:02d}" for i in range(0, 18)]
DEFAULT_BIJECTIVE_TEST_FUNCTION_NAMES = [f"func_{i:02d}" for i in range(18, 24)]
DEFAULT_NONBIJECTIVE_TRAIN_FUNCTION_NAMES = [f"nb_{i:02d}" for i in range(0, 18)]
DEFAULT_NONBIJECTIVE_TEST_FUNCTION_NAMES = [f"nb_{i:02d}" for i in range(18, 24)]

ALL_FUNCTIONS: dict[str, Callable[[str], str]] = {
    name: getattr(atomic_lib, name)
    for name in BIJECTIVE_FUNCTION_NAMES + NONBIJECTIVE_FUNCTION_NAMES
}

HELPER_ORDER = [
    "_d",
    "_c",
    "_shift_all",
    "_shift_alt",
    "_shift_pos_linear",
    "_shift_halves",
    "_shift_mod3",
    "_rotl",
    "_swap_ends",
    "_swap_pairs",
    "_swap_halves",
    "_reverse_halves",
    "_odd_front",
    "_interleave_halves",
    "_reverse_inner",
    "_triplet_right",
    "_prefix_sum_emit",
    "_suffix_sum_emit",
    "_cyclic_pair",
    "_cyclic_triple",
    "_mirror_map",
    "_pair_sort_asc",
    "_pair_sort_desc",
    "_triplet_sort_asc",
    "_triplet_sort_desc",
    "_prefix_min",
    "_prefix_max",
    "_suffix_min",
    "_suffix_max",
    "_state_emit",
]

HELPER_DEPENDENCIES = {
    "_d": [],
    "_c": [],
    "_shift_all": ["_d", "_c"],
    "_shift_alt": ["_d", "_c"],
    "_shift_pos_linear": ["_d", "_c"],
    "_shift_halves": ["_d", "_c"],
    "_shift_mod3": ["_d", "_c"],
    "_rotl": [],
    "_swap_ends": [],
    "_swap_pairs": [],
    "_swap_halves": [],
    "_reverse_halves": [],
    "_odd_front": [],
    "_interleave_halves": [],
    "_reverse_inner": [],
    "_triplet_right": [],
    "_prefix_sum_emit": ["_d", "_c"],
    "_suffix_sum_emit": ["_d", "_c"],
    "_cyclic_pair": ["_d", "_c"],
    "_cyclic_triple": ["_d", "_c"],
    "_mirror_map": ["_d", "_c"],
    "_pair_sort_asc": [],
    "_pair_sort_desc": [],
    "_triplet_sort_asc": [],
    "_triplet_sort_desc": [],
    "_prefix_min": [],
    "_prefix_max": [],
    "_suffix_min": [],
    "_suffix_max": [],
    "_state_emit": ["_d", "_c"],
}

FUNCTION_DEPENDENCIES = {
    "func_00": ["_shift_all"],
    "func_01": ["_rotl", "_shift_all"],
    "func_02": ["_swap_ends", "_shift_all"],
    "func_03": ["_swap_pairs", "_shift_all"],
    "func_04": ["_swap_halves", "_shift_all"],
    "func_05": ["_reverse_halves", "_shift_all"],
    "func_06": ["_odd_front", "_shift_all"],
    "func_07": ["_interleave_halves", "_shift_all"],
    "func_08": ["_reverse_inner", "_shift_all"],
    "func_09": ["_triplet_right", "_shift_all"],
    "func_10": ["_shift_alt"],
    "func_11": ["_rotl", "_shift_alt"],
    "func_12": ["_shift_pos_linear"],
    "func_13": ["_shift_pos_linear"],
    "func_14": ["_shift_halves"],
    "func_15": ["_shift_mod3"],
    "func_16": ["_shift_pos_linear"],
    "func_17": ["_swap_pairs", "_shift_pos_linear"],
    "func_18": ["_odd_front", "_shift_halves"],
    "func_19": ["_swap_halves", "_shift_mod3"],
    "func_20": ["_prefix_sum_emit"],
    "func_21": ["_prefix_sum_emit"],
    "func_22": ["_prefix_sum_emit"],
    "func_23": ["_suffix_sum_emit"],
    "nb_00": ["_cyclic_pair", "_shift_all"],
    "nb_01": ["_cyclic_pair", "_shift_all"],
    "nb_02": ["_cyclic_pair", "_shift_all"],
    "nb_03": ["_cyclic_pair", "_shift_all"],
    "nb_04": ["_cyclic_triple", "_shift_all"],
    "nb_05": ["_cyclic_triple", "_shift_all"],
    "nb_06": ["_cyclic_triple", "_shift_all"],
    "nb_07": ["_cyclic_triple", "_shift_all"],
    "nb_08": ["_mirror_map", "_shift_all"],
    "nb_09": ["_mirror_map", "_shift_all"],
    "nb_10": ["_mirror_map", "_shift_all"],
    "nb_11": ["_mirror_map", "_shift_all"],
    "nb_12": ["_pair_sort_asc", "_shift_all"],
    "nb_13": ["_pair_sort_desc", "_shift_all"],
    "nb_14": ["_triplet_sort_asc", "_shift_all"],
    "nb_15": ["_triplet_sort_desc", "_shift_all"],
    "nb_16": ["_prefix_min", "_shift_all"],
    "nb_17": ["_prefix_max", "_shift_all"],
    "nb_18": ["_suffix_min", "_shift_all"],
    "nb_19": ["_suffix_max", "_shift_all"],
    "nb_20": ["_state_emit", "_shift_all"],
    "nb_21": ["_state_emit", "_shift_all"],
    "nb_22": ["_state_emit", "_shift_all"],
    "nb_23": ["_state_emit"],
}


def _build_function_sets(
    *,
    function_kind: str,
) -> tuple[
    dict[str, Callable[[str], str]],
    dict[str, Callable[[str], str]],
    dict[str, Callable[[str], str]],
]:
    kind = function_kind.lower().strip()
    if kind not in {"bijective", "nonbijective", "both"}:
        raise ValueError(f"Unknown function_kind: {function_kind!r}")

    if kind == "bijective":
        allowed_names = set(BIJECTIVE_FUNCTION_NAMES)
    elif kind == "nonbijective":
        allowed_names = set(NONBIJECTIVE_FUNCTION_NAMES)
    else:
        allowed_names = set(BIJECTIVE_FUNCTION_NAMES) | set(NONBIJECTIVE_FUNCTION_NAMES)

    all_set = {name: ALL_FUNCTIONS[name] for name in sorted(allowed_names)}

    train_names = [
        *[name for name in DEFAULT_BIJECTIVE_TRAIN_FUNCTION_NAMES if name in allowed_names],
        *[name for name in DEFAULT_NONBIJECTIVE_TRAIN_FUNCTION_NAMES if name in allowed_names],
    ]
    test_names = [
        *[name for name in DEFAULT_BIJECTIVE_TEST_FUNCTION_NAMES if name in allowed_names],
        *[name for name in DEFAULT_NONBIJECTIVE_TEST_FUNCTION_NAMES if name in allowed_names],
    ]

    train_set = {name: all_set[name] for name in train_names}
    eval_set = {name: all_set[name] for name in test_names}
    return all_set, train_set, eval_set


def _collect_helper_names(function_names: set[str]) -> list[str]:
    seen: set[str] = set()

    def add_helper(helper_name: str) -> None:
        if helper_name in seen:
            return
        for dep in HELPER_DEPENDENCIES[helper_name]:
            add_helper(dep)
        seen.add(helper_name)

    for function_name in sorted(function_names):
        for helper_name in FUNCTION_DEPENDENCIES[function_name]:
            add_helper(helper_name)

    return [name for name in HELPER_ORDER if name in seen]


def random_expr(depth: int, function_names: list[str]) -> str:
    if depth <= 0:
        return "x"
    expr = "x"
    for _ in range(depth):
        fname = random.choice(function_names)
        expr = f"{fname}({expr})"
    return expr


def generate_full_code(expr: str, *, stage: int, custom_functions: dict[str, Callable[[str], str]]) -> str:
    used_funcs = {fname for fname in custom_functions if fname in expr}
    code_parts: list[str] = []
    if stage == 1:
        code_parts.append("from __future__ import annotations")
        for helper_name in _collect_helper_names(used_funcs):
            code_parts.append(inspect.getsource(getattr(atomic_lib, helper_name)))
        for fname in sorted(used_funcs):
            code_parts.append(inspect.getsource(custom_functions[fname]))

    code_parts.append(f"def main_solution(x):\n    return {expr}")
    return "\n\n".join(code_parts)


def generate_feasible_input(
    func: Callable[[str], object],
    *,
    attempts: int = 100,
    min_len: int = 2,
    max_len: int = 6,
) -> str | None:
    for _ in range(attempts):
        length = random.randint(min_len, max_len)
        test_input = "".join(random.choices(string.digits, k=length))
        try:
            result = func(test_input)
        except Exception:
            continue
        if isinstance(result, str):
            return test_input
    return None


def generate_random_string_function(
    *,
    max_depth: int,
    custom_functions: dict[str, Callable[[str], str]],
) -> tuple[str, Callable[[str], str], str | None]:
    expr = random_expr(max_depth, function_names=list(custom_functions.keys()))
    func_str = f"lambda x: {expr}"
    try:
        func = eval(func_str, custom_functions)
    except Exception as e:
        raise ValueError(f"Error constructing function from expression: {expr}\n{e}") from e
    feasible_input = generate_feasible_input(func)
    return expr, func, feasible_input


def _write_samples(samples: list[dict[str, object]], save_path: str) -> None:
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    if save_path.endswith(".parquet"):
        try:
            from datasets import Dataset  # type: ignore
        except Exception as e:
            raise ModuleNotFoundError(
                "Writing parquet requires the optional dependency `datasets` (and its parquet backend). "
                "Either install it, or use a `.jsonl` save_path."
            ) from e
        Dataset.from_list(samples).to_parquet(save_path)
        return

    with open(save_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--stage", type=int, required=True)
    parser.add_argument("--split", type=str, required=True, choices=["train", "test"])
    parser.add_argument("--direction", type=str, default="forward", choices=["forward", "backward"])
    parser.add_argument(
        "--function_kind",
        type=str,
        default="bijective",
        choices=["bijective", "nonbijective", "both"],
        help="Which function family to sample from.",
    )
    parser.add_argument(
        "--stage1_pool",
        type=str,
        default="all",
        choices=["all", "split"],
        help="For --stage 1, sample from all selected functions or the train/test split pool.",
    )
    parser.add_argument("--min_level", type=int, default=1)
    parser.add_argument("--max_level", type=int, default=1)
    parser.add_argument("--data_num", type=int, default=5000)
    args = parser.parse_args()

    all_set, train_set, eval_set = _build_function_sets(function_kind=args.function_kind)

    if args.stage == 1 and args.stage1_pool == "all":
        custom_functions = all_set
    else:
        custom_functions = train_set if args.split == "train" else eval_set

    uses_nonbijective = any(name in NONBIJECTIVE_FUNCTION_NAMES for name in custom_functions)
    data_source_prefix = "codeio-nb" if uses_nonbijective else "codeio"

    if args.direction == "backward":
        backward_mode = "preimage" if uses_nonbijective else "unique"
    else:
        backward_mode = "none"

    depths = list(range(args.min_level, args.max_level + 1))
    if args.data_num % len(depths) != 0:
        raise ValueError(f"--data_num {args.data_num} should be divisible by level number {len(depths)}")
    num_per_depth = args.data_num // len(depths)

    data: list[dict[str, object]] = []

    random.seed(42)
    for depth in depths:
        count = 0
        print(f"Generating data for depth {depth} ...")
        pbar = tqdm(total=num_per_depth, desc=f"depth={depth}", unit="sample")
        while count < num_per_depth:
            try:
                expr, func, feasible_input = generate_random_string_function(
                    max_depth=depth,
                    custom_functions=custom_functions,
                )
            except Exception:
                continue

            if feasible_input is None:
                continue

            try:
                output = func(feasible_input)
            except Exception:
                continue
            if not isinstance(output, str):
                continue

            full_code = generate_full_code(expr, stage=args.stage, custom_functions=custom_functions)

            if args.direction == "forward":
                data_source = f"{data_source_prefix}-forward-incomplete-depth{depth}"
                prompt_content = FORWARD_PROMPT.format(code=full_code, input=feasible_input)
            else:
                if backward_mode == "unique":
                    data_source = f"{data_source_prefix}-backward-incomplete-depth{depth}"
                    prompt_content = BACKWARD_PROMPT.format(code=full_code, output=output)
                else:
                    data_source = f"{data_source_prefix}-preimage-incomplete-depth{depth}"
                    prompt_content = PREIMAGE_PROMPT.format(code=full_code, output=output)

            sample = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": prompt_content,
                    }
                ],
                "ability": "reasoning",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": json.dumps(
                        {
                            "ref_input": {"x": feasible_input},
                            "ref_output": output,
                            "ref_code": full_code,
                            "funcname": "main_solution",
                        }
                    ),
                },
                "extra_info": {
                    "index": 0,
                    "split": "dummy",
                    "depth": depth,
                    "direction": args.direction,
                    "function_kind": args.function_kind,
                    "function_library": "atomic_functions_v3",
                },
            }

            data.append(sample)
            count += 1
            pbar.update(1)
            if count % 1000 == 0:
                print(f"  Depth {depth}: Generated {count} samples.")
        pbar.close()

    _write_samples(data, args.save_path)
    print(f"Saved {len(data)} samples to {args.save_path}.")
