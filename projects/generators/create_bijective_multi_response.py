"""
Bijective multi-response inverse generator (stage-2 prompt, stage-1 ref_code).

- PROMPT:   stage=2 (only ``def main_solution(x): return func_XX(x)``), so the
            model must recall function semantics from SFT rather than reading code.
- ref_code: stage=1 (full source code), so the reward scorer can execute the
            reference implementation directly without needing the bijective
            function environment loaded (the default scorer env only has the
            non-bijective functions baked in).
"""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path
import random
import sys

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


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from projects.generators import create_bijective as base  # type: ignore  # noqa: E402


MULTI_RESPONSE_WITH_OPTIONAL_SHORTFALL_PROMPT = (
    "You are given a code:\n\n{code}\n\n"
    "You are told that `main_solution(x)` returns the string \"{output}\".\n"
    "Find {num_answers} DIFFERENT digit-string inputs `x` such that `main_solution(x)` returns exactly that string.\n"
    "If fewer than {num_answers} valid answers exist, output only the valid answers.\n\n"
    "Please reason between the XML tags of <think> </think>, and put the answers, which are ONLY the digit strings, "
    "between <answer> </answer> tags separated by commas.\n"
    "The final response should look like <think> ... </think><answer> x1,x2,... </answer>"
)


def _make_gt_info(*, num_answers: int) -> dict[str, object]:
    return {
        "num_answers": num_answers,
        "allow_fewer_answers": True,
        "multi_response_scoring": "bijective_inverse",
    }


def _generate_bijective_sample(
    *,
    custom_functions: dict[str, object],
    depth: int,
    num_answers: int,
    min_len: int,
    max_len: int,
) -> dict[str, object] | None:
    try:
        expr = base.random_expr(depth, function_names=list(custom_functions.keys()))
        func = eval(f"lambda x: {expr}", custom_functions)
        feasible_input = base.generate_feasible_input(func, min_len=min_len, max_len=max_len)
    except Exception:
        return None

    if feasible_input is None:
        return None

    try:
        output = func(feasible_input)
    except Exception:
        return None
    if not isinstance(output, str):
        return None

    # Prompt code: stage=2 (no helper source code — model must recall from SFT).
    prompt_code = base.generate_full_code(expr, stage=2, custom_functions=custom_functions)
    # Reference code: stage=1 (full source code — scorer can execute independently).
    ref_code = base.generate_full_code(expr, stage=1, custom_functions=custom_functions)

    prompt_content = MULTI_RESPONSE_WITH_OPTIONAL_SHORTFALL_PROMPT.format(
        code=prompt_code,
        output=output,
        num_answers=num_answers,
    )
    gt_info = _make_gt_info(num_answers=num_answers)
    known_preimages = [feasible_input]
    ground_truth = {
        "ref_input": {"x": feasible_input},
        "ref_output": output,
        "ref_code": ref_code,
        "funcname": "main_solution",
        "num_answers": num_answers,
        "gt_info": gt_info,
        "known_preimages": known_preimages,
        "preimage_lower_bound": 1,
    }
    return {
        "data_source": f"codeio-multi-preimage-incomplete-depth{depth}",
        "prompt": [{"role": "user", "content": prompt_content}],
        "ability": "reasoning",
        "reward_model": {"style": "rule", "ground_truth": json.dumps(ground_truth)},
        "extra_info": {
            "index": 0,
            "split": "dummy",
            "depth": depth,
            "direction": "backward_multi",
            "num_answers": num_answers,
            "function_library": "atomic_functions_v3",
            "gt_info": gt_info,
            "known_preimages": known_preimages,
            "preimage_lower_bound": 1,
        },
    }


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--split", type=str, required=True, choices=["train", "test"])
    parser.add_argument("--min_level", type=int, default=1)
    parser.add_argument("--max_level", type=int, default=1)
    parser.add_argument("--data_num", type=int, default=5000)
    parser.add_argument("--num_answers", type=int, default=2)
    parser.add_argument("--min_len", type=int, default=2)
    parser.add_argument("--max_len", type=int, default=6)
    args = parser.parse_args()

    if not (1 <= args.num_answers <= 6):
        raise ValueError("--num_answers must be between 1 and 6 (inclusive)")
    if args.min_len < 1 or args.max_len < args.min_len:
        raise ValueError("Invalid --min_len/--max_len")
    if args.min_level < 1 or args.max_level < args.min_level:
        raise ValueError("Invalid --min_level/--max_level")

    # Always use bijective functions only.
    _, train_set, eval_set = base._build_function_sets(function_kind="bijective")
    custom_functions = train_set if args.split == "train" else eval_set

    depths = list(range(args.min_level, args.max_level + 1))
    if args.data_num % len(depths) != 0:
        raise ValueError(f"--data_num {args.data_num} should be divisible by level number {len(depths)}")
    num_per_depth = args.data_num // len(depths)

    data: list[dict[str, object]] = []

    random.seed(42)
    for depth in depths:
        count = 0
        print(f"Generating v3 bijective multi-response data for depth {depth} ...")
        pbar = tqdm(total=num_per_depth, desc=f"depth={depth}", unit="sample")
        while count < num_per_depth:
            sample = _generate_bijective_sample(
                custom_functions=custom_functions,
                depth=depth,
                num_answers=args.num_answers,
                min_len=args.min_len,
                max_len=args.max_len,
            )

            if sample is None:
                continue

            data.append(sample)
            count += 1
            pbar.update(1)
            if count % 1000 == 0:
                print(f"  Depth {depth}: Generated {count} samples.")
        pbar.close()

    base._write_samples(data, args.save_path)
    print(f"Saved {len(data)} samples to {args.save_path}.")
