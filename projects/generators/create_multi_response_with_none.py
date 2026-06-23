"""
Shared multi-response inverse generator for bijective and non-bijective tasks.

This keeps the prompt/output format identical across both families:
- the model always answers with comma-separated digit strings in `<answer>...</answer>`
- the prompt explicitly allows fewer than `num_answers` outputs when fewer valid
  answers exist
"""

from __future__ import annotations

from argparse import ArgumentParser
import json
import random

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


from projects.generators import create_nonbijective as base  # type: ignore
from projects.generators import create_nonbijective_multi as multi_base  # type: ignore


MULTI_RESPONSE_WITH_OPTIONAL_SHORTFALL_PROMPT = (
    "You are given a code:\n\n{code}\n\n"
    "You are told that `main_solution(x)` returns the string \"{output}\".\n"
    "Find {num_answers} DIFFERENT digit-string inputs `x` such that `main_solution(x)` returns exactly that string.\n"
    "If fewer than {num_answers} valid answers exist, output only the valid answers.\n\n"
    "Please reason between the XML tags of <think> </think>, and put the answers, which are ONLY the digit strings, "
    "between <answer> </answer> tags separated by commas.\n"
    "The final response should look like <think> ... </think><answer> x1,x2,... </answer>"
)


def _make_gt_info(*, num_answers: int, scoring_mode: str) -> dict[str, object]:
    return {
        "num_answers": num_answers,
        "allow_fewer_answers": True,
        "multi_response_scoring": scoring_mode,
    }


def _make_sample(
    *,
    data_source: str,
    prompt_content: str,
    depth: int,
    num_answers: int,
    ground_truth: dict[str, object],
    extra_info: dict[str, object],
) -> dict[str, object]:
    return {
        "data_source": data_source,
        "prompt": [{"role": "user", "content": prompt_content}],
        "ability": "reasoning",
        "reward_model": {"style": "rule", "ground_truth": json.dumps(ground_truth)},
        "extra_info": {
            "index": 0,
            "split": "dummy",
            "depth": depth,
            "direction": "backward_multi",
            "num_answers": num_answers,
            **extra_info,
        },
    }


def _generate_bijective_sample(
    *,
    custom_functions: dict[str, object],
    depth: int,
    stage: int,
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

    full_code = base.generate_full_code(expr, stage=stage, custom_functions=custom_functions)
    prompt_content = MULTI_RESPONSE_WITH_OPTIONAL_SHORTFALL_PROMPT.format(
        code=full_code,
        output=output,
        num_answers=num_answers,
    )
    gt_info = _make_gt_info(num_answers=num_answers, scoring_mode="bijective_inverse")
    known_preimages = [feasible_input]
    ground_truth = {
        "ref_input": {"x": feasible_input},
        "ref_output": output,
        "ref_code": full_code,
        "funcname": "main_solution",
        "num_answers": num_answers,
        "gt_info": gt_info,
        "known_preimages": known_preimages,
        "preimage_lower_bound": 1,
    }
    return _make_sample(
        data_source=f"codeio-multi-preimage-incomplete-depth{depth}",
        prompt_content=prompt_content,
        depth=depth,
        num_answers=num_answers,
        ground_truth=ground_truth,
        extra_info={
            "gt_info": gt_info,
            "known_preimages": known_preimages,
            "preimage_lower_bound": 1,
        },
    )


def _generate_nonbijective_sample(
    *,
    custom_functions: dict[str, object],
    depth: int,
    stage: int,
    num_answers: int,
    min_len: int,
    max_len: int,
    search_attempts: int,
    estimate_attempts: int,
    estimate_max_unique: int,
) -> dict[str, object] | None:
    try:
        expr = base.random_expr(depth, function_names=list(custom_functions.keys()))
        func = eval(f"lambda x: {expr}", custom_functions)
    except Exception:
        return None

    found = multi_base.find_output_with_k_preimages(
        func,
        k=num_answers,
        attempts=search_attempts,
        min_len=min_len,
        max_len=max_len,
    )
    if found is None:
        return None

    output, preimages = found
    preimages, estimate_hits = multi_base.extend_preimages_for_fixed_output(
        func,
        output=output,
        preimages=preimages,
        attempts=estimate_attempts,
        min_len=min_len,
        max_len=max_len,
        max_unique=estimate_max_unique,
    )

    known_k = max(5, int(num_answers))
    known_preimages = preimages[: min(len(preimages), known_k)]
    full_code = base.generate_full_code(expr, stage=stage, custom_functions=custom_functions)
    prompt_content = MULTI_RESPONSE_WITH_OPTIONAL_SHORTFALL_PROMPT.format(
        code=full_code,
        output=output,
        num_answers=num_answers,
    )
    gt_info = _make_gt_info(num_answers=num_answers, scoring_mode="nonbijective_preimage")
    ground_truth = {
        "ref_input": {"x": preimages[0]},
        "ref_output": output,
        "ref_code": full_code,
        "funcname": "main_solution",
        "num_answers": num_answers,
        "gt_info": gt_info,
        "known_preimages": known_preimages,
        "preimage_lower_bound": len(preimages),
        "estimate_attempts": estimate_attempts,
        "estimate_hits": estimate_hits,
    }
    return _make_sample(
        data_source=f"codeio-nb-multi-preimage-incomplete-depth{depth}",
        prompt_content=prompt_content,
        depth=depth,
        num_answers=num_answers,
        ground_truth=ground_truth,
        extra_info={
            "gt_info": gt_info,
            "known_preimages": known_preimages,
            "preimage_lower_bound": len(preimages),
            "estimate_attempts": estimate_attempts,
            "estimate_hits": estimate_hits,
        },
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--stage", type=int, required=True)
    parser.add_argument("--split", type=str, required=True, choices=["train", "test"])
    parser.add_argument(
        "--function_kind",
        type=str,
        default="nonbijective_v2",
        choices=["bijective", "nonbijective", "nonbijective_v2"],
    )
    parser.add_argument(
        "--nb_v2_split",
        type=str,
        default="balanced",
        choices=["balanced", "original", "easier"],
        help="Only used when --function_kind nonbijective_v2.",
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
    parser.add_argument("--num_answers", type=int, default=2)
    parser.add_argument("--min_len", type=int, default=2)
    parser.add_argument("--max_len", type=int, default=6)
    parser.add_argument("--search_attempts", type=int, default=2000)
    parser.add_argument("--estimate_attempts", type=int, default=0)
    parser.add_argument("--estimate_max_unique", type=int, default=50)
    args = parser.parse_args()

    if not (1 <= args.num_answers <= 6):
        raise ValueError("--num_answers must be between 1 and 6 (inclusive)")
    if args.min_len < 1 or args.max_len < args.min_len:
        raise ValueError("Invalid --min_len/--max_len")
    if args.min_level < 1 or args.max_level < args.min_level:
        raise ValueError("Invalid --min_level/--max_level")
    if args.search_attempts < 1:
        raise ValueError("--search_attempts must be >= 1")
    if args.estimate_attempts < 0:
        raise ValueError("--estimate_attempts must be >= 0")
    if args.estimate_max_unique < 1:
        raise ValueError("--estimate_max_unique must be >= 1")

    all_set, train_set, eval_set = base._build_function_sets(
        function_kind=args.function_kind,
        nb_v2_split=args.nb_v2_split,
    )
    if args.stage == 1 and args.stage1_pool == "all":
        custom_functions = all_set
    else:
        custom_functions = train_set if args.split == "train" else eval_set

    uses_nonbijective = any(
        name in base.NONBIJECTIVE_FUNCTION_NAMES or name in base.NB_V2_FUNCTION_NAMES for name in custom_functions.keys()
    )

    depths = list(range(args.min_level, args.max_level + 1))
    if args.data_num % len(depths) != 0:
        raise ValueError(f"--data_num {args.data_num} should be divisible by level number {len(depths)}")
    num_per_depth = args.data_num // len(depths)

    data: list[dict[str, object]] = []

    random.seed(42)
    for depth in depths:
        count = 0
        family_label = "non-bijective" if uses_nonbijective else "bijective"
        print(f"Generating {family_label} multi-response data for depth {depth} ...")
        pbar = tqdm(total=num_per_depth, desc=f"depth={depth}", unit="sample")
        while count < num_per_depth:
            if uses_nonbijective:
                sample = _generate_nonbijective_sample(
                    custom_functions=custom_functions,
                    depth=depth,
                    stage=args.stage,
                    num_answers=args.num_answers,
                    min_len=args.min_len,
                    max_len=args.max_len,
                    search_attempts=args.search_attempts,
                    estimate_attempts=args.estimate_attempts,
                    estimate_max_unique=args.estimate_max_unique,
                )
            else:
                sample = _generate_bijective_sample(
                    custom_functions=custom_functions,
                    depth=depth,
                    stage=args.stage,
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

    multi_base._write_samples(data, args.save_path)
    print(f"Saved {len(data)} samples to {args.save_path}.")
