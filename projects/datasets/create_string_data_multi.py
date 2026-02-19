"""
Multi-Preimage Digit-String Operations for RL Research
=====================================================

This script is a companion to `projects/datasets/create_string_data.py`.

It generates *non-bijective* inverse problems where the model must enumerate
multiple (different) digit-string inputs `x` that lead to the same output.

Answers must be placed inside `<answer>...</answer>` and separated by commas.
"""

from __future__ import annotations

from argparse import ArgumentParser
import json
import random
import string
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


# Import the base implementation from the sibling file, so we don't duplicate
# the function families / code rendering logic.
import create_string_data as base  # type: ignore


MULTI_PREIMAGE_PROMPT = (
    "You are given a code:\n\n{code}\n\n"
    "You are told that `main_solution(x)` returns the string \"{output}\".\n"
    "Find {num_answers} DIFFERENT digit-string inputs `x` such that `main_solution(x)` returns exactly that string.\n\n"
    "Please reason between the XML tags of <think> </think>, and put the answers, which are ONLY the digit strings, "
    "between <answer> </answer> tags separated by commas.\n"
    "The final response should look like <think> ... </think><answer> x1,x2,... </answer>"
)


def _random_digit_string(*, min_len: int, max_len: int) -> str:
    length = random.randint(min_len, max_len)
    return "".join(random.choices(string.digits, k=length))


def find_output_with_k_preimages(
    func: Callable[[str], object],
    *,
    k: int,
    attempts: int,
    min_len: int,
    max_len: int,
) -> tuple[str, list[str]] | None:
    """Heuristically find an output with >=k distinct preimages by random sampling."""
    if k <= 0:
        raise ValueError("k must be >= 1")

    buckets: dict[str, list[str]] = {}
    for _ in range(attempts):
        x = _random_digit_string(min_len=min_len, max_len=max_len)
        try:
            y = func(x)
        except Exception:
            continue
        if not isinstance(y, str):
            continue
        xs = buckets.setdefault(y, [])
        if x in xs:
            continue
        xs.append(x)
        if len(xs) >= k:
            return y, xs
    return None


def extend_preimages_for_fixed_output(
    func: Callable[[str], object],
    *,
    output: str,
    preimages: list[str],
    attempts: int,
    min_len: int,
    max_len: int,
    max_unique: int,
) -> tuple[list[str], int]:
    """Heuristically extend a preimage list for a fixed output.

    Returns (extended_preimages, num_hits) where num_hits counts how many sampled
    inputs matched the output (including duplicates).
    """
    if attempts <= 0 or max_unique <= len(preimages):
        return preimages, 0

    seen = set(preimages)
    extended = list(preimages)
    hits = 0
    for _ in range(attempts):
        x = _random_digit_string(min_len=min_len, max_len=max_len)
        try:
            y = func(x)
        except Exception:
            continue
        if not isinstance(y, str):
            continue
        if y != output:
            continue
        hits += 1
        if x in seen:
            continue
        seen.add(x)
        extended.append(x)
        if len(extended) >= max_unique:
            break
    return extended, hits


def _write_samples(samples: list[dict[str, object]], save_path: str) -> None:
    """Persist generated samples to disk (`.jsonl` always; `.parquet` if deps exist)."""
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
    parser.add_argument("--function_kind", type=str, default="nonbijective_v2", choices=["nonbijective", "nonbijective_v2"])
    parser.add_argument(
        "--nb_v2_split",
        type=str,
        default="balanced",
        choices=["balanced", "original", "easier"],
        help="Only used when --function_kind nonbijective_v2: choose which train/test function split to use.",
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
    parser.add_argument("--num_answers", type=int, default=2, help="Number of distinct preimages to request/score (<=5 recommended).")
    parser.add_argument("--min_len", type=int, default=2)
    parser.add_argument("--max_len", type=int, default=6)
    parser.add_argument("--search_attempts", type=int, default=2000, help="Random attempts per sample to find collisions.")
    parser.add_argument(
        "--estimate_attempts",
        type=int,
        default=0,
        help="Extra random attempts per sample to estimate/collect more distinct preimages for the selected output (0 disables).",
    )
    parser.add_argument(
        "--estimate_max_unique",
        type=int,
        default=50,
        help="Cap on unique preimages collected during estimation (for logging/metadata).",
    )
    args = parser.parse_args()

    if not (1 <= args.num_answers <= 5):
        raise ValueError("--num_answers must be between 1 and 5 (inclusive)")
    if args.min_len < 1 or args.max_len < args.min_len:
        raise ValueError("Invalid --min_len/--max_len")
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
    if not uses_nonbijective:
        raise ValueError("This generator is intended for non-bijective function families only.")

    depths = list(range(args.min_level, args.max_level + 1))
    if args.data_num % len(depths) != 0:
        raise ValueError(f"--data_num {args.data_num} should be divisible by level number {len(depths)}")
    num_per_depth = args.data_num // len(depths)

    data: list[dict[str, object]] = []

    random.seed(42)
    for depth in depths:
        count = 0
        print(f"Generating multi-preimage data for depth {depth} ...")
        pbar = tqdm(total=num_per_depth, desc=f"depth={depth}", unit="sample")
        while count < num_per_depth:
            try:
                expr = base.random_expr(depth, function_names=list(custom_functions.keys()))
                func_str = f"lambda x: {expr}"
                func = eval(func_str, custom_functions)
            except Exception:
                continue

            found = find_output_with_k_preimages(
                func,
                k=args.num_answers,
                attempts=args.search_attempts,
                min_len=args.min_len,
                max_len=args.max_len,
            )
            if found is None:
                continue
            output, preimages = found
            preimages, estimate_hits = extend_preimages_for_fixed_output(
                func,
                output=output,
                preimages=preimages,
                attempts=args.estimate_attempts,
                min_len=args.min_len,
                max_len=args.max_len,
                max_unique=args.estimate_max_unique,
            )

            full_code = base.generate_full_code(expr, stage=args.stage, custom_functions=custom_functions)
            prompt_content = MULTI_PREIMAGE_PROMPT.format(code=full_code, output=output, num_answers=args.num_answers)

            data_source = f"codeio-nb-multi-preimage-incomplete-depth{depth}"
            sample = {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": prompt_content}],
                "ability": "reasoning",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": json.dumps(
                        {
                            # Keep `ref_input` for consistency/debug, but scoring is based on ref_code+ref_output.
                            "ref_input": {"x": preimages[0]},
                            "ref_output": output,
                            "ref_code": full_code,
                            "funcname": "main_solution",
                            "num_answers": args.num_answers,
                            "known_preimages": preimages[: min(len(preimages), 5)],
                            "preimage_lower_bound": len(preimages),
                            "estimate_attempts": args.estimate_attempts,
                            "estimate_hits": estimate_hits,
                        }
                    ),
                },
                "extra_info": {
                    "index": 0,
                    "split": "dummy",
                    "depth": depth,
                    "direction": "backward_multi",
                    "num_answers": args.num_answers,
                    "known_preimages": preimages[: min(len(preimages), 5)],
                    "preimage_lower_bound": len(preimages),
                    "estimate_attempts": args.estimate_attempts,
                    "estimate_hits": estimate_hits,
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
