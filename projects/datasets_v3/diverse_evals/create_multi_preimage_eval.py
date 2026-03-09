"""
V3 diverse multi-preimage eval generator for the atomic non-bijective family.
"""

from __future__ import annotations

from argparse import ArgumentParser
import json
import random
import string
import sys
from pathlib import Path
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

from projects.datasets_v3.generators import create_string_data as base  # type: ignore  # noqa: E402


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


def _is_output_len_ok(output: str, *, min_len: int | None, max_len: int | None) -> bool:
    if min_len is not None and len(output) < min_len:
        return False
    if max_len is not None and len(output) > max_len:
        return False
    return True


def find_output_with_k_preimages(
    func: Callable[[str], object],
    *,
    k: int,
    attempts: int,
    min_len: int,
    max_len: int,
    output_min_len: int | None,
    output_max_len: int | None,
) -> tuple[str, list[str]] | None:
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
        if not _is_output_len_ok(y, min_len=output_min_len, max_len=output_max_len):
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


def _resolve_num_answers(*, num_answers: int, num_answers_min: int, num_answers_max: int) -> int:
    if num_answers_min > 0 or num_answers_max > 0:
        if num_answers_min <= 0 or num_answers_max <= 0:
            raise ValueError("--num_answers_min/--num_answers_max must both be set (and >0) when using a range.")
        if num_answers_min > num_answers_max:
            raise ValueError("--num_answers_min must be <= --num_answers_max.")
        return random.randint(num_answers_min, num_answers_max)

    if num_answers <= 0:
        raise ValueError("--num_answers must be > 0.")
    return num_answers


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--stage", type=int, required=True, choices=[1, 2])
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--function_kind", type=str, default="nonbijective", choices=["nonbijective"])
    parser.add_argument("--min_level", type=int, default=1)
    parser.add_argument("--max_level", type=int, default=1)
    parser.add_argument("--data_num", type=int, default=1024)
    parser.add_argument("--num_answers", type=int, default=2)
    parser.add_argument("--num_answers_min", type=int, default=0)
    parser.add_argument("--num_answers_max", type=int, default=0)
    parser.add_argument("--min_len", type=int, default=2)
    parser.add_argument("--max_len", type=int, default=6)
    parser.add_argument("--output_min_len", type=int, default=0)
    parser.add_argument("--output_max_len", type=int, default=0)
    parser.add_argument("--search_attempts", type=int, default=2000)
    parser.add_argument("--estimate_attempts", type=int, default=0)
    parser.add_argument("--estimate_max_unique", type=int, default=50)
    parser.add_argument("--data_source_tag", type=str, default="diverse-eval-v3")
    args = parser.parse_args()

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

    output_min_len = None if args.output_min_len <= 0 else int(args.output_min_len)
    output_max_len = None if args.output_max_len <= 0 else int(args.output_max_len)
    if output_min_len is not None and output_max_len is not None and output_min_len > output_max_len:
        raise ValueError("Invalid --output_min_len/--output_max_len")

    all_set, train_set, eval_set = base._build_function_sets(function_kind=args.function_kind)
    custom_functions = train_set if args.split == "train" else eval_set

    uses_nonbijective = any(name in base.NONBIJECTIVE_FUNCTION_NAMES for name in custom_functions.keys())
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
        print(f"Generating diverse eval multi-preimage data for depth {depth} ...")
        pbar = tqdm(total=num_per_depth, desc=f"depth={depth}", unit="sample")
        while count < num_per_depth:
            k = _resolve_num_answers(
                num_answers=int(args.num_answers),
                num_answers_min=int(args.num_answers_min),
                num_answers_max=int(args.num_answers_max),
            )

            try:
                expr = base.random_expr(depth, function_names=list(custom_functions.keys()))
                func = eval(f"lambda x: {expr}", custom_functions)
            except Exception:
                continue

            found = find_output_with_k_preimages(
                func,
                k=k,
                attempts=int(args.search_attempts),
                min_len=int(args.min_len),
                max_len=int(args.max_len),
                output_min_len=output_min_len,
                output_max_len=output_max_len,
            )
            if found is None:
                continue

            output, preimages = found
            preimages, estimate_hits = extend_preimages_for_fixed_output(
                func,
                output=output,
                preimages=preimages,
                attempts=int(args.estimate_attempts),
                min_len=int(args.min_len),
                max_len=int(args.max_len),
                max_unique=int(args.estimate_max_unique),
            )

            # Keep v3 stage-2 evals self-contained for scorer execution.
            full_code = base.generate_full_code(expr, stage=1, custom_functions=custom_functions)
            prompt_content = MULTI_PREIMAGE_PROMPT.format(code=full_code, output=output, num_answers=k)

            gt = {
                "ref_input": {"x": preimages[0]},
                "ref_output": output,
                "ref_code": full_code,
                "funcname": "main_solution",
                "num_answers": k,
                "gt_info": {"num_answers": k},
                "known_preimages": preimages[: min(len(preimages), 5)],
                "preimage_lower_bound": len(preimages),
                "estimate_attempts": int(args.estimate_attempts),
                "estimate_hits": estimate_hits,
            }
            sample = {
                "data_source": f"codeio-nb-multi-preimage-{args.data_source_tag}-depth{depth}",
                "prompt": [{"role": "user", "content": prompt_content}],
                "ability": "reasoning",
                "reward_model": {"style": "rule", "ground_truth": json.dumps(gt)},
                "extra_info": {
                    "index": 0,
                    "split": "dummy",
                    "depth": depth,
                    "direction": "backward_multi",
                    "num_answers": k,
                    "gt_info": {"num_answers": k},
                    "expr": expr,
                    "output_len": len(output),
                    "function_library": "atomic_functions_v3",
                },
            }

            data.append(sample)
            count += 1
            pbar.update(1)
            if count % 200 == 0:
                print(f"  Depth {depth}: Generated {count} samples.")
        pbar.close()

    base._write_samples(data, args.save_path)
    print(f"Saved {len(data)} samples to {args.save_path}.")
