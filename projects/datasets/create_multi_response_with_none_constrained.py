"""
Constrained multi-response inverse generator for non-bijective tasks.

Extends the standard multi-response generator by adding digit-position
constraints that shrink the solution space.  For each problem:

1.  Randomly pick 2-3 digit positions.
2.  For each position, randomly pick 3-6 digits (0-9) that are NOT allowed.
3.  Express the constraints explicitly in the prompt.
4.  Verify (via constrained search) that valid solutions still exist.
    If not, re-pick constraints.

The constraints are persisted in both ``ground_truth`` and ``extra_info``
so that reward functions can enforce them.
"""

from __future__ import annotations

from argparse import ArgumentParser
import json
import random
import string

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


try:
    from projects.datasets import create_string_data as base  # type: ignore
    from projects.datasets import create_string_data_multi as multi_base  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    import create_string_data as base  # type: ignore
    import create_string_data_multi as multi_base  # type: ignore


# ---------------------------------------------------------------------------
# Prompt template (adds constraint block after the standard instructions)
# ---------------------------------------------------------------------------

MULTI_RESPONSE_CONSTRAINED_PROMPT = (
    "You are given a code:\n\n{code}\n\n"
    "You are told that `main_solution(x)` returns the string \"{output}\".\n"
    "Find {num_answers} DIFFERENT digit-string inputs `x` such that `main_solution(x)` returns exactly that string.\n"
    "If fewer than {num_answers} valid answers exist, output only the valid answers.\n\n"
    "Additionally, each input x must satisfy the following constraints:\n"
    "{constraint_text}\n\n"
    "Please reason between the XML tags of <think> </think>, and put the answers, which are ONLY the digit strings, "
    "between <answer> </answer> tags separated by commas.\n"
    "The final response should look like <think> ... </think><answer> x1,x2,... </answer>"
)

# ---------------------------------------------------------------------------
# Ordinal helpers for prompt readability
# ---------------------------------------------------------------------------

_ORDINALS = {
    1: "1st", 2: "2nd", 3: "3rd", 4: "4th", 5: "5th",
    6: "6th", 7: "7th", 8: "8th", 9: "9th", 10: "10th",
}


def _ordinal(n: int) -> str:
    return _ORDINALS.get(n, f"{n}th")


# ---------------------------------------------------------------------------
# Constraint generation
# ---------------------------------------------------------------------------

def generate_random_constraints(
    num_positions: int,
    available_positions: list[int],
    min_excluded: int = 3,
    max_excluded: int = 6,
) -> dict[int, list[int]]:
    """Generate random digit constraints.

    Returns a dict mapping 0-indexed digit position to a sorted list of
    excluded digit values (ints 0-9).
    """
    chosen_positions = sorted(random.sample(available_positions, num_positions))
    constraints: dict[int, list[int]] = {}
    for pos in chosen_positions:
        num_excluded = random.randint(min_excluded, max_excluded)
        excluded = sorted(random.sample(range(10), num_excluded))
        constraints[pos] = excluded
    return constraints


def constraints_to_prompt_text(constraints: dict[int, list[int]]) -> str:
    """Format constraints as human-readable bullet points (1-indexed)."""
    lines: list[str] = []
    for pos in sorted(constraints.keys()):
        excluded = constraints[pos]
        digit_str = ", ".join(str(d) for d in excluded)
        lines.append(f"- The {_ordinal(pos + 1)} digit of x cannot be {digit_str}")
    return "\n".join(lines)


def constraints_to_json_serializable(constraints: dict[int, list[int]]) -> dict[str, list[int]]:
    """Convert constraints dict to JSON-serializable form (string keys)."""
    return {str(pos): excluded for pos, excluded in constraints.items()}


def constraints_from_json(obj: dict[str, list[int]]) -> dict[int, list[int]]:
    """Inverse of ``constraints_to_json_serializable``."""
    return {int(pos): excluded for pos, excluded in obj.items()}


def input_satisfies_constraints(x: str, constraints: dict[int, list[int]]) -> bool:
    """Check whether digit string ``x`` satisfies all digit constraints.

    If the input is shorter than a constrained position, the constraint
    is considered violated (prevents dodging via short inputs).
    """
    for pos, excluded in constraints.items():
        if pos >= len(x):
            return False
        digit_val = int(x[pos])
        if digit_val in excluded:
            return False
    return True


# ---------------------------------------------------------------------------
# Constrained search
# ---------------------------------------------------------------------------

def _random_constrained_digit_string(
    *,
    min_len: int,
    max_len: int,
    constraints: dict[int, list[int]],
) -> str | None:
    """Generate a random digit string satisfying the given constraints."""
    length = random.randint(min_len, max_len)
    result: list[str] = []
    for i in range(length):
        if i in constraints:
            available = [d for d in range(10) if d not in constraints[i]]
            if not available:
                return None  # impossible position
            result.append(str(random.choice(available)))
        else:
            result.append(random.choice(string.digits))
    return "".join(result)


def find_output_with_k_constrained_preimages(
    func,
    *,
    k: int,
    attempts: int,
    min_len: int,
    max_len: int,
    constraints: dict[int, list[int]],
) -> tuple[str, list[str]] | None:
    """Find an output with >= k distinct constrained preimages."""
    buckets: dict[str, list[str]] = {}
    for _ in range(attempts):
        x = _random_constrained_digit_string(
            min_len=min_len, max_len=max_len, constraints=constraints,
        )
        if x is None:
            continue
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


def extend_constrained_preimages(
    func,
    *,
    output: str,
    preimages: list[str],
    attempts: int,
    min_len: int,
    max_len: int,
    max_unique: int,
    constraints: dict[int, list[int]],
) -> tuple[list[str], int]:
    """Extend a preimage list with additional constrained preimages."""
    if attempts <= 0 or max_unique <= len(preimages):
        return preimages, 0

    seen = set(preimages)
    extended = list(preimages)
    hits = 0
    for _ in range(attempts):
        x = _random_constrained_digit_string(
            min_len=min_len, max_len=max_len, constraints=constraints,
        )
        if x is None:
            continue
        try:
            y = func(x)
        except Exception:
            continue
        if not isinstance(y, str) or y != output:
            continue
        hits += 1
        if x in seen:
            continue
        seen.add(x)
        extended.append(x)
        if len(extended) >= max_unique:
            break
    return extended, hits


# ---------------------------------------------------------------------------
# gt_info helpers (mirrors create_multi_response_with_none.py)
# ---------------------------------------------------------------------------

def _make_gt_info(*, num_answers: int) -> dict[str, object]:
    return {
        "num_answers": num_answers,
        "allow_fewer_answers": True,
        "multi_response_scoring": "nonbijective_preimage_constrained",
    }


# ---------------------------------------------------------------------------
# Sample generation
# ---------------------------------------------------------------------------

_MAX_CONSTRAINT_RETRIES = 10


def _generate_constrained_nonbijective_sample(
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
    min_constrained_positions: int = 2,
    max_constrained_positions: int = 3,
    min_excluded_digits: int = 3,
    max_excluded_digits: int = 6,
) -> dict[str, object] | None:
    try:
        expr = base.random_expr(depth, function_names=list(custom_functions.keys()))
        func = eval(f"lambda x: {expr}", custom_functions)
    except Exception:
        return None

    # Positions we can constrain: 0 .. (min_len - 1) so they always exist.
    available_positions = list(range(min_len))
    max_positions = min(max_constrained_positions, len(available_positions))
    min_positions = min(min_constrained_positions, max_positions)

    if max_positions < min_positions:
        return None

    for _ in range(_MAX_CONSTRAINT_RETRIES):
        num_positions = random.randint(min_positions, max_positions)
        constraints = generate_random_constraints(
            num_positions=num_positions,
            available_positions=available_positions,
            min_excluded=min_excluded_digits,
            max_excluded=max_excluded_digits,
        )

        found = find_output_with_k_constrained_preimages(
            func,
            k=num_answers,
            attempts=search_attempts,
            min_len=min_len,
            max_len=max_len,
            constraints=constraints,
        )
        if found is not None:
            break
    else:
        return None  # all constraint attempts failed

    output, preimages = found
    preimages, estimate_hits = extend_constrained_preimages(
        func,
        output=output,
        preimages=preimages,
        attempts=estimate_attempts,
        min_len=min_len,
        max_len=max_len,
        max_unique=estimate_max_unique,
        constraints=constraints,
    )

    known_k = max(5, int(num_answers))
    known_preimages = preimages[: min(len(preimages), known_k)]
    full_code = base.generate_full_code(expr, stage=stage, custom_functions=custom_functions)

    constraint_text = constraints_to_prompt_text(constraints)
    prompt_content = MULTI_RESPONSE_CONSTRAINED_PROMPT.format(
        code=full_code,
        output=output,
        num_answers=num_answers,
        constraint_text=constraint_text,
    )

    gt_info = _make_gt_info(num_answers=num_answers)
    constraints_json = constraints_to_json_serializable(constraints)

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
        "digit_constraints": constraints_json,
    }

    data_source = f"codeio-nb-multi-preimage-with-constraints-depth{depth}"

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
            "gt_info": gt_info,
            "known_preimages": known_preimages,
            "preimage_lower_bound": len(preimages),
            "estimate_attempts": estimate_attempts,
            "estimate_hits": estimate_hits,
            "digit_constraints": constraints_json,
        },
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--stage", type=int, required=True)
    parser.add_argument("--split", type=str, required=True, choices=["train", "test"])
    parser.add_argument(
        "--function_kind",
        type=str,
        default="nonbijective_v2",
        choices=["nonbijective", "nonbijective_v2"],
    )
    parser.add_argument(
        "--nb_v2_split",
        type=str,
        default="balanced",
        choices=["balanced", "original", "easier"],
    )
    parser.add_argument(
        "--stage1_pool",
        type=str,
        default="all",
        choices=["all", "split"],
    )
    parser.add_argument("--min_level", type=int, default=1)
    parser.add_argument("--max_level", type=int, default=1)
    parser.add_argument("--data_num", type=int, default=5000)
    parser.add_argument("--num_answers", type=int, default=2)
    parser.add_argument("--min_len", type=int, default=3)
    parser.add_argument("--max_len", type=int, default=6)
    parser.add_argument("--search_attempts", type=int, default=4000)
    parser.add_argument("--estimate_attempts", type=int, default=0)
    parser.add_argument("--estimate_max_unique", type=int, default=50)
    # Constraint parameters
    parser.add_argument("--min_constrained_positions", type=int, default=2)
    parser.add_argument("--max_constrained_positions", type=int, default=3)
    parser.add_argument("--min_excluded_digits", type=int, default=3)
    parser.add_argument("--max_excluded_digits", type=int, default=6)
    args = parser.parse_args()

    if not (1 <= args.num_answers <= 6):
        raise ValueError("--num_answers must be between 1 and 6 (inclusive)")
    if args.min_len < 2 or args.max_len < args.min_len:
        raise ValueError("Invalid --min_len/--max_len (min_len must be >= 2)")
    if args.min_level < 1 or args.max_level < args.min_level:
        raise ValueError("Invalid --min_level/--max_level")
    if args.search_attempts < 1:
        raise ValueError("--search_attempts must be >= 1")
    if args.estimate_attempts < 0:
        raise ValueError("--estimate_attempts must be >= 0")
    if args.estimate_max_unique < 1:
        raise ValueError("--estimate_max_unique must be >= 1")
    if args.min_constrained_positions < 1:
        raise ValueError("--min_constrained_positions must be >= 1")
    if args.max_constrained_positions < args.min_constrained_positions:
        raise ValueError("--max_constrained_positions must be >= --min_constrained_positions")
    if args.min_excluded_digits < 1 or args.min_excluded_digits > 9:
        raise ValueError("--min_excluded_digits must be between 1 and 9")
    if args.max_excluded_digits < args.min_excluded_digits or args.max_excluded_digits > 9:
        raise ValueError("--max_excluded_digits must be between --min_excluded_digits and 9")

    all_set, train_set, eval_set = base._build_function_sets(
        function_kind=args.function_kind,
        nb_v2_split=args.nb_v2_split,
    )
    if args.stage == 1 and args.stage1_pool == "all":
        custom_functions = all_set
    else:
        custom_functions = train_set if args.split == "train" else eval_set

    depths = list(range(args.min_level, args.max_level + 1))
    if args.data_num % len(depths) != 0:
        raise ValueError(f"--data_num {args.data_num} should be divisible by level number {len(depths)}")
    num_per_depth = args.data_num // len(depths)

    data: list[dict[str, object]] = []

    random.seed(42)
    for depth in depths:
        count = 0
        print(f"Generating constrained non-bijective multi-response data for depth {depth} ...")
        pbar = tqdm(total=num_per_depth, desc=f"depth={depth}", unit="sample")
        while count < num_per_depth:
            sample = _generate_constrained_nonbijective_sample(
                custom_functions=custom_functions,
                depth=depth,
                stage=args.stage,
                num_answers=args.num_answers,
                min_len=args.min_len,
                max_len=args.max_len,
                search_attempts=args.search_attempts,
                estimate_attempts=args.estimate_attempts,
                estimate_max_unique=args.estimate_max_unique,
                min_constrained_positions=args.min_constrained_positions,
                max_constrained_positions=args.max_constrained_positions,
                min_excluded_digits=args.min_excluded_digits,
                max_excluded_digits=args.max_excluded_digits,
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
    print(f"Saved {len(data)} constrained samples to {args.save_path}.")
