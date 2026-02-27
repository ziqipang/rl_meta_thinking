"""
Eval-only multi-preimage generator for `lenchange_functions.py`.

Unlike the NB-v2 generators, we *construct* multiple preimages directly:
all `lenop_XX` functions only depend on digits modulo 5, so toggling any digit
by +5 (mod 10) keeps the output unchanged.

We still store `num_answers` in both `ground_truth.num_answers` (back-compat)
and `ground_truth.gt_info.num_answers` (preferred for new evals).
"""

from __future__ import annotations

from argparse import ArgumentParser
import json
import random
import string
import sys
from pathlib import Path

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

from projects.datasets import create_string_data as base  # type: ignore  # noqa: E402
from projects.datasets.diverse_evals.lenchange_functions import LENCHANGE_FUNCTIONS  # noqa: E402


MULTI_PREIMAGE_PROMPT = (
    "You are given a code:\n\n{code}\n\n"
    "You are told that `main_solution(x)` returns the string \"{output}\".\n"
    "Find {num_answers} DIFFERENT digit-string inputs `x` such that `main_solution(x)` returns exactly that string.\n\n"
    "Please reason between the XML tags of <think> </think>, and put the answers, which are ONLY the digit strings, "
    "between <answer> </answer> tags separated by commas.\n"
    "The final response should look like <think> ... </think><answer> x1,x2,... </answer>"
)


def _write_samples(samples: list[dict[str, object]], save_path: str) -> None:
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


def _toggle_digit_by_5(ch: str) -> str:
    d = ord(ch) - ord("0")
    return chr(((d + 5) % 10) + ord("0"))


def _make_k_preimages(base_x: str, *, k: int) -> list[str]:
    """Generate k distinct preimages by toggling different positions by +5 (mod 10)."""
    if k <= 0:
        raise ValueError("k must be > 0")
    if not base_x:
        return ["0"] * k

    preimages: list[str] = [base_x]
    i = 0
    # Greedy: toggle one position at a time, then combinations if needed.
    while len(preimages) < k:
        pos = i % len(base_x)
        x_list = list(base_x)
        x_list[pos] = _toggle_digit_by_5(x_list[pos])
        cand = "".join(x_list)
        if cand not in preimages:
            preimages.append(cand)
        i += 1
        if i > 10_000:
            break
    return preimages[:k]


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--data_num", type=int, default=1024)
    parser.add_argument("--stage", type=int, default=1, choices=[1], help="Must be 1 (include source code).")
    parser.add_argument("--output_min_len", type=int, default=6)
    parser.add_argument("--output_max_len", type=int, default=10)
    parser.add_argument("--num_answers_min", type=int, default=3)
    parser.add_argument("--num_answers_max", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.output_min_len < 1 or args.output_max_len < args.output_min_len:
        raise ValueError("Invalid --output_min_len/--output_max_len")
    if args.num_answers_min < 1 or args.num_answers_max < args.num_answers_min:
        raise ValueError("Invalid --num_answers_min/--num_answers_max")

    random.seed(int(args.seed))

    fn_names = sorted(LENCHANGE_FUNCTIONS.keys())
    custom_functions = {name: LENCHANGE_FUNCTIONS[name] for name in fn_names}

    data: list[dict[str, object]] = []
    pbar = tqdm(total=int(args.data_num), desc="lenchange-eval", unit="sample")

    while len(data) < int(args.data_num):
        fname = random.choice(fn_names)
        func = custom_functions[fname]

        # We want y length in [output_min_len, output_max_len]. Our functions shift length by ±1,
        # so choose x length accordingly (then assert/skip if needed).
        y_len = random.randint(int(args.output_min_len), int(args.output_max_len))
        x_len = y_len + 1 if fname in {"lenop_00", "lenop_01", "lenop_02"} else max(1, y_len - 1)
        x = "".join(random.choices(string.digits, k=x_len))

        try:
            y = func(x)  # type: ignore[misc]
        except Exception:
            continue
        if not isinstance(y, str):
            continue
        if not (int(args.output_min_len) <= len(y) <= int(args.output_max_len)):
            continue

        k = random.randint(int(args.num_answers_min), int(args.num_answers_max))
        preimages = _make_k_preimages(x, k=k)

        # Sanity: ensure our constructed candidates indeed collide.
        if any(func(pi) != y for pi in preimages):  # type: ignore[misc]
            continue

        expr = f"{fname}(x)"
        full_code = base.generate_full_code(expr, stage=1, custom_functions=custom_functions)
        prompt_content = MULTI_PREIMAGE_PROMPT.format(code=full_code, output=y, num_answers=k)

        data_source = "codeio-nb-multi-preimage-lenchange-eval-depth1"
        gt = {
            "ref_input": {"x": preimages[0]},
            "ref_output": y,
            "ref_code": full_code,
            "funcname": "main_solution",
            "num_answers": k,
            "gt_info": {"num_answers": k},
            "known_preimages": preimages[: min(len(preimages), 5)],
            "preimage_lower_bound": len(preimages),
        }
        sample = {
            "data_source": data_source,
            "prompt": [{"role": "user", "content": prompt_content}],
            "ability": "reasoning",
            "reward_model": {"style": "rule", "ground_truth": json.dumps(gt)},
            "extra_info": {
                "index": 0,
                "split": "dummy",
                "depth": 1,
                "direction": "backward_multi",
                "expr": expr,
                "func": fname,
                "num_answers": k,
                "gt_info": {"num_answers": k},
                "output_len": len(y),
                "input_len": len(x),
            },
        }
        data.append(sample)
        pbar.update(1)

    pbar.close()
    _write_samples(data, args.save_path)
    print(f"Saved {len(data)} samples to {args.save_path}.")
