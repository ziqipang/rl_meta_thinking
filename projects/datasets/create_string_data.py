"""
24 Bijective Digit-String Operations for RL Research
===============================================
All functions are invertible: given output, the input can be uniquely recovered.
Single argument (digit string) only. Organized by difficulty for LLM mental simulation.

Difficulty is estimated based on:
- Number of characters an LLM must track simultaneously
- Non-locality of transformations
- Mathematical complexity of index/character mappings
- Number of conceptual steps required
"""

from __future__ import annotations

from argparse import ArgumentParser
import inspect
import json
import random
import string
from typing import Callable
from math import gcd

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

# ==================================================
# TIER 1: EASY (Direct pattern, few steps)
# ==================================================

def func_00(s):
    """Reverse the digit string.
    '1234' -> '4321'
    Self-inverse."""
    return s[::-1]


def func_01(s):
    """Rotate left by 1 position.
    '1234' -> '2341'
    Inverse: rotate right by 1."""
    if len(s) <= 1:
        return s
    return s[1:] + s[0]


def func_02(s):
    """Swap first and last characters.
    '12345' -> '52341'
    Self-inverse."""
    if len(s) <= 1:
        return s
    return s[-1] + s[1:-1] + s[0]


def func_03(s):
    """Swap adjacent digit pairs: 0<->1, 2<->3, ..., 8<->9.
    '0123456789' -> '1032547698'
    Self-inverse."""
    result = []
    for ch in s:
        if '0' <= ch <= '9':
            idx = ord(ch) - ord('0')
            new_idx = idx ^ 1  # 0<->1, 2<->3, ..., 8<->9
            result.append(chr(new_idx + ord('0')))
        else:
            result.append(ch)
    return ''.join(result)


def func_04(s):
    """Swap adjacent pairs of characters.
    '123456' -> '214365'
    Self-inverse. Last char unchanged if odd length."""
    result = list(s)
    for i in range(0, len(s) - 1, 2):
        result[i], result[i + 1] = result[i + 1], result[i]
    return ''.join(result)


def func_05(s):
    """Digit ROT5: shift digits by 5 positions (mod 10).
    '0123' -> '5678'
    Self-inverse (5+5=10)."""
    result = []
    for ch in s:
        if '0' <= ch <= '9':
            result.append(chr((ord(ch) - ord('0') + 5) % 10 + ord('0')))
        else:
            result.append(ch)
    return ''.join(result)


# ==================================================
# TIER 2: EASY-MEDIUM (Two-step or half-based)
# ==================================================

def func_06(s):
    """Swap first half with second half.
    '123456' -> '456123'
    For odd length, middle char stays in place.
    Self-inverse."""
    n = len(s)
    if n <= 1:
        return s
    mid = n // 2
    if n % 2 == 0:
        return s[mid:] + s[:mid]
    else:
        return s[mid + 1:] + s[mid] + s[:mid]


def func_07(s):
    """Reverse each half independently.
    '123456' -> '321654'
    For odd length, middle char stays in place.
    Self-inverse."""
    n = len(s)
    if n <= 1:
        return s
    mid = n // 2
    if n % 2 == 0:
        return s[:mid][::-1] + s[mid:][::-1]
    else:
        return s[:mid][::-1] + s[mid] + s[mid + 1:][::-1]


def func_08(s):
    """Digit complement: 0<->9, 1<->8, ..., 9<->0.
    '0123' -> '9876'
    Self-inverse."""
    result = []
    for ch in s:
        if '0' <= ch <= '9':
            result.append(chr(ord('9') - (ord(ch) - ord('0'))))
        else:
            result.append(ch)
    return ''.join(result)


def func_09(s):
    """Move odd-indexed characters to front, even-indexed to back.
    '123456' -> '246135' (indices 1,3,5 then 0,2,4)
    Inverse: interleave back."""
    return s[1::2] + s[0::2]


def func_10(s):
    """Digit shift: add 3 (mod 10).
    '789' -> '012'
    Inverse: shift by -3 (or +7)."""
    result = []
    for ch in s:
        if '0' <= ch <= '9':
            result.append(chr((ord(ch) - ord('0') + 3) % 10 + ord('0')))
        else:
            result.append(ch)
    return ''.join(result)


def func_11(s):
    """Rotate left by 3 positions.
    '123456' -> '456123'
    Inverse: rotate right by 3."""
    if len(s) <= 3:
        return s
    k = 3 % len(s)
    return s[k:] + s[:k]


# ==================================================
# TIER 3: MEDIUM (Index-dependent, interleaving)
# ==================================================

def func_12(s):
    """Interleave first half with second half.
    '123456' -> '142536' (take alternately from each half)
    For odd length, extra char from first half goes at end."""
    n = len(s)
    if n <= 1:
        return s
    mid = (n + 1) // 2
    first, second = s[:mid], s[mid:]
    result = []
    for i in range(len(second)):
        result.append(first[i])
        result.append(second[i])
    if len(first) > len(second):
        result.append(first[-1])
    return ''.join(result)


def func_13(s):
    """Position-dependent digit shift: add i (mod 10) to digit at position i.
    '012345' -> '024680' (0+0,1+1,2+2,3+3,4+4,5+5)
    Inverse: subtract i (mod 10)."""
    result = []
    for i, ch in enumerate(s):
        if '0' <= ch <= '9':
            idx = ord(ch) - ord('0')
            new_idx = (idx + i) % 10
            result.append(chr(new_idx + ord('0')))
        else:
            result.append(ch)
    return ''.join(result)


def func_14(s):
    """Reverse, then swap adjacent pairs.
    '123456' -> '654321' -> '563412'
    Inverse: swap pairs, then reverse."""
    t = s[::-1]
    result = list(t)
    for i in range(0, len(result) - 1, 2):
        result[i], result[i + 1] = result[i + 1], result[i]
    return ''.join(result)


def func_15(s):
    """Cycle groups of 3: abc -> cab, def -> fde, etc.
    '12345678' -> '31264578'
    Within each triplet, rotate right by 1. Leftover chars unchanged."""
    result = list(s)
    for i in range(0, len(s) - 2, 3):
        result[i], result[i + 1], result[i + 2] = result[i + 2], result[i], result[i + 1]
    return ''.join(result)


def func_16(s):
    """Permute using coprime multiplier 3.
    New position i gets character from position (i*3) mod n.
    Inverse: use modular inverse of 3."""
    n = len(s)
    if n == 0:
        return s
    mult = 3
    while gcd(mult, n) != 1:
        mult += 2
    return ''.join(s[(i * mult) % n] for i in range(n))


def func_17(s):
    """Multiplicative position shift: add (i*7) (mod 10) to digit at position i.
    Uses 7 as multiplier for varied shifts.
    Inverse: subtract (i*7) (mod 10)."""
    result = []
    for i, ch in enumerate(s):
        if '0' <= ch <= '9':
            idx = ord(ch) - ord('0')
            shift = (i * 7) % 10
            new_idx = (idx + shift) % 10
            result.append(chr(new_idx + ord('0')))
        else:
            result.append(ch)
    return ''.join(result)


# ==================================================
# TIER 4: MEDIUM-HARD (Multi-step, non-local)
# ==================================================

def func_18(s):
    """Reverse inner substring (exclude first and last char).
    '123456' -> '165432'
    Self-inverse."""
    if len(s) <= 2:
        return s
    return s[0] + s[1:-1][::-1] + s[-1]


def func_19(s):
    """Swap halves, then reverse each half.
    '123456' -> '456123' -> '654321'
    Inverse: reverse each half, then swap."""
    n = len(s)
    if n <= 1:
        return s
    mid = n // 2
    if n % 2 == 0:
        swapped = s[mid:] + s[:mid]
        return swapped[:mid][::-1] + swapped[mid:][::-1]
    else:
        swapped = s[mid + 1:] + s[mid] + s[:mid]
        first_len = n - mid - 1
        return swapped[:first_len][::-1] + swapped[first_len] + swapped[first_len + 1:][::-1]


def func_20(s):
    """Alternating digit shift: even positions +3, odd positions -3 (mod 10).
    Inverse: even positions -3, odd positions +3."""
    result = []
    for i, ch in enumerate(s):
        if '0' <= ch <= '9':
            shift = 3 if i % 2 == 0 else -3
            result.append(chr((ord(ch) - ord('0') + shift) % 10 + ord('0')))
        else:
            result.append(ch)
    return ''.join(result)


def func_21(s):
    """Bit-reversal permutation of indices (padded to power of 2 conceptually).
    For simplicity, reverse the binary representation of each index.
    Falls back to length-aware permutation."""
    n = len(s)
    if n <= 1:
        return s
    # Find smallest power of 2 >= n for bit width
    bits = (n - 1).bit_length()
    
    def reverse_bits(x, width):
        result = 0
        for _ in range(width):
            result = (result << 1) | (x & 1)
            x >>= 1
        return result
    
    # Create permutation, only using valid indices
    perm = []
    used = set()
    for i in range(n):
        rev = reverse_bits(i, bits)
        if rev < n and rev not in used:
            perm.append(rev)
            used.add(rev)
        elif i not in used:
            perm.append(i)
            used.add(i)
    
    # Ensure it's a valid permutation
    if len(perm) != n or len(set(perm)) != n:
        perm = list(range(n))[::-1]  # fallback to reverse
    
    return ''.join(s[perm[i]] for i in range(n))


# ==================================================
# TIER 5: HARD (Complex multi-step, tracking required)
# ==================================================

def func_22(s):
    """Feistel-like round using modular addition in Z_10 on digits.
    Split into equal halves, apply Feistel: L' = R, R' = L + f(R) mod 10.
    f(x) = sum of digit values mod 10.
    For odd length, last char is preserved and excluded from transform.
    Inverse: reverse the Feistel round with subtraction."""
    n = len(s)
    if n <= 1:
        return s
    
    # Handle odd length by preserving last char
    if n % 2 == 1:
        core = s[:-1]
        tail = s[-1]
    else:
        core = s
        tail = ""
    
    mid = len(core) // 2
    left, right = core[:mid], core[mid:]
    
    # f(right) = sum of digit values mod 10
    f_val = sum((ord(c) - ord('0')) for c in right if '0' <= c <= '9') % 10
    
    # Feistel: new_left = right, new_right = (left + f_val) mod 10
    new_left = right
    new_right = []
    for ch in left:
        if '0' <= ch <= '9':
            idx = ord(ch) - ord('0')
            new_idx = (idx + f_val) % 10
            new_right.append(chr(new_idx + ord('0')))
        else:
            new_right.append(ch)
    
    return new_left + ''.join(new_right) + tail


def func_23(s):
    """Three-round substitution-permutation network.
    Round 1: Digit shift +7 (mod 10)
    Round 2: Reverse
    Round 3: Swap pairs
    Inverse: swap pairs, reverse, shift -7."""
    # Round 1: digit shift +7
    t = []
    for ch in s:
        if '0' <= ch <= '9':
            t.append(chr((ord(ch) - ord('0') + 7) % 10 + ord('0')))
        else:
            t.append(ch)
    t = ''.join(t)
    
    # Round 2: Reverse
    t = t[::-1]
    
    # Round 3: Swap pairs
    result = list(t)
    for i in range(0, len(result) - 1, 2):
        result[i], result[i + 1] = result[i + 1], result[i]
    
    return ''.join(result)


# ==================================================
# Function Name Mapping (identity by default)
# ==================================================

func_name_mapping = {f"func_{i:02d}": f"func_{i:02d}" for i in range(24)}


def _build_function_sets() -> tuple[dict[str, Callable[[str], str]], dict[str, Callable[[str], str]], dict[str, Callable[[str], str]]]:
    """Build `{name -> callable}` maps for all/train/eval function pools.

    - `all_set`: `func_00`..`func_23`
    - `train_set`: `func_00`..`func_17`
    - `eval_set`: `func_18`..`func_23`
    """
    all_set: dict[str, Callable[[str], str]] = {f"func_{i:02d}": globals()[f"func_{i:02d}"] for i in range(24)}

    def _idx(name: str) -> int:
        return int(name.split("_", 1)[1])

    train_set: dict[str, Callable[[str], str]] = {k: v for k, v in all_set.items() if _idx(k) <= 17}
    eval_set: dict[str, Callable[[str], str]] = {k: v for k, v in all_set.items() if _idx(k) >= 18}
    return all_set, train_set, eval_set


def random_expr(depth: int, function_names: list[str]) -> str:
    """
    Generate a deterministic expression in terms of 'x' using only single-arg string functions.

    This intentionally avoids:
    - built-in string methods (e.g., .upper())
    - parameterized operators
    - branching / tree-structured expressions
    """
    if depth <= 0:
        return "x"
    expr = "x"
    for _ in range(depth):
        fname = random.choice(function_names)
        expr = f"{fname}({expr})"
    return expr


def generate_full_code(expr: str, *, stage: int, custom_functions: dict[str, Callable[..., object]]) -> str:
    """Create the program text shown in the prompt.

    - `stage==1`: include source code for all referenced helper functions
    - otherwise: only emit `main_solution`, assuming functions are known elsewhere
    """
    used_funcs = {fname for fname in custom_functions.keys() if fname in expr}
    code_parts: list[str] = []
    if stage == 1:
        code_parts.append("from math import gcd")
        for fname in sorted(used_funcs):
            code_parts.append(inspect.getsource(custom_functions[fname]))

    code_parts.append(f"def main_solution(x):\n    return {expr}")
    full_code = "\n\n".join(code_parts)

    # Keep the mapping hook for compatibility with the reference script, but
    # default to identity when using func_XX names directly.
    for func_name, mapped_name in func_name_mapping.items():
        full_code = full_code.replace(func_name, mapped_name)
    return full_code


def generate_feasible_input(
    func: Callable[[str], object],
    *,
    attempts: int = 100,
    min_len: int = 4,
    max_len: int = 10,
) -> str | None:
    """Find a random digit-string input that makes `func(input)` return a string."""
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


def generate_random_string_function(*, max_depth: int, custom_functions: dict[str, Callable[[str], str]]) -> tuple[str, Callable[[str], str], str | None]:
    """Sample a deterministic `lambda x: ...` and a feasible input for it."""
    expr = random_expr(max_depth, function_names=list(custom_functions.keys()))
    func_str = f"lambda x: {expr}"
    try:
        func = eval(func_str, custom_functions)
    except Exception as e:
        raise ValueError(f"Error constructing function from expression: {expr}\n{e}") from e
    feasible_input = generate_feasible_input(func)
    return expr, func, feasible_input


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
    parser.add_argument("--direction", type=str, default="forward", choices=["forward", "backward"])
    parser.add_argument("--min_level", type=int, default=1)
    parser.add_argument("--max_level", type=int, default=1)
    parser.add_argument("--data_num", type=int, default=5000)
    args = parser.parse_args()

    all_set, train_set, eval_set = _build_function_sets()

    if args.stage == 1:
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
        print(f"Generating data for depth {depth} ...")
        pbar = tqdm(total=num_per_depth, desc=f"depth={depth}", unit="sample")
        while count < num_per_depth:

            try:
                expr, func, feasible_input = generate_random_string_function(max_depth=depth, custom_functions=custom_functions)
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
                data_source = f"codeio-forward-incomplete-depth{depth}"
                prompt_content = FORWARD_PROMPT.format(code=full_code, input=feasible_input)
            else:
                data_source = f"codeio-backward-incomplete-depth{depth}"
                prompt_content = BACKWARD_PROMPT.format(code=full_code, output=output)

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
                "extra_info": {"index": 0, "split": "dummy", "depth": depth, "direction": args.direction},
            }

            data.append(sample)
            count += 1
            pbar.update(1)
            if count % 1000 == 0:
                print(f"  Depth {depth}: Generated {count} samples.")
        pbar.close()

    _write_samples(data, args.save_path)
    print(f"Saved {len(data)} samples to {args.save_path}.")
