"""
Digit-String Operations for RL Research
======================================

This file defines:
- `func_00`..`func_23`: bijective digit-string transforms (unique inverse).
- `func_24+`: non-bijective digit-string transforms (many-to-one).
- `nb_00+` (aka nonbijective_v2): a second non-bijective family used via `--function_kind nonbijective_v2`.

All functions take a single digit string and return a string.
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

PREIMAGE_PROMPT = (
    "You are given a code:\n\n{code}\n\n"
    "You are told that `main_solution(x)` returns the string \"{output}\".\n"
    "Find ONE possible digit-string input `x` such that `main_solution(x)` returns exactly that string.\n\n"
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
# NON-BIJECTIVE FUNCTIONS (many-to-one)
# ==================================================

def _digitwise_map(s: str, f: Callable[[int], int]) -> str:
    out: list[str] = []
    for ch in s:
        if "0" <= ch <= "9":
            out.append(str(f(ord(ch) - ord("0"))))
        else:
            out.append(ch)
    return "".join(out)


# ---- Family A: Digitwise many-to-one collapses ----

def func_24(s):
    """PARITY01: y[i] = d_i % 2 -> outputs only {0,1}."""
    return _digitwise_map(s, lambda d: d % 2)


def func_25(s):
    """MOD3: y[i] = d_i % 3 -> outputs {0,1,2}."""
    return _digitwise_map(s, lambda d: d % 3)


def func_26(s):
    """MOD5: y[i] = d_i % 5 -> outputs {0,1,2,3,4}."""
    return _digitwise_map(s, lambda d: d % 5)


def func_27(s):
    """HALF_FLOOR: y[i] = floor(d_i/2) -> outputs {0,1,2,3,4}."""
    return _digitwise_map(s, lambda d: d // 2)


def func_28(s):
    """CLAMP5: y[i] = min(d_i, 5) -> outputs {0,1,2,3,4,5}."""
    return _digitwise_map(s, lambda d: min(d, 5))


def func_29(s):
    """ABS_FROM5: y[i] = abs(d_i - 5) -> outputs {0,1,2,3,4,5}."""
    return _digitwise_map(s, lambda d: abs(d - 5))


def func_30(s):
    """MIRROR_MIN: y[i] = min(d_i, 9 - d_i) -> outputs {0,1,2,3,4}."""
    return _digitwise_map(s, lambda d: min(d, 9 - d))


def func_31(s):
    """SQUARE_MOD10: y[i] = (d_i*d_i) % 10 -> outputs {0,1,4,5,6,9}."""
    return _digitwise_map(s, lambda d: (d * d) % 10)


def func_32(s):
    """BUCKET3_FLOOR: y[i] = 3*floor(d_i/3) -> outputs {0,3,6,9}."""
    return _digitwise_map(s, lambda d: 3 * (d // 3))


def func_33(s):
    """GCD10_MOD10: y[i] = gcd(d_i, 10) % 10 -> outputs {0,1,2,5}."""
    return _digitwise_map(s, lambda d: gcd(d, 10) % 10)


# ---- Family B: Positional masking ----

def func_34(s):
    """MASK_EVEN_IDX_ZERO (0-indexed): set indices i%2==0 to '0'."""
    out = list(s)
    for i in range(0, len(out), 2):
        out[i] = "0"
    return "".join(out)


def func_35(s):
    """MASK_MOD3_EQ0_ZERO (0-indexed): set indices i%3==0 to '0'."""
    out = list(s)
    for i in range(0, len(out), 3):
        out[i] = "0"
    return "".join(out)


def func_36(s):
    """MASK_FIRST_HALF_ZERO: set first ceil(n/2) digits to '0'."""
    n = len(s)
    k = (n + 1) // 2
    return ("0" * k) + s[k:]


# ---- Family C: Structured downsample/aggregate within pairs ----

def func_37(s):
    """REPEAT_EVEN_TO_ODD: within each pair (2i,2i+1), copy even digit to odd."""
    out = list(s)
    for i in range(0, len(out) - 1, 2):
        out[i + 1] = out[i]
    return "".join(out)


def func_38(s):
    """PAIR_MAX_REPEAT: within each pair, output (max(a,b), max(a,b))."""
    out = list(s)
    for i in range(0, len(out) - 1, 2):
        a = ord(out[i]) - ord("0")
        b = ord(out[i + 1]) - ord("0")
        m = str(max(a, b))
        out[i] = m
        out[i + 1] = m
    return "".join(out)


def func_39(s):
    """PAIR_MIN_REPEAT: within each pair, output (min(a,b), min(a,b))."""
    out = list(s)
    for i in range(0, len(out) - 1, 2):
        a = ord(out[i]) - ord("0")
        b = ord(out[i + 1]) - ord("0")
        m = str(min(a, b))
        out[i] = m
        out[i + 1] = m
    return "".join(out)


# ---- Family D: Permutation-invariance via sorting ----

def func_40(s):
    """SORT_PAIRS_ASC: sort digits inside each length-2 block ascending."""
    out = list(s)
    for i in range(0, len(out) - 1, 2):
        a, b = out[i], out[i + 1]
        if a > b:
            out[i], out[i + 1] = b, a
    return "".join(out)


def func_41(s):
    """SORT_TRIPLES_ASC: sort digits inside each length-3 block ascending."""
    out: list[str] = []
    for i in range(0, len(s), 3):
        block = sorted(s[i : i + 3])
        out.extend(block)
    return "".join(out)


def func_42(s):
    """SORT_GLOBAL_ASC: sort all digits globally ascending."""
    return "".join(sorted(s))


# ---- Family E: Neighbor-coupled lossy transform ----

def func_43(s):
    """ADJ_SUM_MOD10: y[i] = (d_i + d_{i+1}) % 10 for i=0..n-2 (length n-1)."""
    if len(s) < 2:
        return ""
    out: list[str] = []
    for i in range(len(s) - 1):
        a = ord(s[i]) - ord("0")
        b = ord(s[i + 1]) - ord("0")
        out.append(str((a + b) % 10))
    return "".join(out)


# ==================================================
# NON-BIJECTIVE V2: Local neighbor/triple/state transforms
# (Designed to be forward-simple but non-trivially invertible)
# ==================================================

def _d(ch: str) -> int:
    return ord(ch) - ord("0")


def _c(v: int) -> str:
    return chr(v + ord("0"))


def _fold0to4(t: int) -> int:
    # fold a digit 0..9 down to 0..4 by symmetry around 4.5
    return t if t <= 4 else 9 - t


# ==================================================
# TIER 1: CYCLIC NEIGHBOR (2-local, overlapping constraints)
# ==================================================

def nb_00(s: str) -> str:
    """Cyclic floor-average of neighbors.
    y[i] = floor((x[i] + x[i+1]) / 2) with wrap-around.
    Non-bijective; x=y almost never works because constraints overlap."""
    n = len(s)
    if n <= 1:
        return s
    out: list[str] = []
    for i in range(n):
        a = _d(s[i])
        b = _d(s[(i + 1) % n])
        out.append(_c((a + b) // 2))
    return "".join(out)


def nb_01(s: str) -> str:
    """Cyclic ceil-average of neighbors.
    y[i] = ceil((x[i] + x[i+1]) / 2) = (a+b+1)//2."""
    n = len(s)
    if n <= 1:
        return s
    out: list[str] = []
    for i in range(n):
        a = _d(s[i])
        b = _d(s[(i + 1) % n])
        out.append(_c((a + b + 1) // 2))
    return "".join(out)


def nb_02(s: str) -> str:
    """Cyclic absolute difference.
    y[i] = |x[i] - x[i+1]| (wrap-around).
    Inversion becomes branching (±) globally, not per-position independent."""
    n = len(s)
    if n <= 1:
        return ("0" * n) if n == 1 else s
    out: list[str] = []
    for i in range(n):
        a = _d(s[i])
        b = _d(s[(i + 1) % n])
        out.append(_c(abs(a - b)))
    return "".join(out)


def nb_03(s: str) -> str:
    """Cyclic capped sum.
    y[i] = min(x[i] + x[i+1], 9).
    Loses carry magnitude; overlapping caps block trivial inverses."""
    n = len(s)
    if n <= 1:
        return s
    out: list[str] = []
    for i in range(n):
        a = _d(s[i])
        b = _d(s[(i + 1) % n])
        out.append(_c(min(a + b, 9)))
    return "".join(out)


def nb_04(s: str) -> str:
    """Cyclic overflow amount past 9.
    y[i] = max(x[i] + x[i+1] - 9, 0).
    Output encodes 'how much overflow' but loses exact operands."""
    n = len(s)
    if n <= 1:
        return ("0" * n) if n == 1 else s
    out: list[str] = []
    for i in range(n):
        a = _d(s[i])
        b = _d(s[(i + 1) % n])
        out.append(_c(max(a + b - 9, 0)))
    return "".join(out)


def nb_05(s: str) -> str:
    """Cyclic distance of neighbor-sum to 9.
    y[i] = |(x[i] + x[i+1]) - 9|.
    Nonlinear; lots of collisions; no projector behavior."""
    n = len(s)
    if n <= 1:
        return s
    out: list[str] = []
    for i in range(n):
        a = _d(s[i])
        b = _d(s[(i + 1) % n])
        out.append(_c(abs(a + b - 9)))
    return "".join(out)


def nb_06(s: str) -> str:
    """Cyclic folded neighbor-sum mod 10 (folds 0..9 to 0..4).
    t = (x[i] + x[i+1]) % 10
    y[i] = fold(t) in 0..4
    Many-to-one + overlapping constraints => no trivial right-inverse."""
    n = len(s)
    if n <= 1:
        return s
    out: list[str] = []
    for i in range(n):
        a = _d(s[i])
        b = _d(s[(i + 1) % n])
        t = (a + b) % 10
        out.append(_c(_fold0to4(t)))
    return "".join(out)


def nb_07(s: str) -> str:
    """Cyclic folded neighbor-difference mod 10.
    t = (x[i] - x[i+1]) % 10
    y[i] = fold(t) in 0..4
    Inversion is not 'copy y' and not independent per digit."""
    n = len(s)
    if n <= 1:
        return s
    out: list[str] = []
    for i in range(n):
        a = _d(s[i])
        b = _d(s[(i + 1) % n])
        t = (a - b) % 10
        out.append(_c(_fold0to4(t)))
    return "".join(out)


def nb_08(s: str) -> str:
    """Cyclic (avg + absdiff) mod 10.
    y[i] = ( floor((a+b)/2) + |a-b| ) % 10
    Designed to break easy inverses; still forward-simple."""
    n = len(s)
    if n <= 1:
        return s
    out: list[str] = []
    for i in range(n):
        a = _d(s[i])
        b = _d(s[(i + 1) % n])
        val = ((a + b) // 2 + abs(a - b)) % 10
        out.append(_c(val))
    return "".join(out)


def nb_09(s: str) -> str:
    """Cyclic 'two-views' merge.
    u = min(a+b, 9)
    v = |a-b|
    y[i] = min(u, v)
    Nonlinear compression; avoids idempotent/projector patterns."""
    n = len(s)
    if n <= 1:
        return s
    out: list[str] = []
    for i in range(n):
        a = _d(s[i])
        b = _d(s[(i + 1) % n])
        u = min(a + b, 9)
        v = abs(a - b)
        out.append(_c(min(u, v)))
    return "".join(out)


# ==================================================
# TIER 2: CYCLIC TRIPLES (3-local, more constraints)
# ==================================================

def nb_10(s: str) -> str:
    """Cyclic triple average floor.
    y[i] = floor((x[i] + x[i+1] + x[i+2]) / 3)."""
    n = len(s)
    if n <= 1:
        return s
    if n == 2:
        out: list[str] = []
        for i in range(n):
            a = _d(s[i])
            b = _d(s[(i + 1) % n])
            out.append(_c((a + b) // 2))
        return "".join(out)
    out: list[str] = []
    for i in range(n):
        a = _d(s[i])
        b = _d(s[(i + 1) % n])
        c = _d(s[(i + 2) % n])
        out.append(_c((a + b + c) // 3))
    return "".join(out)


def nb_11(s: str) -> str:
    """Cyclic triple average ceil.
    y[i] = ceil((a+b+c)/3) = (a+b+c+2)//3."""
    n = len(s)
    if n <= 1:
        return s
    if n == 2:
        out: list[str] = []
        for i in range(n):
            a = _d(s[i])
            b = _d(s[(i + 1) % n])
            out.append(_c((a + b + 1) // 2))
        return "".join(out)
    out: list[str] = []
    for i in range(n):
        a = _d(s[i])
        b = _d(s[(i + 1) % n])
        c = _d(s[(i + 2) % n])
        out.append(_c((a + b + c + 2) // 3))
    return "".join(out)


def nb_12(s: str) -> str:
    """Cyclic triple range.
    y[i] = max(a,b,c) - min(a,b,c)."""
    n = len(s)
    if n <= 1:
        return ("0" * n) if n == 1 else s
    if n == 2:
        out: list[str] = []
        for i in range(n):
            a = _d(s[i])
            b = _d(s[(i + 1) % n])
            out.append(_c(abs(a - b)))
        return "".join(out)
    out: list[str] = []
    for i in range(n):
        a = _d(s[i])
        b = _d(s[(i + 1) % n])
        c = _d(s[(i + 2) % n])
        out.append(_c(max(a, b, c) - min(a, b, c)))
    return "".join(out)


def nb_13(s: str) -> str:
    """Cyclic folded triple sum mod 10.
    t = (a+b+c) % 10; y = fold(t) in 0..4."""
    n = len(s)
    if n <= 1:
        return s
    if n == 2:
        out: list[str] = []
        for i in range(n):
            a = _d(s[i])
            b = _d(s[(i + 1) % n])
            t = (a + b) % 10
            out.append(_c(_fold0to4(t)))
        return "".join(out)
    out: list[str] = []
    for i in range(n):
        a = _d(s[i])
        b = _d(s[(i + 1) % n])
        c = _d(s[(i + 2) % n])
        t = (a + b + c) % 10
        out.append(_c(_fold0to4(t)))
    return "".join(out)


def nb_14(s: str) -> str:
    """Cyclic second-difference magnitude mod 10.
    y[i] = |a - 2b + c| % 10  (nonlinear; overlapping)."""
    n = len(s)
    if n <= 2:
        return ("0" * n) if n > 0 else s
    out: list[str] = []
    for i in range(n):
        a = _d(s[i])
        b = _d(s[(i + 1) % n])
        c = _d(s[(i + 2) % n])
        val = abs(a - 2 * b + c) % 10
        out.append(_c(val))
    return "".join(out)


def nb_15(s: str) -> str:
    """Cyclic clamped 'abs(a+b-c)'.
    y[i] = min(|a + b - c|, 9).
    Many-to-one; avoids projector/fixed-point issues."""
    n = len(s)
    if n <= 1:
        return s
    if n == 2:
        out: list[str] = []
        for i in range(n):
            a = _d(s[i])
            b = _d(s[(i + 1) % n])
            out.append(_c(abs(a + b - 9)))
        return "".join(out)
    out: list[str] = []
    for i in range(n):
        a = _d(s[i])
        b = _d(s[(i + 1) % n])
        c = _d(s[(i + 2) % n])
        out.append(_c(min(abs(a + b - c), 9)))
    return "".join(out)


# ==================================================
# TIER 3: LOSSY STATE (single rule; no template composition)
# ==================================================

def nb_16(s: str) -> str:
    """Prefix-sum -> evenized emission.
    state = (state + digit) % 10
    y[i] = (2*state) % 10  (loses the +/-5 ambiguity each step)."""
    n = len(s)
    if n == 0:
        return s
    out: list[str] = []
    st = 0
    for ch in s:
        st = (st + _d(ch)) % 10
        out.append(_c((2 * st) % 10))
    return "".join(out)


def nb_17(s: str) -> str:
    """Prefix-sum -> folded emission (0..4).
    y[i] = min(state, 9-state)."""
    n = len(s)
    if n == 0:
        return s
    out: list[str] = []
    st = 0
    for ch in s:
        st = (st + _d(ch)) % 10
        out.append(_c(min(st, 9 - st)))
    return "".join(out)


def nb_18(s: str) -> str:
    """Prefix-sum -> square emission.
    y[i] = (state^2) % 10 (many-to-one because squaring mod 10 is not injective)."""
    n = len(s)
    if n == 0:
        return s
    out: list[str] = []
    st = 0
    for ch in s:
        st = (st + _d(ch)) % 10
        out.append(_c((st * st) % 10))
    return "".join(out)


def nb_19(s: str) -> str:
    """Affine state update + folded emission.
    state = (2*state + digit) % 10
    y[i] = min(state, 9-state)
    Strong set-valued behavior; still a single coherent operation."""
    n = len(s)
    if n == 0:
        return s
    out: list[str] = []
    st = 0
    for ch in s:
        st = (2 * st + _d(ch)) % 10
        out.append(_c(min(st, 9 - st)))
    return "".join(out)


def nb_20(s: str) -> str:
    """Cyclic median-of-3.
    y[i] = median(x[i], x[i+1], x[i+2]) with wrap-around.
    Non-bijective + overlapping constraints => x=y almost never works (except special cases)."""
    n = len(s)
    if n <= 1:
        return s
    if n == 2:
        a = _d(s[0])
        b = _d(s[1])
        m = (a + b) // 2
        return _c(m) + _c(m)

    out: list[str] = []
    for i in range(n):
        a = _d(s[i])
        b = _d(s[(i + 1) % n])
        c = _d(s[(i + 2) % n])
        lo = min(a, b, c)
        hi = max(a, b, c)
        med = a + b + c - lo - hi
        out.append(_c(med))
    return "".join(out)


def nb_21(s: str) -> str:
    """Cyclic rank-of-first-in-triple (0/1/2).
    y[i] = 0 if x[i] is (strict) min of {x[i],x[i+1],x[i+2]}
           2 if x[i] is (strict) max
           1 otherwise (including ties/all-equal).
    Very non-bijective; kills x=y hacks because outputs are only in {0,1,2}."""
    n = len(s)
    if n == 0:
        return s
    if n == 1:
        return "1"  # tie by definition
    if n == 2:
        a = _d(s[0])
        b = _d(s[1])
        if a < b:
            r0, r1 = 0, 2
        elif a > b:
            r0, r1 = 2, 0
        else:
            r0, r1 = 1, 1
        return _c(r0) + _c(r1)

    out: list[str] = []
    for i in range(n):
        a = _d(s[i])
        b = _d(s[(i + 1) % n])
        c = _d(s[(i + 2) % n])

        is_min = (a <= b and a <= c) and (a < b or a < c)
        is_max = (a >= b and a >= c) and (a > b or a > c)

        out.append("0" if is_min else ("2" if is_max else "1"))
    return "".join(out)


def nb_22(s: str) -> str:
    """Cyclic capped neighbor product.
    y[i] = min(x[i] * x[i+1], 9) with wrap-around.
    Strongly non-bijective; overlapping constraints; not a projector."""
    n = len(s)
    if n == 0:
        return s
    if n == 1:
        a = _d(s[0])
        return _c(min(a * a, 9))

    out: list[str] = []
    for i in range(n):
        a = _d(s[i])
        b = _d(s[(i + 1) % n])
        out.append(_c(min(a * b, 9)))
    return "".join(out)


def nb_23(s: str) -> str:
    """Cyclic local total variation (3-local).
    y[i] = min(|x[i]-x[i+1]| + |x[i+1]-x[i+2]|, 9) with wrap-around.
    Non-bijective; inversion requires globally consistent assignments."""
    n = len(s)
    if n == 0:
        return s
    if n == 1:
        return "0"
    if n == 2:
        a = _d(s[0])
        b = _d(s[1])
        v = min(2 * abs(a - b), 9)
        return _c(v) + _c(v)

    out: list[str] = []
    for i in range(n):
        a = _d(s[i])
        b = _d(s[(i + 1) % n])
        c = _d(s[(i + 2) % n])
        out.append(_c(min(abs(a - b) + abs(b - c), 9)))
    return "".join(out)


def nb_24(s: str) -> str:
    """Lossy state machine with quantized emission.
    state = (3*state + digit) % 10
    y[i]  = state // 2     (digits 0..4)
    Non-bijective due to quantization; avoids x=y because output alphabet shrinks and depends on history."""
    n = len(s)
    if n == 0:
        return s
    out: list[str] = []
    st = 0
    for ch in s:
        st = (3 * st + _d(ch)) % 10
        out.append(_c(st // 2))
    return "".join(out)


# ==================================================
# Function Name Mapping (identity by default)
# ==================================================

NUM_BIJECTIVE_FUNCS = 24  # func_00..func_23
NUM_NONBIJECTIVE_FUNCS = 20  # func_24..func_43
NUM_NB_V2_FUNCS = 25  # nb_00..nb_24
NUM_TOTAL_FUNCS = NUM_BIJECTIVE_FUNCS + NUM_NONBIJECTIVE_FUNCS

func_name_mapping = {f"func_{i:02d}": f"func_{i:02d}" for i in range(NUM_TOTAL_FUNCS)}
func_name_mapping.update({f"nb_{i:02d}": f"func_{(NUM_BIJECTIVE_FUNCS + i):02d}" for i in range(NUM_NB_V2_FUNCS)})

BIJECTIVE_FUNCTION_NAMES = [f"func_{i:02d}" for i in range(NUM_BIJECTIVE_FUNCS)]
NONBIJECTIVE_FUNCTION_NAMES = [f"func_{i:02d}" for i in range(NUM_BIJECTIVE_FUNCS, NUM_TOTAL_FUNCS)]
NB_V2_FUNCTION_NAMES = [f"nb_{i:02d}" for i in range(NUM_NB_V2_FUNCS)]

# Default function splits (can be overridden via CLI flags).
DEFAULT_TRAIN_FUNCTION_NAMES = [f"func_{i:02d}" for i in range(0, 18)] + [f"func_{i:02d}" for i in range(24, 38)]
DEFAULT_TEST_FUNCTION_NAMES = [f"func_{i:02d}" for i in range(18, 24)] + [f"func_{i:02d}" for i in range(38, 44)]

# --- nonbijective_v2 split options (selected via CLI) ---
# Original split (historical): train nb_00..nb_17, test nb_18..nb_24.
ORIGINAL_NB_V2_TRAIN_FUNCTION_NAMES = [f"nb_{i:02d}" for i in range(0, 18)]
ORIGINAL_NB_V2_TEST_FUNCTION_NAMES = [f"nb_{i:02d}" for i in range(18, 25)]

# Easier-than-original split: make eval more lossy (more preimages per output on average),
# useful as a sanity-check baseline where pass@1 should be noticeably higher.
EASIER_NB_V2_TEST_FUNCTION_NAMES = [
    # Extremely lossy
    "nb_21",  # rank in triple -> outputs only {0,1,2} (and wrap-around constraints)
    "nb_20",  # cyclic median-of-3
    # Lossy folded neighbor ops
    "nb_06",  # folded (a+b)%10 -> 0..4
    "nb_07",  # folded (a-b)%10 -> 0..4
    # Other lossy/compressive ops
    "nb_23",  # local total variation clamped
    "nb_13",  # folded triple sum -> 0..4
    "nb_16",  # prefix-sum evenized emission
]
EASIER_NB_V2_TRAIN_FUNCTION_NAMES = [
    n for n in NB_V2_FUNCTION_NAMES if n not in set(EASIER_NB_V2_TEST_FUNCTION_NAMES)
]

# Balanced split: mix tiers in both train and eval.
# Tiers:
# - Tier 1: nb_00..nb_09
# - Tier 2: nb_10..nb_15
# - Tier 3: nb_16..nb_24
BALANCED_NB_V2_TEST_FUNCTION_NAMES = [
    # Tier 1
    "nb_02",
    "nb_06",
    "nb_09",
    # Tier 2
    "nb_12",
    "nb_14",
    # Tier 3
    "nb_18",
    "nb_24",
]
BALANCED_NB_V2_TRAIN_FUNCTION_NAMES = [
    n for n in NB_V2_FUNCTION_NAMES if n not in set(BALANCED_NB_V2_TEST_FUNCTION_NAMES)
]

# Back-compat defaults (used unless overridden by --nb_v2_split).
DEFAULT_NB_V2_TRAIN_FUNCTION_NAMES = BALANCED_NB_V2_TRAIN_FUNCTION_NAMES
DEFAULT_NB_V2_TEST_FUNCTION_NAMES = BALANCED_NB_V2_TEST_FUNCTION_NAMES

def _build_function_sets(
    *,
    function_kind: str,
    nb_v2_split: str = "balanced",
) -> tuple[dict[str, Callable[[str], str]], dict[str, Callable[[str], str]], dict[str, Callable[[str], str]]]:
    """Build `{name -> callable}` maps for all/train/eval function pools.

    `function_kind`:
      - `bijective`: only `func_00`..`func_23`
      - `nonbijective`: only `func_24+`
      - `both`: union of both sets

    Default split:
      - bijective train: `func_00`..`func_17`
      - bijective test: `func_18`..`func_23`
      - nonbijective train: `func_24`..`func_33` (Family A)
      - nonbijective test: `func_34`..`func_43` (Families B..E)
    """
    kind = function_kind.lower().strip()
    if kind not in {"bijective", "nonbijective", "nonbijective_v2", "both"}:
        raise ValueError(f"Unknown function_kind: {function_kind!r}")

    if kind == "bijective":
        allowed_names = set(BIJECTIVE_FUNCTION_NAMES)
    elif kind == "nonbijective":
        allowed_names = set(NONBIJECTIVE_FUNCTION_NAMES)
    elif kind == "nonbijective_v2":
        allowed_names = set(NB_V2_FUNCTION_NAMES)
    else:
        allowed_names = set(BIJECTIVE_FUNCTION_NAMES) | set(NONBIJECTIVE_FUNCTION_NAMES)

    all_set: dict[str, Callable[[str], str]] = {name: globals()[name] for name in sorted(allowed_names)}

    if kind == "nonbijective_v2":
        split = nb_v2_split.lower().strip()
        if split == "original":
            train_names = ORIGINAL_NB_V2_TRAIN_FUNCTION_NAMES
            test_names = ORIGINAL_NB_V2_TEST_FUNCTION_NAMES
        elif split == "easier":
            train_names = EASIER_NB_V2_TRAIN_FUNCTION_NAMES
            test_names = EASIER_NB_V2_TEST_FUNCTION_NAMES
        elif split == "balanced":
            train_names = DEFAULT_NB_V2_TRAIN_FUNCTION_NAMES
            test_names = DEFAULT_NB_V2_TEST_FUNCTION_NAMES
        else:
            raise ValueError(
                f"Unknown nb_v2_split: {nb_v2_split!r} (expected 'balanced', 'original', or 'easier')"
            )

        train_selected = [n for n in train_names if n in allowed_names]
        test_selected = [n for n in test_names if n in allowed_names]
    else:
        train_selected = [n for n in DEFAULT_TRAIN_FUNCTION_NAMES if n in allowed_names]
        test_selected = [n for n in DEFAULT_TEST_FUNCTION_NAMES if n in allowed_names]

    train_set = {name: all_set[name] for name in train_selected}
    eval_set = {name: all_set[name] for name in test_selected}
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
        code_parts.append("from __future__ import annotations")
        code_parts.append("from math import gcd")
        if any(fname in used_funcs for fname in (f"func_{i:02d}" for i in range(24, 34))):
            code_parts.append(inspect.getsource(_digitwise_map))
        if any(fname in used_funcs for fname in NB_V2_FUNCTION_NAMES):
            for _helper_fn in [_d, _c, _fold0to4]:
                code_parts.append(inspect.getsource(_helper_fn))
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
    min_len: int = 2,
    max_len: int = 6,
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
    parser.add_argument(
        "--function_kind",
        type=str,
        default="bijective",
        choices=["bijective", "nonbijective", "nonbijective_v2", "both"],
        help="Which function family to sample from.",
    )
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
    args = parser.parse_args()

    all_set, train_set, eval_set = _build_function_sets(
        function_kind=args.function_kind,
        nb_v2_split=args.nb_v2_split,
    )

    if args.stage == 1 and args.stage1_pool == "all":
        custom_functions = all_set
    else:
        custom_functions = train_set if args.split == "train" else eval_set

    uses_nonbijective = any(name in NONBIJECTIVE_FUNCTION_NAMES or name in NB_V2_FUNCTION_NAMES for name in custom_functions.keys())
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
