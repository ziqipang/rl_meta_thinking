
"""
Atomic string transformation functions v4.

Design goals:
- 24 bijective + 24 non-bijective transforms.
- Same input/output length.
- Digit-only style, concise and easy to simulate.
- Avoid self-inverse bijections.
- Avoid trivial fixed-point behavior on lengths 2-6 as much as possible.
- Keep the visual / coding style close to the original library.
"""

from __future__ import annotations

from itertools import product


# ==================================================
# Helpers
# ==================================================

def _d(ch: str) -> int:
    return ord(ch) - ord("0")


def _c(v: int) -> str:
    return chr((v % 10) + ord("0"))


def _shift_all(s: str, k: int) -> str:
    return "".join(_c(_d(ch) + k) for ch in s)


def _shift_alt(s: str, a: int, b: int) -> str:
    return "".join(_c(_d(ch) + (a if i % 2 == 0 else b)) for i, ch in enumerate(s))


def _shift_pos_linear(s: str, mul: int, add: int) -> str:
    return "".join(_c(_d(ch) + mul * i + add) for i, ch in enumerate(s))


def _shift_halves(s: str, left: int, right: int, middle: int) -> str:
    n = len(s)
    mid = n // 2
    out: list[str] = []
    for i, ch in enumerate(s):
        if n % 2 == 1 and i == mid:
            out.append(_c(_d(ch) + middle))
        elif i < mid:
            out.append(_c(_d(ch) + left))
        else:
            out.append(_c(_d(ch) + right))
    return "".join(out)


def _shift_mod3(s: str) -> str:
    return "".join(_c(_d(ch) + (i % 3) + 1) for i, ch in enumerate(s))


def _rotl(s: str, k: int) -> str:
    n = len(s)
    if n <= 1:
        return s
    k %= n
    return s[k:] + s[:k]


def _swap_ends(s: str) -> str:
    if len(s) <= 1:
        return s
    return s[-1] + s[1:-1] + s[0]


def _swap_pairs(s: str) -> str:
    result = list(s)
    for i in range(0, len(s) - 1, 2):
        result[i], result[i + 1] = result[i + 1], result[i]
    return "".join(result)


def _swap_halves(s: str) -> str:
    n = len(s)
    if n <= 1:
        return s
    mid = n // 2
    if n % 2 == 0:
        return s[mid:] + s[:mid]
    return s[mid + 1:] + s[mid] + s[:mid]


def _reverse_halves(s: str) -> str:
    n = len(s)
    if n <= 1:
        return s
    mid = n // 2
    if n % 2 == 0:
        return s[:mid][::-1] + s[mid:][::-1]
    return s[:mid][::-1] + s[mid] + s[mid + 1:][::-1]


def _odd_front(s: str) -> str:
    return s[1::2] + s[0::2]


def _interleave_halves(s: str) -> str:
    n = len(s)
    if n <= 1:
        return s
    mid = (n + 1) // 2
    first, second = s[:mid], s[mid:]
    out: list[str] = []
    for i in range(len(second)):
        out.append(first[i])
        out.append(second[i])
    if len(first) > len(second):
        out.append(first[-1])
    return "".join(out)


def _reverse_inner(s: str) -> str:
    if len(s) <= 2:
        return s
    return s[0] + s[1:-1][::-1] + s[-1]


def _triplet_right(s: str) -> str:
    result = list(s)
    for i in range(0, len(s) - 2, 3):
        result[i], result[i + 1], result[i + 2] = result[i + 2], result[i], result[i + 1]
    return "".join(result)


def _prefix_sum_emit(s: str, extra) -> str:
    state = 0
    out: list[str] = []
    for i, ch in enumerate(s):
        state = (state + _d(ch) + extra(i)) % 10
        out.append(_c(state))
    return "".join(out)


def _suffix_sum_emit(s: str, extra) -> str:
    state = 0
    out = [""] * len(s)
    for i in range(len(s) - 1, -1, -1):
        state = (state + _d(s[i]) + extra(i)) % 10
        out[i] = _c(state)
    return "".join(out)


def _cyclic_pair(s: str, rule) -> str:
    n = len(s)
    if n == 0:
        return s
    if n == 1:
        a = _d(s[0])
        return _c(rule(a, a))
    out: list[str] = []
    for i in range(n):
        a = _d(s[i])
        b = _d(s[(i + 1) % n])
        out.append(_c(rule(a, b)))
    return "".join(out)


def _cyclic_triple(s: str, pair_rule, triple_rule) -> str:
    n = len(s)
    if n == 0:
        return s
    if n == 1:
        a = _d(s[0])
        return _c(pair_rule(a, a))
    if n == 2:
        out: list[str] = []
        for i in range(n):
            a = _d(s[i])
            b = _d(s[(i + 1) % n])
            out.append(_c(pair_rule(a, b)))
        return "".join(out)
    out: list[str] = []
    for i in range(n):
        a = _d(s[i])
        b = _d(s[(i + 1) % n])
        c = _d(s[(i + 2) % n])
        out.append(_c(triple_rule(a, b, c)))
    return "".join(out)


def _mirror_map(s: str, rule) -> str:
    n = len(s)
    out: list[str] = []
    for i, ch in enumerate(s):
        a = _d(ch)
        b = _d(s[n - 1 - i])
        out.append(_c(rule(a, b)))
    return "".join(out)


def _pair_sort_asc(s: str) -> str:
    out: list[str] = []
    for i in range(0, len(s) - 1, 2):
        a, b = s[i], s[i + 1]
        if a <= b:
            out.extend([a, b])
        else:
            out.extend([b, a])
    if len(s) % 2 == 1:
        out.append(s[-1])
    return "".join(out)


def _pair_sort_desc(s: str) -> str:
    out: list[str] = []
    for i in range(0, len(s) - 1, 2):
        a, b = s[i], s[i + 1]
        if a >= b:
            out.extend([a, b])
        else:
            out.extend([b, a])
    if len(s) % 2 == 1:
        out.append(s[-1])
    return "".join(out)


def _triplet_sort_asc(s: str) -> str:
    out: list[str] = []
    i = 0
    while i + 2 < len(s):
        out.extend(sorted(s[i:i + 3]))
        i += 3
    rem = s[i:]
    if len(rem) == 2:
        out.extend(sorted(rem))
    elif len(rem) == 1:
        out.extend(rem)
    return "".join(out)


def _triplet_sort_desc(s: str) -> str:
    out: list[str] = []
    i = 0
    while i + 2 < len(s):
        out.extend(sorted(s[i:i + 3], reverse=True))
        i += 3
    rem = s[i:]
    if len(rem) == 2:
        out.extend(sorted(rem, reverse=True))
    elif len(rem) == 1:
        out.extend(rem)
    return "".join(out)


def _prefix_min(s: str) -> str:
    out: list[str] = []
    cur = "9"
    for ch in s:
        if ch < cur:
            cur = ch
        out.append(cur)
    return "".join(out)


def _prefix_max(s: str) -> str:
    out: list[str] = []
    cur = "0"
    for ch in s:
        if ch > cur:
            cur = ch
        out.append(cur)
    return "".join(out)


def _suffix_min(s: str) -> str:
    out = [""] * len(s)
    cur = "9"
    for i in range(len(s) - 1, -1, -1):
        if s[i] < cur:
            cur = s[i]
        out[i] = cur
    return "".join(out)


def _suffix_max(s: str) -> str:
    out = [""] * len(s)
    cur = "0"
    for i in range(len(s) - 1, -1, -1):
        if s[i] > cur:
            cur = s[i]
        out[i] = cur
    return "".join(out)


def _state_emit(s: str, emit, update=None) -> str:
    state = 0
    out: list[str] = []
    for i, ch in enumerate(s):
        d = _d(ch)
        state = update(state, d, i) if update is not None else (state + d) % 10
        out.append(_c(emit(state, d, i)))
    return "".join(out)


# ==================================================
# BIJECTIVE V4
# 24 functions, grouped by behavior class
# ==================================================

# --------------------------------------------------
# TIER 1: SIMPLE PERMUTATION + SMALL DRIFT
# --------------------------------------------------

def func_00(s: str) -> str:
    """Reverse, then add 1 to every digit.
    '1234' -> '4321' -> '5432'
    Inverse: subtract 1, then reverse."""
    return _shift_all(s[::-1], 1)


def func_01(s: str) -> str:
    """Rotate left by 1, then add 3 to every digit.
    '1234' -> '2341' -> '5674'
    Inverse: subtract 3, then rotate right by 1."""
    return _shift_all(_rotl(s, 1), 3)


def func_02(s: str) -> str:
    """Swap first and last characters, then add 3.
    '12345' -> '52341' -> '85674'
    Inverse: subtract 3, then swap first and last."""
    return _shift_all(_swap_ends(s), 3)


def func_03(s: str) -> str:
    """Swap adjacent pairs, then add 1.
    '123456' -> '214365' -> '325476'
    Inverse: subtract 1, then swap adjacent pairs."""
    return _shift_all(_swap_pairs(s), 1)


def func_04(s: str) -> str:
    """Swap the two halves, then add 2.
    '123456' -> '456123' -> '678345'
    Inverse: subtract 2, then swap halves back."""
    return _shift_all(_swap_halves(s), 2)


def func_05(s: str) -> str:
    """Reverse each half, then add 3.
    '123456' -> '321654' -> '654987'
    Inverse: subtract 3, then reverse each half again."""
    return _shift_all(_reverse_halves(s), 3)


# --------------------------------------------------
# TIER 2: PERMUTATION + STRUCTURED DRIFT
# --------------------------------------------------

def func_06(s: str) -> str:
    """Move odd-indexed characters to the front, then add 1.
    '123456' -> '246135' -> '357246'
    Inverse: subtract 1, then interleave back."""
    return _shift_all(_odd_front(s), 1)


def func_07(s: str) -> str:
    """Interleave the two halves, then add 2.
    '123456' -> '142536' -> '364758'
    Inverse: subtract 2, then de-interleave."""
    return _shift_all(_interleave_halves(s), 2)


def func_08(s: str) -> str:
    """Reverse the inner substring, then add 3.
    '123456' -> '154326' -> '487659'
    Inverse: subtract 3, then reverse the inner substring."""
    return _shift_all(_reverse_inner(s), 3)


def func_09(s: str) -> str:
    """Rotate each full group of 3 right by 1, then add 1.
    '12345678' -> '31264578' -> '42375689'
    Inverse: subtract 1, then rotate each triplet left by 1."""
    return _shift_all(_triplet_right(s), 1)


def func_10(s: str) -> str:
    """Reverse, then add 1 to even positions and 2 to odd positions.
    '1234' -> '4321' -> '5543'
    Inverse: subtract the alternating pattern, then reverse."""
    return _shift_alt(s[::-1], 1, 2)


def func_11(s: str) -> str:
    """Rotate left by 1, then add 1/3 on alternating positions.
    '123456' -> '234561' -> '365874'
    Inverse: subtract the alternating pattern, then rotate right by 1."""
    return _shift_alt(_rotl(s, 1), 1, 3)


# --------------------------------------------------
# TIER 3: DIGIT-ONLY POSITIONAL DRIFT
# --------------------------------------------------

def func_12(s: str) -> str:
    """Add i+1 to the digit at position i.
    '012345' -> '135791'
    Inverse: subtract i+1."""
    return _shift_pos_linear(s, 1, 1)


def func_13(s: str) -> str:
    """Add 2i+1 to the digit at position i.
    Shift pattern: 1,3,5,7,9,...
    Inverse: subtract 2i+1."""
    return _shift_pos_linear(s, 2, 1)


def func_14(s: str) -> str:
    """Add 1 on the left half, 3 on the right half, and 2 in the middle.
    '12345' -> '23578'
    Inverse: subtract the same half-based pattern."""
    return _shift_halves(s, 1, 3, 2)


def func_15(s: str) -> str:
    """Add the repeating pattern 1,2,3,1,2,3,...
    '012345' -> '135468'
    Inverse: subtract the repeating pattern."""
    return _shift_mod3(s)


# --------------------------------------------------
# TIER 4: PERMUTATION + POSITION-DEPENDENT DRIFT
# --------------------------------------------------

def func_16(s: str) -> str:
    """Reverse, then add i+1 at position i.
    '1023' -> '3201' -> '4435'
    Inverse: subtract i+1, then reverse."""
    return _shift_pos_linear(s[::-1], 1, 1)


def func_17(s: str) -> str:
    """Swap adjacent pairs, then add i+2 at position i.
    '123456' -> '214365' -> '448811'
    Inverse: subtract i+2, then swap adjacent pairs."""
    return _shift_pos_linear(_swap_pairs(s), 1, 2)


def func_18(s: str) -> str:
    """Move odd-indexed characters to the front, then add 1 on the left half,
    2 on the right half, and 3 in the middle when needed.
    Inverse: subtract the half-based pattern, then interleave back."""
    return _shift_halves(_odd_front(s), 1, 2, 3)


def func_19(s: str) -> str:
    """Swap halves, then add the repeating pattern 1,2,3,1,2,3,...
    Inverse: subtract the repeating pattern, then swap halves back."""
    return _shift_mod3(_swap_halves(s))


# --------------------------------------------------
# TIER 5: ONE-PASS STATEFUL BIJECTIONS
# --------------------------------------------------

def func_20(s: str) -> str:
    """Running sum with +1 drift.
    state = (state + digit + 1) mod 10
    output digit = state
    Inverse: subtract previous state and 1."""
    return _prefix_sum_emit(s, lambda i: 1)


def func_21(s: str) -> str:
    """Running sum with +2 drift.
    state = (state + digit + 2) mod 10
    output digit = state
    Inverse: subtract previous state and 2."""
    return _prefix_sum_emit(s, lambda i: 2)


def func_22(s: str) -> str:
    """Running sum with alternating +1/+2 drift.
    state = (state + digit + (1 or 2)) mod 10
    output digit = state
    Inverse: subtract previous state and the alternating drift."""
    return _prefix_sum_emit(s, lambda i: (i % 2) + 1)


def func_23(s: str) -> str:
    """Right-to-left running sum with +1 drift.
    state is accumulated from the end of the string.
    Inverse: subtract the next output digit and 1."""
    return _suffix_sum_emit(s, lambda i: 1)


# ==================================================
# NON-BIJECTIVE V4
# 24 functions, grouped by behavior class
# ==================================================

# --------------------------------------------------
# TIER 1: CYCLIC NEIGHBOR SUMMARIES
# --------------------------------------------------

def nb_00(s: str) -> str:
    """Cyclic floor-average of neighbors, then add 1.
    y[i] = floor((x[i] + x[i+1]) / 2) + 1 (mod 10)."""
    return _shift_all(_cyclic_pair(s, lambda a, b: (a + b) // 2), 1)


def nb_01(s: str) -> str:
    """Cyclic ceil-average of neighbors, then add 2.
    y[i] = ceil((x[i] + x[i+1]) / 2) + 2 (mod 10)."""
    return _shift_all(_cyclic_pair(s, lambda a, b: (a + b + 1) // 2), 2)


def nb_02(s: str) -> str:
    """Cyclic neighbor minimum, then add 1.
    y[i] = min(x[i], x[i+1]) + 1 (mod 10)."""
    return _shift_all(_cyclic_pair(s, lambda a, b: min(a, b)), 1)


def nb_03(s: str) -> str:
    """Cyclic capped neighbor sum, then add 1.
    y[i] = min(x[i] + x[i+1], 9) + 1 (mod 10)."""
    return _shift_all(_cyclic_pair(s, lambda a, b: min(a + b, 9)), 1)


# --------------------------------------------------
# TIER 2: CYCLIC TRIPLE SUMMARIES
# --------------------------------------------------

def nb_04(s: str) -> str:
    """Cyclic floor-average of triples, then add 1.
    For length 2, falls back to pair floor-average."""
    return _shift_all(
        _cyclic_triple(s, lambda a, b: (a + b) // 2, lambda a, b, c: (a + b + c) // 3),
        1,
    )


def nb_05(s: str) -> str:
    """Cyclic ceil-average of triples, then add 2.
    For length 2, falls back to pair ceil-average."""
    return _shift_all(
        _cyclic_triple(s, lambda a, b: (a + b + 1) // 2, lambda a, b, c: (a + b + c + 2) // 3),
        2,
    )


def nb_06(s: str) -> str:
    """Cyclic triple minimum, then add 1.
    For length 2, falls back to pair minimum."""
    return _shift_all(
        _cyclic_triple(s, lambda a, b: min(a, b), lambda a, b, c: min(a, b, c)),
        1,
    )


def nb_07(s: str) -> str:
    """Cyclic triple median, then add 2.
    For length 2, falls back to pair floor-average."""
    return _shift_all(
        _cyclic_triple(s, lambda a, b: (a + b) // 2, lambda a, b, c: sorted([a, b, c])[1]),
        2,
    )


# --------------------------------------------------
# TIER 3: MIRROR-PAIR SUMMARIES
# --------------------------------------------------

def nb_08(s: str) -> str:
    """Mirror floor-average, then add 1.
    Position i is paired with position n-1-i."""
    return _shift_all(_mirror_map(s, lambda a, b: (a + b) // 2), 1)


def nb_09(s: str) -> str:
    """Mirror ceil-average, then add 2.
    Position i is paired with position n-1-i."""
    return _shift_all(_mirror_map(s, lambda a, b: (a + b + 1) // 2), 2)


def nb_10(s: str) -> str:
    """Mirror minimum, then add 1.
    Position i is paired with position n-1-i."""
    return _shift_all(_mirror_map(s, lambda a, b: min(a, b)), 1)


def nb_11(s: str) -> str:
    """Mirror maximum, then add 2.
    Position i is paired with position n-1-i."""
    return _shift_all(_mirror_map(s, lambda a, b: max(a, b)), 2)


# --------------------------------------------------
# TIER 4: BLOCK SORTING
# --------------------------------------------------

def nb_12(s: str) -> str:
    """Sort each adjacent pair ascending, then add 1.
    Leftover last digit is just shifted."""
    return _shift_all(_pair_sort_asc(s), 1)


def nb_13(s: str) -> str:
    """Sort each adjacent pair descending, then add 2.
    Leftover last digit is just shifted."""
    return _shift_all(_pair_sort_desc(s), 2)


def nb_14(s: str) -> str:
    """Sort each full triplet ascending, then add 1.
    A leftover pair is sorted ascending."""
    return _shift_all(_triplet_sort_asc(s), 1)


def nb_15(s: str) -> str:
    """Sort each full triplet descending, then add 2.
    A leftover pair is sorted descending."""
    return _shift_all(_triplet_sort_desc(s), 2)


# --------------------------------------------------
# TIER 5: CUMULATIVE EXTREMA
# --------------------------------------------------

def nb_16(s: str) -> str:
    """Prefix minimum, then add 1.
    y[i] = min(x[0], ..., x[i]) + 1 (mod 10)."""
    return _shift_all(_prefix_min(s), 1)


def nb_17(s: str) -> str:
    """Prefix maximum, then add 1.
    y[i] = max(x[0], ..., x[i]) + 1 (mod 10)."""
    return _shift_all(_prefix_max(s), 1)


def nb_18(s: str) -> str:
    """Suffix minimum, then add 2.
    y[i] = min(x[i], ..., x[n-1]) + 2 (mod 10)."""
    return _shift_all(_suffix_min(s), 2)


def nb_19(s: str) -> str:
    """Suffix maximum, then add 2.
    y[i] = max(x[i], ..., x[n-1]) + 2 (mod 10)."""
    return _shift_all(_suffix_max(s), 2)


# --------------------------------------------------
# TIER 6: LOSSY STATE EMISSIONS
# --------------------------------------------------

def nb_20(s: str) -> str:
    """Prefix sum, then emit half-state with a small drift.
    state = (state + digit) mod 10
    y[i] = floor(state / 2) + 3 (mod 10)."""
    return _shift_all(_state_emit(s, lambda state, digit, i: state // 2), 3)


def nb_21(s: str) -> str:
    """Prefix sum, then emit folded state.
    y[i] = min(state, 9 - state) + 1 (mod 10)."""
    return _shift_all(_state_emit(s, lambda state, digit, i: min(state, 9 - state)), 1)


def nb_22(s: str) -> str:
    """Prefix sum, then emit state squared mod 10, then add 1.
    y[i] = (state^2 mod 10) + 1 (mod 10)."""
    return _shift_all(_state_emit(s, lambda state, digit, i: (state * state) % 10), 1)


def nb_23(s: str) -> str:
    """Prefix sum, then emit distance to 5.
    y[i] = |state - 5|."""
    return _state_emit(s, lambda state, digit, i: abs(state - 5))


ALL_BIJECTIVE = [
    func_00, func_01, func_02, func_03, func_04, func_05,
    func_06, func_07, func_08, func_09, func_10, func_11,
    func_12, func_13, func_14, func_15, func_16, func_17,
    func_18, func_19, func_20, func_21, func_22, func_23,
]


ALL_NONBIJECTIVE = [
    nb_00, nb_01, nb_02, nb_03, nb_04, nb_05,
    nb_06, nb_07, nb_08, nb_09, nb_10, nb_11,
    nb_12, nb_13, nb_14, nb_15, nb_16, nb_17,
    nb_18, nb_19, nb_20, nb_21, nb_22, nb_23,
]


# ==================================================
# Optional validation helpers
# ==================================================

def _all_digit_strings(n: int):
    for t in product("0123456789", repeat=n):
        yield "".join(t)


def fixed_point_count(func, n: int) -> int:
    return sum(1 for s in _all_digit_strings(n) if func(s) == s)


def is_bijective_on_length(func, n: int) -> bool:
    seen: set[str] = set()
    for s in _all_digit_strings(n):
        y = func(s)
        if y in seen:
            return False
        seen.add(y)
    return True


def has_collision_on_length(func, n: int) -> bool:
    seen: dict[str, str] = {}
    for s in _all_digit_strings(n):
        y = func(s)
        if y in seen and seen[y] != s:
            return True
        seen[y] = s
    return False


def is_involution_on_length(func, n: int) -> bool:
    return all(func(func(s)) == s for s in _all_digit_strings(n))


def quick_self_test() -> dict[str, object]:
    """
    Exhaustive on lengths 2..4:
    - every bijection is injective
    - no bijection is an involution
    - every non-bijection has a collision
    - every function has 0 fixed points

    This is the intended fast audit for development.
    """
    return {
        "bijective_on_2_4": {
            f"func_{i:02d}": {n: is_bijective_on_length(func, n) for n in range(2, 5)}
            for i, func in enumerate(ALL_BIJECTIVE)
        },
        "nonbijective_has_collision_on_2_4": {
            f"nb_{i:02d}": {n: has_collision_on_length(func, n) for n in range(2, 5)}
            for i, func in enumerate(ALL_NONBIJECTIVE)
        },
        "bijective_not_involution_on_2_4": {
            f"func_{i:02d}": {n: (not is_involution_on_length(func, n)) for n in range(2, 5)}
            for i, func in enumerate(ALL_BIJECTIVE)
        },
        "fixed_points_on_2_4": {
            **{f"func_{i:02d}": {n: fixed_point_count(func, n) for n in range(2, 5)}
               for i, func in enumerate(ALL_BIJECTIVE)},
            **{f"nb_{i:02d}": {n: fixed_point_count(func, n) for n in range(2, 5)}
               for i, func in enumerate(ALL_NONBIJECTIVE)},
        },
    }
