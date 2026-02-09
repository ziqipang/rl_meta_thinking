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