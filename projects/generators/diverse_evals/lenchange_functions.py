"""
Novel non-bijective digit-string functions that *change length*.

These are intended as an eval-only generalization set: prompts should include
the function source code (stage-1 style).

All functions are many-to-one because they only depend on digits modulo 5,
so each digit has a {d, d+5} ambiguity.
"""

from __future__ import annotations


def lenop_00(s: str) -> str:
    """Reduce length by 1: non-cyclic neighbor-sum mod 5.

    For input digits x[0..n-1], output is:
      y[i] = (x[i] + x[i+1]) % 5  for i=0..n-2
    """
    n = len(s)
    if n <= 1:
        return ""
    out: list[str] = []
    for i in range(n - 1):
        a = ord(s[i]) - ord("0")
        b = ord(s[i + 1]) - ord("0")
        out.append(str((a + b) % 5))
    return "".join(out)


def lenop_01(s: str) -> str:
    """Reduce length by 1: non-cyclic 'signed' neighbor mix mod 5.

      y[i] = (x[i] - 2*x[i+1]) % 5  for i=0..n-2
    """
    n = len(s)
    if n <= 1:
        return ""
    out: list[str] = []
    for i in range(n - 1):
        a = ord(s[i]) - ord("0")
        b = ord(s[i + 1]) - ord("0")
        out.append(str((a - 2 * b) % 5))
    return "".join(out)


def lenop_02(s: str) -> str:
    """Reduce length by 1: first+last mod 5 becomes new first; drop last.

    Example structure (requested): mix first/last, then remove last digit.
      y[0]   = (x[0] + x[n-1]) % 5
      y[1:]  = x[1:n-1] % 5   (per-digit mod 5 view)
    """
    n = len(s)
    if n <= 1:
        return ""
    x0 = ord(s[0]) - ord("0")
    xl = ord(s[-1]) - ord("0")
    out = [str((x0 + xl) % 5)]
    for ch in s[1:-1]:
        out.append(str((ord(ch) - ord("0")) % 5))
    return "".join(out)


def lenop_03(s: str) -> str:
    """Increase length by 1: cyclic neighbor-sum mod 5 + checksum mod 5.

    y has length n+1:
      y[i] = (x[i] + x[i+1]) % 5 with wrap-around (i=0..n-1)
      y[n] = (sum(x) % 5)
    """
    n = len(s)
    if n == 0:
        return "0"
    digits = [ord(ch) - ord("0") for ch in s]
    out: list[str] = []
    for i in range(n):
        out.append(str((digits[i] + digits[(i + 1) % n]) % 5))
    out.append(str(sum(digits) % 5))
    return "".join(out)


def lenop_04(s: str) -> str:
    """Increase length by 1: mod-5 affine state machine + final state digit.

    state = (3*state + x[i]) % 5
    y[i]  = state
    y[n]  = (state + x[0]) % 5
    """
    n = len(s)
    if n == 0:
        return "0"
    st = 0
    out: list[str] = []
    x0 = ord(s[0]) - ord("0")
    for ch in s:
        st = (3 * st + (ord(ch) - ord("0"))) % 5
        out.append(str(st))
    out.append(str((st + x0) % 5))
    return "".join(out)


def lenop_05(s: str) -> str:
    """Increase length by 1: mod-5 prefix sum (non-cyclic) + start/end check.

    prefix[i] = (x[0] + ... + x[i]) % 5
    y         = prefix[0..n-1] + ((x[0] + x[n-1]) % 5)
    """
    n = len(s)
    if n == 0:
        return "0"
    digits = [ord(ch) - ord("0") for ch in s]
    out: list[str] = []
    st = 0
    for d in digits:
        st = (st + d) % 5
        out.append(str(st))
    out.append(str((digits[0] + digits[-1]) % 5))
    return "".join(out)


LENCHANGE_FUNCTIONS: dict[str, object] = {
    "lenop_00": lenop_00,
    "lenop_01": lenop_01,
    "lenop_02": lenop_02,
    "lenop_03": lenop_03,
    "lenop_04": lenop_04,
    "lenop_05": lenop_05,
}

