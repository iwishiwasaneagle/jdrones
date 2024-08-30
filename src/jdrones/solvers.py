#  Copyright (c) 2024.  Jan-Hendrik Ewers
#  SPDX-License-Identifier: GPL-3.0-only
import numpy as np


def bisection(f, a, b, tol=1e-3, N: int = 100):
    fa = f(a)
    fb = f(b)

    if np.sign(fa) == np.sign(fb):
        raise RuntimeError(
            f"{f(a)=} and {f(b)=} have the same sign. The interval [{a},{b}] "
            f"does not contain a root."
        )

    for _ in range(N):
        c = (a + b) / 2
        fc = f(c)

        if np.isnan(fc):
            return None

        if np.abs(fc) < tol:
            break

        if np.sign(fa) == np.sign(fc):
            a = c
            fa = fc
        else:
            b = c

    return c


def bisection_with_right_expansion(f, a, b, tol=1e-3, N: int = 100):
    fa = f(a)
    fb = f(b)
    while True:
        if np.sign(fa) != np.sign(fb):
            break
        a = b
        fa = fb
        b = 2 * b
        fb = f(b)

    return bisection(f, a, b, tol, N)
