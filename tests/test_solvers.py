#  Copyright (c) 2024.  Jan-Hendrik Ewers
#  SPDX-License-Identifier: GPL-3.0-only
import numpy as np
from jdrones.solvers import bisection
from jdrones.solvers import bisection_with_right_expansion


def test_bisection_root_of_poly():
    a = 1
    b = 2

    c = bisection(lambda x: x**3 - x - 2, a, b, 1e-6)

    assert np.isclose(c, 1.52137970)


def test_bisection_with_right_expansion_root_of_poly():
    a = 0
    b = 1

    c = bisection_with_right_expansion(lambda x: x**3 - x - 2, a, b, 1e-6)

    assert np.isclose(c, 1.52137970)
