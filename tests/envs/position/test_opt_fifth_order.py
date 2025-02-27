#  Copyright 2023 Jan-Hendrik Ewers
#  SPDX-License-Identifier: GPL-3.0-only
import numpy as np
import pytest


@pytest.mark.parametrize("pos", [(-10, -10, 2)], indirect=True)
def test_opt_fifth_order_same_start_and_tgt(
    pos, state, optimalfifthorderpolypositiondroneenv
):
    """
    Expect the drone to instantly exit with term = True since the target and current
    position are the same.
    """
    obs, _, term, trunc, _ = optimalfifthorderpolypositiondroneenv.step(pos)

    assert not term
    assert trunc
    assert len(obs) == 1
    assert np.allclose(state, obs[0])
