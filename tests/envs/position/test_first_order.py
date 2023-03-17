#  Copyright 2023 Jan-Hendrik Ewers
#  SPDX-License-Identifier: GPL-3.0-only
import numpy as np
import pytest


@pytest.mark.parametrize("pos", [(-10, -10, 2)], indirect=True)
def test_first_order_same_start_and_tgt(pos, state, firstorderploypositiondroneenv):
    """
    Expect the drone to instantly exit with term = True since the target and current
    position are the same.
    """
    obs, _, term, trunc, _ = firstorderploypositiondroneenv.step(pos)

    assert term
    assert not trunc
    assert len(obs) == 1
    assert np.allclose(state, obs[0])
