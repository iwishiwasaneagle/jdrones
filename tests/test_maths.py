import numpy as np
from hypothesis import strategies as st, given

from jdrones.maths import clip_scalar, clip


@given(
    value=st.floats(allow_nan=False),
    vmin=st.floats(max_value=0,allow_nan=False),
    vmax=st.floats(min_value=1,allow_nan=False)
)

def test_clip_scalar(value,vmin,vmax):
    clipped = clip_scalar(value,vmin,vmax)
    assert vmin<=clipped<=vmax

@given(
    value=st.lists(st.floats(allow_nan=False),min_size=1),
    vmin=st.floats(max_value=0,allow_nan=False),
    vmax=st.floats(min_value=1,allow_nan=False)
)
def test_clip_array(value,vmin,vmax):
    clipped = clip(value,vmin,vmax)
    assert np.all((vmin<=np.asarray(clipped))<=vmax)