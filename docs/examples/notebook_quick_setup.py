#  Copyright 2023 Jan-Hendrik Ewers
#  SPDX-License-Identifier: GPL-3.0-only
# flake8: noqa

print("Beginning notebook setup...")
import sys
import pathlib

PATH = pathlib.Path("../../src/").resolve()
if str(PATH) not in sys.path:
    sys.path.insert(0, str(PATH))
    print(f"\tAdded {PATH} to path")

import gymnasium

print(f"\tImported gymnasium version {gymnasium.__version__}")
import jdrones
from jdrones.data_models import *
import jdrones.plotting as jplot

print(f"\tImported jdrones version {jdrones.__version__}")

import pandas as pd
import numpy as np
import scipy
import scipy.linalg
import scipy.signal

print(
    f"\tImported scipy=={scipy.__version__}, numpy=={np.__version__}, pandas=={pd.__version__}"
)


import functools
import collections
import itertools

print("\tImported functools, collections and itertools")

from tqdm.auto import tqdm, trange

print("\tImported tqdm (standard and trange)")

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

print(f"\tImported seaborn=={sns.__version__}, matplotlib=={matplotlib.__version__}")


print("End of notebook setup")
