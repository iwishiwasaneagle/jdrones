#  Copyright (c) 2024.  Jan-Hendrik Ewers
#  SPDX-License-Identifier: GPL-3.0-only
from jdrones.data_models import State as _State


class State(_State):
    k: int = 32

    @property
    def target(self):
        return self[20:23]

    @target.setter
    def target(self, val):
        self[20:23] = val

    @property
    def next_target(self):
        return self[29:32]

    @next_target.setter
    def next_target(self, val):
        self[29:32] = val

    @property
    def target_error(self):
        return self[23:26]

    @target_error.setter
    def target_error(self, val):
        self[23:26] = val

    @property
    def target_error_integral(self):
        return self[26:29]

    @target_error_integral.setter
    def target_error_integral(self, val):
        self[26:29] = val
