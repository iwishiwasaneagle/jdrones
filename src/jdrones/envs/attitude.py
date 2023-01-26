from typing import Dict
from typing import Optional
from typing import Tuple

import numpy as np
from gymnasium.core import ActType
from gymnasium.vector.utils import spaces
from jdrones.controllers import PID
from jdrones.controllers import PID_antiwindup
from jdrones.envs.drone import DroneEnv
from jdrones.maths import clip
from jdrones.maths import clip_scalar
from jdrones.types import Action
from jdrones.types import AttitudeAltitudeAction
from jdrones.types import Observation


class AttitudeAltitudeDroneEnv(DroneEnv):
    controllers: Dict[str, PID]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.controllers = self._init_controllers()

    def _init_controllers(self) -> Dict[str, PID]:
        y = PID_antiwindup(10, 0.1, 0.2, self.dt, angle=True)
        dy = PID_antiwindup(15, 0.01, 7, self.dt, gain=-10000)
        p = PID_antiwindup(6, 0.1, 0.3, self.dt, angle=True, gain=-1000)
        r = PID_antiwindup(6, 0.1, 0.3, self.dt, angle=True, gain=1000)

        z = PID_antiwindup(15, 3, 0.1, self.dt, gain=0.1)
        dz = PID_antiwindup(2, 0, 0.00001, self.dt, gain=100000)

        return dict(yaw=y, pitch=p, roll=r, daltitude=dz, dyaw=dy, altitude=z)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[Observation, dict]:
        for controller in self.controllers.values():
            controller.reset()
        return super().reset(seed=seed, options=options)

    def step(self, action: Action) -> Tuple[Observation, float, bool, bool, dict]:
        action = AttitudeAltitudeAction(action)
        # Calculate control action

        r_act = self.controllers["roll"](self.state.rpy[0], action.roll)
        p_act = self.controllers["pitch"](self.state.rpy[1], action.pitch)

        y_act = self.controllers["yaw"](self.state.rpy[2], action.yaw)
        y_act = clip_scalar(y_act, -2, 2)
        dy_act = self.controllers["dyaw"](self.state.ang_vel[2], y_act)

        z_act = self.controllers["altitude"](self.state.pos[2], action.z)
        z_act = clip_scalar(z_act, -10, 10)
        dz_act = self.controllers["daltitude"](self.state.vel[2], z_act)

        propellerAction = clip(
            self.model.rpyT2rpm(r_act, p_act, dy_act, dz_act),
            0,
            3 * self.model.weight / self.model.k_T,
        )

        return super().step(propellerAction)

    @property
    def action_space(self) -> ActType:
        act_bounds = np.array(
            [
                (-0.1, 0.1),  # Roll
                (-0.1, 0.1),  # Pitch
                (-np.pi, np.pi),  # Yaw
                (1, np.inf),  # Altitude
            ]
        )
        return spaces.Box(
            low=act_bounds[:, 0],
            high=act_bounds[:, 1],
            dtype=float,
        )

    def get_info(self) -> dict:
        info = super().get_info()
        info["control"] = dict(
            errors={name: ctrl.e for name, ctrl in self.controllers.items()}
        )
        return info


if __name__ == "__main__":
    import copy
    from collections import deque
    from itertools import count

    from gymnasium.wrappers import TimeLimit
    import sys

    from loguru import logger
    from tqdm.auto import tqdm

    from jdrones.envs.dronemodels import DronePlus
    from jdrones.types import State, SimulationType

    sys.setrecursionlimit(100000)

    T = 100
    dt = 1 / 500
    model = DronePlus
    logger.debug(model)

    initial_state = State()
    initial_state.pos = [0, 0, 10]
    initial_state.rpy = [0, 0, 0]

    env = AttitudeAltitudeDroneEnv(
        model=model,
        initial_state=initial_state,
        simulation_type=SimulationType.DIRECT,
        dt=dt,
    )
    env = TimeLimit(env, max_episode_steps=int(T / dt))

    env.reset()

    trunc, term = (False,) * 2
    c = count()

    pbar = tqdm(range(int(T / dt)), desc="Running simulation")
    observations = deque()
    controller_errors = deque()

    attalt_setpoint = None
    t = 0
    info = {}
    while not (trunc or term):
        t = next(c) * dt

        if t % 10 == 0:
            attalt_setpoint = env.action_space.sample()
        obs, _, term, trunc, info = env.step(attalt_setpoint)

        pbar.update(1)
        controller_errors.append(info["control"]["errors"])
        observations.append(copy.copy(obs))

    if trunc or term:
        print(info)
