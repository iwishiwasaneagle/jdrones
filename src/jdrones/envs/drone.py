import numpy as np
from gymnasium.core import ActType
from gymnasium.core import ObsType
from gymnasium.vector.utils import spaces
from jdrones.envs.base import BaseDroneEnv
from jdrones.transforms import quat_to_rotmat
from jdrones.types import VEC3
from jdrones.types import VEC4


class DroneEnv(BaseDroneEnv):
    def calculate_aerodynamic_forces(self, action: ActType) -> VEC3:
        rotmat_i_b = quat_to_rotmat(self.state.quat)
        drag_factors = -1 * np.array(self.model.drag_coeffs)
        return np.dot(rotmat_i_b, drag_factors * self.state.vel)

    def calculate_external_torques(self, action: ActType) -> VEC3:
        Qi = action * self.model.k_Q
        return (0, 0, -Qi[0] + Qi[1] - Qi[2] + Qi[3])

    def calculate_propulsive_forces(self, action: VEC4) -> VEC4:
        return action * self.model.k_T

    @property
    def action_space(self):
        act_bounds = np.array(
            [
                (0.0, 1e6),  # R1
                (0.0, 1e6),  # R2
                (0.0, 1e6),  # R3
                (0.0, 1e6),  # R4
            ]
        )
        return spaces.Box(
            low=act_bounds[:, 0],
            high=act_bounds[:, 1],
            dtype=float,
        )

    @property
    def observation_space(self):
        obs_bounds = np.array(
            [
                # XYZ
                # Position
                (-np.inf, np.inf),
                (-np.inf, np.inf),
                (0.0, np.inf),
                # Q 1-4
                # Quarternion rotation
                (-1.0, 1.0),
                (-1.0, 1.0),
                (-1.0, 1.0),
                (-1.0, 1.0),
                # RPY
                # Roll pitch yaw rotation
                (-np.pi, np.pi),
                (-np.pi, np.pi),
                (-np.pi, np.pi),
                # V XYZ
                # Velocity
                (-np.inf, np.inf),
                (-np.inf, np.inf),
                (-np.inf, np.inf),
                # V RPY
                # Angular velocity
                (-np.inf, np.inf),
                (-np.inf, np.inf),
                (-np.inf, np.inf),
                # P 0-4
                # Propeller speed
                (0.0, np.inf),
                (0.0, np.inf),
                (0.0, np.inf),
                (0.0, np.inf),
            ]
        )
        return spaces.Box(low=obs_bounds[:, 0], high=obs_bounds[:, 1], dtype=float)

    def get_observation(self) -> ObsType:
        return self.state

    def get_reward(self) -> float:
        return 0

    def get_terminated(self) -> bool:
        return self.on_ground_plane

    def get_truncated(self) -> bool:
        return False

    def get_info(self) -> dict:
        return {}
