#  Copyright 2023 Jan-Hendrik Ewers
#  SPDX-License-Identifier: GPL-3.0-only
import contextlib
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import pybullet_data
from gymnasium.core import ActType
from gymnasium.core import ObsType
from gymnasium.core import RenderFrame
from jdrones.data_models import PyBulletIds
from jdrones.data_models import SimulationType
from jdrones.data_models import State
from jdrones.data_models import URDFModel
from jdrones.envs.base.basedronenev import BaseDroneEnv
from jdrones.envs.dronemodels import DronePlus
from jdrones.transforms import euler_to_quat
from jdrones.transforms import quat_to_euler
from jdrones.transforms import quat_to_rotmat
from jdrones.types import PropellerAction
from jdrones.types import VEC3
from jdrones.types import VEC4

with contextlib.redirect_stdout(None):
    import pybullet as p


class PyBulletDroneEnv(BaseDroneEnv):
    """
    Base drone environment. Handles pybullet loading, and application of forces.
    Generalizes the physics to allow other models to be used.

    >>> import jdrones
    >>> import gymnasium
    >>> gymnasium.make("PyBulletDroneEnv-v0")
    <OrderEnforcing<PassiveEnvChecker<PyBulletDroneEnv<PyBulletDroneEnv-v0>>>>
    """

    ids: PyBulletIds
    """PB IDs"""

    model: URDFModel
    "URDFModel description"

    simulation_type: SimulationType
    """Simulation type to run"""

    def __init__(
        self,
        model: URDFModel = DronePlus,
        initial_state: State = None,
        simulation_type: SimulationType = SimulationType.DIRECT,
        dt: float = 1 / 240,
    ):
        super().__init__(initial_state, dt)
        self.model = model
        self.ids = PyBulletIds()
        self.simulation_type = simulation_type
        self.state = State(np.copy(self.initial_state))
        self._init_simulation()

    def _init_simulation(self):
        """
        Initialise the simulation. Only ran once when the environment is instantiated.

        ..warning::
            Do not call this to reset the environment.

        """
        self.ids.client = p.connect(self.simulation_type)
        # PyBullet parameters
        p.setGravity(0, 0, -self.model.g, physicsClientId=self.ids.client)
        p.setTimeStep(self.dt, physicsClientId=self.ids.client)
        p.setAdditionalSearchPath(
            pybullet_data.getDataPath(), physicsClientId=self.ids.client
        )
        p.setRealTimeSimulation(0, physicsClientId=self.ids.client)
        # Load ground plane
        self.ids.plane = p.loadURDF("plane.urdf", physicsClientId=self.ids.client)

        self.state.quat = euler_to_quat(self.state.rpy)
        self.ids.drone = p.loadURDF(
            self.model.filepath,
            basePosition=self.state.pos,
            baseOrientation=self.state.quat,
            flags=p.URDF_USE_INERTIA_FROM_FILE,
            physicsClientId=self.ids.client,
        )
        p.resetBaseVelocity(
            self.ids.drone,
            linearVelocity=self.state.vel,
            angularVelocity=self.state.ang_vel,
            physicsClientId=self.ids.client,
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[State, dict]:
        """
        Reset the simulation to the initial state.

        .. seealso::
            :func:`gymnasium.Env.reset`

        Parameters
        ----------
        seed : int
            Seed to pass to gymnasium RNG
            (Default = None)
        options : dict
            Additional options to pass to gymnasium API
            (Default = None)

        Returns
        -------
        observation : Observation
            Observation of the initial state. It should be analogous to the info
            returned by :meth:`step`.
        info : dict
            This dictionary contains auxiliary information complementing observation.
            It should be analogous to the info returned by :meth:`step`.
        """
        x = super().reset(seed=seed, options=options)
        # Reset drone
        p.resetBasePositionAndOrientation(
            self.ids.drone,
            self.state.pos,
            self.state.quat,
            physicsClientId=self.ids.client,
        )
        p.resetBaseVelocity(
            self.ids.drone,
            linearVelocity=self.state.vel,
            angularVelocity=self.state.ang_vel,
            physicsClientId=self.ids.client,
        )
        return x

    def close(self):
        p.disconnect(physicsClientId=self.ids.client)

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        pass

    def step(self, action: PropellerAction) -> Tuple[State, float, bool, bool, dict]:
        """
        Run one timestep of the environment’s dynamics using the agent actions

        .. seealso::
            - :func:`gymnasium.Env.step`

        .. note::
            The physics are implemented in the following functions:

            - :meth:`jdrones.envs.PyBulletDroneEnv.calculate_propulsive_forces`
            - :meth:`jdrones.envs.PyBulletDroneEnv.calculate_aerodynamic_forces`
            - :meth:`jdrones.envs.PyBulletDroneEnv.calculate_external_torques`

        Parameters
        ----------
        action : float,float,float,float
            An action provided by the agent to update the environment state

        Returns
        -------
        observation : Observation
            Observation of the state
        reward : float
             The reward as a result of taking the action
        terminated : bool
             Whether the agent reaches the terminal state (as defined under the MDP of
             the task)
        truncated : bool
             Whether the truncation condition outside the scope of the MDP is satisfied
        info : dict
            Contains auxiliary diagnostic information (helpful for debugging, learning,
            and logging)
        """
        propulsive_forces = self.calculate_propulsive_forces(action)
        aerodynamic_forces = self.calculate_aerodynamic_forces(action)
        external_torques = self.calculate_external_torques(action)

        for i in range(4):
            p.applyExternalForce(
                self.ids.drone,
                i,
                forceObj=[0, 0, propulsive_forces[i]],
                posObj=[0, 0, 0],
                flags=p.LINK_FRAME,
                physicsClientId=self.ids.client,
            )
        p.applyExternalTorque(
            self.ids.drone,
            -1,
            torqueObj=external_torques,
            flags=p.LINK_FRAME,
            physicsClientId=self.ids.client,
        )
        p.applyExternalForce(
            self.ids.drone,
            -1,
            forceObj=aerodynamic_forces,
            posObj=[0, 0, 0],
            flags=p.LINK_FRAME,
            physicsClientId=self.ids.client,
        )

        p.stepSimulation(physicsClientId=self.ids.client)

        # Cartesian world coordinates
        self.state = self.get_kinematic_data(self.ids)
        self.state.prop_omega = action

        return (
            self.get_observation(),
            self.get_reward(),
            self.get_terminated(),
            self.get_truncated(),
            self.info,
        )

    @staticmethod
    def get_kinematic_data(ids: PyBulletIds) -> State:
        """
        Get the drones's :class:`~jdrones.data_models.State` from pybullet post
        :meth:`step`

        Parameters
        ----------
        ids : PyBulletIds
            PB IDs to read from

        Returns
        -------
        jdrones.data_models.State
            The current state of the drone
        """
        state = State()
        # Cartesian world coordinates
        state.pos, state.quat = p.getBasePositionAndOrientation(
            ids.drone, physicsClientId=ids.client
        )
        new_rpy = quat_to_euler(state.quat)
        state.rpy = new_rpy
        # Cartesian worldspace coordinates
        state.vel, state.ang_vel = p.getBaseVelocity(
            ids.drone, physicsClientId=ids.client
        )
        return state

    @property
    def on_ground_plane(self) -> bool:
        p.performCollisionDetection(physicsClientId=self.ids.client)
        contact_pts = p.getContactPoints(self.ids.drone, self.ids.plane)
        if len(contact_pts) > 0:
            return True
        return False

    def calculate_aerodynamic_forces(self, action: ActType) -> VEC3:
        rotmat_i_b = quat_to_rotmat(self.state.quat)
        drag_factors = np.array(self.model.drag_coeffs)
        drag_force = -drag_factors * np.dot(rotmat_i_b, np.square(self.state.vel))
        return drag_force

    def calculate_external_torques(self, action: ActType) -> VEC3:
        Qi = np.square(action) * self.model.k_Q
        return (0, 0, Qi[0] - Qi[1] + Qi[2] - Qi[3])

    def calculate_propulsive_forces(self, action: VEC4) -> VEC4:
        return np.square(action) * self.model.k_T

    def get_observation(self) -> ObsType:
        return self.state

    def get_reward(self) -> float:
        return 0

    def get_terminated(self) -> bool:
        term = self.on_ground_plane
        if term:
            self.info["collision"] = f"On ground plane at {self.state.pos}"
        return term

    def get_truncated(self) -> bool:
        return False

    def get_info(self) -> dict:
        return {}
