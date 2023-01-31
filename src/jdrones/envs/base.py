import abc
from copy import copy
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import gymnasium
import numpy as np
import pybullet as p
import pybullet_data
import pydantic
from gymnasium.core import ActType
from gymnasium.core import ObsType
from gymnasium.core import RenderFrame
from jdrones.envs.dronemodels import DronePlus
from jdrones.maths import clip
from jdrones.transforms import euler_to_quat
from jdrones.transforms import quat_to_euler
from jdrones.types import Action
from jdrones.types import Observation
from jdrones.types import PropellerAction
from jdrones.types import SimulationType
from jdrones.types import State
from jdrones.types import URDFModel
from jdrones.types import VEC3
from jdrones.types import VEC4


class PyBulletIds(pydantic.BaseModel):
    client: int = None
    plane: int = None
    drone: int = None


class BaseDroneEnv(gymnasium.Env, abc.ABC):
    state: State
    initial_state: State

    model: URDFModel

    ids: PyBulletIds

    def __init__(
        self,
        model: URDFModel = DronePlus,
        initial_state: State = None,
        simulation_type: SimulationType = SimulationType.DIRECT,
        dt: float = 1 / 240,
    ):
        if initial_state is None:
            initial_state = State()
        self.initial_state = initial_state
        self.state = copy(self.initial_state)
        self.model = model
        self.dt = dt
        self.ids = PyBulletIds()
        self._init_simulation(simulation_type)

    @property
    @abc.abstractmethod
    def observation_space(self) -> ObsType:
        pass

    @property
    @abc.abstractmethod
    def action_space(self) -> ActType:
        pass

    def _init_simulation(self, simulation_type: SimulationType):
        self.ids.client = p.connect(simulation_type)
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
    ) -> Tuple[Observation, dict]:
        super().reset(seed=seed, options=options)
        # Reset state
        self.state = copy(self.initial_state)
        self.state.quat = euler_to_quat(self.state.rpy)
        # Reset drone
        p.resetBasePositionAndOrientation(
            self.ids.drone,
            self.state.pos,
            self.state.quat,
            physicsClientId=self.ids.client,
        )
        return self.get_observation(), self.get_info()

    def close(self):
        p.disconnect(physicsClientId=self.ids.client)

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        pass

    def step(self, action: Action) -> Tuple[Observation, float, bool, bool, dict]:
        propeller_action = clip(action, 0, np.inf)
        propulsive_forces = self.calculate_propulsive_forces(propeller_action)
        aerodynamic_forces = self.calculate_aerodynamic_forces(propeller_action)
        external_torques = self.calculate_external_torques(propeller_action)

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
            4,
            torqueObj=external_torques,
            flags=p.LINK_FRAME,
            physicsClientId=self.ids.client,
        )
        p.applyExternalForce(
            self.ids.drone,
            4,
            forceObj=aerodynamic_forces,
            posObj=[0, 0, 0],
            flags=p.LINK_FRAME,
            physicsClientId=self.ids.client,
        )

        # Cartesian world coordinates
        self.state = self.get_kinematic_data(self.ids)
        self.state.prop_omega = action

        p.stepSimulation(physicsClientId=self.ids.client)

        return (
            self.get_observation(),
            self.get_reward(),
            self.get_terminated(),
            self.get_truncated(),
            self.get_info(),
        )

    @staticmethod
    def get_kinematic_data(ids: PyBulletIds) -> State:
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

    @abc.abstractmethod
    def calculate_propulsive_forces(self, action: PropellerAction) -> VEC4:
        pass

    @abc.abstractmethod
    def calculate_aerodynamic_forces(self, action: PropellerAction) -> VEC3:
        pass

    @abc.abstractmethod
    def calculate_external_torques(self, action: PropellerAction) -> VEC3:
        pass

    @abc.abstractmethod
    def get_observation(self, *args, **kwargs) -> Observation:
        pass

    @abc.abstractmethod
    def get_reward(self, *args, **kwargs) -> float:
        pass

    @abc.abstractmethod
    def get_terminated(self, *args, **kwargs) -> bool:
        pass

    @abc.abstractmethod
    def get_truncated(self, *args, **kwargs) -> bool:
        pass

    @abc.abstractmethod
    def get_info(self, *args, **kwargs) -> dict:
        pass

    @property
    def on_ground_plane(self) -> bool:
        p.performCollisionDetection(physicsClientId=self.ids.client)
        contact_pts = p.getContactPoints(self.ids.drone, self.ids.plane)
        if len(contact_pts) > 0:
            return True
        return False
