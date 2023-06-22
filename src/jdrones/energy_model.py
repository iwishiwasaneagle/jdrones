#  Copyright 2023 Jan-Hendrik Ewers
#  SPDX-License-Identifier: GPL-3.0-only
import abc
from typing import Any

import numpy as np
import numpy.typing as npt
from jdrones.data_models import URDFModel
from jdrones.envs.dronemodels import DronePlus


class BaseEnergyModel(abc.ABC):
    """
    Base composite for energy and power implementations
    """

    dt: float

    model: URDFModel
    """Model parameters"""

    def __init__(self, dt, model=DronePlus, *, _state=None):
        self.dt = dt
        self.model = model

    @abc.abstractmethod
    def power(self, state: dict[str, Any] | npt.ArrayLike):
        """
        Calculate the power usage of the system at the state

        Parameters
        ----------
        state: Any

        Returns
        -------
        list | float
            The calculated power consumption
        """
        pass

    def energy(self, state: dict[str, Any] | npt.ArrayLike):
        """
        :math:`E=P\\Delta t`

        Parameters
        ----------
        state: Any
            The state used to calculate the power

        Returns
        -------
        list | float
            The calculated energy
        """

        return self.power(state) * self.dt


class StaticPropellerVariableVelocityEnergyModel(BaseEnergyModel):
    """
    Implement an energy model from [1,2,3] that is based on variable body velocity
    magnitude. The assumption is that the UAV's propellers aren't changing in
    RPM and is therefore an approximation as this clearly isn't the case in the
    real world.

    [1] D. Ebrahimi, S. Sharafeddine, P.-H. Ho, and C. Assi, ‘Autonomous UAV Trajectory
        for Localizing Ground Objects: A Reinforcement Learning Approach’, IEEE
        Transactions on Mobile Computing, vol. 20, no. 4, pp. 1312–1324, Apr. 2021,
        doi: 10.1109/TMC.2020.2966989.
    [2] A. Filippone, Flight performance of fixed and rotary wing aircraft, Ist ed. in
        Elsevier aerospace engineering series. Amsterdam;
        Boston: Butterworth-Heinemann, 2006.
    [3] H. Sallouha, M. M. Azari, and S. Pollin, ‘Energy-Constrained UAV Trajectory
        Design for Ground Node Localization’, in 2018 IEEE Global Communications
        Conference (GLOBECOM), Dec. 2018, pp. 1–7. doi: 10.1109/GLOCOM.2018.8647530.
    """

    v_b: float
    """Propeller RPM"""
    K: float
    """Constant related to drag induced by the propeller"""
    F: float
    """Constant related to the parasitic dragged"""
    A: float
    """Area of the drone"""

    def __init__(
        self,
        dt: float,
        model: URDFModel,
        *,
        v_b: float = 9000,
        K: float = 1,
        F: float = 0.1,
        A: float = 1,
    ):
        super().__init__(dt, model)
        self.v_b = v_b
        self.K = K
        self.F = F
        self.A = A
        self.m = self.model.mass
        self.g = self.model.g
        self.rho = self.model.rho

    def p_blade(self, v: npt.ArrayLike):
        """
        Power required to turn the blades

        .. math::
            P_{blade} = K(1+3\\frac{v^2}{v_b^2})

        Parameters
        ----------
        v : float | numpy.ndarray
            Body velocity

        Returns
        -------
        float | numpy.ndarray
        """
        return self.K * (1 + 3 * np.square(v) / np.square(self.v_b))

    def p_parasite(self, v: npt.ArrayLike):
        """
        Power used to overcome the drag force

        .. math::
            P_{parasite} = \\frac{1}{2}\\rhov^3F

        Parameters
        ----------
        v : float | numpy.ndarray
            Body velocity

        Returns
        -------
        float | numpy.ndarray
        """
        return 0.5 * self.rho * np.power(v, 3) * self.F

    def v_i(self, v: npt.ArrayLike):
        """
        The mean propellers’ induced velocity in the forward flight

        .. math::
           v_i = \\sqrt{
                \\frac{
                -v^2 + \\sqrt{v^4+(\\frac{mg}{\rho A})^2}
                }{2}
                }

        Parameters
        ----------
        v : float | numpy.ndarray
            Body velocity

        Returns
        -------
        float | numpy.ndarray
        """
        a = -np.square(v)
        b = np.power(v, 4) + np.square(self.m * self.g / (self.rho * self.A))

        return np.sqrt((a + np.sqrt(b)) / 2)

    def p_induced(self, v: npt.ArrayLike):
        """
        Power required to lift the UAV and overcome the drag caused by gravity

        .. math::
            P_{parasite} = \\frac{1}{2}\\rhov^3F

        Parameters
        ----------
        v : float | numpy.ndarray
            Body velocity

        Returns
        -------
        float | numpy.ndarray
        """

        return self.m * self.g * self.v_i(v)

    def power(self, v: npt.ArrayLike):
        """
        Total power required at the system velocity

        .. math::

            P_{total} = P_{blade} + P_{parasite} + P_{induced}


        Parameters
        ----------
        v: float | numpy.ndarray

        Returns
        -------
        list | float
            The calculated total power consumption
        """
        return self.p_blade(v) + self.p_parasite(v) + self.p_induced(v)
