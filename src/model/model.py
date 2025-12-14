import abc

import numpy as np
from ..control_assignment import ControlAssignment


class Model(abc.ABC):

    def __init__(self) -> None:
        super().__init__()
        self._m = 0.0
        self._I = np.diag([0.0, 0.0, 0.0])
        self._g = 9.81
        self._control_assignment: ControlAssignment = ControlAssignment()

    @property
    def g(self):
        return self._g

    @property
    def m(self) -> float:
        return self._m

    @property
    def I(self) -> np.ndarray:
        return self._I

    @property
    def Ixx(self) -> float:
        return self._I[0, 0]

    @property
    def Iyy(self) -> float:
        return self._I[1, 1]

    @property
    def Izz(self) -> float:
        return self._I[2, 2]

    def assign(self, taus: np.ndarray) -> np.ndarray:
        return self._control_assignment.assign(taus)
