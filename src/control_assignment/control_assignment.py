#将期望力矩转换为电机角速度
import abc

import numpy as np


class ControlAssignment(abc.ABC):

    def __init__(self) -> None:
        super().__init__()
        self._M0inv = np.ones(4)
        self._Pinv = np.ones(4)

    def assign(self, taus: np.ndarray) -> np.ndarray:
        omegas = self._M0inv @ self._Pinv @ taus
        return omegas
