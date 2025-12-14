import numpy as np
import spatialmath as sm

from .formation import Formation


class LineFormation(Formation):

    def cal_delta0(self, psi: float) -> np.ndarray:
        R = sm.SO3.Rz(psi)   #绕Z轴旋转psi角度
        return np.array([R.o * self._delta * (((-1) ** (i + 1)) * ((i + 1) // 2)) for i in range(self._count)]).T
