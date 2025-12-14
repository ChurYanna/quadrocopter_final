import numpy as np
import spatialmath as sm

from .formation import Formation


class TriangleFormation(Formation):

    def cal_delta0(self, psi: float) -> np.ndarray:
        R = sm.SO3.Rz(psi)
        return np.array([(- R.n * np.sqrt(3) + R.o * ((-1) ** (i + 1))) * self._delta * ((i + 1) // 2 / 2) for i in
                         range(self._count)]).T
