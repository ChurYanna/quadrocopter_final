import numpy as np
import spatialmath as sm

from .formation import Formation


class FourFormation(Formation):

    def cal_delta0(self, psi: float) -> np.ndarray:
        R = sm.SO3.Rz(psi)
        return np.array([
            # 中心点 (索引 0)
            R.o * 0 * self._delta,
            # 右前方点 (索引 1)
            (R.o + R.n * np.sqrt(3)) * 0.5 * self._delta,
            # 左前方点 (索引 2)
            (R.o - R.n * np.sqrt(3)) * 0.5 * self._delta,
            # 后方点 (索引 3)
            -R.o * 1.0 * self._delta
        ]).T