import numpy as np

from .control_assignment import ControlAssignment


class XControlAssignment(ControlAssignment):
    def __init__(self, dx: float, dy: float, ct: float, cm: float) -> None:
        super().__init__()   #调用父类构造

        self._P = np.diag([ct, ct, ct, cm])  #力矩系数矩阵

        self._M0 = np.array([    #惯性矩阵（x型四旋翼结构）
            [1, 1, 1, 1],
            [-dx, dx, dx, -dx],
            [dy, dy, -dy, -dy],
            [1, -1, 1, -1]
        ])

        self._M0inv = np.linalg.inv(self._M0)   #计算m0逆矩阵
        self._Pinv = np.linalg.inv(self._P)    #计算p的逆矩阵
