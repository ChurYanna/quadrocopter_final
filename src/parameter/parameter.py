import numpy as np
import spatialmath as sm


def jacobian(q: np.ndarray) -> np.ndarray:
    # 计算雅可比矩阵，将关节空间速度映射到操作空间速度
    return np.array([
        [1.0, np.sin(q[0]) * np.tan(q[1]), np.cos(q[0]) * np.tan(q[1])],
        [0.0, np.cos(q[0]), -np.sin(q[0])],
        [0.0, np.sin(q[0]) / np.cos(q[1]), np.cos(q[0]) / np.cos(q[1])]
    ])


class Parameter:
    def __init__(self) -> None:
        super().__init__()
        self._posd = np.zeros(3)
        self._dposd = np.zeros(3)
        self._psid = 0.0
        self._dpsid = 0.0

        self._pos = np.zeros(3)
        self._dpos = np.zeros(3)
        self._ori = np.zeros(3)
        self._omega = np.zeros(3)

        self._phid = 0.0
        self._thetad = 0.0

    @property
    def qd(self) -> np.ndarray:
        return np.hstack((self._posd, self._phid, self._thetad, self._psid))

    @qd.setter
    def qd(self, qd: np.ndarray):
        self._posd = qd[:3].copy()
        self._psid = qd[3].copy()

    @property
    def dqd(self) -> np.ndarray:
        return np.hstack((self._dposd, self._dpsid))

    @dqd.setter
    def dqd(self, dqd: np.ndarray):
        self._dposd = dqd[:3].copy()
        self._dpsid = dqd[3].copy()

    @property
    def q(self) -> np.ndarray:
        return np.hstack((self._pos, self._ori))

    @q.setter
    def q(self, q: np.ndarray):
        self._pos = q[:3].copy()
        self._ori = sm.UnitQuaternion(q[3:]).rpy()

    @property
    def dq(self) -> np.ndarray:
        return np.hstack((self._dpos, self._omega))

    @dq.setter
    def dq(self, dq: np.ndarray):
        self._dpos = dq[:3].copy()
        self._omega = dq[3:].copy()

    @property
    def posd(self) -> np.ndarray:
        return self._posd.copy()

    @property
    def dposd(self) -> np.ndarray:
        return self._dposd.copy()

    @dposd.setter
    def dposd(self, dposd: np.ndarray):
        self._dposd = dposd.copy()

    @property
    def orid(self) -> np.ndarray:
        return np.array([self._phid, self._thetad, self._psid])

    @property
    def psid(self) -> float:
        return self._dpsid

    @property
    def pos(self) -> np.ndarray:
        return self._pos.copy()

    @property
    def dpos(self) -> np.ndarray:
        return self._dpos.copy()

    @property
    def ori(self) -> np.ndarray:
        return self._ori.copy()

    @property
    def phid(self) -> float:
        return self._phid

    @phid.setter
    def phid(self, phid: float):
        self._phid = phid

    @property
    def thetad(self) -> float:
        return self._thetad

    @thetad.setter
    def thetad(self, thetad: float):
        self._thetad = thetad

    @property
    def psid(self) -> float:
        return self._psid

    @psid.setter
    def psid(self, psid: float):
        self._psid = psid

    @property
    def psi(self) -> float:
        return self._ori[2]

    @psi.setter
    def psi(self, psi: float):
        self._ori[2] = psi

    @property
    def omega(self) -> np.ndarray:
        return self._omega.copy()
