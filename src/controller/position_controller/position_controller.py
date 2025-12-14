import numpy as np

from ..pid_controller import PIDController
from src.model import Model


class PositionController:
    def __init__(self, kps: list, kis: list, kds: list, model: Model, ts=0.001, filter_coefficient=100.0) -> None:
        super().__init__()
        self._ts = ts
        self._dof = 3
        self._model = model
        self._pid_controllers = [PIDController(kps[i], kis[i], kds[i], ts, filter_coefficient) for i in
                                 range(self._dof)]
        self._qd_prev = np.zeros(3)
        self._dqd_prev = np.zeros(3)

    def control(self, qd: np.ndarray, q: np.ndarray):
        dqd: np.ndarray = (qd - self._qd_prev) / self._ts
        ddqd: np.ndarray = (dqd - self._dqd_prev) / self._ts
        pid_outs = [self._pid_controllers[i].control(qd[i], q[i]) for i in range(self._dof)]

        self._qd_prev = np.array(qd)
        self._dqd_prev = np.array(dqd)

        u1x = pid_outs[0] + ddqd[0]
        u1y = pid_outs[1] + ddqd[1]
        u1z = pid_outs[2] + self._model.g + ddqd[2]

        # phi = q[5]
        # X = (np.cos(phi) * np.cos(phi) * u1x + np.cos(phi) * np.sin(phi) * u1y) / u1z
        #
        # if X > 1:
        #     sin_thetad = 1.0
        #     thetad = np.pi / 2.0
        # elif X < -1:
        #     sin_thetad = -1.0
        #     thetad = -np.pi / 2.0
        # else:
        #     sin_thetad = X
        #     thetad = np.arcsin(X)
        #
        # psid = np.arctan((np.sin(phi) * np.cos(phi) * u1x - np.cos(phi) * np.cos(phi) * u1y) / u1z)

        # u1 = u1z / (np.cos(phi) * np.cos(psid))

        psi = q[5]
        thetad = np.arctan((u1x * np.cos(psi) + u1y * np.sin(psi)) / u1z)
        phid = np.arctan((u1x * np.sin(psi) - u1y * np.cos(psi)) * np.cos(thetad) / u1z)
        u1 = self._model.m * u1z / (np.cos(phid) * np.cos(thetad))

        return [u1, phid, thetad]
