import numpy as np

from ..pid_controller import PIDController
from src.model import Model
from src.parameter import Parameter


def wrap(theta) -> tuple:
    number = np.floor(theta / (2 * np.pi))
    theta -= 2 * np.pi * number
    if theta < -np.pi:
        theta += 2 * np.pi
        number -= 1
    elif theta > np.pi:
        theta -= 2 * np.pi
        number += 1

    return theta, number


class OrientationController:
    def __init__(self, kps: list, kis: list, kds: list, model: Model, ts=0.001, filter_coefficient=100.0) -> None:
        super().__init__()
        self._ts = ts
        self._dof = 3
        self._model = model
        self._pid_controllers = [PIDController(kps[i], kis[i], kds[i], ts, filter_coefficient) for i in
                                 range(self._dof)]
        self._qd_prev = np.zeros(3)
        self._dqd_prev = np.zeros(3)

    def control(self, qd: np.ndarray, q: np.ndarray, angular_velocity):
        dqd: np.ndarray = (qd - self._qd_prev) / self._ts
        ddqd: np.ndarray = (dqd - self._dqd_prev) / self._ts

        phi = wrap(q[2])[0]
        phid = wrap(qd[2])

        if phi - phid[0] > np.pi:
            phi += (phid[1] - 1) * 2 * np.pi
        elif phi - phid[0] < -np.pi:
            phi += (phid[1] + 1) * 2 * np.pi
        else:
            phi += phid[1] * 2 * np.pi
        q[2] = phi

        pid_outs = [self._pid_controllers[i].control(qd[i], q[i]) for i in range(self._dof)]

        self._qd_prev = np.array(qd)
        self._dqd_prev = np.array(dqd)

        u2 = self._model.Ixx * (ddqd[0] + pid_outs[0]) + angular_velocity[1] * angular_velocity[2] * (
                self._model.Izz - self._model.Iyy)
        u3 = self._model.Iyy * (ddqd[1] + pid_outs[1]) + angular_velocity[0] * angular_velocity[2] * (
                self._model.Ixx - self._model.Izz)
        u4 = self._model.Izz * (ddqd[2] + pid_outs[2]) + angular_velocity[0] * angular_velocity[1] * (
                self._model.Iyy - self._model.Ixx)

        return [u2, u3, u4]

    def control2(self, parameter: Parameter):
        dqd: np.ndarray = (parameter.orid - self._qd_prev) / self._ts
        ddqd: np.ndarray = (dqd - self._dqd_prev) / self._ts

        # IMPORTANT: 不要修改测量到的 yaw(psi)。
        # 这里只需要把期望 yaw 选择到与当前 psi 最接近的等价角度，使得误差是 wrap 后的小量。
        psi = wrap(parameter.ori[2])[0]
        psid_wrapped = wrap(parameter.psid)[0]
        yaw_err = wrap(psid_wrapped - psi)[0]
        psid_for_pid = psi + yaw_err

        pid_outs = [
            self._pid_controllers[0].control(parameter.orid[0], parameter.ori[0]),
            self._pid_controllers[1].control(parameter.orid[1], parameter.ori[1]),
            self._pid_controllers[2].control(psid_for_pid, psi),
        ]

        self._qd_prev = np.array(parameter.orid)
        self._dqd_prev = np.array(dqd)

        angular_velocity = parameter.omega

        u2 = self._model.Ixx * (ddqd[0] * 0 + pid_outs[0]) + angular_velocity[1] * angular_velocity[2] * (
                self._model.Izz - self._model.Iyy)
        u3 = self._model.Iyy * (ddqd[1] * 0 + pid_outs[1]) + angular_velocity[0] * angular_velocity[2] * (
                self._model.Ixx - self._model.Izz)
        u4 = self._model.Izz * (ddqd[2] * 0 + pid_outs[2]) + angular_velocity[0] * angular_velocity[1] * (
                self._model.Iyy - self._model.Ixx)

        return [u2, u3, u4]
