import numpy as np

from src.model import Model
from src.parameter import Parameter

from ..pid_controller import PIDController


class VelocityController:
    def __init__(self, kps: list, kis: list, kds: list, model: Model, ts=0.001, filter_coefficient=100.0,
                 position_gain=0.0) -> None:
        super().__init__()
        self._ts = ts
        self._dof = 3
        self._model = model
        self._pid_controllers = [PIDController(kps[i], kis[i], kds[i], ts, filter_coefficient) for i in
                                 range(self._dof)]
        self.position_gain = position_gain
        self._qd_prev = np.zeros(3)

    def control(self, parameter: Parameter, e=np.zeros(3)):
        dqd: np.ndarray = (parameter.dposd - self._qd_prev) / self._ts
        pid_outs = [self._pid_controllers[i].control(parameter.dposd[i], parameter.dpos[i]) for i in range(self._dof)]

        self._qd_prev = np.array(parameter.dposd)

        u1x = pid_outs[0] + dqd[0] + self.position_gain * e[0]
        u1y = pid_outs[1] + dqd[1] + self.position_gain * e[1]
        u1z = pid_outs[2] + self._model.g + dqd[2] + self.position_gain * e[2]

        psi = parameter.psi
        thetad = np.arctan((u1x * np.cos(psi) + u1y * np.sin(psi)) / u1z)
        phid = np.arctan((u1x * np.sin(psi) - u1y * np.cos(psi)) * np.cos(thetad) / u1z)
        u1 = self._model.m * u1z / (np.cos(phid) * np.cos(thetad))

        parameter.phid = phid
        parameter.thetad = thetad

        return u1
