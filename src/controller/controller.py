import numpy as np

from .position_controller import PositionController
from .orientation_controller import OrientationController

from src.model import Model


class Controller:

    def __init__(self, kps: list, kis: list, kds: list, model: Model, ts=0.001) -> None:
        super().__init__()
        self._position_controller = PositionController(kps[0: 3], kis[0: 3], kds[0: 3], model, ts)
        self._orientation_controller = OrientationController(kps[3: 6], kis[3: 6], kds[3: 6], model, ts)

    def control(self, qd: np.ndarray, q: np.ndarray, angular_velocity):
        qd_position = np.array([qd[0], qd[1], qd[2]])
        q_position = np.array([q[0], q[1], q[2], q[3], q[4], q[5]])
        u1, phid, thetad = self._position_controller.control(qd_position, q_position)

        qd_orientation = np.array([phid, thetad, qd[3]])
        q_orientation = np.array([q[3], q[4], q[5]])
        orientation_controller_out = self._orientation_controller.control(qd_orientation, q_orientation,
                                                                          angular_velocity)
        return [u1, *orientation_controller_out], [phid, thetad]
