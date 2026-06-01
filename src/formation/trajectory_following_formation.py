from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .formation import Formation


@dataclass
class TrajectoryFollowingFormation:
    """把几何编队(相对位姿)变成“轨迹主导”的编队参考。

    给定 leader 的 (pos, vel, yaw, yaw_rate) 与编队几何，输出每架机的：
    - 世界系位置参考 p_ref[i]
    - 世界系速度参考 v_ref[i]

    关键点：offset 会随 leader yaw 旋转；并加入由于 yaw_rate 产生的切向速度项，
    从而转弯时内圈更快、外圈更慢（整体围绕 leader 转弯）。
    """

    geometry: Formation
    ignore_yaw: bool = False

    def offsets_rel_world(self, leader_yaw: float, leader_id: int) -> np.ndarray:
        psi = 0.0 if self.ignore_yaw else float(leader_yaw)
        deltas = self.geometry.cal_deltas(psi)
        offsets_world = np.asarray(deltas[0].T, dtype=float)
        leader_id = int(np.clip(leader_id, 0, offsets_world.shape[0] - 1))
        return offsets_world - offsets_world[leader_id]

    def reference(
        self,
        leader_pos: np.ndarray,
        leader_vel: np.ndarray,
        leader_yaw: float,
        leader_yaw_rate: float,
        leader_id: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (pos_ref_all, vel_ref_all) with shape (N,3)."""

        leader_pos = np.asarray(leader_pos, dtype=float)
        leader_vel = np.asarray(leader_vel, dtype=float)
        offsets_rel = self.offsets_rel_world(leader_yaw, leader_id)

        pos_ref = leader_pos[None, :] + offsets_rel

        # d/dt (Rz(psi) * d) = psi_dot * [-y, x, 0] in world frame for offset vector [x,y,z]
        w = float(0.0 if self.ignore_yaw else leader_yaw_rate)
        v_off = np.zeros_like(offsets_rel)
        v_off[:, 0] = -w * offsets_rel[:, 1]
        v_off[:, 1] = w * offsets_rel[:, 0]
        v_off[:, 2] = 0.0

        vel_ref = leader_vel[None, :] + v_off
        return pos_ref, vel_ref
