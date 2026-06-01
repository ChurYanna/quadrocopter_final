from __future__ import annotations

import copy
from typing import Optional

import numpy as np

from src.formation import Formation
from src.model import Model
from src.parameter import Parameter
from ..orientation_controller import OrientationController
from ..velocity_controller import VelocityController


class FormationController:
    """多机编队控制器（控制与编队解耦后的版本）。

    你可以把它理解成两层：
    1) reference 生成（编队层）：由 formation 提供。
        - 传统几何编队：formation.cal_deltas(psi) -> offsets
        - 轨迹主导编队：formation.reference(...) -> (pos_ref_all, vel_ref_all)
    2) reference 跟踪（控制层）：由本控制器完成。
        - 输出每架机的 dposd/psid，然后调用 VelocityController + OrientationController。

    核心用法：
    - 单飞阶段：外部脚本给每架机写 dposd/psid。
    - 开启编队：调用 request_enable_formation(...)，再每步调用 update_switch() + control()。
    - 换 leader：调用 request_change_leader(new_id) 触发一次新的 blending。
    - 解散编队：调用 request_disable_formation()，外部脚本恢复对各机 dposd/psid 的直接控制。

    状态机：idle -> blending -> active
    - idle：不做编队参考覆盖（外部完全控制 dposd/psid）
    - blending：生成接入轨迹(quintic)，让 follower 平滑接入编队
    - active：正常编队跟随（仍可限速/限加速度/限偏航速）
    """

    def __init__(
            self,
            kps: list,
            kis: list,
            kds: list,
            model: Model,
            count: int,
            ts=0.001,
            position_gain=0.0,
            use_formation=True,
            blend_duration: float = 0.8,
            tracking_mode: str = 'error_coupling',
            leader_id: int = 0,
            offset_pos_kp: float = 1.0,
            offset_vel_kd: float = 0.0,
            offset_pos_kp_z: Optional[float] = None,
            offset_vel_kd_z: Optional[float] = None,
            offset_max_speed: Optional[float] = None,
            offset_max_vz: Optional[float] = None,
            offsets_world: Optional[np.ndarray] = None,
            offsets_ignore_yaw: bool = True,
            dposd_slew_rate: float = 2.0,
            dposd_slew_rate_z: Optional[float] = None,
            yaw_blend_duration: float = 1.0,
                yaw_hold_during_blend: bool = True,
                yaw_rate_limit: Optional[float] = 1.0,
            suppress_dqd_seconds: float = 2.0,

                # --- collision avoidance (3D) ---
                collision_avoid_enable: bool = True,
                collision_safe_distance: float = 0.6,
                collision_hard_distance: float = 0.2,
                collision_alpha: float = 2.0,
                collision_alpha_hard: float = 20.0,
                collision_activation_distance: Optional[float] = None,
                collision_projection_iters: int = 2,
                collision_leader_weight: float = 10.0,
    ) -> None:
        """参数说明（只列关键的、和过渡/编队行为直接相关的）：

        - ts: 控制器内部离散步长；建议与仿真 dt 保持一致。
        - blend_duration: 开启编队/换 leader 时，接入轨迹的持续时间（秒）。越大越平滑。
        - tracking_mode:
            - 'offset_velocity': 以 formation 输出的 (p_ref,v_ref) 为目标，生成 dposd。
            - 其它模式保留兼容，但当前测试主要使用 'offset_velocity'。
        - leader_id: 当前领航机索引。

        follower 速度生成（offset_velocity）：
        - offset_pos_kp / offset_pos_kp_z: 位置误差 -> 速度命令（XY/Z 可分开）。
        - offset_vel_kd / offset_vel_kd_z: 相对速度阻尼项（XY/Z 可分开）。
        - offset_max_speed: XY 平面速度上限（m/s）。
        - offset_max_vz: Z 方向速度上限（m/s）。
        - dposd_slew_rate / dposd_slew_rate_z: 速度命令变化率上限（近似限加速度）。

        yaw 过渡：
        - yaw_hold_during_blend: blending 期间冻结 follower 的 psid（默认 True，可抑制切换瞬间偏航暴走）。
        - yaw_rate_limit: active 时 follower psid 的变化率上限（rad/s）。

        - suppress_dqd_seconds: 切换窗口内把 VelocityController._qd_prev 对齐到 dposd，抑制 dqd 尖峰。

                三维避碰（CBF 安全滤波器，作用在 dposd 上）：
                - collision_safe_distance: 期望的安全距离（m）。本需求设置为 0.6m。
                - collision_hard_distance: 最后的兜底距离（m）。本需求设置为 0.2m。
                    当两机距离逼近 hard 距离时，会更强制地推开，尽量避免进入 <0.2m。
                - collision_activation_distance: 启用避碰约束的距离阈值（m）；None 表示默认取 safe+0.4。
                - collision_projection_iters: 迭代投影次数（越大越接近同时满足所有 pair 约束）。
        """
        super().__init__()
        self._model = copy.deepcopy(model)
        self._count = int(count)
        self.use_formation = bool(use_formation)

        self._velocity_controllers = [
            VelocityController(kps[0:3], kis[0:3], kds[0:3], self._model, ts=ts, position_gain=position_gain)
            for _ in range(self._count)
        ]
        self._orientation_controllers = [
            OrientationController(kps[3:6], kis[3:6], kds[3:6], self._model, ts=ts)
            for _ in range(self._count)
        ]

        self.tracking_mode = str(tracking_mode)
        self.leader_id = int(leader_id)

        self.offset_pos_kp = float(offset_pos_kp)
        self.offset_vel_kd = float(offset_vel_kd)
        self.offset_pos_kp_z = float(offset_pos_kp_z) if offset_pos_kp_z is not None else None
        self.offset_vel_kd_z = float(offset_vel_kd_z) if offset_vel_kd_z is not None else None
        self.offset_max_speed = offset_max_speed
        self.offset_max_vz = offset_max_vz
        self.offsets_world = offsets_world
        self.offsets_ignore_yaw = bool(offsets_ignore_yaw)

        self.dposd_slew_rate = float(dposd_slew_rate)
        self.dposd_slew_rate_z = float(dposd_slew_rate_z) if dposd_slew_rate_z is not None else None
        self.yaw_blend_duration = float(yaw_blend_duration)
        self.yaw_hold_during_blend = bool(yaw_hold_during_blend)
        self.yaw_rate_limit = float(yaw_rate_limit) if yaw_rate_limit is not None else None
        self.suppress_dqd_seconds = float(suppress_dqd_seconds)

        self.switch_state = 'idle'
        self.enable_requested = False
        self.blend_duration = float(blend_duration)
        self.blend_elapsed = 0.0

        self._blend_from_dposd: Optional[list[np.ndarray]] = None
        self._blend_from_psid: Optional[list[float]] = None
        self._cmd_prev_dposd: Optional[list[np.ndarray]] = None
        self._cmd_prev_psid: Optional[list[float]] = None
        self._join_coeffs: Optional[list[Optional[np.ndarray]]] = None
        self._yaw_blend_elapsed = 0.0
        self._suppress_dqd_remaining = 0.0
        self._dt_last = 0.0

        self.collision_avoid_enable = bool(collision_avoid_enable)
        self.collision_safe_distance = float(collision_safe_distance)
        self.collision_hard_distance = float(collision_hard_distance)
        self.collision_alpha = float(collision_alpha)
        self.collision_alpha_hard = float(collision_alpha_hard)
        if collision_activation_distance is None:
            self.collision_activation_distance = float(self.collision_safe_distance + 0.4)
        else:
            self.collision_activation_distance = float(collision_activation_distance)
        self.collision_projection_iters = int(max(0, collision_projection_iters))
        self.collision_leader_weight = float(max(1e-6, collision_leader_weight))

    def _apply_collision_avoidance(
        self,
        parameters: list[Parameter],
        v_nom_all: list[np.ndarray],
        leader_id: int,
    ) -> list[np.ndarray]:
        """在速度命令层做三维避碰安全滤波（无外部依赖）。

        形式上是把 v_nom 投影到一组线性不等式（CBF 近似）所定义的半空间交集上。
        约束针对每一对 i-j：
            h = ||p_i - p_j||^2 - d^2
            h_dot + alpha * h >= 0
        其中 h_dot = 2*(p_i-p_j)^T*(v_i-v_j)
        => (p_i-p_j)^T*(v_i-v_j) >= -(alpha/2) * (||p_i-p_j||^2 - d^2)

        这里同时做两层：safe_distance(0.6m) 与 hard_distance(0.2m)。
        """
        if not self.collision_avoid_enable:
            return v_nom_all

        n_agents = int(self._count)
        if n_agents <= 1:
            return v_nom_all

        v = [np.asarray(vn, dtype=float).copy() for vn in v_nom_all]
        pos = [np.asarray(p.pos, dtype=float).copy() for p in parameters]

        d_safe = float(max(1e-6, self.collision_safe_distance))
        d_hard = float(max(1e-6, min(self.collision_hard_distance, d_safe)))
        d_act = float(max(d_safe, self.collision_activation_distance))
        alpha = float(max(0.0, self.collision_alpha))
        alpha_hard = float(max(alpha, self.collision_alpha_hard))

        weights = np.ones(n_agents, dtype=float)
        if 0 <= leader_id < n_agents:
            weights[leader_id] = float(self.collision_leader_weight)

        def project_halfspace(i: int, j: int, n: np.ndarray, b: float):
            rel = float(np.dot(n, v[i] - v[j]))
            if rel >= b:
                return
            c = float(b - rel)
            nn = float(np.dot(n, n))
            if nn < 1e-12:
                return
            wi = float(weights[i])
            wj = float(weights[j])
            denom = nn * (1.0 / wi + 1.0 / wj)
            if denom < 1e-12:
                return
            lam = c / denom
            v[i] = np.asarray(v[i] + (lam / wi) * n, dtype=float)
            v[j] = np.asarray(v[j] - (lam / wj) * n, dtype=float)

        iters = int(self.collision_projection_iters)
        if iters <= 0:
            return v

        for _ in range(iters):
            for i in range(n_agents):
                for j in range(i + 1, n_agents):
                    dp = pos[i] - pos[j]
                    dist2 = float(np.dot(dp, dp))
                    if dist2 < 1e-12:
                        # 极端重合：随机微扰方向避免数值问题
                        dp = np.array([1.0, 0.0, 0.0], dtype=float)
                        dist2 = 1.0
                    dist = float(np.sqrt(dist2))
                    if dist > d_act:
                        continue

                    # safe layer
                    h = dist2 - d_safe * d_safe
                    b_safe = -0.5 * alpha * h
                    project_halfspace(i, j, dp, b_safe)

                    # hard layer (only meaningful when close)
                    if dist < (d_safe + 1e-6):
                        h2 = dist2 - d_hard * d_hard
                        b_hard = -0.5 * alpha_hard * h2
                        project_halfspace(i, j, dp, b_hard)

            # 速度保护：保持与原逻辑一致
            v = [self._clip_speed(vi) for vi in v]

        return v

    def set_tracking_mode(self, tracking_mode: str):
        self.tracking_mode = str(tracking_mode)

    def set_leader_id(self, leader_id: int):
        self.leader_id = int(leader_id)

    def set_offsets_world(self, offsets_world: Optional[np.ndarray]):
        self.offsets_world = offsets_world

    @staticmethod
    def _wrap_to_pi(angle: float) -> float:
        return float((angle + np.pi) % (2.0 * np.pi) - np.pi)

    @staticmethod
    def _quintic_coeffs_1d(p0: float, v0: float, a0: float, p1: float, v1: float, a1: float, T: float) -> np.ndarray:
        """Return coefficients c0..c5 for p(t)=sum c_k t^k, with boundary conditions at t=0 and t=T."""
        T = float(max(1e-6, T))
        c0 = float(p0)
        c1 = float(v0)
        c2 = float(a0) / 2.0

        t2 = T * T
        t3 = t2 * T
        t4 = t3 * T
        t5 = t4 * T

        A = np.array([
            [t3, t4, t5],
            [3.0 * t2, 4.0 * t3, 5.0 * t4],
            [6.0 * T, 12.0 * t2, 20.0 * t3],
        ], dtype=float)

        b = np.array([
            float(p1) - (c0 + c1 * T + c2 * t2),
            float(v1) - (c1 + 2.0 * c2 * T),
            float(a1) - (2.0 * c2),
        ], dtype=float)

        c3, c4, c5 = np.linalg.solve(A, b)
        return np.array([c0, c1, c2, c3, c4, c5], dtype=float)

    @classmethod
    def _quintic_coeffs_vec(cls, p0: np.ndarray, v0: np.ndarray, a0: np.ndarray,
                            p1: np.ndarray, v1: np.ndarray, a1: np.ndarray, T: float) -> np.ndarray:
        coeffs = np.zeros((3, 6), dtype=float)
        for k in range(3):
            coeffs[k, :] = cls._quintic_coeffs_1d(p0[k], v0[k], a0[k], p1[k], v1[k], a1[k], T)
        return coeffs

    @staticmethod
    def _eval_quintic(coeffs: np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray]:
        t = float(max(0.0, t))
        tt = np.array([1.0, t, t * t, t ** 3, t ** 4, t ** 5], dtype=float)
        dtt = np.array([0.0, 1.0, 2.0 * t, 3.0 * t * t, 4.0 * t ** 3, 5.0 * t ** 4], dtype=float)
        p = coeffs @ tt
        v = coeffs @ dtt
        return np.asarray(p, dtype=float), np.asarray(v, dtype=float)

    def _get_offsets_world(self, formation: Optional[Formation], leader_psi: float) -> np.ndarray:
        if self.offsets_world is not None:
            offsets = np.asarray(self.offsets_world, dtype=float)
            if offsets.ndim != 2 or offsets.shape[1] != 3:
                raise ValueError('offsets_world must have shape (N,3)')
            if offsets.shape[0] < self._count:
                pad = np.zeros((self._count - offsets.shape[0], 3), dtype=float)
                offsets = np.vstack([offsets, pad])
            if offsets.shape[0] > self._count:
                offsets = offsets[:self._count, :]
            return offsets

        if formation is None:
            return np.zeros((self._count, 3), dtype=float)

        psi = 0.0 if self.offsets_ignore_yaw else float(leader_psi)
        deltas = formation.cal_deltas(psi)
        offsets = np.asarray(deltas[0].T, dtype=float)
        if offsets.shape != (self._count, 3):
            offsets2 = np.zeros((self._count, 3), dtype=float)
            n = min(self._count, offsets.shape[0])
            offsets2[:n, :] = offsets[:n, :]
            offsets = offsets2
        return offsets

    def _clip_speed(self, v: np.ndarray) -> np.ndarray:
        v2 = np.asarray(v, dtype=float).copy()
        if self.offset_max_speed is not None:
            vmax = float(self.offset_max_speed)
            if vmax > 0:
                speed_xy = float(np.linalg.norm(v2[:2]))
                if speed_xy > vmax and speed_xy > 1e-9:
                    v2[0:2] = (v2[0:2] / speed_xy) * vmax
        if self.offset_max_vz is not None:
            vzmax = float(self.offset_max_vz)
            if vzmax > 0:
                v2[2] = float(np.clip(v2[2], -vzmax, vzmax))
        return v2

    def _slew_limit_vec(self, v_cmd: np.ndarray, v_prev: np.ndarray) -> np.ndarray:
        dt = float(self._dt_last)
        if dt <= 0.0 or self.dposd_slew_rate <= 0.0:
            return np.asarray(v_cmd, dtype=float)
        dv = np.asarray(v_cmd - v_prev, dtype=float)
        dv_lim_xy = float(self.dposd_slew_rate) * dt
        dv[0] = float(np.clip(dv[0], -dv_lim_xy, dv_lim_xy))
        dv[1] = float(np.clip(dv[1], -dv_lim_xy, dv_lim_xy))
        if self.dposd_slew_rate_z is not None and self.dposd_slew_rate_z > 0.0:
            dv_lim_z = float(self.dposd_slew_rate_z) * dt
            dv[2] = float(np.clip(dv[2], -dv_lim_z, dv_lim_z))
        else:
            dv[2] = float(np.clip(dv[2], -dv_lim_xy, dv_lim_xy))
        return np.asarray(v_prev + dv, dtype=float)

    def request_enable_formation(self, leader_id: Optional[int] = None):
        """请求开启编队。

        - 在 idle 状态下调用会进入 blending（或 blend_duration=0 时直接 active）。
        - 如果传入 leader_id，会先切换 leader。
        - 注意：如果已经在 active 状态，想“重新接入/换 leader”，用 request_change_leader()。
        """
        if leader_id is not None:
            self.set_leader_id(int(leader_id))
        self.enable_requested = True
        if self.switch_state == 'idle':
            self.switch_state = 'blending' if self.blend_duration > 0 else 'active'
            self.blend_elapsed = 0.0
            self._blend_from_dposd = None
            self._blend_from_psid = None
            self._cmd_prev_dposd = None
            self._cmd_prev_psid = None
            self._join_coeffs = None
            self._yaw_blend_elapsed = 0.0
            self._suppress_dqd_remaining = max(0.0, float(self.suppress_dqd_seconds))

    def request_disable_formation(self):
        """解散编队并回到 idle。

        解散后，本控制器不再覆盖 follower 的 dposd/psid。
        外部脚本应恢复对每架机 dposd/psid 的直接赋值（例如各自轨迹飞行）。
        """
        self.enable_requested = False
        self.use_formation = False
        self.switch_state = 'idle'
        self.blend_elapsed = 0.0
        self._blend_from_dposd = None
        self._blend_from_psid = None
        self._cmd_prev_dposd = None
        self._cmd_prev_psid = None
        self._join_coeffs = None
        self._yaw_blend_elapsed = 0.0
        self._suppress_dqd_remaining = 0.0

    def get_state(self):
        return self.switch_state

    def update_switch(self, dt: float, parameters: list[Parameter], formation: Formation):
        """推进简化状态机计时。

        典型每个仿真步调用一次：
        - 先在外部用最新 data.qpos/qvel 更新 parameters
        - 再调用 update_switch(dt, ...)
        - 再调用 control(parameters, formation)
        """
        self._dt_last = float(dt)
        if self._suppress_dqd_remaining > 0.0:
            self._suppress_dqd_remaining = max(0.0, self._suppress_dqd_remaining - float(dt))
        if not self.enable_requested:
            return
        if self.switch_state == 'blending':
            self.blend_elapsed += float(dt)
            if self.blend_elapsed >= self.blend_duration:
                self.switch_state = 'active'
                self.enable_requested = False
                self.use_formation = True
        if self.switch_state == 'active':
            self.enable_requested = False
            self.use_formation = True

    def control(self, parameters: list[Parameter], formation: Formation):
        """生成每架机的控制量（推力+力矩）。

        输入：parameters 为每架机当前状态（pos/dpos/psi/omega 等）以及外部写入的命令(dposd/psid)。
        输出：长度为 4*N 的控制向量（每架机 4 个电机的等效输入），用于写入 mujoco ctrl。
        """
        leader_id = int(np.clip(self.leader_id, 0, self._count - 1))
        leader = parameters[leader_id]
        leader_pos = np.asarray(leader.pos, dtype=float)
        leader_dposd = np.asarray(leader.dposd, dtype=float)
        leader_vel = np.asarray(getattr(leader, 'dpos', np.zeros(3)), dtype=float)
        leader_yaw = float(getattr(leader, 'psi', 0.0))
        leader_yaw_rate = float(np.asarray(getattr(leader, 'omega', np.zeros(3)), dtype=float)[2])

        # 参考生成优先交给 formation（如果它提供轨迹主导 reference 接口）
        if hasattr(formation, 'reference') and callable(getattr(formation, 'reference')):
            pos_ref_all, vel_ref_all = formation.reference(
                leader_pos=leader_pos,
                leader_vel=leader_vel,
                leader_yaw=leader_yaw,
                leader_yaw_rate=leader_yaw_rate,
                leader_id=leader_id,
            )
            offsets_rel = pos_ref_all - leader_pos[None, :]
        else:
            offsets = self._get_offsets_world(formation, leader_yaw)
            offsets_rel = offsets - offsets[leader_id]
            pos_ref_all = leader_pos[None, :] + offsets_rel
            vel_ref_all = np.tile(leader_vel[None, :], (self._count, 1))

        # 生成 followers 的期望 (dposd/psid)
        if self.switch_state in ('blending', 'active'):
            if self._blend_from_dposd is None:
                self._blend_from_dposd = [np.asarray(getattr(p, 'dposd', np.zeros(3))).copy() for p in parameters]
            if self._blend_from_psid is None:
                self._blend_from_psid = [float(getattr(p, 'psid', 0.0)) for p in parameters]
            if self._cmd_prev_dposd is None:
                self._cmd_prev_dposd = [np.asarray(getattr(p, 'dposd', np.zeros(3))).copy() for p in parameters]
            if self._cmd_prev_psid is None:
                self._cmd_prev_psid = [float(getattr(p, 'psid', 0.0)) for p in parameters]

            # 在第一次进入 blending 时，为每个 follower 初始化一段接入轨迹（quintic）。
            # 轨迹目标点使用“预测的编队点”：leader_pos + leader_vel*T + offset。
            if self.switch_state == 'blending' and self._join_coeffs is None:
                T_join = float(max(1e-6, self.blend_duration))
                p_leader_pred = leader_pos + leader_vel * T_join
                psi_pred = float(leader_yaw + leader_yaw_rate * T_join)
                # 预测时刻的编队参考（如果 formation 支持 reference，就用它预测）
                if hasattr(formation, 'reference') and callable(getattr(formation, 'reference')):
                    pos_ref_pred, vel_ref_pred = formation.reference(
                        leader_pos=p_leader_pred,
                        leader_vel=leader_vel,
                        leader_yaw=psi_pred,
                        leader_yaw_rate=leader_yaw_rate,
                        leader_id=leader_id,
                    )
                else:
                    pos_ref_pred = p_leader_pred[None, :] + offsets_rel
                    vel_ref_pred = np.tile(leader_vel[None, :], (self._count, 1))
                self._join_coeffs = [None for _ in range(self._count)]
                for i in range(self._count):
                    if i == leader_id:
                        continue
                    p0 = np.asarray(parameters[i].pos, dtype=float)
                    v0 = np.asarray(parameters[i].dpos, dtype=float)
                    a0 = np.zeros(3, dtype=float)
                    p1 = np.asarray(pos_ref_pred[i], dtype=float)
                    v1 = np.asarray(vel_ref_pred[i], dtype=float)
                    a1 = np.zeros(3, dtype=float)
                    self._join_coeffs[i] = self._quintic_coeffs_vec(p0, v0, a0, p1, v1, a1, T_join)

            alpha = 1.0
            if self.switch_state == 'blending' and self.blend_duration > 0:
                alpha = float(np.clip(self.blend_elapsed / max(1e-9, self.blend_duration), 0.0, 1.0))

            yaw_alpha = 1.0
            if self.yaw_blend_duration > 0.0:
                self._yaw_blend_elapsed += float(self._dt_last)
                yaw_alpha = float(np.clip(self._yaw_blend_elapsed / max(1e-9, self.yaw_blend_duration), 0.0, 1.0))

            for i in range(self._count):
                if i == leader_id:
                    continue
                if self.tracking_mode == 'offset_velocity':
                    kp_z = self.offset_pos_kp if self.offset_pos_kp_z is None else float(self.offset_pos_kp_z)
                    kd_z = self.offset_vel_kd if self.offset_vel_kd_z is None else float(self.offset_vel_kd_z)

                    # blending：使用接入轨迹 (p_ref, v_ref)
                    if self.switch_state == 'blending' and self._join_coeffs is not None and self._join_coeffs[i] is not None:
                        T_join = float(max(1e-6, self.blend_duration))
                        t_join = float(np.clip(self.blend_elapsed, 0.0, T_join))
                        p_ref, v_ref = self._eval_quintic(self._join_coeffs[i], t_join)

                        pos_err = p_ref - np.asarray(parameters[i].pos, dtype=float)
                        v_act = np.asarray(parameters[i].dpos, dtype=float)
                        v_cmd = np.zeros(3, dtype=float)
                        v_cmd[:2] = v_ref[:2] + self.offset_pos_kp * pos_err[:2]
                        v_cmd[2] = float(v_ref[2] + kp_z * pos_err[2])
                        if self.offset_vel_kd > 0.0:
                            v_cmd[:2] = v_cmd[:2] - self.offset_vel_kd * (v_act[:2] - v_ref[:2])
                        if kd_z > 0.0:
                            v_cmd[2] = float(v_cmd[2] - kd_z * (v_act[2] - v_ref[2]))
                    else:
                        # active：正常跟随 formation 给出的参考点
                        p_ref = np.asarray(pos_ref_all[i], dtype=float)
                        pos_err = p_ref - np.asarray(parameters[i].pos, dtype=float)
                        v_act = np.asarray(parameters[i].dpos, dtype=float)
                        v_ref = np.asarray(vel_ref_all[i], dtype=float)
                        v_cmd = np.zeros(3, dtype=float)
                        v_cmd[:2] = v_ref[:2] + self.offset_pos_kp * pos_err[:2]
                        v_cmd[2] = float(v_ref[2] + kp_z * pos_err[2])
                        if self.offset_vel_kd > 0.0:
                            v_cmd[:2] = v_cmd[:2] - self.offset_vel_kd * (v_act[:2] - v_ref[:2])
                        if kd_z > 0.0:
                            v_cmd[2] = float(v_cmd[2] - kd_z * (v_act[2] - v_ref[2]))

                    # 速度限幅与限加速度，最后再过一次限速保护
                    v_cmd = self._clip_speed(v_cmd)
                else:
                    # 兼容 error_coupling 模式：这里只做速度跟随 leader（编队误差交给 e 注入）
                    v_from = np.asarray(self._blend_from_dposd[i], dtype=float)
                    v_cmd = (1.0 - alpha) * v_from + alpha * leader_dposd
                v_cmd = self._clip_speed(v_cmd)
                v_cmd = self._slew_limit_vec(v_cmd, self._cmd_prev_dposd[i])
                v_cmd = self._clip_speed(v_cmd)

                parameters[i].dposd = v_cmd
                self._cmd_prev_dposd[i] = np.asarray(v_cmd, dtype=float)

                # yaw：切换时刻大 yaw 误差会导致电机分配剧烈变化，进而耦合到总推力(Z)。
                # 默认在 blending 期间冻结 yaw(跟随当前测量 psi)，active 再以限速方式对齐到 leader.psid。
                dt = float(self._dt_last) if self._dt_last > 0 else float(getattr(self._velocity_controllers[i], '_ts', 0.001))
                if self.switch_state == 'blending' and self.yaw_hold_during_blend:
                    psid_cmd = float(parameters[i].psi)
                else:
                    psi_t = float(getattr(leader, 'psid', 0.0))
                    psid_prev = float(self._cmd_prev_psid[i])
                    dpsi = self._wrap_to_pi(psi_t - psid_prev)
                    if self.yaw_rate_limit is not None and self.yaw_rate_limit > 0.0:
                        max_step = float(self.yaw_rate_limit) * dt
                        dpsi = float(np.clip(dpsi, -max_step, max_step))
                        psid_cmd = float(psid_prev + dpsi)
                    else:
                        psi0 = float(self._blend_from_psid[i])
                        dpsi0 = self._wrap_to_pi(psi_t - psi0)
                        psid_cmd = float(psi0 + yaw_alpha * dpsi0)
                self._cmd_prev_psid[i] = float(psid_cmd)
                parameters[i].psid = float(psid_cmd)

            # --- collision avoidance (velocity-level safety filter) ---
            # 对 leader + followers 的 dposd 一起做安全滤波，避免靠拢/换 leader 时发生碰撞。
            # 注意：这里的约束是三维球形距离。
            if self.collision_avoid_enable:
                v_nom_all = [np.asarray(getattr(p, 'dposd', np.zeros(3)), dtype=float).copy() for p in parameters]
                v_safe_all = self._apply_collision_avoidance(parameters, v_nom_all, leader_id=leader_id)
                for k in range(self._count):
                    parameters[k].dposd = np.asarray(v_safe_all[k], dtype=float)
                    if self._cmd_prev_dposd is not None:
                        self._cmd_prev_dposd[k] = np.asarray(v_safe_all[k], dtype=float)

        # error_coupling 的误差注入（保持兼容）
        if self.use_formation and self.tracking_mode != 'offset_velocity':
            e = [np.zeros(3) for _ in range(self._count)]
            for i in range(self._count):
                if i == leader_id:
                    continue
                desired_rel = offsets_rel[i]
                actual_rel = np.asarray(parameters[i].pos, dtype=float) - leader_pos
                e[i] = desired_rel - actual_rel
        else:
            e = [np.zeros(3) for _ in range(self._count)]

        out = np.zeros(self._count * 4)
        for i in range(self._count):
            # suppress VelocityController dqd spike during transition window
            if self._suppress_dqd_remaining > 0.0:
                try:
                    self._velocity_controllers[i]._qd_prev = np.asarray(parameters[i].dposd, dtype=float).copy()
                except Exception:
                    pass
            u1 = self._velocity_controllers[i].control(parameters[i], e[i])
            orientation_controller_out = self._orientation_controllers[i].control2(parameters[i])
            control_out = np.array([u1, *orientation_controller_out])
            torques = self._model.assign(control_out)
            out[4 * i: 4 * (i + 1)] = torques
        return out

    def request_change_leader(self, leader_id: int):
        """在飞行中切换领航机。

        这个接口会：
        - 更新 leader_id
        - 触发一次新的 blending（重新生成接入轨迹）
        - 打开 suppress_dqd 窗口

        切换 leader 后，外部脚本需要把“轨迹命令”写到新的 leader 上。
        """
        self.set_leader_id(int(leader_id))
        self.enable_requested = True
        self.switch_state = 'blending' if self.blend_duration > 0 else 'active'
        self.blend_elapsed = 0.0
        self._blend_from_dposd = None
        self._blend_from_psid = None
        self._cmd_prev_dposd = None
        self._cmd_prev_psid = None
        self._join_coeffs = None
        self._yaw_blend_elapsed = 0.0
        self._suppress_dqd_remaining = max(0.0, float(self.suppress_dqd_seconds))
