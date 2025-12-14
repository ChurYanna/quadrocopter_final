import copy
import math

import numpy as np

from src.formation import Formation
from src.model import Model
from src.parameter import Parameter
from ..orientation_controller import OrientationController
from ..velocity_controller import VelocityController


class FormationController:
    # def __init__(self, kps: list, kis: list, kds: list, model: Model, count: int, ts=0.001, position_gain=0.0) -> None:
    #     super().__init__()
    #     self._model = copy.deepcopy(model)
    #     self._count = count
    #     self._velocity_controllers = [
    #         VelocityController(kps[0: 3], kis[0: 3], kds[0: 3], self._model, ts=ts, position_gain=position_gain) for _
    #         in range(self._count)]
    #     self._orientation_controllers = [OrientationController(kps[3: 6], kis[3: 6], kds[3: 6], self._model, ts=ts) for
    #                                      _ in range(self._count)]

    # def control(self, parameters: list[Parameter], formation: Formation):

    #     for i in range(1, self._count):
    #         parameters[i].dposd = parameters[0].dposd
    #         parameters[i].psid = parameters[0].psid

    #     position = [np.array([parameters[i].pos - parameters[j].pos for j in range(self._count)]).T for i in
    #                 range(self._count)]
    #     deltas = formation.cal_deltas(parameters[0].psi)

    #     e = [np.sum(deltas[i] - position[i], 1) for i in range(self._count)]

    #     out = np.zeros(self._count * 4)

    #     for i in range(self._count):
    #         u1 = self._velocity_controllers[i].control(parameters[i], e[i])
    #         orientation_controller_out = self._orientation_controllers[i].control2(parameters[i])
    #         control_out = np.array([u1, *orientation_controller_out])
    #         torques = self._model.assign(control_out)
    #         out[4 * i: 4 * (i + 1)] = torques
    #     return out

    def __init__(self, kps: list, kis: list, kds: list, model: Model, count: int, ts=0.001, position_gain=0.0, use_formation=True,
                 blend_duration: float = 1.2, gate_distance: float = 1.5, gate_vel: float = 1) -> None:
        # 父类初始化（原有逻辑不变，父类无额外参数）
        super().__init__()
        self._model = copy.deepcopy(model)
        self._count = count
        self.use_formation = use_formation  # 新增：编队模式开关（默认开启）
        # 初始化控制器（原有逻辑不变）
        self._velocity_controllers = [
            VelocityController(kps[0: 3], kis[0: 3], kds[0: 3], self._model, ts=ts, position_gain=position_gain) 
            for _ in range(self._count)
        ]
        self._orientation_controllers = [
            OrientationController(kps[3: 6], kis[3: 6], kds[3: 6], self._model, ts=ts) 
            for _ in range(self._count)
        ]
        # switch / blending state
        # 切换状态机相关变量
        # switch_state 表示当前切换状态：'idle' 空闲，'arming' 准备中，'rendezvous' 靠近中，
        # 'blending' 混合过渡中，'active' 已完全进入编队
        self.switch_state = 'idle'
        # 外部请求标志（外部调用 request_enable_formation 会把该标志设为 True）
        self.enable_requested = False
        # 混合（blending）参数：混合持续时间与已过时间
        self.blend_duration = blend_duration
        self.blend_elapsed = 0.0
        # 门限（gate）：允许直接进入混合/编队的最大距离与速度差
        self.gate_distance = gate_distance
        self.gate_vel = gate_vel
        # rendezvous（靠近）参数与状态，若门未满足可触发靠近行为
        self.rendezvous_enabled = True
        self.rendezvous_duration = None  # 动态计算：在进入 rendezvous 状态时按距离/速度确定
        self.rendezvous_speed = 2     # 靠近时的最大移动速度（m/s）
        self.rendezvous_timer = 0.0
        self.rendezvous_tol = 3       # 到达靠近目标的容差（米），小于该值视为到位
        # 保存原始 gains 与旧目标，用于混合恢复
        self.saved_position_gains = None
        self.old_dposds = None
        self.old_psids = None  # 进入rendezvous/blending时保存各机当前偏航，用于偏航渐入
        # rendezvous阶段偏航渐入窗口（秒）：避免进入rendezvous时立即大幅偏航导致姿态尖峰
        self.rendezvous_yaw_smooth_window = 2.0
        self._rendezvous_yaw_elapsed = 0.0
        # 激活编队前允许的最大位置误差阈值（单位：米），超出则拒绝进入 active
        self.activation_tol = self.rendezvous_tol
        # 最小驻留时间：防止在 rendezvous <-> blending 之间快速来回切换（单位：秒）
        self.rendezvous_min_time = 0.6
        # 在从 rendezvous 开始 blending 前要求的较宽松阈值（若误差大于此值则延长 rendezvous）
        self.blend_start_tol = self.activation_tol * 1.5
        # 内部时间累积与上次状态变化时间（用于实现最小驻留 / 冷却）
        self._time_accum = 0.0
        self._last_state_change_time = 0.0
        # 临时 position_gain 保持（由外部请求触发），用于试验脚本不再直接操作 controllers
        # 当 _temp_hold_remaining > 0 时，position_gain 会从 0 平滑恢复到 temp_saved_position_gains
        self._temp_hold_total = 0.0
        self._temp_hold_remaining = 0.0
        self._temp_saved_position_gains = None
        # Active阶段增益二次平滑窗口（秒），用于进一步降低切换瞬态
        self.active_gain_smooth_window = 0.5
        self._active_gain_elapsed = 0.0
        # Active阶段误差权重渐入窗口（秒）（B优化）：在进入active后的短窗内逐步放大编队误差权重，避免瞬时注入导致尖峰
        self.active_error_smooth_window = 0.5
        self._active_error_elapsed = 0.0
        # 软饱和设置（推力与姿态角的限幅），可按需要调整或禁用
        try:
            mass = float(getattr(self._model, 'm', 1.0))
        except Exception:
            mass = 1.0
        self.u1_min = mass * 9.81 * 0.5  # 下限 ~0.5g
        self.u1_max = mass * 9.81 * 2.0  # 上限 ~2g
        self.max_tilt_rad = float(np.deg2rad(30.0))  # 姿态角限幅 ±30°
        # A优化：速度/偏航对齐判据（进入active前需满足）
        self.align_window = 0.8  # 对齐评估滑动窗口（秒）
        self.align_cos_thresh = 0.6  # 速度方向余弦相似度阈值（≥该值）
        self.align_yaw_thresh_rad = float(np.deg2rad(30.0))  # 偏航误差阈值（弧度）
        self._align_accum = 0.0
        self._align_samples = []  # 保存最近窗口的对齐样本：[(cos_sim_i, yaw_err_i), ...]
    # 过去可能用于暂存领航机期望速度的变量（已移除：不再在过渡期间暂停领航机）

    def control(self, parameters: list[Parameter], formation: Formation):
        # 1. 在 active 状态下才同步 dposd/psid（避免在 blending 阶段覆盖外部插值）
        if self.use_formation and getattr(self, 'switch_state', 'idle') == 'active':
            for i in range(1, self._count):
                parameters[i].dposd = parameters[0].dposd
                parameters[i].psid = parameters[0].psid

        # 在 rendezvous / blending 阶段对偏航进行纠正：
        # - rendezvous：将 followers 的 psid 渐入到 leader 的 psid，确保朝向一致。
        # - blending：对偏航做与速度相同的线性渐入（从进入blending时保存的旧偏航 -> leader偏航）。
        elif getattr(self, 'switch_state', 'idle') == 'rendezvous':
            try:
                leader_psi = parameters[0].psid
                # 渐入：使用窗口时间线性插值到leader偏航
                gamma = 1.0
                if self.rendezvous_yaw_smooth_window and self.rendezvous_yaw_smooth_window > 0.0:
                    gamma = min(1.0, self._rendezvous_yaw_elapsed / self.rendezvous_yaw_smooth_window)
                # 若未保存旧偏航，则以当前偏航为旧值
                if self.old_psids is None or len(self.old_psids) != self._count:
                    self.old_psids = [p.psid for p in parameters]
                for i in range(1, self._count):
                    psi0 = self.old_psids[i]
                    dpsi = math.atan2(math.sin(leader_psi - psi0), math.cos(leader_psi - psi0))
                    parameters[i].psid = psi0 + gamma * dpsi
            except Exception:
                pass
        elif getattr(self, 'switch_state', 'idle') == 'blending':
            try:
                leader_psi = parameters[0].psid
                alpha = min(1.0, self.blend_elapsed / max(1e-6, self.blend_duration))
                # 若未保存旧偏航，则以当前偏航为旧值
                if self.old_psids is None or len(self.old_psids) != self._count:
                    self.old_psids = [p.psid for p in parameters]
                for i in range(1, self._count):
                    psi0 = self.old_psids[i]
                    # 角度插值需考虑环绕，使用最短角差线性渐入
                    dpsi = math.atan2(math.sin(leader_psi - psi0), math.cos(leader_psi - psi0))
                    parameters[i].psid = psi0 + alpha * dpsi
            except Exception:
                pass

        # 2. 计算位置误差：在 blending 或 active 阶段引入误差反馈；其他阶段误差为 0
        if getattr(self, 'switch_state', 'idle') in ('blending', 'active'):
            position = [
                np.array([parameters[i].pos - parameters[j].pos for j in range(self._count)]).T
                for i in range(self._count)
            ]
            deltas = formation.cal_deltas(parameters[0].psi)
            e_raw = [np.sum(deltas[i] - position[i], 1) for i in range(self._count)]
            # B优化：active初期在短窗内逐步放大误差权重（alpha_e），避免瞬时注入造成尖峰
            if getattr(self, 'switch_state', 'idle') == 'active' and self.active_error_smooth_window and self.active_error_smooth_window > 0.0:
                try:
                    gamma = min(1.0, self._active_error_elapsed / self.active_error_smooth_window)
                except Exception:
                    gamma = 1.0
                e = [gamma * ei for ei in e_raw]
            else:
                e = e_raw
        else:
            e = [np.zeros(3) for _ in range(self._count)]

        # 3. 计算控制力矩（共用逻辑）
        out = np.zeros(self._count * 4)
        for i in range(self._count):
            # 速度控制器：编队模式受误差e影响，独立模式仅跟踪自身目标（e=0）
            u1 = self._velocity_controllers[i].control(parameters[i], e[i])
            # 姿态控制器：始终跟踪自身偏航角目标
            orientation_controller_out = self._orientation_controllers[i].control2(parameters[i])
            # 软饱和：对姿态角限幅（若 orientation_controller_out 为 [phi, theta, psi] 或相似）
            try:
                orientation_controller_out[0] = float(np.clip(orientation_controller_out[0], -self.max_tilt_rad, self.max_tilt_rad))
                orientation_controller_out[1] = float(np.clip(orientation_controller_out[1], -self.max_tilt_rad, self.max_tilt_rad))
            except Exception:
                pass
            # 整合控制输出
            control_out = np.array([u1, *orientation_controller_out])
            # 可选：对推力进行限幅（若 assign 前支持）
            try:
                control_out[0] = float(np.clip(control_out[0], self.u1_min, self.u1_max))
            except Exception:
                pass
            torques = self._model.assign(control_out)
            out[4 * i: 4 * (i + 1)] = torques
        return out

    # ----------------- switching / blending API -----------------
    def request_enable_formation(self):
        self.enable_requested = True
        if self.switch_state == 'idle':
            self.switch_state = 'arming'

    def request_disable_formation(self):
        # immediate disable
        self.enable_requested = False
        self.use_formation = False
        self.switch_state = 'idle'

    def get_state(self):
        """Return the current switch_state string."""
        return self.switch_state

    def is_gate_ok(self, parameters: list[Parameter]):
        """Public wrapper for gate check used by external scripts for diagnostics."""
        return self._check_gate(parameters)

    def apply_temporary_position_gain_zero(self, duration: float):
        """
        Temporarily set all velocity controllers' position_gain to 0 and schedule a smooth
        restore over `duration` seconds. This encapsulates the trial-script behavior so
        external code doesn't need to touch controllers directly.
        """
        try:
            # only set if not already active
            if (self._temp_saved_position_gains is None) or (self._temp_hold_remaining <= 0):
                self._temp_saved_position_gains = [vc.position_gain for vc in self._velocity_controllers]
                for vc in self._velocity_controllers:
                    vc.position_gain = 0.0
                self._temp_hold_total = max(0.0, float(duration))
                self._temp_hold_remaining = self._temp_hold_total
        except Exception:
            # best-effort; ignore failures
            pass

    def sync_qd_prev_to_leader(self, parameters: list[Parameter]):
        """
        Set each velocity controller's internal _qd_prev to the leader's current dposd
        to avoid a differential jump when switching targets. Safe no-op if attribute missing.
        """
        try:
            leader_dposd = parameters[0].dposd.copy()
            for vc in self._velocity_controllers:
                try:
                    if hasattr(vc, '_qd_prev'):
                        vc._qd_prev = leader_dposd.copy()
                    # reset pids to avoid integral windup
                    for pid in getattr(vc, '_pid_controllers', []):
                        try:
                            pid.reset()
                        except Exception:
                            pass
                except Exception:
                    pass
        except Exception:
            pass

    def _check_gate(self, parameters: list[Parameter]):
        # check max distance to leader and max velocity difference
        try:
            leader_pos = parameters[0].pos
            leader_vel = parameters[0].dq[:3]
            dists = [np.linalg.norm(parameters[i].pos - leader_pos) for i in range(1, self._count)]
            vel_diffs = [np.linalg.norm(parameters[i].dq[:3] - leader_vel) for i in range(1, self._count)]
            max_dist = max(dists) if dists else 0.0
            max_vel = max(vel_diffs) if vel_diffs else 0.0
            return (max_dist <= self.gate_distance) and (max_vel <= self.gate_vel)
        except Exception:
            return False

    def _compute_rendezvous_duration(self, parameters: list[Parameter]) -> float:
        """
        根据当前最远无人机到领航机的距离与 rendezvous_speed 动态计算靠近时长（秒）：ceil(max_dist / speed)。
        失败或速度非法时，返回当前 self.rendezvous_duration 作为回退。
        """
        try:
            speed = float(self.rendezvous_speed)
        except Exception:
            speed = 0.0
        if speed <= 0.0:
            return float(self.rendezvous_duration)
        try:
            leader_pos = parameters[0].pos
            dists = [np.linalg.norm(parameters[i].pos - leader_pos) for i in range(1, self._count)]
            max_dist = max(dists) if dists else 0.0
            duration = math.ceil(max_dist / speed)
            return float(max(1, duration))
        except Exception:
            return float(self.rendezvous_duration)

    def update_switch(self, dt: float, parameters: list[Parameter], formation: Formation):
        """
        Advance the enable/disable state machine. This will modify parameters[*].dposd during blending.
        """
        # advance internal time accumulator (for dwell/hysteresis)
        self._time_accum += dt
        # 维护A优化的滑动窗口计时与样本收集（用于速度方向与偏航对齐评估）
        try:
            self._align_accum += dt
            # 收集当前样本：对followers计算与leader速度方向的余弦相似度与偏航误差
            leader_vel = parameters[0].dq[:3]
            leader_yaw = parameters[0].psi
            # 领航速度方向
            lv_norm = np.linalg.norm(leader_vel)
            lv_dir = leader_vel / lv_norm if lv_norm > 1e-6 else np.zeros(3)
            cos_list = []
            yaw_err_list = []
            for i in range(1, self._count):
                fv = parameters[i].dq[:3]
                fv_norm = np.linalg.norm(fv)
                fv_dir = fv / fv_norm if fv_norm > 1e-6 else np.zeros(3)
                # 只考虑水平分量方向对齐（x,y），以减少竖直速度干扰
                lv_xy = lv_dir[:2]
                fv_xy = fv_dir[:2]
                lv_xy_norm = np.linalg.norm(lv_xy)
                fv_xy_norm = np.linalg.norm(fv_xy)
                if lv_xy_norm > 1e-6 and fv_xy_norm > 1e-6:
                    cos_sim = float(np.clip(np.dot(lv_xy, fv_xy) / (lv_xy_norm * fv_xy_norm), -1.0, 1.0))
                else:
                    cos_sim = 1.0  # 如果速度过小，视为对齐良好（避免噪声导致拒绝）
                yaw_err = float(np.arctan2(np.sin(parameters[i].psi - leader_yaw), np.cos(parameters[i].psi - leader_yaw)))
                cos_list.append(cos_sim)
                yaw_err_list.append(abs(yaw_err))
            # 聚合为单个样本：最小cos与最大yaw_err（保守判据）
            if cos_list and yaw_err_list:
                self._align_samples.append((min(cos_list), max(yaw_err_list)))
            # 修剪到窗口长度
            window_len = int(max(1, round(self.align_window / max(1e-3, dt))))
            if len(self._align_samples) > window_len:
                self._align_samples = self._align_samples[-window_len:]
        except Exception:
            pass

        # --- Handle temporary position_gain hold/restore requested via API ---
        # If a temp hold is active and we're NOT in blending, perform a time-based
        # linear restore of the saved gains. Blending has its own gain restore logic
        # and should take precedence.
        if getattr(self, '_temp_hold_remaining', 0.0) > 0.0 and self.switch_state != 'blending':
            try:
                # decrease remaining time
                self._temp_hold_remaining = max(0.0, self._temp_hold_remaining - dt)
                if (self._temp_saved_position_gains is not None) and (self._temp_hold_total > 0.0):
                    alpha = 1.0 - (self._temp_hold_remaining / self._temp_hold_total)
                    # apply linear interpolation from 0 -> saved_gain
                    for vc, g in zip(self._velocity_controllers, self._temp_saved_position_gains):
                        vc.position_gain = g * alpha
                if self._temp_hold_remaining <= 0.0:
                    # restore final gains and clear temp state
                    if self._temp_saved_position_gains is not None:
                        for vc, g in zip(self._velocity_controllers, self._temp_saved_position_gains):
                            vc.position_gain = g
                    self._temp_saved_position_gains = None
                    self._temp_hold_total = 0.0
            except Exception:
                pass

        # --- 如果收到开启编队请求，且当前处于空闲或arming状态，则准备进入靠近/混合流程 ---
        if self.enable_requested and self.switch_state in ('idle', 'arming'):
            # 不再在过渡期间暂停领航机：保持 leader 的 dposd 不被覆盖或置零，以避免改变外部期望

            # 如果门条件满足直接进入 blending，否则进入 rendezvous（靠近）
            if self._check_gate(parameters):
                # begin blending
                self.switch_state = 'blending'
                self._last_state_change_time = self._time_accum
                self.blend_elapsed = 0.0
                # 保存原始 gains（以便线性恢复）和当前 velocities 作为起点
                self.saved_position_gains = [vc.position_gain for vc in self._velocity_controllers]
                self.old_dposds = [p.dposd.copy() for p in parameters]
                # 保存当前偏航作为渐入起点
                try:
                    self.old_psids = [p.psid for p in parameters]
                except Exception:
                    pass
                # 将 controllers 的 _qd_prev 对齐为 old_dposds 并置零 gains（避免瞬态）
                for idx, vc in enumerate(self._velocity_controllers):
                    try:
                        vc._qd_prev = self.old_dposds[idx].copy()
                        for pid in vc._pid_controllers:
                            pid.reset()
                    except Exception:
                        pass
                    vc.position_gain = 0.0
            else:
                # 进入靠近阶段
                self.switch_state = 'rendezvous'
                self.rendezvous_timer = 0.0
                self._last_state_change_time = self._time_accum
                # 动态计算 rendezvous_duration
                self.rendezvous_duration = self._compute_rendezvous_duration(parameters)
                # 将偏航渐入窗口设置为靠近时长（与靠近时间一致）并重置计时器
                try:
                    self.rendezvous_yaw_smooth_window = float(self.rendezvous_duration)
                except Exception:
                    pass
                self._rendezvous_yaw_elapsed = 0.0
                # 保存原始 gains（仅第一次设置），并将所有 controllers 的 gains 置零以避免位置增益在靠近时引起突发扭矩
                if self.saved_position_gains is None:
                    self.saved_position_gains = [vc.position_gain for vc in self._velocity_controllers]
                for vc in self._velocity_controllers:
                    vc.position_gain = 0.0
                # 对所有 controllers 同步 _qd_prev 为当前各自参数 dposd（或 leader 的零速度），并 reset pid
                for idx, vc in enumerate(self._velocity_controllers):
                    try:
                        # 直接使用对应参数当前的 dposd 作为基准，移除对已保存领航机速度的依赖
                        base = parameters[idx].dposd
                        vc._qd_prev = base.copy()
                        for pid in vc._pid_controllers:
                            pid.reset()
                    except Exception:
                        pass
                # 保存当前偏航作为渐入起点
                try:
                    self.old_psids = [p.psid for p in parameters]
                except Exception:
                    pass

        # --- Rendezvous: 恒定速度靠近到期望编队位置 ---
        if self.switch_state == 'rendezvous':
            try:
                deltas = formation.cal_deltas(parameters[0].psi)
                leader_pos = parameters[0].pos
                all_within = True
                # 对每架 follower 生成恒定速度靠近指令，并同步 controllers
                for i in range(1, self._count):
                    # deltas[0][:, i] 是领航机到第 i 架的期望相对位置偏移（向量）
                    target_pos = leader_pos + deltas[0][:, i]
                    cur_pos = parameters[i].pos
                    vec = target_pos - cur_pos
                    dist = np.linalg.norm(vec)
                    if dist > self.rendezvous_tol:
                        all_within = False
                    # 目标速度为方向 * rendezvous_speed
                    if dist == 0:
                        desired_vel = np.zeros(3)
                    else:
                        # 三维靠近：包含 z 分量的方向向量
                        desired_vel = (vec / dist) * self.rendezvous_speed
                        # 可选：限制竖直速度幅度，避免 z 方向过快
                        try:
                            max_vz = getattr(self, 'rendezvous_vz_max', None)
                            if (max_vz is not None) and (max_vz > 0):
                                desired_vel[2] = np.clip(desired_vel[2], -max_vz, max_vz)
                        except Exception:
                            pass
                    parameters[i].dposd = desired_vel
                    # 同步控制器状态，防止速度跳变导致的 dqd 突变
                    try:
                        vc = self._velocity_controllers[i]
                        vc._qd_prev = desired_vel.copy()
                        for pid in vc._pid_controllers:
                            pid.reset()
                        vc.position_gain = 0.0
                    except Exception:
                        pass

                self.rendezvous_timer += dt

                # 当所有到位或超时后尝试进入 blending（但尊重最小驻留时间与 blend_start_tol）
                if all_within or (self.rendezvous_timer >= self.rendezvous_duration):
                    since = self._time_accum - self._last_state_change_time if self._last_state_change_time is not None else None
                    if (since is not None) and (since < self.rendezvous_min_time):
                        # 在最小驻留时间内，不立即开始 blending
                        return
                    # 计算最大位置误差以决定是否开始 blending
                    deltas_chk = formation.cal_deltas(parameters[0].psi)
                    max_err_chk = 0.0
                    for j in range(self._count):
                        # 对第 j 架而言，期望的相对位置是 deltas_chk[0][:, j]
                        desired_rel_j = deltas_chk[0][:, j]
                        rel_pos_j = parameters[j].pos - parameters[0].pos
                        err_j = np.linalg.norm(desired_rel_j - rel_pos_j)
                        if err_j > max_err_chk:
                            max_err_chk = err_j
                    if max_err_chk > getattr(self, 'blend_start_tol', self.activation_tol * 2):
                        # 误差仍然较大，延长 rendezvous
                        self.rendezvous_timer = 0.0
                        return
                    # 否则开始 blending
                    self.switch_state = 'blending'
                    self._last_state_change_time = self._time_accum
                    self.blend_elapsed = 0.0
                    self.old_dposds = [p.dposd.copy() for p in parameters]
                    # 保存当前偏航作为渐入起点
                    try:
                        self.old_psids = [p.psid for p in parameters]
                    except Exception:
                        pass
                    # 对 controllers 同步 _qd_prev
                    for idx, vc in enumerate(self._velocity_controllers):
                        try:
                            vc._qd_prev = self.old_dposds[idx].copy()
                            for pid in vc._pid_controllers:
                                pid.reset()
                        except Exception:
                            pass
                    # position_gains 已在靠近阶段被置为 0， blending 时线性恢复
            except Exception:
                # 出现意外则回到 arming
                self.switch_state = 'arming'
                return

        # --- Blending: 从 old_dposds 平滑过渡到当前 leader 的速度 ---
        if self.switch_state == 'blending':
            self.blend_elapsed += dt
            alpha = min(1.0, self.blend_elapsed / max(1e-6, self.blend_duration))
            # leader 目前保持暂停（parameters[0].dposd 应为 0），混合目标以 leader 当前速度为准
            leader_dposd = parameters[0].dposd
            for i in range(self._count):
                parameters[i].dposd = (1.0 - alpha) * self.old_dposds[i] + alpha * leader_dposd
            # 线性恢复 position_gain
            if self.saved_position_gains is not None:
                for vc, g in zip(self._velocity_controllers, self.saved_position_gains):
                    vc.position_gain = g * alpha

            # blending 完成后，检测位置误差与A优化的对齐判据决定是否进入 active
            if alpha >= 1.0:
                try:
                    deltas = formation.cal_deltas(parameters[0].psi)
                    max_err = 0.0
                    for i in range(self._count):
                        desired_rel_i = deltas[0][:, i]
                        rel_pos = parameters[i].pos - parameters[0].pos
                        err = np.linalg.norm(desired_rel_i - rel_pos)
                        if err > max_err:
                            max_err = err
                except Exception:
                    max_err = float('inf')
                # A优化：检查速度方向与偏航在滑动窗口内是否对齐充足
                align_ok = True
                try:
                    if self._align_samples:
                        # 使用窗口内的最小cos与最大yaw_err作为保守度量
                        min_cos = min(s[0] for s in self._align_samples)
                        max_yaw = max(s[1] for s in self._align_samples)
                        align_ok = (min_cos >= self.align_cos_thresh) and (max_yaw <= self.align_yaw_thresh_rad)
                except Exception:
                    align_ok = True

                if (max_err > getattr(self, 'activation_tol', self.rendezvous_tol)) or (not align_ok):
                    # abort blending，回到 rendezvous
                    self.switch_state = 'rendezvous'
                    self.rendezvous_timer = 0.0
                    self._last_state_change_time = self._time_accum
                    # 将当前 velocities 作为新的 old_dposds 起点，并同步 controllers
                    self.old_dposds = [p.dposd.copy() for p in parameters]
                    for idx, vc in enumerate(self._velocity_controllers):
                        try:
                            vc._qd_prev = self.old_dposds[idx].copy()
                            for pid in vc._pid_controllers:
                                pid.reset()
                        except Exception:
                            pass
                        vc.position_gain = 0.0
                    return
                else:
                    # 激活：恢复领航机原始期望速度并让所有无人机以该速度跟随；同时重置姿态 PID，降低瞬态
                    self.switch_state = 'active'
                    self._last_state_change_time = self._time_accum
                    self.use_formation = True
                    self.enable_requested = False
                    # 启动 active 增益二次平滑计时器
                    self._active_gain_elapsed = 0.0
                    # 启动 B优化误差权重渐入计时器
                    self._active_error_elapsed = 0.0
                    # 不再恢复任何被暂存的领航机速度（因为我们不再在过渡期间暂停领航机）
                    # 将最终目标速度设置为领航机的速度，并同步 controllers
                    try:
                        final_leader = parameters[0].dposd
                        for i in range(self._count):
                            parameters[i].dposd = final_leader.copy()
                            try:
                                vc = self._velocity_controllers[i]
                                vc._qd_prev = final_leader.copy()
                                for pid in vc._pid_controllers:
                                    pid.reset()
                                # 姿态控制器 PID 重置，避免积分残留
                                oc = self._orientation_controllers[i]
                                for pid in getattr(oc, '_pid_controllers', []):
                                    try:
                                        pid.reset()
                                    except Exception:
                                        pass
                            except Exception:
                                pass
                    except Exception:
                        pass
                    # 完成激活

        # --- Active 增益二次平滑：在切换后短窗内对 position_gain 再做线性渐入 ---
        if self.switch_state == 'active' and self.active_gain_smooth_window and self.active_gain_smooth_window > 0.0:
            try:
                self._active_gain_elapsed = min(self.active_gain_smooth_window, self._active_gain_elapsed + dt)
                beta = self._active_gain_elapsed / self.active_gain_smooth_window  # 0..1
                if self.saved_position_gains is not None:
                    for vc, g in zip(self._velocity_controllers, self.saved_position_gains):
                        vc.position_gain = g * beta
            except Exception:
                pass
        # B优化：active阶段误差权重渐入计时推进
        if self.switch_state == 'active' and self.active_error_smooth_window and self.active_error_smooth_window > 0.0:
            try:
                self._active_error_elapsed = min(self.active_error_smooth_window, self._active_error_elapsed + dt)
            except Exception:
                pass
        # rendezvous阶段偏航渐入计时推进
        if self.switch_state == 'rendezvous' and self.rendezvous_yaw_smooth_window and self.rendezvous_yaw_smooth_window > 0.0:
            try:
                self._rendezvous_yaw_elapsed = min(self.rendezvous_yaw_smooth_window, self._rendezvous_yaw_elapsed + dt)
            except Exception:
                pass
