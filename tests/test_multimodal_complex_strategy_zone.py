import numpy as np

from tests.multimodal_complex_strategy_zone_config import MultimodalComplexStrategyZoneConfig
from tests.stable_passage_demo_kernel import run_stable_mixed_passage_demo
from tests.test_three_snake_dynamic_gates import TestThreeSnakeDynamicGates


def _override_specs(base_specs, updates_by_prefix):
    """Return a copied spec list with demo-local parameter overrides."""
    tuned_specs = []
    for spec in base_specs:
        next_spec = dict(spec)
        next_spec.update(updates_by_prefix.get(str(spec.get('prefix')), {}))
        tuned_specs.append(next_spec)
    return tuned_specs


class TestMultimodalComplexStrategyZone(MultimodalComplexStrategyZoneConfig, TestThreeSnakeDynamicGates):
    """Viewer demo for a visually cluttered eight-obstacle VLM replanning scene."""

    # ======================================================================
    # 底层执行调参区
    # ======================================================================
    # 当前文件使用“确定性兜底 + 在线 VLM 高层策略”模式。
    # 底层参数已调稳；VLM 通过异步链路给出可验证 PassagePlan，验证通过后进入控制缓存。
    USE_LLM_STRATEGY = False
    LLM_PROVIDER = 'deterministic'
    ONLINE_REPLANNING_ENABLE = True
    ONLINE_REPLANNING_PROVIDER = 'qwen_vlm'
    PRESERVE_ONLINE_PASSING_ORDER = True
    ONLINE_VLM_PREVENT_LATER_OBSERVE_X = True
    ONLINE_VLM_PREVENT_SHORTER_CLEAR_X = True
    ONLINE_VLM_MIN_EFFECTIVE_X_DELTA = 0.25
    ONLINE_VLM_MAX_CLEAR_EXTENSION_X = 0.80
    # VLM 返回时效性门控。VLM 候选返回时必须仍然离 observe_x 有足够空间/时间，
    # 否则只进入证据链，不允许改写底层执行缓存。整体飞行速度调慢后，TTC 会自然变大。
    ONLINE_VLM_TIMELINESS_ENABLE = True
    ONLINE_VLM_MIN_REMAINING_X = 2.0
    ONLINE_VLM_MIN_REMAINING_TTC = 2.5
    ONLINE_VLM_TTC_MIN_SPEED_X = 0.25
    VLM_PROVIDER = 'off'

    # 在线感知提前量。开阔视野下，非视觉 SFSC + 前视 RGB 可以在更远处给出可规划几何，
    # 因此让 VLM 在传感器覆盖范围远端就开始判断，给 6-8s 推理留出更宽的 TTC。
    ONLINE_LOOKAHEAD_DISTANCE = 34.0
    ONLINE_DETAIL_REVEAL_DISTANCE = 30.0
    ONLINE_MAX_VISIBLE_OBSTACLES = 3
    ONLINE_REPLAN_MIN_INTERVAL = 4.0

    FINAL_TARGET_X = 112.0
    FINAL_GATE_PASS_CLEAR_X = 2.80

    DEBUG_PASSAGE_PLAN = True
    SEMANTIC_PANEL_ENABLE = True
    SEMANTIC_TRACE_CONSOLE = False
    VISUAL_TRACE_ENABLE = True
    VISUAL_TRACE_WIDTH = 480
    VISUAL_TRACE_HEIGHT = 270

    # 1) 障碍区窗口
    # GATE_OBSERVE_DISTANCE_X：越大越早散队/对准，复杂场景建议先保守调大。
    # GATE_PASS_CLEAR_X：越大越晚切下一个障碍，避免尾机还没完全离开就转向。
    # POST_REFORM_CRUISE_X：最终恢复编队后的巡航速度，只影响障碍区之后。
    GATE_OBSERVE_DISTANCE_X = 7.2
    GATE_PASS_CLEAR_X = 1.65
    POST_REFORM_CRUISE_X = 1.25

    # 2) 蛇形队列间距
    # 后机追太近/互相干扰：优先调大 SNAKE_SPACING_X 和 SNAKE_MIN_GAP_X。
    # 队伍太长导致超时：再逐步调小 SNAKE_SPACING_X 或调大速度。
    SNAKE_SPACING_X = 1.05
    SNAKE_MAX_EXTRA_GAP_X = 2.00
    SNAKE_MIN_GAP_X = 0.70
    SNAKE_FOLLOWER_Y_ALIGN_GAP_X = 0.95
    SNAKE_EXTRA_GAP_Y_GAIN = 0.95
    SNAKE_EXTRA_GAP_Z_GAIN = 0.70
    SNAKE_SPACING_KP = 0.72

    # 3) 前向速度节奏
    # 如果“还没对准就撞障碍”：降低 REACTIVE_*_SPEED_X 或调大观察距离。
    # 如果“能过但太慢”：先小幅提高 REACTIVE_SPEED_X，再提高实体障碍 speed_x。
    REACTIVE_SPEED_X = 0.8
    REACTIVE_NEAR_GATE_SPEED_X = 0.36
    REACTIVE_MIN_SPEED_X = 0.04
    TEMPORAL_SLOT_WAIT_SPEED_X = 0.30
    REACTIVE_COMMIT_SPEED_X = 0.95
    REACTIVE_EXIT_SPEED_X = 1.05
    # y/z 对齐后进入“承诺通过”段。当前障碍都较薄，对齐后直接加速释放，
    # 避免慢速贴着障碍边缘一边修正一边蹭过去。
    GATE_ALIGNED_PASS_SPEED_X = 1.34
    SOLID_BYPASS_ALIGNED_PASS_SPEED_X = 1.18
    GATE_SKIP_TEMPORAL_LIMIT_AFTER_ALIGN = True
    SNAKE_MAX_SPEED_X = 1.25
    SNAKE_MIN_SPEED_X = 0.025
    REACTIVE_BRAKE_KP_X = 0.24

    # 4) 横向/高度对准能力
    # 横向绕不开：提高 REACTIVE_Y_KP 或 MAX_CMD_SPEED_XY。
    # 横向过冲：降低 REACTIVE_Y_KP 或 CMD_SLEW_RATE_XY。
    # 越顶/穿洞高度慢：提高 HEIGHT_HOLD_KP 或 MAX_CMD_SPEED_Z。
    REACTIVE_Y_KP = 1.30
    MAX_CMD_SPEED_XY = 2.00
    CMD_SLEW_RATE_XY = 1.0
    MAX_CMD_SPEED_Z = 1.80
    CMD_SLEW_RATE_Z = 1.0
    HEIGHT_HOLD_KP = 2.05
    HEIGHT_HOLD_DAMPING = 1.55

    # 5) 障碍前对齐保持
    # 这组参数决定“距离障碍多近时，如果 y/z 没对准就先停 x”。
    # 复杂场景里先保守一些，防止未完成侧绕就冲进障碍体。
    OBSTACLE_ALIGN_HOLD_DISTANCE_X = 1.8
    BYPASS_ALIGN_HOLD_DISTANCE_X = 2.8

    # Y 轴对齐容差是当前最容易影响“等多久才允许继续前进”的参数。
    # 数值越大：越不苛刻，过障更快，但更可能擦边。
    # 数值越小：越保守，需要更久对准，适合排查撞门/撞箱问题。
    GATE_Y_ALIGNMENT_TOL = 1.4
    BYPASS_Y_ALIGNMENT_TOL = 3.00

    GATE_ALIGNMENT_Y_TOL = GATE_Y_ALIGNMENT_TOL
    GATE_ALIGNMENT_Z_TOL = 0.50
    # 这两个只用于“是否允许高速承诺穿门”，必须比普通对齐容差严格。
    # 普通容差可以放宽以减少等待，但高速穿越不能用 1m 级别的宽容差。
    GATE_COMMIT_ALIGNMENT_Y_TOL = 0.34
    GATE_COMMIT_ALIGNMENT_Z_TOL = 0.46
    BYPASS_ALIGNMENT_Y_TOL = BYPASS_Y_ALIGNMENT_TOL
    BYPASS_ALIGNMENT_Z_TOL = 0.50
    GATE_COMMIT_DISTANCE_X = 0.90

    # 6) 实体绕障全局默认值
    # 单个障碍物没有写 speed_x/clearance 时会使用这里。
    SOLID_BYPASS_ACTIVATE_X = 7.20
    SOLID_BYPASS_CLEAR_X = 4.00
    SOLID_BYPASS_SIDE_CLEARANCE = 1.20
    SOLID_BYPASS_TOP_CLEARANCE = 0.42
    SOLID_BYPASS_SPEED_X = 0.62
    SOLID_BYPASS_Y_KP = 1.12
    SOLID_BYPASS_Z_BAND = 0.55

    # 7) 控制器层安全距离
    # 互相贴得太近：调大 COLLISION_SAFE_DISTANCE / ACTIVATION_DISTANCE。
    # 避碰投影太强导致队形被推散：小幅调低 SAFE 或 PROJECTION_ITERS。
    COLLISION_SAFE_DISTANCE = 0.90
    COLLISION_HARD_DISTANCE = 0.44
    COLLISION_ACTIVATION_DISTANCE = 1.30
    COLLISION_PROJECTION_ITERS = 5

    # 最后一门后不要过早横向回撤。先让无人机沿 x 方向完全脱离窄洞，
    # 再进入门后等待/恢复编队，避免外侧无人机刚出洞就被拉向 y=±1/±2 擦门柱。
    POST_GATE_HOLD_CLEAR_X = FINAL_GATE_PASS_CLEAR_X
    POST_GATE_HOLD_X = 3.40
    POST_GATE_HOLD_STAGGER_X = 0.55
    POST_GATE_HOLD_KP_XY = 0.62
    POST_GATE_HOLD_MAX_SPEED_XY = 0.90

    # 8) 实体绕障阶段局部避碰
    # 只在 active_solid 绕障时生效；普通窄门蛇形仍由时序槽控制。
    # 追尾风险高：调大 MIN_GAP_X 或 BRAKE_GAIN。
    # 侧向擦碰风险高：调大 LATERAL_RADIUS_XY 或 LATERAL_GAIN，但过大可能扰动绕障路线。
    SOLID_BYPASS_COLLISION_GUARD_ENABLE = True
    SOLID_BYPASS_GUARD_MIN_GAP_X = 1.50
    SOLID_BYPASS_GUARD_HARD_GAP_X = 1.20
    SOLID_BYPASS_GUARD_BRAKE_GAIN = 0.70
    SOLID_BYPASS_GUARD_CLOSE_SPEED_X = 0.06
    SOLID_BYPASS_GUARD_LATERAL_RADIUS_XY = 0.86
    SOLID_BYPASS_GUARD_LATERAL_GAIN = 0.24
    SOLID_BYPASS_GUARD_MAX_Y_SPEED = 0.18
    SOLID_BYPASS_GUARD_X_WINDOW = 3.20

    # 9) 每个动态门的运动难度
    # amplitude/omega 越大门越难追；先调底层时可降低，论文实验再逐步加难度。
    GATE_MOTION_SPECS = _override_specs(
        MultimodalComplexStrategyZoneConfig.GATE_MOTION_SPECS,
        {
            'gate1': {'amplitude': 2.45, 'omega': 0.32, 'omega2': 0.36},
            'gate2': {'amplitude': 1.65, 'omega': 0.27, 'omega2': 0.31},
            'gate3': {'amplitude': 2.45, 'omega': 0.30, 'omega2': 0.34},
            'gate4': {
                'amplitude': 2.55,
                'omega': 0.30,
                'omega2': 0.34,
                'center_lookahead_time': 0.16,
                'max_center_lead_y': 0.14,
            },
        },
    )

    # 10) 每个实体障碍物的绕障参数
    # route_policy 可选：left/right/split_by_lane/over。
    # speed_x 越小越保守；activate_x 越大越早进入绕障；clear_x 越大越久保持绕障。
    SOLID_BYPASS_SPECS = _override_specs(
        MultimodalComplexStrategyZoneConfig.SOLID_BYPASS_SPECS,
        {
            'over_beam': {
                'route_policy': 'over',
                'speed_x': 0.56,
                'align_speed_x': 0.24,
                'top_clearance': 0.36,
                'activate_x': 7.80,
                'clear_x': 4.20,
            },
            'cone_cluster': {
                'route_policy': 'split_by_lane',
                'speed_x': 0.60,
                'align_speed_x': 0.24,
                'side_clearance': 1.18,
                'activate_x': 7.20,
                'clear_x': 4.00,
            },
            'tall_tower': {
                'route_policy': 'right',
                'speed_x': 0.56,
                'align_speed_x': 0.22,
                'side_clearance': 1.32,
                'activate_x': 8.20,
                'clear_x': 4.50,
            },
            'side_box_d': {
                'route_policy': 'split_by_lane',
                'speed_x': 0.60,
                'align_speed_x': 0.24,
                'side_clearance': 1.20,
                'activate_x': 7.20,
                'clear_x': 4.20,
            },
        },
    )

    def _build_gate_runtime_specs(self, model):
        gate_specs = super()._build_gate_runtime_specs(model)
        if gate_specs:
            gate_specs[-1] = dict(gate_specs[-1])
            gate_specs[-1]['pass_clear_x'] = float(self.FINAL_GATE_PASS_CLEAR_X)
        return gate_specs

    def _apply_solid_bypass_collision_guard(
        self,
        drone_id: int,
        predecessor_id: int | None,
        positions,
        desired_vel,
        active_solid: dict,
        route_hint: str,
    ):
        if not bool(self.SOLID_BYPASS_COLLISION_GUARD_ENABLE):
            return np.asarray(desired_vel, dtype=float)

        adjusted = np.asarray(desired_vel, dtype=float).copy()
        own_pos = np.asarray(positions[int(drone_id)], dtype=float)

        if predecessor_id is not None:
            prev_pos = np.asarray(positions[int(predecessor_id)], dtype=float)
            gap_x = float(prev_pos[0] - own_pos[0])
            if 0.0 <= gap_x < float(self.SOLID_BYPASS_GUARD_MIN_GAP_X):
                alpha = 1.0 - gap_x / max(float(self.SOLID_BYPASS_GUARD_MIN_GAP_X), 1e-6)
                speed_cap = float(adjusted[0]) * (1.0 - float(self.SOLID_BYPASS_GUARD_BRAKE_GAIN) * alpha)
                if gap_x < float(self.SOLID_BYPASS_GUARD_HARD_GAP_X):
                    speed_cap = min(speed_cap, float(self.SOLID_BYPASS_GUARD_CLOSE_SPEED_X))
                adjusted[0] = max(0.0, min(float(adjusted[0]), speed_cap))

        route_hint = str(route_hint)
        if route_hint in {'over', 'under'}:
            return adjusted

        solid_x = float(active_solid['x'])
        if abs(float(own_pos[0]) - solid_x) > float(self.SOLID_BYPASS_GUARD_X_WINDOW):
            return adjusted

        y_correction = 0.0
        radius = float(self.SOLID_BYPASS_GUARD_LATERAL_RADIUS_XY)
        for other_id, other_pos in enumerate(positions):
            if int(other_id) == int(drone_id):
                continue
            other_pos = np.asarray(other_pos, dtype=float)
            if abs(float(other_pos[0]) - solid_x) > float(self.SOLID_BYPASS_GUARD_X_WINDOW):
                continue
            delta = own_pos - other_pos
            dist_xy = float(np.linalg.norm(delta[:2]))
            if dist_xy >= radius:
                continue
            if abs(float(delta[1])) < 1e-6:
                direction_y = 1.0 if int(drone_id) % 2 == 0 else -1.0
            else:
                direction_y = float(np.sign(delta[1]))
            strength = (radius - dist_xy) / max(radius, 1e-6)
            y_correction += float(self.SOLID_BYPASS_GUARD_LATERAL_GAIN) * strength * direction_y

        max_y = float(self.SOLID_BYPASS_GUARD_MAX_Y_SPEED)
        adjusted[1] += float(np.clip(y_correction, -max_y, max_y))
        return adjusted

    def _should_hold_for_obstacle_alignment(
        self,
        current_pos,
        obstacle_x: float,
        target_y: float,
        target_z: float,
        y_tol: float,
        z_tol: float,
        hold_distance_x: float,
    ) -> bool:
        """Complex-demo local fix: wait for both lateral and vertical alignment."""
        current_pos = np.asarray(current_pos, dtype=float)
        x_gap = float(obstacle_x - current_pos[0])
        if x_gap < 0.0 or x_gap > float(hold_distance_x):
            return False
        y_error = abs(float(target_y) - float(current_pos[1]))
        z_error = abs(float(target_z) - float(current_pos[2]))
        return y_error > float(y_tol) or z_error > float(z_tol)

    def test_multimodal_complex_strategy_zone_viewer(self):
        if self.DEBUG_PASSAGE_PLAN:
            print('[VIEWER] multimodal complex strategy zone: deterministic fallback bottom-layer tuning')
        run_stable_mixed_passage_demo(self)

        leader_final = self._last_leader_final
        passage_metrics = self._last_passage_metrics
        self.assertTrue(
            float(leader_final[0]) >= self.FINAL_TARGET_X - 1.0,
            msg=f'leader did not reach multimodal complex target, leader={np.round(leader_final, 3)}',
        )
        self.assertTrue(passage_metrics.success, msg=passage_metrics.to_dict())


if __name__ == '__main__':
    import unittest

    unittest.main()
