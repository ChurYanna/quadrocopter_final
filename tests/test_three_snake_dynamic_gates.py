import time
from dataclasses import replace

import numpy as np
import mujoco
import mujoco.viewer

from src.controller import FormationController
from src.formation import LineFormation, TrajectoryFollowingFormation
from src.model import Skydio
from src.parameter import Parameter
from src.passage_planning import (
    FormationState,
    LLMStrategyPipeline,
    MissionPreference,
    MockLLMStrategyGenerator,
    ObstacleField,
    ObstacleFieldEncoder,
    AsyncOnlineReplanningManager,
    OnlineObstaclePerception,
    OnlinePerceptionConfig,
    OnlineReplanner,
    OnlineReplanningConfig,
    PassageRunMetricsRecorder,
    PassabilityEvaluator,
    QwenDashScopeStrategyGenerator,
    SemanticTraceRecorder,
    SpatioTemporalStrategyPlanner,
    StrategyExecutionHelper,
    StrategyValidator,
    build_llm_scene_context,
)
from tests.dynamic_gate_common import DynamicGateSnakeBase


class TestThreeSnakeDynamicGates(DynamicGateSnakeBase):
    """5 机连续蛇形通过三扇动态窄门，最后再恢复编队。

    与单门 demo 的区别：
    - 中途不做门后等待位；
    - 中途不恢复编队；
    - 全队保持蛇形纵队连续穿过 gate1、gate2、gate3；
    - 三扇门同速但相位/幅度错开移动；
    - 全部通过最后一扇门后，再重组并直线飞行一段。
    """

    # ========================= 任务终止与连续门节奏 =========================
    # FINAL_TARGET_X：三门全部通过、重组完成后继续直线飞行到该 x 位置结束。
    # - 想观察重组后稳定性：调大。
    # - 想快速回归测试：调小，但必须大于 gate3.x + 重组所需距离。
    FINAL_TARGET_X = 25.0
    # MAX_TOTAL_TIME：连续三门总超时时间。
    # - 队伍能通过但测试提前结束：调大。
    # - 调参时想更快暴露失败：调小。
    MAX_TOTAL_TIME = 130.0

    # GATE_OBSERVE_DISTANCE_X：进入第一扇门前多远开始解散横队并拉成蛇形。
    # - gate1 前纵向间距还没拉开：调大。
    # - 过早散队导致前半程队形太松：调小。
    GATE_OBSERVE_DISTANCE_X = 5.2
    # GATE_PASS_CLEAR_X：单架无人机超过某门中心多少米后，切换去追下一扇门。
    # - 过门后太早转向下一门、尾部擦当前门：调大。
    # - 已经过门却还继续追当前门，导致下一门对准晚：调小。
    GATE_PASS_CLEAR_X = 1.25
    # POST_REFORM_CRUISE_X：三门后重组完成的最终巡航速度。
    # - 重组后前进太慢：调大。
    # - 重组刚完成时姿态还不稳：调小。
    POST_REFORM_CRUISE_X = 2.0

    # ========================= 三扇动态门运动模型 =========================
    # 三扇门保持相同主速度 omega=0.40，但通过 amplitude/phase/blend 错开轨迹：
    # - amplitude：门心 y 向运动半幅值；越大越难，需要更强横向追踪。
    # - omega / omega2：主/副正弦角速度；越大门移动越快，追不上时优先降低。
    # - phase / phase2：相位；用于让三扇门不要同步移动。
    # - blend：副正弦权重；越大运动越不规则，控制抖动时可降低。
    # 如果只想改变某一扇门难度，优先调该门 amplitude；如果想整体变快/变慢，再统一调 omega。
    GATE_MOTION_SPECS = [
        {
            'prefix': 'gate1',
            'amplitude': 3.60,
            'omega': 0.40,
            'phase': 0.15,
            'blend': 0.28,
            'omega2': 0.40,
            'phase2': 0.80,
        },
        {
            'prefix': 'gate2',
            'amplitude': 3.10,
            'omega': 0.40,
            'phase': 1.35,
            'blend': 0.22,
            'omega2': 0.40,
            'phase2': 2.05,
        },
        {
            'prefix': 'gate3',
            'amplitude': 3.80,
            'omega': 0.40,
            'phase': 2.45,
            'blend': 0.25,
            'omega2': 0.40,
            'phase2': 3.10,
        },
    ]

    # ========================= 连续蛇形队列覆盖参数 =========================
    # 这些参数覆盖单门 demo 的默认值，只服务于“中途不重组、连续追三门”的策略。
    #
    # SNAKE_SPACING_X：连续门中的基础前后间距。
    # - 后机容易碰前机：调大。
    # - 队伍太长、最后一架总是错过下一门节奏：调小。
    SNAKE_SPACING_X = 0.80
    # SNAKE_MAX_EXTRA_GAP_X：未对准时允许后机额外拉开的最大距离。
    # - 后机被前机固定间距拖进门框：调大。
    # - 队伍拉太长导致三门总耗时高：调小。
    SNAKE_MAX_EXTRA_GAP_X = 2.00
    # SNAKE_MAX_SPEED_X：蛇形阶段 x 速度上限。
    # - 整体通过节奏慢：调大。
    # - 横向/高度还没对准就冲到下一门：调小。
    SNAKE_MAX_SPEED_X =  2.20
    # REACTIVE_Y_KP：横向追门心增益。连续门需要比单门更积极，
    # 否则每扇门前都会花太多时间等 y 误差收敛。
    # - 左右对准慢：调大。
    # - 横向过冲或 S 型摆动明显：调小。
    REACTIVE_Y_KP = 2.20
    # MAX_CMD_SPEED_XY：xy 平面速度命令上限。
    # - 追不上横向移动的门：调大。
    # - 姿态倾角过大、xy 控制发抖：调小。
    MAX_CMD_SPEED_XY = 3.00
    # CMD_SLEW_RATE_XY：xy 速度命令变化率限制。
    # - 切换下一门时动作太硬：调小。
    # - 门间转向响应慢：调大。
    CMD_SLEW_RATE_XY = 4.00
    # MAX_CMD_SPEED_Z：z 向速度上限。三门连续时 z 不固定死，
    # 而是追每扇门计算出的 pass_z。
    # - 高度对准慢、到下一门还在下降/上升：调大。
    # - 垂向动作过猛：调小。
    MAX_CMD_SPEED_Z = 1.80
    # CMD_SLEW_RATE_Z：z 速度命令变化率限制。
    # - 高度切换时突变明显：调小。
    # - z 响应拖慢节奏：调大。
    CMD_SLEW_RATE_Z = 1.35
    # HEIGHT_HOLD_KP / HEIGHT_HOLD_DAMPING：高度误差闭环。
    # - z 对齐慢：调大 KP。
    # - z 过冲或上下振荡：调大 DAMPING 或调小 KP。
    HEIGHT_HOLD_KP = 1.65
    HEIGHT_HOLD_DAMPING = 1.25

    # ========================= LLM 策略接口开关 =========================
    # USE_LLM_STRATEGY：默认关闭，保持确定性算法基线。
    # - False：使用本地确定性规划器生成 PassagePlan。
    # - True：走 LLMStrategyPipeline；可选择 mock 或 qwen provider。
    USE_LLM_STRATEGY = True
    # LLM_PROVIDER：
    # - mock：本地闭环，不联网；
    # - qwen：调用千问 DashScope API，需要设置环境变量 DASHSCOPE_API_KEY。
    LLM_PROVIDER = 'qwen'
    # DEBUG_PASSAGE_PLAN：打印 LLM/validator/阶段切换/最终状态，用于定位策略和仿真问题。
    DEBUG_PASSAGE_PLAN = True
    # MOCK_LLM_MODE：
    # - valid：模拟 LLM 输出合法策略；
    # - repairable：模拟轻微时间槽问题，验证 validator 修复；
    # - unsafe / malformed：验证自动回退确定性策略。
    MOCK_LLM_MODE = 'valid'
    # SEMANTIC_TRACE_ENABLE：打开后会输出“LLM 语义证据链”，并写入 logs/semantic_trace_latest.json。
    # 这是给录屏和导师汇报看的语义信息流，不参与底层控制。
    SEMANTIC_TRACE_ENABLE = True
    SEMANTIC_TRACE_CONSOLE = True
    SEMANTIC_TRACE_LOG_DIR = 'logs'
    SEMANTIC_PANEL_ENABLE = False
    SEMANTIC_PANEL_WAIT_ON_FINISH = True
    # ONLINE_REPLANNING_ENABLE：在线语义策略层开关。打开后不再展示“起飞前全局 LLM 看完整地图”，
    # 而是在飞行中基于有限视野逐步触发 mock LLM 局部重规划。
    ONLINE_REPLANNING_ENABLE = False
    ONLINE_REPLANNING_PROVIDER = 'mock'
    ONLINE_LOOKAHEAD_DISTANCE = 14.0
    ONLINE_DETAIL_REVEAL_DISTANCE = 10.0
    ONLINE_MAX_VISIBLE_OBSTACLES = 3
    ONLINE_REPLAN_MIN_INTERVAL = 6.0
    ONLINE_PERCEPTION_UPDATE_INTERVAL = 0.75

    _last_strategy_validation = None
    _last_strategy_used_fallback = False

    def _ensure_semantic_trace(self) -> SemanticTraceRecorder | None:
        if not self.SEMANTIC_TRACE_ENABLE:
            return None
        recorder = getattr(self, '_semantic_trace_recorder', None)
        if recorder is None:
            recorder = SemanticTraceRecorder(
                scenario_id=self.__class__.__name__,
                log_dir=self.SEMANTIC_TRACE_LOG_DIR,
                enabled=True,
                console=self.SEMANTIC_TRACE_CONSOLE,
                panel_enabled=self.SEMANTIC_PANEL_ENABLE,
                reset=True,
            )
            self._semantic_trace_recorder = recorder
        return recorder

    def _build_online_replanning_generator(self):
        provider = str(getattr(self, 'ONLINE_REPLANNING_PROVIDER', 'mock')).lower()
        if provider == 'mock':
            return None
        if provider == 'qwen':
            return QwenDashScopeStrategyGenerator()
        if provider in {'none', 'deterministic'}:
            return None
        raise ValueError(f'unknown online replanning provider: {self.ONLINE_REPLANNING_PROVIDER}')

    def _snake_target_yz(
        self,
        drone_id: int,
        order_index: int,
        positions: list[np.ndarray],
        gate_spec: dict,
        current_t: float,
        start_y_list: list[float],
        passing_order: list[int],
    ) -> tuple[float, float]:
        """连续窄门版本：追动态门心 y，并追该门计算出的穿越高度 z。"""
        current_pos = np.asarray(positions[drone_id], dtype=float)
        center_y = self._gate_center_y(gate_spec, current_t)
        center_vy = self._gate_center_vy(gate_spec, current_t)
        predicted_center_y = float(center_y + center_vy * self.GATE_CENTER_LOOKAHEAD_TIME)
        pass_z = float(gate_spec['pass_z'])

        align_entry_x = float(gate_spec['x'] - self.ALIGN_ENTRY_BUFFER_X)
        observe_x = float(gate_spec['x'] - self.GATE_OBSERVE_DISTANCE_X)
        progress_to_gate = (float(current_pos[0]) - observe_x) / max(1e-6, align_entry_x - observe_x)
        gate_alpha = self._smoothstep(progress_to_gate)

        if order_index == 0:
            line_alpha = 1.0
        else:
            prev_id = int(passing_order[order_index - 1])
            gap_x = float(np.asarray(positions[prev_id], dtype=float)[0] - current_pos[0])
            line_alpha = self._smoothstep(
                (gap_x - self.SNAKE_MIN_GAP_X)
                / max(1e-6, self.SNAKE_FOLLOWER_Y_ALIGN_GAP_X - self.SNAKE_MIN_GAP_X)
            )

        alpha = float(gate_alpha * line_alpha)
        target_y = float((1.0 - alpha) * start_y_list[drone_id] + alpha * predicted_center_y)
        if bool(gate_spec.get('has_bottom_beam', False)):
            safe_pass_z = pass_z
        else:
            safe_pass_z = float(
                np.clip(
                    pass_z,
                    self.INITIAL_Z,
                    self.INITIAL_Z + self.GATE_PASS_MAX_Z_ABOVE_INITIAL,
                )
            )
        target_z = float((1.0 - gate_alpha) * self.INITIAL_Z + gate_alpha * safe_pass_z)
        return target_y, target_z

    def _build_deterministic_passage_plan(
        self,
        model: mujoco.MjModel,
        gate_specs: list[dict],
        original_y_list: list[float],
        solid_bypass_specs: list[dict] | None = None,
    ):
        """四模块确定性核心算法：编码、评估、规划、验证。"""
        formation_state = FormationState(
            num_uavs=self.COUNT,
            original_y=tuple(float(y) for y in original_y_list),
            initial_z=float(self.INITIAL_Z),
            uav_radius_xy=float(self.UAV_RADIUS_XY),
            uav_radius_z=float(self.UAV_RADIUS_Z),
            nominal_speed_x=float(self.SNAKE_MAX_SPEED_X),
            max_speed_xy=float(self.MAX_CMD_SPEED_XY),
            max_speed_z=float(self.MAX_CMD_SPEED_Z),
            max_lateral_speed=float(self.MAX_CMD_SPEED_XY),
        )
        mission = MissionPreference(
            priority='fast_safe_passage',
            allow_disband=True,
            recover_after_last_obstacle=True,
        )
        self._last_formation_state = formation_state
        self._last_mission_preference = mission
        aperture_field = ObstacleFieldEncoder.encode_dynamic_apertures(
            gate_specs,
            aperture_width_fn=lambda gate_spec: self._gate_aperture_width(model, gate_spec),
            formation=formation_state,
        )
        solid_field = ObstacleFieldEncoder.encode_solid_obstacles(solid_bypass_specs or [])
        semantic_field = ObstacleField(aperture_field.obstacles + solid_field.obstacles)
        aperture_reports = PassabilityEvaluator().evaluate_field(aperture_field, formation_state, mission)
        semantic_reports = PassabilityEvaluator().evaluate_field(semantic_field, formation_state, mission)
        semantic_trace = self._ensure_semantic_trace()
        online_mode = bool(getattr(self, 'ONLINE_REPLANNING_ENABLE', False))
        if semantic_trace is not None and not online_mode:
            semantic_trace.record_scene(semantic_field, formation_state, mission, sim_time=0.0)
        deterministic_semantic_plan = SpatioTemporalStrategyPlanner(
            gate_observe_distance_x=self.GATE_OBSERVE_DISTANCE_X,
            gate_pass_clear_x=self.GATE_PASS_CLEAR_X,
            snake_spacing_x=self.SNAKE_SPACING_X,
        ).plan(semantic_field, formation_state, mission, semantic_reports)
        validation = StrategyValidator().validate(deterministic_semantic_plan, semantic_field, formation_state)
        if not validation.valid:
            raise AssertionError(f'passage strategy validation failed: {validation.messages}')
        deterministic_execution_plan = self._aperture_execution_plan(deterministic_semantic_plan, aperture_field)
        if semantic_trace is not None and not online_mode:
            semantic_trace.record_plan(
                deterministic_semantic_plan,
                title='确定性基准策略',
                label='规则算法生成的完整混合障碍基准策略',
                category='deterministic_plan',
                sim_time=0.0,
            )
        self._last_strategy_validation = validation
        self._last_strategy_used_fallback = False
        if online_mode:
            self._last_semantic_obstacle_field = semantic_field
            self._last_semantic_passage_plan = deterministic_semantic_plan
            return aperture_field, aperture_reports, deterministic_execution_plan
        if not self.USE_LLM_STRATEGY:
            if semantic_trace is not None:
                semantic_trace.record_validation(validation, used_fallback=False, sim_time=0.0)
            self._last_semantic_obstacle_field = semantic_field
            self._last_semantic_passage_plan = deterministic_semantic_plan
            return aperture_field, aperture_reports, deterministic_execution_plan

        scene_context = build_llm_scene_context(semantic_field, formation_state, mission, deterministic_semantic_plan)
        if semantic_trace is not None:
            semantic_trace.record_llm_input(scene_context, sim_time=0.0)
        if self.LLM_PROVIDER == 'qwen':
            generator = QwenDashScopeStrategyGenerator()
        elif self.LLM_PROVIDER == 'mock':
            generator = MockLLMStrategyGenerator(deterministic_semantic_plan, mode=self.MOCK_LLM_MODE)
        else:
            raise ValueError(f'unknown LLM provider: {self.LLM_PROVIDER}')
        llm_semantic_plan, llm_validation, used_fallback = LLMStrategyPipeline(
            debug=self.DEBUG_PASSAGE_PLAN,
        ).generate_or_fallback(
            generator,
            scene_context,
            deterministic_semantic_plan,
            semantic_field,
            formation_state,
        )
        if not llm_validation.valid:
            raise AssertionError(f'LLM passage strategy validation failed: {llm_validation.messages}')
        self._last_strategy_validation = llm_validation
        self._last_strategy_used_fallback = bool(used_fallback)
        self._last_semantic_obstacle_field = semantic_field
        self._last_semantic_passage_plan = llm_semantic_plan
        llm_execution_plan = self._aperture_execution_plan(llm_semantic_plan, aperture_field)
        if semantic_trace is not None:
            semantic_trace.record_plan(
                llm_semantic_plan,
                title='LLM 高层策略输出',
                label='LLM 返回并被系统选用的完整混合障碍高层通行策略',
                category='llm_strategy_output',
                sim_time=0.0,
            )
            semantic_trace.record_validation(llm_validation, used_fallback=used_fallback, sim_time=0.0)
        if used_fallback:
            print(f'LLM strategy fallback to deterministic plan: {llm_validation.messages}')
        if self.DEBUG_PASSAGE_PLAN:
            print(
                '[PLAN] selected:',
                f'source={llm_semantic_plan.source}',
                f'id={llm_semantic_plan.plan_id}',
                f'mode={llm_semantic_plan.mode}',
                f'order={llm_semantic_plan.passing_order}',
                f'slot_dt={llm_semantic_plan.time_slot_interval:.3f}',
                f'obstacles={len(llm_semantic_plan.obstacle_strategies)}',
                f'fallback={used_fallback}',
            )
        return aperture_field, aperture_reports, llm_execution_plan

    @staticmethod
    def _aperture_execution_plan(passage_plan, aperture_field):
        aperture_ids = {obstacle.obstacle_id for obstacle in aperture_field.obstacles}
        aperture_strategies = tuple(
            replace(strategy, obstacle_index=index)
            for index, strategy in enumerate(
                strategy
                for strategy in passage_plan.obstacle_strategies
                if strategy.obstacle_id in aperture_ids
            )
        )
        return replace(passage_plan, obstacle_strategies=aperture_strategies, mode='snake_sequence')

    def _online_cleared_obstacle_ids(self, positions: list[np.ndarray]) -> list[str]:
        semantic_plan = getattr(self, '_last_semantic_passage_plan', None)
        if semantic_plan is None:
            return []
        cleared = []
        for strategy in semantic_plan.obstacle_strategies:
            clear_x = float(strategy.clear_x)
            if all(float(np.asarray(pos, dtype=float)[0]) >= clear_x for pos in positions):
                cleared.append(str(strategy.obstacle_id))
        return cleared

    def run_stable_passage_kernel(self):
        model = mujoco.MjModel.from_xml_path(self.SCENE)
        data = mujoco.MjData(model)

        mujoco.mj_setState(model, data, np.append(model.key_qpos, model.key_qvel), mujoco.mjtState.mjSTATE_PHYSICS)
        mujoco.mj_setState(model, data, model.key_ctrl[0, :], mujoco.mjtState.mjSTATE_CTRL)
        mujoco.mj_forward(model, data)

        gate_specs = self._build_gate_runtime_specs(model)
        self._update_dynamic_gates(model, gate_specs, 0.0)
        solid_bypass_specs = self._build_solid_bypass_runtime_specs(model)
        self._update_dynamic_solids(model, solid_bypass_specs, 0.0)
        mujoco.mj_forward(model, data)

        skydio = Skydio()
        count = self.COUNT
        dt = model.opt.timestep

        kps = [1.0, 0.5, 0.5, 50.0, 50.0, 50.0]
        kis = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        kds = [0.1, 0.1, 0.0, 10.0, 10.0, 10.0]

        parameters = [Parameter() for _ in range(count)]
        formation = TrajectoryFollowingFormation(LineFormation(1.0, count), ignore_yaw=True)
        formation_controller = FormationController(
            kps,
            kis,
            kds,
            skydio,
            count,
            ts=dt,
            position_gain=2.0,
            use_formation=False,
            blend_duration=self.BLEND_DURATION,
            tracking_mode='offset_velocity',
            leader_id=0,
            offset_pos_kp=self.OFFSET_POS_KP,
            offset_vel_kd=0.0,
            offset_max_speed=self.OFFSET_MAX_SPEED,
            offsets_world=None,
            offsets_ignore_yaw=True,
            dposd_slew_rate=self.DPOSD_SLEW_RATE,
            collision_avoid_enable=True,
            collision_safe_distance=self.COLLISION_SAFE_DISTANCE,
            collision_hard_distance=self.COLLISION_HARD_DISTANCE,
            collision_activation_distance=self.COLLISION_ACTIVATION_DISTANCE,
            collision_projection_iters=self.COLLISION_PROJECTION_ITERS,
        )

        initial_positions = [
            np.array([0.0, 0.0, self.INITIAL_Z]),
            np.array([0.0, 1.0, self.INITIAL_Z]),
            np.array([0.0, -1.0, self.INITIAL_Z]),
            np.array([0.0, 2.0, self.INITIAL_Z]),
            np.array([0.0, -2.0, self.INITIAL_Z]),
        ]
        original_y_list = [float(position[1]) for position in initial_positions]
        obstacle_field, passability_reports, passage_plan = self._build_deterministic_passage_plan(
            model,
            gate_specs,
            original_y_list,
            solid_bypass_specs=solid_bypass_specs,
        )
        semantic_trace = self._ensure_semantic_trace()
        if semantic_trace is not None and not self.ONLINE_REPLANNING_ENABLE:
            semantic_trace.record_runtime_solids(solid_bypass_specs, sim_time=0.0)
        strategy_executor = StrategyExecutionHelper(passage_plan)
        passing_order = list(passage_plan.passing_order)
        aperture_orders = [passing_order.copy() for _ in gate_specs]
        aperture_order_locked = [False for _ in gate_specs]
        if aperture_order_locked:
            aperture_order_locked[0] = True
        metrics_recorder = PassageRunMetricsRecorder(
            scenario_id=self.__class__.__name__,
            plan=passage_plan,
            uav_count=count,
            obstacle_count=obstacle_field.count,
            final_target_x=self.FINAL_TARGET_X,
            validation_result=self._last_strategy_validation,
            used_fallback=self._last_strategy_used_fallback,
            start_time=0.0,
        )
        online_perception = None
        online_replanner = None
        async_online_replanner = None
        online_replanning_generator = None
        online_last_update_time = float('-inf')
        if self.ONLINE_REPLANNING_ENABLE:
            online_perception = OnlineObstaclePerception(
                self._last_semantic_obstacle_field,
                OnlinePerceptionConfig(
                    lookahead_distance=self.ONLINE_LOOKAHEAD_DISTANCE,
                    detail_reveal_distance=self.ONLINE_DETAIL_REVEAL_DISTANCE,
                    max_visible_obstacles=self.ONLINE_MAX_VISIBLE_OBSTACLES,
                ),
            )
            online_replanner = OnlineReplanner(
                formation=self._last_formation_state,
                mission=self._last_mission_preference,
                config=OnlineReplanningConfig(
                    min_replan_interval=self.ONLINE_REPLAN_MIN_INTERVAL,
                    gate_observe_distance_x=self.GATE_OBSERVE_DISTANCE_X,
                    gate_pass_clear_x=self.GATE_PASS_CLEAR_X,
                    snake_spacing_x=self.SNAKE_SPACING_X,
                    use_mock_when_generator_missing=False,
                ),
            )
            online_replanning_generator = self._build_online_replanning_generator()
            async_online_replanner = AsyncOnlineReplanningManager(
                online_replanner,
                online_replanning_generator,
            )

        rotate_planners = [
            self._build_segment_planner(
                np.array([0.0, position[1], position[2], 0.0], dtype=float),
                np.array([0.0, position[1], position[2], np.pi], dtype=float),
                self.INIT_ROTATE_TIME,
            )
            for position in initial_positions
        ]

        positions = [np.zeros(3, dtype=float) for _ in range(count)]
        velocities = [np.zeros(3, dtype=float) for _ in range(count)]
        prev_cmds = [np.zeros(3, dtype=float) for _ in range(count)]
        obstacle_crossing_times = [dict() for _ in range(len(gate_specs))]
        solid_bypass_locks = [None for _ in range(count)]
        last_reported_phase_keys = [None for _ in range(count)]
        start_y_list = original_y_list.copy()
        stage = 'rotate'
        last_stage = stage
        final_stable_elapsed = 0.0
        post_reform_target_x = self.FINAL_TARGET_X

        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running() and data.time <= self.MAX_TOTAL_TIME:
                step_start = time.time()
                current_t = float(data.time)

                self._update_dynamic_gates(model, gate_specs, current_t)
                self._update_dynamic_solids(model, solid_bypass_specs, current_t)
                mujoco.mj_forward(model, data)

                for drone_id in range(count):
                    parameters[drone_id].q = data.qpos[7 * drone_id: 7 * (drone_id + 1)]
                    parameters[drone_id].dq = data.qvel[6 * drone_id: 6 * (drone_id + 1)]
                    positions[drone_id] = np.asarray(parameters[drone_id].pos, dtype=float).copy()
                    velocities[drone_id] = np.asarray(parameters[drone_id].dpos, dtype=float).copy()
                metrics_recorder.update_positions(current_t, positions)
                if (
                    self.ONLINE_REPLANNING_ENABLE
                    and online_perception is not None
                    and online_replanner is not None
                    and async_online_replanner is not None
                    and current_t - online_last_update_time >= self.ONLINE_PERCEPTION_UPDATE_INTERVAL
                ):
                    online_last_update_time = current_t
                    cleared_ids = self._online_cleared_obstacle_ids(positions)
                    if cleared_ids:
                        async_online_replanner.mark_cleared(cleared_ids)
                    online_window = online_perception.update(
                        leader_x=float(positions[0][0]),
                        planned_obstacle_ids=async_online_replanner.buffer.planned_obstacle_ids,
                        cleared_obstacle_ids=async_online_replanner.buffer.cleared_obstacle_ids,
                    )
                    if semantic_trace is not None and (
                        online_window.has_new_obstacles or online_window.has_unplanned_obstacles
                    ):
                        semantic_trace.record_perception_window(current_t, online_window)
                    if semantic_trace is not None:
                        for event in async_online_replanner.poll_completed(current_t):
                            semantic_trace.record_async_online_replan_event(event)
                        for event in async_online_replanner.submit_if_needed(online_window, current_t):
                            semantic_trace.record_async_online_replan_event(event)
                    else:
                        async_online_replanner.poll_completed(current_t)
                        async_online_replanner.submit_if_needed(online_window, current_t)

                if stage == 'rotate':
                    for drone_id in range(count):
                        rotate_pose = rotate_planners[drone_id].interpolate(min(current_t, self.INIT_ROTATE_TIME))
                        parameters[drone_id].dposd = np.zeros(3, dtype=float)
                        parameters[drone_id].psid = float(rotate_pose[3])
                    if current_t >= self.INIT_ROTATE_TIME:
                        formation_controller.request_enable_formation(leader_id=0)
                        metrics_recorder.mark_stage_transition(current_t, stage, 'formation')
                        if semantic_trace is not None:
                            semantic_trace.record_stage_transition(current_t, stage, 'formation')
                        stage = 'formation'
                        if self.DEBUG_PASSAGE_PLAN and stage != last_stage:
                            print(f'[STAGE] t={current_t:.2f}: rotate -> formation')
                            last_stage = stage

                elif stage == 'formation':
                    parameters[0].dposd = np.array([self.CRUISE_SPEED_X, 0.0, 0.0], dtype=float)
                    parameters[0].psid = np.pi
                    if float(positions[0][0]) >= strategy_executor.first_obstacle_observe_x():
                        if passage_plan.mode != 'snake_sequence':
                            raise AssertionError(f'expected a snake sequence plan, got {passage_plan.mode}')
                        start_y_list = [float(pos[1]) for pos in positions]
                        prev_cmds = [np.asarray(parameters[idx].dposd, dtype=float).copy() for idx in range(count)]
                        formation_controller.request_disable_formation()
                        metrics_recorder.mark_stage_transition(current_t, stage, 'snake')
                        if semantic_trace is not None:
                            semantic_trace.record_stage_transition(current_t, stage, 'snake')
                        stage = 'snake'
                        if self.DEBUG_PASSAGE_PLAN and stage != last_stage:
                            print(
                                f'[STAGE] t={current_t:.2f}: formation -> snake',
                                f'leader_x={positions[0][0]:.2f}',
                            )
                            last_stage = stage

                elif stage == 'snake':
                    for gate_index, gate_spec in enumerate(gate_specs):
                        strategy_executor.record_obstacle_crossings(
                            obstacle_crossing_times[gate_index],
                            positions,
                            obstacle_x=float(gate_spec['x']),
                            current_time=current_t,
                        )
                    active_gate_indices = {
                        strategy_executor.active_obstacle_index_for_position(pos)
                        for pos in positions
                    }
                    for gate_index in sorted(active_gate_indices):
                        if aperture_order_locked[gate_index]:
                            continue
                        if gate_index > 0 and not all(aperture_order_locked[:gate_index]):
                            continue
                        gate_spec = gate_specs[gate_index]
                        previous_gate_x = (
                            float('-inf')
                            if gate_index == 0
                            else float(gate_specs[gate_index - 1]['x'])
                        )
                        if not self._solid_barriers_before_gate_are_clear(
                            positions,
                            solid_bypass_specs,
                            previous_gate_x=previous_gate_x,
                            gate_x=float(gate_spec['x']),
                        ):
                            continue
                        center_y = self._gate_center_y(gate_spec, current_t)
                        aperture_orders[gate_index] = sorted(
                            range(count),
                            key=lambda idx: (
                                abs(float(gate_spec['x']) - float(positions[idx][0])),
                                abs(float(positions[idx][1]) - center_y),
                                idx,
                            ),
                        )
                        aperture_order_locked[gate_index] = True
                        if self.DEBUG_PASSAGE_PLAN:
                            print(
                                f'[ORDER] t={current_t:.2f}: gate{gate_index + 1} adaptive order',
                                aperture_orders[gate_index],
                            )
                        if semantic_trace is not None:
                            semantic_trace.record_adaptive_order(
                                current_t,
                                f'第 {gate_index + 1} 个穿越型障碍',
                                aperture_orders[gate_index],
                            )
                    aperture_snapshots = [
                        {
                            'x': float(gate_spec['x']),
                            'center_y': self._gate_center_y(gate_spec, current_t),
                            'aperture_width': self._gate_aperture_width(model, gate_spec),
                        }
                        for gate_spec in gate_specs
                    ]
                    metrics_recorder.update_aperture_clearance(
                        current_t,
                        positions,
                        aperture_snapshots,
                        uav_radius_xy=self.UAV_RADIUS_XY,
                        x_window=0.28,
                    )
                    drone_gate_indices = {}
                    drone_solid_states = {}
                    drone_phase_keys = {}
                    for phase_drone_id in range(count):
                        phase_gate_index = strategy_executor.active_obstacle_index_for_position(
                            positions[phase_drone_id]
                        )
                        locked_solid = self._solid_spec_by_prefix(
                            solid_bypass_specs,
                            solid_bypass_locks[phase_drone_id],
                        )
                        if (
                            locked_solid is not None
                            and self._solid_bypass_is_cleared_by_drone(
                                positions[phase_drone_id],
                                locked_solid,
                            )
                        ):
                            locked_solid = None
                            solid_bypass_locks[phase_drone_id] = None
                        phase_solid = locked_solid
                        if (
                            phase_solid is None
                            or not self._solid_bypass_is_released(
                                positions[phase_drone_id], positions, phase_solid, gate_specs
                            )
                        ):
                            candidate_solid = self._active_solid_bypass(
                                positions[phase_drone_id],
                                solid_bypass_specs,
                                current_t,
                            )
                            if candidate_solid is None:
                                candidate_solid = self._active_solid_bypass_by_x(
                                    positions[phase_drone_id],
                                    solid_bypass_specs,
                                )
                            if (
                                candidate_solid is not None
                                and self._solid_bypass_is_released(
                                    positions[phase_drone_id],
                                    positions,
                                    candidate_solid,
                                    gate_specs,
                                )
                            ):
                                phase_solid = candidate_solid
                                solid_bypass_locks[phase_drone_id] = str(candidate_solid['prefix'])
                        phase_blocked_solid = None
                        if (
                            phase_solid is None
                            and locked_solid is None
                        ):
                            candidate_solid = self._active_solid_bypass_by_x(
                                positions[phase_drone_id],
                                solid_bypass_specs,
                            )
                            if (
                                candidate_solid is not None
                                and not self._solid_bypass_is_released(
                                    positions[phase_drone_id],
                                    positions,
                                    candidate_solid,
                                    gate_specs,
                                )
                            ):
                                phase_solid_x_gap = float(candidate_solid['x']) - float(positions[phase_drone_id][0])
                                if 0.0 <= phase_solid_x_gap <= float(self.BYPASS_ALIGN_HOLD_DISTANCE_X):
                                    phase_blocked_solid = candidate_solid
                        pending_solid = phase_solid if phase_solid is not None else phase_blocked_solid
                        if pending_solid is not None:
                            upstream_gate_index = self._solid_upstream_gate_index(pending_solid, gate_specs)
                            if upstream_gate_index is not None:
                                phase_gate_index = upstream_gate_index
                        drone_gate_indices[phase_drone_id] = phase_gate_index
                        drone_solid_states[phase_drone_id] = (phase_solid, phase_blocked_solid)
                        if phase_solid is not None:
                            drone_phase_keys[phase_drone_id] = ('solid', str(phase_solid['prefix']))
                        elif phase_blocked_solid is not None:
                            drone_phase_keys[phase_drone_id] = ('blocked_solid', str(phase_blocked_solid['prefix']))
                        else:
                            drone_phase_keys[phase_drone_id] = ('gate', int(phase_gate_index))
                    if semantic_trace is not None:
                        for phase_drone_id in range(count):
                            phase_key = drone_phase_keys[phase_drone_id]
                            if last_reported_phase_keys[phase_drone_id] == phase_key:
                                continue
                            last_reported_phase_keys[phase_drone_id] = phase_key
                            phase_mode, phase_target = phase_key
                            phase_solid, phase_blocked_solid = drone_solid_states[phase_drone_id]
                            if phase_mode == 'solid' and phase_solid is not None:
                                route_policy = str(phase_solid.get('route_policy', 'split_by_lane'))
                                route_text = {
                                    'split_by_lane': '按原始横向位置选择左/右侧绕',
                                    'left': '左侧绕行',
                                    'right': '右侧绕行',
                                    'over': '顶部越障',
                                }.get(route_policy, route_policy)
                                semantic_trace.record_execution_mode(
                                    current_t,
                                    phase_drone_id,
                                    'solid',
                                    str(phase_solid['prefix']),
                                    {
                                        '绕行方式': route_text,
                                        '障碍中心 x': f'{float(phase_solid["x"]):.2f}',
                                    },
                                )
                            elif phase_mode == 'blocked_solid' and phase_blocked_solid is not None:
                                semantic_trace.record_execution_mode(
                                    current_t,
                                    phase_drone_id,
                                    'blocked_solid',
                                    str(phase_blocked_solid['prefix']),
                                    {
                                        '含义': '当前无人机还未完全释放上一个穿越障碍，先在实体障碍前保底对齐。',
                                    },
                                )
                            else:
                                gate_index = int(phase_target)
                                semantic_trace.record_execution_mode(
                                    current_t,
                                    phase_drone_id,
                                    'gate',
                                    f'第 {gate_index + 1} 个穿越型障碍',
                                    {
                                        '目标策略': '预测洞口中心穿越',
                                        '通过高度': f'{float(gate_specs[gate_index]["pass_z"]):.2f} m',
                                    },
                                )
                    for drone_id in range(count):
                        active_gate_index = int(drone_gate_indices[drone_id])
                        active_gate = gate_specs[active_gate_index]
                        active_order = aperture_orders[active_gate_index]
                        order_index = StrategyExecutionHelper.order_index_in(active_order, drone_id)

                        # 最后一扇门之后属于“最后一个障碍物”场景：
                        # 先通过的无人机去门后等待位，给后续无人机让出门心通道；
                        # 等所有无人机都清过 gate3 后，再统一恢复编队。
                        alignment_hold_x = False
                        if (
                            active_gate_index == len(gate_specs) - 1
                            and strategy_executor.should_hold_after_final_obstacle(
                                positions[drone_id],
                                self.POST_GATE_HOLD_CLEAR_X,
                            )
                        ):
                            hold_target = self._post_gate_hold_target(
                                drone_id,
                                order_index,
                                active_gate,
                                original_y_list,
                            )
                            desired_vel = self._post_gate_hold_velocity(
                                positions[drone_id],
                                velocities[drone_id],
                                hold_target,
                            )
                        else:
                            target_y, target_z = self._snake_target_yz(
                                drone_id,
                                order_index,
                                positions,
                                active_gate,
                                current_t,
                                start_y_list,
                                active_order,
                            )
                            active_solid, blocked_solid = drone_solid_states[drone_id]
                            gate_straight_through = (
                                active_solid is None
                                and blocked_solid is None
                                and self._gate_straight_through_zone(positions[drone_id], active_gate)
                            )
                            active_solid_align_alpha = 0.0
                            if active_solid is not None:
                                solid_center = np.array([
                                    float(active_solid['x']),
                                    self._solid_center_y(active_solid, current_t),
                                    float(active_solid['z']),
                                ], dtype=float)
                                route_policy = str(active_solid.get('route_policy', 'split_by_lane'))
                                if route_policy in {'left', 'right', 'over'}:
                                    route_hint = route_policy
                                else:
                                    route_hint = (
                                        'left'
                                        if float(original_y_list[drone_id]) <= float(solid_center[1])
                                        else 'right'
                                    )
                                bypass_target = strategy_executor.bypass_target(
                                    position=positions[drone_id],
                                    obstacle_center=solid_center,
                                    obstacle_size=np.asarray(active_solid['size'], dtype=float),
                                    original_y=float(original_y_list[drone_id]),
                                    clearance_xy=float(active_solid['side_clearance']) + self.UAV_RADIUS_XY,
                                    clearance_z=float(active_solid['top_clearance']) + self.UAV_RADIUS_Z,
                                    route_hint=route_hint,
                                )
                                target_y = float(bypass_target[1])
                                if route_hint == 'over':
                                    target_z = float(bypass_target[2])
                                    z_error = abs(float(target_z) - float(positions[drone_id][2]))
                                    active_solid_align_alpha = self._smoothstep(
                                        1.0 - z_error / max(1e-6, 2.5 * float(self.BYPASS_ALIGNMENT_Z_TOL))
                                    )
                                else:
                                    nominal_gate_z = float(active_gate['pass_z'])
                                    target_z = float(
                                        np.clip(
                                            nominal_gate_z,
                                            self.INITIAL_Z,
                                            self.INITIAL_Z + self.SOLID_BYPASS_Z_BAND,
                                    )
                                )
                                    y_error = abs(float(target_y) - float(positions[drone_id][1]))
                                    active_solid_align_alpha = self._smoothstep(
                                        1.0 - y_error / max(1e-6, 2.5 * float(self.BYPASS_ALIGNMENT_Y_TOL))
                                    )
                            if active_solid is not None:
                                alignment_hold_x = self._should_hold_for_obstacle_alignment(
                                    positions[drone_id],
                                    obstacle_x=float(active_solid['x']),
                                    target_y=target_y,
                                    target_z=target_z,
                                    y_tol=self.BYPASS_ALIGNMENT_Y_TOL,
                                    z_tol=self.BYPASS_ALIGNMENT_Z_TOL,
                                    hold_distance_x=self.BYPASS_ALIGN_HOLD_DISTANCE_X,
                                )
                            elif blocked_solid is not None:
                                blocked_x_gap = float(blocked_solid['x']) - float(positions[drone_id][0])
                                alignment_hold_x = (
                                    0.0 <= blocked_x_gap <= float(self.BYPASS_ALIGN_HOLD_DISTANCE_X)
                                )
                            else:
                                alignment_hold_x = self._should_hold_for_obstacle_alignment(
                                    positions[drone_id],
                                    obstacle_x=float(active_gate['x']),
                                    target_y=target_y,
                                    target_z=target_z,
                                    y_tol=self.GATE_ALIGNMENT_Y_TOL,
                                    z_tol=self.GATE_ALIGNMENT_Z_TOL,
                                    hold_distance_x=self.OBSTACLE_ALIGN_HOLD_DISTANCE_X,
                                )
                            speed_x = self._snake_speed_x(
                                drone_id,
                                order_index,
                                positions,
                                active_gate,
                                target_y,
                                target_z,
                                active_order,
                                predecessor_id=self._phase_aware_predecessor(
                                    drone_id,
                                    active_order,
                                    drone_phase_keys,
                                ),
                            )
                            if active_solid is not None:
                                align_speed_x = float(
                                    active_solid.get('align_speed_x', self.TEMPORAL_SLOT_WAIT_SPEED_X)
                                )
                                bypass_speed_x = (
                                    align_speed_x
                                    + active_solid_align_alpha * (float(active_solid['speed_x']) - align_speed_x)
                                )
                                speed_x = max(speed_x, bypass_speed_x)
                            if gate_straight_through:
                                speed_x = max(speed_x, float(self.REACTIVE_EXIT_SPEED_X))
                            temporal_limit = None
                            if active_solid is None:
                                temporal_limit = strategy_executor.temporal_speed_limit(
                                    drone_id=drone_id,
                                    current_time=current_t,
                                    obstacle_crossing_times=obstacle_crossing_times[active_gate_index],
                                    current_x=float(positions[drone_id][0]),
                                    obstacle_x=float(active_gate['x']),
                                    commit_distance_x=self.GATE_COMMIT_DISTANCE_X,
                                    wait_speed_x=self.TEMPORAL_SLOT_WAIT_SPEED_X,
                                    current_speed_x=speed_x,
                                    min_speed_x=self.SNAKE_MIN_SPEED_X,
                                    passing_order=active_order,
                                )
                            if temporal_limit is not None:
                                speed_x = min(speed_x, temporal_limit)
                                metrics_recorder.record_temporal_gate_hold()
                            if alignment_hold_x:
                                speed_x = 0.0
                                metrics_recorder.record_temporal_gate_hold()
                            desired_vel = self._snake_velocity(
                                positions[drone_id],
                                velocities[drone_id],
                                speed_x,
                                target_y,
                                target_z,
                            )
                            if active_solid is not None:
                                desired_vel[1] *= float(self.SOLID_BYPASS_Y_KP / max(self.REACTIVE_Y_KP, 1e-6))
                        cmd = self._slew_limit_velocity(prev_cmds[drone_id], desired_vel, dt)
                        cmd = self._clip_velocity_limits(cmd)
                        if not alignment_hold_x and not (
                            active_gate_index == len(gate_specs) - 1
                            and strategy_executor.should_hold_after_final_obstacle(
                                positions[drone_id],
                                self.POST_GATE_HOLD_CLEAR_X,
                            )
                        ):
                            cmd[0] = float(max(self.SNAKE_MIN_SPEED_X, cmd[0]))
                        prev_cmds[drone_id] = cmd.copy()
                        parameters[drone_id].dposd = cmd
                        parameters[drone_id].psid = np.pi

                    if self._all_cleared_gate(positions, gate_specs[-1]):
                        formation_controller.request_enable_formation(leader_id=0)
                        final_stable_elapsed = 0.0
                        metrics_recorder.mark_stage_transition(current_t, stage, 'reform')
                        if semantic_trace is not None:
                            semantic_trace.record_stage_transition(current_t, stage, 'reform')
                        stage = 'reform'
                        if self.DEBUG_PASSAGE_PLAN and stage != last_stage:
                            xs = [round(float(pos[0]), 2) for pos in positions]
                            print(f'[STAGE] t={current_t:.2f}: snake -> reform xs={xs}')
                            last_stage = stage

                elif stage == 'reform':
                    parameters[0].dposd = np.array([self.CRUISE_SPEED_X, 0.0, 0.0], dtype=float)
                    parameters[0].psid = np.pi

                    if self._formation_is_stable(positions, positions[0], formation):
                        final_stable_elapsed += dt
                    else:
                        final_stable_elapsed = 0.0

                    if final_stable_elapsed >= self.FINAL_REFORM_STABLE_TIME:
                        post_reform_target_x = min(
                            self.FINAL_TARGET_X,
                            float(positions[0][0] + self.POST_REFORM_CRUISE_X),
                        )
                        metrics_recorder.mark_stage_transition(current_t, stage, 'post_formation')
                        if semantic_trace is not None:
                            semantic_trace.record_stage_transition(current_t, stage, 'post_formation')
                        stage = 'post_formation'
                        if self.DEBUG_PASSAGE_PLAN and stage != last_stage:
                            print(
                                f'[STAGE] t={current_t:.2f}: reform -> post_formation',
                                f'post_target_x={post_reform_target_x:.2f}',
                            )
                            last_stage = stage

                else:
                    if float(positions[0][0]) >= post_reform_target_x:
                        parameters[0].dposd = np.zeros(3, dtype=float)
                        parameters[0].psid = np.pi
                        metrics_recorder.mark_stage_transition(current_t, stage, 'complete')
                        if semantic_trace is not None:
                            semantic_trace.record_stage_transition(current_t, stage, 'complete')
                        stage = 'complete'
                        if self.DEBUG_PASSAGE_PLAN:
                            print(f'[STAGE] t={current_t:.2f}: post_formation complete')
                        break
                    parameters[0].dposd = np.array([self.CRUISE_SPEED_X, 0.0, 0.0], dtype=float)
                    parameters[0].psid = np.pi

                formation_controller.update_switch(dt, parameters, formation)
                torques = formation_controller.control(parameters, formation)
                mujoco.mj_setState(model, data, torques, mujoco.mjtState.mjSTATE_CTRL)
                mujoco.mj_step(model, data)

                with viewer.lock():
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(data.time % 2)
                viewer.sync()

                time_until_next_step = dt - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

        if async_online_replanner is not None:
            async_online_replanner.close()

        leader_final = data.qpos[0:3].copy()
        final_positions = [data.qpos[7 * idx: 7 * idx + 3].copy() for idx in range(count)]
        cleared_last_obstacle = self._all_cleared_gate(final_positions, gate_specs[-1])
        run_success = bool(float(leader_final[0]) >= self.FINAL_TARGET_X - 0.8 and cleared_last_obstacle)
        passage_metrics = metrics_recorder.finish(
            current_time=float(data.time),
            final_stage=stage,
            leader_final=leader_final,
            final_positions=final_positions,
            cleared_last_obstacle=cleared_last_obstacle,
            success=run_success,
        )
        if semantic_trace is not None:
            semantic_trace.record_final_metrics(passage_metrics)
            if self.SEMANTIC_PANEL_WAIT_ON_FINISH and semantic_trace.panel is not None:
                print('[SEMANTIC_PANEL] demo finished; close the semantic panel window to finish unittest.')
                semantic_trace.wait_for_panel_close()
        self._last_leader_final = leader_final.copy()
        self._last_final_positions = [pos.copy() for pos in final_positions]
        self._last_stage = stage
        self._last_passage_metrics = passage_metrics
        if self.DEBUG_PASSAGE_PLAN:
            print(
                '[FINAL]',
                f't={data.time:.2f}',
                f'stage={stage}',
                f'viewer_running={viewer.is_running()}',
                f'leader={np.round(leader_final, 3).tolist()}',
                f'target_x={self.FINAL_TARGET_X:.2f}',
                f'cleared_last={cleared_last_obstacle}',
            )
            print(passage_metrics.summary_line())
        self.assertTrue(
            float(leader_final[0]) >= self.FINAL_TARGET_X - 0.8,
            msg=(
                'leader did not reach the final target after three snake dynamic gates; '
                f't={data.time:.2f}, stage={stage}, leader={leader_final}, '
                f'viewer_running={viewer.is_running()}'
            ),
        )
        self.assertTrue(
            cleared_last_obstacle,
            msg='not all drones cleared the last dynamic gate',
        )
        self.assertEqual(obstacle_field.count, len(gate_specs))
        self.assertTrue(all(report.recommended_mode == 'snake_sequence' for report in passability_reports))
        self.assertEqual(passage_plan.mode, 'snake_sequence')

        expected_pos, _ = formation.reference(
            leader_pos=leader_final,
            leader_vel=np.zeros(3, dtype=float),
            leader_yaw=np.pi,
            leader_yaw_rate=0.0,
            leader_id=0,
        )
        for drone_id in range(count):
            final_pos = data.qpos[7 * drone_id: 7 * drone_id + 3].copy()
            self.assertTrue(
                np.linalg.norm(final_pos - expected_pos[drone_id]) < 2.6,
                msg=f'drone {drone_id} did not return near formation after three snake dynamic gates',
            )

    def test_three_snake_dynamic_gates(self):
        self.run_stable_passage_kernel()
