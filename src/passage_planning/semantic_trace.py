from __future__ import annotations

import json
import math
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .metrics import PassageRunMetrics
from .multimodal_vlm import (
    MockVLMStrategyGenerator,
    QwenDashScopeVLMStrategyGenerator,
    build_multimodal_vlm_input_packet,
    summarize_multimodal_vlm_packet,
    summarize_vlm_strategy_result,
)
from .semantic_presenter import (
    passage_mode_cn,
    summarize_adaptive_order,
    summarize_llm_input,
    summarize_plan,
    summarize_scene,
    summarize_stage_transition,
    summarize_validation,
    target_policy_cn,
)
from .semantic_panel import SemanticTracePanel
from .structures import FormationState, MissionPreference, ObstacleField, PassagePlan, ValidationResult


@dataclass(frozen=True)
class SemanticTraceEvent:
    """One human-readable SFSC/LLM event for passage visualization."""

    event_id: int
    wall_time: str
    sim_time: float | None
    category: str
    title: str
    summary: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            'event_id': int(self.event_id),
            'wall_time': self.wall_time,
            'sim_time': None if self.sim_time is None else float(self.sim_time),
            'category': self.category,
            'title': self.title,
            'summary': self.summary,
            'details': _json_safe(self.details),
        }


class SemanticTraceRecorder:
    """Records and displays the SFSC/LLM evidence flow around planning.

    The recorder is intentionally independent from MuJoCo.  Current demos can
    write events to JSON/JSONL and print a Chinese evidence-chain preview; the
    future real-time panel can consume the same latest JSON file.
    """

    def __init__(
        self,
        scenario_id: str,
        log_dir: str | Path = 'logs',
        enabled: bool = True,
        console: bool = True,
        panel_enabled: bool = False,
        reset: bool = True,
        vlm_provider: str = 'mock',
    ):
        self.scenario_id = str(scenario_id)
        self.log_dir = Path(log_dir)
        self.enabled = bool(enabled)
        self.console = bool(console)
        self.panel_enabled = bool(panel_enabled)
        self.panel: SemanticTracePanel | None = None
        self.events: list[SemanticTraceEvent] = []
        self.vlm_provider = str(vlm_provider).lower()
        self.mock_vlm_generator = MockVLMStrategyGenerator()
        self.vlm_generator = self._build_vlm_generator(self.vlm_provider)
        self._record_lock = threading.Lock()
        self.latest_path = self.log_dir / 'semantic_trace_latest.json'
        self.history_path = self.log_dir / 'semantic_trace_history.jsonl'
        self.vlm_packet_dir = self.log_dir / 'vlm_packets'
        if self.enabled:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.vlm_packet_dir.mkdir(parents=True, exist_ok=True)
            if reset:
                self.latest_path.write_text('{}\n', encoding='utf-8')
                self.history_path.write_text('', encoding='utf-8')
                for packet_path in self.vlm_packet_dir.glob('*.json'):
                    try:
                        packet_path.unlink()
                    except OSError:
                        pass
            if self.panel_enabled:
                panel = SemanticTracePanel(title=f'SV-STCP SFSC+VLM 证据链 - {self.scenario_id}')
                if panel.start():
                    self.panel = panel

    def record(
        self,
        category: str,
        title: str,
        summary: str,
        details: dict[str, Any] | None = None,
        sim_time: float | None = None,
    ) -> SemanticTraceEvent | None:
        if not self.enabled:
            return None
        with self._record_lock:
            event = SemanticTraceEvent(
                event_id=len(self.events) + 1,
                wall_time=datetime.now().isoformat(timespec='seconds'),
                sim_time=None if sim_time is None else float(sim_time),
                category=str(category),
                title=str(title),
                summary=str(summary),
                details=details or {},
            )
            self.events.append(event)
            self._write_files(event)
        if self.panel is not None:
            self.panel.submit(event.to_dict())
        if self.console:
            self._print_event(event)
        return event

    def record_scene(
        self,
        field: ObstacleField,
        formation: FormationState,
        mission: MissionPreference,
        sim_time: float | None = None,
    ) -> None:
        payload = summarize_scene(field, formation, mission)
        self.record(
            'scene_semantics',
            'SFSC 传感器融合结构化上下文',
            payload['summary'],
            payload['details'],
            sim_time=sim_time,
        )

    def record_llm_input(self, scene_context: dict[str, Any], sim_time: float | None = None) -> None:
        payload = summarize_llm_input(scene_context)
        self.record(
            'llm_input',
            '输入给外部策略模型的 SFSC 上下文',
            payload['summary'],
            payload['details'],
            sim_time=sim_time,
        )

    def record_runtime_solids(
        self,
        solid_specs: list[dict[str, Any]] | tuple[dict[str, Any], ...],
        sim_time: float | None = None,
    ) -> None:
        if not solid_specs:
            return
        lines = []
        for index, solid in enumerate(solid_specs, start=1):
            size = solid.get('size', None)
            if hasattr(size, '__len__') and len(size) >= 3:
                size_text = f'{float(size[0]):.2f} x {float(size[1]):.2f} x {float(size[2]):.2f} m'
            else:
                size_text = '尺寸未知'
            route_policy = str(solid.get('route_policy', 'split_by_lane'))
            route_text = {
                'split_by_lane': '按无人机原始横向位置分流侧绕',
                'left': '左侧绕行',
                'right': '右侧绕行',
                'over': '顶部越障',
            }.get(route_policy, route_policy)
            lines.append(
                f'第 {index} 个实体障碍 {solid.get("prefix", "solid")}：'
                f'中心 x={float(solid.get("x", 0.0)):.2f}，尺寸约 {size_text}，'
                f'推荐执行方式为 {route_text}，横向运动幅值约 {float(solid.get("amplitude", 0.0)):.2f} m'
            )
        self.record(
            'runtime_semantics',
            'SFSC 实体绕行上下文补充',
            (
                f'检测到 {len(solid_specs)} 个实体障碍；它们已经进入 SFSC 高层上下文，'
                '底层执行器负责将 bypass/over 策略落实为连续侧绕或越顶控制。'
            ),
            {
                '实体障碍数量': len(solid_specs),
                '实体障碍摘要': lines,
            },
            sim_time=sim_time,
        )

    def record_execution_mode(
        self,
        current_time: float,
        drone_id: int,
        mode: str,
        obstacle_label: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        mode_text = {
            'gate': '穿越型障碍对齐/穿洞',
            'solid': '实体障碍侧绕/越顶',
            'blocked_solid': '实体障碍前保底对齐',
        }.get(str(mode), str(mode))
        self.record(
            'execution_feedback',
            '单机执行模式切换',
            f'UAV {int(drone_id)} 切换为“{mode_text}”，当前目标：{obstacle_label}。',
            {
                '无人机': f'UAV {int(drone_id)}',
                '执行模式': mode_text,
                '当前目标': obstacle_label,
                **(details or {}),
            },
            sim_time=current_time,
        )

    def record_perception_window(self, current_time: float, window) -> None:
        visible_lines = []
        for perceived in window.visible_obstacles:
            reveal_text = '细节已看清' if perceived.details_revealed else '仅粗略发现'
            plan_text = '需要局部规划' if perceived.requires_planning else '已有策略或暂不规划'
            visible_lines.append(
                f'{perceived.obstacle_id}: {perceived.function}, '
                f'距离 {perceived.relative_distance_x:.2f} m, {reveal_text}, {plan_text}'
            )
        self.record(
            'online_perception',
            '有限视野 SFSC 感知窗口',
            (
                f'当前有限视野范围 x={window.x_min:.2f} 到 x={window.x_max:.2f}；'
                f'发现 {len(window.visible_obstacles)} 个可见障碍。'
            ),
            {
                '新发现障碍': list(window.newly_observed_ids),
                '待规划障碍': list(window.planning_candidate_ids),
                '可见障碍摘要': visible_lines,
            },
            sim_time=current_time,
        )

    def record_online_replan_result(self, result) -> None:
        if not result.triggered:
            return
        if result.accepted:
            summary = (
                f'在线局部重规划已接受，触发原因：{result.reason}；'
                f'覆盖障碍：{list(result.planning_obstacle_ids)}。'
            )
        else:
            summary = (
                f'在线局部重规划未接受，原因：{result.reason}；'
                f'候选障碍：{list(result.planning_obstacle_ids)}。'
            )
        strategy_lines = []
        if result.selected_plan is not None:
            for strategy in result.selected_plan.obstacle_strategies:
                strategy_lines.append(
                    f'{strategy.obstacle_id}: mode={strategy.mode}, policy={strategy.target_policy}, '
                    f'observe_x={strategy.observe_x:.2f}, clear_x={strategy.clear_x:.2f}'
                )
        validation = result.validation
        details = {
            '可见障碍': list(result.visible_obstacle_ids),
            '本次规划障碍': list(result.planning_obstacle_ids),
            '策略来源': None if result.selected_plan is None else result.selected_plan.source,
            '计划编号': None if result.selected_plan is None else result.selected_plan.plan_id,
            '是否使用回退': bool(result.used_fallback),
            '验证通过': None if validation is None else bool(validation.valid),
            '策略缓存已覆盖': [] if result.buffer is None else sorted(result.buffer.planned_obstacle_ids),
            '局部策略': strategy_lines,
        }
        self.record(
            'online_replanning',
            '在线外部策略局部重规划',
            summary,
            details,
            sim_time=result.current_time,
        )

    def record_visual_frame(self, frame) -> None:
        should_submit_vlm = self._should_submit_vlm_for_frame(frame)
        details = {
            '图片路径': frame.path,
            '触发标签': frame.label,
            '触发原因': frame.reason,
            '图像尺寸': f'{frame.width} x {frame.height}',
            '图像类型': getattr(frame, 'render_mode', 'uav_front_rgb'),
            '相机名称': getattr(frame, 'camera_name', None),
            '来源无人机': None if getattr(frame, 'source_uav_id', None) is None else f'UAV {frame.source_uav_id}',
            '视角选择策略': getattr(frame, 'selection_policy', 'frontmost_uav'),
            '融合输入形式': 'frontmost UAV RGB + SFSC 传感器融合结构化上下文',
            'VLM关注点': [
                '前方障碍外观、孔洞/实体/遮挡形态',
                '局部通道是否被视觉遮挡或结构异常',
                '视觉证据是否支持 SFSC 中的可通行性判断',
            ],
            'VLM输出边界': [
                '只输出高层策略建议和解释',
                '不直接输出电机推力或逐帧速度',
                '所有建议必须经过安全验证器后才能进入策略缓冲区',
            ],
            '后续用途': '作为多模态 VLM 输入证据帧；当前阶段只记录，不影响控制。',
            '是否提交VLM': bool(should_submit_vlm and not frame.error),
        }
        if not should_submit_vlm:
            details['VLM跳过原因'] = '非规划关键帧或任务后处理阶段，仅保留视觉证据，不提交 VLM。'
        if frame.error:
            details['采样错误'] = frame.error
            summary = f'无人机前视 RGB 关键帧采样失败：{frame.label}。'
        else:
            summary = f'已保存无人机前视 RGB 关键帧：{frame.label}。'
        self.record(
            'visual_observation',
            '无人机前视 RGB 关键帧',
            summary,
            details,
            sim_time=frame.sim_time,
        )
        if not frame.error and should_submit_vlm and self.vlm_provider != 'off':
            self.record_multimodal_vlm_input_packet(frame)

    def record_multimodal_vlm_input_packet(self, frame) -> None:
        sfsc_events = self._recent_sfsc_events(limit=5)
        packet_id = f'vlm_{len(self.events) + 1:04d}_{float(frame.sim_time):08.2f}_{_safe_packet_label(frame.label)}'
        packet = build_multimodal_vlm_input_packet(
            packet_id=packet_id,
            scenario_id=self.scenario_id,
            frame=frame,
            sfsc_events=sfsc_events,
        )
        packet_path = self.vlm_packet_dir / f'{packet_id}.json'
        packet_path.write_text(
            json.dumps(packet.to_dict(), ensure_ascii=False, indent=2) + '\n',
            encoding='utf-8',
        )
        payload = summarize_multimodal_vlm_packet(packet, str(packet_path))
        self.record(
            'multimodal_vlm_input',
            '多模态 VLM 输入包',
            payload['summary'],
            payload['details'],
            sim_time=frame.sim_time,
        )
        if self.vlm_provider == 'qwen':
            self.record(
                'vlm_strategy_output',
                'VLM 视觉复核请求已提交',
                '已异步提交 Qwen VLM 请求；viewer 和底层控制不会等待模型返回。',
                {
                    '链路阶段': 'VLM 视觉复核链路（不进入控制）',
                    '一句话说明': '对当前前视 RGB + SFSC 做视觉复核，结果只用于解释和展示。',
                    '输入包编号': packet_id,
                    '输入包路径': str(packet_path),
                    'VLM来源': 'qwen_vlm',
                    '是否阻塞控制': False,
                    '失败处理': '若真实 VLM 调用失败，将记录错误并使用 mock VLM 结果用于展示对照。',
                },
                sim_time=frame.sim_time,
            )
            thread = threading.Thread(
                target=self._run_async_vlm_request,
                args=(packet, packet_id, frame.sim_time),
                name=f'vlm-request-{packet_id}',
                daemon=True,
            )
            thread.start()
            return
        result = self.vlm_generator.generate(packet)
        result_path = self.vlm_packet_dir / f'{packet_id}_mock_result.json'
        result_path.write_text(
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + '\n',
            encoding='utf-8',
        )
        result_payload = summarize_vlm_strategy_result(result)
        result_details = {
            '链路阶段': 'VLM 视觉复核链路（不进入控制）',
            '一句话说明': 'mock VLM 只模拟视觉复核输出格式，不改变在线控制缓存。',
            **result_payload['details'],
            '结果文件路径': str(result_path),
        }
        self.record(
            'vlm_strategy_output',
            'Mock VLM 高层策略输出',
            result_payload['summary'],
            result_details,
            sim_time=frame.sim_time,
        )

    def _run_async_vlm_request(self, packet, packet_id: str, sim_time: float) -> None:
        try:
            result = self.vlm_generator.generate(packet)
            suffix = 'qwen_result'
            title = 'Qwen VLM 视觉复核结果'
            summary_prefix = ''
        except Exception as exc:
            self.record(
                'vlm_strategy_output',
                '真实 VLM 调用失败',
                f'Qwen VLM 调用失败：{exc}',
                {
                    '链路阶段': 'VLM 视觉复核链路（不进入控制）',
                    '一句话说明': '真实 VLM 请求失败，本条只说明失败原因。',
                    '输入包编号': packet_id,
                    'VLM来源': 'qwen_vlm',
                    '错误信息': str(exc),
                    '回退展示': '使用 mock VLM 生成一份结构化输出，便于面板链路继续完整展示。',
                    '是否进入控制': False,
                },
                sim_time=sim_time,
            )
            result = self.mock_vlm_generator.generate(packet)
            suffix = 'mock_after_qwen_error'
            title = 'Mock VLM 复核结果（回退展示）'
            summary_prefix = '真实 VLM 失败后，'
        result_path = self.vlm_packet_dir / f'{packet_id}_{suffix}.json'
        result_path.write_text(
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + '\n',
            encoding='utf-8',
        )
        result_payload = summarize_vlm_strategy_result(result)
        result_details = {
            '链路阶段': 'VLM 视觉复核链路（不进入控制）',
            '一句话说明': '真实 Qwen VLM 基于前视 RGB + SFSC 返回解释性高层建议。',
            **result_payload['details'],
            '结果文件路径': str(result_path),
            '是否阻塞控制': False,
        }
        self.record(
            'vlm_strategy_output',
            title,
            summary_prefix + result_payload['summary'],
            result_details,
            sim_time=sim_time,
        )

    def record_async_online_replan_event(self, event) -> None:
        if event.event_type == 'not_triggered':
            return
        source_hint = str(getattr(event, 'strategy_source', '') or '').lower()
        is_vlm_control = 'vlm' in source_hint
        actor = 'VLM' if is_vlm_control else '外部模型'
        title_by_type = {
            'fallback_accepted': '在线规则兜底策略',
            'fallback_rejected': '在线规则兜底失败',
            'submitted': f'异步 {actor} 策略请求已提交',
            'pending': f'异步 {actor} 策略计算中',
            'completed_accepted': f'{actor} 策略候选已通过验证',
            'completed_rejected': f'{actor} 策略未通过验证',
        }
        title = title_by_type.get(event.event_type, '异步在线重规划事件')
        result = event.result or event.fallback_result
        selected_plan = None if result is None else result.selected_plan
        validation = None if result is None else result.validation
        fallback_plan = self._event_fallback_plan(event)
        strategy_source = source_hint or (None if selected_plan is None else selected_plan.source)
        contribution = self._online_external_contribution(event)
        if event.event_type == 'completed_accepted' and contribution['changed'] is False:
            title = f'{actor} 原始策略确认兜底'
        elif event.event_type == 'completed_accepted' and contribution['changed'] is True:
            title = f'{actor} 原始策略提出改写'
        details = {
            '链路阶段': (
                'VLM 多模态在线策略链路（候选策略，等待时效性门控与安全投影后进入执行）'
                if is_vlm_control
                else '外部语义在线策略链路（候选策略，等待时效性门控与安全投影后进入执行）'
            ),
            '一句话说明': self._online_replan_one_line(
                event.event_type,
                is_vlm_control=is_vlm_control,
                contribution=contribution,
            ),
            '事件类型': event.event_type,
            '障碍集合': list(event.obstacle_ids),
            '外部模型耗时': None if event.latency_s is None else f'{event.latency_s:.2f} s',
            '策略来源': strategy_source,
            '计划编号': None if selected_plan is None else selected_plan.plan_id,
            '确定性兜底策略': self._plan_strategy_summary_for_obstacles(fallback_plan, event.obstacle_ids),
            'VLM是否改变兜底策略': contribution['display'],
            '策略变化摘要': contribution['summary'],
            '执行准入状态': self._online_execution_admission_hint(event.event_type, is_vlm_control),
            '是否使用回退': None if result is None else bool(result.used_fallback),
            '验证通过': None if validation is None else bool(validation.valid),
            '当前缓存覆盖': [] if result is None or result.buffer is None else sorted(result.buffer.planned_obstacle_ids),
        }
        front_rgb = None if result is None or result.scene_context is None else result.scene_context.get('front_rgb')
        if isinstance(front_rgb, dict):
            details.update(
                {
                    '前视图路径': front_rgb.get('path') or front_rgb.get('image_path'),
                    '视觉来源': (
                        f'UAV {front_rgb.get("source_uav_id")}'
                        if front_rgb.get('source_uav_id') is not None
                        else 'unknown'
                    ),
                    '相机名称': front_rgb.get('camera_name'),
                }
            )
        self.record(
            'online_replanning',
            title,
            event.message,
            details,
            sim_time=event.current_time,
        )

    @staticmethod
    def _event_fallback_plan(event) -> PassagePlan | None:
        fallback_result = getattr(event, 'fallback_result', None)
        if fallback_result is not None:
            return fallback_result.selected_plan or fallback_result.deterministic_plan
        result = getattr(event, 'result', None)
        if result is not None:
            return result.deterministic_plan
        return None

    @staticmethod
    def _plan_strategy_summary_for_obstacles(
        plan: PassagePlan | None,
        obstacle_ids,
    ) -> list[str]:
        if plan is None:
            return ['暂无兜底计划信息。']
        wanted = {str(obstacle_id) for obstacle_id in obstacle_ids}
        lines = []
        for strategy in plan.obstacle_strategies:
            if wanted and str(strategy.obstacle_id) not in wanted:
                continue
            lines.append(
                f'{strategy.obstacle_id}: {passage_mode_cn(strategy.mode)} / '
                f'{target_policy_cn(strategy.target_policy)} '
                f'(observe_x={float(strategy.observe_x):.2f}, clear_x={float(strategy.clear_x):.2f})'
            )
        return lines or ['当前事件没有匹配的兜底障碍策略。']

    def record_plan(
        self,
        plan: PassagePlan,
        title: str,
        label: str,
        category: str = 'strategy_plan',
        sim_time: float | None = None,
    ) -> None:
        payload = summarize_plan(plan, label=label)
        self.record(category, title, payload['summary'], payload['details'], sim_time=sim_time)

    def record_validation(
        self,
        validation: ValidationResult,
        used_fallback: bool = False,
        sim_time: float | None = None,
    ) -> None:
        payload = summarize_validation(validation, used_fallback=used_fallback)
        self.record(
            'safety_validation',
            '安全验证结果',
            payload['summary'],
            payload['details'],
            sim_time=sim_time,
        )

    def record_stage_transition(
        self,
        current_time: float,
        from_stage: str,
        to_stage: str,
    ) -> None:
        payload = summarize_stage_transition(from_stage, to_stage)
        self.record(
            'execution_feedback',
            '执行阶段切换',
            payload['summary'],
            payload['details'],
            sim_time=current_time,
        )

    def record_adaptive_order(
        self,
        current_time: float,
        obstacle_label: str,
        order: list[int] | tuple[int, ...],
    ) -> None:
        payload = summarize_adaptive_order(obstacle_label, order)
        self.record(
            'execution_feedback',
            '自适应通行顺序',
            payload['summary'],
            payload['details'],
            sim_time=current_time,
        )

    def record_final_metrics(self, metrics: PassageRunMetrics) -> None:
        self.record(
            'execution_feedback',
            '任务执行结果',
            (
                f'任务{"成功" if metrics.success else "未成功"}，总耗时 {metrics.total_time:.2f} s，'
                f'最终阶段 {metrics.final_stage}。'
            ),
            {
                '任务是否成功': bool(metrics.success),
                '总耗时': f'{metrics.total_time:.2f} s',
                '最小机间距': _optional_fmt(metrics.min_inter_uav_distance, 'm'),
                '最小洞口净空': _optional_fmt(metrics.min_aperture_lateral_clearance, 'm'),
                '时间槽保持次数': int(metrics.temporal_gate_holds),
                '是否使用回退策略': bool(metrics.used_fallback),
                '是否发生策略修复': bool(metrics.validation_repaired),
            },
            sim_time=metrics.end_time,
        )

    def wait_for_panel_close(self) -> None:
        if self.panel is not None:
            self.panel.wait_until_closed()

    def snapshot(self) -> dict[str, Any]:
        return {
            'scenario_id': self.scenario_id,
            'event_count': len(self.events),
            'latest_event': None if not self.events else self.events[-1].to_dict(),
            'events': [event.to_dict() for event in self.events],
        }

    def _write_files(self, event: SemanticTraceEvent) -> None:
        snapshot = self.snapshot()
        self.latest_path.write_text(
            json.dumps(snapshot, ensure_ascii=False, indent=2) + '\n',
            encoding='utf-8',
        )
        with self.history_path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(event.to_dict(), ensure_ascii=False) + '\n')

    def _print_event(self, event: SemanticTraceEvent) -> None:
        time_text = '初始化' if event.sim_time is None else f't={event.sim_time:.2f}s'
        print()
        print('==================== SFSC + VLM 证据链 ====================')
        print(f'[{event.event_id:02d}] {event.title} | {time_text}')
        print(event.summary)
        for key, value in event.details.items():
            if isinstance(value, list):
                print(f'- {key}:')
                for item in value:
                    print(f'  · {item}')
            else:
                print(f'- {key}: {value}')
        print('==========================================================')

    def _recent_sfsc_events(self, limit: int = 5) -> list[dict[str, Any]]:
        categories = {
            'scene_semantics',
            'llm_input',
            'runtime_semantics',
            'online_perception',
            'online_replanning',
            'safety_validation',
            'strategy_plan',
            'llm_strategy_output',
        }
        selected = [
            event.to_dict()
            for event in reversed(self.events)
            if event.category in categories
        ]
        return list(reversed(selected[: max(0, int(limit))]))

    @staticmethod
    def _should_submit_vlm_for_frame(frame) -> bool:
        label = str(getattr(frame, 'label', '')).lower()
        skipped_labels = {
            'stage_reform',
            'stage_post_formation',
            'stage_complete',
            'final_state',
        }
        if label in skipped_labels:
            return False
        if label.startswith('stage_complete') or label.startswith('final'):
            return False
        if label.startswith('stage_reform') or label.startswith('stage_post_formation'):
            return False
        planning_prefixes = (
            'initial_scene',
            'stage_formation',
            'stage_snake',
            'online_',
            'adaptive_order_',
        )
        return label.startswith(planning_prefixes)

    @staticmethod
    def _online_replan_one_line(
        event_type: str,
        is_vlm_control: bool = False,
        contribution: dict[str, Any] | None = None,
    ) -> str:
        actor = 'VLM' if is_vlm_control else '外部模型'
        if str(event_type) == 'completed_accepted' and contribution is not None:
            changed = contribution.get('changed')
            if changed is True:
                return f'{actor} 返回的原始候选与本地兜底不同，已通过验证，等待时效性门控和安全投影后决定实际执行改动。'
            if changed is False:
                return f'{actor} 返回的原始候选与本地兜底一致或等价，已通过验证并进入闭环复核。'
        mapping = {
            'fallback_accepted': f'{actor} 尚未返回，确定性策略已先进入控制缓存。',
            'fallback_rejected': '确定性兜底策略未通过验证，需要继续保持原策略。',
            'submitted': f'新的在线 {actor} 策略请求已提交，控制端继续执行当前缓存策略。',
            'pending': f'在线 {actor} 仍在计算，控制端未等待。',
            'completed_accepted': f'外部 {actor} 策略候选已通过验证，后续由时效性门控和安全投影决定实际执行内容。',
            'completed_rejected': f'外部 {actor} 策略未通过验证，未进入控制缓存。',
        }
        return mapping.get(str(event_type), '在线重规划事件。')

    @staticmethod
    def _online_execution_admission_hint(event_type: str, is_vlm_control: bool) -> str:
        if str(event_type) == 'completed_accepted' and is_vlm_control:
            return '候选已通过格式/安全验证；是否实际进入执行，请看 execution_feedback 的时效性验证和安全投影。'
        if str(event_type) in {'fallback_accepted', 'submitted', 'pending'} and is_vlm_control:
            return '控制端继续执行当前确定性兜底/已缓存策略。'
        if str(event_type) == 'completed_rejected':
            return '候选未通过验证，未进入执行。'
        return '无额外执行准入裁决。'

    @staticmethod
    def _online_external_contribution(event) -> dict[str, Any]:
        if event.event_type not in {'completed_accepted', 'completed_rejected'}:
            return {
                'changed': None,
                'display': '等待外部模型返回',
                'summary': ['当前由确定性兜底策略保持控制连续性。'],
            }
        result = getattr(event, 'result', None)
        fallback_result = getattr(event, 'fallback_result', None)
        candidate = None if result is None else result.selected_plan
        fallback = None if fallback_result is None else (
            fallback_result.selected_plan or fallback_result.deterministic_plan
        )
        if candidate is None or fallback is None:
            return {
                'changed': None,
                'display': '无法判定',
                'summary': ['缺少 VLM 策略或兜底策略，无法进行差异比较。'],
            }
        changes = SemanticTraceRecorder._plan_diff_summary(fallback, candidate)
        changed = bool(changes)
        if bool(getattr(result, 'used_fallback', False)):
            return {
                'changed': False,
                'display': '否，外部模型失败后沿用兜底',
                'summary': ['外部模型输出未能形成可执行改写，安全管线继续使用确定性兜底策略。'],
            }
        return {
            'changed': changed,
            'display': '是，原始候选提出改写' if changed else '否，原始候选确认/复用兜底',
            'summary': changes or ['VLM 原始候选与本地兜底在通行模式、顺序和障碍策略上等价。'],
        }

    @staticmethod
    def _plan_diff_summary(reference: PassagePlan, candidate: PassagePlan) -> list[str]:
        changes: list[str] = []
        if reference.mode != candidate.mode:
            changes.append(f'全局模式: {reference.mode} -> {candidate.mode}')
        if tuple(reference.passing_order) != tuple(candidate.passing_order):
            changes.append(f'通行顺序: {list(reference.passing_order)} -> {list(candidate.passing_order)}')
        if abs(float(reference.time_slot_interval) - float(candidate.time_slot_interval)) > 1e-6:
            changes.append(
                f'时间槽间隔: {float(reference.time_slot_interval):.2f}s -> '
                f'{float(candidate.time_slot_interval):.2f}s'
            )
        ref_by_id = {strategy.obstacle_id: strategy for strategy in reference.obstacle_strategies}
        cand_by_id = {strategy.obstacle_id: strategy for strategy in candidate.obstacle_strategies}
        for obstacle_id in sorted(set(ref_by_id) | set(cand_by_id)):
            ref_strategy = ref_by_id.get(obstacle_id)
            cand_strategy = cand_by_id.get(obstacle_id)
            if ref_strategy is None:
                changes.append(f'{obstacle_id}: VLM 新增策略 {cand_strategy.mode}/{cand_strategy.target_policy}')
                continue
            if cand_strategy is None:
                changes.append(f'{obstacle_id}: VLM 删除了兜底策略')
                continue
            local_changes = []
            if ref_strategy.mode != cand_strategy.mode:
                local_changes.append(f'mode {ref_strategy.mode}->{cand_strategy.mode}')
            if ref_strategy.target_policy != cand_strategy.target_policy:
                local_changes.append(f'policy {ref_strategy.target_policy}->{cand_strategy.target_policy}')
            if abs(float(ref_strategy.observe_x) - float(cand_strategy.observe_x)) > 0.25:
                local_changes.append(f'observe_x {ref_strategy.observe_x:.2f}->{cand_strategy.observe_x:.2f}')
            if abs(float(ref_strategy.clear_x) - float(cand_strategy.clear_x)) > 0.25:
                local_changes.append(f'clear_x {ref_strategy.clear_x:.2f}->{cand_strategy.clear_x:.2f}')
            if local_changes:
                changes.append(f'{obstacle_id}: ' + ', '.join(local_changes))
        return changes

    @staticmethod
    def _build_vlm_generator(provider: str):
        provider = str(provider).lower()
        if provider == 'off':
            return None
        if provider in {'mock', 'none', 'deterministic'}:
            return MockVLMStrategyGenerator()
        if provider == 'qwen':
            return QwenDashScopeVLMStrategyGenerator()
        raise ValueError(f'unknown VLM provider: {provider}')


def _optional_fmt(value: float | None, unit: str = '') -> str:
    if value is None:
        return 'n/a'
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 'n/a'
    if not math.isfinite(number):
        return 'n/a'
    suffix = f' {unit}' if unit else ''
    return f'{number:.3f}{suffix}'


def _safe_packet_label(label: str) -> str:
    text = re.sub(r'[^A-Za-z0-9_.-]+', '_', str(label).strip())
    return text.strip('_') or 'vlm_packet'


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)
