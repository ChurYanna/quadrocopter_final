from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .metrics import PassageRunMetrics
from .semantic_presenter import (
    summarize_adaptive_order,
    summarize_llm_input,
    summarize_plan,
    summarize_scene,
    summarize_stage_transition,
    summarize_validation,
)
from .semantic_panel import SemanticTracePanel
from .structures import FormationState, MissionPreference, ObstacleField, PassagePlan, ValidationResult


@dataclass(frozen=True)
class SemanticTraceEvent:
    """One human-readable semantic event for LLM passage visualization."""

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
    """Records and displays the semantic information flow around LLM planning.

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
    ):
        self.scenario_id = str(scenario_id)
        self.log_dir = Path(log_dir)
        self.enabled = bool(enabled)
        self.console = bool(console)
        self.panel_enabled = bool(panel_enabled)
        self.panel: SemanticTracePanel | None = None
        self.events: list[SemanticTraceEvent] = []
        self.latest_path = self.log_dir / 'semantic_trace_latest.json'
        self.history_path = self.log_dir / 'semantic_trace_history.jsonl'
        if self.enabled:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            if reset:
                self.latest_path.write_text('{}\n', encoding='utf-8')
                self.history_path.write_text('', encoding='utf-8')
            if self.panel_enabled:
                panel = SemanticTracePanel(title=f'SV-STCP 语义证据链 - {self.scenario_id}')
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
            '场景语义提取',
            payload['summary'],
            payload['details'],
            sim_time=sim_time,
        )

    def record_llm_input(self, scene_context: dict[str, Any], sim_time: float | None = None) -> None:
        payload = summarize_llm_input(scene_context)
        self.record(
            'llm_input',
            '输入给 LLM 的语义信息',
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
            '底层执行器识别的实体绕行语义',
            (
                f'检测到 {len(solid_specs)} 个实体障碍；它们已经进入 LLM 高层语义输入，'
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
            '有限视野语义感知',
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
            '在线 LLM 局部重规划',
            summary,
            details,
            sim_time=result.current_time,
        )

    def record_async_online_replan_event(self, event) -> None:
        if event.event_type == 'not_triggered':
            return
        title_by_type = {
            'fallback_accepted': '在线规则兜底策略',
            'fallback_rejected': '在线规则兜底失败',
            'submitted': '异步 LLM 请求已提交',
            'pending': '异步 LLM 计算中',
            'completed_accepted': '异步 LLM 策略已接收',
            'completed_rejected': '异步 LLM 策略被拒绝',
        }
        title = title_by_type.get(event.event_type, '异步在线重规划事件')
        result = event.result or event.fallback_result
        selected_plan = None if result is None else result.selected_plan
        validation = None if result is None else result.validation
        details = {
            '事件类型': event.event_type,
            '障碍集合': list(event.obstacle_ids),
            'LLM耗时': None if event.latency_s is None else f'{event.latency_s:.2f} s',
            '策略来源': None if selected_plan is None else selected_plan.source,
            '计划编号': None if selected_plan is None else selected_plan.plan_id,
            '是否使用回退': None if result is None else bool(result.used_fallback),
            '验证通过': None if validation is None else bool(validation.valid),
            '当前缓存覆盖': [] if result is None or result.buffer is None else sorted(result.buffer.planned_obstacle_ids),
        }
        self.record(
            'online_replanning',
            title,
            event.message,
            details,
            sim_time=event.current_time,
        )

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
        print('==================== LLM 语义证据链 ====================')
        print(f'[{event.event_id:02d}] {event.title} | {time_text}')
        print(event.summary)
        for key, value in event.details.items():
            if isinstance(value, list):
                print(f'- {key}:')
                for item in value:
                    print(f'  · {item}')
            else:
                print(f'- {key}: {value}')
        print('=========================================================')


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
