from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .structures import PassagePlan, ValidationResult


@dataclass(frozen=True)
class StageTransition:
    """One execution-stage transition recorded during a passage run."""

    time: float
    from_stage: str
    to_stage: str

    def to_dict(self) -> dict[str, Any]:
        return {
            'time': float(self.time),
            'from_stage': self.from_stage,
            'to_stage': self.to_stage,
        }


@dataclass(frozen=True)
class PassageRunMetrics:
    """Serializable summary of one multi-UAV passage experiment."""

    scenario_id: str
    plan_id: str
    plan_source: str
    plan_mode: str
    uav_count: int
    obstacle_count: int
    final_target_x: float
    start_time: float
    end_time: float
    final_stage: str
    success: bool
    cleared_last_obstacle: bool
    used_fallback: bool
    validation_valid: bool
    validation_repaired: bool
    validation_issue_codes: tuple[str, ...] = field(default_factory=tuple)
    validation_messages: tuple[str, ...] = field(default_factory=tuple)
    stage_transitions: tuple[StageTransition, ...] = field(default_factory=tuple)
    min_inter_uav_distance: float | None = None
    min_aperture_lateral_clearance: float | None = None
    leader_final_x: float | None = None
    total_path_length: float | None = None
    temporal_gate_holds: int = 0

    @property
    def total_time(self) -> float:
        return float(max(0.0, self.end_time - self.start_time))

    @property
    def formation_recovery_time(self) -> float | None:
        reform_time = self._first_transition_time_to('reform')
        post_time = self._first_transition_time_to('post_formation')
        if reform_time is None or post_time is None:
            return None
        return float(max(0.0, post_time - reform_time))

    def to_dict(self) -> dict[str, Any]:
        return {
            'scenario_id': self.scenario_id,
            'plan_id': self.plan_id,
            'plan_source': self.plan_source,
            'plan_mode': self.plan_mode,
            'uav_count': int(self.uav_count),
            'obstacle_count': int(self.obstacle_count),
            'final_target_x': float(self.final_target_x),
            'start_time': float(self.start_time),
            'end_time': float(self.end_time),
            'total_time': self.total_time,
            'final_stage': self.final_stage,
            'success': bool(self.success),
            'cleared_last_obstacle': bool(self.cleared_last_obstacle),
            'used_fallback': bool(self.used_fallback),
            'validation_valid': bool(self.validation_valid),
            'validation_repaired': bool(self.validation_repaired),
            'validation_issue_codes': list(self.validation_issue_codes),
            'validation_messages': list(self.validation_messages),
            'stage_transitions': [transition.to_dict() for transition in self.stage_transitions],
            'min_inter_uav_distance': self._optional_float(self.min_inter_uav_distance),
            'min_aperture_lateral_clearance': self._optional_float(self.min_aperture_lateral_clearance),
            'leader_final_x': self._optional_float(self.leader_final_x),
            'total_path_length': self._optional_float(self.total_path_length),
            'formation_recovery_time': self._optional_float(self.formation_recovery_time),
            'temporal_gate_holds': int(self.temporal_gate_holds),
        }

    def summary_line(self) -> str:
        return (
            f'[METRICS] scenario={self.scenario_id} success={self.success} '
            f'total_time={self.total_time:.2f}s final_stage={self.final_stage} '
            f'min_uav_dist={self._format_optional(self.min_inter_uav_distance)} '
            f'min_aperture_clearance={self._format_optional(self.min_aperture_lateral_clearance)} '
            f'temporal_holds={self.temporal_gate_holds} fallback={self.used_fallback} '
            f'repaired={self.validation_repaired}'
        )

    def _first_transition_time_to(self, stage: str) -> float | None:
        for transition in self.stage_transitions:
            if transition.to_stage == stage:
                return float(transition.time)
        return None

    @staticmethod
    def _optional_float(value: float | None) -> float | None:
        if value is None:
            return None
        if not math.isfinite(float(value)):
            return None
        return float(value)

    @staticmethod
    def _format_optional(value: float | None) -> str:
        if value is None or not math.isfinite(float(value)):
            return 'n/a'
        return f'{float(value):.3f}'


class PassageRunMetricsRecorder:
    """Online recorder for viewer and batch passage experiments.

    The recorder is intentionally simulator-agnostic.  Control loops feed it
    positions, stage transitions, and optional aperture snapshots; it returns a
    compact summary that can later be saved as JSON or aggregated into tables.
    """

    def __init__(
        self,
        scenario_id: str,
        plan: PassagePlan,
        uav_count: int,
        obstacle_count: int,
        final_target_x: float,
        validation_result: ValidationResult | None = None,
        used_fallback: bool = False,
        start_time: float = 0.0,
    ):
        self.scenario_id = str(scenario_id)
        self.plan = plan
        self.uav_count = int(uav_count)
        self.obstacle_count = int(obstacle_count)
        self.final_target_x = float(final_target_x)
        self.validation_result = validation_result
        self.used_fallback = bool(used_fallback)
        self.start_time = float(start_time)

        self._stage_transitions: list[StageTransition] = []
        self._min_inter_uav_distance = math.inf
        self._min_aperture_lateral_clearance = math.inf
        self._last_positions: list[np.ndarray] | None = None
        self._path_lengths = [0.0 for _ in range(self.uav_count)]
        self._end_time = self.start_time
        self._final_stage = 'unknown'
        self._success = False
        self._cleared_last_obstacle = False
        self._leader_final_x: float | None = None
        self._temporal_gate_holds = 0

    def mark_stage_transition(self, current_time: float, from_stage: str, to_stage: str) -> None:
        if from_stage == to_stage:
            return
        self._stage_transitions.append(
            StageTransition(
                time=float(current_time),
                from_stage=str(from_stage),
                to_stage=str(to_stage),
            )
        )

    def update_positions(self, current_time: float, positions: list[np.ndarray] | tuple[np.ndarray, ...]) -> None:
        self._end_time = float(current_time)
        current_positions = [np.asarray(position, dtype=float).copy() for position in positions]
        if len(current_positions) != self.uav_count:
            raise ValueError(f'expected {self.uav_count} UAV positions, got {len(current_positions)}')

        for first in range(len(current_positions)):
            for second in range(first + 1, len(current_positions)):
                distance = float(np.linalg.norm(current_positions[first] - current_positions[second]))
                self._min_inter_uav_distance = min(self._min_inter_uav_distance, distance)

        if self._last_positions is not None:
            for idx, position in enumerate(current_positions):
                step_distance = float(np.linalg.norm(position - self._last_positions[idx]))
                if math.isfinite(step_distance):
                    self._path_lengths[idx] += step_distance
        self._last_positions = current_positions

    def update_aperture_clearance(
        self,
        current_time: float,
        positions: list[np.ndarray] | tuple[np.ndarray, ...],
        aperture_snapshots: list[dict[str, Any]] | tuple[dict[str, Any], ...],
        uav_radius_xy: float,
        x_window: float,
    ) -> None:
        """Track the smallest lateral aperture margin for UAVs near a gate.

        The metric is positive when the UAV center stays inside the aperture
        with the requested radius margin, and negative when it violates the
        lateral gate clearance.  It is a lightweight proxy that works for the
        current dynamic-gate experiments and can be extended for rings/solids.
        """
        self._end_time = float(current_time)
        radius = float(uav_radius_xy)
        window = float(max(0.0, x_window))
        for position in positions:
            pos = np.asarray(position, dtype=float)
            for snapshot in aperture_snapshots:
                obstacle_x = float(snapshot['x'])
                if abs(float(pos[0]) - obstacle_x) > window:
                    continue
                width = float(snapshot['aperture_width'])
                center_y = float(snapshot['center_y'])
                lateral_margin = 0.5 * width - abs(float(pos[1]) - center_y) - radius
                if math.isfinite(lateral_margin):
                    self._min_aperture_lateral_clearance = min(
                        self._min_aperture_lateral_clearance,
                        float(lateral_margin),
                    )

    def record_temporal_gate_hold(self) -> None:
        self._temporal_gate_holds += 1

    def finish(
        self,
        current_time: float,
        final_stage: str,
        leader_final: np.ndarray,
        final_positions: list[np.ndarray] | tuple[np.ndarray, ...],
        cleared_last_obstacle: bool,
        success: bool,
    ) -> PassageRunMetrics:
        self.update_positions(current_time, final_positions)
        self._final_stage = str(final_stage)
        self._leader_final_x = float(np.asarray(leader_final, dtype=float)[0])
        self._cleared_last_obstacle = bool(cleared_last_obstacle)
        self._success = bool(success)
        return self.snapshot()

    def snapshot(self) -> PassageRunMetrics:
        validation = self.validation_result
        issue_codes = tuple(issue.code for issue in validation.issues) if validation is not None else tuple()
        messages = tuple(validation.messages) if validation is not None else tuple()
        return PassageRunMetrics(
            scenario_id=self.scenario_id,
            plan_id=self.plan.plan_id,
            plan_source=self.plan.source,
            plan_mode=self.plan.mode,
            uav_count=self.uav_count,
            obstacle_count=self.obstacle_count,
            final_target_x=self.final_target_x,
            start_time=self.start_time,
            end_time=self._end_time,
            final_stage=self._final_stage,
            success=self._success,
            cleared_last_obstacle=self._cleared_last_obstacle,
            used_fallback=self.used_fallback,
            validation_valid=bool(validation.valid) if validation is not None else False,
            validation_repaired=bool(validation.repaired) if validation is not None else False,
            validation_issue_codes=issue_codes,
            validation_messages=messages,
            stage_transitions=tuple(self._stage_transitions),
            min_inter_uav_distance=self._finite_or_none(self._min_inter_uav_distance),
            min_aperture_lateral_clearance=self._finite_or_none(self._min_aperture_lateral_clearance),
            leader_final_x=self._leader_final_x,
            total_path_length=float(sum(self._path_lengths)),
            temporal_gate_holds=int(self._temporal_gate_holds),
        )

    @staticmethod
    def _finite_or_none(value: float) -> float | None:
        if not math.isfinite(float(value)):
            return None
        return float(value)
