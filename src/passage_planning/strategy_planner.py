from __future__ import annotations

from .structures import (
    FormationState,
    MissionPreference,
    ObstacleField,
    ObstaclePassageStrategy,
    PassagePlan,
    PassageSlot,
    PassabilityReport,
)


class SpatioTemporalStrategyPlanner:
    """Creates deterministic passage strategies from passability reports."""

    def __init__(
        self,
        gate_observe_distance_x: float,
        gate_pass_clear_x: float,
        snake_spacing_x: float,
        nominal_slot_interval: float | None = None,
    ):
        self.gate_observe_distance_x = float(gate_observe_distance_x)
        self.gate_pass_clear_x = float(gate_pass_clear_x)
        self.snake_spacing_x = float(snake_spacing_x)
        self.nominal_slot_interval = nominal_slot_interval

    def plan(
        self,
        field: ObstacleField,
        formation: FormationState,
        mission: MissionPreference,
        reports: tuple[PassabilityReport, ...],
    ) -> PassagePlan:
        report_by_id = {report.obstacle_id: report for report in reports}
        modes = [report_by_id[obstacle.obstacle_id].recommended_mode for obstacle in field.obstacles]

        if any(mode == 'snake_sequence' for mode in modes):
            global_mode = 'snake_sequence'
        elif any(mode == 'bypass' for mode in modes):
            global_mode = 'bypass'
        else:
            global_mode = 'formation'

        passing_order = self._center_first_order(formation.original_y)
        slot_interval = self._slot_interval(formation)
        slots = tuple(
            PassageSlot(
                drone_id=drone_id,
                order_index=order_index,
                nominal_entry_time=order_index * slot_interval,
                nominal_exit_time=(order_index + 1) * slot_interval,
            )
            for order_index, drone_id in enumerate(passing_order)
        )

        obstacle_strategies = tuple(
            ObstaclePassageStrategy(
                obstacle_id=obstacle.obstacle_id,
                obstacle_index=index,
                mode=report_by_id[obstacle.obstacle_id].recommended_mode,
                target_policy='predictive_center_crossing' if obstacle.function == 'aperture' else 'edge_bypass',
                obstacle_x=float(obstacle.x),
                clear_x=float(obstacle.x + self._clearance_after_obstacle(obstacle)),
                observe_x=float(obstacle.x - self.gate_observe_distance_x),
            )
            for index, obstacle in enumerate(field.obstacles)
        )

        confidence = min((report.confidence for report in reports), default=0.0)
        return PassagePlan(
            plan_id='deterministic_dynamic_aperture_sequence',
            mode=global_mode,
            passing_order=passing_order,
            slots=slots,
            obstacle_strategies=obstacle_strategies,
            recover_after_last_obstacle=bool(mission.recover_after_last_obstacle),
            time_slot_interval=slot_interval,
            confidence=float(confidence),
        )

    @staticmethod
    def _center_first_order(original_y: tuple[float, ...]) -> tuple[int, ...]:
        return tuple(sorted(range(len(original_y)), key=lambda idx: (abs(original_y[idx]), idx)))

    def _slot_interval(self, formation: FormationState) -> float:
        if self.nominal_slot_interval is not None:
            return float(self.nominal_slot_interval)
        return float(max(0.35, self.snake_spacing_x / max(formation.nominal_speed_x, 1e-6)))

    def _clearance_after_obstacle(self, obstacle) -> float:
        if obstacle.function == 'solid' and obstacle.size_x is not None:
            return float(0.5 * abs(float(obstacle.size_x)) + self.gate_pass_clear_x)
        return float(self.gate_pass_clear_x)
