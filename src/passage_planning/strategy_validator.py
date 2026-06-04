from __future__ import annotations

import math
from dataclasses import replace

from .structures import (
    FormationState,
    ObstacleField,
    ObstaclePassageStrategy,
    PassagePlan,
    PassageSlot,
    ValidationIssue,
    ValidationResult,
)


class StrategyValidator:
    """Deterministic safety gate for rule-based or LLM-generated strategies.

    The validator is deliberately stricter than the controller.  A strategy
    can only reach the MuJoCo execution loop after it passes structural,
    geometric, temporal, and obstacle-mode checks.
    """

    VALID_MODES = {'formation', 'snake_sequence', 'bypass'}
    SOLID_BYPASS_TARGET_POLICIES = {
        'edge_bypass',
        'side_bypass_left',
        'side_bypass_right',
        'split_by_lane_bypass',
        'overpass',
        'underpass',
        'hybrid_over_or_side',
    }
    VALID_TARGET_POLICIES = {'predictive_center_crossing', 'hold_and_reform'} | SOLID_BYPASS_TARGET_POLICIES

    def __init__(
        self,
        min_time_slot_interval: float = 0.25,
        min_clearance_x: float = 0.30,
        min_observe_distance_x: float = 0.20,
        formation_width_margin: float = 0.20,
        single_file_width_margin: float = 0.16,
        min_confidence_warning: float = 0.50,
        min_observe_travel_time: float = 0.70,
        max_observe_travel_time: float = 12.0,
        dynamic_lateral_speed_safety_factor: float = 1.15,
        high_risk_lateral_margin: float = 0.20,
        min_final_recovery_clearance_x: float = 1.00,
        min_inter_obstacle_clearance_x: float = 0.50,
        max_overpass_target_z: float = 3.40,
    ):
        self.min_time_slot_interval = float(min_time_slot_interval)
        self.min_clearance_x = float(min_clearance_x)
        self.min_observe_distance_x = float(min_observe_distance_x)
        self.formation_width_margin = float(formation_width_margin)
        self.single_file_width_margin = float(single_file_width_margin)
        self.min_confidence_warning = float(min_confidence_warning)
        self.min_observe_travel_time = float(min_observe_travel_time)
        self.max_observe_travel_time = float(max_observe_travel_time)
        self.dynamic_lateral_speed_safety_factor = float(dynamic_lateral_speed_safety_factor)
        self.high_risk_lateral_margin = float(high_risk_lateral_margin)
        self.min_final_recovery_clearance_x = float(min_final_recovery_clearance_x)
        self.min_inter_obstacle_clearance_x = float(min_inter_obstacle_clearance_x)
        self.max_overpass_target_z = float(max_overpass_target_z)

    def validate(self, plan: PassagePlan, field: ObstacleField, formation: FormationState) -> ValidationResult:
        issues = self._validate(plan, field, formation, repaired=False)
        return self._result(issues, repaired=False)

    def validate_and_repair(
        self,
        plan: PassagePlan,
        field: ObstacleField,
        formation: FormationState,
    ) -> tuple[PassagePlan, ValidationResult]:
        """Repair minor JSON-level issues, then validate the repaired plan.

        Repairs are intentionally conservative:
        - slot timing/order can be regenerated from passing_order;
        - obstacle strategies can be sorted and their clear/observe x clamped;
        - missing mode consistency is not guessed if it would change behavior.
        """
        repaired_plan = self._repair_minor_issues(plan, field, formation)
        repaired = repaired_plan != plan
        issues = self._validate(repaired_plan, field, formation, repaired=repaired)
        if repaired:
            issues.append(ValidationIssue('repair', 'plan_repaired', 'minor strategy fields were repaired deterministically'))
        return repaired_plan, self._result(issues, repaired=repaired)

    def _validate(
        self,
        plan: PassagePlan,
        field: ObstacleField,
        formation: FormationState,
        repaired: bool,
    ) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        self._validate_top_level(plan, formation, issues)
        self._validate_slots(plan, formation, issues)
        self._validate_obstacle_strategies(plan, field, formation, issues)

        if plan.mode == 'formation' and any(strategy.mode != 'formation' for strategy in plan.obstacle_strategies):
            self._error(issues, 'global_mode_conflict', 'global formation mode conflicts with obstacle-level strategy')
        if plan.mode == 'snake_sequence' and not any(
            strategy.mode == 'snake_sequence' for strategy in plan.obstacle_strategies
        ):
            self._error(issues, 'global_mode_conflict', 'global snake_sequence mode has no snake obstacle strategy')
        if plan.recover_after_last_obstacle is False:
            self._warning(issues, 'no_recovery', 'plan does not request formation recovery after the final obstacle')
        if repaired:
            self._warning(issues, 'repaired_before_validation', 'plan was repaired before final validation')

        return issues

    def _validate_top_level(self, plan: PassagePlan, formation: FormationState, issues: list[ValidationIssue]):
        if plan.mode not in self.VALID_MODES:
            self._error(issues, 'invalid_plan_mode', f'invalid global mode: {plan.mode}')
        if not self._finite(plan.confidence) or not (0.0 <= float(plan.confidence) <= 1.0):
            self._error(issues, 'invalid_confidence', 'plan confidence must be finite and inside [0, 1]')
        elif float(plan.confidence) < self.min_confidence_warning:
            self._warning(issues, 'low_confidence', 'plan confidence is low')

        if not self._finite(plan.time_slot_interval):
            self._error(issues, 'invalid_time_slot_interval', 'time_slot_interval must be finite')
        elif plan.time_slot_interval < self.min_time_slot_interval:
            self._error(issues, 'time_slot_too_dense', 'time_slot_interval is below minimum safety interval')

        expected_ids = set(range(formation.num_uavs))
        if set(plan.passing_order) != expected_ids or len(plan.passing_order) != formation.num_uavs:
            self._error(issues, 'invalid_passing_order', 'passing_order must contain each UAV exactly once')

    def _validate_slots(self, plan: PassagePlan, formation: FormationState, issues: list[ValidationIssue]):
        if len(plan.slots) != formation.num_uavs:
            self._error(issues, 'slot_count_mismatch', 'slot count must match UAV count')
            return

        slot_by_drone: dict[int, PassageSlot] = {}
        for slot in plan.slots:
            if slot.drone_id in slot_by_drone:
                self._error(issues, 'duplicate_slot_drone', f'duplicate slot for UAV {slot.drone_id}')
            slot_by_drone[int(slot.drone_id)] = slot

            if not all(self._finite(value) for value in (slot.nominal_entry_time, slot.nominal_exit_time)):
                self._error(issues, 'invalid_slot_time', f'slot times must be finite for UAV {slot.drone_id}')
                continue
            if slot.nominal_entry_time < 0.0:
                self._error(issues, 'negative_slot_time', f'slot entry time is negative for UAV {slot.drone_id}')
            if slot.nominal_exit_time <= slot.nominal_entry_time:
                self._error(issues, 'nonpositive_slot_duration', f'slot exit must be after entry for UAV {slot.drone_id}')

        previous_entry: float | None = None
        for expected_order, drone_id in enumerate(plan.passing_order):
            slot = slot_by_drone.get(int(drone_id))
            if slot is None:
                self._error(issues, 'missing_slot', f'missing slot for UAV {drone_id}')
                continue
            if slot.order_index != expected_order:
                self._error(issues, 'slot_order_mismatch', f'slot order mismatch for UAV {drone_id}')
            if previous_entry is not None:
                gap = float(slot.nominal_entry_time - previous_entry)
                if gap + 1e-9 < self.min_time_slot_interval:
                    self._error(issues, 'slot_gap_too_small', f'time gap before UAV {drone_id} is too small')
            previous_entry = float(slot.nominal_entry_time)

    def _validate_obstacle_strategies(
        self,
        plan: PassagePlan,
        field: ObstacleField,
        formation: FormationState,
        issues: list[ValidationIssue],
    ):
        if len(plan.obstacle_strategies) != field.count:
            self._error(issues, 'strategy_count_mismatch', 'strategy count must match obstacle count')
            return

        obstacle_by_id = {obstacle.obstacle_id: obstacle for obstacle in field.obstacles}
        previous_index = -1
        previous_x = -math.inf
        for strategy in plan.obstacle_strategies:
            obstacle = obstacle_by_id.get(strategy.obstacle_id)
            if obstacle is None:
                self._error(issues, 'unknown_obstacle_id', f'unknown obstacle strategy id: {strategy.obstacle_id}')
                continue

            if strategy.mode not in self.VALID_MODES:
                self._error(issues, 'invalid_strategy_mode', f'invalid strategy mode for {strategy.obstacle_id}')
            if strategy.mode == 'formation':
                self._error(
                    issues,
                    'formation_mode_not_allowed_in_obstacle_zone',
                    f'formation mode is reserved for post-obstacle recovery, not obstacle strategy: {strategy.obstacle_id}',
                )
            if strategy.target_policy not in self.VALID_TARGET_POLICIES:
                self._error(issues, 'invalid_target_policy', f'invalid target policy for {strategy.obstacle_id}')
            if strategy.obstacle_index <= previous_index:
                self._error(issues, 'obstacle_order_mismatch', 'obstacle strategies must be ordered by obstacle_index')
            if strategy.obstacle_x + 1e-9 < previous_x:
                self._error(issues, 'obstacle_x_not_monotonic', 'obstacle strategies must be ordered by x')
            previous_index = int(strategy.obstacle_index)
            previous_x = float(strategy.obstacle_x)

            self._validate_obstacle_geometry(strategy, obstacle, formation, issues)
            self._validate_obstacle_executability(strategy, obstacle, formation, issues)

        self._validate_strategy_spacing(plan, issues)

    def _validate_obstacle_geometry(
        self,
        strategy: ObstaclePassageStrategy,
        obstacle,
        formation: FormationState,
        issues: list[ValidationIssue],
    ):
        numeric_values = (
            strategy.obstacle_x,
            strategy.clear_x,
            strategy.observe_x,
            obstacle.x,
            obstacle.center_y,
            obstacle.pass_z,
        )
        if not all(self._finite(value) for value in numeric_values):
            self._error(issues, 'nonfinite_obstacle_value', f'obstacle strategy has non-finite value: {strategy.obstacle_id}')
            return

        if abs(float(strategy.obstacle_x) - float(obstacle.x)) > 0.25:
            self._error(issues, 'obstacle_x_mismatch', f'obstacle_x does not match encoded obstacle {strategy.obstacle_id}')
        if strategy.observe_x > obstacle.x - self.min_observe_distance_x:
            self._error(issues, 'observe_x_too_close', f'observe_x is too close to obstacle {strategy.obstacle_id}')
        if strategy.clear_x < obstacle.x + self.min_clearance_x:
            self._error(issues, 'clear_x_too_close', f'clear_x is too close for obstacle {strategy.obstacle_id}')
        if strategy.clear_x <= strategy.observe_x:
            self._error(issues, 'invalid_observe_clear_order', f'clear_x must be after observe_x for {strategy.obstacle_id}')

        if obstacle.function == 'solid' and strategy.mode != 'bypass':
            self._error(issues, 'solid_requires_bypass', f'solid obstacle requires bypass mode: {strategy.obstacle_id}')
        if obstacle.function == 'solid' and strategy.target_policy not in self.SOLID_BYPASS_TARGET_POLICIES:
            self._error(issues, 'solid_policy_mismatch', f'solid obstacle requires a bypass target policy: {strategy.obstacle_id}')
        if obstacle.function == 'aperture' and strategy.mode == 'bypass':
            self._warning(issues, 'aperture_bypass', f'aperture obstacle is bypassed instead of crossed: {strategy.obstacle_id}')
        if obstacle.function == 'aperture' and strategy.mode == 'snake_sequence' and strategy.target_policy != 'predictive_center_crossing':
            self._error(issues, 'aperture_snake_policy_mismatch', f'aperture snake strategy must use predictive center crossing: {strategy.obstacle_id}')
        if strategy.target_policy == 'underpass':
            if obstacle.size_z is None:
                self._error(
                    issues,
                    'underpass_geometry_unknown',
                    f'underpass requires known solid obstacle height for {strategy.obstacle_id}',
                )
                return
            bottom_clearance = float(obstacle.pass_z) - 0.5 * abs(float(obstacle.size_z or 0.0))
            required_bottom_clearance = float(formation.initial_z + formation.uav_radius_z + 0.08)
            if bottom_clearance < required_bottom_clearance:
                self._error(
                    issues,
                    'underpass_clearance_too_small',
                    f'underpass target is not geometrically feasible for {strategy.obstacle_id}',
                )
        if strategy.target_policy == 'overpass':
            if obstacle.size_z is None:
                self._error(
                    issues,
                    'overpass_geometry_unknown',
                    f'overpass requires known solid obstacle height for {strategy.obstacle_id}',
                )
                return
            top_height = float(obstacle.pass_z) + 0.5 * abs(float(obstacle.size_z or 0.0))
            required_overpass_z = top_height + float(formation.uav_radius_z + 0.08)
            if required_overpass_z > self.max_overpass_target_z:
                self._error(
                    issues,
                    'overpass_target_too_high',
                    f'overpass target is too high for {strategy.obstacle_id}',
                )

        aperture_width = float(obstacle.aperture_width or 0.0)
        if strategy.mode == 'snake_sequence':
            required = formation.single_file_width + self.single_file_width_margin
            if aperture_width < required:
                self._error(issues, 'aperture_too_narrow_for_snake', f'aperture too narrow for snake sequence: {strategy.obstacle_id}')
            if strategy.target_policy != 'predictive_center_crossing':
                self._error(issues, 'snake_policy_mismatch', f'snake strategy must use predictive center crossing: {strategy.obstacle_id}')
        if strategy.mode == 'formation':
            required = formation.width_y + self.formation_width_margin
            if aperture_width < required:
                self._error(issues, 'aperture_too_narrow_for_formation', f'aperture too narrow for formation mode: {strategy.obstacle_id}')

        if obstacle.risk_level == 'high' and strategy.mode == 'formation':
            self._warning(issues, 'high_risk_formation_passage', f'high-risk aperture uses formation mode: {strategy.obstacle_id}')

    def _validate_obstacle_executability(
        self,
        strategy: ObstaclePassageStrategy,
        obstacle,
        formation: FormationState,
        issues: list[ValidationIssue],
    ):
        observe_distance = float(strategy.obstacle_x - strategy.observe_x)
        if observe_distance <= 0.0:
            return

        travel_time = observe_distance / max(float(formation.nominal_speed_x), 1e-6)
        if travel_time < self.min_observe_travel_time:
            self._error(
                issues,
                'observe_window_too_short',
                f'observe window is too short for {strategy.obstacle_id}',
            )
        elif travel_time > self.max_observe_travel_time:
            self._warning(
                issues,
                'observe_window_too_long',
                f'observe window is very long and may disperse the queue before {strategy.obstacle_id}',
            )

        if strategy.mode not in {'snake_sequence', 'formation'} or obstacle.function != 'aperture':
            return

        aperture_width = float(obstacle.aperture_width or 0.0)
        if strategy.mode == 'snake_sequence':
            static_margin = 0.5 * aperture_width - formation.uav_radius_xy
        else:
            static_margin = 0.5 * (aperture_width - formation.width_y)

        if static_margin < 0.0:
            return
        if static_margin < self.high_risk_lateral_margin:
            self._warning(
                issues,
                'low_lateral_aperture_margin',
                f'lateral aperture margin is small for {strategy.obstacle_id}',
            )

        obstacle_peak_speed = self._obstacle_peak_motion_speed(obstacle)
        required_lateral_speed = obstacle_peak_speed * self.dynamic_lateral_speed_safety_factor
        lateral_capacity = float(formation.lateral_tracking_speed)
        if required_lateral_speed > lateral_capacity + 1e-9:
            self._error(
                issues,
                'dynamic_aperture_too_fast',
                f'dynamic aperture motion exceeds lateral tracking capacity for {strategy.obstacle_id}',
            )
        elif required_lateral_speed > 0.80 * lateral_capacity:
            self._warning(
                issues,
                'dynamic_aperture_near_lateral_limit',
                f'dynamic aperture motion is near lateral tracking limit for {strategy.obstacle_id}',
            )

        max_initial_offset = max((abs(float(y) - float(obstacle.center_y)) for y in formation.original_y), default=0.0)
        reachable_lateral_shift = lateral_capacity * travel_time
        required_shift = max(0.0, max_initial_offset - static_margin)
        if required_shift > reachable_lateral_shift + 1e-9:
            self._error(
                issues,
                'aperture_center_not_reachable',
                f'UAV lateral shift is not reachable before {strategy.obstacle_id}',
            )
        elif required_shift > 0.85 * reachable_lateral_shift and required_shift > 1e-9:
            self._warning(
                issues,
                'aperture_center_near_reach_limit',
                f'UAV lateral shift is near reach limit before {strategy.obstacle_id}',
            )

    def _validate_strategy_spacing(self, plan: PassagePlan, issues: list[ValidationIssue]):
        strategies = tuple(plan.obstacle_strategies)
        if len(strategies) < 1:
            return

        for previous, current in zip(strategies, strategies[1:]):
            gap = float(current.observe_x - previous.clear_x)
            if gap < -1e-9:
                self._warning(
                    issues,
                    'overlapping_obstacle_windows',
                    f'observe window for {current.obstacle_id} starts before {previous.obstacle_id} clears',
                )
            elif gap < self.min_inter_obstacle_clearance_x:
                self._warning(
                    issues,
                    'tight_inter_obstacle_window',
                    f'obstacle windows are tight between {previous.obstacle_id} and {current.obstacle_id}',
                )

        final_strategy = strategies[-1]
        final_clearance = float(final_strategy.clear_x - final_strategy.obstacle_x)
        if plan.recover_after_last_obstacle and final_clearance < self.min_final_recovery_clearance_x:
            self._error(
                issues,
                'final_recovery_clearance_too_short',
                'final obstacle clearance is too short before recovery',
            )

    @staticmethod
    def _obstacle_peak_motion_speed(obstacle) -> float:
        return abs(float(obstacle.motion_amplitude) * float(obstacle.motion_omega))

    def _repair_minor_issues(self, plan: PassagePlan, field: ObstacleField, formation: FormationState) -> PassagePlan:
        time_slot_interval = max(float(plan.time_slot_interval), self.min_time_slot_interval)
        passing_order = tuple(int(drone_id) for drone_id in plan.passing_order)
        if set(passing_order) != set(range(formation.num_uavs)) or len(passing_order) != formation.num_uavs:
            passing_order = tuple(range(formation.num_uavs))

        slots = tuple(
            PassageSlot(
                drone_id=drone_id,
                order_index=order_index,
                nominal_entry_time=float(order_index * time_slot_interval),
                nominal_exit_time=float((order_index + 1) * time_slot_interval),
            )
            for order_index, drone_id in enumerate(passing_order)
        )

        obstacle_by_id = {obstacle.obstacle_id: obstacle for obstacle in field.obstacles}
        strategies = []
        for strategy in plan.obstacle_strategies:
            obstacle = obstacle_by_id.get(strategy.obstacle_id)
            if obstacle is None:
                strategies.append(strategy)
                continue
            strategies.append(
                replace(
                    strategy,
                    obstacle_index=int(max(0, min(strategy.obstacle_index, field.count - 1))),
                    obstacle_x=float(obstacle.x),
                    clear_x=float(max(strategy.clear_x, obstacle.x + self.min_clearance_x)),
                    observe_x=float(min(strategy.observe_x, obstacle.x - self.min_observe_distance_x)),
                )
            )
        strategies = tuple(sorted(strategies, key=lambda item: (item.obstacle_index, item.obstacle_x)))

        return replace(
            plan,
            passing_order=passing_order,
            slots=slots,
            obstacle_strategies=strategies,
            time_slot_interval=time_slot_interval,
        )

    @staticmethod
    def _finite(value: float) -> bool:
        return math.isfinite(float(value))

    @staticmethod
    def _error(issues: list[ValidationIssue], code: str, message: str):
        issues.append(ValidationIssue('error', code, message))

    @staticmethod
    def _warning(issues: list[ValidationIssue], code: str, message: str):
        issues.append(ValidationIssue('warning', code, message))

    @staticmethod
    def _result(issues: list[ValidationIssue], repaired: bool) -> ValidationResult:
        errors = tuple(issue for issue in issues if issue.severity == 'error')
        return ValidationResult(
            valid=not errors,
            repaired=bool(repaired),
            issues=tuple(issues),
            messages=tuple(issue.message for issue in issues),
        )
