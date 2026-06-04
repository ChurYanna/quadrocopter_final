import unittest
from dataclasses import replace

import mujoco
import numpy as np

from src.passage_planning import (
    FormationState,
    MissionPreference,
    ObstacleFieldEncoder,
    PassabilityEvaluator,
    PassageSlot,
    SpatioTemporalStrategyPlanner,
    StrategyExecutionHelper,
    StrategyValidator,
    ObstacleDescriptor,
    ObstacleField,
    passage_plan_from_json,
    passage_plan_to_dict,
    passage_plan_to_json,
)
from tests.dynamic_gate_common import DynamicGateSnakeBase


class ThreeGatePlanningFixture(DynamicGateSnakeBase):
    FINAL_TARGET_X = 25.0
    MAX_TOTAL_TIME = 130.0
    GATE_OBSERVE_DISTANCE_X = 5.2
    GATE_PASS_CLEAR_X = 1.25
    POST_REFORM_CRUISE_X = 2.0

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

    SNAKE_SPACING_X = 0.86
    SNAKE_MAX_EXTRA_GAP_X = 1.25
    SNAKE_MAX_SPEED_X = 1.50
    REACTIVE_Y_KP = 1.90
    MAX_CMD_SPEED_XY = 3.00
    CMD_SLEW_RATE_XY = 4.00
    MAX_CMD_SPEED_Z = 0.80
    CMD_SLEW_RATE_Z = 1.35
    HEIGHT_HOLD_KP = 1.65
    HEIGHT_HOLD_DAMPING = 1.25


class TestPassageStrategyFramework(unittest.TestCase):
    """Fast non-viewer checks for the deterministic passage-planning layer."""

    def _build_strategy_context(self):
        demo = ThreeGatePlanningFixture()
        model = mujoco.MjModel.from_xml_path(demo.SCENE)
        gate_specs = demo._build_gate_runtime_specs(model)
        demo._update_dynamic_gates(model, gate_specs, 0.0)

        original_y = (0.0, 1.0, -1.0, 2.0, -2.0)
        formation = FormationState(
            num_uavs=demo.COUNT,
            original_y=original_y,
            initial_z=demo.INITIAL_Z,
            uav_radius_xy=demo.UAV_RADIUS_XY,
            uav_radius_z=demo.UAV_RADIUS_Z,
            nominal_speed_x=demo.SNAKE_MAX_SPEED_X,
            max_speed_xy=demo.MAX_CMD_SPEED_XY,
            max_speed_z=demo.MAX_CMD_SPEED_Z,
            max_lateral_speed=demo.MAX_CMD_SPEED_XY,
        )
        mission = MissionPreference(allow_disband=True, recover_after_last_obstacle=True)

        field = ObstacleFieldEncoder.encode_dynamic_apertures(
            gate_specs,
            aperture_width_fn=lambda gate_spec: demo._gate_aperture_width(model, gate_spec),
            formation=formation,
        )
        reports = PassabilityEvaluator().evaluate_field(field, formation, mission)
        plan = SpatioTemporalStrategyPlanner(
            gate_observe_distance_x=demo.GATE_OBSERVE_DISTANCE_X,
            gate_pass_clear_x=demo.GATE_PASS_CLEAR_X,
            snake_spacing_x=demo.SNAKE_SPACING_X,
        ).plan(field, formation, mission, reports)
        return demo, field, reports, plan, formation

    def test_three_gate_strategy_is_valid_and_executable(self):
        demo, field, reports, plan, formation = self._build_strategy_context()
        validation = StrategyValidator().validate(plan, field, formation)
        executor = StrategyExecutionHelper(plan)

        self.assertEqual(field.count, 3)
        self.assertTrue(validation.valid, msg=validation.messages)
        self.assertEqual(plan.mode, 'snake_sequence')
        self.assertEqual(plan.passing_order, (0, 1, 2, 3, 4))
        self.assertTrue(all(report.recommended_mode == 'snake_sequence' for report in reports))
        self.assertEqual(executor.active_obstacle_index_for_position([0.0, 0.0, demo.INITIAL_Z]), 0)
        self.assertEqual(executor.active_obstacle_index_for_position([7.5, 0.0, demo.INITIAL_Z]), 1)
        self.assertEqual(executor.active_obstacle_index_for_position([14.5, 0.0, demo.INITIAL_Z]), 2)

    def test_strategy_json_round_trip_keeps_plan_valid(self):
        _, field, _, plan, formation = self._build_strategy_context()

        payload = passage_plan_to_dict(plan)
        self.assertEqual(payload['version'], '1.0')
        self.assertEqual(payload['mode'], 'snake_sequence')
        self.assertEqual(payload['passing_order'], [0, 1, 2, 3, 4])

        loaded_plan = passage_plan_from_json(passage_plan_to_json(plan))
        validation = StrategyValidator().validate(loaded_plan, field, formation)
        executor = StrategyExecutionHelper(loaded_plan)

        self.assertTrue(validation.valid, msg=validation.messages)
        self.assertEqual(loaded_plan.passing_order, plan.passing_order)
        self.assertEqual(loaded_plan.obstacle_strategies[0].obstacle_id, 'gate1')
        self.assertEqual(executor.first_obstacle_observe_x(), plan.obstacle_strategies[0].observe_x)

    def test_validator_rejects_unsafe_llm_like_strategy(self):
        _, field, _, plan, formation = self._build_strategy_context()
        bad_strategy = replace(
            plan.obstacle_strategies[0],
            mode='formation',
            clear_x=field.obstacles[0].x + 0.05,
            observe_x=field.obstacles[0].x,
        )
        bad_plan = replace(
            plan,
            mode='formation',
            passing_order=(0, 1, 1, 3, 4),
            time_slot_interval=0.05,
            obstacle_strategies=(bad_strategy,) + plan.obstacle_strategies[1:],
        )

        validation = StrategyValidator().validate(bad_plan, field, formation)
        codes = {issue.code for issue in validation.errors}

        self.assertFalse(validation.valid)
        self.assertIn('invalid_passing_order', codes)
        self.assertIn('time_slot_too_dense', codes)
        self.assertIn('aperture_too_narrow_for_formation', codes)
        self.assertIn('clear_x_too_close', codes)

    def test_validator_repairs_minor_slot_timing_issues(self):
        _, field, _, plan, formation = self._build_strategy_context()
        bad_slots = tuple(
            PassageSlot(
                drone_id=slot.drone_id,
                order_index=slot.order_index,
                nominal_entry_time=0.0,
                nominal_exit_time=0.1,
            )
            for slot in plan.slots
        )
        repairable_plan = replace(plan, time_slot_interval=0.05, slots=bad_slots)

        repaired_plan, validation = StrategyValidator().validate_and_repair(
            repairable_plan,
            field,
            formation,
        )

        self.assertTrue(validation.valid, msg=validation.messages)
        self.assertTrue(validation.repaired)
        self.assertGreaterEqual(repaired_plan.time_slot_interval, 0.25)
        self.assertEqual(
            [slot.nominal_entry_time for slot in repaired_plan.slots],
            [0.0, 0.25, 0.5, 0.75, 1.0],
        )

    def test_validator_rejects_dynamic_aperture_beyond_tracking_capacity(self):
        _, field, _, plan, formation = self._build_strategy_context()
        constrained_formation = replace(
            formation,
            max_speed_xy=0.4,
            max_lateral_speed=0.4,
        )

        validation = StrategyValidator().validate(plan, field, constrained_formation)
        codes = {issue.code for issue in validation.errors}

        self.assertFalse(validation.valid)
        self.assertIn('dynamic_aperture_too_fast', codes)

    def test_validator_rejects_unreachable_short_observe_window(self):
        _, field, _, plan, formation = self._build_strategy_context()
        short_observe_strategy = replace(
            plan.obstacle_strategies[0],
            observe_x=plan.obstacle_strategies[0].obstacle_x - 0.05,
        )
        risky_plan = replace(
            plan,
            obstacle_strategies=(short_observe_strategy,) + plan.obstacle_strategies[1:],
        )

        validation = StrategyValidator().validate(risky_plan, field, formation)
        codes = {issue.code for issue in validation.errors}

        self.assertFalse(validation.valid)
        self.assertIn('observe_x_too_close', codes)
        self.assertIn('observe_window_too_short', codes)

    def test_solid_obstacle_plans_bypass_and_helper_selects_stable_side(self):
        formation = FormationState(
            num_uavs=1,
            original_y=(-1.0,),
            initial_z=0.3,
            uav_radius_xy=0.38,
            uav_radius_z=0.22,
            nominal_speed_x=1.0,
            max_speed_xy=2.0,
            max_speed_z=0.8,
            max_lateral_speed=2.0,
        )
        field = ObstacleField((
            ObstacleDescriptor(
                obstacle_id='box1',
                obstacle_type='box',
                function='solid',
                x=4.0,
                center_y=0.0,
                pass_z=0.8,
                size_x=1.0,
                size_y=2.0,
                size_z=1.0,
            ),
        ))
        reports = PassabilityEvaluator().evaluate_field(field, formation, MissionPreference())
        plan = SpatioTemporalStrategyPlanner(
            gate_observe_distance_x=2.0,
            gate_pass_clear_x=1.0,
            snake_spacing_x=0.8,
        ).plan(field, formation, MissionPreference(), reports)
        validation = StrategyValidator().validate(plan, field, formation)
        target = StrategyExecutionHelper.bypass_target(
            position=np.array([2.0, -1.0, 0.3]),
            obstacle_center=np.array([4.0, 0.0, 0.8]),
            obstacle_size=np.array([1.0, 2.0, 1.0]),
            original_y=-1.0,
            clearance_xy=0.6,
            clearance_z=0.4,
            route_hint='left',
        )

        self.assertTrue(validation.valid, msg=validation.messages)
        self.assertEqual(reports[0].recommended_mode, 'bypass')
        self.assertEqual(plan.mode, 'bypass')
        self.assertEqual(plan.obstacle_strategies[0].target_policy, 'edge_bypass')
        self.assertLess(target[1], -1.0)

    def test_solid_route_family_policies_are_validated(self):
        formation = FormationState(
            num_uavs=1,
            original_y=(0.0,),
            initial_z=0.3,
            uav_radius_xy=0.38,
            uav_radius_z=0.22,
            nominal_speed_x=1.0,
            max_speed_xy=2.0,
            max_speed_z=0.8,
            max_lateral_speed=2.0,
        )
        field = ObstacleField((
            ObstacleDescriptor(
                obstacle_id='beam1',
                obstacle_type='elevated_beam',
                function='solid',
                x=4.0,
                center_y=0.0,
                pass_z=2.0,
                size_x=1.0,
                size_y=3.0,
                size_z=1.0,
            ),
        ))
        reports = PassabilityEvaluator().evaluate_field(field, formation, MissionPreference())
        plan = SpatioTemporalStrategyPlanner(
            gate_observe_distance_x=2.0,
            gate_pass_clear_x=1.0,
            snake_spacing_x=0.8,
        ).plan(field, formation, MissionPreference(), reports)

        overpass_plan = replace(
            plan,
            obstacle_strategies=(
                replace(plan.obstacle_strategies[0], target_policy='overpass'),
            ),
        )
        underpass_plan = replace(
            plan,
            obstacle_strategies=(
                replace(plan.obstacle_strategies[0], target_policy='underpass'),
            ),
        )
        under_target = StrategyExecutionHelper.bypass_target(
            position=np.array([2.0, 0.0, 0.9]),
            obstacle_center=np.array([4.0, 0.0, 2.0]),
            obstacle_size=np.array([1.0, 3.0, 1.0]),
            original_y=0.0,
            clearance_xy=0.6,
            clearance_z=0.4,
            route_hint='under',
        )

        self.assertTrue(StrategyValidator().validate(overpass_plan, field, formation).valid)
        self.assertTrue(StrategyValidator().validate(underpass_plan, field, formation).valid)
        self.assertLess(under_target[2], 1.5)
        self.assertGreaterEqual(under_target[2], 0.3)

    def test_underpass_policy_rejects_low_solid_obstacle(self):
        formation = FormationState(
            num_uavs=1,
            original_y=(0.0,),
            initial_z=0.3,
            uav_radius_xy=0.38,
            uav_radius_z=0.22,
            nominal_speed_x=1.0,
        )
        field = ObstacleField((
            ObstacleDescriptor(
                obstacle_id='low_box',
                obstacle_type='box',
                function='solid',
                x=4.0,
                center_y=0.0,
                pass_z=0.8,
                size_x=1.0,
                size_y=2.0,
                size_z=1.0,
            ),
        ))
        reports = PassabilityEvaluator().evaluate_field(field, formation, MissionPreference())
        plan = SpatioTemporalStrategyPlanner(
            gate_observe_distance_x=2.0,
            gate_pass_clear_x=1.0,
            snake_spacing_x=0.8,
        ).plan(field, formation, MissionPreference(), reports)
        underpass_plan = replace(
            plan,
            obstacle_strategies=(
                replace(plan.obstacle_strategies[0], target_policy='underpass'),
            ),
        )
        validation = StrategyValidator().validate(underpass_plan, field, formation)
        codes = {issue.code for issue in validation.issues}

        self.assertFalse(validation.valid)
        self.assertIn('underpass_clearance_too_small', codes)

    def test_underpass_policy_rejects_unknown_solid_height(self):
        formation = FormationState(
            num_uavs=1,
            original_y=(0.0,),
            initial_z=0.3,
            uav_radius_xy=0.38,
            uav_radius_z=0.22,
            nominal_speed_x=1.0,
        )
        field = ObstacleField((
            ObstacleDescriptor(
                obstacle_id='unknown_box',
                obstacle_type='box',
                function='solid',
                x=4.0,
                center_y=0.0,
                pass_z=0.8,
                size_x=1.0,
                size_y=2.0,
                size_z=None,
            ),
        ))
        reports = PassabilityEvaluator().evaluate_field(field, formation, MissionPreference())
        plan = SpatioTemporalStrategyPlanner(
            gate_observe_distance_x=2.0,
            gate_pass_clear_x=1.0,
            snake_spacing_x=0.8,
        ).plan(field, formation, MissionPreference(), reports)
        underpass_plan = replace(
            plan,
            obstacle_strategies=(
                replace(plan.obstacle_strategies[0], target_policy='underpass'),
            ),
        )
        validation = StrategyValidator().validate(underpass_plan, field, formation)
        codes = {issue.code for issue in validation.issues}

        self.assertFalse(validation.valid)
        self.assertIn('underpass_geometry_unknown', codes)

    def test_overpass_policy_rejects_tall_solid_obstacle(self):
        formation = FormationState(
            num_uavs=1,
            original_y=(0.0,),
            initial_z=0.3,
            uav_radius_xy=0.38,
            uav_radius_z=0.22,
            nominal_speed_x=1.0,
        )
        field = ObstacleField((
            ObstacleDescriptor(
                obstacle_id='tower',
                obstacle_type='very_tall_obstacle',
                function='solid',
                x=4.0,
                center_y=0.0,
                pass_z=3.4,
                size_x=1.0,
                size_y=3.0,
                size_z=6.8,
            ),
        ))
        reports = PassabilityEvaluator().evaluate_field(field, formation, MissionPreference())
        plan = SpatioTemporalStrategyPlanner(
            gate_observe_distance_x=2.0,
            gate_pass_clear_x=1.0,
            snake_spacing_x=0.8,
        ).plan(field, formation, MissionPreference(), reports)
        overpass_plan = replace(
            plan,
            obstacle_strategies=(
                replace(plan.obstacle_strategies[0], target_policy='overpass'),
            ),
        )
        validation = StrategyValidator().validate(overpass_plan, field, formation)
        codes = {issue.code for issue in validation.issues}

        self.assertFalse(validation.valid)
        self.assertIn('overpass_target_too_high', codes)

    def test_execution_helper_enforces_temporal_slot_at_commit_region(self):
        _, _, _, plan, _ = self._build_strategy_context()
        executor = StrategyExecutionHelper(plan)
        drone_id = int(plan.passing_order[1])
        gate_x = float(plan.obstacle_strategies[0].obstacle_x)

        cap_before_previous_crosses = executor.temporal_speed_limit(
            drone_id=drone_id,
            current_time=10.0,
            obstacle_crossing_times={},
            current_x=gate_x - 0.5,
            obstacle_x=gate_x,
            commit_distance_x=1.1,
            wait_speed_x=0.42,
            current_speed_x=1.0,
        )
        cap_too_soon = executor.temporal_speed_limit(
            drone_id=drone_id,
            current_time=10.2,
            obstacle_crossing_times={int(plan.passing_order[0]): 10.0},
            current_x=gate_x - 0.16,
            obstacle_x=gate_x,
            commit_distance_x=1.1,
            wait_speed_x=0.42,
            current_speed_x=1.0,
        )
        cap_after_slot = executor.temporal_speed_limit(
            drone_id=drone_id,
            current_time=10.0 + plan.time_slot_interval + 0.05,
            obstacle_crossing_times={int(plan.passing_order[0]): 10.0},
            current_x=gate_x - 0.5,
            obstacle_x=gate_x,
            commit_distance_x=1.1,
            wait_speed_x=0.42,
            current_speed_x=1.0,
        )
        no_cap_when_arrival_is_not_early = executor.temporal_speed_limit(
            drone_id=drone_id,
            current_time=10.2,
            obstacle_crossing_times={int(plan.passing_order[0]): 10.0},
            current_x=gate_x - 1.05,
            obstacle_x=gate_x,
            commit_distance_x=1.1,
            wait_speed_x=0.42,
            current_speed_x=0.35,
        )

        self.assertIsNotNone(cap_before_previous_crosses)
        self.assertGreaterEqual(cap_before_previous_crosses, 0.04)
        self.assertLessEqual(cap_before_previous_crosses, 0.42)
        self.assertIsNotNone(cap_too_soon)
        self.assertGreaterEqual(cap_too_soon, 0.04)
        self.assertLessEqual(cap_too_soon, 0.42)
        self.assertIsNone(cap_after_slot)
        self.assertIsNone(no_cap_when_arrival_is_not_early)

    def test_execution_helper_accepts_per_obstacle_adaptive_order(self):
        _, _, _, plan, _ = self._build_strategy_context()
        executor = StrategyExecutionHelper(plan)
        adaptive_order = (3, 0, 2, 1, 4)

        self.assertEqual(StrategyExecutionHelper.order_index_in(adaptive_order, 3), 0)
        self.assertEqual(executor.previous_drone_in_order(0, passing_order=adaptive_order), 3)
        self.assertIsNone(executor.previous_drone_in_order(3, passing_order=adaptive_order))


if __name__ == '__main__':
    unittest.main()
