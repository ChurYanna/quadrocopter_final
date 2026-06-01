import unittest

import mujoco

from src.passage_planning import (
    FormationState,
    MissionPreference,
    ObstacleFieldEncoder,
    PassabilityEvaluator,
    SpatioTemporalStrategyPlanner,
    StrategyExecutionHelper,
    StrategyValidator,
)
from tests.dynamic_mixed_bypass_obstacle_zone_config import DynamicMixedBypassObstacleZoneConfig


class TestComplexStrategyFramework(unittest.TestCase):
    """Fast non-viewer checks for the stable dynamic mixed-obstacle demo."""

    def test_five_dynamic_apertures_are_plannable(self):
        demo = DynamicMixedBypassObstacleZoneConfig()
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
        validation = StrategyValidator().validate(plan, field, formation)
        executor = StrategyExecutionHelper(plan)

        self.assertEqual(len(gate_specs), 5)
        self.assertEqual(field.count, 5)
        self.assertTrue(validation.valid, msg=validation.messages)
        self.assertEqual(plan.mode, 'snake_sequence')
        self.assertEqual(plan.passing_order, (0, 1, 2, 3, 4))
        self.assertTrue(all(report.recommended_mode == 'snake_sequence' for report in reports))
        self.assertEqual(executor.active_obstacle_index_for_position([0.0, 0.0, demo.INITIAL_Z]), 0)
        self.assertEqual(executor.active_obstacle_index_for_position([12.8, 0.0, demo.INITIAL_Z]), 2)
        self.assertEqual(executor.active_obstacle_index_for_position([23.8, 0.0, demo.INITIAL_Z]), 3)


if __name__ == '__main__':
    unittest.main()
