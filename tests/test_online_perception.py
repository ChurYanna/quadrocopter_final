import unittest

from src.passage_planning import (
    ObstacleDescriptor,
    ObstacleField,
    OnlineObstaclePerception,
    OnlinePerceptionConfig,
    hide_obstacle_details,
)


class TestOnlineObstaclePerception(unittest.TestCase):
    def _field(self):
        return ObstacleField((
            ObstacleDescriptor(
                obstacle_id='solid_1',
                obstacle_type='dynamic_box',
                function='solid',
                x=6.0,
                center_y=0.0,
                pass_z=1.0,
                motion_model='sinusoidal',
                motion_amplitude=1.0,
                motion_omega=0.3,
                size_x=2.0,
                size_y=4.0,
                size_z=2.5,
            ),
            ObstacleDescriptor(
                obstacle_id='gate_1',
                obstacle_type='dynamic_wall_hole',
                function='aperture',
                x=12.0,
                center_y=0.0,
                pass_z=0.8,
                aperture_width=1.2,
                motion_model='sinusoidal',
                motion_amplitude=2.0,
                motion_omega=0.4,
            ),
            ObstacleDescriptor(
                obstacle_id='solid_2',
                obstacle_type='static_wall',
                function='solid',
                x=22.0,
                center_y=0.0,
                pass_z=2.0,
                motion_model='static',
                size_x=1.5,
                size_y=8.0,
                size_z=3.0,
            ),
        ))

    def test_detects_only_obstacles_inside_forward_window(self):
        perception = OnlineObstaclePerception(
            self._field(),
            OnlinePerceptionConfig(lookahead_distance=14.0, detail_reveal_distance=10.0, max_visible_obstacles=3),
        )

        window = perception.update(leader_x=0.0)

        self.assertEqual([item.obstacle_id for item in window.visible_obstacles], ['solid_1', 'gate_1'])
        self.assertEqual(window.newly_observed_ids, ('solid_1', 'gate_1'))
        self.assertEqual(window.planning_candidate_ids, ('solid_1',))
        self.assertTrue(window.visible_obstacles[0].details_revealed)
        self.assertFalse(window.visible_obstacles[1].details_revealed)

    def test_reveals_hidden_geometry_when_obstacle_is_close_enough(self):
        perception = OnlineObstaclePerception(
            self._field(),
            OnlinePerceptionConfig(lookahead_distance=16.0, detail_reveal_distance=10.0),
        )

        far_window = perception.update(leader_x=0.0)
        far_visible = far_window.visible_field(revealed_only=True)
        gate = next(obstacle for obstacle in far_visible.obstacles if obstacle.obstacle_id == 'gate_1')
        self.assertIsNone(gate.aperture_width)

        near_window = perception.update(leader_x=3.0)
        near_visible = near_window.visible_field(revealed_only=True)
        gate = next(obstacle for obstacle in near_visible.obstacles if obstacle.obstacle_id == 'gate_1')
        self.assertEqual(gate.aperture_width, 1.2)
        self.assertIn('gate_1', near_window.planning_candidate_ids)

    def test_planned_and_cleared_obstacles_do_not_trigger_replanning(self):
        perception = OnlineObstaclePerception(
            self._field(),
            OnlinePerceptionConfig(lookahead_distance=16.0, detail_reveal_distance=16.0),
        )

        window = perception.update(
            leader_x=0.0,
            planned_obstacle_ids={'solid_1'},
            cleared_obstacle_ids={'gate_1'},
        )

        self.assertEqual([item.obstacle_id for item in window.visible_obstacles], ['solid_1'])
        self.assertEqual(window.planning_candidate_ids, tuple())
        self.assertFalse(window.has_unplanned_obstacles)

    def test_newly_observed_ids_are_reported_only_once(self):
        perception = OnlineObstaclePerception(
            self._field(),
            OnlinePerceptionConfig(lookahead_distance=14.0, detail_reveal_distance=10.0),
        )

        first = perception.update(leader_x=0.0)
        second = perception.update(leader_x=1.0)

        self.assertEqual(first.newly_observed_ids, ('solid_1', 'gate_1'))
        self.assertEqual(second.newly_observed_ids, tuple())

    def test_planning_field_contains_only_detailed_unplanned_obstacles(self):
        perception = OnlineObstaclePerception(
            self._field(),
            OnlinePerceptionConfig(lookahead_distance=24.0, detail_reveal_distance=10.0, max_visible_obstacles=3),
        )

        window = perception.update(leader_x=3.0, planned_obstacle_ids={'solid_1'})
        planning_field = window.planning_field()

        self.assertEqual([obstacle.obstacle_id for obstacle in planning_field.obstacles], ['gate_1'])

    def test_hide_obstacle_details_removes_actionable_geometry(self):
        solid, aperture, _ = self._field().obstacles

        hidden_solid = hide_obstacle_details(solid)
        hidden_aperture = hide_obstacle_details(aperture)

        self.assertIsNone(hidden_solid.size_x)
        self.assertIsNone(hidden_solid.size_y)
        self.assertIsNone(hidden_solid.size_z)
        self.assertIsNone(hidden_aperture.aperture_width)


if __name__ == '__main__':
    unittest.main()
