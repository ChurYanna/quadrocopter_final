import unittest

from src.passage_planning import (
    AsyncOnlineReplanningManager,
    FormationState,
    MissionPreference,
    MockLLMStrategyGenerator,
    ObstacleDescriptor,
    ObstacleField,
    OnlineObstaclePerception,
    OnlinePerceptionConfig,
    OnlineReplanner,
    OnlineReplanningConfig,
    PassabilityEvaluator,
)


class TestOnlineReplanner(unittest.TestCase):
    def _formation(self):
        return FormationState(
            num_uavs=5,
            original_y=(0.0, 1.0, -1.0, 2.0, -2.0),
            initial_z=0.5,
            uav_radius_xy=0.38,
            uav_radius_z=0.22,
            nominal_speed_x=1.6,
            max_speed_xy=3.0,
            max_speed_z=1.2,
            max_lateral_speed=3.0,
        )

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
                risk_level='medium',
                size_x=2.0,
                size_y=5.0,
                size_z=3.0,
            ),
            ObstacleDescriptor(
                obstacle_id='gate_1',
                obstacle_type='dynamic_wall_hole',
                function='aperture',
                x=12.0,
                center_y=0.0,
                pass_z=0.8,
                aperture_width=1.3,
                motion_model='sinusoidal',
                motion_amplitude=2.0,
                motion_omega=0.35,
                risk_level='high',
            ),
            ObstacleDescriptor(
                obstacle_id='solid_2',
                obstacle_type='static_wall',
                function='solid',
                x=25.0,
                center_y=0.0,
                pass_z=2.5,
                motion_model='static',
                risk_level='medium',
                size_x=2.0,
                size_y=8.0,
                size_z=4.0,
            ),
        ))

    def _perception(self, field=None):
        return OnlineObstaclePerception(
            field or self._field(),
            OnlinePerceptionConfig(
                lookahead_distance=16.0,
                detail_reveal_distance=10.0,
                max_visible_obstacles=3,
            ),
        )

    def _replanner(self, min_interval=6.0):
        return OnlineReplanner(
            formation=self._formation(),
            mission=MissionPreference(allow_disband=True, recover_after_last_obstacle=True),
            config=OnlineReplanningConfig(
                min_replan_interval=min_interval,
                gate_observe_distance_x=5.0,
                gate_pass_clear_x=1.2,
                snake_spacing_x=0.8,
            ),
        )

    def test_triggers_local_plan_for_first_visible_solid_obstacle(self):
        perception = self._perception()
        replanner = self._replanner()
        window = perception.update(leader_x=0.0)

        result = replanner.evaluate(window, current_time=0.0)

        self.assertTrue(result.triggered)
        self.assertTrue(result.accepted, msg=result.to_dict())
        self.assertEqual(result.reason, 'new_obstacle_window_detected')
        self.assertEqual(result.planning_obstacle_ids, ('solid_1',))
        self.assertIn('solid_1', result.buffer.planned_obstacle_ids)
        self.assertEqual(result.selected_plan.obstacle_strategies[0].mode, 'bypass')
        self.assertEqual(result.selected_plan.obstacle_strategies[0].target_policy, 'edge_bypass')

    def test_does_not_replan_when_interval_has_not_elapsed(self):
        perception = self._perception()
        replanner = self._replanner(min_interval=6.0)
        first_window = perception.update(leader_x=0.0)
        first_result = replanner.evaluate(first_window, current_time=0.0)
        self.assertTrue(first_result.accepted)

        second_window = perception.update(
            leader_x=3.0,
            planned_obstacle_ids=replanner.buffer.planned_obstacle_ids,
        )
        second_result = replanner.evaluate(second_window, current_time=2.0)

        self.assertFalse(second_result.triggered)
        self.assertEqual(second_result.reason, 'min_replan_interval_not_elapsed')

    def test_replans_next_aperture_after_interval_elapsed(self):
        perception = self._perception()
        replanner = self._replanner(min_interval=1.0)
        first_window = perception.update(leader_x=0.0)
        first_result = replanner.evaluate(first_window, current_time=0.0)
        self.assertTrue(first_result.accepted)

        second_window = perception.update(
            leader_x=3.0,
            planned_obstacle_ids=replanner.buffer.planned_obstacle_ids,
        )
        second_result = replanner.evaluate(second_window, current_time=1.5)

        self.assertTrue(second_result.triggered)
        self.assertTrue(second_result.accepted, msg=second_result.to_dict())
        self.assertEqual(second_result.planning_obstacle_ids, ('gate_1',))
        self.assertIn('gate_1', second_result.buffer.planned_obstacle_ids)
        self.assertEqual(second_result.selected_plan.obstacle_strategies[0].mode, 'snake_sequence')

    def test_unsafe_llm_output_falls_back_to_deterministic_local_plan(self):
        perception = self._perception()
        replanner = self._replanner()
        window = perception.update(leader_x=0.0)

        local_field = window.planning_field()
        reports = PassabilityEvaluator().evaluate_field(
            local_field,
            self._formation(),
            MissionPreference(),
        )
        deterministic_plan = replanner.planner.plan(local_field, self._formation(), MissionPreference(), reports)
        result = replanner.evaluate(
            window,
            current_time=0.0,
            generator=MockLLMStrategyGenerator(deterministic_plan, mode='unsafe'),
        )

        self.assertTrue(result.triggered)
        self.assertTrue(result.accepted)
        self.assertTrue(result.used_fallback)
        self.assertEqual(result.selected_plan.source, 'deterministic')
        self.assertIn('solid_1', result.buffer.planned_obstacle_ids)

    def test_buffer_mark_cleared_removes_planned_strategy(self):
        perception = self._perception()
        replanner = self._replanner()
        window = perception.update(leader_x=0.0)
        result = replanner.evaluate(window, current_time=0.0)
        self.assertTrue(result.accepted)

        replanner.mark_cleared(['solid_1'])

        self.assertIn('solid_1', replanner.buffer.cleared_obstacle_ids)
        self.assertNotIn('solid_1', replanner.buffer.planned_obstacle_ids)
        self.assertIsNone(replanner.buffer.strategy_for('solid_1'))

    def test_async_manager_accepts_fallback_immediately_then_merges_llm_result(self):
        perception = self._perception()
        replanner = OnlineReplanner(
            formation=self._formation(),
            mission=MissionPreference(allow_disband=True, recover_after_last_obstacle=True),
            config=OnlineReplanningConfig(
                min_replan_interval=1.0,
                gate_observe_distance_x=5.0,
                gate_pass_clear_x=1.2,
                snake_spacing_x=0.8,
                use_mock_when_generator_missing=False,
            ),
        )
        window = perception.update(leader_x=0.0)
        local_field = window.planning_field()
        reports = PassabilityEvaluator().evaluate_field(local_field, self._formation(), MissionPreference())
        deterministic_plan = replanner.planner.plan(local_field, self._formation(), MissionPreference(), reports)
        manager = AsyncOnlineReplanningManager(
            replanner,
            MockLLMStrategyGenerator(deterministic_plan, mode='valid'),
        )

        submit_events = manager.submit_if_needed(window, current_time=0.0)

        self.assertEqual([event.event_type for event in submit_events], ['fallback_accepted', 'submitted'])
        self.assertIn('solid_1', manager.buffer.planned_obstacle_ids)

        completed = []
        for _ in range(20):
            completed = manager.poll_completed(current_time=0.5)
            if completed:
                break

        manager.close()
        self.assertEqual(len(completed), 1)
        self.assertEqual(completed[0].event_type, 'completed_accepted')
        self.assertEqual(completed[0].result.selected_plan.source, 'mock_llm')
        self.assertIn('solid_1', manager.buffer.planned_obstacle_ids)


if __name__ == '__main__':
    unittest.main()
