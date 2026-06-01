import unittest

import numpy as np

from src.passage_planning import (
    ObstaclePassageStrategy,
    PassagePlan,
    PassageRunMetricsRecorder,
    PassageSlot,
    ValidationResult,
)


class TestPassageMetrics(unittest.TestCase):
    def _plan(self):
        slots = tuple(
            PassageSlot(
                drone_id=drone_id,
                order_index=drone_id,
                nominal_entry_time=float(drone_id),
                nominal_exit_time=float(drone_id + 1),
            )
            for drone_id in range(3)
        )
        strategies = (
            ObstaclePassageStrategy(
                obstacle_id='gate1',
                obstacle_index=0,
                mode='snake_sequence',
                target_policy='predictive_center_crossing',
                obstacle_x=5.0,
                clear_x=6.2,
                observe_x=0.0,
            ),
        )
        return PassagePlan(
            plan_id='metrics_fixture',
            mode='snake_sequence',
            passing_order=(0, 1, 2),
            slots=slots,
            obstacle_strategies=strategies,
            recover_after_last_obstacle=True,
            time_slot_interval=1.0,
            confidence=0.9,
        )

    def test_recorder_summarizes_distances_stages_and_clearance(self):
        recorder = PassageRunMetricsRecorder(
            scenario_id='unit_metrics',
            plan=self._plan(),
            uav_count=3,
            obstacle_count=1,
            final_target_x=8.0,
            validation_result=ValidationResult(valid=True, repaired=True, messages=('repaired slots',)),
            used_fallback=True,
        )

        recorder.mark_stage_transition(1.0, 'rotate', 'formation')
        recorder.mark_stage_transition(2.0, 'formation', 'snake')
        recorder.mark_stage_transition(6.0, 'snake', 'reform')
        recorder.mark_stage_transition(7.5, 'reform', 'post_formation')
        recorder.update_positions(
            1.0,
            [
                np.array([0.0, 0.0, 0.3]),
                np.array([0.0, 1.0, 0.3]),
                np.array([0.0, 2.0, 0.3]),
            ],
        )
        positions = [
            np.array([5.0, 0.2, 0.3]),
            np.array([4.0, 0.9, 0.3]),
            np.array([3.0, 1.9, 0.3]),
        ]
        recorder.update_positions(5.0, positions)
        recorder.update_aperture_clearance(
            5.0,
            positions,
            aperture_snapshots=({'x': 5.0, 'center_y': 0.0, 'aperture_width': 1.4},),
            uav_radius_xy=0.3,
            x_window=0.25,
        )
        recorder.record_temporal_gate_hold()
        recorder.record_temporal_gate_hold()

        metrics = recorder.finish(
            current_time=8.0,
            final_stage='post_formation',
            leader_final=np.array([8.1, 0.0, 0.3]),
            final_positions=positions,
            cleared_last_obstacle=True,
            success=True,
        )
        payload = metrics.to_dict()

        self.assertTrue(metrics.success)
        self.assertTrue(metrics.used_fallback)
        self.assertTrue(metrics.validation_repaired)
        self.assertEqual(metrics.formation_recovery_time, 1.5)
        self.assertAlmostEqual(metrics.min_inter_uav_distance, 1.0, places=6)
        self.assertAlmostEqual(metrics.min_aperture_lateral_clearance, 0.2)
        self.assertEqual(metrics.temporal_gate_holds, 2)
        self.assertEqual(payload['temporal_gate_holds'], 2)
        self.assertEqual(payload['stage_transitions'][0]['to_stage'], 'formation')
        self.assertIn('unit_metrics', metrics.summary_line())


if __name__ == '__main__':
    unittest.main()
