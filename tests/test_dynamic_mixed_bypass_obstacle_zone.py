import numpy as np

from tests.dynamic_mixed_bypass_obstacle_zone_config import DynamicMixedBypassObstacleZoneConfig
from tests.stable_passage_demo_kernel import run_stable_mixed_passage_demo
from tests.test_three_snake_dynamic_gates import TestThreeSnakeDynamicGates


class TestDynamicMixedBypassObstacleZone(DynamicMixedBypassObstacleZoneConfig, TestThreeSnakeDynamicGates):
    """Viewer test for dynamic aperture passage plus an executable solid bypass obstacle."""

    USE_LLM_STRATEGY = True
    LLM_PROVIDER = 'qwen'
    DEBUG_PASSAGE_PLAN = True
    SEMANTIC_PANEL_ENABLE = True
    SEMANTIC_TRACE_CONSOLE = False
    VISUAL_TRACE_ENABLE = True
    VLM_PROVIDER = 'off'
    ONLINE_REPLANNING_ENABLE = True
    ONLINE_REPLANNING_PROVIDER = 'qwen_vlm'

    def test_dynamic_mixed_bypass_obstacle_zone_viewer(self):
        if self.DEBUG_PASSAGE_PLAN:
            print('[VIEWER] dynamic mixed bypass zone: wall apertures + moving solid bypass obstacles')
        run_stable_mixed_passage_demo(self)

        leader_final = self._last_leader_final
        passage_metrics = self._last_passage_metrics
        self.assertTrue(
            float(leader_final[0]) >= self.FINAL_TARGET_X - 1.0,
            msg=f'leader did not reach dynamic mixed-bypass target, leader={np.round(leader_final, 3)}',
        )
        self.assertTrue(passage_metrics.success, msg=passage_metrics.to_dict())


if __name__ == '__main__':
    import unittest

    unittest.main()
