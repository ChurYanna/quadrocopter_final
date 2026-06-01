import json
import unittest
from unittest.mock import patch

import mujoco

from src.passage_planning import (
    FormationState,
    LLMStrategyPipeline,
    MissionPreference,
    MockLLMStrategyGenerator,
    ObstacleFieldEncoder,
    PassabilityEvaluator,
    QwenDashScopeStrategyGenerator,
    SpatioTemporalStrategyPlanner,
    StrategyExecutionHelper,
    build_llm_scene_context,
    passage_plan_from_json,
    passage_plan_to_json,
)
from tests.test_passage_strategy_framework import ThreeGatePlanningFixture


class TestLLMStrategyInterface(unittest.TestCase):
    """Local closed-loop tests before connecting a real LLM API."""

    def _build_context(self):
        demo = ThreeGatePlanningFixture()
        model = mujoco.MjModel.from_xml_path(demo.SCENE)
        gate_specs = demo._build_gate_runtime_specs(model)
        demo._update_dynamic_gates(model, gate_specs, 0.0)

        formation = FormationState(
            num_uavs=demo.COUNT,
            original_y=(0.0, 1.0, -1.0, 2.0, -2.0),
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
        deterministic_plan = SpatioTemporalStrategyPlanner(
            gate_observe_distance_x=demo.GATE_OBSERVE_DISTANCE_X,
            gate_pass_clear_x=demo.GATE_PASS_CLEAR_X,
            snake_spacing_x=demo.SNAKE_SPACING_X,
        ).plan(field, formation, mission, reports)
        scene_context = build_llm_scene_context(field, formation, mission, deterministic_plan)
        return demo, field, formation, deterministic_plan, scene_context

    def test_context_contains_semantic_scene_for_llm(self):
        _, _, _, _, scene_context = self._build_context()

        self.assertEqual(scene_context['task'], 'multi_uav_dynamic_obstacle_passage')
        self.assertEqual(scene_context['uav_team']['num_uavs'], 5)
        self.assertEqual(len(scene_context['obstacle_field']), 3)
        self.assertEqual(scene_context['required_output']['schema'], 'PassagePlan')
        self.assertIn('deterministic_reference_plan', scene_context)
        self.assertIn('passing_order is a list of UAV ids', scene_context['required_output']['hard_rules'][0])

    def test_valid_mock_llm_plan_is_used(self):
        _, field, formation, deterministic_plan, scene_context = self._build_context()
        plan, validation, used_fallback = LLMStrategyPipeline().generate_or_fallback(
            MockLLMStrategyGenerator(deterministic_plan, mode='valid'),
            scene_context,
            deterministic_plan,
            field,
            formation,
        )

        self.assertFalse(used_fallback)
        self.assertTrue(validation.valid, msg=validation.messages)
        self.assertEqual(plan.source, 'mock_llm')
        self.assertEqual(plan.plan_id, 'mock_llm_valid_plan')

    def test_repairable_mock_llm_plan_is_repaired_and_used(self):
        _, field, formation, deterministic_plan, scene_context = self._build_context()
        plan, validation, used_fallback = LLMStrategyPipeline().generate_or_fallback(
            MockLLMStrategyGenerator(deterministic_plan, mode='repairable'),
            scene_context,
            deterministic_plan,
            field,
            formation,
        )

        self.assertFalse(used_fallback)
        self.assertTrue(validation.valid, msg=validation.messages)
        self.assertTrue(validation.repaired)
        self.assertGreaterEqual(plan.time_slot_interval, 0.25)

    def test_unsafe_mock_llm_plan_falls_back(self):
        _, field, formation, deterministic_plan, scene_context = self._build_context()
        plan, validation, used_fallback = LLMStrategyPipeline().generate_or_fallback(
            MockLLMStrategyGenerator(deterministic_plan, mode='unsafe'),
            scene_context,
            deterministic_plan,
            field,
            formation,
        )
        executor = StrategyExecutionHelper(plan)

        self.assertTrue(used_fallback)
        self.assertTrue(validation.valid, msg=validation.messages)
        self.assertEqual(plan.source, 'deterministic')
        self.assertEqual(executor.active_obstacle_index_for_position([7.5, 0.0, 0.3]), 1)

    def test_malformed_mock_llm_plan_falls_back(self):
        _, field, formation, deterministic_plan, scene_context = self._build_context()
        plan, validation, used_fallback = LLMStrategyPipeline().generate_or_fallback(
            MockLLMStrategyGenerator(deterministic_plan, mode='malformed'),
            scene_context,
            deterministic_plan,
            field,
            formation,
        )

        self.assertTrue(used_fallback)
        self.assertTrue(validation.valid, msg=validation.messages)
        self.assertEqual(plan.source, 'deterministic')
        self.assertTrue(any('llm_strategy_parse_failed' in msg for msg in validation.messages))

    def test_qwen_generator_parses_openai_compatible_response(self):
        _, _, _, deterministic_plan, scene_context = self._build_context()
        qwen_plan_json = passage_plan_to_json(deterministic_plan)

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return json.dumps({
                    'choices': [
                        {'message': {'content': qwen_plan_json}},
                    ],
                }).encode('utf-8')

        def fake_urlopen(request, timeout):
            self.assertIn('/chat/completions', request.full_url)
            self.assertEqual(timeout, 3.0)
            self.assertEqual(request.headers['Authorization'], 'Bearer test-key')
            return FakeResponse()

        with patch.dict('os.environ', {'DASHSCOPE_API_KEY': 'test-key'}, clear=False):
            with patch('urllib.request.urlopen', fake_urlopen):
                output = QwenDashScopeStrategyGenerator(timeout=3.0).generate_strategy_json(scene_context)

        self.assertIn('"version": "1.0"', output)
        self.assertIn('"mode": "snake_sequence"', output)

    def test_qwen_system_prompt_disambiguates_uav_order_and_obstacle_order(self):
        prompt = QwenDashScopeStrategyGenerator._system_prompt()

        self.assertIn('passing_order MUST be UAV ids', prompt)
        self.assertIn('MUST NOT contain obstacle ids', prompt)
        self.assertIn('deterministic_reference_plan', prompt)

    def test_strategy_json_accepts_common_llm_string_indices(self):
        _, _, _, deterministic_plan, _ = self._build_context()
        payload = json.loads(passage_plan_to_json(deterministic_plan))
        payload['passing_order'] = ['uav0', 'uav1', 'uav2', 'uav3', 'uav4']
        payload['slots'][0]['drone_id'] = 'uav0'
        payload['slots'][0]['order_index'] = '0'
        payload['obstacle_strategies'][0]['obstacle_index'] = 'gate1'

        parsed_plan = passage_plan_from_json(json.dumps(payload))

        self.assertEqual(parsed_plan.passing_order, (0, 1, 2, 3, 4))
        self.assertEqual(parsed_plan.slots[0].drone_id, 0)
        self.assertEqual(parsed_plan.obstacle_strategies[0].obstacle_index, 0)

    def test_strategy_json_fills_missing_slot_drone_id_from_order(self):
        _, _, _, deterministic_plan, _ = self._build_context()
        payload = json.loads(passage_plan_to_json(deterministic_plan))
        for slot in payload['slots']:
            slot.pop('drone_id')

        parsed_plan = passage_plan_from_json(json.dumps(payload))

        self.assertEqual(
            [slot.drone_id for slot in parsed_plan.slots],
            list(parsed_plan.passing_order),
        )

    def test_qwen_missing_api_key_falls_back(self):
        _, field, formation, deterministic_plan, scene_context = self._build_context()
        with patch.dict('os.environ', {}, clear=True):
            with patch.object(QwenDashScopeStrategyGenerator, '_load_local_env', lambda self: None):
                plan, validation, used_fallback = LLMStrategyPipeline().generate_or_fallback(
                    QwenDashScopeStrategyGenerator(timeout=0.01),
                    scene_context,
                    deterministic_plan,
                    field,
                    formation,
                )

        self.assertTrue(used_fallback)
        self.assertTrue(validation.valid, msg=validation.messages)
        self.assertEqual(plan.source, 'deterministic')
        self.assertTrue(any('missing API key' in msg for msg in validation.messages))


if __name__ == '__main__':
    unittest.main()
