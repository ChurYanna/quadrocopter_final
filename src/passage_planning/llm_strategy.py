from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import replace
from pathlib import Path
from typing import Protocol

from .strategy_json import StrategyJsonError, passage_plan_from_json, passage_plan_to_json
from .strategy_validator import StrategyValidator
from .structures import FormationState, ObstacleField, PassagePlan, ValidationResult


class LLMStrategyGenerator(Protocol):
    """Interface implemented by local mock generators and future API clients."""

    def generate_strategy_json(self, scene_context: dict) -> str:
        ...


class LLMStrategyError(RuntimeError):
    """Raised when an external LLM strategy request fails."""


class MockLLMStrategyGenerator:
    """Local stand-in for an LLM response.

    Modes:
    - `valid`: returns the deterministic baseline as a JSON strategy.
    - `repairable`: returns valid structure but bad time slots, expecting repair.
    - `unsafe`: returns a risky strategy that the validator must reject.
    - `malformed`: returns invalid JSON.
    """

    def __init__(self, baseline_plan: PassagePlan, mode: str = 'valid'):
        self.baseline_plan = baseline_plan
        self.mode = mode

    def generate_strategy_json(self, scene_context: dict) -> str:
        if self.mode == 'valid':
            return passage_plan_to_json(replace(self.baseline_plan, source='mock_llm', plan_id='mock_llm_valid_plan'))
        if self.mode == 'repairable':
            slots = tuple(
                replace(slot, nominal_entry_time=0.0, nominal_exit_time=0.1)
                for slot in self.baseline_plan.slots
            )
            plan = replace(
                self.baseline_plan,
                source='mock_llm',
                plan_id='mock_llm_repairable_plan',
                time_slot_interval=0.05,
                slots=slots,
            )
            return passage_plan_to_json(plan)
        if self.mode == 'unsafe':
            first_strategy = replace(
                self.baseline_plan.obstacle_strategies[0],
                mode='formation',
                target_policy='predictive_center_crossing',
                clear_x=self.baseline_plan.obstacle_strategies[0].obstacle_x + 0.05,
                observe_x=self.baseline_plan.obstacle_strategies[0].obstacle_x,
            )
            plan = replace(
                self.baseline_plan,
                source='mock_llm',
                plan_id='mock_llm_unsafe_plan',
                mode='formation',
                confidence=0.95,
                passing_order=(0, 1, 1, 3, 4),
                time_slot_interval=0.05,
                obstacle_strategies=(first_strategy,) + self.baseline_plan.obstacle_strategies[1:],
            )
            return passage_plan_to_json(plan)
        if self.mode == 'malformed':
            return '{"version": "1.0", "mode": "snake_sequence"'
        raise ValueError(f'unknown mock LLM mode: {self.mode}')


class QwenDashScopeStrategyGenerator:
    """Qwen/DashScope OpenAI-compatible strategy generator.

    The API key is read from `DASHSCOPE_API_KEY`.  The generator returns the
    assistant message content as raw PassagePlan JSON; parsing and safety checks
    are handled by LLMStrategyPipeline and StrategyValidator.
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key_env: str = 'DASHSCOPE_API_KEY',
        timeout: float = 30.0,
        temperature: float = 0.0,
    ):
        self.model = model or os.getenv('DASHSCOPE_MODEL', 'qwen-plus')
        self.base_url = (base_url or os.getenv('DASHSCOPE_BASE_URL', 'https://dashscope.aliyuncs.com/compatible-mode/v1')).rstrip('/')
        self.api_key_env = api_key_env
        self.timeout = float(timeout)
        self.temperature = float(temperature)

    def generate_strategy_json(self, scene_context: dict) -> str:
        self._load_local_env()
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise LLMStrategyError(f'missing API key environment variable: {self.api_key_env}')

        payload = {
            'model': self.model,
            'messages': [
                {
                    'role': 'system',
                    'content': self._system_prompt(),
                },
                {
                    'role': 'user',
                    'content': json.dumps(scene_context, ensure_ascii=False, sort_keys=True),
                },
            ],
            'temperature': self.temperature,
            'response_format': {'type': 'json_object'},
        }
        request = urllib.request.Request(
            url=f'{self.base_url}/chat/completions',
            data=json.dumps(payload, ensure_ascii=False).encode('utf-8'),
            headers={
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json',
            },
            method='POST',
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                response_payload = json.loads(response.read().decode('utf-8'))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode('utf-8', errors='replace')
            raise LLMStrategyError(f'Qwen HTTP error {exc.code}: {body}') from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise LLMStrategyError(f'Qwen request failed: {exc}') from exc

        try:
            content = response_payload['choices'][0]['message']['content']
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMStrategyError(f'Qwen response missing message content: {response_payload}') from exc
        return self._extract_json_content(str(content))

    @staticmethod
    def _load_local_env(path: str = '.env.local') -> None:
        """Load local key/value overrides for convenient private debugging.

        This intentionally supports only simple KEY=VALUE lines and does not
        overwrite variables already exported in the shell.
        """
        env_path = Path(path)
        if not env_path.exists():
            return
        for raw_line in env_path.read_text(encoding='utf-8').splitlines():
            line = raw_line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value

    @staticmethod
    def _system_prompt() -> str:
        return (
            'You are a high-level strategy planner for multi-UAV dynamic obstacle passage. '
            'Return only one JSON object. Do not return Markdown, comments, prose, or explanations. '
            'The output must exactly match PassagePlan JSON version 1.0. '
            'Required top-level keys: version, plan_id, source, mode, confidence, '
            'recover_after_last_obstacle, time_slot_interval, passing_order, slots, obstacle_strategies. '
            'passing_order MUST be UAV ids, for example [0,1,2,3,4]. '
            'passing_order MUST NOT contain obstacle ids such as gate1/gate2/gate3. '
            'slots MUST be UAV time slots, one per UAV, and every slot MUST include '
            'drone_id, order_index, nominal_entry_time, nominal_exit_time. '
            'obstacle_strategies MUST be obstacle strategies, one per obstacle, and every strategy MUST include '
            'obstacle_id, obstacle_index, mode, target_policy, obstacle_x, clear_x, observe_x. '
            'Do not invent fields such as slot_start_time, slot_duration, pass_z, entry_offset_y, timing_adjustment_s. '
            'Use only obstacle strategy modes snake_sequence and bypass. '
            'Do not use formation for any obstacle_strategy; formation recovery happens only after the full obstacle zone is cleared. '
            'Use target_policy predictive_center_crossing for aperture snake_sequence obstacles. '
            'For solid bypass obstacles, target_policy may be edge_bypass, side_bypass_left, side_bypass_right, '
            'split_by_lane_bypass, overpass, underpass, or hybrid_over_or_side. These are high-level route families only; '
            'do not assign per-UAV low-level paths. Choose underpass only when underpass_feasible is true and bottom clearance is explicit. '
            'Choose overpass only when overpass_reasonable is true. When both are feasible, follow preferred_vertical_route or the smaller vertical_delta_z. '
            'If the user payload contains deterministic_reference_plan, prefer copying it exactly and set source to qwen. '
            'Do not output low-level velocity commands, motor commands, or raw trajectories.'
        )

    @staticmethod
    def _extract_json_content(content: str) -> str:
        text = content.strip()
        if text.startswith('```'):
            lines = [line for line in text.splitlines() if not line.strip().startswith('```')]
            text = '\n'.join(lines).strip()
        start = text.find('{')
        end = text.rfind('}')
        if start == -1 or end == -1 or end < start:
            raise LLMStrategyError('Qwen response did not contain a JSON object')
        return text[start:end + 1]


class LLMStrategyPipeline:
    """Parse, validate, repair, and fallback wrapper for external strategies."""

    def __init__(self, validator: StrategyValidator | None = None, debug: bool = False):
        self.validator = validator or StrategyValidator()
        self.debug = bool(debug)

    def generate_or_fallback(
        self,
        generator: LLMStrategyGenerator,
        scene_context: dict,
        deterministic_plan: PassagePlan,
        field: ObstacleField,
        formation: FormationState,
    ) -> tuple[PassagePlan, ValidationResult, bool]:
        try:
            raw_strategy_json = generator.generate_strategy_json(scene_context)
            if self.debug:
                print('[LLM] raw strategy JSON preview:')
                print(self._preview(raw_strategy_json))
            candidate = passage_plan_from_json(raw_strategy_json)
            if self.debug:
                print(
                    '[LLM] parsed candidate:',
                    f'plan_id={candidate.plan_id}',
                    f'source={candidate.source}',
                    f'mode={candidate.mode}',
                    f'order={candidate.passing_order}',
                    f'slots={len(candidate.slots)}',
                    f'obstacles={len(candidate.obstacle_strategies)}',
                )
        except Exception as exc:
            fallback_result = self.validator.validate(deterministic_plan, field, formation)
            message = f'llm_strategy_parse_failed: {exc}'
            if self.debug:
                print(f'[LLM] parse/request failed, fallback to deterministic plan: {exc}')
            return deterministic_plan, replace(
                fallback_result,
                messages=fallback_result.messages + (message,),
            ), True

        repaired_candidate, validation = self.validator.validate_and_repair(candidate, field, formation)
        if self.debug:
            print(
                '[LLM] validation:',
                f'valid={validation.valid}',
                f'repaired={validation.repaired}',
                f'issues={[issue.code for issue in validation.issues]}',
            )
        if validation.valid:
            return repaired_candidate, validation, False

        fallback_result = self.validator.validate(deterministic_plan, field, formation)
        if self.debug:
            print('[LLM] validation failed, fallback to deterministic plan:', validation.messages)
        return deterministic_plan, replace(
            fallback_result,
            messages=fallback_result.messages + tuple(f'fallback_after_llm_error: {msg}' for msg in validation.messages),
        ), True

    @staticmethod
    def _preview(text: str, limit: int = 1600) -> str:
        text = str(text)
        if len(text) <= limit:
            return text
        return text[:limit] + f'\n... <truncated {len(text) - limit} chars>'
