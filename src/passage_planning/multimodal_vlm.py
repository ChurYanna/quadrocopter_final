from __future__ import annotations

import base64
import json
import mimetypes
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MultimodalVLMInputPacket:
    """Structured packet for a future front-RGB + SFSC VLM request."""

    packet_id: str
    scenario_id: str
    sim_time: float
    trigger_label: str
    trigger_reason: str
    front_rgb: dict[str, Any]
    sfsc_context: dict[str, Any]
    request: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            'packet_id': self.packet_id,
            'scenario_id': self.scenario_id,
            'sim_time': float(self.sim_time),
            'trigger_label': self.trigger_label,
            'trigger_reason': self.trigger_reason,
            'front_rgb': _json_safe(self.front_rgb),
            'sfsc_context': _json_safe(self.sfsc_context),
            'request': _json_safe(self.request),
        }


@dataclass(frozen=True)
class VLMStrategyResult:
    """Mock/future VLM high-level strategy result.

    The result is intentionally not a controller command.  It is a strategy
    suggestion that must go through validation before it can affect execution.
    """

    result_id: str
    packet_id: str
    source: str
    confidence: float
    accepted_for_control: bool
    visual_assessment: dict[str, Any]
    strategy_suggestion: dict[str, Any]
    obstacle_strategies: list[dict[str, Any]]
    safety_notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            'result_id': self.result_id,
            'packet_id': self.packet_id,
            'source': self.source,
            'confidence': float(self.confidence),
            'accepted_for_control': bool(self.accepted_for_control),
            'visual_assessment': _json_safe(self.visual_assessment),
            'strategy_suggestion': _json_safe(self.strategy_suggestion),
            'obstacle_strategies': _json_safe(self.obstacle_strategies),
            'safety_notes': _json_safe(self.safety_notes),
        }


class MockVLMStrategyGenerator:
    """Deterministic mock VLM for testing multimodal evidence flow."""

    def generate(self, packet: MultimodalVLMInputPacket) -> VLMStrategyResult:
        data = packet.to_dict()
        front_rgb = data['front_rgb']
        sfsc_events = data['sfsc_context'].get('recent_events', [])
        obstacle_ids = _extract_candidate_obstacle_ids(sfsc_events)
        suggested_mode, target_policy = _suggest_mode_from_events(sfsc_events)
        confidence = 0.76 if front_rgb.get('available') else 0.35
        source_uav = front_rgb.get('source_uav_id')
        camera_name = front_rgb.get('camera_name') or 'unknown'
        visual_summary = (
            f'使用当前最前方 UAV {source_uav} 的前视 RGB ({camera_name}) 作为视觉证据；'
            'mock VLM 不读取图像像素，只模拟真实 VLM 的输出结构。'
        )
        if not front_rgb.get('available'):
            visual_summary = '前视 RGB 不可用，mock VLM 建议保持确定性 fallback。'
            suggested_mode, target_policy = 'hold', 'hold_and_reform'
        return VLMStrategyResult(
            result_id=f'mock_result_for_{packet.packet_id}',
            packet_id=packet.packet_id,
            source='mock_vlm',
            confidence=confidence,
            accepted_for_control=False,
            visual_assessment={
                'image_understanding': visual_summary,
                'sfsc_consistency': 'consistent' if front_rgb.get('available') else 'uncertain',
                'occlusion_risk': 'low' if front_rgb.get('selection_policy') == 'frontmost_uav' else 'medium',
                'morphology_risk': 'medium',
            },
            strategy_suggestion={
                'mode': suggested_mode,
                'target_policy': target_policy,
                'passing_order': [0, 1, 2, 3, 4],
                'reason': (
                    'mock VLM 根据 SFSC 最近事件和前视 RGB 可用性生成高层建议；'
                    '当前结果只进入证据链，不进入控制。'
                ),
            },
            obstacle_strategies=[
                {
                    'obstacle_id': obstacle_id,
                    'mode': suggested_mode,
                    'target_policy': target_policy,
                    'risk_note': 'mock result; requires validator before execution',
                }
                for obstacle_id in (obstacle_ids or ['visible_obstacle'])
            ],
            safety_notes=[
                'VLM output is advisory only in the mock stage.',
                'Keep deterministic fallback active during VLM latency or uncertainty.',
                'Safety validator must remain the only gate into the execution buffer.',
            ],
        )


class QwenDashScopeVLMStrategyGenerator:
    """DashScope OpenAI-compatible VLM generator for front-RGB + SFSC packets."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key_env: str = 'DASHSCOPE_API_KEY',
        timeout: float = 45.0,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ):
        self._load_local_env()
        self.model = model or os.getenv('DASHSCOPE_VLM_MODEL', 'qwen3-vl-flash')
        self.base_url = (base_url or os.getenv('DASHSCOPE_BASE_URL', 'https://dashscope.aliyuncs.com/compatible-mode/v1')).rstrip('/')
        self.api_key_env = api_key_env
        self.timeout = float(timeout)
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens or os.getenv('DASHSCOPE_VLM_MAX_TOKENS', '900'))

    def generate(self, packet: MultimodalVLMInputPacket) -> VLMStrategyResult:
        self._load_local_env()
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise RuntimeError(f'missing API key environment variable: {self.api_key_env}')

        packet_dict = packet.to_dict()
        image_url = self._image_data_url(packet_dict['front_rgb'].get('image_path'))
        prompt_payload = {
            'packet_id': packet.packet_id,
            'scenario_id': packet.scenario_id,
            'sim_time': packet.sim_time,
            'front_rgb': {
                key: value
                for key, value in packet_dict['front_rgb'].items()
                if key not in {'image_path', 'error'}
            },
            'sfsc_context': packet_dict['sfsc_context'],
            'expected_output_schema': _expected_vlm_output_schema(),
            'hard_rules': packet_dict['request']['hard_rules'],
        }
        payload = {
            'model': self.model,
            'messages': [
                {
                    'role': 'system',
                    'content': self._system_prompt(),
                },
                {
                    'role': 'user',
                    'content': [
                        {
                            'type': 'text',
                            'text': json.dumps(prompt_payload, ensure_ascii=False, sort_keys=True),
                        },
                        {
                            'type': 'image_url',
                            'image_url': {'url': image_url},
                        },
                    ],
                },
            ],
            'temperature': self.temperature,
            'response_format': {'type': 'json_object'},
            'max_tokens': self.max_tokens,
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
            raise RuntimeError(f'Qwen VLM HTTP error {exc.code}: {body}') from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f'Qwen VLM request failed: {exc}') from exc

        try:
            content = response_payload['choices'][0]['message']['content']
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f'Qwen VLM response missing message content: {response_payload}') from exc
        return self._result_from_json(packet.packet_id, self._extract_json_content(str(content)))

    @staticmethod
    def _system_prompt() -> str:
        return (
            'You are a multimodal high-level strategy advisor for multi-UAV dynamic obstacle passage. '
            'You receive one onboard front RGB image from the current frontmost UAV and one SFSC '
            '(Sensor-Fused Structured Context) packet. Return exactly one JSON object. '
            'Do not return Markdown or prose. Do not output motor thrust, raw velocity commands, '
            'or low-level trajectories. Your output is advisory only and must be validated before execution. '
            'Required keys: confidence, visual_assessment, strategy_suggestion, obstacle_strategies, safety_notes. '
            'visual_assessment must include image_understanding, sfsc_consistency, occlusion_risk, morphology_risk. '
            'strategy_suggestion must include mode, target_policy, passing_order, reason. '
            'passing_order must be numeric UAV ids such as [0,1,2,3,4], never strings such as ["uav0"]. '
            'Use obstacle strategy modes only from snake_sequence, bypass, hold. '
            'Do not recommend formation for obstacle-zone passage; formation recovery is post-obstacle only. '
            'Do not invent numeric safety thresholds such as clearance or altitude unless the SFSC packet explicitly provides them. '
            'Use qualitative safety notes when numeric constraints are not available. '
            'If the image is unclear or conflicts with SFSC, report uncertainty and recommend deterministic fallback.'
        )

    @staticmethod
    def _load_local_env(path: str = '.env.local') -> None:
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
    def _image_data_url(image_path: str | None) -> str:
        if not image_path:
            raise RuntimeError('front RGB image path is missing')
        path = Path(str(image_path))
        if not path.exists():
            raise RuntimeError(f'front RGB image does not exist: {path}')
        mime_type = mimetypes.guess_type(path.name)[0] or 'image/png'
        encoded = base64.b64encode(path.read_bytes()).decode('ascii')
        return f'data:{mime_type};base64,{encoded}'

    @staticmethod
    def _extract_json_content(content: str) -> str:
        text = content.strip()
        if text.startswith('```'):
            lines = [line for line in text.splitlines() if not line.strip().startswith('```')]
            text = '\n'.join(lines).strip()
        start = text.find('{')
        end = text.rfind('}')
        if start == -1 or end == -1 or end < start:
            raise RuntimeError('Qwen VLM response did not contain a JSON object')
        return text[start:end + 1]

    @staticmethod
    def _result_from_json(packet_id: str, text: str) -> VLMStrategyResult:
        try:
            data = _loads_json_lenient(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f'Qwen VLM returned invalid JSON: {text}') from exc
        visual_assessment = data.get('visual_assessment') if isinstance(data.get('visual_assessment'), dict) else {}
        strategy_suggestion = data.get('strategy_suggestion') if isinstance(data.get('strategy_suggestion'), dict) else {}
        obstacle_strategies = data.get('obstacle_strategies') if isinstance(data.get('obstacle_strategies'), list) else []
        safety_notes = data.get('safety_notes') if isinstance(data.get('safety_notes'), list) else []
        confidence = data.get('confidence', 0.5)
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.5
        return VLMStrategyResult(
            result_id=str(data.get('result_id', f'qwen_vlm_result_for_{packet_id}')),
            packet_id=packet_id,
            source='qwen_vlm',
            confidence=max(0.0, min(1.0, confidence)),
            accepted_for_control=False,
            visual_assessment={
                'image_understanding': visual_assessment.get('image_understanding', 'qwen_vlm returned no image description'),
                'sfsc_consistency': visual_assessment.get('sfsc_consistency', 'uncertain'),
                'occlusion_risk': visual_assessment.get('occlusion_risk', 'medium'),
                'morphology_risk': visual_assessment.get('morphology_risk', 'medium'),
            },
            strategy_suggestion={
                'mode': _normalize_mode(strategy_suggestion.get('mode', 'hold')),
                'target_policy': str(strategy_suggestion.get('target_policy', 'hold_and_reform')),
                'passing_order': _normalize_passing_order(strategy_suggestion.get('passing_order')),
                'reason': strategy_suggestion.get('reason', 'qwen_vlm did not provide a reason'),
            },
            obstacle_strategies=[
                _normalize_obstacle_strategy(item)
                for item in obstacle_strategies
                if isinstance(item, dict)
            ],
            safety_notes=[str(item) for item in safety_notes],
        )


class QwenDashScopeVLMPlanGenerator(QwenDashScopeVLMStrategyGenerator):
    """Qwen VLM generator that returns executable PassagePlan JSON."""

    def generate_strategy_json(self, scene_context: dict[str, Any]) -> str:
        self._load_local_env()
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise RuntimeError(f'missing API key environment variable: {self.api_key_env}')
        front_rgb = scene_context.get('front_rgb', {})
        image_url = self._image_data_url(front_rgb.get('path') or front_rgb.get('image_path'))
        prompt_context = dict(scene_context)
        prompt_context['front_rgb'] = {
            key: value
            for key, value in front_rgb.items()
            if key not in {'path', 'image_path', 'error'}
        }
        prompt_context['route_decision_focus'] = _build_route_decision_focus(scene_context)
        payload = {
            'model': self.model,
            'messages': [
                {
                    'role': 'system',
                    'content': self._passage_plan_system_prompt(),
                },
                {
                    'role': 'user',
                    'content': [
                        {
                            'type': 'text',
                            'text': json.dumps(prompt_context, ensure_ascii=False, sort_keys=True),
                        },
                        {
                            'type': 'image_url',
                            'image_url': {'url': image_url},
                        },
                    ],
                },
            ],
            'temperature': self.temperature,
            'response_format': {'type': 'json_object'},
            'max_tokens': self.max_tokens,
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
            raise RuntimeError(f'Qwen VLM plan HTTP error {exc.code}: {body}') from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f'Qwen VLM plan request failed: {exc}') from exc
        try:
            content = response_payload['choices'][0]['message']['content']
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f'Qwen VLM plan response missing message content: {response_payload}') from exc
        text = self._extract_json_content(str(content))
        try:
            payload = _loads_json_lenient(text)
            payload['source'] = 'qwen_vlm'
            if 'plan_id' in payload and not str(payload['plan_id']).endswith('_vlm'):
                payload['plan_id'] = f'{payload["plan_id"]}_vlm'
            return json.dumps(payload, ensure_ascii=False)
        except json.JSONDecodeError:
            return text

    @staticmethod
    def _passage_plan_system_prompt() -> str:
        return (
            'You are a multimodal high-level planner for multi-UAV dynamic obstacle passage. '
            'You receive one onboard front RGB image from the current frontmost UAV and one SFSC context. '
            'Return only one JSON object. Do not return Markdown, prose, explanations, motor commands, raw velocities, or trajectories. '
            'The output must exactly match PassagePlan JSON version 1.0. '
            'Required top-level keys: version, plan_id, source, mode, confidence, recover_after_last_obstacle, '
            'time_slot_interval, passing_order, slots, obstacle_strategies. '
            'Set source to qwen_vlm. passing_order must be numeric UAV ids, for example [0,1,2,3,4], never strings such as ["uav0"]. '
            'slots must be one slot per UAV and include drone_id, order_index, nominal_entry_time, nominal_exit_time. '
            'obstacle_strategies must include one strategy per obstacle in obstacle_field and every strategy must include '
            'obstacle_id, obstacle_index, mode, target_policy, obstacle_x, clear_x, observe_x. '
            'Use only obstacle strategy modes snake_sequence and bypass. '
            'Do not use formation for any obstacle_strategy; formation recovery happens only after the full obstacle zone is cleared. '
            'Solid obstacles must use mode bypass. For solid target_policy choose one high-level route family: '
            'edge_bypass, side_bypass_left, side_bypass_right, split_by_lane_bypass, overpass, underpass, or hybrid_over_or_side. '
            'Choose only the obstacle-level route family; the bottom layer maps it to per-UAV continuous execution and collision avoidance. '
            'Never choose underpass unless SFSC explicitly reports underpass_feasible=true and bottom_clearance_z is sufficient for the UAV radius. '
            'Never choose overpass unless SFSC explicitly reports overpass_reasonable=true and required_overpass_z is modest. '
            'When both underpass and overpass are feasible, compare underpass_vertical_delta_z and overpass_vertical_delta_z, or follow preferred_vertical_route. '
            'For floating/elevated solid obstacles with open bottom clearance, prefer underpass when it is the shorter vertical route. '
            'For non-floating solids with no bottom clearance, do not choose underpass. '
            'Use side_bypass_left/right for asymmetric side clearance, and split_by_lane_bypass for central clutter where the formation can naturally divide. '
            'Use vertical passage over side bypass only when the corresponding vertical route is explicitly feasible and shorter/reasonable. '
            'Use hybrid_over_or_side when either side or top route is acceptable and the controller should select by local cost. '
            'Aperture obstacles should use snake_sequence and predictive_center_crossing when single-file passage is feasible. '
            'The deterministic_reference_plan is a conservative safety fallback, not a mandatory answer. '
            'When front RGB morphology and SFSC both indicate a validator-safe higher-level alternative, you may revise '
            'local obstacle mode, target_policy, observe_x, or clear_x instead of copying the fallback. '
            'Do not change global passing_order during obstacle-zone execution unless scene_context explicitly allows it. '
            'For local timing, prefer earlier observe_x or longer clear_x for uncertain/high-risk obstacles; do not delay observe_x '
            'unless the scene_context explicitly requests a delayed commitment. '
            'If scene_context contains vlm_replanning_objective, follow its encouraged_safe_modifications and forbidden_modifications. '
            'If scene_context contains route_decision_focus, use it as the primary concise route-family decision aid for solid obstacles. '
            'Do not ignore route_decision_focus by copying deterministic_reference_plan when it explicitly recommends a feasible route-family change. '
            'Do not invent pass_z, slot_duration, velocity commands, clearance thresholds, or fields not in PassagePlan. '
            'If uncertain, copy deterministic_reference_plan exactly, set source to qwen_vlm, and keep the same plan_id with suffix _vlm. '
            'If you intentionally revise the fallback, use a plan_id suffix such as _vlm_revised.'
        )


def build_multimodal_vlm_input_packet(
    *,
    packet_id: str,
    scenario_id: str,
    frame: Any,
    sfsc_events: list[dict[str, Any]],
) -> MultimodalVLMInputPacket:
    """Build the VLM input packet from one front RGB frame and recent SFSC events."""

    return MultimodalVLMInputPacket(
        packet_id=str(packet_id),
        scenario_id=str(scenario_id),
        sim_time=float(frame.sim_time),
        trigger_label=str(frame.label),
        trigger_reason=str(frame.reason),
        front_rgb={
            'image_path': str(frame.path),
            'view_type': 'onboard_front_rgb',
            'camera_name': getattr(frame, 'camera_name', None),
            'source_uav_id': getattr(frame, 'source_uav_id', None),
            'selection_policy': getattr(frame, 'selection_policy', 'frontmost_uav'),
            'width': int(frame.width),
            'height': int(frame.height),
            'render_mode': getattr(frame, 'render_mode', 'uav_front_rgb'),
            'available': not bool(getattr(frame, 'error', None)),
            'error': getattr(frame, 'error', None),
        },
        sfsc_context={
            'context_type': 'SFSC',
            'context_name': 'Sensor-Fused Structured Context',
            'source_note': (
                'SFSC is the fast non-visual sensor-fusion context: odometry/IMU, '
                'range sensing, inter-UAV communication, controller state, and '
                'local obstacle estimates. In this MuJoCo demo it is reproduced '
                'from runtime state for controlled experiments.'
            ),
            'recent_events': [_compact_event(event) for event in sfsc_events],
        },
        request={
            'target_model_role': 'multimodal_high_level_strategy_advisor',
            'input_modalities': ['frontmost_uav_rgb', 'sfsc'],
            'output_format': 'json_only',
            'expected_output_schema': _expected_vlm_output_schema(),
            'hard_rules': [
                'VLM must not output motor thrust.',
                'VLM must not output raw per-frame velocity commands.',
                'VLM strategy suggestions must be validated before execution.',
                'passing_order must be numeric UAV ids such as [0, 1, 2, 3, 4], never strings such as ["uav0"].',
                'Do not invent numeric safety thresholds that are not present in SFSC.',
                'Prefer qualitative safety notes unless SFSC explicitly provides a numeric value.',
                'If visual evidence conflicts with SFSC, report the conflict instead of forcing a strategy.',
                'If uncertain, recommend keeping the deterministic fallback strategy.',
            ],
        },
    )


def summarize_multimodal_vlm_packet(packet: MultimodalVLMInputPacket, packet_path: str) -> dict[str, Any]:
    data = packet.to_dict()
    front_rgb = data['front_rgb']
    recent_events = data['sfsc_context'].get('recent_events', [])
    return {
        'summary': (
            '已构造多模态 VLM 输入包：当前最前方 UAV 前视 RGB + 最近 SFSC 上下文；'
            '本阶段只记录和展示，不直接改变控制。'
        ),
        'details': {
            '链路阶段': 'VLM 视觉复核链路（输入准备，不进入控制）',
            '一句话说明': '把当前前视 RGB 和最近 SFSC 打包，供 VLM 做视觉复核。',
            '输入包编号': data['packet_id'],
            '输入包路径': packet_path,
            '前视图路径': front_rgb.get('image_path'),
            '视觉来源': (
                f'UAV {front_rgb.get("source_uav_id")}'
                if front_rgb.get('source_uav_id') is not None
                else 'unknown'
            ),
            '相机名称': front_rgb.get('camera_name'),
            '视角策略': front_rgb.get('selection_policy'),
            '视觉帧可用': bool(front_rgb.get('available')),
            '融合模态': 'frontmost_uav_rgb + SFSC',
            '引用SFSC事件数量': len(recent_events),
            '引用SFSC事件': [
                f'#{event.get("event_id")} {event.get("title")} ({event.get("category")})'
                for event in recent_events
            ],
            'VLM任务': '根据前视图像和 SFSC 约束输出可验证的高层通行策略建议',
            'VLM期望输出': [
                'visual_assessment: 图像理解、遮挡风险、与 SFSC 是否一致',
                'strategy_suggestion: 穿越/侧绕/越顶/等待/重排序等高层建议',
                'obstacle_strategies: 针对可见障碍的策略建议',
                'safety_notes: 风险解释和必须保留的安全约束',
            ],
        },
    }


def summarize_vlm_strategy_result(result: VLMStrategyResult) -> dict[str, Any]:
    data = result.to_dict()
    assessment = data['visual_assessment']
    suggestion = data['strategy_suggestion']
    source_text = 'Mock VLM' if str(data.get('source')) == 'mock_vlm' else '真实 Qwen VLM'
    return {
        'summary': (
            f'{source_text} 已返回高层策略建议：mode={suggestion.get("mode")}，'
            f'policy={suggestion.get("target_policy")}，confidence={float(data["confidence"]):.2f}；'
            '当前仅展示，不进入控制。'
        ),
        'details': {
            '结果编号': data['result_id'],
            '对应输入包': data['packet_id'],
            '来源': data['source'],
            '置信度': f'{float(data["confidence"]):.2f}',
            '是否进入控制': bool(data['accepted_for_control']),
            '图像理解': assessment.get('image_understanding'),
            'SFSC一致性': assessment.get('sfsc_consistency'),
            '遮挡风险': assessment.get('occlusion_risk'),
            '形态风险': assessment.get('morphology_risk'),
            '建议模式': suggestion.get('mode'),
            '目标策略': suggestion.get('target_policy'),
            '建议通行顺序': suggestion.get('passing_order'),
            '建议原因': suggestion.get('reason'),
            '障碍策略建议': [
                (
                    f'{item.get("obstacle_id")}: mode={item.get("mode")}, '
                    f'policy={item.get("target_policy")}, note={item.get("risk_note")}'
                )
                for item in data.get('obstacle_strategies', [])
            ],
            '安全备注': list(data.get('safety_notes', [])),
        },
    }


def _compact_event(event: dict[str, Any]) -> dict[str, Any]:
    details = event.get('details', {})
    return {
        'event_id': event.get('event_id'),
        'sim_time': event.get('sim_time'),
        'category': event.get('category'),
        'title': event.get('title'),
        'summary': event.get('summary'),
        'details': details,
    }


def _expected_vlm_output_schema() -> dict[str, Any]:
    return {
        'version': '1.0',
        'source': 'vlm',
        'confidence': 'float in [0, 1]',
        'visual_assessment': {
            'image_understanding': 'short description of obstacle appearance',
            'sfsc_consistency': 'consistent | partially_consistent | conflict | uncertain',
            'occlusion_risk': 'low | medium | high',
            'morphology_risk': 'low | medium | high',
        },
        'strategy_suggestion': {
            'mode': 'snake_sequence | bypass | hold',
            'target_policy': (
                'predictive_center_crossing | edge_bypass | side_bypass_left | side_bypass_right | '
                'split_by_lane_bypass | overpass | underpass | hybrid_over_or_side | hold_and_reform'
            ),
            'passing_order': 'list of UAV ids when needed',
            'reason': 'short executable explanation',
        },
        'obstacle_strategies': [
            {
                'obstacle_id': 'string',
                'mode': 'snake_sequence | bypass | hold',
                'target_policy': 'string',
                'risk_note': 'short note',
            }
        ],
        'safety_notes': ['short safety constraints'],
    }


def _build_route_decision_focus(scene_context: dict[str, Any]) -> dict[str, Any]:
    reference_plan = scene_context.get('deterministic_reference_plan', {})
    reference_by_id = {}
    if isinstance(reference_plan, dict):
        for strategy in reference_plan.get('obstacle_strategies', []):
            if isinstance(strategy, dict) and strategy.get('obstacle_id') is not None:
                reference_by_id[str(strategy.get('obstacle_id'))] = strategy

    solid_items = []
    for obstacle in scene_context.get('obstacle_field', []):
        if not isinstance(obstacle, dict) or obstacle.get('function') != 'solid':
            continue
        obstacle_id = str(obstacle.get('obstacle_id'))
        preferred_vertical = obstacle.get('preferred_vertical_route')
        underpass_feasible = bool(obstacle.get('underpass_feasible') is True)
        overpass_reasonable = bool(obstacle.get('overpass_reasonable') is True)
        deterministic_policy = reference_by_id.get(obstacle_id, {}).get('target_policy')
        if preferred_vertical == 'underpass' and underpass_feasible:
            decision_hint = (
                'SFSC geometry indicates a floating/open-bottom obstacle. '
                'Choose target_policy underpass unless the RGB image clearly contradicts the open bottom.'
            )
        elif preferred_vertical == 'overpass' and overpass_reasonable:
            decision_hint = (
                'SFSC geometry indicates the top route is the shorter feasible vertical route. '
                'Choose target_policy overpass unless RGB contradicts top clearance.'
            )
        elif not underpass_feasible and not overpass_reasonable:
            decision_hint = 'Vertical routes are not feasible/reasonable; choose a side or split bypass route.'
        else:
            decision_hint = 'Compare vertical_delta_z fields and choose the shortest feasible route family.'
        solid_items.append({
            'obstacle_id': obstacle_id,
            'obstacle_type': obstacle.get('obstacle_type'),
            'deterministic_fallback_target_policy': deterministic_policy,
            'bottom_clearance_z': obstacle.get('bottom_clearance_z'),
            'underpass_feasible': obstacle.get('underpass_feasible'),
            'estimated_underpass_target_z': obstacle.get('estimated_underpass_target_z'),
            'underpass_vertical_delta_z': obstacle.get('underpass_vertical_delta_z'),
            'required_overpass_z': obstacle.get('required_overpass_z'),
            'overpass_reasonable': obstacle.get('overpass_reasonable'),
            'overpass_vertical_delta_z': obstacle.get('overpass_vertical_delta_z'),
            'preferred_vertical_route': preferred_vertical,
            'route_affordances': obstacle.get('route_affordances'),
            'decision_hint': decision_hint,
        })
    return {
        'purpose': 'Concise SFSC route-family decision summary for solid obstacles; use this before copying fallback.',
        'hard_rules': [
            'Do not choose underpass unless underpass_feasible is true.',
            'Do not choose overpass unless overpass_reasonable is true.',
            'If preferred_vertical_route is underpass and RGB does not contradict open bottom clearance, choose target_policy underpass.',
            'If vertical routes are infeasible or visually contradicted, choose side_bypass_left/right or split_by_lane_bypass.',
        ],
        'solid_obstacles': solid_items,
    }


def _extract_candidate_obstacle_ids(events: list[dict[str, Any]]) -> list[str]:
    obstacle_ids: list[str] = []
    for event in events:
        details = event.get('details', {})
        for key in ('待规划障碍', '新发现障碍', '策略缓存已覆盖'):
            values = details.get(key, [])
            if isinstance(values, list):
                obstacle_ids.extend(str(item) for item in values)
        for line in details.get('可见障碍摘要', []) if isinstance(details.get('可见障碍摘要', []), list) else []:
            obstacle_id = str(line).split(':', 1)[0].strip()
            if obstacle_id:
                obstacle_ids.append(obstacle_id)
    deduped: list[str] = []
    for obstacle_id in obstacle_ids:
        if obstacle_id and obstacle_id not in deduped:
            deduped.append(obstacle_id)
    return deduped[:3]


def _suggest_mode_from_events(events: list[dict[str, Any]]) -> tuple[str, str]:
    text = ' '.join(
        f'{event.get("summary", "")} {event.get("details", "")}'
        for event in events
    ).lower()
    if 'solid' in text or '实体' in text or 'bypass' in text:
        return 'bypass', 'edge_bypass'
    if 'aperture' in text or 'gate' in text or '穿越' in text or '洞' in text:
        return 'snake_sequence', 'predictive_center_crossing'
    return 'snake_sequence', 'predictive_center_crossing'


def _loads_json_lenient(text: str) -> dict[str, Any]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        repaired = _repair_truncated_json_object(text)
        data = json.loads(repaired)
    if not isinstance(data, dict):
        raise json.JSONDecodeError('top-level VLM output is not a JSON object', str(text), 0)
    return data


def _repair_truncated_json_object(text: str) -> str:
    """Repair common VLM truncation where closing brackets/braces are missing."""

    repaired = str(text).strip()
    stack: list[str] = []
    in_string = False
    escape = False
    for char in repaired:
        if escape:
            escape = False
            continue
        if char == '\\':
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == '{':
            stack.append('}')
        elif char == '[':
            stack.append(']')
        elif char in {'}', ']'}:
            if stack and stack[-1] == char:
                stack.pop()
    if in_string:
        repaired += '"'
    while stack:
        repaired += stack.pop()
    return repaired


def _normalize_passing_order(value: Any) -> list[int]:
    if not isinstance(value, list):
        return [0, 1, 2, 3, 4]
    normalized: list[int] = []
    for item in value:
        drone_id = _normalize_uav_id(item)
        if drone_id is not None and drone_id not in normalized:
            normalized.append(drone_id)
    return normalized or [0, 1, 2, 3, 4]


def _normalize_uav_id(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    text = str(value).strip().lower()
    if text.startswith('uav'):
        text = text[3:]
    if text.isdigit():
        return int(text)
    return None


def _normalize_mode(value: Any) -> str:
    mode = str(value)
    if mode in {'formation', 'snake_sequence', 'bypass', 'hold'}:
        return mode
    return 'hold'


def _normalize_obstacle_strategy(item: dict[str, Any]) -> dict[str, Any]:
    return {
        'obstacle_id': str(item.get('obstacle_id', 'visible_obstacle')),
        'mode': _normalize_mode(item.get('mode', 'hold')),
        'target_policy': str(item.get('target_policy', 'hold_and_reform')),
        'risk_note': str(item.get('risk_note', 'VLM did not provide a risk note')),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)
