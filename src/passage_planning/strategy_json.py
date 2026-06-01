from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .structures import ObstaclePassageStrategy, PassagePlan, PassageSlot


STRATEGY_JSON_VERSION = '1.0'


class StrategyJsonError(ValueError):
    """Raised when a strategy JSON payload is malformed."""


def passage_plan_to_dict(plan: PassagePlan) -> dict[str, Any]:
    """Serialize a PassagePlan to a JSON-friendly dictionary."""
    return {
        'version': STRATEGY_JSON_VERSION,
        'plan_id': plan.plan_id,
        'source': plan.source,
        'mode': plan.mode,
        'confidence': float(plan.confidence),
        'recover_after_last_obstacle': bool(plan.recover_after_last_obstacle),
        'time_slot_interval': float(plan.time_slot_interval),
        'passing_order': [int(drone_id) for drone_id in plan.passing_order],
        'slots': [
            {
                'drone_id': int(slot.drone_id),
                'order_index': int(slot.order_index),
                'nominal_entry_time': float(slot.nominal_entry_time),
                'nominal_exit_time': float(slot.nominal_exit_time),
            }
            for slot in plan.slots
        ],
        'obstacle_strategies': [
            {
                'obstacle_id': strategy.obstacle_id,
                'obstacle_index': int(strategy.obstacle_index),
                'mode': strategy.mode,
                'target_policy': strategy.target_policy,
                'obstacle_x': float(strategy.obstacle_x),
                'clear_x': float(strategy.clear_x),
                'observe_x': float(strategy.observe_x),
            }
            for strategy in plan.obstacle_strategies
        ],
    }


def passage_plan_from_dict(payload: dict[str, Any]) -> PassagePlan:
    """Deserialize a strategy dictionary into a PassagePlan."""
    if not isinstance(payload, dict):
        raise StrategyJsonError('strategy payload must be a JSON object')

    version = str(payload.get('version', ''))
    if version != STRATEGY_JSON_VERSION:
        raise StrategyJsonError(f'unsupported strategy JSON version: {version!r}')

    required_keys = {
        'plan_id',
        'source',
        'mode',
        'confidence',
        'recover_after_last_obstacle',
        'time_slot_interval',
        'passing_order',
        'slots',
        'obstacle_strategies',
    }
    missing = sorted(required_keys - set(payload))
    if missing:
        raise StrategyJsonError(f'missing strategy keys: {missing}')

    passing_order = _tuple_of_ints(payload['passing_order'], 'passing_order')
    slots = tuple(_slot_from_dict(item, passing_order) for item in _list(payload['slots'], 'slots'))
    obstacle_strategies = tuple(
        _obstacle_strategy_from_dict(item)
        for item in _list(payload['obstacle_strategies'], 'obstacle_strategies')
    )

    return PassagePlan(
        plan_id=str(payload['plan_id']),
        source=str(payload['source']),
        mode=_mode(payload['mode'], 'mode'),
        confidence=float(payload['confidence']),
        recover_after_last_obstacle=bool(payload['recover_after_last_obstacle']),
        time_slot_interval=float(payload['time_slot_interval']),
        passing_order=passing_order,
        slots=slots,
        obstacle_strategies=obstacle_strategies,
    )


def passage_plan_to_json(plan: PassagePlan, indent: int = 2) -> str:
    return json.dumps(passage_plan_to_dict(plan), ensure_ascii=False, indent=indent, sort_keys=True)


def passage_plan_from_json(text: str) -> PassagePlan:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise StrategyJsonError(f'invalid strategy JSON: {exc}') from exc
    return passage_plan_from_dict(payload)


def save_passage_plan_json(plan: PassagePlan, path: str | Path) -> None:
    Path(path).write_text(passage_plan_to_json(plan) + '\n', encoding='utf-8')


def load_passage_plan_json(path: str | Path) -> PassagePlan:
    return passage_plan_from_json(Path(path).read_text(encoding='utf-8'))


def _slot_from_dict(payload: Any, passing_order: tuple[int, ...]) -> PassageSlot:
    payload = _dict(payload, 'slot')
    order_index = _int_like(_required(payload, 'order_index', 'slot'), 'slot.order_index')
    drone_id_raw = payload.get('drone_id', None)
    if drone_id_raw is None:
        if 0 <= order_index < len(passing_order):
            drone_id = int(passing_order[order_index])
        else:
            raise StrategyJsonError('missing slot.drone_id and order_index cannot map to passing_order')
    else:
        drone_id = _int_like(drone_id_raw, 'slot.drone_id')
    return PassageSlot(
        drone_id=drone_id,
        order_index=order_index,
        nominal_entry_time=float(_required(payload, 'nominal_entry_time', 'slot')),
        nominal_exit_time=float(_required(payload, 'nominal_exit_time', 'slot')),
    )


def _obstacle_strategy_from_dict(payload: Any) -> ObstaclePassageStrategy:
    payload = _dict(payload, 'obstacle_strategy')
    return ObstaclePassageStrategy(
        obstacle_id=str(_required(payload, 'obstacle_id', 'obstacle_strategy')),
        obstacle_index=_int_like(
            _required(payload, 'obstacle_index', 'obstacle_strategy'),
            'obstacle_strategy.obstacle_index',
            one_based_name=True,
        ),
        mode=_mode(_required(payload, 'mode', 'obstacle_strategy'), 'obstacle_strategy.mode'),
        target_policy=str(_required(payload, 'target_policy', 'obstacle_strategy')),
        obstacle_x=float(_required(payload, 'obstacle_x', 'obstacle_strategy')),
        clear_x=float(_required(payload, 'clear_x', 'obstacle_strategy')),
        observe_x=float(_required(payload, 'observe_x', 'obstacle_strategy')),
    )


def _required(payload: dict[str, Any], key: str, where: str) -> Any:
    if key not in payload:
        raise StrategyJsonError(f'missing {where}.{key}')
    return payload[key]


def _dict(payload: Any, name: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise StrategyJsonError(f'{name} must be a JSON object')
    return payload


def _list(payload: Any, name: str) -> list[Any]:
    if not isinstance(payload, list):
        raise StrategyJsonError(f'{name} must be a JSON array')
    return payload


def _tuple_of_ints(payload: Any, name: str) -> tuple[int, ...]:
    return tuple(_int_like(item, name) for item in _list(payload, name))


def _int_like(value: Any, name: str, one_based_name: bool = False) -> int:
    """Parse integers and common LLM labels such as `gate1` or `uav_0`.

    `one_based_name=True` is used for obstacle names like gate1/gate2, where
    the human label is one-based but PassagePlan uses zero-based indices.
    """
    if isinstance(value, bool):
        raise StrategyJsonError(f'{name} must be an integer-like value, got boolean')
    if isinstance(value, int):
        return int(value)
    text = str(value).strip()
    if re.fullmatch(r'[+-]?\d+', text):
        return int(text)
    match = re.search(r'(\d+)$', text)
    if match:
        parsed = int(match.group(1))
        if one_based_name and re.search(r'[A-Za-z]', text):
            return parsed - 1
        return parsed
    raise StrategyJsonError(f'{name} must be an integer-like value, got {value!r}')


def _mode(value: Any, name: str) -> str:
    mode = str(value)
    if mode not in {'formation', 'snake_sequence', 'bypass'}:
        raise StrategyJsonError(f'{name} has invalid mode: {mode!r}')
    return mode
