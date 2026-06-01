from __future__ import annotations

from typing import Any

from .strategy_json import passage_plan_to_dict
from .structures import FormationState, MissionPreference, ObstacleField
from .structures import PassagePlan


def build_llm_scene_context(
    field: ObstacleField,
    formation: FormationState,
    mission: MissionPreference,
    reference_plan: PassagePlan | None = None,
) -> dict[str, Any]:
    """Build the structured prompt payload for an LLM strategy generator.

    The LLM receives semantic planning context, not raw MuJoCo state.  This
    keeps the model in the high-level strategy layer and makes its output
    independent from simulator-specific objects.
    """
    context = {
        'task': 'multi_uav_dynamic_obstacle_passage',
        'uav_team': {
            'num_uavs': int(formation.num_uavs),
            'formation': 'line',
            'original_y': [float(y) for y in formation.original_y],
            'initial_z': float(formation.initial_z),
            'uav_radius_xy': float(formation.uav_radius_xy),
            'uav_radius_z': float(formation.uav_radius_z),
            'formation_width_y': float(formation.width_y),
            'single_file_width': float(formation.single_file_width),
            'nominal_speed_x': float(formation.nominal_speed_x),
        },
        'obstacle_field': [
            {
                'obstacle_id': obstacle.obstacle_id,
                'obstacle_type': obstacle.obstacle_type,
                'function': obstacle.function,
                'x': float(obstacle.x),
                'center_y': float(obstacle.center_y),
                'pass_z': float(obstacle.pass_z),
                'aperture_width': None if obstacle.aperture_width is None else float(obstacle.aperture_width),
                'motion_axis': obstacle.motion_axis,
                'motion_model': obstacle.motion_model,
                'motion_amplitude': float(obstacle.motion_amplitude),
                'motion_omega': float(obstacle.motion_omega),
                'risk_level': obstacle.risk_level,
                'size_x': None if obstacle.size_x is None else float(obstacle.size_x),
                'size_y': None if obstacle.size_y is None else float(obstacle.size_y),
                'size_z': None if obstacle.size_z is None else float(obstacle.size_z),
            }
            for obstacle in field.obstacles
        ],
        'mission_preference': {
            'priority': mission.priority,
            'allow_disband': bool(mission.allow_disband),
            'recover_after_last_obstacle': bool(mission.recover_after_last_obstacle),
            'preferred_mode': mission.preferred_mode,
        },
        'allowed_modes': ['formation', 'snake_sequence', 'bypass'],
        'allowed_target_policies': ['predictive_center_crossing', 'edge_bypass', 'hold_and_reform'],
        'required_output': {
            'schema': 'PassagePlan',
            'version': '1.0',
            'format': 'json_only',
            'hard_rules': [
                'passing_order is a list of UAV ids, not obstacle ids',
                'passing_order length must equal uav_team.num_uavs',
                'slots is a list of UAV time slots, not obstacle time slots',
                'each slot must include drone_id, order_index, nominal_entry_time, nominal_exit_time',
                'obstacle_strategies length must equal obstacle_field length',
                'each obstacle_strategy must include obstacle_id, obstacle_index, mode, target_policy, obstacle_x, clear_x, observe_x',
                'solid obstacles MUST use mode bypass and target_policy edge_bypass',
                'aperture obstacles SHOULD use mode snake_sequence and target_policy predictive_center_crossing when formation width is larger than aperture width',
                'do not invent fields such as slot_start_time, slot_duration, pass_z, or timing_adjustment_s',
                'if uncertain, copy deterministic_reference_plan exactly and only set source to qwen',
            ],
            'field_meanings': {
                'passing_order': 'UAV ids in crossing order, e.g. [0, 1, 2, 3, 4]',
                'slots': 'one slot per UAV, ordered by passing_order',
                'obstacle_strategies': 'one strategy per obstacle, ordered by obstacle_index',
            },
        },
    }
    if reference_plan is not None:
        context['deterministic_reference_plan'] = passage_plan_to_dict(reference_plan)
    return context
