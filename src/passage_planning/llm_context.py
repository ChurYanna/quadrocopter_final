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
    """Build the SFSC prompt payload for an LLM strategy generator.

    SFSC means Sensor-Fused Structured Context.  In a real system it is the
    compact non-visual context produced by fast sensor fusion: odometry/IMU,
    range sensing, inter-UAV communication, controller state and local obstacle
    estimates.  The current MuJoCo demo constructs the same payload from
    simulator state so the planner can be reproduced deterministically.

    The LLM receives SFSC plus optional visual evidence later, not body ids,
    qpos/qvel arrays or low-level controller errors.  This keeps the model in
    the high-level strategy layer.
    """
    context = {
        'task': 'multi_uav_dynamic_obstacle_passage',
        'context_type': 'SFSC',
        'context_name': 'Sensor-Fused Structured Context',
        'context_source': (
            'fast non-visual sensor fusion: odometry/IMU, range sensing, '
            'inter-UAV communication, controller state, and local obstacle estimates'
        ),
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
            _obstacle_context_item(obstacle, formation)
            for obstacle in field.obstacles
        ],
        'mission_preference': {
            'priority': mission.priority,
            'allow_disband': bool(mission.allow_disband),
            'recover_after_last_obstacle': bool(mission.recover_after_last_obstacle),
            'preferred_mode': mission.preferred_mode,
            'solid_route_preference': (
                'choose underpass only when underpass_feasible is true; choose overpass only when overpass_reasonable is true; '
                'when both vertical routes are feasible, follow preferred_vertical_route or the smaller vertical_delta_z'
            ),
        },
        'allowed_modes': ['snake_sequence', 'bypass'],
        'allowed_target_policies': [
            'predictive_center_crossing',
            'edge_bypass',
            'side_bypass_left',
            'side_bypass_right',
            'split_by_lane_bypass',
            'overpass',
            'underpass',
            'hybrid_over_or_side',
            'hold_and_reform',
        ],
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
                (
                    'solid obstacles MUST use mode bypass and one bypass target_policy: edge_bypass, '
                    'side_bypass_left, side_bypass_right, split_by_lane_bypass, overpass, underpass, '
                    'or hybrid_over_or_side'
                ),
                'for floating solids with underpass_feasible=true, prefer underpass before overpass or side bypass',
                'for tall solids whose required_overpass_z is high, do not choose overpass',
                'for non-floating solids or unknown bottom clearance, do not choose underpass',
                'when both underpass and overpass are feasible, compare underpass_vertical_delta_z and overpass_vertical_delta_z',
                'VLM/LLM route policies are high-level route families only; do not assign per-UAV low-level paths',
                'formation mode is NOT allowed for obstacle_strategies; formation recovery happens only after the full obstacle zone is cleared',
                'aperture obstacles SHOULD use mode snake_sequence and target_policy predictive_center_crossing when single-file passage is feasible',
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


def _obstacle_context_item(obstacle, formation: FormationState) -> dict[str, Any]:
    size_z = None if obstacle.size_z is None else float(obstacle.size_z)
    bottom_clearance_z = None
    top_height_z = None
    required_overpass_z = None
    estimated_underpass_target_z = None
    underpass_vertical_delta_z = None
    overpass_vertical_delta_z = None
    overpass_reasonable = None
    underpass_feasible = None
    preferred_vertical_route = None
    if obstacle.function == 'solid' and size_z is not None:
        bottom_clearance_z = float(obstacle.pass_z) - 0.5 * abs(size_z)
        top_height_z = float(obstacle.pass_z) + 0.5 * abs(size_z)
        required_overpass_z = float(top_height_z + formation.uav_radius_z + 0.08)
        estimated_underpass_target_z = float(
            max(formation.initial_z, bottom_clearance_z - formation.uav_radius_z - 0.08)
        )
        underpass_vertical_delta_z = float(abs(estimated_underpass_target_z - formation.initial_z))
        overpass_vertical_delta_z = float(abs(required_overpass_z - formation.initial_z))
        overpass_reasonable = bool(required_overpass_z <= 3.40)
        required_underpass_z = float(formation.initial_z + formation.uav_radius_z + 0.08)
        underpass_feasible = bool(bottom_clearance_z >= required_underpass_z)
        if underpass_feasible and overpass_reasonable:
            preferred_vertical_route = (
                'underpass'
                if underpass_vertical_delta_z <= overpass_vertical_delta_z
                else 'overpass'
            )
    route_affordances = []
    if obstacle.function == 'solid':
        route_affordances.extend(['side_bypass_left', 'side_bypass_right', 'split_by_lane_bypass'])
        if overpass_reasonable:
            route_affordances.append('overpass')
        if underpass_feasible:
            route_affordances.insert(0, 'underpass')
    return {
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
        'size_z': size_z,
        'bottom_clearance_z': bottom_clearance_z,
        'top_height_z': top_height_z,
        'required_overpass_z': required_overpass_z,
        'estimated_underpass_target_z': estimated_underpass_target_z,
        'underpass_vertical_delta_z': underpass_vertical_delta_z,
        'overpass_vertical_delta_z': overpass_vertical_delta_z,
        'overpass_reasonable': overpass_reasonable,
        'underpass_feasible': underpass_feasible,
        'preferred_vertical_route': preferred_vertical_route,
        'route_affordances': route_affordances,
    }
