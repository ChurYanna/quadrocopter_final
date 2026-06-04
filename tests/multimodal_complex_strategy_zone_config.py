from tests.dynamic_gate_common import DynamicGateSnakeBase


class MultimodalComplexStrategyZoneConfig(DynamicGateSnakeBase):
    """Complex multimodal scene designed to expose VLM-vs-fallback differences."""

    SCENE = 'assets/skydio_x2/scene_multimodal_complex_strategy_zone5.xml'

    FINAL_TARGET_X = 112.0
    MAX_TOTAL_TIME = 280.0

    GATE_OBSERVE_DISTANCE_X = 5.6
    GATE_PASS_CLEAR_X = 1.25
    POST_REFORM_CRUISE_X = 2.2

    GATE_MOTION_SPECS = [
        {
            'prefix': 'gate1',
            'amplitude': 2.65,
            'omega': 0.36,
            'phase': 0.20,
            'blend': 0.20,
            'omega2': 0.42,
            'phase2': 1.10,
        },
        {
            'prefix': 'gate2',
            'amplitude': 1.85,
            'omega': 0.30,
            'phase': 1.40,
            'blend': 0.18,
            'omega2': 0.34,
            'phase2': 2.20,
        },
        {
            'prefix': 'gate3',
            'amplitude': 2.90,
            'omega': 0.34,
            'phase': 2.55,
            'blend': 0.22,
            'omega2': 0.38,
            'phase2': 0.85,
        },
        {
            'prefix': 'gate4',
            'amplitude': 3.10,
            'omega': 0.35,
            'phase': 3.70,
            'blend': 0.24,
            'omega2': 0.40,
            'phase2': 2.70,
        },
    ]

    SOLID_BYPASS_SPECS = [
        {
            'prefix': 'over_beam',
            'type': 'long_floating_block_with_under_over_options',
            'x': 19.0,
            'base_center_y': 0.0,
            'z': 1.20,
            'size_x': 2.80,
            'size_y': 7.80,
            'size_z': 2.40,
            'amplitude': 0.45,
            'omega': 0.24,
            'phase': 0.65,
            'route_policy': 'over',
            'speed_x': 0.90,
            'align_speed_x': 0.42,
            'top_clearance': 0.45,
            'activate_x': 5.30,
            'clear_x': 2.80,
            'risk_level': 'high',
        },
        {
            'prefix': 'cone_cluster',
            'type': 'stacked_cone_like_cluster',
            'x': 45.0,
            'base_center_y': 0.0,
            'z': 0.65,
            'size_x': 1.90,
            'size_y': 2.20,
            'size_z': 2.30,
            'amplitude': 0.70,
            'omega': 0.28,
            'phase': 1.25,
            'route_policy': 'split_by_lane',
            'speed_x': 0.92,
            'align_speed_x': 0.42,
            'side_clearance': 0.90,
            'clear_x': 2.70,
            'risk_level': 'medium',
        },
        {
            'prefix': 'tall_tower',
            'type': 'very_tall_side_bypass_obstacle',
            'x': 58.0,
            'base_center_y': 0.0,
            'z': 3.40,
            'size_x': 2.10,
            'size_y': 5.10,
            'size_z': 6.80,
            'amplitude': 0.35,
            'omega': 0.22,
            'phase': 2.10,
            'route_policy': 'right',
            'speed_x': 0.88,
            'align_speed_x': 0.40,
            'side_clearance': 1.05,
            'clear_x': 3.10,
            'risk_level': 'high',
        },
        {
            'prefix': 'side_box_d',
            'type': 'floating_underpass_box_after_elevated_gate',
            'x': 86.0,
            'base_center_y': 0.0,
            'z': 2.25,
            'size_x': 2.10,
            'size_y': 5.20,
            'size_z': 1.10,
            'amplitude': 0.55,
            'omega': 0.30,
            'phase': 3.10,
            'route_policy': 'split_by_lane',
            'speed_x': 0.90,
            'align_speed_x': 0.42,
            'side_clearance': 0.95,
            'clear_x': 2.90,
            'risk_level': 'medium',
        },
    ]

    SOLID_BYPASS_ACTIVATE_X = 5.80
    SOLID_BYPASS_CLEAR_X = 3.20
    SOLID_BYPASS_SIDE_CLEARANCE = 0.98
    SOLID_BYPASS_TOP_CLEARANCE = 0.50
    SOLID_BYPASS_SPEED_X = 0.90
    SOLID_BYPASS_Y_KP = 1.05

    SNAKE_SPACING_X = 0.84
    SNAKE_MAX_EXTRA_GAP_X = 1.40
    SNAKE_MAX_SPEED_X = 1.55
    REACTIVE_Y_KP = 1.95
    MAX_CMD_SPEED_XY = 3.05
    CMD_SLEW_RATE_XY = 4.10
    MAX_CMD_SPEED_Z = 1.30
    CMD_SLEW_RATE_Z = 2.10
    HEIGHT_HOLD_KP = 2.35
    HEIGHT_HOLD_DAMPING = 1.75

    @staticmethod
    def _latest_multimodal_context(visual_capture):
        if visual_capture is None:
            return None
        frame = getattr(visual_capture, 'last_frame_record', None)
        if frame is None or getattr(frame, 'error', None):
            return None
        return {
            'front_rgb': {
                'path': frame.path,
                'view_type': 'onboard_front_rgb',
                'camera_name': frame.camera_name,
                'source_uav_id': frame.source_uav_id,
                'selection_policy': frame.selection_policy,
                'width': frame.width,
                'height': frame.height,
                'sim_time': frame.sim_time,
                'trigger_label': frame.label,
                'trigger_reason': frame.reason,
            },
            'multimodal_policy': {
                'external_strategy_source': 'qwen_vlm',
                'input_modalities': ['frontmost_uav_rgb', 'sfsc'],
                'control_entry_rule': 'PassagePlan must pass StrategyValidator before merging into OnlineStrategyBuffer',
            },
            'vlm_replanning_objective': {
                'scenario_role': 'online_multimodal_strategy_refinement_after_stable_bottom_layer',
                'deterministic_reference_plan_role': (
                    'safety fallback and bottom-layer continuity anchor; VLM may refine local obstacle strategy '
                    'when visual evidence and SFSC indicate a validator-safe improvement'
                ),
                'encouraged_safe_modifications': [
                    'keep aperture obstacles in snake_sequence when single-file passage is feasible',
                    'adjust observe_x or clear_x when visual morphology requires earlier commitment or longer clearance',
                    (
                        'refine solid-obstacle target_policy as a route family: side_bypass_left, side_bypass_right, '
                        'split_by_lane_bypass, overpass, underpass, or hybrid_over_or_side'
                    ),
                    'confirm deterministic fallback when it is already visually consistent and safe',
                ],
                'preferred_safe_revision_example': {
                    'intent': (
                        'If the front RGB view is clear enough and this does not violate validation, revise only local '
                        'obstacle timing or policy fields so the experiment can measure VLM contribution without '
                        'breaking bottom-layer continuity.'
                    ),
                    'reason': (
                        'The complex scene contains asymmetric side clutter and mixed over/side bypass obstacles; '
                        'VLM should use the front RGB to refine obstacle morphology interpretation while preserving '
                        'the controller-managed sequence.'
                    ),
                    'x_window_candidate': (
                        'For bulky solid obstacles or tall clutter, observe_x may be moved 0.3 to 0.6 m earlier '
                        'and clear_x may be extended 0.3 to 0.6 m if all geometric constraints remain valid.'
                    ),
                    'route_policy_candidate': (
                        'For solid obstacles, choose underpass only when bottom clearance is explicit and underpass_feasible is true. '
                        'Choose overpass only when overpass_reasonable is true. When both vertical routes are feasible, '
                        'follow preferred_vertical_route or the smaller vertical_delta_z; otherwise use side_bypass_right/left.'
                    ),
                    'underpass_test_obstacle': (
                        'side_box_d is intentionally modeled as a floating solid obstacle with open bottom clearance; '
                        'if SFSC underpass_feasible is true and the RGB frame is not contradictory, prefer target_policy underpass '
                        'over side_bypass_right/left so the demo can test VLM route-family selection.'
                    ),
                },
                'forbidden_modifications': [
                    'do not change global passing_order during obstacle-zone execution',
                    'do not change any solid obstacle out of bypass mode',
                    'do not invent obstacle ids or low-level velocity commands',
                    'do not output unsafe slot intervals or nonnumeric UAV ids',
                ],
            },
        }
