from tests.dynamic_gate_common import DynamicGateSnakeBase


class DynamicMixedBypassObstacleZoneConfig(DynamicGateSnakeBase):
    """Mixed zone with dynamic solid obstacles placed in the passage corridor."""

    SCENE = 'assets/skydio_x2/scene_dynamic_mixed_bypass_zone5.xml'

    FINAL_TARGET_X = 48.0
    MAX_TOTAL_TIME = 250.0

    GATE_OBSERVE_DISTANCE_X = 5.4
    GATE_PASS_CLEAR_X = 1.25
    POST_REFORM_CRUISE_X = 2.2

    GATE_MOTION_SPECS = [
        {
            'prefix': 'gate1',
            'amplitude': 3.40,
            'omega': 0.38,
            'phase': 0.10,
            'blend': 0.24,
            'omega2': 0.42,
            'phase2': 0.70,
        },
        {
            'prefix': 'gate2',
            'amplitude': 2.85,
            'omega': 0.40,
            'phase': 1.15,
            'blend': 0.20,
            'omega2': 0.36,
            'phase2': 2.10,
        },
        {
            'prefix': 'gate3',
            'amplitude': 3.25,
            'omega': 0.34,
            'phase': 2.35,
            'blend': 0.26,
            'omega2': 0.40,
            'phase2': 3.05,
        },
        {
            'prefix': 'gate4',
            'amplitude': 2.95,
            'omega': 0.39,
            'phase': 3.20,
            'blend': 0.22,
            'omega2': 0.37,
            'phase2': 1.40,
        },
        {
            'prefix': 'gate5',
            'amplitude': 3.55,
            'omega': 0.36,
            'phase': 4.10,
            'blend': 0.24,
            'omega2': 0.41,
            'phase2': 2.75,
        },
    ]

    SOLID_BYPASS_SPECS = [
        {
            'prefix': 'side_box_b',
            'x': 22.2,
            'base_center_y': 0.0,
            'z': 1.35,
            'size_x': 2.40,
            'size_y': 4.80,
            'size_z': 3.10,
            'amplitude': 1.05,
            'omega': 0.36,
            'phase': 0.40,
            'route_policy': 'split_by_lane',
            'speed_x': 0.92,
            'align_speed_x': 0.42,
            'side_clearance': 0.95,
            'clear_x': 3.40,
        },
        {
            'prefix': 'side_box_c',
            'x': 34.0,
            'base_center_y': 0.0,
            'z': 0.90,
            'size_x': 2.10,
            'size_y': 51.00,
            'size_z': 14.40,
            'amplitude': 0.78,
            'omega': 0.34,
            'phase': 1.70,
            'route_policy': 'over',
            'speed_x': 0.88,
            'align_speed_x': 0.44,
            'top_clearance': 0.50,
            'activate_x': 4.40,
            'clear_x': 2.75,
        },
    ]

    SOLID_BYPASS_ACTIVATE_X = 5.80
    SOLID_BYPASS_CLEAR_X = 3.40
    SOLID_BYPASS_SIDE_CLEARANCE = 0.95
    SOLID_BYPASS_TOP_CLEARANCE = 0.55
    SOLID_BYPASS_SPEED_X = 0.92
    SOLID_BYPASS_Y_KP = 1.05

    SNAKE_SPACING_X = 0.84
    SNAKE_MAX_EXTRA_GAP_X = 1.35
    SNAKE_MAX_SPEED_X = 1.55
    REACTIVE_Y_KP = 1.95
    MAX_CMD_SPEED_XY = 3.05
    CMD_SLEW_RATE_XY = 4.10
    MAX_CMD_SPEED_Z = 1.25
    CMD_SLEW_RATE_Z = 2.10
    HEIGHT_HOLD_KP = 2.35
    HEIGHT_HOLD_DAMPING = 1.75
