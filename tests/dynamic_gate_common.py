import unittest

import numpy as np
import mujoco

from src.motion_planning.trajectory_planning import TrajectoryParameter, TrajectoryPlanner
from src.motion_planning.trajectory_planning.path_planning.joint_planning import JointParameter
from src.motion_planning.trajectory_planning.velocity_planning.quintic_velocity_planning import (
    QuinticVelocityParameter,
)


class DynamicGateSnakeBase(unittest.TestCase):
    """Common utilities for dynamic-aperture multi-UAV passage tests.

    This class intentionally contains no `test_*` method.  It keeps the
    stable three-gate demo independent from older exploratory demos and gives
    us a clean place to extract strategy-driven planning in the next step.
    """

    SCENE = 'assets/skydio_x2/scene_dynamic_moving_gates5.xml'

    COUNT = 5
    CRUISE_SPEED_X = 0.95

    INIT_ROTATE_TIME = 5.0
    INITIAL_Z = 0.30
    UAV_RADIUS_XY = 0.38
    UAV_RADIUS_Z = 0.22

    GATE_TOP_CLEARANCE_Z = 0.28
    GATE_PASS_DOWN_BIAS_Z = 0.10
    GATE_CENTER_LOOKAHEAD_TIME = 0.42
    ALIGN_ENTRY_BUFFER_X = 0.38
    GATE_PASS_MAX_Z_ABOVE_INITIAL = 0.72
    APERTURE_TOP_EXTRA_MARGIN_Z = 0.10
    APERTURE_BOTTOM_EXTRA_MARGIN_Z = 0.06

    REFORM_POSITION_TOL_X = 0.65
    REFORM_POSITION_TOL_Y = 0.55
    REFORM_POSITION_TOL_Z = 0.30

    REACTIVE_SPEED_X = 0.82
    REACTIVE_NEAR_GATE_SPEED_X = 0.56
    REACTIVE_BRAKE_KP_X = 0.18
    REACTIVE_MIN_SPEED_X = 0.08
    TEMPORAL_SLOT_WAIT_SPEED_X = 0.50
    REACTIVE_COMMIT_SPEED_X = 1.35
    REACTIVE_EXIT_SPEED_X = 1.45
    GATE_COMMIT_DISTANCE_X = 1.10
    GATE_ALIGNMENT_Y_TOL = 0.60
    GATE_ALIGNMENT_Z_TOL = 0.30

    SNAKE_MIN_GAP_X = 0.48
    SNAKE_SPACING_KP = 0.85
    SNAKE_MIN_SPEED_X = 0.04
    SNAKE_FOLLOWER_Y_ALIGN_GAP_X = 0.62
    SNAKE_EXTRA_GAP_Y_GAIN = 0.75
    SNAKE_EXTRA_GAP_Z_GAIN = 0.55

    POST_GATE_HOLD_X = 2.25
    POST_GATE_HOLD_STAGGER_X = 0.42
    POST_GATE_HOLD_KP_XY = 0.85
    POST_GATE_HOLD_KP_Z = 0.95
    POST_GATE_HOLD_MAX_SPEED_XY = 1.15
    POST_GATE_HOLD_CLEAR_X = 0.92

    BLEND_DURATION = 0.9
    OFFSET_POS_KP = 1.35
    OFFSET_MAX_SPEED = 2.25
    DPOSD_SLEW_RATE = 2.6
    FINAL_REFORM_STABLE_TIME = 0.6

    COLLISION_SAFE_DISTANCE = 0.78
    COLLISION_HARD_DISTANCE = 0.38
    COLLISION_ACTIVATION_DISTANCE = 1.12
    COLLISION_PROJECTION_ITERS = 4

    SOLID_BYPASS_SPECS = []
    SOLID_BYPASS_ACTIVATE_X = 2.25
    SOLID_BYPASS_CLEAR_X = 1.15
    SOLID_BYPASS_SIDE_CLEARANCE = 1.05
    SOLID_BYPASS_TOP_CLEARANCE = 0.55
    SOLID_BYPASS_SPEED_X = 0.72
    SOLID_BYPASS_Y_KP = 1.35
    SOLID_BYPASS_Z_BAND = 0.45
    OBSTACLE_ALIGN_HOLD_DISTANCE_X = 1.65
    BYPASS_ALIGN_HOLD_DISTANCE_X = 2.40
    BYPASS_ALIGNMENT_Y_TOL = 0.72
    BYPASS_ALIGNMENT_Z_TOL = 0.42
    GATE_STRAIGHT_THROUGH_MARGIN_X = 0.10

    @staticmethod
    def _build_segment_planner(start_pose: np.ndarray, end_pose: np.ndarray, duration: float) -> TrajectoryPlanner:
        joint_param = JointParameter(np.asarray(start_pose, dtype=float), np.asarray(end_pose, dtype=float))
        velocity_param = QuinticVelocityParameter(float(duration))
        trajectory_param = TrajectoryParameter(joint_param, velocity_param)
        return TrajectoryPlanner(trajectory_param)

    @staticmethod
    def _smoothstep(alpha: float) -> float:
        alpha = float(np.clip(alpha, 0.0, 1.0))
        return float(alpha * alpha * (3.0 - 2.0 * alpha))

    @staticmethod
    def _passing_order_from_y(original_y_list: list[float]) -> list[int]:
        """Centerline UAV passes first; outer UAVs follow by center distance."""
        return sorted(range(len(original_y_list)), key=lambda idx: (abs(original_y_list[idx]), idx))

    @staticmethod
    def _gate_geom_ids(model: mujoco.MjModel, prefix: str) -> dict:
        ids = {
            'left': mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f'{prefix}_left_post'),
            'right': mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f'{prefix}_right_post'),
            'top': mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f'{prefix}_top_beam'),
        }
        bottom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f'{prefix}_bottom_beam')
        if bottom_id >= 0:
            ids['bottom'] = bottom_id
        return ids

    def _safe_aperture_pass_z(self, top_inner: float, bottom_inner: float, down_bias: float) -> float:
        bottom_safe = float(bottom_inner + self.UAV_RADIUS_Z + self.APERTURE_BOTTOM_EXTRA_MARGIN_Z)
        top_safe = float(top_inner - self.UAV_RADIUS_Z - self.APERTURE_TOP_EXTRA_MARGIN_Z)
        nominal = float(0.5 * (top_inner + bottom_inner) - down_bias)
        return float(np.clip(nominal, bottom_safe, top_safe))

    def _gate_center_y(self, gate_spec: dict, current_t: float) -> float:
        primary = np.sin(gate_spec['omega'] * current_t + gate_spec['phase'])
        secondary = gate_spec['blend'] * np.sin(gate_spec['omega2'] * current_t + gate_spec['phase2'])
        return float(gate_spec['base_center_y'] + gate_spec['amplitude'] * (primary + secondary))

    def _gate_center_vy(self, gate_spec: dict, current_t: float) -> float:
        primary = gate_spec['omega'] * np.cos(gate_spec['omega'] * current_t + gate_spec['phase'])
        secondary = gate_spec['blend'] * gate_spec['omega2'] * np.cos(gate_spec['omega2'] * current_t + gate_spec['phase2'])
        return float(gate_spec['amplitude'] * (primary + secondary))

    def _build_gate_runtime_specs(self, model: mujoco.MjModel) -> list[dict]:
        gate_specs = []
        for gate_motion in self.GATE_MOTION_SPECS:
            ids = self._gate_geom_ids(model, gate_motion['prefix'])
            left_pos = np.asarray(model.geom_pos[ids['left']], dtype=float).copy()
            right_pos = np.asarray(model.geom_pos[ids['right']], dtype=float).copy()
            top_pos = np.asarray(model.geom_pos[ids['top']], dtype=float).copy()
            top_size = np.asarray(model.geom_size[ids['top']], dtype=float).copy()

            top_inner = float(top_pos[2] - top_size[2])
            bottom_inner = float(self.INITIAL_Z)
            if 'bottom' in ids:
                bottom_pos = np.asarray(model.geom_pos[ids['bottom']], dtype=float).copy()
                bottom_size = np.asarray(model.geom_size[ids['bottom']], dtype=float).copy()
                bottom_inner = float(bottom_pos[2] + bottom_size[2])
            pass_z = self._safe_aperture_pass_z(
                top_inner=top_inner - self.GATE_TOP_CLEARANCE_Z,
                bottom_inner=bottom_inner,
                down_bias=self.GATE_PASS_DOWN_BIAS_Z,
            )
            base_positions = {
                'left': left_pos,
                'right': right_pos,
                'top': top_pos,
            }
            if 'bottom' in ids:
                base_positions['bottom'] = np.asarray(model.geom_pos[ids['bottom']], dtype=float).copy()

            gate_specs.append({
                'prefix': str(gate_motion['prefix']),
                'ids': ids,
                'x': float(top_pos[0]),
                'base_center_y': float(top_pos[1]),
                'pass_z': float(pass_z),
                'aperture_bottom_inner': float(bottom_inner),
                'aperture_top_inner': float(top_inner),
                'has_bottom_beam': 'bottom' in ids,
                'half_thickness_x': float(abs(top_size[0])),
                'base_positions': base_positions,
                'amplitude': float(gate_motion['amplitude']),
                'omega': float(gate_motion['omega']),
                'phase': float(gate_motion['phase']),
                'blend': float(gate_motion['blend']),
                'omega2': float(gate_motion['omega2']),
                'phase2': float(gate_motion['phase2']),
                'center_lookahead_time': float(
                    gate_motion.get('center_lookahead_time', self.GATE_CENTER_LOOKAHEAD_TIME)
                ),
                'max_center_lead_y': (
                    None
                    if gate_motion.get('max_center_lead_y') is None
                    else float(gate_motion.get('max_center_lead_y'))
                ),
            })
        return gate_specs

    def _build_solid_bypass_runtime_specs(self, model: mujoco.MjModel) -> list[dict]:
        solid_specs = []
        for solid_spec in getattr(self, 'SOLID_BYPASS_SPECS', []):
            geom_id = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_GEOM,
                str(solid_spec['prefix']),
            )
            if geom_id < 0:
                raise ValueError(f"unknown solid bypass geom: {solid_spec['prefix']}")
            base_pos = np.asarray(model.geom_pos[geom_id], dtype=float).copy()
            geom_size = np.asarray(model.geom_size[geom_id], dtype=float).copy()
            obstacle_size = np.array([
                float(solid_spec.get('size_x', 2.0 * geom_size[0])),
                float(solid_spec.get('size_y', 2.0 * geom_size[1])),
                float(solid_spec.get('size_z', 2.0 * geom_size[2])),
            ], dtype=float)
            if len(geom_size) >= 3:
                model.geom_size[geom_id] = 0.5 * obstacle_size
            route_policy = str(solid_spec.get('route_policy', 'split_by_lane'))
            solid_specs.append({
                'prefix': str(solid_spec['prefix']),
                'id': int(geom_id),
                'x': float(solid_spec.get('x', base_pos[0])),
                'base_center_y': float(solid_spec.get('base_center_y', solid_spec.get('center_y', base_pos[1]))),
                'z': float(solid_spec.get('z', base_pos[2])),
                'size': obstacle_size,
                'amplitude': float(solid_spec.get('amplitude', 0.0)),
                'omega': float(solid_spec.get('omega', 0.0)),
                'phase': float(solid_spec.get('phase', 0.0)),
                'route_policy': route_policy,
                'default_route_policy': route_policy,
                'route_policy_source': 'deterministic_config',
                'speed_x': float(solid_spec.get('speed_x', self.SOLID_BYPASS_SPEED_X)),
                'align_speed_x': float(solid_spec.get('align_speed_x', self.TEMPORAL_SLOT_WAIT_SPEED_X)),
                'side_clearance': float(solid_spec.get('side_clearance', self.SOLID_BYPASS_SIDE_CLEARANCE)),
                'top_clearance': float(solid_spec.get('top_clearance', self.SOLID_BYPASS_TOP_CLEARANCE)),
                'activate_x': float(solid_spec.get('activate_x', self.SOLID_BYPASS_ACTIVATE_X)),
                'clear_x': float(solid_spec.get('clear_x', self.SOLID_BYPASS_CLEAR_X)),
                'release_policy': str(solid_spec.get('release_policy', 'per_drone_after_upstream_gate')),
            })
        return solid_specs

    def _solid_center_y(self, solid_spec: dict, current_t: float) -> float:
        return float(
            solid_spec['base_center_y']
            + solid_spec['amplitude'] * np.sin(solid_spec['omega'] * current_t + solid_spec['phase'])
        )

    def _update_dynamic_solids(self, model: mujoco.MjModel, solid_specs: list[dict], current_t: float):
        for solid_spec in solid_specs:
            model.geom_pos[solid_spec['id']] = np.array([
                float(solid_spec['x']),
                self._solid_center_y(solid_spec, current_t),
                float(solid_spec['z']),
            ], dtype=float)

    def _active_solid_bypass(self, current_pos: np.ndarray, solid_specs: list[dict], current_t: float) -> dict | None:
        current_pos = np.asarray(current_pos, dtype=float)
        for solid_spec in solid_specs:
            x = float(solid_spec['x'])
            if current_pos[0] < x - float(solid_spec.get('activate_x', self.SOLID_BYPASS_ACTIVATE_X)):
                continue
            half_x = 0.5 * float(abs(solid_spec['size'][0]))
            if current_pos[0] > x + half_x + float(solid_spec.get('clear_x', self.SOLID_BYPASS_CLEAR_X)):
                continue
            center_y = self._solid_center_y(solid_spec, current_t)
            half_y = 0.5 * float(solid_spec['size'][1])
            influence_y = half_y + self.SOLID_BYPASS_SIDE_CLEARANCE + self.UAV_RADIUS_XY
            if abs(float(current_pos[1]) - center_y) <= influence_y:
                return solid_spec
        return None

    def _active_solid_bypass_by_x(self, current_pos: np.ndarray, solid_specs: list[dict]) -> dict | None:
        current_x = float(np.asarray(current_pos, dtype=float)[0])
        for solid_spec in solid_specs:
            enter_x = float(solid_spec['x']) - float(solid_spec.get('activate_x', self.SOLID_BYPASS_ACTIVATE_X))
            exit_x = self._solid_bypass_exit_x(solid_spec)
            if enter_x <= current_x <= exit_x:
                return solid_spec
        return None

    @staticmethod
    def _solid_bypass_exit_x(solid_spec: dict) -> float:
        half_x = 0.5 * float(abs(solid_spec['size'][0]))
        return float(solid_spec['x']) + half_x + float(solid_spec['clear_x'])

    def _solid_bypass_is_cleared_by_drone(self, current_pos: np.ndarray, solid_spec: dict) -> bool:
        current_x = float(np.asarray(current_pos, dtype=float)[0])
        return current_x >= self._solid_bypass_exit_x(solid_spec)

    @staticmethod
    def _solid_spec_by_prefix(solid_specs: list[dict], prefix: str) -> dict | None:
        for solid_spec in solid_specs:
            if str(solid_spec['prefix']) == str(prefix):
                return solid_spec
        return None

    def _solid_upstream_gate_index(self, solid_spec: dict, gate_specs: list[dict]) -> int | None:
        solid_x = float(solid_spec['x'])
        upstream_indices = [
            gate_index
            for gate_index, gate_spec in enumerate(gate_specs)
            if float(gate_spec['x']) < solid_x
        ]
        if not upstream_indices:
            return None
        return int(max(upstream_indices))

    def _gate_straight_through_zone(self, current_pos: np.ndarray, gate_spec: dict) -> bool:
        current_x = float(np.asarray(current_pos, dtype=float)[0])
        half_thickness = float(gate_spec.get('half_thickness_x', 0.0))
        enter_x = float(gate_spec['x']) - half_thickness + float(self.GATE_STRAIGHT_THROUGH_MARGIN_X)
        exit_x = float(gate_spec['x']) + half_thickness + float(self.GATE_PASS_CLEAR_X)
        return enter_x <= current_x <= exit_x

    def _solid_bypass_is_released_after_upstream_gate(
        self,
        current_pos: np.ndarray,
        solid_spec: dict,
        gate_specs: list[dict],
    ) -> bool:
        """Allow solid-bypass response only after this UAV has cleared the upstream aperture."""
        current_x = float(np.asarray(current_pos, dtype=float)[0])
        solid_x = float(solid_spec['x'])
        upstream_gates = [
            gate_spec
            for gate_spec in gate_specs
            if float(gate_spec['x']) < solid_x
        ]
        if not upstream_gates:
            return True
        upstream_gate = max(upstream_gates, key=lambda gate_spec: float(gate_spec['x']))
        release_x = float(upstream_gate['x'] + self.GATE_PASS_CLEAR_X)
        return current_x >= release_x

    def _solid_bypass_is_released(
        self,
        current_pos: np.ndarray,
        positions: list[np.ndarray],
        solid_spec: dict,
        gate_specs: list[dict],
    ) -> bool:
        solid_x = float(solid_spec['x'])
        upstream_gates = [
            gate_spec
            for gate_spec in gate_specs
            if float(gate_spec['x']) < solid_x
        ]
        if not upstream_gates:
            return True
        upstream_gate = max(upstream_gates, key=lambda gate_spec: float(gate_spec['x']))
        release_policy = str(solid_spec.get('release_policy', 'per_drone_after_upstream_gate'))
        if release_policy == 'all_after_upstream_gate':
            return self._all_cleared_gate(positions, upstream_gate)
        return self._solid_bypass_is_released_after_upstream_gate(
            current_pos,
            solid_spec,
            gate_specs,
        )

    def _should_hold_for_obstacle_alignment(
        self,
        current_pos: np.ndarray,
        obstacle_x: float,
        target_y: float,
        target_z: float,
        y_tol: float,
        z_tol: float,
        hold_distance_x: float,
    ) -> bool:
        current_pos = np.asarray(current_pos, dtype=float)
        x_gap = float(obstacle_x - current_pos[0])
        if x_gap < 0.0 or x_gap > float(hold_distance_x):
            return False
        z_error = abs(float(target_z) - float(current_pos[2]))
        return z_error > float(z_tol)

    def _phase_aware_predecessor(
        self,
        drone_id: int,
        passing_order: list[int],
        phase_keys: dict[int, tuple],
    ) -> int | None:
        own_phase = phase_keys.get(int(drone_id))
        previous_same_phase = None
        for candidate_id in passing_order:
            candidate_id = int(candidate_id)
            if candidate_id == int(drone_id):
                return previous_same_phase
            if phase_keys.get(candidate_id) == own_phase:
                previous_same_phase = candidate_id
        return None

    def _solid_barriers_before_gate_are_clear(
        self,
        positions: list[np.ndarray],
        solid_specs: list[dict],
        previous_gate_x: float,
        gate_x: float,
    ) -> bool:
        for solid_spec in solid_specs:
            solid_x = float(solid_spec['x'])
            if not (float(previous_gate_x) < solid_x < float(gate_x)):
                continue
            clear_x = solid_x + float(solid_spec.get('clear_x', self.SOLID_BYPASS_CLEAR_X))
            if not all(float(np.asarray(pos, dtype=float)[0]) >= clear_x for pos in positions):
                return False
        return True

    def _update_dynamic_gates(self, model: mujoco.MjModel, gate_specs: list[dict], current_t: float):
        for gate_spec in gate_specs:
            center_y = self._gate_center_y(gate_spec, current_t)
            y_shift = float(center_y - gate_spec['base_center_y'])
            for key, geom_id in gate_spec['ids'].items():
                base_pos = np.asarray(gate_spec['base_positions'][key], dtype=float)
                model.geom_pos[geom_id] = np.array([base_pos[0], base_pos[1] + y_shift, base_pos[2]], dtype=float)

    def _clip_velocity_limits(self, velocity_xyz: np.ndarray) -> np.ndarray:
        velocity_xyz = np.asarray(velocity_xyz, dtype=float).copy()
        speed_xy = float(np.linalg.norm(velocity_xyz[:2]))
        if speed_xy > self.MAX_CMD_SPEED_XY and speed_xy > 1e-9:
            velocity_xyz[:2] = velocity_xyz[:2] / speed_xy * self.MAX_CMD_SPEED_XY
        velocity_xyz[2] = float(np.clip(velocity_xyz[2], -self.MAX_CMD_SPEED_Z, self.MAX_CMD_SPEED_Z))
        return velocity_xyz

    def _slew_limit_velocity(self, prev_vel: np.ndarray, target_vel: np.ndarray, dt: float) -> np.ndarray:
        prev_vel = np.asarray(prev_vel, dtype=float)
        target_vel = np.asarray(target_vel, dtype=float)
        dt = float(max(dt, 1e-6))
        delta = target_vel - prev_vel
        delta_xy_max = float(self.CMD_SLEW_RATE_XY * dt)
        delta_z_max = float(self.CMD_SLEW_RATE_Z * dt)
        delta[0] = float(np.clip(delta[0], -delta_xy_max, delta_xy_max))
        delta[1] = float(np.clip(delta[1], -delta_xy_max, delta_xy_max))
        delta[2] = float(np.clip(delta[2], -delta_z_max, delta_z_max))
        return prev_vel + delta

    def _gate_aperture_width(self, model: mujoco.MjModel, gate_spec: dict) -> float:
        ids = gate_spec['ids']
        left_pos = np.asarray(model.geom_pos[ids['left']], dtype=float)
        right_pos = np.asarray(model.geom_pos[ids['right']], dtype=float)
        left_size = np.asarray(model.geom_size[ids['left']], dtype=float)
        right_size = np.asarray(model.geom_size[ids['right']], dtype=float)
        left_inner = float(left_pos[1] + left_size[1])
        right_inner = float(right_pos[1] - right_size[1])
        return float(max(0.0, right_inner - left_inner))

    def _infer_dynamic_gate_strategy(
        self,
        model: mujoco.MjModel,
        gate_spec: dict,
        original_y_list: list[float],
    ) -> str:
        aperture_width = self._gate_aperture_width(model, gate_spec)
        formation_width = float(max(original_y_list) - min(original_y_list) + 2.0 * self.UAV_RADIUS_XY)
        single_width = float(2.0 * self.UAV_RADIUS_XY)
        if aperture_width < min(formation_width, 4.0 * single_width):
            return 'snake'
        return 'formation'

    def _height_track_velocity(self, current_z: float, current_vz: float, target_z: float) -> float:
        vz_cmd = float(
            self.HEIGHT_HOLD_KP * (float(target_z) - float(current_z))
            - self.HEIGHT_HOLD_DAMPING * float(current_vz)
        )
        return float(np.clip(vz_cmd, -self.MAX_CMD_SPEED_Z, self.MAX_CMD_SPEED_Z))

    def _snake_speed_x(
        self,
        drone_id: int,
        order_index: int,
        positions: list[np.ndarray],
        gate_spec: dict,
        target_y: float,
        target_z: float,
        passing_order: list[int],
        predecessor_id: int | None = None,
    ) -> float:
        current_pos = np.asarray(positions[drone_id], dtype=float)
        near_gate = abs(float(current_pos[0]) - float(gate_spec['x'])) <= 0.95
        base_speed_x = self.REACTIVE_NEAR_GATE_SPEED_X if near_gate else self.REACTIVE_SPEED_X

        y_error = abs(float(target_y) - float(current_pos[1]))
        z_error = abs(float(target_z) - float(current_pos[2]))
        speed_x = float(max(self.REACTIVE_MIN_SPEED_X, base_speed_x - self.REACTIVE_BRAKE_KP_X * y_error))

        close_enough_to_commit = float(current_pos[0]) >= float(gate_spec['x'] - self.GATE_COMMIT_DISTANCE_X)
        aligned_for_gate = y_error <= float(self.GATE_ALIGNMENT_Y_TOL) and z_error <= float(self.GATE_ALIGNMENT_Z_TOL)
        if close_enough_to_commit and aligned_for_gate:
            speed_x = max(speed_x, float(self.REACTIVE_COMMIT_SPEED_X))
        if float(current_pos[0]) >= float(gate_spec['x'] - 0.10):
            speed_x = max(speed_x, float(self.REACTIVE_EXIT_SPEED_X))

        if order_index > 0 and predecessor_id is not None:
            prev_id = int(predecessor_id)
            prev_x = float(np.asarray(positions[prev_id], dtype=float)[0])
            gap_x = float(prev_x - current_pos[0])
            extra_gap_x = min(
                float(self.SNAKE_MAX_EXTRA_GAP_X),
                float(self.SNAKE_EXTRA_GAP_Y_GAIN * y_error + self.SNAKE_EXTRA_GAP_Z_GAIN * z_error),
            )
            desired_gap_x = float(self.SNAKE_SPACING_X + extra_gap_x)
            spacing_error = gap_x - desired_gap_x

            y_align_alpha = 1.0 - np.clip(y_error / max(1e-6, self.GATE_ALIGNMENT_Y_TOL), 0.0, 1.0)
            z_align_alpha = 1.0 - np.clip(z_error / max(1e-6, self.GATE_ALIGNMENT_Z_TOL), 0.0, 1.0)
            align_alpha = self._smoothstep(min(float(y_align_alpha), float(z_align_alpha)))

            if spacing_error > 0.0:
                speed_x = float(speed_x + self.SNAKE_SPACING_KP * spacing_error * align_alpha)
            else:
                speed_x = float(speed_x + self.SNAKE_SPACING_KP * spacing_error)
            if gap_x < self.SNAKE_MIN_GAP_X:
                speed_x = min(speed_x, self.SNAKE_MIN_SPEED_X)

        return float(np.clip(speed_x, self.SNAKE_MIN_SPEED_X, self.SNAKE_MAX_SPEED_X))

    def _snake_velocity(
        self,
        current_pos: np.ndarray,
        current_vel: np.ndarray,
        speed_x: float,
        target_y: float,
        target_z: float,
    ) -> np.ndarray:
        cmd = np.array([
            max(0.0, float(speed_x)),
            self.REACTIVE_Y_KP * (float(target_y) - float(current_pos[1])),
            self._height_track_velocity(current_pos[2], current_vel[2], float(target_z)),
        ], dtype=float)
        return self._clip_velocity_limits(cmd)

    def _post_gate_hold_target(
        self,
        drone_id: int,
        order_index: int,
        gate_spec: dict,
        original_y_list: list[float],
    ) -> np.ndarray:
        x = float(gate_spec['x'] + self.POST_GATE_HOLD_X + order_index * self.POST_GATE_HOLD_STAGGER_X)
        return np.array([x, float(original_y_list[drone_id]), self.INITIAL_Z], dtype=float)

    def _post_gate_hold_velocity(
        self,
        current_pos: np.ndarray,
        current_vel: np.ndarray,
        hold_target: np.ndarray,
    ) -> np.ndarray:
        current_pos = np.asarray(current_pos, dtype=float)
        current_vel = np.asarray(current_vel, dtype=float)
        hold_target = np.asarray(hold_target, dtype=float)
        err = hold_target - current_pos
        cmd = np.array([
            self.POST_GATE_HOLD_KP_XY * err[0],
            self.POST_GATE_HOLD_KP_XY * err[1],
            self._height_track_velocity(current_pos[2], current_vel[2], hold_target[2]),
        ], dtype=float)
        speed_xy = float(np.linalg.norm(cmd[:2]))
        if speed_xy > self.POST_GATE_HOLD_MAX_SPEED_XY and speed_xy > 1e-9:
            cmd[:2] = cmd[:2] / speed_xy * self.POST_GATE_HOLD_MAX_SPEED_XY
        cmd[2] = float(np.clip(cmd[2], -self.MAX_CMD_SPEED_Z, self.MAX_CMD_SPEED_Z))
        return cmd

    def _all_cleared_gate(self, positions: list[np.ndarray], gate_spec: dict) -> bool:
        pass_clear_x = float(gate_spec.get('pass_clear_x', self.GATE_PASS_CLEAR_X))
        clear_x = float(gate_spec['x'] + pass_clear_x)
        return all(float(np.asarray(pos, dtype=float)[0]) >= clear_x for pos in positions)

    def _formation_is_stable(self, positions: list[np.ndarray], leader_pos: np.ndarray, formation) -> bool:
        pos_ref_all, _ = formation.reference(
            leader_pos=np.asarray(leader_pos, dtype=float),
            leader_vel=np.array([self.CRUISE_SPEED_X, 0.0, 0.0], dtype=float),
            leader_yaw=np.pi,
            leader_yaw_rate=0.0,
            leader_id=0,
        )
        for drone_id, pos in enumerate(positions):
            err = np.asarray(pos, dtype=float) - np.asarray(pos_ref_all[drone_id], dtype=float)
            if abs(float(err[0])) > self.REFORM_POSITION_TOL_X:
                return False
            if abs(float(err[1])) > self.REFORM_POSITION_TOL_Y:
                return False
            if abs(float(err[2])) > self.REFORM_POSITION_TOL_Z:
                return False
        return True
