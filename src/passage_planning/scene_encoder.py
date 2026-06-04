from __future__ import annotations

from collections.abc import Callable, Sequence

from .structures import FormationState, ObstacleDescriptor, ObstacleField


class ObstacleFieldEncoder:
    """Encodes obstacle observations into SFSC descriptors.

    In hardware terms this layer corresponds to fast non-VLM processing over
    range sensing, odometry and local obstacle estimates.  The MuJoCo demo uses
    scenario/runtime data to emulate that sensor-fused structured context.
    """

    @staticmethod
    def encode_dynamic_apertures(
        gate_specs: Sequence[dict],
        aperture_width_fn: Callable[[dict], float],
        formation: FormationState,
    ) -> ObstacleField:
        obstacles: list[ObstacleDescriptor] = []
        for index, gate_spec in enumerate(gate_specs):
            aperture_width = float(aperture_width_fn(gate_spec))
            motion_amplitude = float(gate_spec.get('amplitude', 0.0))
            risk_level = ObstacleFieldEncoder._aperture_risk_level(
                aperture_width=aperture_width,
                single_file_width=formation.single_file_width,
                motion_amplitude=motion_amplitude,
            )
            obstacles.append(
                ObstacleDescriptor(
                    obstacle_id=str(gate_spec.get('prefix', f'gate{index + 1}')),
                    obstacle_type='dynamic_gate',
                    function='aperture',
                    x=float(gate_spec['x']),
                    center_y=float(gate_spec['base_center_y']),
                    pass_z=float(gate_spec['pass_z']),
                    aperture_width=aperture_width,
                    motion_axis='y',
                    motion_model='sinusoidal',
                    motion_amplitude=motion_amplitude,
                    motion_omega=float(gate_spec.get('omega', 0.0)),
                    motion_phase=float(gate_spec.get('phase', 0.0)),
                    risk_level=risk_level,
                )
            )
        return ObstacleField(tuple(obstacles))

    @staticmethod
    def _aperture_risk_level(aperture_width: float, single_file_width: float, motion_amplitude: float) -> str:
        width_ratio = aperture_width / max(single_file_width, 1e-6)
        if width_ratio < 1.8 or motion_amplitude >= 3.5:
            return 'high'
        if width_ratio < 2.6 or motion_amplitude >= 2.0:
            return 'medium'
        return 'low'

    @staticmethod
    def encode_solid_obstacles(solid_specs: Sequence[dict]) -> ObstacleField:
        obstacles: list[ObstacleDescriptor] = []
        for index, solid_spec in enumerate(solid_specs):
            obstacles.append(
                ObstacleDescriptor(
                    obstacle_id=str(solid_spec.get('prefix', f'solid{index + 1}')),
                    obstacle_type=str(solid_spec.get('type', 'solid_obstacle')),
                    function='solid',
                    x=float(solid_spec['x']),
                    center_y=float(solid_spec.get('center_y', solid_spec.get('base_center_y', solid_spec.get('y', 0.0)))),
                    pass_z=float(solid_spec.get('pass_z', solid_spec.get('z', 0.0))),
                    aperture_width=None,
                    motion_axis=str(solid_spec.get('motion_axis', 'y' if float(solid_spec.get('amplitude', 0.0)) else 'none')),
                    motion_model=str(solid_spec.get('motion_model', 'sinusoidal' if float(solid_spec.get('amplitude', 0.0)) else 'static')),
                    motion_amplitude=float(solid_spec.get('amplitude', 0.0)),
                    motion_omega=float(solid_spec.get('omega', 0.0)),
                    motion_phase=float(solid_spec.get('phase', 0.0)),
                    risk_level=str(solid_spec.get('risk_level', 'medium')),
                    size_x=ObstacleFieldEncoder._solid_size(solid_spec, 'x'),
                    size_y=ObstacleFieldEncoder._solid_size(solid_spec, 'y'),
                    size_z=ObstacleFieldEncoder._solid_size(solid_spec, 'z'),
                )
            )
        return ObstacleField(tuple(obstacles))

    @staticmethod
    def _solid_size(solid_spec: dict, axis: str) -> float | None:
        size = solid_spec.get(f'size_{axis}')
        if size is not None:
            return float(size)
        sizes = solid_spec.get('size')
        axis_index = {'x': 0, 'y': 1, 'z': 2}[axis]
        if isinstance(sizes, (list, tuple)) and len(sizes) > axis_index:
            return float(sizes[axis_index])
        return None
