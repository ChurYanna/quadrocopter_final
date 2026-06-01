from __future__ import annotations

from .structures import FormationState, MissionPreference, ObstacleDescriptor, ObstacleField, PassabilityReport


class PassabilityEvaluator:
    """Deterministic passability reasoning for formation-level passage."""

    def __init__(self, formation_pass_margin: float = 0.20, single_file_margin: float = 0.16):
        self.formation_pass_margin = float(formation_pass_margin)
        self.single_file_margin = float(single_file_margin)

    def evaluate_field(
        self,
        field: ObstacleField,
        formation: FormationState,
        mission: MissionPreference,
    ) -> tuple[PassabilityReport, ...]:
        return tuple(self.evaluate_obstacle(obstacle, formation, mission) for obstacle in field.obstacles)

    def evaluate_obstacle(
        self,
        obstacle: ObstacleDescriptor,
        formation: FormationState,
        mission: MissionPreference,
    ) -> PassabilityReport:
        if obstacle.function == 'solid':
            return PassabilityReport(
                obstacle_id=obstacle.obstacle_id,
                recommended_mode='bypass',
                confidence=0.80,
                reasons=('solid obstacle requires edge or over-pass strategy',),
            )

        aperture_width = float(obstacle.aperture_width or 0.0)
        formation_required_width = formation.width_y + self.formation_pass_margin
        single_file_required_width = formation.single_file_width + self.single_file_margin

        if mission.preferred_mode is not None:
            return PassabilityReport(
                obstacle_id=obstacle.obstacle_id,
                recommended_mode=mission.preferred_mode,
                confidence=0.75,
                reasons=(f'mission preferred mode is {mission.preferred_mode}',),
            )

        if aperture_width >= formation_required_width:
            return PassabilityReport(
                obstacle_id=obstacle.obstacle_id,
                recommended_mode='formation',
                confidence=0.88,
                reasons=('aperture width is sufficient for current formation',),
            )

        if mission.allow_disband and aperture_width >= single_file_required_width:
            confidence = 0.90 if obstacle.risk_level != 'high' else 0.82
            return PassabilityReport(
                obstacle_id=obstacle.obstacle_id,
                recommended_mode='snake_sequence',
                confidence=confidence,
                reasons=(
                    'formation width exceeds aperture width',
                    'single-file passage remains feasible',
                ),
            )

        return PassabilityReport(
            obstacle_id=obstacle.obstacle_id,
            recommended_mode='bypass',
            confidence=0.72,
            reasons=(
                'aperture is too narrow for both formation and single-file passage',
                'fallback to bypass or waiting strategy',
            ),
        )
