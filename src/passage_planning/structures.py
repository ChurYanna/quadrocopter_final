from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


ObstacleFunction = Literal['aperture', 'solid']
PassageMode = Literal['formation', 'snake_sequence', 'bypass']


@dataclass(frozen=True)
class ObstacleDescriptor:
    """Structured semantic description of one obstacle in the passage field."""

    obstacle_id: str
    obstacle_type: str
    function: ObstacleFunction
    x: float
    center_y: float
    pass_z: float
    aperture_width: float | None = None
    motion_axis: str = 'y'
    motion_model: str = 'sinusoidal'
    motion_amplitude: float = 0.0
    motion_omega: float = 0.0
    motion_phase: float = 0.0
    risk_level: str = 'unknown'
    size_x: float | None = None
    size_y: float | None = None
    size_z: float | None = None


@dataclass(frozen=True)
class ObstacleField:
    """Ordered obstacle-zone description used by the deterministic planner."""

    obstacles: tuple[ObstacleDescriptor, ...]

    def __post_init__(self):
        ordered = tuple(sorted(self.obstacles, key=lambda item: item.x))
        object.__setattr__(self, 'obstacles', ordered)

    @property
    def count(self) -> int:
        return len(self.obstacles)


@dataclass(frozen=True)
class FormationState:
    """Compact multi-UAV formation state for passability reasoning."""

    num_uavs: int
    original_y: tuple[float, ...]
    initial_z: float
    uav_radius_xy: float
    uav_radius_z: float
    nominal_speed_x: float
    max_speed_xy: float | None = None
    max_speed_z: float | None = None
    max_lateral_speed: float | None = None

    @property
    def width_y(self) -> float:
        if not self.original_y:
            return 0.0
        return float(max(self.original_y) - min(self.original_y) + 2.0 * self.uav_radius_xy)

    @property
    def single_file_width(self) -> float:
        return float(2.0 * self.uav_radius_xy)

    @property
    def lateral_tracking_speed(self) -> float:
        if self.max_lateral_speed is not None:
            return float(self.max_lateral_speed)
        if self.max_speed_xy is not None:
            return float(self.max_speed_xy)
        return float(self.nominal_speed_x)

    @property
    def vertical_tracking_speed(self) -> float:
        if self.max_speed_z is not None:
            return float(self.max_speed_z)
        return float(self.nominal_speed_x)


@dataclass(frozen=True)
class MissionPreference:
    """Task-level preference that can later come from rules or LLM output."""

    priority: str = 'fast_safe_passage'
    allow_disband: bool = True
    recover_after_last_obstacle: bool = True
    preferred_mode: PassageMode | None = None


@dataclass(frozen=True)
class PassabilityReport:
    """Passability decision for one obstacle or for the whole field."""

    obstacle_id: str
    recommended_mode: PassageMode
    confidence: float
    reasons: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class PassageSlot:
    """Spatio-temporal slot assigned to one UAV."""

    drone_id: int
    order_index: int
    nominal_entry_time: float
    nominal_exit_time: float


@dataclass(frozen=True)
class ObstaclePassageStrategy:
    """Executable high-level strategy for one obstacle."""

    obstacle_id: str
    obstacle_index: int
    mode: PassageMode
    target_policy: str
    obstacle_x: float
    clear_x: float
    observe_x: float


@dataclass(frozen=True)
class PassagePlan:
    """Complete deterministic strategy consumed by the MuJoCo demo."""

    plan_id: str
    mode: PassageMode
    passing_order: tuple[int, ...]
    slots: tuple[PassageSlot, ...]
    obstacle_strategies: tuple[ObstaclePassageStrategy, ...]
    recover_after_last_obstacle: bool
    time_slot_interval: float
    confidence: float
    source: str = 'deterministic'

    def order_index(self, drone_id: int) -> int:
        return self.passing_order.index(int(drone_id))


@dataclass(frozen=True)
class ValidationIssue:
    """One validator finding with machine-readable code and severity."""

    severity: Literal['error', 'warning', 'repair']
    code: str
    message: str


@dataclass(frozen=True)
class ValidationResult:
    """Strategy validation result before execution."""

    valid: bool
    repaired: bool = False
    messages: tuple[str, ...] = field(default_factory=tuple)
    issues: tuple[ValidationIssue, ...] = field(default_factory=tuple)

    @property
    def errors(self) -> tuple[ValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == 'error')

    @property
    def warnings(self) -> tuple[ValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == 'warning')
