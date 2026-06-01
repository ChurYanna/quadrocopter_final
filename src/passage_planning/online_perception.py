from __future__ import annotations

from dataclasses import dataclass, replace

from .structures import ObstacleDescriptor, ObstacleField


@dataclass(frozen=True)
class OnlinePerceptionConfig:
    """Finite-view perception settings for online semantic replanning."""

    lookahead_distance: float = 14.0
    behind_tolerance: float = 1.0
    detail_reveal_distance: float = 10.0
    max_visible_obstacles: int = 3


@dataclass(frozen=True)
class PerceivedObstacle:
    """One obstacle currently inside the finite perception window."""

    obstacle: ObstacleDescriptor
    relative_distance_x: float
    newly_observed: bool
    details_revealed: bool
    requires_planning: bool

    @property
    def obstacle_id(self) -> str:
        return self.obstacle.obstacle_id

    @property
    def function(self) -> str:
        return self.obstacle.function


@dataclass(frozen=True)
class PerceptionWindow:
    """Snapshot of what the online planner is allowed to know now."""

    leader_x: float
    x_min: float
    x_max: float
    visible_obstacles: tuple[PerceivedObstacle, ...]
    newly_observed_ids: tuple[str, ...]
    planning_candidate_ids: tuple[str, ...]

    @property
    def has_new_obstacles(self) -> bool:
        return bool(self.newly_observed_ids)

    @property
    def has_unplanned_obstacles(self) -> bool:
        return bool(self.planning_candidate_ids)

    @property
    def next_obstacle(self) -> PerceivedObstacle | None:
        if not self.visible_obstacles:
            return None
        return self.visible_obstacles[0]

    def visible_field(self, revealed_only: bool = True) -> ObstacleField:
        """Return a semantic field for the visible window.

        When `revealed_only=True`, obstacles whose detailed geometry is not yet
        within the reveal distance are returned with width/size hidden.  This
        lets the next online replanning layer distinguish coarse detection from
        actionable semantic planning.
        """
        obstacles = []
        for perceived in self.visible_obstacles:
            obstacle = perceived.obstacle
            if revealed_only and not perceived.details_revealed:
                obstacle = hide_obstacle_details(obstacle)
            obstacles.append(obstacle)
        return ObstacleField(tuple(obstacles))

    def planning_field(self) -> ObstacleField:
        """Return visible, detailed, unplanned obstacles for local replanning."""
        candidates = [
            perceived.obstacle
            for perceived in self.visible_obstacles
            if perceived.requires_planning and perceived.details_revealed
        ]
        return ObstacleField(tuple(candidates))


class OnlineObstaclePerception:
    """Finite-lookahead semantic perception for online replanning demos.

    The class models a simple onboard perception assumption: the vehicle can
    detect obstacles inside a forward window, but detailed geometry becomes
    reliable only after the obstacle enters a nearer reveal distance.
    """

    def __init__(self, field: ObstacleField, config: OnlinePerceptionConfig | None = None):
        self.field = field
        self.config = config or OnlinePerceptionConfig()
        self.observed_obstacle_ids: set[str] = set()

    def update(
        self,
        leader_x: float,
        planned_obstacle_ids: set[str] | frozenset[str] | tuple[str, ...] | list[str] | None = None,
        cleared_obstacle_ids: set[str] | frozenset[str] | tuple[str, ...] | list[str] | None = None,
    ) -> PerceptionWindow:
        planned_ids = {str(item) for item in (planned_obstacle_ids or set())}
        cleared_ids = {str(item) for item in (cleared_obstacle_ids or set())}

        x_min = float(leader_x) - float(self.config.behind_tolerance)
        x_max = float(leader_x) + float(self.config.lookahead_distance)
        candidates = [
            obstacle
            for obstacle in self.field.obstacles
            if x_min <= float(obstacle.x) <= x_max and obstacle.obstacle_id not in cleared_ids
        ]
        candidates = candidates[: max(0, int(self.config.max_visible_obstacles))]

        perceived_obstacles: list[PerceivedObstacle] = []
        newly_observed_ids: list[str] = []
        planning_candidate_ids: list[str] = []
        for obstacle in candidates:
            obstacle_id = str(obstacle.obstacle_id)
            relative_distance = float(obstacle.x) - float(leader_x)
            newly_observed = obstacle_id not in self.observed_obstacle_ids
            if newly_observed:
                newly_observed_ids.append(obstacle_id)
            details_revealed = relative_distance <= float(self.config.detail_reveal_distance)
            requires_planning = (
                details_revealed
                and obstacle_id not in planned_ids
                and obstacle_id not in cleared_ids
            )
            if requires_planning:
                planning_candidate_ids.append(obstacle_id)
            perceived_obstacles.append(
                PerceivedObstacle(
                    obstacle=obstacle,
                    relative_distance_x=relative_distance,
                    newly_observed=newly_observed,
                    details_revealed=details_revealed,
                    requires_planning=requires_planning,
                )
            )

        self.observed_obstacle_ids.update(newly_observed_ids)
        return PerceptionWindow(
            leader_x=float(leader_x),
            x_min=x_min,
            x_max=x_max,
            visible_obstacles=tuple(perceived_obstacles),
            newly_observed_ids=tuple(newly_observed_ids),
            planning_candidate_ids=tuple(planning_candidate_ids),
        )

    def reset(self) -> None:
        self.observed_obstacle_ids.clear()


def hide_obstacle_details(obstacle: ObstacleDescriptor) -> ObstacleDescriptor:
    """Hide geometry that a finite-view sensor has not reliably measured yet."""
    if obstacle.function == 'aperture':
        return replace(obstacle, aperture_width=None)
    if obstacle.function == 'solid':
        return replace(obstacle, size_x=None, size_y=None, size_z=None)
    return obstacle
