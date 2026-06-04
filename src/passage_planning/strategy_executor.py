from __future__ import annotations

import math
import numpy as np

from .structures import PassagePlan, PassageSlot


class StrategyExecutionHelper:
    """Small adapter that lets legacy MuJoCo control loops consume PassagePlan."""

    def __init__(self, plan: PassagePlan):
        self.plan = plan

    def order_index(self, drone_id: int) -> int:
        return self.plan.order_index(drone_id)

    def slot_for_drone(self, drone_id: int) -> PassageSlot:
        for slot in self.plan.slots:
            if int(slot.drone_id) == int(drone_id):
                return slot
        raise KeyError(f'missing passage slot for UAV {drone_id}')

    def nominal_slot_entry_time(self, drone_id: int) -> float:
        return float(self.slot_for_drone(drone_id).nominal_entry_time)

    @staticmethod
    def order_index_in(passing_order: tuple[int, ...] | list[int], drone_id: int) -> int:
        return tuple(int(item) for item in passing_order).index(int(drone_id))

    def previous_drone_in_order(
        self,
        drone_id: int,
        passing_order: tuple[int, ...] | list[int] | None = None,
    ) -> int | None:
        if passing_order is None:
            passing_order = self.plan.passing_order
        passing_order = tuple(int(item) for item in passing_order)
        order_index = self.order_index_in(passing_order, drone_id)
        if order_index <= 0:
            return None
        return int(passing_order[order_index - 1])

    def temporal_speed_limit(
        self,
        drone_id: int,
        current_time: float,
        obstacle_crossing_times: dict[int, float],
        current_x: float,
        obstacle_x: float,
        commit_distance_x: float,
        wait_speed_x: float,
        current_speed_x: float | None = None,
        min_speed_x: float = 0.04,
        release_margin_s: float = 0.16,
        passing_order: tuple[int, ...] | list[int] | None = None,
    ) -> float | None:
        """Return a smooth x-speed cap when a UAV is predicted too early.

        Slots are enforced near each obstacle's commit region.  This preserves
        upstream alignment and only gates the safety-critical crossing moment.
        """
        previous_drone = self.previous_drone_in_order(drone_id, passing_order=passing_order)
        if previous_drone is None:
            return None
        if float(current_x) < float(obstacle_x) - float(commit_distance_x):
            return None
        if float(current_x) >= float(obstacle_x):
            return None

        remaining_distance = max(0.0, float(obstacle_x) - float(current_x))
        previous_crossing_time = obstacle_crossing_times.get(previous_drone)
        if previous_crossing_time is None:
            if remaining_distance <= 0.18:
                return None
            return self._soft_arrival_speed_cap(
                remaining_distance=remaining_distance,
                remaining_time=float(self.plan.time_slot_interval),
                current_speed_x=current_speed_x,
                wait_speed_x=wait_speed_x,
                min_speed_x=min_speed_x,
            )

        elapsed_since_previous = float(current_time) - float(previous_crossing_time)
        remaining_time = float(self.plan.time_slot_interval) - elapsed_since_previous
        if remaining_time > float(release_margin_s):
            predicted_time_to_cross = remaining_distance / max(float(current_speed_x or 0.0), 1e-6)
            if predicted_time_to_cross + float(release_margin_s) >= remaining_time:
                return None
            return self._soft_arrival_speed_cap(
                remaining_distance=remaining_distance,
                remaining_time=remaining_time,
                current_speed_x=current_speed_x,
                wait_speed_x=wait_speed_x,
                min_speed_x=min_speed_x,
            )
        return None

    @staticmethod
    def _soft_arrival_speed_cap(
        remaining_distance: float,
        remaining_time: float,
        current_speed_x: float | None,
        wait_speed_x: float,
        min_speed_x: float,
    ) -> float:
        target_speed = float(remaining_distance) / max(float(remaining_time) + 0.12, 1e-6)
        current_speed = float(current_speed_x) if current_speed_x is not None else float(wait_speed_x)
        speed_cap = 0.65 * current_speed + 0.35 * target_speed
        speed_cap = min(float(wait_speed_x), speed_cap)
        return float(max(float(min_speed_x), speed_cap))

    @staticmethod
    def record_obstacle_crossings(
        crossing_times: dict[int, float],
        positions: list[np.ndarray] | tuple[np.ndarray, ...],
        obstacle_x: float,
        current_time: float,
    ) -> None:
        if not math.isfinite(float(obstacle_x)):
            return
        for drone_id, position in enumerate(positions):
            if drone_id in crossing_times:
                continue
            if float(np.asarray(position, dtype=float)[0]) >= float(obstacle_x):
                crossing_times[int(drone_id)] = float(current_time)

    def active_obstacle_index_for_position(self, position: np.ndarray) -> int:
        x = float(np.asarray(position, dtype=float)[0])
        for strategy in self.plan.obstacle_strategies:
            if x < float(strategy.clear_x):
                return int(strategy.obstacle_index)
        return int(len(self.plan.obstacle_strategies) - 1)

    def should_hold_after_final_obstacle(self, position: np.ndarray, hold_clear_x: float) -> bool:
        if not self.plan.recover_after_last_obstacle:
            return False
        final_strategy = self.plan.obstacle_strategies[-1]
        return float(np.asarray(position, dtype=float)[0]) >= float(final_strategy.obstacle_x + hold_clear_x)

    def first_obstacle_observe_x(self) -> float:
        return float(self.plan.obstacle_strategies[0].observe_x)

    def all_strategies_are(self, mode: str) -> bool:
        return all(strategy.mode == mode for strategy in self.plan.obstacle_strategies)

    @staticmethod
    def bypass_target(
        position: np.ndarray,
        obstacle_center: np.ndarray,
        obstacle_size: np.ndarray,
        original_y: float,
        clearance_xy: float,
        clearance_z: float,
        route_hint: str = 'auto',
    ) -> np.ndarray:
        """Generate a deterministic left/right/over/under waypoint for a solid obstacle.

        This helper is intentionally simple and stable.  It gives the future
        mixed-obstacle executor a bounded local target without pretending to be
        a full global planner.
        """
        pos = np.asarray(position, dtype=float)
        center = np.asarray(obstacle_center, dtype=float)
        size = np.asarray(obstacle_size, dtype=float)
        half_y = 0.5 * abs(float(size[1]))
        half_z = 0.5 * abs(float(size[2]))
        target_x = float(center[0] + 0.5 * abs(float(size[0])) + clearance_xy)
        raw_under_z = float(center[2] - half_z - clearance_z)
        under_z = max(0.30, raw_under_z)

        candidates = {
            'left': np.array([target_x, center[1] - half_y - clearance_xy, pos[2]], dtype=float),
            'right': np.array([target_x, center[1] + half_y + clearance_xy, pos[2]], dtype=float),
            'over': np.array([target_x, float(original_y), center[2] + half_z + clearance_z], dtype=float),
            'under': np.array([target_x, float(original_y), under_z], dtype=float),
        }
        if route_hint in candidates:
            return candidates[route_hint]

        preferred_side = 'left' if float(original_y) <= float(center[1]) else 'right'
        side_target = candidates[preferred_side]
        side_cost = float(np.linalg.norm(side_target[:2] - pos[:2]))

        vertical_candidates = [('over', candidates['over'])]
        if raw_under_z > 0.35:
            vertical_candidates.insert(0, ('under', candidates['under']))
        _, vertical_target = min(
            vertical_candidates,
            key=lambda item: float(
                np.linalg.norm(item[1][:2] - pos[:2])
                + 0.45 * abs(float(item[1][2] - pos[2]))
            ),
        )
        vertical_cost = float(
            np.linalg.norm(vertical_target[:2] - pos[:2])
            + 0.45 * abs(float(vertical_target[2] - pos[2]))
        )

        # UAVs can usually exploit vertical free space faster than a wide
        # lateral detour, so auto mode deliberately prefers feasible over/under
        # routes unless the side route is clearly cheaper.
        if vertical_cost <= 1.25 * side_cost:
            return vertical_target
        return side_target
