from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from typing import Any

from .llm_context import build_llm_scene_context
from .llm_strategy import LLMStrategyGenerator, LLMStrategyPipeline, MockLLMStrategyGenerator
from .online_perception import PerceptionWindow
from .passability import PassabilityEvaluator
from .strategy_planner import SpatioTemporalStrategyPlanner
from .strategy_validator import StrategyValidator
from .structures import FormationState, MissionPreference, ObstacleField, ObstaclePassageStrategy, PassagePlan, ValidationResult


@dataclass(frozen=True)
class OnlineReplanningConfig:
    """Policy settings for event-triggered online LLM replanning."""

    min_replan_interval: float = 6.0
    gate_observe_distance_x: float = 5.2
    gate_pass_clear_x: float = 1.25
    snake_spacing_x: float = 0.8
    use_mock_when_generator_missing: bool = True


@dataclass(frozen=True)
class OnlineStrategyBuffer:
    """Accepted local strategies accumulated during online flight."""

    planned_obstacle_ids: frozenset[str] = field(default_factory=frozenset)
    cleared_obstacle_ids: frozenset[str] = field(default_factory=frozenset)
    obstacle_strategies: tuple[ObstaclePassageStrategy, ...] = field(default_factory=tuple)
    accepted_plan_ids: tuple[str, ...] = field(default_factory=tuple)

    def with_cleared(self, obstacle_ids: set[str] | list[str] | tuple[str, ...]) -> 'OnlineStrategyBuffer':
        cleared = frozenset(set(self.cleared_obstacle_ids).union(str(item) for item in obstacle_ids))
        planned = frozenset(item for item in self.planned_obstacle_ids if item not in cleared)
        strategies = tuple(strategy for strategy in self.obstacle_strategies if strategy.obstacle_id not in cleared)
        return OnlineStrategyBuffer(
            planned_obstacle_ids=planned,
            cleared_obstacle_ids=cleared,
            obstacle_strategies=strategies,
            accepted_plan_ids=self.accepted_plan_ids,
        )

    def with_plan(self, plan: PassagePlan) -> 'OnlineStrategyBuffer':
        strategies_by_id = {strategy.obstacle_id: strategy for strategy in self.obstacle_strategies}
        for strategy in plan.obstacle_strategies:
            if strategy.obstacle_id not in self.cleared_obstacle_ids:
                strategies_by_id[strategy.obstacle_id] = strategy
        ordered = tuple(sorted(strategies_by_id.values(), key=lambda item: (item.obstacle_x, item.obstacle_index)))
        planned = frozenset(set(self.planned_obstacle_ids).union(strategy.obstacle_id for strategy in plan.obstacle_strategies))
        return OnlineStrategyBuffer(
            planned_obstacle_ids=planned,
            cleared_obstacle_ids=self.cleared_obstacle_ids,
            obstacle_strategies=ordered,
            accepted_plan_ids=self.accepted_plan_ids + (plan.plan_id,),
        )

    def strategy_for(self, obstacle_id: str) -> ObstaclePassageStrategy | None:
        for strategy in self.obstacle_strategies:
            if strategy.obstacle_id == obstacle_id:
                return strategy
        return None


@dataclass(frozen=True)
class OnlineReplanResult:
    """Outcome of one online replanning evaluation."""

    triggered: bool
    accepted: bool
    reason: str
    current_time: float
    visible_obstacle_ids: tuple[str, ...]
    planning_obstacle_ids: tuple[str, ...]
    deterministic_plan: PassagePlan | None = None
    selected_plan: PassagePlan | None = None
    validation: ValidationResult | None = None
    used_fallback: bool = False
    scene_context: dict[str, Any] | None = None
    buffer: OnlineStrategyBuffer | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            'triggered': bool(self.triggered),
            'accepted': bool(self.accepted),
            'reason': self.reason,
            'current_time': float(self.current_time),
            'visible_obstacle_ids': list(self.visible_obstacle_ids),
            'planning_obstacle_ids': list(self.planning_obstacle_ids),
            'selected_plan_id': None if self.selected_plan is None else self.selected_plan.plan_id,
            'selected_plan_source': None if self.selected_plan is None else self.selected_plan.source,
            'validation_valid': None if self.validation is None else bool(self.validation.valid),
            'validation_repaired': None if self.validation is None else bool(self.validation.repaired),
            'used_fallback': bool(self.used_fallback),
            'planned_buffer_ids': [] if self.buffer is None else sorted(self.buffer.planned_obstacle_ids),
        }


class OnlineReplanner:
    """Event-triggered local strategy generator for finite-view flight.

    The replanner consumes a `PerceptionWindow`, creates a local planning field
    from detailed and unplanned obstacles, asks an LLM-compatible generator for
    a local `PassagePlan`, validates it, and merges accepted strategies into an
    online strategy buffer.  It does not command the UAVs directly.
    """

    def __init__(
        self,
        formation: FormationState,
        mission: MissionPreference,
        config: OnlineReplanningConfig | None = None,
        planner: SpatioTemporalStrategyPlanner | None = None,
        validator: StrategyValidator | None = None,
        pipeline: LLMStrategyPipeline | None = None,
    ):
        self.formation = formation
        self.mission = mission
        self.config = config or OnlineReplanningConfig()
        self.planner = planner or SpatioTemporalStrategyPlanner(
            gate_observe_distance_x=self.config.gate_observe_distance_x,
            gate_pass_clear_x=self.config.gate_pass_clear_x,
            snake_spacing_x=self.config.snake_spacing_x,
        )
        self.validator = validator or StrategyValidator()
        self.pipeline = pipeline or LLMStrategyPipeline(validator=self.validator, debug=False)
        self.buffer = OnlineStrategyBuffer()
        self.last_replan_time: float | None = None

    def mark_cleared(self, obstacle_ids: set[str] | list[str] | tuple[str, ...]) -> None:
        self.buffer = self.buffer.with_cleared(obstacle_ids)

    def should_trigger(self, window: PerceptionWindow, current_time: float) -> tuple[bool, str]:
        if not window.visible_obstacles:
            return False, 'no_visible_obstacles'
        local_field = self._local_planning_field(window)
        if not local_field.obstacles:
            return False, 'no_detailed_unplanned_obstacles'
        if self.last_replan_time is not None:
            elapsed = float(current_time) - float(self.last_replan_time)
            if elapsed < float(self.config.min_replan_interval):
                return False, 'min_replan_interval_not_elapsed'
        if window.newly_observed_ids:
            return True, 'new_obstacle_window_detected'
        return True, 'unplanned_detailed_obstacle_window'

    def evaluate(
        self,
        window: PerceptionWindow,
        current_time: float,
        generator: LLMStrategyGenerator | None = None,
    ) -> OnlineReplanResult:
        visible_ids = tuple(item.obstacle_id for item in window.visible_obstacles)
        local_field = self._local_planning_field(window)
        local_ids = tuple(obstacle.obstacle_id for obstacle in local_field.obstacles)
        trigger, reason = self.should_trigger(window, current_time)
        if not trigger:
            return OnlineReplanResult(
                triggered=False,
                accepted=False,
                reason=reason,
                current_time=float(current_time),
                visible_obstacle_ids=visible_ids,
                planning_obstacle_ids=local_ids,
                buffer=self.buffer,
            )

        reports = PassabilityEvaluator().evaluate_field(local_field, self.formation, self.mission)
        deterministic_plan = self.planner.plan(local_field, self.formation, self.mission, reports)
        deterministic_validation = self.validator.validate(deterministic_plan, local_field, self.formation)
        if not deterministic_validation.valid:
            return OnlineReplanResult(
                triggered=True,
                accepted=False,
                reason='deterministic_local_plan_invalid',
                current_time=float(current_time),
                visible_obstacle_ids=visible_ids,
                planning_obstacle_ids=tuple(obstacle.obstacle_id for obstacle in local_field.obstacles),
                deterministic_plan=deterministic_plan,
                validation=deterministic_validation,
                buffer=self.buffer,
            )

        scene_context = build_llm_scene_context(local_field, self.formation, self.mission, deterministic_plan)
        if generator is None and self.config.use_mock_when_generator_missing:
            generator = MockLLMStrategyGenerator(deterministic_plan, mode='valid')
        if generator is None:
            selected_plan = deterministic_plan
            validation = deterministic_validation
            used_fallback = False
        else:
            selected_plan, validation, used_fallback = self.pipeline.generate_or_fallback(
                generator,
                scene_context,
                deterministic_plan,
                local_field,
                self.formation,
            )

        if not validation.valid:
            return OnlineReplanResult(
                triggered=True,
                accepted=False,
                reason='local_llm_plan_invalid',
                current_time=float(current_time),
                visible_obstacle_ids=visible_ids,
                planning_obstacle_ids=tuple(obstacle.obstacle_id for obstacle in local_field.obstacles),
                deterministic_plan=deterministic_plan,
                selected_plan=selected_plan,
                validation=validation,
                used_fallback=used_fallback,
                scene_context=scene_context,
                buffer=self.buffer,
            )

        self.buffer = self.buffer.with_plan(selected_plan)
        self.last_replan_time = float(current_time)
        return OnlineReplanResult(
            triggered=True,
            accepted=True,
            reason=reason,
            current_time=float(current_time),
            visible_obstacle_ids=visible_ids,
            planning_obstacle_ids=tuple(obstacle.obstacle_id for obstacle in local_field.obstacles),
            deterministic_plan=deterministic_plan,
            selected_plan=selected_plan,
            validation=validation,
            used_fallback=used_fallback,
            scene_context=scene_context,
            buffer=self.buffer,
        )

    def _local_planning_field(self, window: PerceptionWindow) -> ObstacleField:
        obstacles = []
        for perceived in window.visible_obstacles:
            obstacle_id = str(perceived.obstacle_id)
            if not perceived.details_revealed:
                continue
            if obstacle_id in self.buffer.planned_obstacle_ids:
                continue
            if obstacle_id in self.buffer.cleared_obstacle_ids:
                continue
            if not perceived.requires_planning and obstacle_id not in window.planning_candidate_ids:
                continue
            obstacles.append(perceived.obstacle)
        return ObstacleField(tuple(obstacles))


@dataclass(frozen=True)
class AsyncOnlineReplanEvent:
    """Non-blocking replanning event emitted to the control loop."""

    event_type: str
    current_time: float
    obstacle_ids: tuple[str, ...]
    message: str
    result: OnlineReplanResult | None = None
    fallback_result: OnlineReplanResult | None = None
    latency_s: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            'event_type': self.event_type,
            'current_time': float(self.current_time),
            'obstacle_ids': list(self.obstacle_ids),
            'message': self.message,
            'latency_s': self.latency_s,
            'result': None if self.result is None else self.result.to_dict(),
            'fallback_result': None if self.fallback_result is None else self.fallback_result.to_dict(),
        }


@dataclass
class _PendingOnlineRequest:
    obstacle_ids: tuple[str, ...]
    submitted_at: float
    future: Future
    fallback_result: OnlineReplanResult


class AsyncOnlineReplanningManager:
    """Asynchronous qwen/LLM wrapper around OnlineReplanner.

    A deterministic local strategy is accepted immediately as a safety fallback.
    The external LLM request runs in a background thread; when it returns, the
    result is validated and can replace/augment the same strategy buffer.
    """

    def __init__(
        self,
        replanner: OnlineReplanner,
        generator: LLMStrategyGenerator | None,
        max_workers: int = 1,
    ):
        self.replanner = replanner
        self.generator = generator
        self.executor = ThreadPoolExecutor(max_workers=max(1, int(max_workers)))
        self.pending: dict[tuple[str, ...], _PendingOnlineRequest] = {}
        self.closed = False

    @property
    def buffer(self) -> OnlineStrategyBuffer:
        return self.replanner.buffer

    def mark_cleared(self, obstacle_ids: set[str] | list[str] | tuple[str, ...]) -> None:
        self.replanner.mark_cleared(obstacle_ids)

    def submit_if_needed(self, window: PerceptionWindow, current_time: float) -> list[AsyncOnlineReplanEvent]:
        if self.closed:
            return []
        visible_ids = tuple(item.obstacle_id for item in window.visible_obstacles)
        local_field = self.replanner._local_planning_field(window)
        local_ids = tuple(obstacle.obstacle_id for obstacle in local_field.obstacles)
        trigger, reason = self.replanner.should_trigger(window, current_time)
        if not trigger:
            return [
                AsyncOnlineReplanEvent(
                    event_type='not_triggered',
                    current_time=float(current_time),
                    obstacle_ids=visible_ids,
                    message=reason,
                )
            ]
        if local_ids in self.pending:
            return [
                AsyncOnlineReplanEvent(
                    event_type='pending',
                    current_time=float(current_time),
                    obstacle_ids=local_ids,
                    message='LLM request already pending; control loop continues with fallback strategy',
                    fallback_result=self.pending[local_ids].fallback_result,
                )
            ]

        fallback_result = self.replanner.evaluate(window, current_time=current_time, generator=None)
        events = [
            AsyncOnlineReplanEvent(
                event_type='fallback_accepted' if fallback_result.accepted else 'fallback_rejected',
                current_time=float(current_time),
                obstacle_ids=local_ids,
                message=(
                    'deterministic local fallback accepted immediately'
                    if fallback_result.accepted
                    else 'deterministic local fallback was rejected'
                ),
                fallback_result=fallback_result,
            )
        ]
        if self.generator is None or not fallback_result.accepted:
            return events

        future = self.executor.submit(
            self._run_external_llm,
            local_field,
            fallback_result.deterministic_plan,
            tuple(fallback_result.visible_obstacle_ids),
            tuple(fallback_result.planning_obstacle_ids),
            float(current_time),
        )
        self.pending[local_ids] = _PendingOnlineRequest(
            obstacle_ids=local_ids,
            submitted_at=float(current_time),
            future=future,
            fallback_result=fallback_result,
        )
        events.append(
            AsyncOnlineReplanEvent(
                event_type='submitted',
                current_time=float(current_time),
                obstacle_ids=local_ids,
                message='external LLM request submitted asynchronously; viewer/control loop is not blocked',
                fallback_result=fallback_result,
            )
        )
        return events

    def poll_completed(self, current_time: float) -> list[AsyncOnlineReplanEvent]:
        events: list[AsyncOnlineReplanEvent] = []
        for key, pending in list(self.pending.items()):
            if not pending.future.done():
                continue
            try:
                result = pending.future.result()
            except Exception as exc:
                result = OnlineReplanResult(
                    triggered=True,
                    accepted=False,
                    reason=f'external_llm_exception: {exc}',
                    current_time=float(current_time),
                    visible_obstacle_ids=tuple(),
                    planning_obstacle_ids=pending.obstacle_ids,
                    buffer=self.replanner.buffer,
                )
            latency = float(current_time) - float(pending.submitted_at)
            if result.accepted:
                self.replanner.buffer = self.replanner.buffer.with_plan(result.selected_plan)
                self.replanner.last_replan_time = float(current_time)
                result = replace(result, buffer=self.replanner.buffer)
            events.append(
                AsyncOnlineReplanEvent(
                    event_type='completed_accepted' if result.accepted else 'completed_rejected',
                    current_time=float(current_time),
                    obstacle_ids=pending.obstacle_ids,
                    message=(
                        'external LLM strategy accepted and merged into online buffer'
                        if result.accepted
                        else 'external LLM strategy rejected; fallback strategy remains active'
                    ),
                    result=result,
                    fallback_result=pending.fallback_result,
                    latency_s=latency,
                )
            )
            del self.pending[key]
        return events

    def close(self) -> None:
        self.closed = True
        self.executor.shutdown(wait=False, cancel_futures=True)

    def _run_external_llm(
        self,
        local_field: ObstacleField,
        deterministic_plan: PassagePlan,
        visible_ids: tuple[str, ...],
        planning_ids: tuple[str, ...],
        request_time: float,
    ) -> OnlineReplanResult:
        scene_context = build_llm_scene_context(local_field, self.replanner.formation, self.replanner.mission, deterministic_plan)
        selected_plan, validation, used_fallback = self.replanner.pipeline.generate_or_fallback(
            self.generator,
            scene_context,
            deterministic_plan,
            local_field,
            self.replanner.formation,
        )
        accepted = bool(validation.valid)
        return OnlineReplanResult(
            triggered=True,
            accepted=accepted,
            reason='external_llm_completed',
            current_time=float(request_time),
            visible_obstacle_ids=visible_ids,
            planning_obstacle_ids=planning_ids,
            deterministic_plan=deterministic_plan,
            selected_plan=selected_plan,
            validation=validation,
            used_fallback=used_fallback,
            scene_context=scene_context,
            buffer=self.replanner.buffer,
        )
