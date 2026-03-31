from __future__ import annotations

from dataclasses import dataclass
from math import inf
from typing import Iterable

from .models import PlanResult, PipelineSpec, StagePlacement


@dataclass(frozen=True)
class GoalWeights:
    latency: float
    energy: float
    switches: float


@dataclass(frozen=True)
class OptimizerConfig:
    beam_width: int = 128
    exact_search_limit: int = 100_000

    def __post_init__(self) -> None:
        if self.beam_width <= 0:
            raise ValueError("beam_width must be greater than zero.")
        if self.exact_search_limit <= 0:
            raise ValueError("exact_search_limit must be greater than zero.")


@dataclass(frozen=True)
class _Normalizers:
    latency: float
    energy: float


@dataclass(frozen=True)
class _BeamState:
    prev_device: str | None
    placements: tuple[StagePlacement, ...]
    total_latency_ms: float
    total_energy_mj: float
    memory_by_device: dict[str, float]
    switches: int
    optimistic_score: float


GOAL_PRESETS: dict[str, GoalWeights] = {
    "latency": GoalWeights(latency=1.0, energy=0.20, switches=0.08),
    "efficiency": GoalWeights(latency=0.35, energy=1.0, switches=0.08),
    "balanced": GoalWeights(latency=0.75, energy=0.75, switches=0.08),
}


class EdgeInferenceOptimizer:
    def __init__(self, config: OptimizerConfig | None = None) -> None:
        self.config = config or OptimizerConfig()

    def optimize(self, pipeline: PipelineSpec, goal: str = "balanced", top_k: int = 3) -> list[PlanResult]:
        if not pipeline.devices:
            raise ValueError("Pipeline must define at least one device.")
        if not pipeline.stages:
            raise ValueError("Pipeline must define at least one stage.")
        if goal not in GOAL_PRESETS:
            supported = ", ".join(sorted(GOAL_PRESETS))
            raise ValueError(f"Unsupported goal {goal!r}. Supported goals: {supported}.")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        weights = GOAL_PRESETS[goal]
        normalizers = self._build_normalizers(pipeline)
        remaining_min_latency = self._suffix_minima(
            min(profile.latency_ms for profile in stage.profiles.values()) for stage in pipeline.stages
        )
        remaining_min_energy = self._suffix_minima(
            min(profile.energy_mj for profile in stage.profiles.values()) for stage in pipeline.stages
        )
        search_space = self._estimate_search_space(pipeline)

        if search_space <= self.config.exact_search_limit:
            return self._search_exact(
                pipeline=pipeline,
                goal=goal,
                top_k=top_k,
                weights=weights,
                normalizers=normalizers,
                remaining_min_latency=remaining_min_latency,
                remaining_min_energy=remaining_min_energy,
            )

        return self._search_beam(
            pipeline=pipeline,
            goal=goal,
            top_k=top_k,
            weights=weights,
            normalizers=normalizers,
            remaining_min_latency=remaining_min_latency,
            remaining_min_energy=remaining_min_energy,
        )

    def _search_exact(
        self,
        *,
        pipeline: PipelineSpec,
        goal: str,
        top_k: int,
        weights: GoalWeights,
        normalizers: _Normalizers,
        remaining_min_latency: list[float],
        remaining_min_energy: list[float],
    ) -> list[PlanResult]:
        best_results: list[PlanResult] = []
        zero_memory = {device_name: 0.0 for device_name in pipeline.devices}

        def maybe_add_result(result: PlanResult) -> None:
            best_results.append(result)
            best_results.sort(key=self._result_sort_key)
            del best_results[top_k:]

        def worst_score() -> float:
            if len(best_results) < top_k:
                return inf
            return best_results[-1].total_score

        def dfs(
            index: int,
            prev_device: str | None,
            placements: tuple[StagePlacement, ...],
            total_latency_ms: float,
            total_energy_mj: float,
            memory_by_device: dict[str, float],
            switches: int,
        ) -> None:
            if index == len(pipeline.stages):
                maybe_add_result(
                    self._build_result(
                        pipeline=pipeline,
                        goal=goal,
                        strategy="exact",
                        placements=placements,
                        total_latency_ms=total_latency_ms,
                        total_energy_mj=total_energy_mj,
                        memory_by_device=memory_by_device,
                        switches=switches,
                        weights=weights,
                        normalizers=normalizers,
                    )
                )
                return

            optimistic_score = self._score(
                latency_ms=total_latency_ms + remaining_min_latency[index],
                energy_mj=total_energy_mj + remaining_min_energy[index],
                switches=switches,
                weights=weights,
                normalizers=normalizers,
            )
            if optimistic_score >= worst_score():
                return

            stage = pipeline.stages[index]
            previous_output_mb = pipeline.stages[index - 1].output_mb if index > 0 else 0.0

            for device_name, profile in stage.profiles.items():
                transfer_latency_ms, transfer_energy_mj = 0.0, 0.0
                did_switch = prev_device is not None and prev_device != device_name
                if did_switch:
                    transfer_cost = self._transfer_cost_or_none(
                        prev_device,
                        device_name,
                        previous_output_mb,
                        pipeline,
                    )
                    if transfer_cost is None:
                        continue
                    transfer_latency_ms, transfer_energy_mj = transfer_cost

                new_total_latency_ms = total_latency_ms + transfer_latency_ms + profile.latency_ms
                new_total_energy_mj = total_energy_mj + transfer_energy_mj + profile.energy_mj
                if not self._within_constraints(pipeline, new_total_latency_ms, new_total_energy_mj):
                    continue

                new_memory_by_device = dict(memory_by_device)
                new_memory_by_device[device_name] += profile.memory_mb
                if new_memory_by_device[device_name] > pipeline.devices[device_name].memory_mb:
                    continue

                placement = StagePlacement(
                    stage_name=stage.name,
                    device_name=device_name,
                    execution_latency_ms=profile.latency_ms,
                    execution_energy_mj=profile.energy_mj,
                    memory_mb=profile.memory_mb,
                    transfer_latency_ms=transfer_latency_ms,
                    transfer_energy_mj=transfer_energy_mj,
                )

                dfs(
                    index=index + 1,
                    prev_device=device_name,
                    placements=placements + (placement,),
                    total_latency_ms=new_total_latency_ms,
                    total_energy_mj=new_total_energy_mj,
                    memory_by_device=new_memory_by_device,
                    switches=switches + int(did_switch),
                )

        dfs(
            index=0,
            prev_device=None,
            placements=(),
            total_latency_ms=0.0,
            total_energy_mj=0.0,
            memory_by_device=zero_memory,
            switches=0,
        )
        return best_results

    def _search_beam(
        self,
        *,
        pipeline: PipelineSpec,
        goal: str,
        top_k: int,
        weights: GoalWeights,
        normalizers: _Normalizers,
        remaining_min_latency: list[float],
        remaining_min_energy: list[float],
    ) -> list[PlanResult]:
        frontier = [
            _BeamState(
                prev_device=None,
                placements=(),
                total_latency_ms=0.0,
                total_energy_mj=0.0,
                memory_by_device={device_name: 0.0 for device_name in pipeline.devices},
                switches=0,
                optimistic_score=0.0,
            )
        ]

        for index, stage in enumerate(pipeline.stages):
            next_frontier: list[_BeamState] = []
            previous_output_mb = pipeline.stages[index - 1].output_mb if index > 0 else 0.0

            for state in frontier:
                for device_name, profile in stage.profiles.items():
                    transfer_latency_ms, transfer_energy_mj = 0.0, 0.0
                    did_switch = state.prev_device is not None and state.prev_device != device_name
                    if did_switch:
                        transfer_cost = self._transfer_cost_or_none(
                            state.prev_device,
                            device_name,
                            previous_output_mb,
                            pipeline,
                        )
                        if transfer_cost is None:
                            continue
                        transfer_latency_ms, transfer_energy_mj = transfer_cost

                    total_latency_ms = state.total_latency_ms + transfer_latency_ms + profile.latency_ms
                    total_energy_mj = state.total_energy_mj + transfer_energy_mj + profile.energy_mj
                    if not self._within_constraints(pipeline, total_latency_ms, total_energy_mj):
                        continue

                    memory_by_device = dict(state.memory_by_device)
                    memory_by_device[device_name] += profile.memory_mb
                    if memory_by_device[device_name] > pipeline.devices[device_name].memory_mb:
                        continue

                    placement = StagePlacement(
                        stage_name=stage.name,
                        device_name=device_name,
                        execution_latency_ms=profile.latency_ms,
                        execution_energy_mj=profile.energy_mj,
                        memory_mb=profile.memory_mb,
                        transfer_latency_ms=transfer_latency_ms,
                        transfer_energy_mj=transfer_energy_mj,
                    )

                    optimistic_score = self._score(
                        latency_ms=total_latency_ms + remaining_min_latency[index + 1],
                        energy_mj=total_energy_mj + remaining_min_energy[index + 1],
                        switches=state.switches + int(did_switch),
                        weights=weights,
                        normalizers=normalizers,
                    )
                    next_frontier.append(
                        _BeamState(
                            prev_device=device_name,
                            placements=state.placements + (placement,),
                            total_latency_ms=total_latency_ms,
                            total_energy_mj=total_energy_mj,
                            memory_by_device=memory_by_device,
                            switches=state.switches + int(did_switch),
                            optimistic_score=optimistic_score,
                        )
                    )

            next_frontier.sort(
                key=lambda state: (
                    state.optimistic_score,
                    state.total_latency_ms,
                    state.total_energy_mj,
                    state.switches,
                )
            )
            frontier = next_frontier[: self.config.beam_width]
            if not frontier:
                return []

        results = [
            self._build_result(
                pipeline=pipeline,
                goal=goal,
                strategy="beam",
                placements=state.placements,
                total_latency_ms=state.total_latency_ms,
                total_energy_mj=state.total_energy_mj,
                memory_by_device=state.memory_by_device,
                switches=state.switches,
                weights=weights,
                normalizers=normalizers,
            )
            for state in frontier
        ]
        results.sort(key=self._result_sort_key)
        return results[:top_k]

    def _build_normalizers(self, pipeline: PipelineSpec) -> _Normalizers:
        latency = sum(min(profile.latency_ms for profile in stage.profiles.values()) for stage in pipeline.stages)
        energy = sum(min(profile.energy_mj for profile in stage.profiles.values()) for stage in pipeline.stages)
        return _Normalizers(latency=max(latency, 1e-6), energy=max(energy, 1e-6))

    def _suffix_minima(self, values: Iterable[float]) -> list[float]:
        minima = list(values)
        suffix = [0.0] * (len(minima) + 1)
        for index in range(len(minima) - 1, -1, -1):
            suffix[index] = suffix[index + 1] + minima[index]
        return suffix

    def _estimate_search_space(self, pipeline: PipelineSpec) -> int:
        total = 1
        for stage in pipeline.stages:
            total *= len(stage.profiles)
        return total

    def _transfer_cost_or_none(
        self,
        source: str,
        target: str,
        data_mb: float,
        pipeline: PipelineSpec,
    ) -> tuple[float, float] | None:
        if source == target or data_mb <= 0:
            return 0.0, 0.0

        link = pipeline.transfer_links.get((source, target))
        if link is None:
            return None
        return link.cost_for(data_mb)

    def _score(
        self,
        *,
        latency_ms: float,
        energy_mj: float,
        switches: int,
        weights: GoalWeights,
        normalizers: _Normalizers,
    ) -> float:
        normalized_latency = latency_ms / normalizers.latency
        normalized_energy = energy_mj / normalizers.energy
        return (
            weights.latency * normalized_latency
            + weights.energy * normalized_energy
            + weights.switches * switches
        )

    def _within_constraints(
        self, pipeline: PipelineSpec, total_latency_ms: float, total_energy_mj: float
    ) -> bool:
        latency_cap = pipeline.constraints.max_total_latency_ms
        if latency_cap is not None and total_latency_ms > latency_cap:
            return False

        energy_cap = pipeline.constraints.max_total_energy_mj
        if energy_cap is not None and total_energy_mj > energy_cap:
            return False

        return True

    def _build_result(
        self,
        *,
        pipeline: PipelineSpec,
        goal: str,
        strategy: str,
        placements: tuple[StagePlacement, ...],
        total_latency_ms: float,
        total_energy_mj: float,
        memory_by_device: dict[str, float],
        switches: int,
        weights: GoalWeights,
        normalizers: _Normalizers,
    ) -> PlanResult:
        total_score = self._score(
            latency_ms=total_latency_ms,
            energy_mj=total_energy_mj,
            switches=switches,
            weights=weights,
            normalizers=normalizers,
        )
        explanation = self._build_explanation(pipeline, placements, memory_by_device)
        return PlanResult(
            pipeline_name=pipeline.name,
            goal=goal,
            strategy=strategy,
            assignments=placements,
            total_latency_ms=total_latency_ms,
            total_energy_mj=total_energy_mj,
            memory_by_device=dict(memory_by_device),
            total_score=total_score,
            switches=switches,
            explanation=explanation,
        )

    def _build_explanation(
        self,
        pipeline: PipelineSpec,
        placements: tuple[StagePlacement, ...],
        memory_by_device: dict[str, float],
    ) -> tuple[str, ...]:
        if not placements:
            return ()

        slowest_stage = max(placements, key=lambda placement: placement.total_latency_ms)
        lines = [
            (
                f"{slowest_stage.stage_name} is the latency bottleneck on {slowest_stage.device_name} "
                f"at {slowest_stage.total_latency_ms:.2f} ms."
            )
        ]

        highest_util_device = max(
            pipeline.devices,
            key=lambda device_name: memory_by_device[device_name] / pipeline.devices[device_name].memory_mb,
        )
        utilization = memory_by_device[highest_util_device] / pipeline.devices[highest_util_device].memory_mb
        lines.append(
            f"{highest_util_device} uses {utilization:.0%} of its memory budget in this placement."
        )

        if any(placement.transfer_latency_ms > 0 for placement in placements):
            heaviest_transfer = max(placements, key=lambda placement: placement.transfer_latency_ms)
            if heaviest_transfer.transfer_latency_ms > 0:
                lines.append(
                    (
                        f"The handoff into {heaviest_transfer.stage_name} adds "
                        f"{heaviest_transfer.transfer_latency_ms:.2f} ms of copy overhead."
                    )
                )
        else:
            lines.append("The pipeline stays on one accelerator end-to-end, so transfer overhead is zero.")

        return tuple(lines)

    def _result_sort_key(self, result: PlanResult) -> tuple[float, float, float, int]:
        return (result.total_score, result.total_latency_ms, result.total_energy_mj, result.switches)
