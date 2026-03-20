from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DeviceProfile:
    name: str
    label: str
    memory_mb: float


@dataclass(frozen=True)
class TransferLink:
    source: str
    target: str
    latency_ms_per_mb: float
    energy_mj_per_mb: float

    def cost_for(self, data_mb: float) -> tuple[float, float]:
        return self.latency_ms_per_mb * data_mb, self.energy_mj_per_mb * data_mb


@dataclass(frozen=True)
class ExecutionProfile:
    latency_ms: float
    energy_mj: float
    memory_mb: float


@dataclass(frozen=True)
class StageSpec:
    name: str
    output_mb: float
    profiles: dict[str, ExecutionProfile]
    description: str = ""


@dataclass(frozen=True)
class Constraints:
    max_total_latency_ms: float | None = None
    max_total_energy_mj: float | None = None


@dataclass(frozen=True)
class PipelineSpec:
    name: str
    description: str
    devices: dict[str, DeviceProfile]
    stages: tuple[StageSpec, ...]
    transfer_links: dict[tuple[str, str], TransferLink] = field(default_factory=dict)
    constraints: Constraints = field(default_factory=Constraints)

    def transfer_cost(self, source: str, target: str, data_mb: float) -> tuple[float, float]:
        if source == target or data_mb <= 0:
            return 0.0, 0.0

        link = self.transfer_links.get((source, target))
        if link is None:
            raise ValueError(f"No transfer link defined between {source!r} and {target!r}.")

        return link.cost_for(data_mb)


@dataclass(frozen=True)
class StagePlacement:
    stage_name: str
    device_name: str
    execution_latency_ms: float
    execution_energy_mj: float
    memory_mb: float
    transfer_latency_ms: float = 0.0
    transfer_energy_mj: float = 0.0

    @property
    def total_latency_ms(self) -> float:
        return self.execution_latency_ms + self.transfer_latency_ms

    @property
    def total_energy_mj(self) -> float:
        return self.execution_energy_mj + self.transfer_energy_mj

    def to_dict(self) -> dict[str, float | str]:
        return {
            "stage_name": self.stage_name,
            "device_name": self.device_name,
            "execution_latency_ms": self.execution_latency_ms,
            "transfer_latency_ms": self.transfer_latency_ms,
            "total_latency_ms": self.total_latency_ms,
            "execution_energy_mj": self.execution_energy_mj,
            "transfer_energy_mj": self.transfer_energy_mj,
            "total_energy_mj": self.total_energy_mj,
            "memory_mb": self.memory_mb,
        }


@dataclass(frozen=True)
class PlanResult:
    pipeline_name: str
    goal: str
    strategy: str
    assignments: tuple[StagePlacement, ...]
    total_latency_ms: float
    total_energy_mj: float
    memory_by_device: dict[str, float]
    total_score: float
    switches: int
    explanation: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "pipeline_name": self.pipeline_name,
            "goal": self.goal,
            "strategy": self.strategy,
            "total_latency_ms": self.total_latency_ms,
            "total_energy_mj": self.total_energy_mj,
            "total_score": self.total_score,
            "switches": self.switches,
            "memory_by_device": self.memory_by_device,
            "explanation": list(self.explanation),
            "assignments": [assignment.to_dict() for assignment in self.assignments],
        }
