from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .models import Constraints, DeviceProfile, ExecutionProfile, PipelineSpec, StageSpec, TransferLink


def load_pipeline(path: str | Path) -> PipelineSpec:
    scenario_path = Path(path)
    with scenario_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return pipeline_from_dict(payload)


def pipeline_from_dict(data: Mapping[str, Any]) -> PipelineSpec:
    name = str(data["name"])
    description = str(data.get("description", ""))

    devices: dict[str, DeviceProfile] = {}
    for raw_device in data["devices"]:
        device = DeviceProfile(
            name=str(raw_device["name"]),
            label=str(raw_device.get("label", raw_device["name"])),
            memory_mb=float(raw_device["memory_mb"]),
        )
        devices[device.name] = device

    transfer_links: dict[tuple[str, str], TransferLink] = {}
    for raw_link in data.get("links", []):
        source = str(raw_link["source"])
        target = str(raw_link["target"])
        _validate_device_name(devices, source)
        _validate_device_name(devices, target)
        link = TransferLink(
            source=source,
            target=target,
            latency_ms_per_mb=float(raw_link["latency_ms_per_mb"]),
            energy_mj_per_mb=float(raw_link["energy_mj_per_mb"]),
        )
        transfer_links[(source, target)] = link
        if raw_link.get("bidirectional", True):
            transfer_links[(target, source)] = TransferLink(
                source=target,
                target=source,
                latency_ms_per_mb=link.latency_ms_per_mb,
                energy_mj_per_mb=link.energy_mj_per_mb,
            )

    stages: list[StageSpec] = []
    for raw_stage in data["stages"]:
        profiles: dict[str, ExecutionProfile] = {}
        for device_name, raw_profile in raw_stage["profiles"].items():
            _validate_device_name(devices, device_name)
            profiles[device_name] = ExecutionProfile(
                latency_ms=float(raw_profile["latency_ms"]),
                energy_mj=float(raw_profile["energy_mj"]),
                memory_mb=float(raw_profile["memory_mb"]),
            )
        if not profiles:
            raise ValueError(f"Stage {raw_stage['name']!r} must define at least one execution profile.")

        stages.append(
            StageSpec(
                name=str(raw_stage["name"]),
                output_mb=float(raw_stage.get("output_mb", 0.0)),
                profiles=profiles,
                description=str(raw_stage.get("description", "")),
            )
        )

    constraints = Constraints(
        max_total_latency_ms=_optional_float(data.get("constraints", {}).get("max_total_latency_ms")),
        max_total_energy_mj=_optional_float(data.get("constraints", {}).get("max_total_energy_mj")),
    )

    return PipelineSpec(
        name=name,
        description=description,
        devices=devices,
        stages=tuple(stages),
        transfer_links=transfer_links,
        constraints=constraints,
    )


def _validate_device_name(devices: Mapping[str, DeviceProfile], device_name: str) -> None:
    if device_name not in devices:
        raise ValueError(f"Unknown device {device_name!r} referenced in scenario.")


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)
