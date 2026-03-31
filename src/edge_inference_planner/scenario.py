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

    raw_devices = data["devices"]
    if not raw_devices:
        raise ValueError("Scenario must define at least one device.")

    devices: dict[str, DeviceProfile] = {}
    for raw_device in raw_devices:
        device_name = str(raw_device["name"])
        _validate_non_empty_name(device_name, "Device name")
        if device_name in devices:
            raise ValueError(f"Duplicate device name {device_name!r} in scenario.")

        device = DeviceProfile(
            name=device_name,
            label=str(raw_device.get("label", device_name)),
            memory_mb=_require_positive_float(raw_device["memory_mb"], f"Device {device_name!r} memory_mb"),
        )
        devices[device.name] = device

    transfer_links: dict[tuple[str, str], TransferLink] = {}
    for raw_link in data.get("links", []):
        source = str(raw_link["source"])
        target = str(raw_link["target"])
        _validate_non_empty_name(source, "Link source")
        _validate_non_empty_name(target, "Link target")
        _validate_device_name(devices, source)
        _validate_device_name(devices, target)
        link = TransferLink(
            source=source,
            target=target,
            latency_ms_per_mb=_require_nonnegative_float(
                raw_link["latency_ms_per_mb"],
                f"Transfer link {source!r}->{target!r} latency_ms_per_mb",
            ),
            energy_mj_per_mb=_require_nonnegative_float(
                raw_link["energy_mj_per_mb"],
                f"Transfer link {source!r}->{target!r} energy_mj_per_mb",
            ),
        )
        _register_transfer_link(transfer_links, link)
        if raw_link.get("bidirectional", True):
            _register_transfer_link(
                transfer_links,
                TransferLink(
                    source=target,
                    target=source,
                    latency_ms_per_mb=link.latency_ms_per_mb,
                    energy_mj_per_mb=link.energy_mj_per_mb,
                ),
            )

    raw_stages = data["stages"]
    if not raw_stages:
        raise ValueError("Scenario must define at least one stage.")

    stages: list[StageSpec] = []
    stage_names: set[str] = set()
    for raw_stage in raw_stages:
        stage_name = str(raw_stage["name"])
        _validate_non_empty_name(stage_name, "Stage name")
        if stage_name in stage_names:
            raise ValueError(f"Duplicate stage name {stage_name!r} in scenario.")
        stage_names.add(stage_name)

        profiles: dict[str, ExecutionProfile] = {}
        for device_name, raw_profile in raw_stage["profiles"].items():
            _validate_device_name(devices, device_name)
            profiles[device_name] = ExecutionProfile(
                latency_ms=_require_nonnegative_float(
                    raw_profile["latency_ms"],
                    f"Stage {stage_name!r} profile {device_name!r} latency_ms",
                ),
                energy_mj=_require_nonnegative_float(
                    raw_profile["energy_mj"],
                    f"Stage {stage_name!r} profile {device_name!r} energy_mj",
                ),
                memory_mb=_require_nonnegative_float(
                    raw_profile["memory_mb"],
                    f"Stage {stage_name!r} profile {device_name!r} memory_mb",
                ),
            )
        if not profiles:
            raise ValueError(f"Stage {stage_name!r} must define at least one execution profile.")

        stages.append(
            StageSpec(
                name=stage_name,
                output_mb=_require_nonnegative_float(
                    raw_stage.get("output_mb", 0.0),
                    f"Stage {stage_name!r} output_mb",
                ),
                profiles=profiles,
                description=str(raw_stage.get("description", "")),
            )
        )

    constraints = Constraints(
        max_total_latency_ms=_optional_nonnegative_float(
            data.get("constraints", {}).get("max_total_latency_ms"),
            "constraints.max_total_latency_ms",
        ),
        max_total_energy_mj=_optional_nonnegative_float(
            data.get("constraints", {}).get("max_total_energy_mj"),
            "constraints.max_total_energy_mj",
        ),
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


def _validate_non_empty_name(value: str, field_name: str) -> None:
    if not value.strip():
        raise ValueError(f"{field_name} must not be empty.")


def _register_transfer_link(
    transfer_links: dict[tuple[str, str], TransferLink], link: TransferLink
) -> None:
    key = (link.source, link.target)
    if key in transfer_links:
        raise ValueError(f"Duplicate transfer link {link.source!r}->{link.target!r} in scenario.")
    transfer_links[key] = link


def _require_positive_float(value: Any, field_name: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise ValueError(f"{field_name} must be greater than zero.")
    return parsed


def _require_nonnegative_float(value: Any, field_name: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise ValueError(f"{field_name} must be greater than or equal to zero.")
    return parsed


def _optional_nonnegative_float(value: Any, field_name: str) -> float | None:
    if value is None:
        return None
    return _require_nonnegative_float(value, field_name)
