from __future__ import annotations

import json
from typing import Sequence

from .models import PlanResult, PipelineSpec


def render_results_text(pipeline: PipelineSpec, results: Sequence[PlanResult]) -> str:
    if not results:
        return "No feasible placement found."

    best = results[0]
    lines: list[str] = [
        f"Pipeline: {pipeline.name}",
        f"Goal: {best.goal} | Strategy: {best.strategy} | Score: {best.total_score:.2f}",
        (
            f"Latency: {best.total_latency_ms:.2f} ms | Energy: {best.total_energy_mj:.2f} mJ "
            f"| Switches: {best.switches}"
        ),
        "Memory: " + _memory_summary(pipeline, best),
        "",
        "Placement",
        _render_stage_table(best),
    ]

    if best.explanation:
        lines.append("")
        lines.append("Rationale")
        lines.extend(f"- {line}" for line in best.explanation)

    if len(results) > 1:
        lines.append("")
        lines.append("Alternatives")
        lines.append(_render_alternative_table(results[1:]))

    return "\n".join(lines)


def results_to_json(results: Sequence[PlanResult]) -> str:
    return json.dumps([result.to_dict() for result in results], indent=2)


def _memory_summary(pipeline: PipelineSpec, result: PlanResult) -> str:
    parts = []
    for device_name, device in pipeline.devices.items():
        used = result.memory_by_device[device_name]
        parts.append(f"{device_name} {used:.0f}/{device.memory_mb:.0f} MB")
    return ", ".join(parts)


def _render_stage_table(result: PlanResult) -> str:
    headers = [
        "stage",
        "device",
        "exec ms",
        "xfer ms",
        "total ms",
        "exec mJ",
        "xfer mJ",
        "memory MB",
    ]
    rows = [
        [
            placement.stage_name,
            placement.device_name,
            f"{placement.execution_latency_ms:.2f}",
            f"{placement.transfer_latency_ms:.2f}",
            f"{placement.total_latency_ms:.2f}",
            f"{placement.execution_energy_mj:.2f}",
            f"{placement.transfer_energy_mj:.2f}",
            f"{placement.memory_mb:.0f}",
        ]
        for placement in result.assignments
    ]
    return _render_table(headers, rows)


def _render_alternative_table(results: Sequence[PlanResult]) -> str:
    headers = ["rank", "score", "latency ms", "energy mJ", "switches", "device path"]
    rows = []
    for rank, result in enumerate(results, start=2):
        rows.append(
            [
                str(rank),
                f"{result.total_score:.2f}",
                f"{result.total_latency_ms:.2f}",
                f"{result.total_energy_mj:.2f}",
                str(result.switches),
                " -> ".join(placement.device_name for placement in result.assignments),
            ]
        )
    return _render_table(headers, rows)


def _render_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def render_row(row: list[str]) -> str:
        return " | ".join(cell.ljust(widths[index]) for index, cell in enumerate(row))

    separator = "-+-".join("-" * width for width in widths)
    rendered = [render_row(headers), separator]
    rendered.extend(render_row(row) for row in rows)
    return "\n".join(rendered)
