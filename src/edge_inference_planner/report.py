from __future__ import annotations

import csv
import html
import json
from io import StringIO
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


def results_to_csv(results: Sequence[PlanResult]) -> str:
    buffer = StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(
        [
            "rank",
            "pipeline_name",
            "goal",
            "strategy",
            "total_score",
            "total_latency_ms",
            "total_energy_mj",
            "switches",
            "stage_name",
            "device_name",
            "execution_latency_ms",
            "transfer_latency_ms",
            "total_latency_ms",
            "execution_energy_mj",
            "transfer_energy_mj",
            "total_energy_mj",
            "memory_mb",
        ]
    )

    for rank, result in enumerate(results, start=1):
        for placement in result.assignments:
            writer.writerow(
                [
                    rank,
                    result.pipeline_name,
                    result.goal,
                    result.strategy,
                    f"{result.total_score:.4f}",
                    f"{result.total_latency_ms:.4f}",
                    f"{result.total_energy_mj:.4f}",
                    result.switches,
                    placement.stage_name,
                    placement.device_name,
                    f"{placement.execution_latency_ms:.4f}",
                    f"{placement.transfer_latency_ms:.4f}",
                    f"{placement.total_latency_ms:.4f}",
                    f"{placement.execution_energy_mj:.4f}",
                    f"{placement.transfer_energy_mj:.4f}",
                    f"{placement.total_energy_mj:.4f}",
                    f"{placement.memory_mb:.4f}",
                ]
            )
    return buffer.getvalue()


def results_to_html(pipeline: PipelineSpec, results: Sequence[PlanResult]) -> str:
    title = f"Edge Inference Planner Report - {pipeline.name}"
    if not results:
        body = "<p>No feasible placement found.</p>"
    else:
        best = results[0]
        summary_items = [
            ("Goal", best.goal),
            ("Strategy", best.strategy),
            ("Score", f"{best.total_score:.2f}"),
            ("Latency", f"{best.total_latency_ms:.2f} ms"),
            ("Energy", f"{best.total_energy_mj:.2f} mJ"),
            ("Switches", str(best.switches)),
            ("Memory", _memory_summary(pipeline, best)),
        ]
        summary_markup = "".join(
            f"<li><strong>{html.escape(label)}:</strong> {html.escape(value)}</li>"
            for label, value in summary_items
        )

        body_parts = [
            f"<h1>{html.escape(title)}</h1>",
            "<section>",
            "<h2>Best Plan</h2>",
            f"<ul>{summary_markup}</ul>",
            _html_table(
                "Placement",
                [
                    "Stage",
                    "Device",
                    "Exec ms",
                    "Xfer ms",
                    "Total ms",
                    "Exec mJ",
                    "Xfer mJ",
                    "Memory MB",
                ],
                [
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
                    for placement in best.assignments
                ],
            ),
            "</section>",
        ]

        if best.explanation:
            body_parts.extend(
                [
                    "<section>",
                    "<h2>Rationale</h2>",
                    "<ul>",
                    *[f"<li>{html.escape(line)}</li>" for line in best.explanation],
                    "</ul>",
                    "</section>",
                ]
            )

        if len(results) > 1:
            body_parts.extend(
                [
                    "<section>",
                    _html_table(
                        "Alternatives",
                        ["Rank", "Score", "Latency ms", "Energy mJ", "Switches", "Device Path"],
                        [
                            [
                                str(rank),
                                f"{result.total_score:.2f}",
                                f"{result.total_latency_ms:.2f}",
                                f"{result.total_energy_mj:.2f}",
                                str(result.switches),
                                " -> ".join(placement.device_name for placement in result.assignments),
                            ]
                            for rank, result in enumerate(results[1:], start=2)
                        ],
                    ),
                    "</section>",
                ]
            )

        body = "\n".join(body_parts)

    return "\n".join(
        [
            "<!DOCTYPE html>",
            "<html lang=\"en\">",
            "<head>",
            "  <meta charset=\"utf-8\">",
            f"  <title>{html.escape(title)}</title>",
            "  <style>",
            "    body { font-family: Arial, sans-serif; margin: 2rem; color: #1f2933; }",
            "    h1, h2 { color: #102a43; }",
            "    table { border-collapse: collapse; width: 100%; margin: 1rem 0 2rem; }",
            "    th, td { border: 1px solid #d9e2ec; padding: 0.5rem; text-align: left; }",
            "    th { background: #f0f4f8; }",
            "    ul { padding-left: 1.25rem; }",
            "    section { margin-bottom: 2rem; }",
            "  </style>",
            "</head>",
            "<body>",
            body,
            "</body>",
            "</html>",
        ]
    )


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


def _html_table(title: str, headers: list[str], rows: list[list[str]]) -> str:
    header_markup = "".join(f"<th>{html.escape(header)}</th>" for header in headers)
    row_markup = []
    for row in rows:
        cells = "".join(f"<td>{html.escape(cell)}</td>" for cell in row)
        row_markup.append(f"<tr>{cells}</tr>")

    return "\n".join(
        [
            f"<h2>{html.escape(title)}</h2>",
            "<table>",
            f"<thead><tr>{header_markup}</tr></thead>",
            f"<tbody>{''.join(row_markup)}</tbody>",
            "</table>",
        ]
    )
