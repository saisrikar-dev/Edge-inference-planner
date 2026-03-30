from __future__ import annotations

import json
from pathlib import Path

from edge_inference_planner.cli import main
from edge_inference_planner.optimizer import EdgeInferenceOptimizer
from edge_inference_planner.scenario import load_pipeline, pipeline_from_dict


def test_sample_scenario_has_feasible_balanced_plan() -> None:
    scenario_path = Path(__file__).resolve().parents[1] / "scenarios" / "mobile_vision_pipeline.json"
    pipeline = load_pipeline(scenario_path)

    results = EdgeInferenceOptimizer().optimize(pipeline, goal="balanced", top_k=3)

    assert results
    best = results[0]
    assert best.total_latency_ms <= pipeline.constraints.max_total_latency_ms
    assert best.total_energy_mj <= pipeline.constraints.max_total_energy_mj
    assert best.assignments[0].device_name == "gpu"
    assert best.assignments[-1].device_name == "gpu"
    assert any(placement.device_name == "npu" for placement in best.assignments)


def test_goal_preset_changes_preferred_accelerator() -> None:
    pipeline = pipeline_from_dict(
        {
            "name": "goal_sensitivity",
            "devices": [
                {"name": "cpu", "memory_mb": 4096},
                {"name": "gpu", "memory_mb": 4096},
                {"name": "npu", "memory_mb": 4096},
            ],
            "links": [
                {"source": "cpu", "target": "gpu", "latency_ms_per_mb": 0.02, "energy_mj_per_mb": 0.02},
                {"source": "cpu", "target": "npu", "latency_ms_per_mb": 0.02, "energy_mj_per_mb": 0.02},
                {"source": "gpu", "target": "npu", "latency_ms_per_mb": 0.02, "energy_mj_per_mb": 0.02},
            ],
            "stages": [
                {
                    "name": "prep",
                    "output_mb": 4,
                    "profiles": {
                        "cpu": {"latency_ms": 2.0, "energy_mj": 1.1, "memory_mb": 120},
                        "gpu": {"latency_ms": 1.2, "energy_mj": 1.5, "memory_mb": 180},
                        "npu": {"latency_ms": 1.3, "energy_mj": 0.9, "memory_mb": 100},
                    },
                },
                {
                    "name": "backbone",
                    "output_mb": 2,
                    "profiles": {
                        "cpu": {"latency_ms": 10.0, "energy_mj": 9.5, "memory_mb": 500},
                        "gpu": {"latency_ms": 4.0, "energy_mj": 7.2, "memory_mb": 900},
                        "npu": {"latency_ms": 5.2, "energy_mj": 3.5, "memory_mb": 640},
                    },
                },
                {
                    "name": "head",
                    "output_mb": 0,
                    "profiles": {
                        "cpu": {"latency_ms": 3.0, "energy_mj": 2.3, "memory_mb": 160},
                        "gpu": {"latency_ms": 1.5, "energy_mj": 2.0, "memory_mb": 240},
                        "npu": {"latency_ms": 1.7, "energy_mj": 1.0, "memory_mb": 140},
                    },
                },
            ],
        }
    )

    optimizer = EdgeInferenceOptimizer()
    latency_plan = optimizer.optimize(pipeline, goal="latency", top_k=1)[0]
    efficiency_plan = optimizer.optimize(pipeline, goal="efficiency", top_k=1)[0]

    assert latency_plan.assignments[1].device_name == "gpu"
    assert efficiency_plan.assignments[1].device_name == "npu"
    assert latency_plan.total_latency_ms < efficiency_plan.total_latency_ms
    assert latency_plan.total_energy_mj > efficiency_plan.total_energy_mj


def test_memory_constraints_force_work_off_small_npu() -> None:
    pipeline = pipeline_from_dict(
        {
            "name": "memory_budget",
            "devices": [
                {"name": "cpu", "memory_mb": 4096},
                {"name": "npu", "memory_mb": 700},
            ],
            "links": [
                {"source": "cpu", "target": "npu", "latency_ms_per_mb": 0.05, "energy_mj_per_mb": 0.03},
            ],
            "stages": [
                {
                    "name": "encoder",
                    "output_mb": 6,
                    "profiles": {
                        "cpu": {"latency_ms": 7.0, "energy_mj": 5.0, "memory_mb": 250},
                        "npu": {"latency_ms": 3.2, "energy_mj": 2.1, "memory_mb": 400},
                    },
                },
                {
                    "name": "decoder",
                    "output_mb": 0,
                    "profiles": {
                        "cpu": {"latency_ms": 4.0, "energy_mj": 3.0, "memory_mb": 220},
                        "npu": {"latency_ms": 2.8, "energy_mj": 1.8, "memory_mb": 350},
                    },
                },
            ],
        }
    )

    result = EdgeInferenceOptimizer().optimize(pipeline, goal="balanced", top_k=1)[0]

    assert result.memory_by_device["npu"] <= 700
    assert [placement.device_name for placement in result.assignments] != ["npu", "npu"]


def test_cli_json_output(tmp_path, capsys) -> None:
    scenario_path = tmp_path / "scenario.json"
    scenario_path.write_text(
        json.dumps(
            {
                "name": "cli_smoke",
                "devices": [
                    {"name": "cpu", "memory_mb": 1024},
                    {"name": "gpu", "memory_mb": 1024},
                ],
                "links": [
                    {
                        "source": "cpu",
                        "target": "gpu",
                        "latency_ms_per_mb": 0.01,
                        "energy_mj_per_mb": 0.01,
                    }
                ],
                "stages": [
                    {
                        "name": "stage_a",
                        "output_mb": 1,
                        "profiles": {
                            "cpu": {"latency_ms": 2.0, "energy_mj": 1.5, "memory_mb": 50},
                            "gpu": {"latency_ms": 1.0, "energy_mj": 1.8, "memory_mb": 70},
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(["plan", str(scenario_path), "--format", "json", "--top-k", "1"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert '"pipeline_name": "cli_smoke"' in captured.out


def test_cli_csv_output_to_file(tmp_path, capsys) -> None:
    scenario_path = tmp_path / "scenario.json"
    output_path = tmp_path / "exports" / "plan.csv"
    scenario_path.write_text(
        json.dumps(
            {
                "name": "csv_export",
                "devices": [
                    {"name": "cpu", "memory_mb": 1024},
                    {"name": "gpu", "memory_mb": 1024},
                ],
                "links": [
                    {
                        "source": "cpu",
                        "target": "gpu",
                        "latency_ms_per_mb": 0.01,
                        "energy_mj_per_mb": 0.01,
                    }
                ],
                "stages": [
                    {
                        "name": "stage_a",
                        "output_mb": 1,
                        "profiles": {
                            "cpu": {"latency_ms": 2.0, "energy_mj": 1.5, "memory_mb": 50},
                            "gpu": {"latency_ms": 1.0, "energy_mj": 1.8, "memory_mb": 70},
                        },
                    },
                    {
                        "name": "stage_b",
                        "output_mb": 0,
                        "profiles": {
                            "cpu": {"latency_ms": 1.5, "energy_mj": 1.1, "memory_mb": 20},
                            "gpu": {"latency_ms": 0.8, "energy_mj": 1.0, "memory_mb": 30},
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "plan",
            str(scenario_path),
            "--format",
            "csv",
            "--output",
            str(output_path),
            "--top-k",
            "1",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert output_path.exists()
    content = output_path.read_text(encoding="utf-8")
    assert "rank,pipeline_name,goal,strategy,total_score" in content
    assert "csv_export" in content
    assert "stage_a" in content
    assert "\n\n" not in content
    assert f"Wrote csv report to {output_path}" in captured.out


def test_cli_html_output_to_file(tmp_path, capsys) -> None:
    scenario_path = tmp_path / "scenario.json"
    output_path = tmp_path / "plan.html"
    scenario_path.write_text(
        json.dumps(
            {
                "name": "html_export",
                "devices": [
                    {"name": "cpu", "memory_mb": 1024},
                    {"name": "gpu", "memory_mb": 1024},
                ],
                "links": [
                    {
                        "source": "cpu",
                        "target": "gpu",
                        "latency_ms_per_mb": 0.01,
                        "energy_mj_per_mb": 0.01,
                    }
                ],
                "stages": [
                    {
                        "name": "stage_a",
                        "output_mb": 0,
                        "profiles": {
                            "cpu": {"latency_ms": 2.0, "energy_mj": 1.5, "memory_mb": 50},
                            "gpu": {"latency_ms": 1.0, "energy_mj": 1.8, "memory_mb": 70},
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "plan",
            str(scenario_path),
            "--format",
            "html",
            "--output",
            str(output_path),
            "--top-k",
            "1",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert output_path.exists()
    content = output_path.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in content
    assert "Edge Inference Planner Report - html_export" in content
    assert "<table>" in content
    assert f"Wrote html report to {output_path}" in captured.out
