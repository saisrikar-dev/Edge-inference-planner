# Edge Inference Planner

Hardware-aware optimizer for partitioning an edge AI pipeline across CPU, GPU, and NPU devices while respecting latency, energy, and memory budgets.

This repo is built to signal the kind of engineering depth that companies like NVIDIA and Apple care about:
- hardware-aware scheduling instead of CRUD-heavy app work
- performance tradeoff reasoning across latency, energy, and transfer overhead
- exact search for small design spaces and beam search for larger ones
- explainable results that make placement decisions easy to inspect

## Problem

Modern on-device AI systems rarely run on one accelerator end-to-end. Real products split work across CPU, GPU, and NPU blocks depending on:
- stage-level compute characteristics
- model residency pressure on each device
- transfer cost between accelerators
- end-to-end latency and battery constraints

`Edge Inference Planner` turns that into a reproducible optimization problem.

## What It Does

Given a pipeline scenario in JSON, the planner:
- models per-stage execution profiles on each device
- applies inter-device transfer penalties when stages move across accelerators
- tracks cumulative model memory on every device
- enforces latency and energy caps
- returns the best placements for latency, efficiency, or balanced goals

## Quickstart

### 1. Create a virtual environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 2. Install the project

```bash
pip install -e .[dev]
```

### 3. Run the sample workload

```bash
edge-inference-planner plan scenarios/mobile_vision_pipeline.json --goal balanced --top-k 3
```

### 4. Export reports

```bash
edge-inference-planner plan scenarios/mobile_vision_pipeline.json --format csv --output reports/mobile_vision.csv
edge-inference-planner plan scenarios/mobile_vision_pipeline.json --format html --output reports/mobile_vision.html
```

## Example Output

```text
Pipeline: mobile_vision_stack
Goal: balanced | Strategy: exact | Score: 1.85
Latency: 27.97 ms | Energy: 32.18 mJ | Switches: 2

Placement
stage                | device | exec ms | xfer ms | total ms | exec mJ | xfer mJ | memory MB
---------------------+--------+---------+---------+----------+---------+---------+----------
frame_decode         | gpu    | 3.10    | 0.00    | 3.10     | 5.40    | 0.00    | 420
resize_normalize     | gpu    | 1.90    | 0.00    | 1.90     | 3.40    | 0.00    | 350
backbone_embedding   | npu    | 7.40    | 1.08    | 8.48     | 8.10    | 0.72    | 1240
multimodal_fusion    | npu    | 4.70    | 0.00    | 4.70     | 4.90    | 0.00    | 700
detector_heads       | npu    | 6.20    | 0.00    | 6.20     | 5.10    | 0.00    | 540
temporal_smoother    | npu    | 1.40    | 0.00    | 1.40     | 1.20    | 0.00    | 96
renderer             | gpu    | 2.10    | 0.09    | 2.19     | 3.30    | 0.06    | 260
```

## Architecture

```text
Scenario JSON
    |
    v
PipelineSpec -> Device profiles + stage profiles + transfer graph + constraints
    |
    v
Optimizer
    |- Exact branch-and-bound search for small spaces
    |- Beam search fallback for larger spaces
    |
    v
PlanResult
    |- ranked placements
    |- memory utilization summary
    |- stage-by-stage transfer costs
    |- optimization rationale
```

Detailed design notes live in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Scenario Schema

Each scenario includes:
- `devices`: accelerator memory budgets
- `links`: transfer cost between accelerators
- `stages`: per-device latency, energy, and memory requirements
- `constraints`: optional end-to-end caps

See [scenarios/mobile_vision_pipeline.json](scenarios/mobile_vision_pipeline.json) for a complete example.

## Project Structure

```text
Edge Inference Planner/
|-- docs/
|-- scenarios/
|-- src/edge_inference_planner/
|   |-- cli.py
|   |-- models.py
|   |-- optimizer.py
|   |-- report.py
|   `-- scenario.py
|-- tests/
|-- pyproject.toml
`-- README.md
```

## Why This Repo Is Stronger Than a Generic Portfolio App

- It demonstrates optimization and systems reasoning instead of only framework familiarity.
- It produces inspectable outputs with tradeoffs that are easy to discuss in interviews.
- It maps cleanly to edge AI, silicon, graphics, and applied ML platform teams.

## Roadmap

- Add DAG support for non-linear pipelines
- Add thermal throttling models and quantization knobs
- Plug in measured hardware benchmarks instead of hand-authored scenario profiles

## License

MIT License. See [LICENSE](LICENSE).
