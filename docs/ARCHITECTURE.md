# Architecture

## Design Goal

The planner models a realistic edge inference placement problem:
- each pipeline stage can run on a subset of devices
- execution cost changes by device
- moving tensors between accelerators costs time and energy
- model memory accumulates on the device that hosts a stage

That produces a search problem over device assignments.

## Core Objects

### `PipelineSpec`

Holds the full workload:
- devices and memory budgets
- stage execution profiles
- transfer links between devices
- optional global latency and energy limits

### `StageSpec`

Each stage owns:
- output tensor size in MB
- a map of device-specific execution profiles

### `ExecutionProfile`

Per-stage, per-device metrics:
- latency in milliseconds
- energy in millijoules
- memory residency in megabytes

## Search Strategies

### Exact Search

For small design spaces the optimizer uses branch-and-bound search:
- recursively assign a device to each stage
- prune states that break latency, energy, or memory constraints
- compute a lower-bound score using the minimum remaining latency and energy
- keep only the best `top_k` complete placements

This is deterministic and returns optimal results for the explored objective.

### Beam Search

When the placement space becomes too large, the optimizer falls back to beam search:
- expand the current frontier stage by stage
- rank partial states by optimistic score
- retain the top `beam_width` states per layer

This trades exactness for scalability while preserving explainable intermediate logic.

## Objective Function

Three goal presets ship with the repo:
- `latency`
- `efficiency`
- `balanced`

The score uses normalized latency and energy totals plus a penalty for device switches:

```text
score =
  latency_weight * (total_latency / baseline_latency)
  + energy_weight * (total_energy / baseline_energy)
  + switch_weight * total_switches
```

The baseline terms are the stage-wise minimum latency and energy across all available devices. That keeps the score stable across scenarios with different raw magnitudes.

## Explainability

Every `PlanResult` includes:
- stage-level placement and transfer costs
- total latency and energy
- per-device memory usage
- short rationale lines that highlight bottlenecks and dominant transfer costs

## Extension Points

Natural follow-up work for this repo:
- support branching DAGs instead of linear pipelines
- add batch-size-aware profiles
- model quantization choices as additional stage variants
- feed the planner with measured benchmark traces
