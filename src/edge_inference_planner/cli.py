from __future__ import annotations

import argparse
import sys

from .optimizer import EdgeInferenceOptimizer, OptimizerConfig
from .report import render_results_text, results_to_json
from .scenario import load_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="edge-inference-planner",
        description="Optimize edge inference stage placement across CPU, GPU, and NPU devices.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser("plan", help="Optimize a scenario file.")
    plan_parser.add_argument("scenario", help="Path to the scenario JSON file.")
    plan_parser.add_argument(
        "--goal",
        default="balanced",
        choices=["latency", "efficiency", "balanced"],
        help="Optimization goal preset.",
    )
    plan_parser.add_argument("--top-k", type=int, default=3, help="Number of ranked plans to emit.")
    plan_parser.add_argument(
        "--beam-width",
        type=int,
        default=128,
        help="Beam width for large search spaces.",
    )
    plan_parser.add_argument(
        "--exact-search-limit",
        type=int,
        default=100000,
        help="Maximum placement combinations before switching from exact to beam search.",
    )
    plan_parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "plan":
        optimizer = EdgeInferenceOptimizer(
            OptimizerConfig(
                beam_width=args.beam_width,
                exact_search_limit=args.exact_search_limit,
            )
        )
        pipeline = load_pipeline(args.scenario)
        results = optimizer.optimize(pipeline, goal=args.goal, top_k=args.top_k)
        if not results:
            print("No feasible placement found under the current constraints.", file=sys.stderr)
            return 1

        if args.format == "json":
            print(results_to_json(results))
        else:
            print(render_results_text(pipeline, results))
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
