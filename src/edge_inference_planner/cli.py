from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .optimizer import EdgeInferenceOptimizer, OptimizerConfig
from .report import render_results_text, results_to_csv, results_to_html, results_to_json
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
        choices=["text", "json", "csv", "html"],
        default="text",
        help="Output format.",
    )
    plan_parser.add_argument(
        "--output",
        help="Optional output file path. Prints to stdout when omitted.",
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
            rendered = results_to_json(results)
        elif args.format == "csv":
            rendered = results_to_csv(results)
        elif args.format == "html":
            rendered = results_to_html(pipeline, results)
        else:
            rendered = render_results_text(pipeline, results)

        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(rendered, encoding="utf-8")
            print(f"Wrote {args.format} report to {output_path}")
        else:
            print(rendered)
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
