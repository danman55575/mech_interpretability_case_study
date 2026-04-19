"""
Unified CLI entry point for the Mechanistic Interpretability pipeline.
"""

from __future__ import annotations

import argparse
import sys

from src.config import get_default_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Mechanistic Interpretability via Sparse Autoencoders",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    subparsers = parser.add_subparsers(dest="command", help="Pipeline stage to run")

    # collect
    subparsers.add_parser("collect", help="Cache LLM activations to disk")

    # train
    subparsers.add_parser("train", help="Train the Sparse Autoencoder")

    # evaluate
    subparsers.add_parser("evaluate", help="Evaluate trained SAE quality")

    # interpret
    subparsers.add_parser("interpret", help="Build feature dictionary from activations")

    # steer
    steer_parser = subparsers.add_parser("steer", help="Run activation steering experiments")
    steer_parser.add_argument(
        "--feature_id", type=int, default=None,
        help="SAE feature ID to steer (overrides config)"
    )
    steer_parser.add_argument(
        "--alpha", type=float, default=None,
        help="Steering coefficient (overrides config)"
    )
    steer_parser.add_argument(
        "--prompt", type=str, default=None,
        help="Generation prompt (overrides config)"
    )

    # pipeline
    subparsers.add_parser(
        "pipeline",
        help="Run full pipeline: collect -> train -> evaluate -> interpret"
    )

    # Global overrides
    parser.add_argument(
        "--override", action="append", default=[],
        metavar="SECTION.PARAM=VALUE",
        help="Override config parameter (e.g., --override train.learning_rate=5e-4)"
    )

    return parser


def parse_overrides(override_list: list[str]) -> dict[str, str]:
    """Parses CLI override strings into a dictionary."""
    overrides = {}
    for item in override_list:
        if "=" not in item:
            print(f"[ERROR] Invalid override format: '{item}'. Expected 'section.param=value'")
            sys.exit(1)
        key, value = item.split("=", 1)
        overrides[key] = value
    return overrides


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    # Build and configure
    cfg = get_default_config()

    # Apply global overrides
    if args.override:
        overrides = parse_overrides(args.override)
        cfg.override(overrides)

    # Apply steer-specific CLI args
    if args.command == "steer":
        if args.feature_id is not None:
            cfg.steer.target_feature_id = args.feature_id
        if args.alpha is not None:
            cfg.steer.steering_coeff = args.alpha
        if args.prompt is not None:
            cfg.steer.prompt = args.prompt

    # Dispatch to pipeline stage
    if args.command == "collect":
        from src.data_collection import run
        run(cfg)

    elif args.command == "train":
        from src.train import run
        run(cfg)

    elif args.command == "evaluate":
        from src.evaluate import run
        run(cfg)

    elif args.command == "interpret":
        from src.interpret import run
        run(cfg)

    elif args.command == "steer":
        from src.steer import run
        run(cfg)

    elif args.command == "pipeline":
        print("=" * 70)
        print("RUNNING FULL PIPELINE")
        print("=" * 70)

        print("\n[STAGE 1/4] Collecting activations...")
        from src.data_collection import run as collect_run
        collect_run(cfg)

        print("\n[STAGE 2/4] Training SAE...")
        from src.train import run as train_run
        train_run(cfg)

        print("\n[STAGE 3/4] Evaluating SAE...")
        from src.evaluate import run as eval_run
        eval_run(cfg)

        print("\n[STAGE 4/4] Building feature dictionary...")
        from src.interpret import run as interp_run
        interp_run(cfg)

        print("\n" + "=" * 70)
        print("PIPELINE COMPLETE")
        print("=" * 70)


if __name__ == "__main__":
    main()
