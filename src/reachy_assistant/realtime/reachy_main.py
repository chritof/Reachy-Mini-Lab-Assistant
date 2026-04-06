from __future__ import annotations

import argparse
import sys

from reachy_mini import ReachyMini

from reachy_assistant.realtime.config import RealtimeConfig
from reachy_assistant.robot.pipeline.realtime_pipeline import ReachyRealtimePipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the OpenAI Realtime pipeline on Reachy Mini.")
    parser.add_argument("--reachy-host", default="reachy-mini.local")
    parser.add_argument("--reachy-port", type=int, default=8000)
    parser.add_argument(
        "--reachy-connection-mode",
        choices=["auto", "localhost_only", "network"],
        default="auto",
    )
    parser.add_argument("--reachy-use-sim", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        config = RealtimeConfig.from_env()
        config.allow_interruptions = False
        config.suppress_input_while_speaking = True
        config.input_resume_delay_ms = 900
        print(
            f"[realtime] Starting Reachy assistant with model={config.model} "
            f"at {args.reachy_host}:{args.reachy_port} "
            f"(mode={args.reachy_connection_mode}, sim={args.reachy_use_sim}, "
            f"interruptions={config.allow_interruptions})"
        )
        mini = ReachyMini(
            host=args.reachy_host,
            port=args.reachy_port,
            connection_mode=args.reachy_connection_mode,
            use_sim=args.reachy_use_sim,
        )
        pipeline = ReachyRealtimePipeline(mini=mini, config=config).build()
        pipeline.run()
    except Exception as exc:
        print(f"[realtime] Fatal error: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
