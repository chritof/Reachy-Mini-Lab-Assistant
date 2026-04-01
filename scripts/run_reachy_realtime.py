from __future__ import annotations

import argparse

from dotenv import load_dotenv
from reachy_mini import ReachyMini

from reachy_assistant.realtime.config import RealtimeConfig
from reachy_assistant.robot.pipeline.realtime_pipeline import ReachyRealtimePipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the OpenAI Realtime pipeline on Reachy Mini.")
    parser.add_argument(
        "--connection-mode",
        default="auto",
        choices=["auto", "localhost_only"],
        help="Reachy Mini SDK connection mode.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Optional Reachy daemon port. Useful with localhost_only.",
    )
    parser.add_argument(
        "--media-backend",
        default="default",
        help="Reachy media backend to use.",
    )
    return parser


def main() -> None:
    load_dotenv()
    args = build_parser().parse_args()
    config = RealtimeConfig.from_env()

    reachy_kwargs: dict[str, object] = {
        "connection_mode": args.connection_mode,
        "media_backend": args.media_backend,
    }
    if args.port is not None:
        reachy_kwargs["port"] = args.port

    with ReachyMini(**reachy_kwargs) as mini:
        pipeline = ReachyRealtimePipeline(mini=mini, config=config).build()
        pipeline.run()


if __name__ == "__main__":
    main()
