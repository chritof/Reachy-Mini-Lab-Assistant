from __future__ import annotations

import argparse
import sys

from reachy_assistant.realtime.config import RealtimeConfig
from reachy_assistant.realtime.engine import RealtimeConversationEngine
from reachy_assistant.realtime.pc_audio import PcRealtimeAudioSink, PcRealtimeAudioSource
from reachy_assistant.realtime.pipeline import RealtimeConversationPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the OpenAI Realtime pipeline on a PC.")
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--chunk-ms", type=int, default=100)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        config = RealtimeConfig.from_env()
        config.input_sample_rate = args.sample_rate
        config.output_sample_rate = args.sample_rate
        config.chunk_ms = args.chunk_ms
        print(
            f"[realtime] Starting assistant with model={config.model}, "
            f"sample_rate={config.input_sample_rate}, chunk_ms={config.chunk_ms}"
        )

        source = PcRealtimeAudioSource(
            sample_rate=config.input_sample_rate,
            chunk_frames=config.chunk_frames,
        )
        sink = PcRealtimeAudioSink()
        engine = RealtimeConversationEngine(config=config, source=source, sink=sink)
        pipeline = RealtimeConversationPipeline(engine=engine)
        pipeline.run()
    except Exception as exc:
        print(f"[realtime] Fatal error: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
