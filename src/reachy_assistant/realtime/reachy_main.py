"""
CLI-entrypoint for realtime-assistenten på Reachy Mini, inkludert wakeword-oppsett.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
    parser.add_argument("--wakeword-model", default="")
    parser.add_argument("--sleepword-model", default="")
    parser.add_argument("--wakeword-timeout-sec", type=float, default=20.0)
    return parser


def _find_default_model(pattern: str) -> str | None:
    models_dir = Path(__file__).resolve().parents[3] / "models"
    matches = sorted(models_dir.glob(pattern))
    if not matches:
        return None
    return str(matches[-1])


def _find_default_wakeword_model() -> str | None:
    return _find_default_model("hei_reachee*.onnx")


def _find_default_sleepword_model() -> str | None:
    return _find_default_model("Stop*.onnx")


def _apply_reachy_lab_audio_defaults(config: RealtimeConfig) -> None:
    config.vad_threshold = 0.72
    config.vad_silence_ms = 950
    config.vad_prefix_padding_ms = 180


def main() -> None:
    args = build_parser().parse_args()
    try:
        config = RealtimeConfig.from_env()
        _apply_reachy_lab_audio_defaults(config)
        config.allow_interruptions = False
        config.suppress_input_while_speaking = True
        config.input_resume_delay_ms = 900
        config.wake_phrase_enabled = False
        config.wake_phrase_timeout_sec = args.wakeword_timeout_sec
        wakeword_model = args.wakeword_model.strip() or _find_default_wakeword_model()
        sleepword_model = args.sleepword_model.strip() or _find_default_sleepword_model()
        if not wakeword_model:
            raise FileNotFoundError(
                "No openWakeWord model found. Put a model like 'hei_reachee*.onnx' in the project's models/ folder "
                "or pass --wakeword-model <path>."
            )
        if not sleepword_model:
            print(
                "[realtime] No sleep word model found. Put a model like 'Stop*.onnx' in the project's models/ folder "
                "or pass --sleepword-model <path> to enable voice sleep."
            )
        print(
            f"[realtime] Starting Reachy assistant with model={config.model} "
            f"at {args.reachy_host}:{args.reachy_port} "
            f"(mode={args.reachy_connection_mode}, sim={args.reachy_use_sim}, "
            f"interruptions={config.allow_interruptions}, sleepword={bool(sleepword_model)}, "
            f"vad_threshold={config.vad_threshold}, vad_silence_ms={config.vad_silence_ms})"
        )
        mini = ReachyMini(
            host=args.reachy_host,
            port=args.reachy_port,
            connection_mode=args.reachy_connection_mode,
            use_sim=args.reachy_use_sim,
        )
        pipeline = ReachyRealtimePipeline(
            mini=mini,
            config=config,
            wakeword_model_path=wakeword_model,
            sleepword_model_path=sleepword_model,
        ).build()
        pipeline.run()
    except Exception as exc:
        print(f"[realtime] Fatal error: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
