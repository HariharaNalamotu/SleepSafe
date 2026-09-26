"""Run the full pipeline on a recording (one file, or a directory of chunk files).

    python scripts/run_session.py recording.wav --out night.json [--demo-mode]
    python scripts/run_session.py chunks_dir/ --out night.json
"""
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sleepsafe.pipeline import SleepSafePipeline, assemble_chunks, load_audio  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--out", default="night.json")
    ap.add_argument("--model", default=str(ROOT / "runs" / "v2" / "model.pt"))
    ap.add_argument("--encoder", default=str(ROOT / "data" / "pretrained" / "Cnn14_16k.pth"))
    ap.add_argument("--session-id", default=None)
    ap.add_argument("--start-utc", default=None, help="ISO time of the first sample (default: now)")
    ap.add_argument("--demo-mode", action="store_true", help="count events regardless of sleep/wake")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    src = Path(args.input)
    if src.is_dir():
        files = [p for p in src.iterdir() if p.suffix.lower() in (".flac", ".wav", ".ogg")]
        audio, valid = assemble_chunks(files)
    else:
        audio, valid = load_audio(src), None
    start = datetime.fromisoformat(args.start_utc) if args.start_utc else datetime.now(timezone.utc)

    t0 = time.time()
    pipe = SleepSafePipeline(args.encoder, args.model, args.device)
    night = pipe.run(audio, args.session_id or src.stem, start, valid, demo_mode=args.demo_mode)
    Path(args.out).write_text(json.dumps(night, indent=2))
    s = night["summary"]
    print(f"{len(audio) / 16000 / 60:.1f} min processed in {time.time() - t0:.1f}s on {pipe.device}")
    print(f"events/hour {s['respiratory_events_per_hour']} ({s['severity_estimate']}), counts {s['counts']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
