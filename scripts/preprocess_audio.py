#!/usr/bin/env python3
"""Standardize a WAV and save mono 16 kHz windows plus a JSON manifest."""

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.audio import (
    compute_quality_metrics,
    inspect_original_audio,
    split_into_windows,
    standardize_audio,
)
from preprocessing.manifest import build_window_manifest, write_manifest_to_delta


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input WAV path")
    parser.add_argument("--session-id", required=True, help="Stable session identifier")
    parser.add_argument("--output-dir", required=True, help="Directory for prepared window WAVs")
    parser.add_argument("--window-seconds", type=float, default=1200)
    parser.add_argument("--target-sr", type=int, default=16000)
    parser.add_argument("--write-delta", action="store_true", help="Append manifest records to Delta")
    parser.add_argument("--delta-table", default="workspace.default.sleepsafe_audio_windows")
    args = parser.parse_args(argv)

    original = inspect_original_audio(args.input)
    audio = standardize_audio(args.input, target_sr=args.target_sr)
    overall_quality = compute_quality_metrics(audio, args.target_sr)
    windows = split_into_windows(audio, args.target_sr, window_seconds=args.window_seconds)
    output_dir = Path(args.output_dir)
    records = build_window_manifest(args.session_id, windows, args.target_sr, output_dir)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps({
        "session_id": args.session_id,
        "source": {"audio_path": str(Path(args.input)), **original},
        "standardized": {
            "sample_rate": args.target_sr,
            "channels": 1,
            **overall_quality,
        },
        "window_seconds": args.window_seconds,
        "windows": records,
    }, indent=2) + "\n")

    print(f"Session: {args.session_id}")
    print(f"Original audio: {original['sample_rate']} Hz, {original['channels']} channel(s), {original['duration_s']:.2f} s")
    print(f"Prepared audio: {args.target_sr} Hz mono, {overall_quality['duration_s']:.2f} s, {len(records)} window(s)")
    print(f"Manifest: {manifest_path}")
    for record in records:
        print(f"  window_{record['window_id']:03d}: {record['start_offset_s']:.2f}-{record['end_offset_s']:.2f} s, "
              f"{record['status']}, {record['audio_path']}")
    if args.write_delta:
        written = write_manifest_to_delta(records, table_name=args.delta_table)
        print(f"Delta: appended {written} record(s) to {args.delta_table}")
    return records


if __name__ == "__main__":
    main()
