"""Strict output shape and defensive report validation, without external packages."""
from datetime import datetime
from math import isfinite

CLASSIFICATIONS = {
    "no_significant_respiratory_pattern", "apnea_hypopnea_pattern",
    "snoring_dominant_pattern", "mixed_respiratory_pattern",
    "poor_audio_quality", "insufficient_evidence",
}
OUTPUT_KEYS = ("classification", "severity", "primary_findings", "notable_events",
               "data_quality", "summary", "caveats")


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(value)


def parse_utc(value):
    if not isinstance(value, str):
        raise ValueError("UTC time must be an ISO 8601 string")
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError("UTC time must include a timezone")
    return dt


def validate_report(report):
    if not isinstance(report, dict):
        raise ValueError("report is not a JSON object")
    if not isinstance(report.get("session_id"), str) or not report["session_id"]:
        raise ValueError("missing session_id")
    recording = report.get("recording")
    if not isinstance(recording, dict):
        raise ValueError("missing recording")
    for key in ("duration_s", "valid_audio_s"):
        if not number(recording.get(key)) or recording[key] < 0:
            raise ValueError(f"missing or invalid recording.{key}")
    if recording["duration_s"] == 0 or recording["valid_audio_s"] > recording["duration_s"]:
        raise ValueError("invalid recording duration")
    summary = report.get("summary")
    if summary is not None and not isinstance(summary, dict):
        raise ValueError("malformed summary")
    events = report.get("events")
    if events is not None and not isinstance(events, list):
        raise ValueError("malformed events")
    if isinstance(events, list):
        for i, event in enumerate(events):
            if not isinstance(event, dict) or not isinstance(event.get("type"), str):
                raise ValueError(f"malformed event at index {i}")
            for key in ("start_offset_s", "end_offset_s"):
                if key in event and (not number(event[key]) or event[key] < 0):
                    raise ValueError(f"invalid event {i}.{key}")
            if "confidence" in event and (not number(event["confidence"]) or not 0 <= event["confidence"] <= 1):
                raise ValueError(f"invalid event {i}.confidence")
    return report


def validate_output(value):
    if not isinstance(value, dict) or set(value) != set(OUTPUT_KEYS):
        raise ValueError("output must have exactly the seven required keys")
    if value["classification"] not in CLASSIFICATIONS:
        raise ValueError("invalid classification")
    for key in ("severity", "data_quality", "summary"):
        if not isinstance(value[key], str):
            raise ValueError(f"{key} must be a string")
    for key in ("primary_findings", "notable_events", "caveats"):
        if not isinstance(value[key], list) or not all(isinstance(x, str) for x in value[key]):
            raise ValueError(f"{key} must be a list of strings")
    return value
