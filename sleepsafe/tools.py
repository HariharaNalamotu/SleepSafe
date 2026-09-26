"""Read-only tools. No tool invents missing observations."""
from datetime import timezone

from .repository import SampleRepository
from .schema import parse_utc, validate_report

_repo = SampleRepository()


def set_repository(repository):
    global _repo
    _repo = repository


def get_sleep_report(session_id):
    report = _repo.get_session(session_id)
    if report is None:
        raise KeyError(f"Unknown session_id: {session_id}")
    return validate_report(report)


def get_events(session_id, event_type=None):
    events = get_sleep_report(session_id).get("events")
    if events is None:
        raise ValueError("Report has no events field")
    return [e for e in events if event_type is None or e["type"] == event_type]


def get_event_near_time(session_id, utc_time):
    report = get_sleep_report(session_id)
    target = parse_utc(utc_time).astimezone(timezone.utc)
    events = report.get("events")
    if not events:
        return None
    def distance(event):
        start = parse_utc(event["start_utc"]).astimezone(timezone.utc)
        end = parse_utc(event.get("end_utc", event["start_utc"])).astimezone(timezone.utc)
        return max((start-target).total_seconds(), (target-end).total_seconds(), 0)
    valid = [e for e in events if e.get("start_utc")]
    if not valid:
        return None
    # If a long snore episode overlaps a short apnea, prefer the specific event.
    event = min(valid, key=lambda e: (distance(e), e.get("duration_s", float("inf"))))
    return {"event": event, "distance_s": distance(event)}


def get_audio_quality(session_id):
    report = get_sleep_report(session_id)
    r = report["recording"]
    valid_fraction = r["valid_audio_s"] / r["duration_s"]
    return {"duration_s": r["duration_s"], "valid_audio_s": r["valid_audio_s"],
            "valid_fraction": round(valid_fraction, 4), "excluded_s": r.get("excluded_s", {}),
            "caveats": report.get("caveats", []),
            "quality_periods": [e for e in report.get("events", []) or []
                                if e["type"] in ("low_signal", "dropout")]}
