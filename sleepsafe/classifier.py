"""Deterministic classification. Thresholds are hackathon heuristics, not clinical rules."""
import logging

from .schema import number, validate_output
from .tools import get_sleep_report, get_audio_quality

log = logging.getLogger(__name__)


def _base(classification, severity="unknown"):
    return {"classification": classification, "severity": severity,
            "primary_findings": [], "notable_events": [], "data_quality": "",
            "summary": "", "caveats": []}


def classify_sleep_session(session_id):
    report = get_sleep_report(session_id)
    quality = get_audio_quality(session_id)
    s = report.get("summary")
    severity = str(s.get("severity_estimate", "unknown")) if isinstance(s, dict) else "unknown"
    result = _base("insufficient_evidence", severity)
    result["caveats"] = list(report.get("caveats", []))
    result["data_quality"] = (f"{quality['valid_audio_s']} of {quality['duration_s']} seconds valid "
                              f"({quality['valid_fraction']:.1%}); excluded: {quality['excluded_s']}.")
    if quality["valid_fraction"] < 0.70:
        result["classification"] = "poor_audio_quality"
        result["summary"] = "Recording quality is too limited for a reliable pattern interpretation."
        return validate_output(result)
    if not isinstance(s, dict) or not isinstance(s.get("counts"), dict):
        result["summary"] = "The report lacks a usable summary or event counts."
        return validate_output(result)
    counts = s["counts"]
    required = ("apnea", "hypopnea")
    if any(not number(counts.get(k)) or counts[k] < 0 for k in required):
        result["summary"] = "The report lacks valid apnea or hypopnea counts."
        return validate_output(result)
    ahi = s.get("estimated_ahi")
    snore = s.get("snore_pct_of_sleep")
    if not number(ahi) or ahi < 0 or not number(snore) or not 0 <= snore <= 100:
        result["summary"] = "The report lacks valid estimated AHI or snoring percentage."
        return validate_output(result)
    if report.get("events") is None:
        result["summary"] = "The event list is missing, so event evidence cannot be checked."
        return validate_output(result)
    respiratory = counts["apnea"] + counts["hypopnea"]
    ap_pattern = ahi >= 5 and respiratory > 0
    sn_pattern = snore >= 30
    if ap_pattern and sn_pattern:
        label = "mixed_respiratory_pattern"
    elif ap_pattern:
        label = "apnea_hypopnea_pattern"
    elif sn_pattern:
        label = "snoring_dominant_pattern"
    elif ahi < 5 and respiratory == 0:
        label = "no_significant_respiratory_pattern"
    else:
        label = "insufficient_evidence"
    result["classification"] = label
    result["primary_findings"] = [
        f"Model-reported estimated AHI: {ahi}; model-reported severity: {severity}.",
        f"Model summary counts: {counts['apnea']} apnea, {counts['hypopnea']} hypopnea.",
        f"Model-reported snoring: {snore}% of estimated sleep.",
    ]
    events = report["events"]
    trained = [e for e in events if e.get("confidence_basis") == "trained"]
    zero = [e for e in events if e.get("confidence_basis") == "zero_shot"]
    if zero:
        result["caveats"].append(f"{len(zero)} listed zero-shot detections have weaker, uncalibrated evidence than trained detections.")
    if not events:
        result["caveats"].append("No individual events are listed; only summary-level interpretation is possible.")
    if not report.get("events_truncated", False) and len([e for e in events if e["type"] == "apnea"]) != counts["apnea"]:
        result["caveats"].append("Listed apnea events do not match the summary count although events_truncated is false.")
    for e in sorted(trained, key=lambda x: x.get("duration_s", 0), reverse=True):
        if e["type"] in ("apnea", "hypopnea", "snore_episode"):
            result["notable_events"].append(f"{e.get('id', 'unidentified')}: {e['type']} at {e.get('start_utc', 'unknown time')}, duration {e.get('duration_s', 'unknown')} s, trained confidence {e.get('confidence', 'unreported')}.")
        if len(result["notable_events"]) >= 3:
            break
    result["summary"] = (f"The model findings support {label.replace('_', ' ')}. "
                         "This is an audio-based pattern interpretation, not a diagnosis.")
    log.info("classified session %s as %s", session_id, label)
    return validate_output(result)
