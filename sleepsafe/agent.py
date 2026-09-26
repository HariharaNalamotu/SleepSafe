"""Grounded Q&A. Deterministic answers first; optional Databricks model for other questions."""
import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .classifier import classify_sleep_session
from .tools import get_sleep_report, get_events, get_event_near_time, get_audio_quality

SYSTEM_PROMPT = (Path(__file__).parent / "system_prompt.txt").read_text()

TOOL_FUNCTIONS = {"get_sleep_report": get_sleep_report, "get_events": get_events,
                  "get_event_near_time": get_event_near_time,
                  "get_audio_quality": get_audio_quality,
                  "classify_sleep_session": classify_sleep_session}

TOOL_SCHEMAS = [{"type": "function", "function": {"name": name,
    "description": desc, "parameters": {"type": "object", "properties": properties,
    "required": required, "additionalProperties": False}}}
    for name, desc, properties, required in [
        ("get_sleep_report", "Read model summary and report metadata", {"session_id": {"type": "string"}}, ["session_id"]),
        ("get_events", "Read listed events, optionally filtered by type", {"session_id": {"type": "string"}, "event_type": {"type": "string"}}, ["session_id"]),
        ("get_event_near_time", "Find nearest listed event to UTC timestamp", {"session_id": {"type": "string"}, "utc_time": {"type": "string"}}, ["session_id", "utc_time"]),
        ("get_audio_quality", "Read valid audio, exclusions and caveats", {"session_id": {"type": "string"}}, ["session_id"]),
        ("classify_sleep_session", "Read deterministic classification", {"session_id": {"type": "string"}}, ["session_id"]),
    ]]


def _time_from_question(question, report):
    match = re.search(r"\b(1[0-2]|0?[1-9]):([0-5]\d)\s*(am|pm)\b", question, re.I)
    if not match:
        return None
    hour = int(match[1]) % 12 + (12 if match[3].lower() == "pm" else 0)
    minute = int(match[2])
    start = datetime.fromisoformat(report["recording"]["start_utc"].replace("Z", "+00:00"))
    end = datetime.fromisoformat(report["recording"]["end_utc"].replace("Z", "+00:00"))
    day = start.date()
    while day <= end.date():
        candidate = datetime(day.year, day.month, day.day, hour, minute, tzinfo=timezone.utc)
        if start <= candidate <= end:
            return candidate.isoformat().replace("+00:00", "Z")
        day += timedelta(days=1)
    return None


def _direct_answer(session_id, question):
    q = question.lower()
    report = get_sleep_report(session_id)
    if "previous" in q or "different from" in q:
        return "A previous-night report is not available, so I cannot compare nights yet."
    if "classified" in q or "classification" in q or "why" in q:
        c = classify_sleep_session(session_id)
        return f"{c['classification']}: {' '.join(c['primary_findings'])} {c['data_quality']} {' '.join(c['caveats'])}".strip()
    if "quality" in q or "low-quality" in q or "low quality" in q:
        a = get_audio_quality(session_id)
        periods = a["quality_periods"]
        return (f"{a['valid_audio_s']} of {a['duration_s']} seconds were valid. Excluded: {a['excluded_s']}. "
                f"Listed quality periods: {json.dumps(periods)}. Caveats: {a['caveats']}")
    if "longest apnea" in q:
        longest = report.get("summary", {}).get("longest_apnea_s")
        listed = [e for e in get_events(session_id, "apnea") if isinstance(e.get("duration_s"), (int, float))]
        if longest is None:
            return "The report does not state the longest apnea duration."
        detail = max(listed, key=lambda e: e["duration_s"]) if listed else None
        return (f"The model summary reports a longest apnea of {longest} seconds. "
                + (f"The longest listed apnea is {detail['duration_s']} seconds at {detail.get('start_utc')}; "
                   "the report does not provide details for the summary maximum." if detail and detail["duration_s"] != longest
                   else (f"Listed event: {detail.get('id')} at {detail.get('start_utc')}." if detail else "No matching event detail is listed.")))
    if "snor" in q and ("how much" in q or "percent" in q):
        pct = report.get("summary", {}).get("snore_pct_of_sleep")
        return f"The model reports snoring during {pct}% of estimated sleep." if pct is not None else "Snoring percentage is unavailable."
    if "zero-shot" in q or "zero shot" in q:
        found = [e for e in get_events(session_id) if e.get("confidence_basis") == "zero_shot"]
        return "Listed zero-shot detections: " + (", ".join(f"{e.get('id')} ({e['type']}, {e.get('start_utc')})" for e in found) if found else "none") + ". These thresholds are uncalibrated."
    when = _time_from_question(question, report)
    if when:
        near = get_event_near_time(session_id, when)
        if near is None:
            return "No individual events are listed for this session."
        e = near["event"]
        return f"Nearest listed event to {when}: {e['type']} {e.get('id')} from {e.get('start_utc')} to {e.get('end_utc')}; distance {near['distance_s']} seconds."
    if "most concerning" in q:
        hourly = report.get("summary", {}).get("events_per_hour_by_hour", [])
        if not hourly:
            return "The report does not identify a most active period."
        i = max(range(len(hourly)), key=lambda j: hourly[j])
        return f"The highest model-reported hourly event rate is {hourly[i]} in hour {i+1} of the hourly summary. This is a screening signal, not a diagnosis."
    if "what events" in q or "events detected" in q:
        s = report.get("summary", {}).get("counts", {})
        return f"Model summary counts: {s}. The supplied event list has {len(get_events(session_id))} entries; it may not show each counted event."
    return None


def _llm_answer(session_id, question):
    from databricks.sdk import WorkspaceClient
    endpoint = os.environ["SLEEPSAFE_LLM_ENDPOINT"]
    client = WorkspaceClient().serving_endpoints.get_open_ai_client()
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Session ID: {session_id}\nQuestion: {question}"}]
    for _ in range(4):
        response = client.chat.completions.create(model=endpoint, messages=messages,
                                                  tools=TOOL_SCHEMAS, tool_choice="auto")
        msg = response.choices[0].message
        calls = msg.tool_calls or []
        if not calls:
            return msg.content or "No answer returned."
        messages.append(msg.model_dump(exclude_none=True))
        for call in calls:
            name = call.function.name
            try:
                args = json.loads(call.function.arguments)
                args["session_id"] = session_id  # never let model cross sessions
                value = TOOL_FUNCTIONS[name](**args)
            except (ValueError, KeyError, TypeError) as exc:
                value = {"error": str(exc)}
            messages.append({"role": "tool", "tool_call_id": call.id,
                             "content": json.dumps(value, default=str)})
    return "I could not complete the tool-backed answer. Please ask a more specific question."


def answer_question(session_id, question):
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be nonempty")
    direct = _direct_answer(session_id, question)
    if direct is not None:
        return direct
    if os.environ.get("SLEEPSAFE_LLM_ENDPOINT"):
        return _llm_answer(session_id, question)
    return "I can answer questions about events, snoring, recording quality, classification, and the hourly summary. Set SLEEPSAFE_LLM_ENDPOINT for broader questions."
