"""Replace SampleRepository with DeltaRepository when the CRNN writes tables."""
import json
import os
from copy import deepcopy
from pathlib import Path

DATA = Path(__file__).parent / "data" / "sample_report.json"


class SampleRepository:
    def get_session(self, session_id):
        report = json.loads(DATA.read_text())
        return deepcopy(report) if session_id == report.get("session_id") else None

    def get_events(self, session_id):
        report = self.get_session(session_id)
        return report.get("events") if report else None


def get_session_from_delta(session_id):
    """Read report_json from workspace.default.sleepsafe_sessions."""
    rows = _query("SELECT report_json FROM workspace.default.sleepsafe_sessions "
                  "WHERE session_id = ? LIMIT 1", session_id)
    return json.loads(rows[0][0]) if rows else None


def get_events_from_delta(session_id):
    """Read event_json rows from workspace.default.sleepsafe_events."""
    rows = _query("SELECT event_json FROM workspace.default.sleepsafe_events "
                  "WHERE session_id = ? ORDER BY start_offset_s", session_id)
    return [json.loads(row[0]) for row in rows]


def _query(statement, session_id):
    from databricks import sql
    from databricks.sdk.core import Config
    cfg = Config()  # Databricks App service principal or local CLI profile
    http_path = os.environ["DATABRICKS_HTTP_PATH"]
    with sql.connect(server_hostname=cfg.host, http_path=http_path,
                     credentials_provider=lambda: cfg.authenticate) as connection:
        with connection.cursor() as cursor:
            cursor.execute(statement, [session_id])
            return cursor.fetchall()


class DeltaRepository:
    def get_session(self, session_id):
        report = get_session_from_delta(session_id)
        if report is not None:
            report["events"] = self.get_events(session_id)
        return report

    def get_events(self, session_id):
        return get_events_from_delta(session_id)
