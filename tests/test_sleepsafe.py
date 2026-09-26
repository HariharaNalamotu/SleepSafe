import copy
import unittest

from sleepsafe import tools
from sleepsafe.classifier import classify_sleep_session
from sleepsafe.agent import answer_question


class FixtureRepository:
    def __init__(self, report): self.report = report
    def get_session(self, session_id):
        return copy.deepcopy(self.report) if self.report and session_id == self.report["session_id"] else None
    def get_events(self, session_id):
        report = self.get_session(session_id)
        return report.get("events") if report else None


class SleepSafeTests(unittest.TestCase):
    def setUp(self):
        self.original = tools._repo
        self.base = self.original.get_session("a3f9c2e1-lapel-0926")
    def tearDown(self): tools.set_repository(self.original)
    def use(self, report): tools.set_repository(FixtureRepository(report))
    def test_sample_and_questions(self):
        self.assertEqual(classify_sleep_session(self.base["session_id"])["classification"], "mixed_respiratory_pattern")
        self.assertIn("41 seconds", answer_question(self.base["session_id"], "What was the longest apnea event?"))
        self.assertIn("e031", answer_question(self.base["session_id"], "What happened around 12:18 AM?"))
    def test_normal(self):
        x = copy.deepcopy(self.base); x["summary"].update(estimated_ahi=0, snore_pct_of_sleep=0, severity_estimate="normal")
        x["summary"]["counts"].update(apnea=0, hypopnea=0); x["events"] = []
        self.use(x); self.assertEqual(classify_sleep_session(x["session_id"])["classification"], "no_significant_respiratory_pattern")
    def test_apnea_heavy(self):
        x = copy.deepcopy(self.base); x["summary"]["snore_pct_of_sleep"] = 5
        self.use(x); self.assertEqual(classify_sleep_session(x["session_id"])["classification"], "apnea_hypopnea_pattern")
    def test_snoring_heavy(self):
        x = copy.deepcopy(self.base); x["summary"]["estimated_ahi"] = 0
        x["summary"]["counts"].update(apnea=0, hypopnea=0)
        self.use(x); self.assertEqual(classify_sleep_session(x["session_id"])["classification"], "snoring_dominant_pattern")
    def test_poor_audio(self):
        x = copy.deepcopy(self.base); x["recording"]["valid_audio_s"] = 1000
        self.use(x); self.assertEqual(classify_sleep_session(x["session_id"])["classification"], "poor_audio_quality")
    def test_zero_shot_heavy(self):
        x = copy.deepcopy(self.base); x["summary"].update(estimated_ahi=0, snore_pct_of_sleep=0)
        x["summary"]["counts"].update(apnea=0, hypopnea=0)
        x["events"] = [e for e in x["events"] if e.get("confidence_basis") == "zero_shot"]
        self.use(x); self.assertEqual(classify_sleep_session(x["session_id"])["classification"], "no_significant_respiratory_pattern")
    def test_missing_data(self):
        x = copy.deepcopy(self.base); x.pop("summary")
        self.use(x); self.assertEqual(classify_sleep_session(x["session_id"])["classification"], "insufficient_evidence")
        x.pop("events"); self.use(x); self.assertEqual(classify_sleep_session(x["session_id"])["classification"], "insufficient_evidence")
        with self.assertRaises(KeyError): classify_sleep_session("missing")
    def test_empty_events(self):
        x = copy.deepcopy(self.base); x["events"] = []
        self.use(x); self.assertIn("No individual events", classify_sleep_session(x["session_id"])["caveats"][-2])


if __name__ == "__main__": unittest.main()
