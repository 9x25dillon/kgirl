"""Regression: every console job hung "running" forever. Hub.emit(kind, **data) collided with the job's
own kind= field at job_start (and job_end), raising TypeError AFTER registration but BEFORE the worker
thread started. These start REAL jobs through JobManager.start with trivial functions — no model, no HTTP.
Written, not run.
"""

import time
import unittest

import tests.harness.console  # noqa: F401  (sys.path)
from kgirl.harness.console.core import Hub, Job, JobManager


def collect_until_end(q, job_id, timeout=5.0):
    """Drain the hub queue until this job's job_end arrives (the worker's last act), then return its events."""
    deadline = time.time() + timeout
    seen = []
    while time.time() < deadline:
        while not q.empty():
            seen.append(q.get_nowait())
        mine = [e for e in seen if e.get("job") == job_id]
        if any(e["type"] == "job_end" for e in mine):
            return mine
        time.sleep(0.01)
    return [e for e in seen if e.get("job") == job_id]


class Jobs(unittest.TestCase):
    def setUp(self):
        self.hub = Hub()
        self.jobs = JobManager(self.hub)
        self.q = self.hub.subscribe()
        self.addCleanup(self.hub.unsubscribe, self.q)

    def test_emit_lets_the_jobs_kind_field_ride_in_data(self):
        ev = self.hub.emit("job_start", job="x", kind="chat", label="l")
        self.assertEqual(ev["type"], "job_start")
        self.assertEqual(ev["kind"], "chat")
        self.assertEqual(ev["label"], "l")

    def test_real_heavy_job_completes_and_fires_start_and_end(self):
        job = self.jobs.start("chat", "trivial", lambda emit: {"answer": 42}, heavy=True)
        events = collect_until_end(self.q, job.id)
        self.assertEqual(job.status, "done")
        self.assertEqual(job.result, {"answer": 42})
        types = [e["type"] for e in events]
        self.assertIn("job_start", types)
        self.assertIn("job_end", types)
        self.assertEqual(events[types.index("job_start")]["kind"], "chat")
        self.assertEqual(events[types.index("job_end")]["status"], "done")

    def test_failing_job_reports_error_and_fires_end(self):
        def boom(emit):
            raise ValueError("nope")
        job = self.jobs.start("build", "boom", boom)
        events = collect_until_end(self.q, job.id)
        self.assertEqual(job.status, "error")
        self.assertIn("ValueError: nope", job.error)
        ends = [e for e in events if e["type"] == "job_end"]
        self.assertEqual(len(ends), 1)
        self.assertEqual(ends[0]["status"], "error")
        self.assertEqual(ends[0]["kind"], "build")

    def test_job_step_events_reach_the_hub_and_the_job_log(self):
        def chatty(emit):
            emit("status", text="working")
            return 1
        job = self.jobs.start("write", "chatty", chatty)
        events = collect_until_end(self.q, job.id)
        self.assertEqual(job.status, "done")
        wrapped = [e for e in events if e["type"] == "job"]
        self.assertTrue(any(e["event"]["type"] == "status" and e["event"]["text"] == "working"
                            for e in wrapped))
        self.assertEqual(job.events[0]["type"], "status")

    def test_summary_last_event_measures_silence_for_the_tray(self):
        job = Job("abc123", "chat", "a label")
        self.assertEqual(job.summary()["last_event"], job.created)   # silent since creation
        job.events.append({"type": "status", "ts": job.created + 5, "text": "working"})
        self.assertEqual(job.summary()["last_event"], job.created + 5)
        job.events.append({"type": "delta", "ts": job.created + 9, "text": "x"})
        self.assertEqual(job.summary()["last_event"], job.created + 9)   # deltas count as events too


if __name__ == "__main__":
    unittest.main()
