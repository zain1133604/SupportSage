"""
trace_wrapper.py
----------------
Wraps every agent.chat() call with LangSmith tracing AND
exposes the full persistent audit trail to the Gradio UI panel.
"""

from langsmith import traceable
import time


def trace_agent_call(func):
    @traceable(name="SupportSage-Agent-Call")
    def wrapper(self, *args, **kwargs):
        start  = time.time()
        result = func(self, *args, **kwargs)
        end    = time.time()

        latency = round(end - start, 3)

        # --- In-memory events logged this turn ---
        in_mem_events = []
        if hasattr(self.agent, "audit_log") and self.agent.audit_log:
            in_mem_events = self.agent.audit_log[-10:]

        # --- Persistent audit stats (survive restarts) ---
        persistent_stats = {}
        if hasattr(self.agent, "audit_db"):
            try:
                persistent_stats = self.agent.audit_db.summary_stats()
            except Exception:
                persistent_stats = {"error": "Could not read persistent audit store"}

        # --- Pending review queue size ---
        pending_reviews = []
        if hasattr(self.agent, "audit_db"):
            try:
                pending_reviews = self.agent.audit_db.get_pending_review_queue()
            except Exception:
                pending_reviews = []

        audit_summary = {
            "latency_seconds":      latency,
            "session_id":           getattr(self.agent, "session_id", "unknown"),
            "actor":                getattr(self.agent, "user_id", "unknown"),
            "events_this_turn":     len(in_mem_events),
            "audit_trail":          in_mem_events,
            # Randy's production requirement: persisted counters
            "persistent_audit": {
                **persistent_stats,
                "pending_review_count": len(pending_reviews),
                "store_path":           getattr(self.agent, "audit_db", None) and self.agent.audit_db.db_path,
            },
        }

        # Attach to self so app.py can read it via agent._last_trace
        self._last_trace = audit_summary
        return result

    return wrapper