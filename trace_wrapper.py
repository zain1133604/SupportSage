from langsmith import traceable
import time

# Wrap ANY function you want to observe

def trace_agent_call(func):
    @traceable(name="SupportSage-Agent-Call")
    def wrapper(self, *args, **kwargs):
        start = time.time()

        result = func(self, *args, **kwargs)

        end = time.time()

        # Randy's requirement: expose the full audit trail after every call.
        # Captures: source snippets, SQL tool calls + rollback paths,
        # routing confidence, and low-confidence review flags.
        audit_summary = {}
        if hasattr(self.agent, "audit_log") and self.agent.audit_log:
            last_events = self.agent.audit_log[-10:]  # Last 10 events for this turn
            audit_summary = {
                "latency_seconds": round(end - start, 3),
                "events_this_turn": len(last_events),
                "audit_trail": last_events
            }

        # Attach to self so app.py can read it
        self._last_trace = audit_summary

        return result

    return wrapper