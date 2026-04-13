from langsmith import traceable
import time

# Wrap ANY function you want to observe

def trace_agent_call(func):
    @traceable(name="SupportSage-Agent-Call")
    def wrapper(*args, **kwargs):
        start = time.time()

        result = func(*args, **kwargs)

        end = time.time()

        return result

    return wrapper