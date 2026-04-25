from agent import AgenticStripeScout
from trace_wrapper import trace_agent_call

class TracedAgent:

    def __init__(self, db_path, user_id, password):
        self.agent = AgenticStripeScout(db_path, user_id, password)

    @property
    def history(self):
        # Reach into the actual agent to get the history list
        return self.agent.history

    @trace_agent_call
    def chat(self, query):
        return self.agent.chat(query)