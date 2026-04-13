import os
from langsmith import Client
from dotenv import load_dotenv

load_dotenv()

# Load keys from environment
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "LANGCHAIN_API_KEY"
os.environ["LANGCHAIN_PROJECT"] = "SupportSage-Pro"

client = Client()

def get_langsmith_client():
    return client 