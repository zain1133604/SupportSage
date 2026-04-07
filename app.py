import gradio as gr
import os
import logging
import shutil
from database import ChromaVectorDB
from chunking import AscendedRAGPipeline
from embedding import EmbeddingEngine
from agent import AgenticStripeScout

# --- LOGGING ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SupportSage.Production")

DB_PATH = r"D:\project dataset\RAG project\chromadb"
db_manager = ChromaVectorDB(persist_dir=DB_PATH)

# --- TAB 1: INGESTION LOGIC (Keep as is) ---
def process_upload(user_id, password, files):
    if not user_id or not password or not files:
        return "❌ Please provide ID, Password, and Files."
    try:
        if not db_manager.register_user(user_id, password):
            try:
                db_manager.authenticate(user_id, password)
            except:
                return "❌ User exists but Password incorrect."

        temp_dir = f"./temp_{user_id}"
        os.makedirs(temp_dir, exist_ok=True)
        for f in files:
            shutil.copy(f.name, os.path.join(temp_dir, os.path.basename(f.name)))

        pipeline = AscendedRAGPipeline(base_path=temp_dir)
        parents, children = pipeline.process()
        embedder = EmbeddingEngine(model_name="BAAI/bge-m3", batch_size=4) 
        embedded_data = embedder.split_parent_child(parents, children)
        db_manager.insert_user_data(user_id, embedded_data)

        shutil.rmtree(temp_dir)
        return f"✅ Database for '{user_id}' ready! Go to the Chat tab."
    except Exception as e:
        return f"❌ Error: {str(e)}"

# --- TAB 2: CHAT LOGIC (Updated for Chatbot) ---
active_agents = {}

def chat_bridge(user_id, password, query, chat_history):
    session_key = f"{user_id}_{password}"
    
    try:
        if session_key not in active_agents:
            # Note: Ensure your AgenticStripeScout is in agent.py
            agent = AgenticStripeScout(db_path=DB_PATH, user_id=user_id, password=password)
            active_agents[session_key] = agent
        
        agent = active_agents[session_key]
        
        # 1. Get the real response from your agent
        answer = agent.chat(query)
        
        # 2. Update Chatbot History (Format: [[user, bot], [user, bot]])
        chat_history.append((query, answer))
        
        # 3. Create a Trace log for the recruiter to see
        trace = {
            "last_query": query,
            "session_user": user_id,
            "gpu_hardware": "RTX 3060 Ti",
            "agent_memory": f"{len(agent.history)} messages"
        }
        
        return "", chat_history, trace
    
    except Exception as e:
        chat_history.append((query, f"❌ System Error: {str(e)}"))
        return "", chat_history, {"error": str(e)}

# --- 🎨 THE UI DESIGN (Updated for Chat Model feel) ---
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate"), title="SupportSage Pro") as demo:
    gr.Markdown("# 🛡️ SupportSage Pro: Multi-Tenant RAG")

    with gr.Tabs():
        # --- TAB 1: DATA FORGE ---
        with gr.TabItem("🏗️ Data Forge"):
            with gr.Row():
                with gr.Column(scale=1):
                    u_id = gr.Textbox(label="User ID", placeholder="e.g., zain_ali")
                    u_pw = gr.Textbox(label="Password", type="password")
                    file_output = gr.File(label="Upload Documents", file_count="multiple")
                    upload_btn = gr.Button("🚀 Build Vector Intelligence", variant="primary")
                with gr.Column(scale=1):
                    status_output = gr.Textbox(label="System Logs", interactive=False)

        # --- TAB 2: INTELLIGENCE CONSOLE (CHAT MODEL UI) ---
        with gr.TabItem("🧠 Intelligence Console"):
            with gr.Row():
                # Side Panel for Auth
                with gr.Column(scale=1):
                    gr.Markdown("### 🔐 Session Login")
                    login_id = gr.Textbox(label="User ID")
                    login_pw = gr.Textbox(label="Password", type="password")
                    gr.Markdown("---")
                    trace_json = gr.JSON(label="Live Logic Trace")

                # Main Chat Area
                with gr.Column(scale=2):
                    chatbot = gr.Chatbot(label="Agentic Support Bot", height=500)
                    with gr.Row():
                        chat_query = gr.Textbox(
                            label="Message", 
                            placeholder="Ask anything about your data...", 
                            scale=4
                        )
                        send_btn = gr.Button("Send", variant="primary", scale=1)
                    clear_btn = gr.Button("Clear Chat")

    # --- BINDING ---
    upload_btn.click(process_upload, [u_id, u_pw, file_output], status_output)
    
    # We pass the chatbot state (history) back and forth
    send_btn.click(
        chat_bridge, 
        inputs=[login_id, login_pw, chat_query, chatbot], 
        outputs=[chat_query, chatbot, trace_json]
    )
    # Also allow pressing 'Enter' to send
    chat_query.submit(
        chat_bridge, 
        inputs=[login_id, login_pw, chat_query, chatbot], 
        outputs=[chat_query, chatbot, trace_json]
    )
    
    clear_btn.click(lambda: None, None, chatbot, queue=False)


if __name__ == "__main__":
    # This is the "Production" launch command
    demo.launch(
        share=True, 
        server_name="0.0.0.0", 
        server_port=7860,
        show_api=False # Hides the 'view API' button from recruiters for a cleaner look
    )