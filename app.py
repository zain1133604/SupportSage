import gradio as gr
import os
import logging
import shutil
from database import ChromaVectorDB
from chunking import AscendedRAGPipeline
from embedding import EmbeddingEngine
from agent_traced import TracedAgent
from compliance_agent import ComplianceAgent

# --- LOGGING ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SupportSage.Production")

DB_PATH = r"D:\project dataset\RAG project\chromadb"
db_manager = ChromaVectorDB(persist_dir=DB_PATH)

# --- TAB 1: INGESTION LOGIC ---
def process_upload(user_id, password, files):
    if not user_id or not password or not files:
        return "Please provide ID, Password and Files."
    try:
        if not db_manager.register_user(user_id, password):
            try:
                db_manager.authenticate(user_id, password)
            except:
                return "User exists but Password incorrect."

        temp_dir = f"./temp_{user_id}"
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)

        if isinstance(files, list):
            for f in files:
                dest_path = os.path.join(temp_dir, os.path.basename(f.name))
                shutil.copy(f.name, dest_path)
        else:
            shutil.copytree(files, temp_dir, dirs_exist_ok=True)

        logger.info(f"Data prepared in {temp_dir}. Starting RAG Pipeline...")

        pipeline = AscendedRAGPipeline(base_path=temp_dir)
        parents, children = pipeline.process()

        if not parents:
            return "No valid text/pdf/code files found in the upload."

        embedder = EmbeddingEngine(model_name="BAAI/bge-m3", batch_size=4)
        embedded_data = embedder.split_parent_child(parents, children)
        db_manager.insert_user_data(user_id, embedded_data)

        shutil.rmtree(temp_dir)
        return f"Database for '{user_id}' ready! Go to the Chat or Compliance tab."
    except Exception as e:
        logger.error(f"Ingestion Error: {str(e)}")
        return f"Ingestion Error: {str(e)}"


def handle_deletion(user_id, password):
    if not user_id or not password:
        return "ID and Password required to delete."
    try:
        db_manager.delete_user_account(user_id, password)
        return f"Database for '{user_id}' has been permanently deleted."
    except Exception as e:
        return f"Deletion failed: {str(e)}"


# --- TAB 2: CHAT LOGIC ---
active_agents = {}

def chat_bridge(user_id, password, query, chat_history):
    session_key = f"{user_id}_{password}"
    try:
        if session_key not in active_agents:
            agent = TracedAgent(db_path=DB_PATH, user_id=user_id, password=password)
            active_agents[session_key] = agent

        agent = active_agents[session_key]
        answer = agent.chat(query)
        chat_history.append((query, answer))

        base_trace = {
            "last_query":   query,
            "session_user": user_id,
            "gpu_hardware": "RTX 3060 Ti",
            "agent_memory": f"{len(agent.history)} messages"
        }
        audit_data = getattr(agent, "_last_trace", {})
        trace = {**base_trace, **audit_data}

        return "", chat_history, trace

    except Exception as e:
        chat_history.append((query, f"System Error: {str(e)}"))
        return "", chat_history, {"error": str(e)}


# --- TAB 3: COMPLIANCE CHECK LOGIC ---
def run_compliance(user_id, password, framework):
    """
    Run a full compliance check against the selected framework.
    Returns a formatted markdown report + summary JSON.
    """
    if not user_id or not password:
        return "Please enter your User ID and Password.", {}

    if not framework:
        return "Please select a compliance framework.", {}

    try:
        comp_agent = ComplianceAgent(
            db_path      = DB_PATH,
            user_id      = user_id,
            password     = password,
            audit_db_path= "./audit_store.db",
        )

        results = comp_agent.run_compliance_check(framework)
        summary = ComplianceAgent.summarise(results)

        # --- Build markdown report ---
        lines = []
        lines.append(f"## {framework} Compliance Report")
        lines.append(f"**Score: {summary['score']}%** — "
                     f"{summary['satisfied']} Satisfied | "
                     f"{summary['partial']} Partial | "
                     f"{summary['gaps']} Gaps | "
                     f"{summary['total']} Total Controls\n")
        lines.append("---")

        for r in results:
            if r["status"] == "SATISFIED":
                icon = "✅"
            elif r["status"] == "PARTIAL":
                icon = "⚠️"
            else:
                icon = "❌"

            lines.append(f"### {icon} {r['control_id']} — {r['control_name']}")
            lines.append(f"**Status:** {r['status']}  |  **Confidence:** {round(r['confidence'] * 100)}%")
            lines.append(f"**Requirement:** {r['requirement']}")

            if r["evidence_snippet"]:
                lines.append(f"**Evidence:** *\"{r['evidence_snippet']}\"*")
                lines.append(f"**Source:** `{r['source']}`")
            else:
                lines.append("**Evidence:** No matching evidence found in uploaded documents.")

            lines.append(f"**Reasoning:** {r['reasoning']}")
            lines.append("---")

        report_text = "\n\n".join(lines)

        summary_display = {
            "framework":          framework,
            "compliance_score":   f"{summary['score']}%",
            "satisfied_controls": summary["satisfied"],
            "partial_controls":   summary["partial"],
            "gap_controls":       summary["gaps"],
            "total_controls":     summary["total"],
            "audit_saved_to":     "./audit_store.db",
            "note":               "All results persisted — queryable without replaying the agent.",
        }

        return report_text, summary_display

    except Exception as e:
        logger.error(f"Compliance check error: {str(e)}")
        return f"Error: {str(e)}", {"error": str(e)}


# --- UI ---
with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate"),
    title="SupportSage Pro"
) as demo:

    gr.Markdown("# 🛡️ SupportSage Pro: Multi-Tenant RAG + Compliance")

    with gr.Tabs():

        # ── TAB 1: DATA FORGE ──────────────────────────────────────────
        with gr.TabItem("Data Forge"):
            with gr.Row():
                with gr.Column(scale=1):
                    u_id        = gr.Textbox(label="User ID", placeholder="e.g., zain_ali")
                    u_pw        = gr.Textbox(label="Password", type="password")
                    file_output = gr.File(label="Upload Documents", file_count="directory")
                    upload_btn  = gr.Button("Build Vector Intelligence", variant="primary")
                    gr.Markdown("---")
                    gr.Markdown("### Danger Zone")
                    delete_btn  = gr.Button("Delete My Database", variant="stop")
                with gr.Column(scale=1):
                    status_output = gr.Textbox(label="System Logs", interactive=False)

        # ── TAB 2: INTELLIGENCE CONSOLE ────────────────────────────────
        with gr.TabItem("Intelligence Console"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Session Login")
                    login_id = gr.Textbox(label="User ID")
                    login_pw = gr.Textbox(label="Password", type="password")
                    gr.Markdown("---")
                    trace_json = gr.JSON(label="Live Logic Trace")

                with gr.Column(scale=2):
                    chatbot = gr.Chatbot(
                        label="Agentic Support Bot",
                        height=500,
                        type="messages"
                    )
                    with gr.Row():
                        chat_query = gr.Textbox(
                            label="Message",
                            placeholder="Ask anything about your data...",
                            scale=4
                        )
                        send_btn = gr.Button("Send", variant="primary", scale=1)
                    clear_btn = gr.Button("Clear Chat")

        # ── TAB 3: COMPLIANCE CHECK ────────────────────────────────────
        with gr.TabItem("Compliance Check"):
            gr.Markdown("""
### Evidence-to-Control Mapping
Upload your policy and procedure documents in **Data Forge** first,
then run a compliance check here. The agent will scan every control
in the selected framework, find supporting evidence from your documents,
and tell you what is **Satisfied**, **Partial**, or a **Gap**.

All results are saved to `audit_store.db` — queryable without replaying the agent.
""")
            with gr.Row():
                with gr.Column(scale=1):
                    comp_user_id   = gr.Textbox(label="User ID")
                    comp_password  = gr.Textbox(label="Password", type="password")
                    framework_drop = gr.Dropdown(
                        choices  = ["CMMC", "NIST", "SOC2"],
                        label    = "Select Compliance Framework",
                        value    = "CMMC",
                    )
                    comp_btn = gr.Button("Run Compliance Check", variant="primary")
                    gr.Markdown("*This may take 1–3 minutes depending on framework size.*")
                    comp_summary = gr.JSON(label="Summary")

                with gr.Column(scale=2):
                    comp_report = gr.Markdown(label="Compliance Report")

    # ── BINDINGS ───────────────────────────────────────────────────────
    upload_btn.click(process_upload, [u_id, u_pw, file_output], status_output)
    delete_btn.click(handle_deletion, [u_id, u_pw], status_output)

    chat_query.submit(
        chat_bridge,
        inputs  = [login_id, login_pw, chat_query, chatbot],
        outputs = [chat_query, chatbot, trace_json]
    )
    send_btn.click(
        chat_bridge,
        inputs  = [login_id, login_pw, chat_query, chatbot],
        outputs = [chat_query, chatbot, trace_json]
    )
    clear_btn.click(lambda: [], None, chatbot, queue=False)

    comp_btn.click(
        run_compliance,
        inputs  = [comp_user_id, comp_password, framework_drop],
        outputs = [comp_report, comp_summary]
    )


if __name__ == "__main__":
    demo.launch(
        share       = True,
        server_name = "0.0.0.0",
        server_port = 7860,
        show_api    = False,
    )