import gradio as gr
import os
import logging
import shutil
from database import ChromaVectorDB
from chunking import AscendedRAGPipeline
from embedding import EmbeddingEngine
from agent_traced import TracedAgent
from compliance_agent import ComplianceAgent
from audit_db import AuditDatabase

AUDIT_DB_PATH = "./audit_store.db"

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


# --- TAB 4: AUDIT DASHBOARD LOGIC ---

def audit_get_stats():
    """Load summary stats when dashboard tab opens."""
    try:
        db = AuditDatabase(db_path=AUDIT_DB_PATH)
        return db.summary_stats()
    except Exception as e:
        return {"error": str(e)}


def audit_search_event(event_id):
    """Search for a single event by its ID."""
    if not event_id or not event_id.strip():
        return "Please enter an Event ID.", {}
    try:
        db     = AuditDatabase(db_path=AUDIT_DB_PATH)
        result = db.get_event(event_id.strip())
        if not result:
            return f"No event found for ID: `{event_id}`", {}
        import json
        details = json.loads(result.get("details_json", "{}"))
        display = {
            "event_id":          result["event_id"],
            "actor":             result["actor"],
            "session_id":        result["session_id"],
            "event_type":        result["event_type"],
            "timestamp_utc":     result["timestamp_utc"],
            "source_file":       result["source_file"],
            "control_ref":       result["control_ref"],
            "approval_state":    result["approval_state"],
            "needs_human_review": bool(result["needs_human_review"]),
            "tool_input_hash":   result["tool_input_hash"],
            "tool_output_hash":  result["tool_output_hash"],
            "rollback_sql":      result["rollback_sql"],
            "details":           details,
        }
        return f"✅ Event found: `{event_id}`", display
    except Exception as e:
        return f"Error: {str(e)}", {}


def audit_search_actor(actor):
    """Get all events for a user/actor."""
    if not actor or not actor.strip():
        return [], "Please enter a User ID."
    try:
        db     = AuditDatabase(db_path=AUDIT_DB_PATH)
        events = db.get_actor_trail(actor.strip(), limit=50)
        if not events:
            return [], f"No events found for actor: `{actor}`"
        rows = [
            [
                e["event_id"][:16] + "...",
                e["event_type"],
                e["timestamp_utc"][:19],
                e["control_ref"] or "-",
                e["approval_state"],
                "⚠️ Yes" if e["needs_human_review"] else "No",
                e["source_file"] or "-",
            ]
            for e in events
        ]
        return rows, f"Found {len(events)} events for actor: `{actor}`"
    except Exception as e:
        return [], f"Error: {str(e)}"


def audit_search_control(control_ref):
    """Get all events for a control reference — answers 'why did this control pass'."""
    if not control_ref or not control_ref.strip():
        return [], "Please enter a Control Reference (e.g. CMMC.AC.1.001)"
    try:
        db     = AuditDatabase(db_path=AUDIT_DB_PATH)
        events = db.explain_control(control_ref.strip())
        if not events:
            return [], f"No events found for control: `{control_ref}`"
        rows = [
            [
                e["event_id"][:16] + "...",
                e["actor"],
                e["event_type"],
                e["timestamp_utc"][:19],
                e["approval_state"],
                e["rollback_sql"][:60] + "..." if e["rollback_sql"] else "-",
            ]
            for e in events
        ]
        return rows, f"Found {len(events)} evidence records for control: `{control_ref}`"
    except Exception as e:
        return [], f"Error: {str(e)}"


def audit_get_review_queue():
    """Get all events pending human review."""
    try:
        db     = AuditDatabase(db_path=AUDIT_DB_PATH)
        events = db.get_pending_review_queue()
        if not events:
            return [], "✅ Review queue is empty — no pending actions."
        rows = [
            [
                e["event_id"][:16] + "...",
                e["actor"],
                e["event_type"],
                e["timestamp_utc"][:19],
                e["control_ref"] or "-",
                e["approval_state"],
            ]
            for e in events
        ]
        return rows, f"⚠️ {len(events)} items waiting for human review."
    except Exception as e:
        return [], f"Error: {str(e)}"


def audit_get_sql_actions(actor_filter):
    """Get all state-changing SQL actions, optionally filtered by actor."""
    try:
        db     = AuditDatabase(db_path=AUDIT_DB_PATH)
        actor  = actor_filter.strip() if actor_filter and actor_filter.strip() else None
        events = db.get_sql_actions(actor=actor)
        if not events:
            msg = f"No SQL actions found" + (f" for actor: `{actor}`" if actor else ".")
            return [], msg
        rows = [
            [
                e["event_id"][:16] + "...",
                e["actor"],
                e["timestamp_utc"][:19],
                e["source_file"] or "-",
                e["approval_state"],
                e["rollback_sql"][:60] + "..." if e["rollback_sql"] else "-",
            ]
            for e in events
        ]
        msg = f"Found {len(events)} SQL actions" + (f" for actor: `{actor}`" if actor else ".")
        return rows, msg
    except Exception as e:
        return [], f"Error: {str(e)}"


def audit_get_rollback(event_id):
    """Get rollback SQL for a specific event."""
    if not event_id or not event_id.strip():
        return "Please enter an Event ID."
    try:
        db  = AuditDatabase(db_path=AUDIT_DB_PATH)
        sql = db.get_rollback_info(event_id.strip())
        if not sql:
            return f"No rollback SQL recorded for event: `{event_id}`"
        return f"**Rollback SQL for `{event_id}`:**\n\n```sql\n{sql}\n```"
    except Exception as e:
        return f"Error: {str(e)}"


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

        # ── TAB 4: AUDIT DASHBOARD ────────────────────────────────────
        with gr.TabItem("Audit Dashboard"):
            gr.Markdown("""
### Immutable Audit Trail — Query Without Replaying the Agent
Every action the system takes is saved here permanently.
Search by Event ID, User, Control Reference, or view the human review queue.
""")
            # --- Stats Row ---
            with gr.Row():
                stats_json  = gr.JSON(label="Live Audit Stats")
                refresh_btn = gr.Button("Refresh Stats", variant="secondary")

            gr.Markdown("---")

            with gr.Tabs():

                # Search by Event ID
                with gr.TabItem("Search by Event ID"):
                    with gr.Row():
                        event_id_input  = gr.Textbox(
                            label="Event ID",
                            placeholder="Paste any event_id here e.g. 7dafa605f8cb48e2..."
                        )
                        event_search_btn = gr.Button("Search", variant="primary")
                    event_status  = gr.Markdown()
                    event_result  = gr.JSON(label="Event Record")

                # Search by Actor / User
                with gr.TabItem("Search by User"):
                    with gr.Row():
                        actor_input     = gr.Textbox(
                            label="User ID (Actor)",
                            placeholder="e.g. zainali1122"
                        )
                        actor_search_btn = gr.Button("Search", variant="primary")
                    actor_status = gr.Markdown()
                    actor_table  = gr.Dataframe(
                        headers=["Event ID", "Type", "Timestamp", "Control Ref", "Approval", "Needs Review", "Source"],
                        label="User Events (last 50)",
                        interactive=False,
                        wrap=True,
                    )

                # Search by Control Reference
                with gr.TabItem("Search by Control"):
                    gr.Markdown("*Answer: 'Show me why this control passed' — without replaying the agent.*")
                    with gr.Row():
                        control_input     = gr.Textbox(
                            label="Control Reference",
                            placeholder="e.g. CMMC.AC.1.001 or ORDER_WRITE"
                        )
                        control_search_btn = gr.Button("Search", variant="primary")
                    control_status = gr.Markdown()
                    control_table  = gr.Dataframe(
                        headers=["Event ID", "Actor", "Type", "Timestamp", "Approval", "Rollback SQL"],
                        label="Evidence Records",
                        interactive=False,
                        wrap=True,
                    )

                # Human Review Queue
                with gr.TabItem("Review Queue"):
                    gr.Markdown("*All actions flagged for human approval — refunds, complaints, low-confidence answers.*")
                    review_load_btn = gr.Button("Load Review Queue", variant="primary")
                    review_status   = gr.Markdown()
                    review_table    = gr.Dataframe(
                        headers=["Event ID", "Actor", "Type", "Timestamp", "Control Ref", "Approval State"],
                        label="Pending Human Review",
                        interactive=False,
                        wrap=True,
                    )

                # SQL Actions
                with gr.TabItem("SQL Actions"):
                    gr.Markdown("*All state-changing database actions — cancels, address changes, modifies. Filter by user optionally.*")
                    with gr.Row():
                        sql_actor_input = gr.Textbox(
                            label="Filter by User ID (optional)",
                            placeholder="Leave empty to see all, or enter a user ID"
                        )
                        sql_actions_btn = gr.Button("Load SQL Actions", variant="primary")
                    sql_actions_status = gr.Markdown()
                    sql_actions_table  = gr.Dataframe(
                        headers=["Event ID", "Actor", "Timestamp", "Table", "Approval", "Rollback SQL"],
                        label="SQL Action Log",
                        interactive=False,
                        wrap=True,
                    )

                # Rollback SQL Lookup
                with gr.TabItem("Rollback SQL"):
                    gr.Markdown("*Get the exact SQL to undo any state-changing action.*")
                    with gr.Row():
                        rollback_input = gr.Textbox(
                            label="Event ID",
                            placeholder="Paste event_id of the action you want to undo"
                        )
                        rollback_btn = gr.Button("Get Rollback SQL", variant="primary")
                    rollback_output = gr.Markdown()

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

    # Audit Dashboard bindings
    refresh_btn.click(audit_get_stats, inputs=[], outputs=[stats_json])

    event_search_btn.click(
        audit_search_event,
        inputs  = [event_id_input],
        outputs = [event_status, event_result]
    )

    actor_search_btn.click(
        audit_search_actor,
        inputs  = [actor_input],
        outputs = [actor_table, actor_status]
    )

    control_search_btn.click(
        audit_search_control,
        inputs  = [control_input],
        outputs = [control_table, control_status]
    )

    review_load_btn.click(
        audit_get_review_queue,
        inputs  = [],
        outputs = [review_table, review_status]
    )

    rollback_btn.click(
        audit_get_rollback,
        inputs  = [rollback_input],
        outputs = [rollback_output]
    )

    sql_actions_btn.click(
        audit_get_sql_actions,
        inputs  = [sql_actor_input],
        outputs = [sql_actions_table, sql_actions_status]
    )


if __name__ == "__main__":
    demo.launch(
        share       = True,
        server_name = "0.0.0.0",
        server_port = 7860,
        show_api    = False,
    )