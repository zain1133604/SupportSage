"""
audit_db.py — Persistent Immutable Audit Store
------------------------------------------------
Implements Randy's production requirement:
  - Immutable audit events persisted to SQLite (survives restarts)
  - Every record carries: actor, source, timestamp, tool input/output hash,
    control/evidence linkage, approval state, and rollback reference
  - Queryable without replaying the agent — an auditor can ask
    "show me why this control passed" and get a direct answer
"""

import sqlite3
import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_DDL = """
CREATE TABLE IF NOT EXISTS audit_events (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id            TEXT    NOT NULL UNIQUE,          -- UUID for this event
    session_id          TEXT    NOT NULL,                 -- groups events per chat session
    actor               TEXT    NOT NULL,                 -- user_id who triggered the action
    source_file         TEXT,                             -- document / table / tool that was touched
    event_type          TEXT    NOT NULL,                 -- ROUTING | RAG_RETRIEVAL | SQL_ACTION | REFLECTION_SCORE | COMPLAINT
    timestamp_utc       TEXT    NOT NULL,                 -- ISO-8601, always UTC
    tool_input_hash     TEXT,                             -- SHA-256 of the serialised tool input
    tool_output_hash    TEXT,                             -- SHA-256 of the serialised tool output
    control_ref         TEXT,                             -- compliance control this event satisfies (e.g. CMMC AC.1.001)
    approval_state      TEXT    NOT NULL DEFAULT 'AUTO',  -- AUTO | PENDING | APPROVED | REJECTED
    rollback_sql        TEXT,                             -- exact SQL to undo a state-changing action
    details_json        TEXT    NOT NULL,                 -- full event payload (input, output, scores …)
    needs_human_review  INTEGER NOT NULL DEFAULT 0        -- 1 = flagged for review queue
);

CREATE INDEX IF NOT EXISTS idx_audit_session   ON audit_events(session_id);
CREATE INDEX IF NOT EXISTS idx_audit_actor     ON audit_events(actor);
CREATE INDEX IF NOT EXISTS idx_audit_type      ON audit_events(event_type);
CREATE INDEX IF NOT EXISTS idx_audit_approval  ON audit_events(approval_state);
CREATE INDEX IF NOT EXISTS idx_audit_review    ON audit_events(needs_human_review);
"""


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _sha256(obj: Any) -> str:
    """Deterministic SHA-256 hash of any JSON-serialisable object."""
    raw = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(raw).hexdigest()


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_event_id() -> str:
    import uuid
    return uuid.uuid4().hex


# ---------------------------------------------------------------------------
# AuditDatabase
# ---------------------------------------------------------------------------

class AuditDatabase:
    """
    Thread-safe SQLite-backed audit store.

    All writes use WAL mode so reads never block writes, and every INSERT is
    committed immediately — there is no 'batch' mode to preserve immutability.
    """

    def __init__(self, db_path: str = "./audit_store.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._init_db()
        logger.info(f"[AuditDB] Persistent store ready at: {db_path}")

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_db(self):
        with self._connect() as conn:
            conn.executescript(_DDL)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")   # concurrent reads + writes
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    # ------------------------------------------------------------------
    # Core write path
    # ------------------------------------------------------------------

    def log_event(
        self,
        *,
        session_id: str,
        actor: str,
        event_type: str,
        details: Dict[str, Any],
        source_file: Optional[str] = None,
        tool_input: Optional[Any] = None,
        tool_output: Optional[Any] = None,
        control_ref: Optional[str] = None,
        approval_state: str = "AUTO",
        rollback_sql: Optional[str] = None,
        needs_human_review: bool = False,
    ) -> str:
        """
        Persist one immutable audit event.  Returns the event_id.

        Parameters
        ----------
        session_id          : identifies the chat / API session
        actor               : user_id who triggered the action
        event_type          : e.g. 'SQL_ACTION', 'RAG_RETRIEVAL'
        details             : full event payload — stored as JSON
        source_file         : document name / DB table touched
        tool_input          : raw input to the tool (hashed for tamper detection)
        tool_output         : raw output from the tool (hashed)
        control_ref         : compliance control reference this event maps to
        approval_state      : AUTO | PENDING | APPROVED | REJECTED
        rollback_sql        : exact SQL to undo a state change
        needs_human_review  : True → event lands in the review queue
        """
        event_id       = _new_event_id()
        timestamp      = _now_utc()
        input_hash     = _sha256(tool_input)  if tool_input  is not None else None
        output_hash    = _sha256(tool_output) if tool_output is not None else None
        details_json   = json.dumps(details, default=str)

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO audit_events (
                    event_id, session_id, actor, source_file,
                    event_type, timestamp_utc,
                    tool_input_hash, tool_output_hash,
                    control_ref, approval_state, rollback_sql,
                    details_json, needs_human_review
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    event_id, session_id, actor, source_file,
                    event_type, timestamp,
                    input_hash, output_hash,
                    control_ref, approval_state, rollback_sql,
                    details_json, int(needs_human_review),
                ),
            )
            conn.commit()

        logger.info(f"[AuditDB] {event_type} persisted | event_id={event_id} | actor={actor}")
        return event_id

    # ------------------------------------------------------------------
    # Approval state management (for PENDING → APPROVED / REJECTED flow)
    # ------------------------------------------------------------------

    def update_approval(self, event_id: str, new_state: str, reviewer: str) -> bool:
        """
        Update the approval state of a PENDING event.
        Logs a follow-up audit record so the approval itself is traceable.
        """
        allowed = {"APPROVED", "REJECTED"}
        if new_state not in allowed:
            raise ValueError(f"approval_state must be one of {allowed}")

        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE audit_events SET approval_state = ? WHERE event_id = ? AND approval_state = 'PENDING'",
                (new_state, event_id),
            )
            conn.commit()
            updated = cur.rowcount > 0

        if updated:
            logger.info(f"[AuditDB] event_id={event_id} → {new_state} by {reviewer}")
        return updated

    # ------------------------------------------------------------------
    # Query helpers — answering auditor questions without agent replay
    # ------------------------------------------------------------------

    def get_event(self, event_id: str) -> Optional[Dict]:
        """Fetch a single event by ID."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM audit_events WHERE event_id = ?", (event_id,)
            ).fetchone()
        return dict(row) if row else None

    def get_session_trail(self, session_id: str) -> List[Dict]:
        """Full chronological audit trail for one session."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM audit_events WHERE session_id = ? ORDER BY id ASC",
                (session_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_actor_trail(self, actor: str, limit: int = 100) -> List[Dict]:
        """All events for a given actor (user_id), most recent first."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM audit_events WHERE actor = ? ORDER BY id DESC LIMIT ?",
                (actor, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_pending_review_queue(self) -> List[Dict]:
        """
        Returns every event that needs a human to review.
        Answers: 'show me low-confidence answers waiting for approval.'
        """
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM audit_events
                WHERE needs_human_review = 1
                  AND approval_state IN ('AUTO', 'PENDING')
                ORDER BY id ASC
                """,
            ).fetchall()
        return [dict(r) for r in rows]

    def explain_control(self, control_ref: str) -> List[Dict]:
        """
        'Show me why control X passed.'
        Returns all evidence events mapped to a compliance control reference.
        """
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM audit_events
                WHERE control_ref = ?
                ORDER BY id ASC
                """,
                (control_ref,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_sql_actions(self, actor: Optional[str] = None) -> List[Dict]:
        """All state-changing SQL actions, optionally filtered by actor."""
        query  = "SELECT * FROM audit_events WHERE event_type = 'SQL_ACTION'"
        params: list = []
        if actor:
            query += " AND actor = ?"
            params.append(actor)
        query += " ORDER BY id DESC"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def get_rollback_info(self, event_id: str) -> Optional[str]:
        """
        Retrieve the exact rollback SQL for a state-changing event.
        Lets an ops team undo any action without touching the agent.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT rollback_sql FROM audit_events WHERE event_id = ?",
                (event_id,),
            ).fetchone()
        return row["rollback_sql"] if row else None

    def summary_stats(self) -> Dict[str, Any]:
        """Quick health-check numbers for the ops dashboard."""
        with self._connect() as conn:
            total       = conn.execute("SELECT COUNT(*) FROM audit_events").fetchone()[0]
            pending     = conn.execute("SELECT COUNT(*) FROM audit_events WHERE approval_state='PENDING'").fetchone()[0]
            review_q    = conn.execute("SELECT COUNT(*) FROM audit_events WHERE needs_human_review=1").fetchone()[0]
            sql_actions = conn.execute("SELECT COUNT(*) FROM audit_events WHERE event_type='SQL_ACTION'").fetchone()[0]
        return {
            "total_events":       total,
            "pending_approvals":  pending,
            "review_queue_size":  review_q,
            "sql_actions_logged": sql_actions,
        }