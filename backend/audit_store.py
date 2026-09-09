"""
audit_store.py — persistence layer for CureMyBill "audits" (one per bill
analysis). Backed by a simple SQLite file so it works with zero external
dependencies and zero configuration.

Two consumers share this same database file:
1. app.py (the Streamlit app) — writes the extracted/comparison/letter data,
   reads it back after a page reload via the `audit` URL parameter.
2. webhook_server.py (optional, deploy separately) — marks an audit as paid
   once Paddle's webhook confirms a real transaction, which is the only
   fully trustworthy way to unlock a paywall in production.

This file has no Streamlit-specific code so it can be imported by both.
"""

import json
import os
import sqlite3
import time
import uuid

DB_PATH = os.path.join(os.path.dirname(__file__), "curemybill_audits.db")


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS audits (
            audit_id TEXT PRIMARY KEY,
            data TEXT NOT NULL,
            paid INTEGER NOT NULL DEFAULT 0,
            paid_plan TEXT,
            purchased_addons TEXT,
            customer_email TEXT,
            email_sent INTEGER NOT NULL DEFAULT 0,
            created_at REAL NOT NULL
        )
        """
    )
    # Backfill columns for databases created before this field existed.
    existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(audits)")}
    if "customer_email" not in existing_cols:
        conn.execute("ALTER TABLE audits ADD COLUMN customer_email TEXT")
    if "email_sent" not in existing_cols:
        conn.execute("ALTER TABLE audits ADD COLUMN email_sent INTEGER NOT NULL DEFAULT 0")
    return conn


def new_audit_id() -> str:
    """Short, URL-friendly unique id for one bill analysis session."""
    return uuid.uuid4().hex[:12]


def save_audit(audit_id: str, data: dict) -> None:
    """Create or update the stored data for an audit (extracted bill info,
    comparison table, generated letters — anything needed to rebuild the
    page after a refresh)."""
    conn = _get_conn()
    try:
        conn.execute(
            """
            INSERT INTO audits (audit_id, data, created_at)
            VALUES (?, ?, ?)
            ON CONFLICT(audit_id) DO UPDATE SET data = excluded.data
            """,
            (audit_id, json.dumps(data), time.time()),
        )
        conn.commit()
    finally:
        conn.close()


def load_audit(audit_id: str) -> dict | None:
    """Fetch a stored audit, including its payment status. Returns None if
    the audit_id doesn't exist (e.g. expired, wrong id, or first visit)."""
    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT data, paid, paid_plan, purchased_addons FROM audits WHERE audit_id = ?",
            (audit_id,),
        ).fetchone()
    finally:
        conn.close()

    if row is None:
        return None

    data_json, paid, paid_plan, addons_json = row
    result = json.loads(data_json)
    result["_paid"] = bool(paid)
    result["_paid_plan"] = paid_plan
    result["_purchased_addons"] = json.loads(addons_json) if addons_json else []
    return result


def mark_paid(audit_id: str, plan: str, addons: list, customer_email: str = None) -> bool:
    """Called by the webhook (or, as a Sandbox/local fallback, by the app
    itself) once a payment is confirmed. Returns False if the audit_id is
    unknown (e.g. a forged/expired id), True if it was updated."""
    conn = _get_conn()
    try:
        cursor = conn.execute(
            "UPDATE audits SET paid = 1, paid_plan = ?, purchased_addons = ?, customer_email = ? WHERE audit_id = ?",
            (plan, json.dumps(addons), customer_email, audit_id),
        )
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def mark_email_sent(audit_id: str) -> None:
    conn = _get_conn()
    try:
        conn.execute("UPDATE audits SET email_sent = 1 WHERE audit_id = ?", (audit_id,))
        conn.commit()
    finally:
        conn.close()


def is_paid(audit_id: str) -> tuple[bool, str | None, list, str | None, bool]:
    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT paid, paid_plan, purchased_addons, customer_email, email_sent FROM audits WHERE audit_id = ?",
            (audit_id,),
        ).fetchone()
    finally:
        conn.close()

    if row is None:
        return False, None, [], None, False
    paid, plan, addons_json, customer_email, email_sent = row
    return bool(paid), plan, (json.loads(addons_json) if addons_json else []), customer_email, bool(email_sent)
