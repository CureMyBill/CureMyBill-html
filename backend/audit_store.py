"""
audit_store.py — persistence layer for CureMyBill "audits" (one per bill
analysis). Backed by a Postgres database (Render) so data survives
redeploys and server restarts — a local SQLite file would not.

Retention policy (see cleanup_expired_audits):
- Unpaid audits (someone uploaded a bill but never bought anything) are
  deleted after 48 hours.
- Paid audits are deleted 30 days after payment. A minimal receipt (no
  bill/health data — just audit_id, plan, email, paid_at) is kept
  indefinitely in `purchase_receipts` for accounting/support/legal
  purposes, matching what the privacy policy promises.
"""

import json
import os
import time
import uuid
from contextlib import contextmanager

import psycopg2
import psycopg2.extras
from cryptography.fernet import Fernet, InvalidToken

DATABASE_URL = os.getenv("DATABASE_URL", "")
ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY", "")
_fernet = Fernet(ENCRYPTION_KEY.encode()) if ENCRYPTION_KEY else None

UNPAID_RETENTION_HOURS = 48
PAID_RETENTION_DAYS = 30


def _encrypt(plaintext: str) -> str:
    """Encrypt a string before it's written to the database. If no
    encryption key is configured, data is stored as-is (fails loud in
    production since ENCRYPTION_KEY should always be set there)."""
    if plaintext is None:
        return None
    if not _fernet:
        return plaintext
    return _fernet.encrypt(plaintext.encode()).decode()


def _decrypt(value: str) -> str:
    """Decrypt a value read from the database. Falls back to returning the
    raw value if it isn't a valid encrypted token — this only happens for
    old test rows written before encryption was added, and lets us read
    them without crashing instead of silently losing data."""
    if value is None:
        return None
    if not _fernet:
        return value
    try:
        return _fernet.decrypt(value.encode()).decode()
    except (InvalidToken, ValueError):
        return value


@contextmanager
def _get_conn():
    conn = psycopg2.connect(DATABASE_URL)
    try:
        yield conn
    finally:
        conn.close()


def init_db() -> None:
    """Create tables if they don't exist yet. Safe to call on every startup."""
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS audits (
                    audit_id TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    paid BOOLEAN NOT NULL DEFAULT FALSE,
                    paid_plan TEXT,
                    purchased_addons JSONB,
                    customer_email TEXT,
                    email_sent BOOLEAN NOT NULL DEFAULT FALSE,
                    created_at DOUBLE PRECISION NOT NULL,
                    paid_at DOUBLE PRECISION
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS purchase_receipts (
                    audit_id TEXT PRIMARY KEY,
                    plan TEXT,
                    customer_email TEXT,
                    paid_at DOUBLE PRECISION NOT NULL
                )
                """
            )
            # One-time migration: the `data` column was originally created as
            # JSONB before field-level encryption was added. Encrypted values
            # are opaque text, not valid JSON, so the column type must be TEXT.
            # This only ever matters for the handful of rows created before
            # this change (all test data, pre-launch).
            cur.execute(
                """
                SELECT data_type FROM information_schema.columns
                WHERE table_name = 'audits' AND column_name = 'data'
                """
            )
            row = cur.fetchone()
            if row and row[0] == "jsonb":
                cur.execute("ALTER TABLE audits ALTER COLUMN data TYPE TEXT USING data::text")
        conn.commit()


def new_audit_id() -> str:
    """Short, URL-friendly unique id for one bill analysis session."""
    return uuid.uuid4().hex[:12]


def save_audit(audit_id: str, data: dict) -> None:
    """Create or update the stored data for an audit (extracted bill info,
    comparison table, generated letters — anything needed to rebuild the
    page after a refresh). Encrypted at rest."""
    encrypted = _encrypt(json.dumps(data))
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO audits (audit_id, data, created_at)
                VALUES (%s, %s, %s)
                ON CONFLICT (audit_id) DO UPDATE SET data = EXCLUDED.data
                """,
                (audit_id, encrypted, time.time()),
            )
        conn.commit()


def load_audit(audit_id: str) -> dict | None:
    """Fetch a stored audit, including its payment status. Returns None if
    the audit_id doesn't exist (e.g. expired, wrong id, or first visit)."""
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT data, paid, paid_plan, purchased_addons FROM audits WHERE audit_id = %s",
                (audit_id,),
            )
            row = cur.fetchone()

    if row is None:
        return None

    data_text, paid, paid_plan, addons_json = row
    result = json.loads(_decrypt(data_text))
    result["_paid"] = bool(paid)
    result["_paid_plan"] = paid_plan
    result["_purchased_addons"] = addons_json or []
    return result


def mark_paid(audit_id: str, plan: str, addons: list, customer_email: str = None) -> bool:
    """Called by the webhook once a payment is confirmed. Returns False if
    the audit_id is unknown (e.g. a forged/expired id), True if updated."""
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE audits
                SET paid = TRUE, paid_plan = %s, purchased_addons = %s,
                    customer_email = %s, paid_at = %s
                WHERE audit_id = %s
                """,
                (plan, psycopg2.extras.Json(addons), _encrypt(customer_email), time.time(), audit_id),
            )
            updated = cur.rowcount > 0
        conn.commit()
        return updated


def mark_email_sent(audit_id: str) -> None:
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE audits SET email_sent = TRUE WHERE audit_id = %s", (audit_id,))
        conn.commit()


def is_paid(audit_id: str) -> tuple[bool, str | None, list, str | None, bool]:
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT paid, paid_plan, purchased_addons, customer_email, email_sent FROM audits WHERE audit_id = %s",
                (audit_id,),
            )
            row = cur.fetchone()

    if row is None:
        return False, None, [], None, False
    paid, plan, addons_json, customer_email, email_sent = row
    return bool(paid), plan, (addons_json or []), _decrypt(customer_email), bool(email_sent)


def cleanup_expired_audits() -> dict:
    """Delete audits per the retention policy described at the top of this
    file. Intended to run on a schedule (see cleanup.py / the Render Cron
    Job), not on every request. Returns counts for logging."""
    now = time.time()
    unpaid_cutoff = now - UNPAID_RETENTION_HOURS * 3600
    paid_cutoff = now - PAID_RETENTION_DAYS * 86400

    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM audits WHERE paid = FALSE AND created_at < %s",
                (unpaid_cutoff,),
            )
            deleted_unpaid = cur.rowcount

            # Archive a minimal, non-health receipt before deleting the full
            # record for old paid audits.
            cur.execute(
                """
                SELECT audit_id, paid_plan, customer_email, paid_at
                FROM audits
                WHERE paid = TRUE AND paid_at IS NOT NULL AND paid_at < %s
                """,
                (paid_cutoff,),
            )
            old_paid = cur.fetchall()
            for audit_id, plan, customer_email, paid_at in old_paid:
                cur.execute(
                    """
                    INSERT INTO purchase_receipts (audit_id, plan, customer_email, paid_at)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (audit_id) DO NOTHING
                    """,
                    (audit_id, plan, customer_email, paid_at),
                )
            cur.execute(
                "DELETE FROM audits WHERE paid = TRUE AND paid_at IS NOT NULL AND paid_at < %s",
                (paid_cutoff,),
            )
            deleted_paid = cur.rowcount
        conn.commit()

    return {"deleted_unpaid": deleted_unpaid, "deleted_paid_archived": deleted_paid}
