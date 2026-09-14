"""
cleanup.py — run daily (via a Render Cron Job) to enforce CureMyBill's data
retention policy: delete unpaid audits after 48h, and delete paid audits
30 days after payment (keeping only a minimal, non-health receipt for
accounting). See audit_store.cleanup_expired_audits for the exact rules.

Usage: python cleanup.py
"""

import audit_store

if __name__ == "__main__":
    audit_store.init_db()
    result = audit_store.cleanup_expired_audits()
    print(f"Cleanup done: {result}")
