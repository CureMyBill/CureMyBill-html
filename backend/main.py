"""
main.py — CureMyBill backend API.

Serves the frontend (plain HTML/CSS/JS, in ../frontend) and exposes the
endpoints it calls: analyze a bill, generate letters, generate PDFs, restore
an audit, and receive Paddle webhooks.

Run locally with:  uvicorn main:app --reload --port 8000
"""

import hashlib
import hmac
import json
import os
from typing import Optional

import audit_store
import logic
import requests
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="CureMyBill API")


@app.on_event("startup")
def _init_database():
    audit_store.init_db()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to your real domain once deployed
    allow_methods=["*"],
    allow_headers=["*"],
)

FEE_SCHEDULE = logic.load_fee_schedule()


def _client():
    try:
        return logic.get_client()
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))


# --------------------------------------------------------------------------
# Bill analysis
# --------------------------------------------------------------------------


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...), turnstile_token: str = Form(default="")):
    if not logic.verify_turnstile(turnstile_token):
        raise HTTPException(status_code=403, detail="Verification failed. Please refresh the page and try again.")

    raw_bytes = await file.read()
    client = _client()
    file_block = logic.bytes_to_content_block(raw_bytes, file.filename or "", file.content_type or "")

    try:
        extracted = logic.extract_bill_data(client, file_block)
    except json.JSONDecodeError:
        raise HTTPException(status_code=422, detail="Could not read a usable result from the bill.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    comparison_df = logic.compare_to_schedule(extracted.get("line_items", []), FEE_SCHEDULE)
    npi_result = logic.verify_provider_npi(
        extracted.get("provider_name", ""), extracted.get("provider_address", "")
    )
    audit_id = audit_store.new_audit_id()
    audit_store.save_audit(
        audit_id,
        {"extracted": extracted, "comparison_records": comparison_df.to_dict(orient="records")},
    )

    return {
        "audit_id": audit_id,
        "extracted": extracted,
        "comparison": comparison_df.to_dict(orient="records"),
        "npi_verification": npi_result,
    }


@app.post("/api/demo")
async def demo():
    sample_extracted = {
        "provider_name": "St. Jude Community Hospital",
        "provider_address": "123 Health Ave, Austin, TX 78701",
        "patient_name": "John Doe",
        "bill_date": "05/12/2026",
        "account_number": "987654321",
        "line_items": [
            {"cpt_code": "99285", "description": "Emergency Department Visit, Level 5", "quantity": 1, "billed_amount": 2450.00},
            {"cpt_code": "70450", "description": "CT Scan, Head/Brain without Contrast", "quantity": 1, "billed_amount": 3200.00},
            {"cpt_code": "70450", "description": "CT Scan, Head/Brain without Contrast (duplicate entry)", "quantity": 1, "billed_amount": 3200.00},
            {"cpt_code": "J7030", "description": "Saline IV Solution, 1000ml", "quantity": 1, "billed_amount": 380.00},
            {"cpt_code": "96374", "description": "IV push, single/initial substance", "quantity": 1, "billed_amount": 180.00},
        ],
        "total_billed": 9410.00,
    }
    comparison_df = logic.compare_to_schedule(sample_extracted["line_items"], FEE_SCHEDULE)
    audit_id = audit_store.new_audit_id()
    audit_store.save_audit(
        audit_id,
        {"extracted": sample_extracted, "comparison_records": comparison_df.to_dict(orient="records")},
    )
    return {
        "audit_id": audit_id,
        "extracted": sample_extracted,
        "comparison": comparison_df.to_dict(orient="records"),
        # Skipped for the demo — "St. Jude Community Hospital" is fictional,
        # a real registry check would just show "not found" and confuse people.
        "npi_verification": {"checked": False, "found": False, "npi": None, "matched_name": None, "address": None},
    }


@app.get("/api/audit/{audit_id}")
async def get_audit(audit_id: str):
    data = audit_store.load_audit(audit_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Audit not found.")
    return data


# --------------------------------------------------------------------------
# Letters
# --------------------------------------------------------------------------


@app.post("/api/letter")
async def create_letter(payload: dict):
    client = _client()
    try:
        letter = logic.generate_dispute_letter(
            client,
            payload.get("sender_name", ""),
            payload.get("sender_address", ""),
            payload.get("sender_city_state_zip", ""),
            payload.get("provider_name", ""),
            payload.get("provider_address", ""),
            payload.get("patient_name", ""),
            payload.get("account_number", ""),
            payload.get("bill_date", ""),
            payload.get("letter_date", ""),
            payload.get("disputed_rows", []),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    audit_id = payload.get("audit_id")
    disputed_rows = payload.get("disputed_rows", [])
    if audit_id:
        existing = audit_store.load_audit(audit_id) or {}
        existing["letter"] = letter
        existing["sender_name"] = payload.get("sender_name", "")
        existing["sender_address"] = payload.get("sender_address", "")
        existing["sender_city_state_zip"] = payload.get("sender_city_state_zip", "")
        existing["bill_date"] = payload.get("bill_date", "")
        existing["letter_date"] = payload.get("letter_date", "")
        existing["disputed_rows"] = disputed_rows
        audit_store.save_audit(audit_id, existing)

    return {"letter": logic.render_letter_display_text(letter, disputed_rows)}


@app.post("/api/save-addon-info")
async def save_addon_info(payload: dict):
    """Store optional insurer details (for the Insurance Appeal Letter add-on)
    against an audit before checkout, so the webhook can use them once
    payment is confirmed."""
    audit_id = payload.get("audit_id", "")
    if not audit_id:
        raise HTTPException(status_code=400, detail="audit_id is required.")
    existing = audit_store.load_audit(audit_id) or {}
    existing["insurer_name"] = payload.get("insurer_name", "")
    existing["member_id"] = payload.get("member_id", "")
    existing["claim_number"] = payload.get("claim_number", "")
    audit_store.save_audit(audit_id, existing)
    return {"status": "ok"}


@app.post("/api/followup")
async def create_followup(payload: dict):
    client = _client()
    disputed_rows = payload.get("disputed_rows", [])
    letter = logic.generate_followup_letter(client, payload.get("original_letter", ""), payload.get("original_date", ""), disputed_rows)
    return {"letter": letter}


@app.post("/api/insurance-appeal")
async def create_insurance_appeal(payload: dict):
    client = _client()
    disputed_rows = payload.get("disputed_rows", [])
    letter = logic.generate_insurance_appeal_letter(
        client,
        payload.get("sender_name", ""),
        payload.get("sender_address", ""),
        payload.get("sender_city_state_zip", ""),
        payload.get("insurer_name", ""),
        payload.get("member_id", ""),
        payload.get("claim_number", ""),
        payload.get("patient_name", ""),
        disputed_rows,
    )
    return {"letter": logic.render_letter_display_text(letter, disputed_rows)}


@app.post("/api/phone-script")
async def create_phone_script(payload: dict):
    client = _client()
    script = logic.generate_phone_script(
        client, payload.get("patient_name", ""), payload.get("account_number", ""), payload.get("disputed_rows", [])
    )
    return {"letter": script}


def _generate_addon_pdf(kind: str, audit: dict) -> bytes:
    """Generate one purchased add-on document (phone script, follow-up
    letter, or insurance appeal letter) from stored audit data."""
    client = logic.get_client()
    disputed_rows = audit.get("disputed_rows") or []
    extracted = audit.get("extracted") or {}
    patient_name = extracted.get("patient_name", "")
    if kind == "phone":
        text = logic.generate_phone_script(client, patient_name, extracted.get("account_number", ""), disputed_rows)
        return logic.generate_pdf_bytes(text)
    elif kind == "followup":
        text = logic.generate_followup_letter(client, audit.get("letter", ""), audit.get("letter_date", ""), disputed_rows)
        return logic.generate_pdf_bytes(text)
    elif kind == "insurance":
        text = logic.generate_insurance_appeal_letter(
            client,
            audit.get("sender_name", ""),
            audit.get("sender_address", ""),
            audit.get("sender_city_state_zip", ""),
            audit.get("insurer_name", ""),
            audit.get("member_id", ""),
            audit.get("claim_number", ""),
            patient_name,
            disputed_rows,
        )
        return logic.generate_pdf_bytes(text, disputed_rows)
    else:
        raise ValueError(f"Unknown add-on kind: {kind}")


@app.post("/api/pdf")
async def create_pdf(payload: dict):
    audit_id = payload.get("audit_id", "")
    kind = payload.get("kind", "letter")
    if not audit_id:
        raise HTTPException(status_code=400, detail="audit_id is required.")

    # Server-side payment check — this is the actual paywall. The letter
    # text is also pulled from our own stored audit, never trusted from the
    # client, so there's nothing for a visitor to fake or bypass client-side.
    paid, _plan, addons, _customer_email, _email_sent = audit_store.is_paid(audit_id)
    if not paid:
        raise HTTPException(status_code=403, detail="Payment required before the PDF can be downloaded.")

    audit = audit_store.load_audit(audit_id) or {}

    if kind == "letter":
        letter_text = audit.get("letter", "")
        if not letter_text:
            raise HTTPException(status_code=404, detail="No letter found for this audit.")
        pdf_bytes = logic.generate_pdf_bytes(letter_text, audit.get("disputed_rows"))
    else:
        if kind not in addons:
            raise HTTPException(status_code=403, detail="This add-on wasn't purchased for this audit.")
        pdf_bytes = _generate_addon_pdf(kind, audit)

    return Response(content=pdf_bytes, media_type="application/pdf")


# --------------------------------------------------------------------------
# Paddle webhook (real, server-verified payment confirmation)
# --------------------------------------------------------------------------

import hashlib
import hmac

from fastapi import Header, Request

PADDLE_WEBHOOK_SECRET = os.getenv("PADDLE_WEBHOOK_SECRET", "")
PRICE_ID_PRO = os.getenv("PADDLE_PRICE_PRO", "")
PRICE_ID_INSURANCE_APPEAL = os.getenv("PADDLE_PRICE_INSURANCE_APPEAL", "")
PRICE_ID_PHONE_SCRIPT = os.getenv("PADDLE_PRICE_PHONE_SCRIPT", "")
PRICE_ID_FOLLOWUP = os.getenv("PADDLE_PRICE_FOLLOWUP", "")


def verify_paddle_signature(raw_body: bytes, signature_header: str) -> bool:
    if not PADDLE_WEBHOOK_SECRET:
        return False
    try:
        parts = dict(p.split("=", 1) for p in signature_header.split(";"))
        ts, h1 = parts["ts"], parts["h1"]
    except Exception:
        return False
    signed_payload = f"{ts}:{raw_body.decode('utf-8')}"
    expected = hmac.new(PADDLE_WEBHOOK_SECRET.encode(), signed_payload.encode(), hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, h1)


@app.post("/api/paddle-webhook")
async def paddle_webhook(request: Request, paddle_signature: str = Header(default="")):
    raw_body = await request.body()
    if not verify_paddle_signature(raw_body, paddle_signature):
        raise HTTPException(status_code=401, detail="Invalid Paddle signature")

    event = json.loads(raw_body)
    if event.get("event_type") == "transaction.completed":
        data = event.get("data", {})
        custom_data = data.get("custom_data") or {}
        audit_id = custom_data.get("audit_id")
        price_ids = [item.get("price", {}).get("id") for item in data.get("items", [])]
        plan = "pro" if PRICE_ID_PRO and PRICE_ID_PRO in price_ids else "standard"
        if plan == "pro":
            # Pro is a flat-price bundle that always includes all three add-ons.
            addons = ["insurance", "phone", "followup"]
        else:
            addons = []
            if PRICE_ID_INSURANCE_APPEAL and PRICE_ID_INSURANCE_APPEAL in price_ids:
                addons.append("insurance")
            if PRICE_ID_PHONE_SCRIPT and PRICE_ID_PHONE_SCRIPT in price_ids:
                addons.append("phone")
            if PRICE_ID_FOLLOWUP and PRICE_ID_FOLLOWUP in price_ids:
                addons.append("followup")
        if audit_id:
            customer_id = data.get("customer_id", "")
            customer_email = logic.get_paddle_customer_email(customer_id) if customer_id else ""

            # Idempotency guard: Paddle can (and sometimes does) redeliver the
            # same event. Don't re-send the email if we already sent it once.
            already_paid, _prev_plan, _prev_addons, _prev_email, already_sent = audit_store.is_paid(audit_id)
            audit_store.mark_paid(audit_id, plan, addons, customer_email)

            if customer_email and not already_sent:
                audit = audit_store.load_audit(audit_id)
                letter_text = (audit or {}).get("letter")
                if letter_text:
                    pdf_bytes = logic.generate_pdf_bytes(letter_text, audit.get("disputed_rows"))
                    patient_name = (audit.get("extracted") or {}).get("patient_name", "")
                    extra_attachments = []
                    filenames = {"phone": "phone_negotiation_script.pdf", "followup": "followup_letter.pdf", "insurance": "insurance_appeal_letter.pdf"}
                    for kind, filename in filenames.items():
                        if kind in addons:
                            try:
                                extra_attachments.append({"filename": filename, "content": _generate_addon_pdf(kind, audit)})
                            except Exception:
                                pass  # don't let one add-on failure block the main letter email
                    sent = logic.send_letter_email(customer_email, patient_name, pdf_bytes, extra_attachments)
                    if sent:
                        audit_store.mark_email_sent(audit_id)

    return {"status": "ok"}


@app.get("/api/payment-status/{audit_id}")
async def payment_status(audit_id: str):
    paid, plan, addons, customer_email, email_sent = audit_store.is_paid(audit_id)
    return {
        "paid": paid,
        "plan": plan,
        "addons": addons,
        "customer_email": customer_email,
        "email_sent": email_sent,
    }


# --------------------------------------------------------------------------
# Serve the frontend
# --------------------------------------------------------------------------

_frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.isdir(_frontend_dir):
    app.mount("/", StaticFiles(directory=_frontend_dir, html=True), name="frontend")
