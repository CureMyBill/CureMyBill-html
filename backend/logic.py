"""
logic.py — All the "brains" of CureMyBill: bill extraction, Medicare
comparison, letter drafting, and PDF generation. No web-framework code here
on purpose, so it can be called from FastAPI, a CLI, tests, or anything else.
"""

import base64
import io
import os
import re
from xml.sax.saxutils import escape as xml_escape

import pandas as pd
import requests
from anthropic import Anthropic
from reportlab.lib.pagesizes import letter as LETTER_PAGESIZE
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

MODEL = "claude-sonnet-5"
FEE_SCHEDULE_PATH = os.path.join(os.path.dirname(__file__), "fee_schedule.csv")
NPI_REGISTRY_URL = "https://npiregistry.cms.hhs.gov/api/"
TURNSTILE_VERIFY_URL = "https://challenges.cloudflare.com/turnstile/v0/siteverify"


def get_client() -> Anthropic:
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set on the server.")
    return Anthropic(api_key=api_key)


def load_fee_schedule() -> pd.DataFrame:
    df = pd.read_csv(FEE_SCHEDULE_PATH, dtype={"cpt_code": str})
    df["cpt_code"] = df["cpt_code"].str.strip().str.upper()
    return df


def bytes_to_content_block(raw_bytes: bytes, filename: str, mime_type: str = "") -> dict:
    """Turn raw uploaded file bytes into an Anthropic API content block."""
    b64 = base64.b64encode(raw_bytes).decode("utf-8")

    if mime_type == "application/pdf" or filename.lower().endswith(".pdf"):
        return {
            "type": "document",
            "source": {"type": "base64", "media_type": "application/pdf", "data": b64},
        }
    else:
        if mime_type not in ("image/png", "image/jpeg", "image/webp", "image/gif"):
            mime_type = "image/png"
        return {
            "type": "image",
            "source": {"type": "base64", "media_type": mime_type, "data": b64},
        }


EXTRACTION_SYSTEM_PROMPT = """You are a medical billing data extraction assistant.
You will be shown a US medical bill (image or PDF). Extract every billable line
item you can find.

Respond with ONLY a valid JSON object, no markdown fences, no commentary,
matching exactly this schema:

{
  "provider_name": string or null,
  "provider_address": string or null,
  "patient_name": string or null,
  "bill_date": string or null,
  "account_number": string or null,
  "line_items": [
    {
      "cpt_code": string or null,
      "description": string,
      "quantity": number,
      "billed_amount": number
    }
  ],
  "total_billed": number or null
}

Rules:
- account_number is the account, claim, invoice, or reference number printed
  on the bill (labelled things like "Account #", "Claim Number", "Invoice No",
  "Patient Account"), or null if none is visible.
- provider_address is the hospital/clinic's mailing or billing address as
  printed on the bill, or null if not visible.
- cpt_code should be the 5-character CPT/HCPCS code exactly as printed (letters
  uppercase), or null if none is visible for that line.
- billed_amount is the amount charged for that line item in US dollars, as a
  plain number (no $ sign, no commas).
- If a field is not present on the bill, use null.
- Do not invent CPT codes that are not shown or clearly implied by the bill.
- Output must be valid JSON and nothing else.
"""


def extract_bill_data(client: Anthropic, file_block: dict) -> dict:
    import json

    response = client.messages.create(
        model=MODEL,
        max_tokens=2000,
        system=EXTRACTION_SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": [
                    file_block,
                    {"type": "text", "text": "Extract the billing data from this medical bill as JSON."},
                ],
            }
        ],
    )
    text = "".join(block.text for block in response.content if block.type == "text").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    return json.loads(text)


def compare_to_schedule(line_items: list, fee_schedule: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for item in line_items:
        cpt = (item.get("cpt_code") or "").strip().upper()
        billed = item.get("billed_amount") or 0
        qty = item.get("quantity") or 1
        match = fee_schedule[fee_schedule["cpt_code"] == cpt]

        if not match.empty:
            standard = float(match.iloc[0]["standard_price"])
            diff = billed - standard
            pct = (diff / standard * 100) if standard else None
            ratio = (billed / standard) if standard else None
            if ratio is None:
                status = "No reference"
            elif ratio <= 5:
                status = "Typical range"
            elif ratio <= 10:
                status = "Above typical markup"
            else:
                status = "Well above typical — worth disputing"
        else:
            standard, diff, pct, status = None, None, None, "No reference"

        rows.append(
            {
                "cpt_code": cpt or "—",
                "description": item.get("description", ""),
                "quantity": qty,
                "billed": billed,
                "medicare_rate": standard,
                "difference": diff,
                "difference_pct": round(pct, 1) if pct is not None else None,
                "status": status,
            }
        )
    return pd.DataFrame(rows)


LETTER_SYSTEM_PROMPT = """You are an expert patient-advocacy assistant who writes
clear, firm, professional medical bill dispute letters on behalf of US patients.

Write in formal English business-letter style. Be factual and assertive but
polite. Cite the specific CPT codes and dollar amounts provided, comparing the
billed amount to the official Medicare national reimbursement rate for that
code. Frame this as "billed at X times the Medicare rate" rather than
asserting the hospital committed fraud or overcharged unfairly — Medicare
rates are a documented public benchmark, not a legal ceiling on what a
provider may charge, so the letter should request justification and an
itemized review rather than assert wrongdoing. Ask explicitly for an itemized
re-review, a corrected bill, and a written response within 30 days. Do not
invent facts beyond what is given. Keep it to one page.
"""


def generate_dispute_letter(
    client: Anthropic,
    sender_name: str,
    sender_address: str,
    sender_city_state_zip: str,
    provider_name: str,
    provider_address: str,
    patient_name: str,
    account_number: str,
    bill_date: str,
    letter_date: str,
    disputed_rows: list,
) -> str:
    def _val(v, placeholder):
        v = (v or "").strip()
        return v if v else placeholder

    items_text = "\n".join(
        f"- CPT {row['cpt_code']}: {row['description']} — billed ${row['billed']:.2f}, "
        f"Medicare national rate ${row['medicare_rate']:.2f} "
        f"(+{row['difference_pct']}% above the Medicare rate)"
        for row in disputed_rows
    )

    user_prompt = f"""Write a medical bill dispute letter using these EXACT details.
Do not use bracket placeholders for any field listed below — use the real
value given. Only use a bracket placeholder (e.g. [Account/Claim Number]) for
a field that is explicitly marked as "not provided".

Sender (the person sending this letter):
Name: {_val(sender_name, "[Your Name]")}
Address: {_val(sender_address, "[Your Address] — not provided")}
City/State/ZIP: {_val(sender_city_state_zip, "[City, State ZIP] — not provided")}
Letter date: {_val(letter_date, "[Date]")}

Recipient / provider:
Provider name: {_val(provider_name, "[Provider Name] — not provided")}
Provider address: {_val(provider_address, "[Provider Address] — not provided")}

Patient and account details:
Patient name: {_val(patient_name, "[Patient Name] — not provided")}
Account/Claim number: {_val(account_number, "[Account/Claim Number] — not provided")}
Bill date: {_val(bill_date, "[Bill Date] — not provided")}

Disputed line items (billed amount vs. the official Medicare national reimbursement rate for that CPT code):
{items_text}

The letter should request an itemized review, ask the provider to justify the
charges against standard pricing benchmarks (e.g. Medicare or regional rates),
and request a corrected invoice or written explanation within 30 days.
Format it as a ready-to-print business letter: sender block, date, recipient
block, subject line, body, closing, and signature line with the sender's name.
"""

    response = client.messages.create(
        model=MODEL,
        max_tokens=1500,
        system=LETTER_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


FOLLOWUP_SYSTEM_PROMPT = """You are an expert patient-advocacy assistant writing a
FOLLOW-UP letter because the original dispute letter received no response within
30 days. Keep it formal but firmer in tone: reference the original letter and its
date, note that no response was received within the requested 30-day window, and
escalate the requested next steps — mention that the patient may file a complaint
with the state insurance commissioner, the hospital's patient advocacy or
compliance office, and/or the Consumer Financial Protection Bureau if billing
collections are involved. Do not threaten legal action explicitly or invent facts.
Keep it to one page.
"""


def generate_followup_letter(client: Anthropic, original_letter: str, original_date: str) -> str:
    user_prompt = f"""Here is the original dispute letter, sent on {original_date or "[original date]"},
which has not received a response within 30 days:

---
{original_letter}
---

Write a follow-up letter referencing this original letter, its date, and the
lack of response, escalating the request for resolution as described in your
instructions.
"""
    response = client.messages.create(
        model=MODEL,
        max_tokens=1200,
        system=FOLLOWUP_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


INSURANCE_APPEAL_SYSTEM_PROMPT = """You are an expert patient-advocacy assistant
writing a FORMAL APPEAL LETTER to a health insurance company (not the hospital).
This is used when the hospital's charges are high partly because the insurer
denied or underpaid a claim. Write in formal English business-letter style.
Reference the patient's policy/member ID and claim number when given. Ask the
insurer to reprocess the claim, provide a written explanation for any denial
or reduced payment, and clarify in-network/out-of-network status if relevant.
Mention the patient's right to a formal internal appeal and, if unresolved, an
external review, without asserting specific legal violations. Do not invent
facts beyond what is given. Keep it to one page.
"""


def generate_insurance_appeal_letter(
    client: Anthropic,
    sender_name: str,
    sender_address: str,
    sender_city_state_zip: str,
    insurer_name: str,
    member_id: str,
    claim_number: str,
    patient_name: str,
    disputed_rows: list,
) -> str:
    def _val(v, placeholder):
        v = (v or "").strip()
        return v if v else placeholder

    items_text = "\n".join(
        f"- CPT {row['cpt_code']}: {row['description']} — billed ${row['billed']:.2f}, "
        f"Medicare national rate ${row['medicare_rate']:.2f}"
        for row in disputed_rows
    )

    user_prompt = f"""Write an insurance appeal letter using these details:

Sender: {_val(sender_name, "[Your Name]")}
Address: {_val(sender_address, "[Your Address]")}
City/State/ZIP: {_val(sender_city_state_zip, "[City, State ZIP]")}

Insurance company: {_val(insurer_name, "[Insurance Company Name]")}
Member/Policy ID: {_val(member_id, "[Member ID]")}
Claim number: {_val(claim_number, "[Claim Number]")}
Patient name: {_val(patient_name, "[Patient Name]")}

Disputed / underpaid line items:
{items_text}

Request reprocessing of the claim, a written explanation of any denial or
reduced payment, and clarification of network status if relevant.
"""
    response = client.messages.create(
        model=MODEL,
        max_tokens=1200,
        system=INSURANCE_APPEAL_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


PHONE_SCRIPT_SYSTEM_PROMPT = """You are an expert patient-advocacy assistant
writing a SHORT, PRACTICAL PHONE SCRIPT (not a letter) that a patient can read
almost word-for-word when calling a hospital billing department. Plain,
conversational English, organized in clear steps: opening the call, stating
the purpose, citing the specific overbilled line items, asking about a
self-pay/cash discount or financial assistance program, and what to say if
the representative pushes back. Keep it concise and actionable — this is a
cheat sheet, not a formal document. Do not invent facts beyond what is given.
"""


def generate_phone_script(client: Anthropic, patient_name: str, account_number: str, disputed_rows: list) -> str:
    items_text = "\n".join(
        f"- CPT {row['cpt_code']}: {row['description']} — billed ${row['billed']:.2f}, "
        f"Medicare national rate ${row['medicare_rate']:.2f}"
        for row in disputed_rows
    )
    user_prompt = f"""Write a phone negotiation script for this patient calling
the hospital billing department:

Patient name: {patient_name or "[Patient Name]"}
Account number: {account_number or "[Account Number]"}

Overbilled line items to mention:
{items_text}

Include a line asking about a self-pay/cash discount and any financial
assistance / charity care program the hospital may offer.
"""
    response = client.messages.create(
        model=MODEL,
        max_tokens=900,
        system=PHONE_SCRIPT_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


_LETTER_CLOSINGS = ("sincerely", "regards", "respectfully", "best regards", "yours truly")


def generate_pdf_bytes(letter_text: str) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=LETTER_PAGESIZE,
        topMargin=1 * inch,
        bottomMargin=1 * inch,
        leftMargin=1 * inch,
        rightMargin=1 * inch,
        title="Medical Bill Dispute Letter",
    )
    styles = getSampleStyleSheet()
    body_style = ParagraphStyle(
        "LetterBody", parent=styles["Normal"], fontName="Times-Roman",
        fontSize=11, leading=16, spaceAfter=12,
    )

    story = []
    paragraphs = [p.strip() for p in letter_text.strip().split("\n\n") if p.strip()]
    for para in paragraphs:
        safe_html = xml_escape(para).replace("\n", "<br/>")
        story.append(Paragraph(safe_html, body_style))
        first_line = para.strip().split("\n")[0].strip().lower().rstrip(",")
        if first_line in _LETTER_CLOSINGS:
            story.append(Spacer(1, 50))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


_US_STATE_ABBR_RE = re.compile(r"\b([A-Z]{2})\b\s+\d{5}")


def verify_provider_npi(provider_name: str, provider_address: str = "") -> dict:
    """Check the hospital/provider name against the official CMS NPI Registry
    (a free, public, keyless government API). This confirms the name exists
    in that registry — it is NOT a guarantee the provider is currently
    licensed or in good standing, and the phrasing used anywhere with this
    result should stay modest and factual for that reason.

    Fails silently (returns not-found) on any network/parsing issue, since
    this is a "nice to have" trust signal and must never block the core
    bill-analysis flow.
    """
    result = {"checked": True, "found": False, "npi": None, "matched_name": None, "address": None}

    if not provider_name or not provider_name.strip():
        result["checked"] = False
        return result

    state_match = _US_STATE_ABBR_RE.search(provider_address or "")
    params = {
        "organization_name": provider_name.strip(),
        "enumeration_type": "NPI-2",
        "version": "2.1",
        "limit": 5,
    }
    if state_match:
        params["state"] = state_match.group(1)

    try:
        resp = requests.get(NPI_REGISTRY_URL, params=params, timeout=8)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        if results:
            top = results[0]
            basic = top.get("basic", {})
            addresses = top.get("addresses", [])
            addr = addresses[0] if addresses else {}
            result["found"] = True
            result["npi"] = top.get("number")
            result["matched_name"] = basic.get("organization_name") or basic.get("name")
            if addr:
                result["address"] = ", ".join(
                    filter(None, [addr.get("address_1"), addr.get("city"), addr.get("state"), addr.get("postal_code")])
                )
    except Exception:
        pass  # Non-critical enhancement — never break bill analysis over this.

    return result


def verify_turnstile(token: str, remote_ip: str = None) -> bool:
    """Verify a Cloudflare Turnstile token server-side (mandatory per Cloudflare's
    own docs — the client-side widget alone proves nothing).

    Fails CLOSED (returns False) on a missing token or a verification/network
    error, since this specifically guards a paid Claude API call from bot abuse.
    The one exception: if TURNSTILE_SECRET_KEY isn't configured yet at all,
    this fails OPEN (returns True) so the app doesn't break before setup is
    finished — once the key is set, verification is enforced normally.
    """
    secret = os.getenv("TURNSTILE_SECRET_KEY", "")
    if not secret:
        return True  # Not configured yet — don't block real users over it.

    if not token:
        return False

    try:
        data = {"secret": secret, "response": token}
        if remote_ip:
            data["remoteip"] = remote_ip
        resp = requests.post(TURNSTILE_VERIFY_URL, data=data, timeout=8)
        resp.raise_for_status()
        result = resp.json()
        return bool(result.get("success"))
    except Exception:
        return False
