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
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle, HRFlowable

MODEL = "claude-sonnet-5"
FEE_SCHEDULE_PATH = os.path.join(os.path.dirname(__file__), "fee_schedule.csv")
NPI_REGISTRY_URL = "https://npiregistry.cms.hhs.gov/api/"
TURNSTILE_VERIFY_URL = "https://challenges.cloudflare.com/turnstile/v0/siteverify"
RESEND_API_URL = "https://api.resend.com/emails"
PADDLE_API_BASE = "https://sandbox-api.paddle.com"  # Sandbox — switch to api.paddle.com when going Live

# A letter can't hand the AI model responsibility for laying out a clean
# table of dollar amounts — LLM-written "tables" in prose render as an
# unaligned wall of text once wrapped into a PDF paragraph. Instead, the
# model is instructed to place this exact token where the itemized charges
# belong, and generate_pdf_bytes swaps it for a real ReportLab Table.
ITEMIZED_TABLE_TOKEN = "{{ITEMIZED_TABLE}}"


def _items_plain_text(disputed_rows: list) -> str:
    """Plain-text rendering of the disputed line items — used for on-screen
    preview and as context text fed back into later prompts (e.g. the
    follow-up letter), never for the PDF itself (which uses a real table)."""
    return "\n".join(
        f"- CPT {row['cpt_code']}: {row['description']} — billed ${row['billed']:.2f}, "
        f"Medicare national rate ${row['medicare_rate']:.2f}"
        + (f" (+{row['difference_pct']}% above the Medicare rate)" if "difference_pct" in row else "")
        for row in disputed_rows
    )


def render_letter_display_text(letter_text: str, disputed_rows: list) -> str:
    """Replace the itemized-table token with a readable plain-text list, for
    contexts that aren't the PDF (the on-screen preview, and feeding this
    letter as context into a later prompt like the follow-up letter)."""
    if not letter_text:
        return letter_text
    replacement = _items_plain_text(disputed_rows) if disputed_rows else ""
    return letter_text.replace(ITEMIZED_TABLE_TOKEN, replacement)


def _build_items_table(disputed_rows: list) -> Table:
    header_style = ParagraphStyle("TableHeader", fontName="Helvetica-Bold", fontSize=8.5, textColor=colors.white, leading=11)
    cell_style = ParagraphStyle("TableCell", fontName="Helvetica", fontSize=8.5, leading=11)
    cell_style_right = ParagraphStyle("TableCellRight", parent=cell_style, alignment=2)

    header = [
        Paragraph("CPT Code", header_style),
        Paragraph("Description", header_style),
        Paragraph("Billed", header_style),
        Paragraph("Medicare Rate", header_style),
        Paragraph("Over By", header_style),
    ]
    data = [header]
    for row in disputed_rows:
        over_by = f"+{row['difference_pct']}%" if "difference_pct" in row else ""
        data.append([
            Paragraph(xml_escape(str(row.get("cpt_code", ""))), cell_style),
            Paragraph(xml_escape(str(row.get("description", ""))), cell_style),
            Paragraph(f"${row['billed']:.2f}", cell_style_right),
            Paragraph(f"${row['medicare_rate']:.2f}", cell_style_right),
            Paragraph(over_by, cell_style_right),
        ])

    col_widths = [0.85 * inch, 2.55 * inch, 0.95 * inch, 1.15 * inch, 1.0 * inch]
    table = Table(data, colWidths=col_widths, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#23281F")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F0E3")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E1D9C4")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    return table


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

Disputed line items (for your reference only — billed amount vs. the official
Medicare national reimbursement rate for that CPT code):
{items_text}

Do NOT reproduce these line items yourself as a list or table in the letter.
Instead, write one short sentence introducing that the following charges are
being disputed, then insert the exact token {ITEMIZED_TABLE_TOKEN} alone on
its own line immediately after that sentence — a formatted table will be
inserted there automatically. Continue the letter normally after the token.

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


def generate_followup_letter(client: Anthropic, original_letter: str, original_date: str, disputed_rows: list = None) -> str:
    clean_original = render_letter_display_text(original_letter, disputed_rows or [])
    user_prompt = f"""Here is the original dispute letter, sent on {original_date or "[original date]"},
which has not received a response within 30 days:

---
{clean_original}
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
    letter_date: str = "",
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
Letter date: {_val(letter_date, "[Date]")}

Insurance company: {_val(insurer_name, "[Insurance Company Name]")}
Member/Policy ID: {_val(member_id, "[Member ID]")}
Claim number: {_val(claim_number, "[Claim Number]")}
Patient name: {_val(patient_name, "[Patient Name]")}

Disputed / underpaid line items (for your reference only):
{items_text}

Do NOT reproduce these line items yourself as a list or table in the letter.
Instead, write one short sentence introducing that the following claims/
charges are being disputed, then insert the exact token {ITEMIZED_TABLE_TOKEN}
alone on its own line immediately after that sentence — a formatted table
will be inserted there automatically. Continue the letter normally after the
token.

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

This will be rendered as plain text in a PDF, not displayed as Markdown/HTML.
Do NOT use any Markdown syntax: no #, ##, or ### headers, no ** or * for bold
or italic, no --- or *** dividers, and no - or * bullet markers. For section
titles, just write "1. Opening the call" etc. as plain text on its own line.
For emphasis, use quotation marks or plain wording instead of bold/italic.
For a list of items, write each on its own line starting with the item name
directly (no dash or bullet character needed).
"""

PHONE_SCRIPT_BILINGUAL_SYSTEM_PROMPT = """You are an expert patient-advocacy
assistant writing a SHORT, PRACTICAL BILINGUAL PHONE SCRIPT for a Spanish-
speaking patient calling a US hospital billing department, where staff most
likely speak English. This is a free companion to the English-only script —
its purpose is to help the patient understand exactly what they're saying,
even if their English isn't strong.

Start with an opening line asking whether someone who speaks Spanish is
available, or whether an interpreter line can be used — in English first
(exactly as the patient should say it), then its Spanish translation on the
next line in parentheses. Continue this same pattern for every single spoken
line in the script: English line first, Spanish translation directly below
it in parentheses. Cover the same steps as a normal negotiation call: stating
the purpose, citing the specific overbilled line items, asking about a
self-pay/cash discount or financial assistance/charity care program, and what
to say if the representative pushes back. Keep it concise and actionable —
this is a cheat sheet, not a formal document. Do not invent facts beyond what
is given.

This will be rendered as plain text in a PDF, not displayed as Markdown/HTML.
Do NOT use any Markdown syntax: no #, ##, or ### headers, no ** or * for bold
or italic, no --- or *** dividers, and no - or * bullet markers. For section
titles, just write "1. Opening the call" etc. as plain text on its own line.
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


def generate_phone_script_bilingual(client: Anthropic, patient_name: str, account_number: str, disputed_rows: list) -> str:
    """Free companion document to generate_phone_script: the same call,
    with an English/Spanish line for every spoken part, so a Spanish-
    speaking patient understands what they're saying on an English-language
    call. Delivered as a separate PDF — the English-only script is unchanged."""
    items_text = "\n".join(
        f"- CPT {row['cpt_code']}: {row['description']} — billed ${row['billed']:.2f}, "
        f"Medicare national rate ${row['medicare_rate']:.2f}"
        for row in disputed_rows
    )
    user_prompt = f"""Write a bilingual (English/Spanish) phone negotiation
script for this patient calling the hospital billing department:

Patient name: {patient_name or "[Patient Name]"}
Account number: {account_number or "[Account Number]"}

Overbilled line items to mention:
{items_text}

Include a line asking about a self-pay/cash discount and any financial
assistance / charity care program the hospital may offer.
"""
    response = client.messages.create(
        model=MODEL,
        max_tokens=1300,
        system=PHONE_SCRIPT_BILINGUAL_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


_LETTER_CLOSINGS = ("sincerely", "regards", "respectfully", "best regards", "yours truly")

# Safety net: the model is instructed not to use Markdown, but if it slips
# (as LLMs occasionally do, especially for the more casual phone script),
# this converts common Markdown syntax to real PDF formatting instead of
# letting literal #, **, and --- characters show up in the document.
_MD_HR_RE = re.compile(r"^(-{3,}|\*{3,}|_{3,})$")
_MD_HEADER_RE = re.compile(r"^#{1,6}\s*")
_MD_BULLET_RE = re.compile(r"^[\-\*]\s+")
_MD_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_MD_ITALIC_RE = re.compile(r"(?<!\*)\*([^*]+?)\*(?!\*)")


def _markdown_line_to_html(line: str):
    """Convert one line of (possibly Markdown) text to safe Paragraph HTML.
    Returns None for lines that should be dropped entirely (e.g. a bare
    '---' divider)."""
    stripped = line.strip()
    if _MD_HR_RE.match(stripped):
        return None
    is_header = bool(_MD_HEADER_RE.match(stripped))
    text = _MD_HEADER_RE.sub("", stripped)
    text = _MD_BULLET_RE.sub("• ", text)
    escaped = xml_escape(text)
    escaped = _MD_BOLD_RE.sub(r"<b>\1</b>", escaped)
    escaped = _MD_ITALIC_RE.sub(r"<i>\1</i>", escaped)
    return f"<b>{escaped}</b>" if is_header else escaped


# Visual style lifted from the approved design mockup — colors, fonts, and
# layout only. None of this touches what the letters say or how bills are
# analyzed; it only changes how the existing text is laid out on the page.
_DOC_INK = colors.HexColor("#23281F")
_DOC_MUTED = colors.HexColor("#4A5044")
_DOC_ACCENT = colors.HexColor("#7A4A1E")
_DOC_LINE = colors.HexColor("#E1D9C4")
_DOC_FOOTER_GRAY = colors.HexColor("#8A8470")
_DOC_SIG_LINE = colors.HexColor("#B9B29D")
_DOC_LABEL_ROW = colors.HexColor("#6E6857")


class _LetterheadCanvas(canvas.Canvas):
    """Buffers pages so the footer can show 'Page X of Y' (reportlab doesn't
    know the total page count until every page has been drawn)."""

    def __init__(self, *args, **kwargs):
        self._header_kwargs = kwargs.pop("header_kwargs")
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        total_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self._draw_letterhead(total_pages)
            super().showPage()
        super().save()

    def _draw_letterhead(self, total_pages):
        hk = self._header_kwargs
        width, height = LETTER_PAGESIZE
        top = height - 0.8 * inch

        self.setFont("Times-Roman", 15)
        self.setFillColor(_DOC_INK)
        self.drawString(0.9 * inch, top, hk["title"])

        self.setFont("Times-Roman", 8.5)
        self.setFillColor(_DOC_MUTED)
        right_lines = hk.get("right_lines") or []
        ry = top + 2
        for line in reversed(right_lines):
            self.drawRightString(width - 0.9 * inch, ry, line)
            ry += 11

        self.setStrokeColor(_DOC_INK)
        self.setLineWidth(1.3)
        self.line(0.9 * inch, top - 8, width - 0.9 * inch, top - 8)

        self.setFont("Helvetica", 8)
        self.setFillColor(_DOC_ACCENT)
        self.drawString(0.9 * inch, top - 22, hk.get("kind_label", "").upper())

        self.setFont("Times-Roman", 8)
        self.setFillColor(_DOC_FOOTER_GRAY)
        self.drawCentredString(width / 2, 0.6 * inch, f"Page {self._pageNumber} of {total_pages}")
        self.setStrokeColor(_DOC_LINE)
        self.setLineWidth(0.75)
        self.line(0.9 * inch, 0.72 * inch, width - 0.9 * inch, 0.72 * inch)


def generate_pdf_bytes(
    letter_text: str,
    disputed_rows: list = None,
    header_title: str = "",
    header_kind_label: str = "",
    header_right_lines: list = None,
    summary_rows: list = None,
) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=LETTER_PAGESIZE,
        topMargin=1.35 * inch,
        bottomMargin=0.95 * inch,
        leftMargin=0.9 * inch,
        rightMargin=0.9 * inch,
        title=header_title or "Correspondence",
    )
    body_style = ParagraphStyle(
        "LetterBody", fontName="Times-Roman", fontSize=11, leading=17,
        spaceAfter=13, textColor=_DOC_INK,
    )

    def _paragraphs_for(text_block: str) -> list:
        flowables = []
        for para in (p.strip() for p in text_block.strip().split("\n\n") if p.strip()):
            html_lines = [h for h in (_markdown_line_to_html(l) for l in para.split("\n")) if h is not None]
            if not html_lines:
                continue
            safe_html = "<br/>".join(html_lines)
            flowables.append(Paragraph(safe_html, body_style))
            first_line = para.strip().split("\n")[0].strip().lower().rstrip(",")
            if first_line in _LETTER_CLOSINGS:
                flowables.append(HRFlowable(width=3.1 * inch, thickness=1, color=_DOC_SIG_LINE,
                                             spaceBefore=0.42 * inch, spaceAfter=6, hAlign="LEFT"))
        return flowables

    story = []
    if summary_rows:
        label_style = ParagraphStyle("SummaryLabel", fontName="Times-Roman", fontSize=10, textColor=_DOC_LABEL_ROW)
        value_style = ParagraphStyle("SummaryValue", fontName="Times-Roman", fontSize=10, textColor=_DOC_INK)
        rows = [[Paragraph(xml_escape(label), label_style), Paragraph(xml_escape(str(value)), value_style)]
                for label, value in summary_rows]
        t = Table(rows, colWidths=[1.6 * inch, 4.0 * inch], hAlign="LEFT")
        t.setStyle(TableStyle([
            ("TOPPADDING", (0, 0), (-1, -1), 1), ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ]))
        story.append(t)
        story.append(Spacer(1, 18))

    if disputed_rows and ITEMIZED_TABLE_TOKEN in letter_text:
        before, _, after = letter_text.partition(ITEMIZED_TABLE_TOKEN)
        story.extend(_paragraphs_for(before))
        story.append(Spacer(1, 4))
        story.append(_build_items_table(disputed_rows))
        story.append(Spacer(1, 16))
        story.extend(_paragraphs_for(after))
    else:
        # No table to insert (phone script, follow-up letter, or the model
        # didn't include the token) — render as plain paragraphs, and strip
        # a stray token if one slipped through with no rows to fill it.
        story.extend(_paragraphs_for(letter_text.replace(ITEMIZED_TABLE_TOKEN, "")))

    header_kwargs = {"title": header_title, "kind_label": header_kind_label, "right_lines": header_right_lines}

    def _make_canvas(*args, **kwargs):
        return _LetterheadCanvas(*args, header_kwargs=header_kwargs, **kwargs)

    doc.build(story, canvasmaker=_make_canvas)
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


def get_paddle_customer_email(customer_id: str) -> str:
    """Look up a customer's email from their Paddle customer_id. The webhook
    payload only ever includes the id, not the email itself, so this extra
    call is required. Returns "" on any failure — the caller should treat
    that as 'email unknown' rather than crash the webhook.
    """
    api_key = os.getenv("PADDLE_API_KEY", "")
    if not api_key or not customer_id:
        return ""
    try:
        resp = requests.get(
            f"{PADDLE_API_BASE}/customers/{customer_id}",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=8,
        )
        resp.raise_for_status()
        return resp.json().get("data", {}).get("email", "") or ""
    except Exception:
        return ""


def send_letter_email(to_email: str, patient_name: str, pdf_bytes: bytes, extra_attachments: list = None) -> bool:
    """Email the finished dispute letter (as a PDF attachment) via Resend,
    plus any purchased add-on documents (phone script, follow-up letter,
    insurance appeal letter) passed in extra_attachments as
    [{"filename": ..., "content": <pdf bytes>}, ...].
    Returns True only on a confirmed send — the caller decides what to do
    if this fails (e.g. still let the customer download it in-browser as a
    fallback, rather than leaving them with nothing).
    """
    api_key = os.getenv("RESEND_API_KEY", "")
    from_address = os.getenv("RESEND_FROM_ADDRESS", "CureMyBill <onboarding@resend.dev>")
    if not api_key or not to_email:
        return False

    extra_attachments = extra_attachments or []
    attached_kinds = {a["filename"] for a in extra_attachments}

    # Friendly name + one-line explanation for each possible attachment.
    ATTACHMENT_INFO = {
        "dispute_letter.pdf": ("Dispute Letter", "Addressed to your hospital's billing department — print and mail it."),
        "phone_negotiation_script.pdf": ("Phone Negotiation Script", "What to say when you call the billing department."),
        "phone_negotiation_script_bilingual.pdf": ("Bilingual Phone Script (free)", "Same call, with the Spanish translation for every line."),
        "followup_letter.pdf": ("Follow-up Letter", "Send this only if you don't hear back within 30 days."),
        "insurance_appeal_letter.pdf": ("Insurance Appeal Letter", "Addressed to your insurance company's claims department."),
    }
    all_filenames = ["dispute_letter.pdf"] + [a["filename"] for a in extra_attachments]
    attachments_html = "".join(
        f'<li style="margin-bottom:8px;"><strong>{xml_escape(ATTACHMENT_INFO[f][0])}</strong> — {xml_escape(ATTACHMENT_INFO[f][1])}</li>'
        for f in all_filenames if f in ATTACHMENT_INFO
    )

    # Next steps, built based on what was actually purchased.
    steps = ["Open the attached <strong>Dispute Letter</strong> and double-check your name, address, and the hospital's details.",
             "Print it and mail it to your hospital's billing department (the address is already in the letter)."]
    if "phone_negotiation_script.pdf" in attached_kinds:
        steps.append("Use the <strong>Phone Negotiation Script</strong> if you'd like to call the billing department directly.")
    if "insurance_appeal_letter.pdf" in attached_kinds:
        steps.append("Mail the <strong>Insurance Appeal Letter</strong> to your insurance company's claims department.")
    if "followup_letter.pdf" in attached_kinds:
        steps.append("If you don't hear back within 30 days, send the <strong>Follow-up Letter</strong> that's also attached.")
    steps_html = "".join(f'<li style="margin-bottom:10px;">{s}</li>' for s in steps)

    greeting_name = (patient_name or "").strip() or "there"
    html_body = f"""
    <div style="font-family: Georgia, 'Times New Roman', serif; background:#FAF6EA; padding:32px 16px;">
      <div style="max-width:520px; margin:0 auto; background:#FFFFFF; border-radius:10px; overflow:hidden; border:1px solid #E3D9BC;">
        <div style="background:#2C6E9E; padding:22px 28px;">
          <span style="font-family: Georgia, serif; font-size:1.3rem; font-weight:bold; color:#FFFFFF;">CureMyBill</span>
        </div>
        <div style="padding:28px; color:#1C2333; line-height:1.6; font-size:15px;">
          <h2 style="margin-top:0;">Your dispute letter is ready</h2>
          <p>Hi {xml_escape(greeting_name)},</p>
          <p>Here's what's attached to this email:</p>
          <ul style="padding-left:20px; margin:0 0 20px;">{attachments_html}</ul>
          <p style="font-weight:bold; margin-bottom:6px;">Next steps:</p>
          <ol style="padding-left:20px; margin:0 0 20px;">{steps_html}</ol>
          <p>— CureMyBill</p>
          <p style="font-size:12px; color:#6b7488; margin-top:28px; border-top:1px solid #E3D9BC; padding-top:14px;">
          CureMyBill is an automated document-assistance tool. It does not provide
          medical or legal advice.</p>
        </div>
      </div>
    </div>
    """
    subject = "Your CureMyBill Dispute Pack is ready" if extra_attachments else "Your CureMyBill dispute letter is ready"
    payload = {
        "from": from_address,
        "to": [to_email],
        "subject": subject,
        "html": html_body,
        "attachments": [
            {
                "filename": "dispute_letter.pdf",
                "content": base64.b64encode(pdf_bytes).decode("utf-8"),
            }
        ] + [
            {"filename": a["filename"], "content": base64.b64encode(a["content"]).decode("utf-8")}
            for a in extra_attachments
        ],
    }
    try:
        resp = requests.post(
            RESEND_API_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=15,
        )
        resp.raise_for_status()
        return True
    except Exception:
        return False
