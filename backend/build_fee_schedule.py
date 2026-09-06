"""
build_fee_schedule.py — Downloads the real, official CMS Medicare Physician
Fee Schedule (Relative Value File) and computes national payment amounts for
every active code, producing fee_schedule.csv.

This runs automatically on every Render deploy (see the build command in
render's service settings), because Render's servers have normal internet
access to cms.gov — unlike the sandbox this was developed in.

Formula (per CMS's own published documentation):
    National Non-Facility Payment = (Work RVU + Non-Facility PE RVU + MP RVU)
                                     × National Conversion Factor

This uses the NATIONAL average (no geographic/GPCI adjustment) — consistent
with how the rest of CureMyBill already frames Medicare rates: a single,
citable public benchmark, not a locality-specific figure.

Safety net: if anything about this script fails (CMS changes their file
format, network hiccup, etc.), it falls back to the last known-good
fee_schedule.csv instead of breaking the whole app's deployment.
"""

import io
import os
import sys
import zipfile

import pandas as pd
import requests

RVU_ZIP_URL = "https://www.cms.gov/files/zip/rvu26c-updated-06-30-2026.zip"
# CY2026 non-QP conversion factor (final rule, effective Jan 1 2026).
CONVERSION_FACTOR = 33.40

HERE = os.path.dirname(__file__)
OUTPUT_PATH = os.path.join(HERE, "fee_schedule.csv")
FALLBACK_PATH = os.path.join(HERE, "fee_schedule_fallback.csv")
EXTRA_PATH = os.path.join(HERE, "fee_schedule_extra.csv")  # labs + J-codes, not in the PFS


def log(msg: str) -> None:
    print(f"[build_fee_schedule] {msg}", flush=True)


def download_zip(url: str) -> zipfile.ZipFile:
    log(f"Downloading {url} ...")
    resp = requests.get(url, timeout=90, headers={"User-Agent": "CureMyBill/1.0"})
    resp.raise_for_status()
    log(f"Downloaded {len(resp.content) / 1_000_000:.1f} MB")
    return zipfile.ZipFile(io.BytesIO(resp.content))


def find_rvu_dataframe(zf: zipfile.ZipFile) -> pd.DataFrame:
    names = zf.namelist()
    log(f"Zip contains: {names}")

    candidates = [n for n in names if "PPRRVU" in n.upper() and n.lower().endswith(".csv")]
    if not candidates:
        candidates = [n for n in names if "PPRRVU" in n.upper()]
    if not candidates:
        candidates = [n for n in names if n.lower().endswith((".csv", ".xlsx", ".xls"))]
    if not candidates:
        raise RuntimeError(f"No RVU data file found in zip. Contents: {names}")

    name = candidates[0]
    log(f"Using file inside zip: {name}")
    raw = zf.read(name)

    # CMS's PPRRVU CSV wraps each column header across TWO physical rows
    # (e.g. "STATUS" then "CODE" directly below it). Read a chunk with no
    # header, find the row containing "HCPCS", and merge it with the row
    # directly above to reconstruct full column names.
    preview = pd.read_csv(io.BytesIO(raw), header=None, nrows=20, dtype=str)

    header_row_idx = None
    for i in range(len(preview)):
        row_vals = [str(v).strip().upper() for v in preview.iloc[i].tolist()]
        if "HCPCS" in row_vals:
            header_row_idx = i
            break

    if header_row_idx is None:
        raise RuntimeError("Could not find a row containing 'HCPCS' in the first 20 rows.")

    log(f"Found 'HCPCS' header row at index {header_row_idx}")

    top_row = preview.iloc[header_row_idx - 1] if header_row_idx > 0 else None
    bottom_row = preview.iloc[header_row_idx]

    combined_cols = []
    for j in range(len(bottom_row)):
        bottom = str(bottom_row.iloc[j]).strip()
        if bottom.lower() == "nan":
            bottom = ""
        top = ""
        if top_row is not None:
            top = str(top_row.iloc[j]).strip()
            if top.lower() == "nan":
                top = ""
        combined = f"{top} {bottom}".strip().upper()
        combined_cols.append(combined if combined else f"COL_{j}")

    log(f"Reconstructed column names: {combined_cols}")

    df = pd.read_csv(io.BytesIO(raw), skiprows=header_row_idx + 1, header=None, dtype=str)
    df.columns = combined_cols[: len(df.columns)]
    return df


def find_col(df: pd.DataFrame, *candidates: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    # Fuzzy fallback: partial match
    for c in candidates:
        for existing in df.columns:
            if c in existing:
                return existing
    raise KeyError(f"None of {candidates} found in columns: {list(df.columns)}")


def to_float(x):
    try:
        v = float(str(x).strip())
        return v
    except (TypeError, ValueError):
        return None


def build_from_pfs(df: pd.DataFrame) -> pd.DataFrame:
    hcpcs_col = find_col(df, "HCPCS")
    desc_col = find_col(df, "DESCRIPTION")
    status_col = find_col(df, "STATUS CODE", "STATUS")
    work_col = find_col(df, "WORK RVU")
    nonfac_pe_col = find_col(
        df, "NON-FACILITY PE RVU", "NON FACILITY PE RVU", "NONFACILITY PE RVU", "NON-FAC PE RVU"
    )
    mp_col = find_col(df, "MP RVU")

    log(
        f"Column mapping: HCPCS={hcpcs_col!r} DESC={desc_col!r} STATUS={status_col!r} "
        f"WORK={work_col!r} NONFAC_PE={nonfac_pe_col!r} MP={mp_col!r}"
    )

    rows = []
    skipped_inactive, skipped_zero, skipped_missing = 0, 0, 0

    for _, r in df.iterrows():
        status = str(r.get(status_col, "")).strip().upper()
        if status != "A":
            skipped_inactive += 1
            continue

        work = to_float(r.get(work_col))
        pe = to_float(r.get(nonfac_pe_col))
        mp = to_float(r.get(mp_col))
        if work is None or pe is None or mp is None:
            skipped_missing += 1
            continue

        total_rvu = work + pe + mp
        if total_rvu <= 0:
            skipped_zero += 1
            continue

        code = str(r.get(hcpcs_col, "")).strip().upper()
        desc = str(r.get(desc_col, "")).strip()
        if not code or not desc or desc.lower() == "nan":
            skipped_missing += 1
            continue

        rows.append(
            {
                "cpt_code": code,
                "description": desc,
                "standard_price": round(total_rvu * CONVERSION_FACTOR, 2),
            }
        )

    log(
        f"Rows: kept={len(rows)}, skipped_inactive={skipped_inactive}, "
        f"skipped_zero_rvu={skipped_zero}, skipped_missing_data={skipped_missing}"
    )

    out = pd.DataFrame(rows).drop_duplicates(subset="cpt_code")
    if out.empty:
        raise RuntimeError("Parsed 0 usable codes from the RVU file — format likely changed.")
    return out


def main() -> None:
    try:
        zf = download_zip(RVU_ZIP_URL)
        raw_df = find_rvu_dataframe(zf)
        log(f"Loaded raw RVU table: {len(raw_df)} rows")

        pfs_df = build_from_pfs(raw_df)
        log(f"Built {len(pfs_df)} active Physician Fee Schedule codes.")

        frames = [pfs_df]
        if os.path.exists(EXTRA_PATH):
            extra_df = pd.read_csv(EXTRA_PATH, dtype={"cpt_code": str})
            log(f"Merging {len(extra_df)} curated codes (labs, drug J-codes) not in the PFS.")
            frames.append(extra_df)

        final_df = pd.concat(frames, ignore_index=True)
        final_df = final_df.drop_duplicates(subset="cpt_code", keep="last")
        final_df = final_df.sort_values("cpt_code").reset_index(drop=True)

        final_df.to_csv(OUTPUT_PATH, index=False)
        log(f"SUCCESS — wrote {len(final_df)} total codes to {OUTPUT_PATH}")

    except Exception as e:
        log(f"FAILED to build from live CMS data: {e!r}")
        if os.path.exists(FALLBACK_PATH):
            log("Falling back to the last known-good fee_schedule.csv so the app still works.")
            with open(FALLBACK_PATH, "rb") as src, open(OUTPUT_PATH, "wb") as dst:
                dst.write(src.read())
        else:
            log("No fallback file available either — leaving any existing fee_schedule.csv in place.")
        # Exit 0 on purpose: a stale-but-working fee schedule should never block deployment.
        sys.exit(0)


if __name__ == "__main__":
    main()
