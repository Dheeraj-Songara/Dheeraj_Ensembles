"""
PubMed to Excel
---------------
Searches PubMed for a list of queries and exports results to .xlsx files.
Uses NCBI E-utilities API — no browser, no Selenium required.

Usage:
    python pubmed_to_excel.py

Optional: Set your NCBI API key as env var NCBI_API_KEY for higher rate limits (10 req/s vs 3 req/s).
Get a free key at: https://www.ncbi.nlm.nih.gov/account/
"""

import os
import time
import requests
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

# ── Configuration ────────────────────────────────────────────────────────────

QUERIES = [
    "Hippocampus and engram",
    "Susume Tonegawa",
    "Sheena Josselyn"
]

MAX_RESULTS   = 60    # max papers per query (PubMed returns up to 10,000)
MAX_WORKERS   = 5     # parallel fetches for paper details
OUTPUT_DIR    = Path(".")  # folder to save .xlsx files

NCBI_API_KEY  = os.getenv("NCBI_API_KEY", "")  # optional but recommended
BASE_URL      = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# ── NCBI API helpers ──────────────────────────────────────────────────────────

def _params(extra: dict) -> dict:
    p = {"db": "pubmed", "retmode": "xml", **extra}
    if NCBI_API_KEY:
        p["api_key"] = NCBI_API_KEY
    return p


def search_pmids(query: str) -> list[str]:
    """Return list of PMIDs for a PubMed query."""
    r = requests.get(
        f"{BASE_URL}/esearch.fcgi",
        params=_params({"term": query, "retmax": MAX_RESULTS, "usehistory": "n"}),
        timeout=30,
    )
    r.raise_for_status()
    root = ET.fromstring(r.content)
    return [id_el.text for id_el in root.findall(".//Id")]


def fetch_paper(pmid: str) -> dict:
    """Fetch metadata for a single PMID via eFetch."""
    r = requests.get(
        f"{BASE_URL}/efetch.fcgi",
        params=_params({"id": pmid, "rettype": "abstract"}),
        timeout=30,
    )
    r.raise_for_status()

    root = ET.fromstring(r.content)
    article = root.find(".//PubmedArticle")
    if article is None:
        return {"pmid": pmid}

    def text(path):
        el = article.find(path)
        return el.text.strip() if el is not None and el.text else ""

    # Title
    title = text(".//ArticleTitle")

    # Abstract — join structured sections if present
    abstract_texts = article.findall(".//AbstractText")
    if abstract_texts:
        parts = []
        for ab in abstract_texts:
            label = ab.get("Label")
            t = (ab.text or "").strip()
            parts.append(f"{label}: {t}" if label else t)
        abstract = " ".join(parts)
    else:
        abstract = ""

    # Authors
    author_els = article.findall(".//Author")
    authors = []
    for a in author_els:
        last  = (a.findtext("LastName")  or "").strip()
        first = (a.findtext("ForeName")  or "").strip()
        if last:
            authors.append(f"{last} {first}".strip())
    authors_str = "; ".join(authors)

    # Journal + date
    journal = text(".//Journal/Title")
    year    = text(".//PubDate/Year") or text(".//PubDate/MedlineDate")[:4]
    month   = text(".//PubDate/Month")
    date    = f"{year}/{month}" if month else year

    # DOI
    doi = ""
    for id_el in article.findall(".//ArticleId"):
        if id_el.get("IdType") == "doi":
            doi = id_el.text.strip()
            break

    return {
        "pmid":     pmid,
        "doi":      doi,
        "journal":  journal,
        "date":     date,
        "authors":  authors_str,
        "title":    title,
        "abstract": abstract,
    }


def fetch_all_papers(pmids: list[str]) -> list[dict]:
    """Fetch papers in parallel, respecting NCBI rate limits."""
    # NCBI allows 3 req/s without a key, 10/s with one
    delay = 0.1 if NCBI_API_KEY else 0.34
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {}
        for pmid in pmids:
            futures[executor.submit(fetch_paper, pmid)] = pmid
            time.sleep(delay)
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                print(f"  ✗ PMID {futures[future]}: {e}")
    return results

# ── Excel formatting ──────────────────────────────────────────────────────────

HEADER_FILL  = PatternFill("solid", start_color="1F4E79")
HEADER_FONT  = Font(bold=True, color="FFFFFF", name="Arial", size=11)
ROW_FILL_ALT = PatternFill("solid", start_color="EBF2FA")
BORDER_SIDE  = Side(style="thin", color="CCCCCC")
THIN_BORDER  = Border(bottom=BORDER_SIDE)

COL_WIDTHS = {
    "pmid": 12, "doi": 32, "journal": 28,
    "date": 10, "authors": 36, "title": 50, "abstract": 70,
}


def style_sheet(ws):
    columns = list(COL_WIDTHS.keys())

    # Header row
    for col_idx, col_name in enumerate(columns, start=1):
        cell = ws.cell(row=1, column=col_idx)
        cell.value     = col_name.upper()
        cell.font      = HEADER_FONT
        cell.fill      = HEADER_FILL
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        ws.column_dimensions[get_column_letter(col_idx)].width = COL_WIDTHS[col_name]

    ws.row_dimensions[1].height = 22

    # Data rows
    for row_idx in range(2, ws.max_row + 1):
        fill = ROW_FILL_ALT if row_idx % 2 == 0 else None
        for col_idx in range(1, len(columns) + 1):
            cell = ws.cell(row=row_idx, column=col_idx)
            cell.font      = Font(name="Arial", size=10)
            cell.alignment = Alignment(vertical="top", wrap_text=(col_idx == len(columns)))
            cell.border    = THIN_BORDER
            if fill:
                cell.fill = fill

    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions


def save_xlsx(records: list[dict], query: str):
    safe_name = "".join(c if c.isalnum() or c in " _-" else "_" for c in query).strip()
    path = OUTPUT_DIR / f"{safe_name}.xlsx"

    df = pd.DataFrame(records, columns=["pmid", "doi", "journal", "date", "authors", "title", "abstract"])
    df.sort_values("date", ascending=False, inplace=True)
    df.to_excel(path, index=False, sheet_name="Papers")

    wb = load_workbook(path)
    style_sheet(wb.active)
    wb.save(path)
    return path

# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for query in QUERIES:
        print(f"\n🔍 Searching: {query!r}")
        pmids = search_pmids(query)
        if not pmids:
            print("  No results found.")
            continue
        print(f"  Found {len(pmids)} papers. Fetching details…")
        records = fetch_all_papers(pmids)
        path = save_xlsx(records, query)
        print(f"  ✓ Saved {len(records)} papers → {path}")

if __name__ == "__main__":
    run()
