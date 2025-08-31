# -*- coding: utf-8 -*-
# Den Haag RIS downloader (Gemeenteraad) – Python 3.8/3.9 compatible

import os
import re
import time
import hashlib
import mimetypes
import cgi
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
import pandas as pd
from slugify import slugify

# ========== Instellingen ==========
BASE = "https://denhaag.raadsinformatie.nl"
YEAR = 2025
YEAR_URL = f"{BASE}/sitemap/meetings/{YEAR}"

# Welke documenttypes wil je minimaal?
ONLY_TYPES = {"notulen", "besluitenlijst", "raadsvoorstel", "motie", "amendement"}
# Zet op None of een lege set() om alles te bewaren.

DATA_DIR = Path(f"./data/denhaag/{YEAR}")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ========== HTTP session met retries ==========
session = requests.Session()
session.headers.update({
    "User-Agent": "DenHaag-RIS-scraper/1.0 (research; contact: you@example.com)"
})
retry = Retry(total=5, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
session.mount("https://", HTTPAdapter(max_retries=retry))
session.mount("http://",  HTTPAdapter(max_retries=retry))

# ========== Helpers ==========

def resolve_document_download_url(url, referer=None):
    """
    Sommige NotuBiz-links geven eerst een HTML viewer terug.
    Deze functie haalt de HTML op en zoekt de échte PDF-download (of iframe/src).
    """
    headers = {}
    if referer:
        headers["Referer"] = referer

    r = session.get(url, timeout=30, headers=headers, allow_redirects=True)
    r.raise_for_status()

    ctype = r.headers.get("content-type", "").lower()
    if "application/pdf" in ctype:
        # We kregen al direct een PDF
        return url

    # Alleen bij HTML: zoek naar download/iframe/meta-refresh
    html = r.text
    soup = BeautifulSoup(html, "html.parser")

    # 1) expliciete download-link
    a = soup.select_one("a[href*='/document/'][href*='download']")
    if a and a.get("href"):
        return urljoin(url, a["href"])

    # 2) iframe met document
    iframe = soup.select_one("iframe[src*='/document/']")
    if iframe and iframe.get("src"):
        return urljoin(url, iframe["src"])

    # 3) meta refresh
    meta = soup.select_one("meta[http-equiv='refresh' i]")
    if meta and "url=" in (meta.get("content","").lower()):
        import re
        m = re.search(r"url=(.+)$", meta["content"], flags=re.I)
        if m:
            return urljoin(url, m.group(1).strip())

    # 4) fallback: /document/<id>/download gokken
    m = re.search(r"/document/(\d+)", url)
    if m:
        doc_id = m.group(1)
        return f"https://denhaag.raadsinformatie.nl/document/{doc_id}/download"

    return url  # geen betere gevonden

def safe_filename(name, ext=None, maxlen=120):
    base = slugify(name or "document")[:maxlen].strip("-_")
    return f"{base}{ext or ''}"

def uniquify_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem, suffix = path.stem, path.suffix
    i = 2
    while True:
        cand = path.with_name(f"{stem}-{i}{suffix}")
        if not cand.exists():
            return cand
        i += 1

def meeting_id_from_url(url):
    m = re.search(r"/vergadering/([^/]+)/?", url)
    return m.group(1) if m else hashlib.md5(url.encode()).hexdigest()[:12]

def infer_ext_from_url(url, default=".pdf"):
    p = urlparse(url).path.lower()
    for ext in [".pdf",".docx",".doc",".rtf",".txt",".xlsx",".csv",".pptx"]:
        if p.endswith(ext):
            return ext
    return default

def infer_ext_from_headers(resp, fallback=".pdf"):
    cd = resp.headers.get("content-disposition")
    if cd:
        _, params = cgi.parse_header(cd)
        fn = params.get("filename")
        if fn:
            guess = Path(fn).suffix
            if guess:
                return guess
    ctype = resp.headers.get("content-type", "").split(";")[0].strip()
    ext = mimetypes.guess_extension(ctype) or ""
    if ctype == "application/pdf":
        return ".pdf"
    if ctype in (
        "application/msword",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ):
        return ".docx"
    return ext or fallback

def looks_like_doc(href, text=""):
    href_l = (href or "").lower()
    text_l = (text or "").lower()
    DOC_EXTS = (".pdf",".docx",".doc",".rtf")
    if href_l.endswith(DOC_EXTS):
        return True
    if "/document/" in href_l:
        return True
    if any(k in href_l for k in ["document","download","bestand","file"]):
        return True
    KEYWORDS = ["notulen","besluiten","raadsvoorstel","voorstel","motie","amendement",
                "bijlage","agenda","toelichting","memo","raadsinformatie"]
    if any(k in text_l for k in KEYWORDS):
        return True
    return False

def classify_doc(label_or_url):
    t = (label_or_url or "").lower()
    if "notulen" in t: return "notulen"
    if "besluit" in t: return "besluitenlijst"
    if "motie" in t: return "motie"
    if "amendement" in t: return "amendement"
    if "raadsvoorstel" in t or "voorstel" in t: return "raadsvoorstel"
    if "agenda" in t: return "agenda"
    if "bijlage" in t: return "bijlage"
    return "overig"

def extract_doc_id(link):
    m = re.search(r"/document/(\d+)", link or "")
    return m.group(1) if m else None

# ========== Scrapers ==========
def get_meeting_list():
    r = session.get(YEAR_URL, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    rows = []
    for a in soup.select("a[href*='/vergadering/']"):
        url = urljoin(BASE, a.get("href"))
        title = a.get_text(" ", strip=True)
        rows.append({"title": title, "url": url})

    df = pd.DataFrame(rows).drop_duplicates("url").reset_index(drop=True)

    # alleen Gemeenteraad (pas aan als je ook commissies wilt)
    df = df[df["title"].str.contains(r"\bGemeenteraad\b", case=False, na=False)].copy()

    # datum uit titel
    MONTHS = {
        "januari":1,"februari":2,"maart":3,"april":4,"mei":5,"juni":6,
        "juli":7,"augustus":8,"september":9,"oktober":10,"november":11,"december":12
    }
    def parse_date_from_title(t):
        m = re.search(
            r"(\d{1,2})\s+(januari|februari|maart|april|mei|juni|juli|augustus|september|oktober|november|december)\s+(\d{4})",
            t, re.I
        )
        if m:
            d, mon, y = int(m.group(1)), m.group(2).lower(), int(m.group(3))
            return pd.Timestamp(year=y, month=MONTHS[mon], day=d)
        m = re.search(r"(\d{1,2})-(\d{1,2})-(\d{4})", t)
        if m:
            d, mth, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
            return pd.Timestamp(year=y, month=mth, day=d)
        return pd.NaT

    df["date"] = df["title"].apply(parse_date_from_title)
    df["title"] = df["title"].str.replace(r"^\s*>\s*", "", regex=True).str.strip()
    df = df.sort_values("date").reset_index(drop=True)
    return df

def extract_documents(meeting_url, throttle=0.4):
    r = session.get(meeting_url, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    docs = []
    for a in soup.select('a[href]'):
        href = a.get("href")
        label = a.get_text(" ", strip=True)
        if not looks_like_doc(href, label):
            continue
        link = urljoin(BASE, href)
        header = a.find_previous(["h2","h3"])
        agenda = header.get_text(" ", strip=True) if header else None

        doc_id = extract_doc_id(link)
        doc_type = classify_doc(label or link)

        docs.append({
            "doc_id": doc_id,
            "doc_type": doc_type,
            "label": label or "document",
            "link": link,
            "agenda": agenda
        })

    time.sleep(throttle)

    # Optioneel filter op documenttypes
    if ONLY_TYPES:
        filtered = [d for d in docs if d["doc_type"] in ONLY_TYPES]
        if filtered:
            return filtered
    return docs

def download_file(url, dest_path, max_mb=60, referer=None):
    headers = {}
    if referer:
        headers["Referer"] = referer

    # Belangrijk: eerst de viewer-URL omzetten naar echte download
    resolved = resolve_document_download_url(url, referer=referer)

    with session.get(resolved, stream=True, timeout=60, headers=headers, allow_redirects=True) as resp:
        resp.raise_for_status()

        # Als het tóch HTML is, stoppen (geen lege .html opslaan)
        ctype = resp.headers.get("content-type", "").lower()
        if "text/html" in ctype:
            raise RuntimeError(f"Server geeft HTML terug i.p.v. bestand: {resolved}")

        # Extensie bepalen/aanpassen
        real_ext = infer_ext_from_headers(resp, fallback=dest_path.suffix or ".pdf")
        if real_ext and real_ext != dest_path.suffix:
            dest_path = dest_path.with_suffix(real_ext)

        dest_path = uniquify_path(dest_path)
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        total = 0
        with open(dest_path, "wb") as f:
            for chunk in resp.iter_content(1024 * 64):
                if not chunk:
                    break
                total += len(chunk)
                if total > max_mb * 1024 * 1024:
                    raise RuntimeError("File too large (> {} MB)".format(max_mb))
                f.write(chunk)
    return dest_path


# ========== ETL-runner ==========
def run_etl(limit_meetings=None):
    meetings = get_meeting_list()
    if meetings.empty:
        print("⚠️ Geen vergaderingen gevonden op de jaarpagina.")
        return pd.DataFrame()

    if limit_meetings:
        meetings = meetings.head(limit_meetings)

    meta_rows = []

    for _, row in meetings.iterrows():
        meeting_url = row["url"]
        meeting_title = row.get("title")
        meeting_date = row.get("date")  # kan NaT zijn
        mid = meeting_id_from_url(meeting_url)
        mdir = DATA_DIR / mid
        (mdir / "files").mkdir(parents=True, exist_ok=True)

        docs = extract_documents(meeting_url) or []
        print("[{}] docs gevonden: {}".format(meeting_title, len(docs)))
        if docs:
            print("  voorbeeld:", docs[0]["label"], "->", docs[0]["link"])

        if not docs:
            meta_rows.append({
                "meeting_id": mid,
                "meeting_title": meeting_title,
                "meeting_date": meeting_date,
                "meeting_url": meeting_url,
                "agenda_label": None,
                "doc_id": None,
                "doc_type": None,
                "doc_label": None,
                "doc_url": None,
                "local_path": None,
                "status": "no_documents"
            })

        for d in docs:
            label = d.get("label") or "document"
            link = d.get("link")
            agenda = d.get("agenda")
            doc_id = d.get("doc_id") or "doc"
            doc_type = d.get("doc_type") or "overig"

            base_name = "{}-{}-{}".format(doc_type, doc_id, label) if doc_id else "{}-{}".format(doc_type, label)
            ext_guess = infer_ext_from_url(link, default=".pdf")
            fname = safe_filename(base_name, ext=ext_guess)
            fpath = mdir / "files" / fname

            saved_path = None
            status = "skipped"
            if link:
                try:
                    saved_path = download_file(link, fpath, referer=meeting_url)
                    status = "downloaded" if saved_path else "error"
                except Exception as e:
                    status = "error: {}".format(e)

            meta_rows.append({
                "meeting_id": mid,
                "meeting_title": meeting_title,
                "meeting_date": meeting_date,
                "meeting_url": meeting_url,
                "agenda_label": agenda,
                "doc_id": doc_id,
                "doc_type": doc_type,
                "doc_label": label,
                "doc_url": link,
                "local_path": str(saved_path) if saved_path else None,
                "status": status
            })

    meta = pd.DataFrame(meta_rows)

    if "meeting_date" in meta.columns:
        meta["meeting_date"] = pd.to_datetime(meta["meeting_date"], errors="coerce")

    sort_cols = [c for c in ["meeting_date", "meeting_id", "doc_type", "doc_label"] if c in meta.columns]
    if sort_cols:
        meta = meta.sort_values(sort_cols, na_position="last")

    meta["year"] = YEAR
    meta["has_file"] = meta["local_path"].notna() & meta["status"].isin(["downloaded","cached"])

    out_csv = DATA_DIR / "documents_metadata.csv"
    meta.to_csv(out_csv, index=False)
    print("✔ Metadata opgeslagen:", out_csv)
    return meta

# ========== Main ==========
if __name__ == "__main__":
    pd.set_option("display.max_colwidth", None)
    meta = run_etl(limit_meetings=None)  # zet bv. 3 om te testen
    print(meta[["meeting_title","doc_type","doc_label","status","local_path"]].head(20).to_string(index=False))
    print("Files in:", DATA_DIR.resolve())
