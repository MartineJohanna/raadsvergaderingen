# rag_den_haag.py
# Minimalistische RAG voor Den Haag raadsstukken
# Vereisten: pip install pandas pdfminer.six python-docx sentence-transformers faiss-cpu numpy tqdm openai orjson
# (optioneel voor lokale modellen): pip install ollama  (of gebruik requests naar http://localhost:11434)

import os, re, sys, json, orjson, argparse, hashlib, pickle
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# --- Tekstextractie ---
from pdfminer.high_level import extract_text as pdf_extract_text
import docx

# --- Embeddings & index ---
from sentence_transformers import SentenceTransformer
import faiss

# --- LLM backends (kies OpenAI of Ollama) ---
# OpenAI: zet OPENAI_API_KEY in env
try:
    import openai
except Exception:
    openai = None

import requests

# ========== Config ==========
CHUNK_CHARS = 1000
CHUNK_OVERLAP = 200
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # snel & licht
INDEX_DIRNAME = "rag_index"  # wordt aangemaakt in jouw data dir

# ========== Hulpfuncties ==========
def read_doc(path: Path) -> str:
    sfx = path.suffix.lower()
    try:
        if sfx == ".pdf":
            return pdf_extract_text(str(path))
        if sfx == ".docx":
            d = docx.Document(str(path))
            return "\n".join(p.text for p in d.paragraphs)
        if sfx in {".txt", ".md"}:
            return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""
    return ""

def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def chunk_text(text: str, chunk_chars=CHUNK_CHARS, overlap=CHUNK_OVERLAP) -> List[str]:
    text = normalize_ws(text)
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        j = min(i + chunk_chars, n)
        chunks.append(text[i:j])
        if j == n:
            break
        i = j - overlap
        if i < 0:
            i = 0
    return chunks

def hash_str(s: str) -> str:
    return hashlib.md5(s.encode("utf-8", errors="ignore")).hexdigest()

# ========== Index build ==========
def build_index(data_dir: Path):
    meta_csv = data_dir / "documents_metadata.csv"
    assert meta_csv.exists(), f"Niet gevonden: {meta_csv}"
    df = pd.read_csv(meta_csv)
    # alleen rijen met lokaal bestand
    df = df[df["local_path"].notna() & df["status"].isin(["downloaded","cached"])].copy()

    # Laad/initialiseer embedder
    print(">> Laad embedding model:", EMB_MODEL)
    embedder = SentenceTransformer(EMB_MODEL)

    passages = []
    mapping = []  # per passage: bron info
    total_docs = 0

    print(">> Extraheer en chunk documenten...")
    for _, r in tqdm(df.iterrows(), total=len(df)):
        p = Path(str(r["local_path"]))
        if not p.exists():
            continue
        text = read_doc(p)
        if not text:
            continue

        # optioneel: snelle heuristiek om boilerplate te snoeien
        # (bv. paginanummers, zeer korte lijnen)
        # hier laten we 'm simpel

        chunks = chunk_text(text)
        if not chunks:
            continue

        for k, ch in enumerate(chunks):
            passages.append(ch)
            mapping.append({
                "meeting_id": r.get("meeting_id"),
                "meeting_title": r.get("meeting_title"),
                "meeting_date": r.get("meeting_date"),
                "doc_type": r.get("doc_type"),
                "doc_label": r.get("doc_label"),
                "doc_url": r.get("doc_url"),
                "local_path": str(p),
                "chunk_id": k,
                "chunk_hash": hash_str(ch),
            })
        total_docs += 1

    assert passages, "Geen passages gevonden; controleer of je PDF/DOCX tekst bevatten."

    print(f">> Totaal passages: {len(passages)} uit {total_docs} documenten. Embedden...")
    embs = embedder.encode(passages, batch_size=64, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    dim = embs.shape[1]

    print(">> FAISS index bouwen...")
    index = faiss.IndexFlatIP(dim)
    index.add(embs)

    # Opslaan
    out_dir = data_dir / INDEX_DIRNAME
    out_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_dir / "index.faiss"))
    with open(out_dir / "passages.pkl", "wb") as f:
        pickle.dump(passages, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(out_dir / "mapping.pkl", "wb") as f:
        pickle.dump(mapping, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("✔ Index opgeslagen in", out_dir)

# ========== Retrieval ==========
def load_index(data_dir: Path):
    out_dir = data_dir / INDEX_DIRNAME
    with open(out_dir / "passages.pkl", "rb") as f:
        passages = pickle.load(f)
    with open(out_dir / "mapping.pkl", "rb") as f:
        mapping = pickle.load(f)
    index = faiss.read_index(str(out_dir / "index.faiss"))
    embedder = SentenceTransformer(EMB_MODEL)
    return index, embedder, passages, mapping

def retrieve(query: str, index, embedder, passages, mapping, k=8) -> List[Dict]:
    qemb = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    D, I = index.search(qemb, k)
    I = I[0].tolist()
    D = D[0].tolist()
    out = []
    for score, idx in zip(D, I):
        if idx == -1:
            continue
        out.append({
            "score": float(score),
            "passage": passages[idx],
            "meta": mapping[idx]
        })
    return out

# ========== LLM backends ==========
def call_openai(prompt: str, model="gpt-4o-mini") -> str:
    assert openai is not None, "openai package niet geïnstalleerd"
    api_key = os.environ.get("OPENAI_API_KEY")
    assert api_key, "Zet OPENAI_API_KEY in je omgeving"
    client = openai.OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role":"system","content":"Je bent een behulpzame Nederlandse assistent."},
                  {"role":"user","content":prompt}],
        temperature=0.2,
        max_tokens=800
    )
    return resp.choices[0].message.content.strip()

def call_ollama(prompt: str, model="llama3") -> str:
    # Vereist een draaiende Ollama (localhost:11434) en model: `ollama pull llama3`
    r = requests.post("http://localhost:11434/api/generate",
                      json={"model": model, "prompt": prompt, "stream": False},
                      timeout=120)
    r.raise_for_status()
    data = r.json()
    return data.get("response","").strip()

def build_prompt(question: str, contexts: List[Dict]) -> str:
    # contextblokken + bronvermelding
    blocks = []
    for i, ctx in enumerate(contexts, 1):
        m = ctx["meta"]
        src = m.get("doc_label") or m.get("local_path")
        when = str(m.get("meeting_date") or "")
        url = m.get("doc_url") or ""
        head = f"[Bron {i}] {src} ({when})\nURL: {url}\n"
        body = ctx["passage"]
        blocks.append(head + body)
    context_txt = "\n\n---\n\n".join(blocks)

    return f"""Beantwoord de vraag in het Nederlands, beknopt en feitelijk.
Gebruik uitsluitend de context hieronder; fantaseer niet. Citeer aan het eind de gebruikte bronnen als [1], [2], ...
Als iets niet in de bronnen staat, zeg dat eerlijk.

Vraag:
{question}

Context:
{context_txt}

Antwoord (met korte verwijzingen [1], [2], ...):"""

def answer_question(data_dir: Path, question: str, llm_backend: str = "openai", model: str = None, k: int = 8) -> Tuple[str, List[Dict]]:
    index, embedder, passages, mapping = load_index(data_dir)
    ctxs = retrieve(question, index, embedder, passages, mapping, k=k)
    prompt = build_prompt(question, ctxs)
    if llm_backend == "openai":
        model = model or "gpt-4o-mini"
        ans = call_openai(prompt, model=model)
    elif llm_backend == "ollama":
        model = model or "llama3"
        ans = call_ollama(prompt, model=model)
    else:
        raise ValueError("llm_backend moet 'openai' of 'ollama' zijn")
    return ans, ctxs

# ========== CLI ==========
def main():
    ap = argparse.ArgumentParser(description="RAG over Den Haag raadsstukken")
    sub = ap.add_subparsers(dest="cmd")

    b = sub.add_parser("build", help="bouw/refresh FAISS index")
    b.add_argument("--data", required=True, help="pad naar ./data/denhaag/2025")

    q = sub.add_parser("ask", help="stel een vraag")
    q.add_argument("--data", default="./data/denhaag/2025", help="pad naar data dir")
    q.add_argument("--q", required=True, help="je vraag")
    q.add_argument("--llm", choices=["openai","ollama"], default="openai")
    q.add_argument("--model", default=None, help="specifiek model (bv. gpt-4o-mini of llama3)")
    q.add_argument("--k", type=int, default=8, help="aantal contextpassages")

    args = ap.parse_args()

    if args.cmd == "build":
        build_index(Path(args.data))
    elif args.cmd == "ask":
        ans, ctxs = answer_question(Path(args.data), args.q, llm_backend=args.llm, model=args.model, k=args.k)
        print("\n=== ANTWOORD ===\n")
        print(ans)
        print("\n=== GEBRUIKTE BRONNEN ===")
        for i, c in enumerate(ctxs, 1):
            m = c["meta"]
            print(f"[{i}] {m.get('doc_label') or Path((m.get('local_path') or '')).name} | {m.get('meeting_date')} | {m.get('doc_url')}")
    else:
        ap.print_help()

if __name__ == "__main__":
    main()
