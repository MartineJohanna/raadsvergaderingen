# graph_from_llm.py
import argparse, pickle, json, re, subprocess, signal, sys
from pathlib import Path
import networkx as nx
from pyvis.network import Network  # pyvis voor interactieve HTML
import hashlib, os

# ---------- helpers: cache & keys ----------

def chunk_key(text: str, meta: dict | None) -> str:
    if meta and isinstance(meta, dict) and meta.get("chunk_hash"):
        return meta["chunk_hash"]
    return hashlib.md5((text or "").encode("utf-8")).hexdigest()

def _safe_append_lines(path: Path, lines: list[str]) -> None:
    """Append lines atomisch + fsync zodat we niets kwijtraken bij crash/stop."""
    if not lines:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    # Open met low-level os zodat we kunnen fsync'en
    fd = os.open(path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o644)
    try:
        with os.fdopen(fd, "a", encoding="utf-8", buffering=1) as f:
            for line in lines:
                f.write(line)
                if not line.endswith("\n"):
                    f.write("\n")
            f.flush()
            os.fsync(f.fileno())
    finally:
        try:
            os.close(fd)
        except OSError:
            pass

# ---------- LLM ----------

def query_llm(text, model="llama3"):
    """
    Vraagt Ollama om uitsluitend geldige JSON terug te geven:
    {"personen": ["..."], "onderwerpen": ["..."]}
    """
    prompt = f"""Geef uitsluitend geldige JSON in exact dit formaat:
{{
  "personen": ["..."],
  "onderwerpen": ["..."]
}}
Zonder extra uitleg of tekst erbuiten.

Tekst:
{text[:1500]}"""  # truncate om tokens te beperken

    result = subprocess.run(
        ["ollama", "run", model],
        input=prompt.encode("utf-8"),
        capture_output=True,
    )
    out = result.stdout.decode("utf-8", errors="ignore")

    # probeer expliciet alleen het JSON-gedeelte te pakken
    try:
        start = out.find("{")
        end = out.rfind("}")
        obj = json.loads(out[start:end+1])
        if not isinstance(obj, dict):
            raise ValueError("not a dict")
        personen = obj.get("personen", [])
        onderwerpen = obj.get("onderwerpen", [])
        # normaliseer basaal
        personen = [p.strip() for p in personen if isinstance(p, str) and p.strip()]
        onderwerpen = [t.strip().lower() for t in onderwerpen if isinstance(t, str) and t.strip()]
        return {"personen": personen, "onderwerpen": onderwerpen}
    except Exception:
        return {"personen": [], "onderwerpen": []}

# ---------- hoofdlogica ----------

def build_graphs(index_dir: Path, out_dir: Path, model="llama3",
                 limit=None, min_edge_weight_for_export=1,
                 flush_every=200, checkpoint_every=500):
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(index_dir / "passages.pkl", "rb") as f:
        passages = pickle.load(f)
    with open(index_dir / "mapping.pkl", "rb") as f:
        mapping = pickle.load(f)

    if limit:
        passages = passages[:limit]
        mapping  = mapping[:limit]

    G = nx.Graph()

    # ---- cache inlezen (jsonl; key -> {"p":[...],"t":[...]})
    cache_path = out_dir / "ner_cache_llm.jsonl"
    cache = {}
    if cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    # steun zowel oude velden ('personen','onderwerpen') als compacte 'p','t'
                    k = row.get("k")
                    if not k:
                        continue
                    p = row.get("p", row.get("personen", []))
                    t = row.get("t", row.get("onderwerpen", []))
                    cache[k] = {"p": p, "t": t}
                except Exception:
                    # beschadigde regel overslaan
                    pass

    new_cache_lines = []

    # Checkpoint helpers
    ckpt_graphml = out_dir / "rag_graph.graphml.tmp"
    ckpt_gpickle  = out_dir / "rag_graph.gpickle.tmp"

    def _checkpoint(i_passages_done: int):
        """Schrijf snelle checkpoint weg (gpickle + graphml)."""
        try:
            nx.write_gpickle(G, ckpt_gpickle)
        except Exception:
            pass
        # graphml kan groot zijn; alleen schrijven als het zinvol is
        try:
            nx.write_graphml(G, ckpt_graphml)
        except Exception:
            pass
        # cache flushen
        _safe_append_lines(cache_path, new_cache_lines)
        new_cache_lines.clear()
        print(f"[checkpoint] na {i_passages_done} passages: cache + graph opgeslagen")

    # SIGINT/SIGTERM netjes afvangen en eerst flushen
    interrupted = {"flag": False}
    def _handle_signal(signum, frame):
        print(f"\n== Signaal {signum} ontvangen; checkpoint wegschrijven en stoppen… ==")
        interrupted["flag"] = True
        _checkpoint(i_passages_done=i+1 if 'i' in locals() else 0)
        sys.exit(130)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    # ---- bouwen
    for i, text in enumerate(passages):
        # probeer meta per passage; mapping kan lijst/tuple/dict zijn
        meta = None
        try:
            mi = mapping[i]
            if isinstance(mi, dict):
                meta = mi
            elif isinstance(mi, (list, tuple)) and mi and isinstance(mi[-1], dict):
                meta = mi[-1]
        except Exception:
            meta = None

        k = chunk_key(text, meta)

        # uit cache?
        if k in cache:
            persons = cache[k]["p"]
            topics  = cache[k]["t"]
        else:
            ents = query_llm(text, model=model)
            persons = ents.get("personen", [])
            topics  = ents.get("onderwerpen", [])
            # append-alleen regel voorbereiden (compacte velden p/t)
            new_cache_lines.append(json.dumps({"k": k, "p": persons, "t": topics}, ensure_ascii=False))

        # nodes
        for p in persons:
            G.add_node(p, kind="person")
        for t in topics:
            G.add_node(t, kind="topic")

        # persoon-topic edges
        for p in persons:
            for t in topics:
                if G.has_edge(p, t):
                    G[p][t]["weight"] += 1
                else:
                    G.add_edge(p, t, weight=1)

        # persoon-persoon edges
        for a in range(len(persons)):
            for b in range(a+1, len(persons)):
                u, v = persons[a], persons[b]
                if G.has_edge(u, v):
                    G[u][v]["weight"] += 1
                else:
                    G.add_edge(u, v, weight=1)

        # periodiek status + flush
        if (i+1) % 20 == 0:
            print(f"... {i+1} passages verwerkt")
        if (i+1) % flush_every == 0:
            _safe_append_lines(cache_path, new_cache_lines)
            new_cache_lines.clear()
        if (i+1) % checkpoint_every == 0:
            _checkpoint(i_passages_done=i+1)

    # laatste flush
    _safe_append_lines(cache_path, new_cache_lines)
    new_cache_lines.clear()

    # export (optioneel prunen op gewicht)
    if min_edge_weight_for_export > 1:
        H = nx.Graph()
        for u, v, d in G.edges(data=True):
            if d.get("weight", 0) >= min_edge_weight_for_export:
                H.add_node(u, **G.nodes[u])
                H.add_node(v, **G.nodes[v])
                H.add_edge(u, v, **d)
    else:
        H = G

    out_path = out_dir / "rag_graph.graphml"
    nx.write_graphml(H, out_path)
    # definitieve gpickle voor snel herladen
    nx.write_gpickle(G, out_dir / "rag_graph.gpickle")
    print("GraphML opgeslagen:", out_path)
    print("GPickle opgeslagen:", out_dir / "rag_graph.gpickle")

    return G, H  # geef zowel volledige als geprunde graph terug

# ---------- visualisaties ----------

def plot_graph(G, min_weight=2, out_png: Path = None):
    import matplotlib.pyplot as plt

    # prunen voor leesbaarheid
    H = nx.Graph()
    for u, v, d in G.edges(data=True):
        if d.get("weight", 0) >= min_weight:
            H.add_node(u, **G.nodes[u])
            H.add_node(v, **G.nodes[v])
            H.add_edge(u, v, **d)

    if H.number_of_nodes() == 0:
        print("Niets te plotten na pruning; verlaag min_weight.")
        return

    pos = nx.spring_layout(H, seed=42)
    sizes = [120 + 60*H.degree(n) for n in H.nodes()]
    colors = []
    for n in H.nodes():
        kind = H.nodes[n].get("kind", "")
        colors.append("#1f77b4" if kind=="person" else "#2ca02c" if kind=="topic" else "#888888")

    nx.draw(H, pos, with_labels=True, node_size=sizes, font_size=8, edge_color="#AAAAAA", node_color=colors)
    plt.title(f"RAG Graph (min edge weight ≥ {min_weight})")
    if out_png:
        plt.savefig(out_png, dpi=200, bbox_inches="tight")
        print("Plot opgeslagen:", out_png)
    else:
        plt.show()

def save_html_graph(G, html_path: Path, min_weight=2, height="800px", width="100%"):
    """
    Sla een interactieve HTML-graph op (PyVis).
    - Kleurt 'person' en 'topic' anders
    - Verbergt lichte randen met min_weight
    """
    # prune voor leesbaarheid
    H = nx.Graph()
    for u, v, d in G.edges(data=True):
        if d.get("weight", 0) >= min_weight:
            H.add_node(u, **G.nodes[u])
            H.add_node(v, **G.nodes[v])
            H.add_edge(u, v, **d)

    net = Network(height=height, width=width, bgcolor="#ffffff", font_color="#222")
    net.force_atlas_2based()

    # nodes met kleur op 'kind'
    for n, data in H.nodes(data=True):
        kind = data.get("kind", "")
        color = "#1f77b4" if kind == "person" else "#2ca02c" if kind == "topic" else "#888888"
        title = f"{n} ({kind})" if kind else str(n)
        net.add_node(n, label=str(n), title=title, color=color)

    # edges met gewicht als dikte
    for u, v, d in H.edges(data=True):
        w = int(d.get("weight", 1))
        net.add_edge(u, v, value=w, title=f"weight: {w}")

    # ---- JSON-options (géén JS-string) ----
    options = {
        "nodes": {"shape": "dot", "scaling": {"min": 5, "max": 40}},
        "edges": {"arrows": {"to": {"enabled": False}}, "smooth": False},
        "physics": {"stabilization": {"iterations": 200}, "solver": "forceAtlas2Based"},
        "interaction": {"hover": True, "tooltipDelay": 150, "multiselect": True, "dragView": True, "zoomView": True}
    }
    net.set_options(json.dumps(options))

    html_path.parent.mkdir(parents=True, exist_ok=True)
    net.write_html(str(html_path), notebook=False, local=True)
    print("Interactie-graph opgeslagen:", html_path)

# ---------- cli ----------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="./data/denhaag/2025/rag_index", help="pad naar rag_index/")
    ap.add_argument("--out",   default="./data/denhaag/2025/graphs",   help="output map")
    ap.add_argument("--limit", type=int, default=None, help="aantal passages (None = alles)")
    ap.add_argument("--model", default="llama3", help="Ollama modelnaam")
    ap.add_argument("--plot", action="store_true", help="sla ook een PNG-plot op")
    ap.add_argument("--min_weight", type=int, default=2, help="minimale edge weight (pruning)")
    ap.add_argument("--html", action="store_true", help="sla ook interactieve HTML op")
    ap.add_argument("--flush_every", type=int, default=200, help="elke N passages cache naar schijf")
    ap.add_argument("--checkpoint_every", type=int, default=500, help="elke N passages graph + cache checkpoint")
    args = ap.parse_args()

    G, H = build_graphs(
        Path(args.index), Path(args.out),
        model=args.model,
        limit=args.limit,
        min_edge_weight_for_export=1,
        flush_every=args.flush_every,
        checkpoint_every=args.checkpoint_every
    )

    if args.plot:
        plot_graph(H, min_weight=args.min_weight, out_png=Path(args.out) / "rag_graph.png")

    if args.html:
        save_html_graph(H, Path(args.out) / "rag_graph.html", min_weight=args.min_weight)
