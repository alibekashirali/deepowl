import re
import json
import sqlite3
from datetime import datetime
from pathlib import Path

import networkx as nx

from deepowl.llm import call_llm


# ── Extraction ────────────────────────────────────────────────────────────────

_EXTRACT_SYSTEM = """\
Extract 1-3 key concepts from the given text.
For each concept use EXACTLY this format (one concept per block):

CONCEPT: <short name, 2-5 words>
DESC: <one sentence description>
RELATED: <concept1>, <concept2>   (use "none" if no relations)

Rules:
- Concept names must be specific, not generic ("HDFS Block Replication", not "storage")
- RELATED must reference other concepts from THIS text only
- Output nothing else — no preamble, no explanation
"""


def extract_concepts(chunk_content: str, source: str, provider: str, model: str) -> list[dict]:
    text = call_llm(provider, model, system=_EXTRACT_SYSTEM, user=chunk_content[:2000])
    return _parse_concepts(text, source)


def _parse_concepts(text: str, source: str) -> list[dict]:
    concepts = []
    blocks = re.split(r"\n(?=CONCEPT:)", text.strip())
    for block in blocks:
        name_m = re.search(r"CONCEPT:\s*(.+)", block)
        desc_m = re.search(r"DESC:\s*(.+)", block)
        rel_m = re.search(r"RELATED:\s*(.+)", block)
        if not name_m:
            continue
        name = name_m.group(1).strip()
        desc = desc_m.group(1).strip() if desc_m else ""
        related_raw = rel_m.group(1).strip() if rel_m else ""
        related = [
            r.strip()
            for r in related_raw.split(",")
            if r.strip() and r.strip().lower() != "none"
        ]
        concepts.append({"name": name, "description": desc, "source": source, "related": related})
    return concepts


# ── Persistence ───────────────────────────────────────────────────────────────

def save_concepts(conn: sqlite3.Connection, concepts: list[dict]) -> None:
    now = datetime.now().isoformat()
    for c in concepts:
        existing = conn.execute(
            "SELECT id FROM concepts WHERE name = ?", (c["name"],)
        ).fetchone()

        if existing:
            conn.execute(
                "UPDATE concepts SET description = ?, source_files = ?, last_updated = ? WHERE name = ?",
                (c["description"], json.dumps([c["source"]]), now, c["name"]),
            )
            concept_id = existing["id"]
        else:
            conn.execute(
                "INSERT INTO concepts (name, description, source_files, last_updated) VALUES (?, ?, ?, ?)",
                (c["name"], c["description"], json.dumps([c["source"]]), now),
            )
            concept_id = conn.execute(
                "SELECT id FROM concepts WHERE name = ?", (c["name"],)
            ).fetchone()["id"]
            conn.execute(
                "INSERT INTO progress (concept_id, status) VALUES (?, 'not_started')",
                (concept_id,),
            )

    # Save relations (second pass — all concepts must exist first)
    for c in concepts:
        from_id = conn.execute(
            "SELECT id FROM concepts WHERE name = ?", (c["name"],)
        ).fetchone()
        if not from_id:
            continue
        for related_name in c["related"]:
            to_id = conn.execute(
                "SELECT id FROM concepts WHERE name = ?", (related_name,)
            ).fetchone()
            if not to_id:
                continue
            exists = conn.execute(
                "SELECT 1 FROM concept_relations WHERE from_concept_id = ? AND to_concept_id = ?",
                (from_id["id"], to_id["id"]),
            ).fetchone()
            if not exists:
                conn.execute(
                    "INSERT INTO concept_relations (from_concept_id, to_concept_id) VALUES (?, ?)",
                    (from_id["id"], to_id["id"]),
                )
    conn.commit()


# ── Graph ─────────────────────────────────────────────────────────────────────

def build_graph(conn: sqlite3.Connection) -> nx.DiGraph:
    """Build a NetworkX DiGraph from stored concepts and relations."""
    graph = nx.DiGraph()

    for row in conn.execute("SELECT id, name FROM concepts"):
        graph.add_node(row["name"], concept_id=row["id"])

    for row in conn.execute("""
        SELECT c1.name AS from_name, c2.name AS to_name
        FROM concept_relations cr
        JOIN concepts c1 ON cr.from_concept_id = c1.id
        JOIN concepts c2 ON cr.to_concept_id = c2.id
    """):
        graph.add_edge(row["from_name"], row["to_name"])

    return graph


def curriculum_order(graph: nx.DiGraph, conn: sqlite3.Connection) -> list[str]:
    """Return concept names ordered: prerequisites first, low confidence first."""
    if not graph.nodes:
        return []

    # Topological sort; fall back on cycle-safe ordering
    try:
        topo = list(nx.topological_sort(graph))
    except nx.NetworkXUnfeasible:
        pr = nx.pagerank(graph)
        topo = sorted(graph.nodes(), key=lambda n: pr.get(n, 0), reverse=True)

    # Attach progress info
    progress_map = {}
    for row in conn.execute("""
        SELECT c.name, p.confidence, p.status
        FROM concepts c
        LEFT JOIN progress p ON p.concept_id = c.id
    """):
        progress_map[row["name"]] = {
            "confidence": row["confidence"] or 0,
            "status": row["status"] or "not_started",
        }

    # Sort within topological order: outdated > in_progress > not_started > done
    status_priority = {"outdated": 0, "in_progress": 1, "not_started": 2, "done": 99}

    def sort_key(name: str):
        p = progress_map.get(name, {"status": "not_started", "confidence": 0})
        topo_rank = topo.index(name) if name in topo else 999
        return (status_priority.get(p["status"], 2), p["confidence"], topo_rank)

    return sorted(graph.nodes(), key=sort_key)
