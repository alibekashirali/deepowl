import sqlite3
import networkx as nx


def next_concept(conn: sqlite3.Connection, graph: nx.DiGraph) -> dict | None:
    """Return the next concept to study, or None if all done."""
    from deepowl.graph.builder import curriculum_order

    ordered = curriculum_order(graph, conn)
    for name in ordered:
        row = conn.execute("""
            SELECT c.id, c.name, c.description, p.status, p.confidence
            FROM concepts c
            LEFT JOIN progress p ON p.concept_id = c.id
            WHERE c.name = ? AND (p.status IS NULL OR p.status != 'done')
        """, (name,)).fetchone()
        if row:
            return dict(row)
    return None


def update_progress(conn: sqlite3.Connection, concept_id: int, score: int) -> None:
    """Update confidence and status for a concept after a study round."""
    current = conn.execute(
        "SELECT confidence, attempts FROM progress WHERE concept_id = ?", (concept_id,)
    ).fetchone()

    if not current:
        return

    # Weighted average: new score counts 40%, history 60%
    old_conf = current["confidence"] or 0
    new_conf = int(old_conf * 0.6 + score * 0.4)
    attempts = (current["attempts"] or 0) + 1

    status = "done" if new_conf >= 80 and attempts >= 2 else "in_progress"

    conn.execute("""
        UPDATE progress
        SET confidence = ?, status = ?, attempts = ?, last_studied = datetime('now')
        WHERE concept_id = ?
    """, (new_conf, status, attempts, concept_id))
    conn.commit()


def mark_outdated(conn: sqlite3.Connection, source_file: str) -> int:
    """Mark all concepts from a changed file as outdated. Returns count."""
    rows = conn.execute("""
        SELECT id FROM concepts
        WHERE source_files LIKE ?
    """, (f"%{source_file}%",)).fetchall()

    for row in rows:
        conn.execute(
            "UPDATE progress SET status = 'outdated' WHERE concept_id = ?", (row["id"],)
        )
    conn.commit()
    return len(rows)


def get_progress_summary(conn: sqlite3.Connection) -> dict:
    rows = conn.execute("""
        SELECT p.status, COUNT(*) as count
        FROM progress p
        GROUP BY p.status
    """).fetchall()
    summary = {"not_started": 0, "in_progress": 0, "done": 0, "outdated": 0}
    for row in rows:
        summary[row["status"]] = row["count"]
    return summary
