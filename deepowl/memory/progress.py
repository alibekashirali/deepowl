import sqlite3
import hashlib
from pathlib import Path
from datetime import datetime


def get_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    _init_schema(conn)
    return conn


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS files (
            id INTEGER PRIMARY KEY,
            path TEXT UNIQUE,
            hash TEXT,
            last_indexed TIMESTAMP,
            last_modified TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY,
            file_id INTEGER REFERENCES files(id),
            content TEXT,
            embedding_id TEXT,
            chunk_index INTEGER
        );

        CREATE TABLE IF NOT EXISTS concepts (
            id INTEGER PRIMARY KEY,
            name TEXT UNIQUE,
            description TEXT,
            source_files TEXT,
            last_updated TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS progress (
            id INTEGER PRIMARY KEY,
            concept_id INTEGER REFERENCES concepts(id),
            status TEXT DEFAULT 'not_started',
            confidence INTEGER DEFAULT 0,
            last_studied TIMESTAMP,
            attempts INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY,
            started_at TIMESTAMP,
            ended_at TIMESTAMP,
            concepts_covered TEXT
        );

        CREATE TABLE IF NOT EXISTS concept_relations (
            id INTEGER PRIMARY KEY,
            from_concept_id INTEGER REFERENCES concepts(id),
            to_concept_id   INTEGER REFERENCES concepts(id),
            relation_type   TEXT DEFAULT 'related'
        );

        CREATE TABLE IF NOT EXISTS chunk_reviews (
            id          INTEGER PRIMARY KEY,
            chunk_id    INTEGER REFERENCES chunks(id),
            reviewed_at TIMESTAMP,
            score       INTEGER DEFAULT 0
        );
    """)
    conn.commit()


def hash_file(path: Path) -> str:
    h = hashlib.md5()
    h.update(path.read_bytes())
    return h.hexdigest()


def get_file_record(conn: sqlite3.Connection, path: str):
    return conn.execute("SELECT * FROM files WHERE path = ?", (path,)).fetchone()


def upsert_file(conn: sqlite3.Connection, path: str, file_hash: str) -> int:
    now = datetime.now().isoformat()
    existing = get_file_record(conn, path)
    if existing:
        conn.execute(
            "UPDATE files SET hash = ?, last_indexed = ? WHERE path = ?",
            (file_hash, now, path),
        )
    else:
        conn.execute(
            "INSERT INTO files (path, hash, last_indexed, last_modified) VALUES (?, ?, ?, ?)",
            (path, file_hash, now, now),
        )
    conn.commit()
    return conn.execute("SELECT id FROM files WHERE path = ?", (path,)).fetchone()["id"]


def save_chunks(conn: sqlite3.Connection, file_id: int, chunks: list[dict]) -> None:
    conn.execute("DELETE FROM chunks WHERE file_id = ?", (file_id,))
    conn.executemany(
        "INSERT INTO chunks (file_id, content, embedding_id, chunk_index) VALUES (?, ?, ?, ?)",
        [(file_id, c["content"], c.get("embedding_id", ""), c["index"]) for c in chunks],
    )
    conn.commit()


def get_stats(conn: sqlite3.Connection) -> dict:
    files    = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
    chunks   = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    concepts = conn.execute("SELECT COUNT(*) FROM concepts").fetchone()[0]
    done     = conn.execute("SELECT COUNT(*) FROM progress WHERE status = 'done'").fetchone()[0]
    outdated = conn.execute("SELECT COUNT(*) FROM progress WHERE status = 'outdated'").fetchone()[0]
    return {"files": files, "chunks": chunks, "concepts": concepts, "done": done, "outdated": outdated}
