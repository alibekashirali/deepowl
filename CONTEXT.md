# deepowl — Project Context for Claude Code

## What is this?

**deepowl** is a local-first AI-powered learning tool. It reads your documents (PDF, MD, DOCX, TXT) and teaches you the content through an interactive chat — explanations, questions, feedback, and progress tracking. Everything runs locally, no data leaves your machine.

Think: NotebookLM, but fully private, with memory of your progress, and built as a CLI tool.

---

## Core idea

```
you drop docs →  deepowl reads them →  builds a knowledge graph
             →  creates a curriculum →  teaches you via chat
             →  remembers your progress across sessions
```

The teaching loop:
1. **Explain** — short explanation of a concept (from your docs)
2. **Ask** — question to check understanding
3. **Evaluate** — assess the answer, give feedback
4. **Adapt** — if wrong, explain differently; if right, go deeper
5. **Next** — move to the next concept

---

## Tech Stack

| Layer | Tool |
|-------|------|
| CLI | Python + Typer + Rich |
| Web UI (optional) | FastAPI + simple HTML |
| Document parsing | LlamaIndex |
| Vector store | ChromaDB (local) |
| Embeddings | Ollama (`nomic-embed-text`) |
| Local LLM | Ollama (`qwen2.5:7b` default, swappable) |
| Knowledge graph | NetworkX |
| Progress storage | SQLite |
| Config | YAML (`~/.deepowl/config.yaml`) |

---

## Project Structure

```
deepowl/
├── cli.py                  # Typer CLI entry point
├── config.py               # config loader (YAML)
├── ingest/
│   ├── parser.py           # reads PDF, MD, DOCX, TXT, EPUB
│   ├── chunker.py          # splits into ~500 token chunks with overlap
│   └── embedder.py         # ChromaDB + Ollama embeddings
├── graph/
│   └── builder.py          # NetworkX concept graph from chunks
├── teaching/
│   ├── curriculum.py       # builds learning plan from graph
│   ├── tutor.py            # main Socratic dialogue loop
│   └── quiz.py             # question generation + answer evaluation
├── memory/
│   └── progress.py         # SQLite: files, concepts, sessions, progress
├── web/
│   └── app.py              # optional FastAPI web UI
└── tests/
```

---

## SQLite Schema

```sql
-- indexed files with hash for change detection
CREATE TABLE files (
    id INTEGER PRIMARY KEY,
    path TEXT UNIQUE,
    hash TEXT,
    last_indexed TIMESTAMP,
    last_modified TIMESTAMP
);

-- document chunks linked to files
CREATE TABLE chunks (
    id INTEGER PRIMARY KEY,
    file_id INTEGER REFERENCES files(id),
    content TEXT,
    embedding_id TEXT,
    chunk_index INTEGER
);

-- extracted concepts (nodes in knowledge graph)
CREATE TABLE concepts (
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE,
    description TEXT,
    source_files TEXT,   -- JSON list of file paths
    last_updated TIMESTAMP
);

-- user progress per concept
CREATE TABLE progress (
    id INTEGER PRIMARY KEY,
    concept_id INTEGER REFERENCES concepts(id),
    status TEXT,         -- 'not_started' | 'in_progress' | 'done' | 'outdated'
    confidence INTEGER,  -- 0–100
    last_studied TIMESTAMP,
    attempts INTEGER DEFAULT 0
);

-- learning sessions
CREATE TABLE sessions (
    id INTEGER PRIMARY KEY,
    started_at TIMESTAMP,
    ended_at TIMESTAMP,
    concepts_covered TEXT  -- JSON list
);
```

---

## CLI Commands

```bash
deepowl ingest ./docs/          # index a folder
deepowl ingest ./docs/file.md   # index a single file
deepowl watch ./docs/           # watch folder for changes (background)
deepowl start                   # start a learning session
deepowl quiz                    # standalone quiz mode
deepowl quiz --topic kafka      # quiz on a specific topic
deepowl status                  # show knowledge base stats + progress
deepowl sync                    # force full re-index
deepowl config                  # show/edit config
```

---

## Config File (`~/.deepowl/config.yaml`)

```yaml
model:
  provider: ollama          # ollama | anthropic | openai
  name: qwen2.5:7b          # model name
  embedding: nomic-embed-text

storage:
  db_path: ~/.deepowl/deepowl.db
  chroma_path: ~/.deepowl/chroma

teaching:
  language: auto            # auto | en | ru | kk
  style: socratic           # socratic | explain-first | quiz-heavy
  session_length: 20        # minutes
  spaced_repetition: true
```

---

## Key Behaviors

### Incremental ingestion
- Every file is hashed (MD5) on first ingest and stored in SQLite
- On re-ingest, only new or changed files are processed
- If a studied concept changes in the docs → marked as `outdated` → added to review queue

### Concept graph
- LLM extracts concepts and relationships from chunks
- Graph stored in NetworkX, nodes colored by progress status:
  - 🟢 `done` — studied and confident
  - 🟡 `in_progress` — partially studied
  - 🔴 `not_started` — not yet covered
  - ⚠️ `outdated` — docs changed since last study

### Teaching loop (tutor.py)
```python
# pseudocode
concept = curriculum.next_concept()
chunks = rag.retrieve(concept)
explanation = llm.explain(concept, chunks)
question = llm.generate_question(concept, chunks)
answer = input(question)
feedback = llm.evaluate(answer, concept, chunks)
progress.update(concept, score=feedback.score)
```

### Source grounding
- Every explanation must cite the source file + chunk
- Format: `[source: architecture.md, section: L2]`
- Prevents hallucination, builds trust

---

## Development Priorities

Build in this order:

1. **`deepowl ingest`** — parser + chunker + ChromaDB + SQLite file tracking
2. **`deepowl start`** — basic tutor loop (explain → ask → evaluate → next)
3. **Progress memory** — SQLite progress tracking across sessions
4. **Concept graph** — NetworkX extraction + curriculum ordering
5. **`deepowl watch`** — incremental updates on file changes
6. **Web UI** — optional FastAPI interface

---

## What to avoid

- Do NOT build a generic RAG chatbot — this is a structured learning system
- Do NOT skip source citations — every claim must reference a chunk
- Do NOT make setup complex — `pip install deepowl-learn && deepowl ingest ./docs && deepowl start` must work in under 3 minutes
- Do NOT store any data outside `~/.deepowl/` — everything local

---

## Example session

```
$ deepowl start

📚 Knowledge base: 47 files, 312 concepts
🟡 Continuing from last session...
⚠  3 concepts are outdated (docs changed) — will review later

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Block 2 · HDFS Architecture
Source: hadoop_cheatsheet.md · L2
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

HDFS splits large files into 128MB blocks and distributes
them across DataNodes. Each block is replicated 3 times
by default for fault tolerance. The NameNode tracks
where every block lives — it's the brain of the system.

Ready? Press Enter to continue...

🤖 Question: A DataNode crashes. What happens to the
   data it held? Who detects this and what's the response?

> The NameNode detects it via heartbeat timeout, then
  re-replicates the affected blocks from other replicas

🤖 Exactly right. The NameNode uses heartbeats (every 3s
   by default) to monitor DataNodes. On timeout, it
   identifies under-replicated blocks and instructs other
   DataNodes to create new replicas. ✓ +10 confidence

[████████░░] Block 2/6 complete · 3 concepts mastered today
```

---

## Notes

- Default language auto-detected from document content
- Works offline after initial `ollama pull`
- All progress persists in `~/.deepowl/deepowl.db`
- Optional: export progress to Anki (`deepowl export --format anki`)
- Optional: export knowledge base to Obsidian (`deepowl export --format obsidian`)
