# deepowl

<img width="1536" height="1024" alt="logo_deepowlv1" src="https://github.com/user-attachments/assets/bfd4963f-77f0-4347-9024-1951276fd814" />


> Local AI tutor for your private documents.

Drop your docs. deepowl reads them, builds a knowledge graph,
and teaches you through Socratic dialogue — questions, feedback,
and memory of what you already know.

No cloud. No API keys. Runs entirely on your machine.

```
you drop docs → deepowl reads them → builds a knowledge graph
             → creates a curriculum → teaches you via chat
             → remembers your progress across sessions
```

---

## Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) installed and running

Pull the required models:

```bash
ollama pull qwen2.5:7b          # LLM for teaching
ollama pull nomic-embed-text    # embeddings
```

---

## Installation

from source:

```bash
git clone https://github.com/alibekashirali/deepowl.git
cd deepowl
pip install -e .
```

---

## Quick start

```bash
# 1. Index your documents (PDF, MD, TXT, DOCX, EPUB)
deepowl ingest ./docs/

# 2. Extract concepts and build the knowledge graph
deepowl build

# 3. Start a learning session
deepowl start
```

That's it. deepowl will explain concepts from your docs, ask questions, evaluate your answers, and remember your progress.

---

## Example session

```
$ deepowl start

📚 47 files indexed · 312 concepts found
🟡 Continuing from last session...
⚠  3 concepts outdated (docs changed) — will review later

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Kafka Partitions
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

■ Explanation

A Kafka topic is divided into partitions — ordered, immutable
sequences of messages. Producers write to partitions, consumers
read from them independently. Partitions are the unit of
parallelism in Kafka: more partitions = more throughput.
[source: kafka_guide.md, chunk 4]

Press Enter to continue...

■ Question

A producer sends messages with the same key. Which partition
do they land on, and why?

> messages with the same key always go to the same partition
  because kafka hashes the key to pick the partition

■ Feedback

Exactly right. Kafka uses murmur2 hash on the message key
modulo the number of partitions. This guarantees ordering
per key across all messages. ✓

[████████░░] 82/100

Enter to continue · q to quit · ? to ask
```

---

## Commands

| Command | Description |
|---------|-------------|
| `deepowl ingest <path>` | Index a file or folder |
| `deepowl build` | Extract concepts and build the knowledge graph |
| `deepowl start` | Start a learning session |
| `deepowl watch <path>` | Watch a folder and re-index on file changes |
| `deepowl status` | Show knowledge base stats and progress |

### During a session

| Key | Action |
|-----|--------|
| `Enter` | Continue to next concept |
| `q` | Quit session |
| `?` | Ask your own question (searches your docs) |
| `? what is X` | Ask inline without extra prompt |

---

## How it works

1. **Ingest** — parses your documents, splits into ~500-token chunks, generates embeddings via Ollama, stores in ChromaDB locally
2. **Build** — LLM extracts concepts and relationships from chunks, builds a NetworkX knowledge graph stored in SQLite
3. **Learn** — curriculum orders concepts (prerequisites first, low confidence first), retrieves relevant chunks via RAG, teaches through explanation → question → evaluation loop
4. **Progress** — confidence per concept (0–100), weighted across sessions; concepts marked `outdated` when source docs change

---

## Configuration

Config is auto-created at `~/.deepowl/config.yaml` on first run:

```yaml
model:
  provider: ollama
  name: qwen2.5:7b           # swap for any Ollama model
  embedding: nomic-embed-text

storage:
  db_path: ~/.deepowl/deepowl.db
  chroma_path: ~/.deepowl/chroma

teaching:
  language: auto             # auto | en | ru | kk
  style: socratic
  session_length: 20
  spaced_repetition: true
```

All data is stored in `~/.deepowl/` — nothing else, nowhere else.

---

## Supported formats

`.pdf` `.md` `.txt` `.docx` `.epub`
