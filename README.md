# deepowl

<img width="1424" height="491" alt="logo_deepowlv1" src="https://github.com/user-attachments/assets/a67a62f9-3129-4504-8b37-3a6d0cb0b0b6" />

> Local-first AI tutor for your private documents.

Drop your docs. deepowl reads them, builds a knowledge graph,
and teaches you through Socratic dialogue — questions, feedback,
and memory of what you already know.

Runs locally by default (Ollama). Optionally connects to OpenAI, Anthropic, or Groq.

```
you drop docs → deepowl reads them → builds a knowledge graph
             → creates a curriculum → teaches you via chat
             → remembers your progress across sessions
```

---

## Prerequisites

- Python 3.11+

**For local mode (default):** [Ollama](https://ollama.com) installed and running

```bash
ollama pull qwen3:latest        # LLM for teaching
ollama pull nomic-embed-text    # embeddings
```

**For cloud providers:** set the corresponding API key as an environment variable (see [Configuration](#configuration)).

---

## Installation

From source:

```bash
git clone https://github.com/alibekashirali/deepowl.git
cd deepowl
pip install -e .
```

Directly from GitHub (no clone needed):

```bash
pip install git+https://github.com/alibekashirali/deepowl.git
```

With optional provider SDKs:

```bash
pip install -e '.[openai]'      # OpenAI + Groq
pip install -e '.[anthropic]'   # Anthropic
pip install -e '.[all]'         # everything
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

deepowl — 0 done  0 in progress  47 not started
model: ollama/qwen3:latest

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
per key across all messages.

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
| `deepowl graph` | Show concept graph as a tree |
| `deepowl config show` | Print current configuration |
| `deepowl config set <key> <value>` | Change a config value |
| `deepowl config get <key>` | Print a single config value |

### During a session

| Key | Action |
|-----|--------|
| `Enter` | Continue to next concept |
| `q` | Quit session |
| `?` | Ask your own question (searches your docs) |
| `? what is X` | Ask inline without extra prompt |

---

## How it works

1. **Ingest** — parses your documents, splits into ~500-token chunks, generates embeddings, stores in ChromaDB locally
2. **Build** — LLM extracts concepts and relationships from chunks, builds a NetworkX knowledge graph stored in SQLite
3. **Learn** — curriculum orders concepts (prerequisites first, low confidence first), retrieves relevant chunks via RAG, teaches through explanation → question → evaluation loop
4. **Progress** — confidence per concept (0–100), weighted across sessions; concepts marked `outdated` when source docs change

---

## Configuration

Config is auto-created at `~/.deepowl/config.yaml` on first run.

```yaml
model:
  provider: ollama              # ollama | openai | anthropic | groq
  name: qwen3:latest
  embedding: nomic-embed-text
  embedding_provider: ollama    # ollama | openai

storage:
  db_path: ~/.deepowl/deepowl.db
  chroma_path: ~/.deepowl/chroma

teaching:
  language: auto                # auto | en | ru | kk
  style: socratic
  session_length: 20
  spaced_repetition: true
```

Edit with the CLI:

```bash
deepowl config set model.provider openai   # switches provider + sets model defaults
deepowl config set model.name gpt-4o       # override model
deepowl config show                        # print full config
```

### Provider setup

**Ollama (default, local)**

```bash
ollama serve
ollama pull qwen3:latest
ollama pull nomic-embed-text
```

No API key needed. Everything stays on your machine.

**OpenAI**

```bash
pip install 'deepowl-learn[openai]'
export OPENAI_API_KEY=sk-...
deepowl config set model.provider openai
```

Defaults to `gpt-4o-mini` + `text-embedding-3-small`.

**Anthropic**

```bash
pip install 'deepowl-learn[anthropic]'
export ANTHROPIC_API_KEY=sk-ant-...
deepowl config set model.provider anthropic
```

Defaults to `claude-haiku-4-5-20251001`. Embeddings fall back to Ollama (`nomic-embed-text`) — Anthropic doesn't provide an embeddings API.

**Groq**

```bash
pip install 'deepowl-learn[openai]'   # uses openai SDK
export GROQ_API_KEY=gsk_...
deepowl config set model.provider groq
```

Defaults to `llama-3.1-8b-instant`. Embeddings fall back to Ollama.

---

## Supported formats

`.pdf` `.md` `.txt` `.docx` `.epub`

---

## Data & privacy

All data is stored in `~/.deepowl/` — your documents, embeddings, and progress never leave your machine unless you explicitly choose a cloud provider.
