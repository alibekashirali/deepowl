import re
import sys
import sqlite3
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.rule import Rule

from deepowl.llm import call_llm, get_embedding

console = Console()


# ── RAG ───────────────────────────────────────────────────────────────────────

def _retrieve_chunks(query: str, embed_provider: str, embed_model: str, collection, n: int = 4) -> list[dict]:
    embedding = get_embedding(embed_provider, embed_model, query)
    results = collection.query(
        query_embeddings=[embedding],
        n_results=n,
        include=["documents", "metadatas"],
    )
    if not results["ids"][0]:
        return []
    return [
        {
            "content": results["documents"][0][i],
            "source": Path(results["metadatas"][0][i]["source"]).name,
            "chunk_index": results["metadatas"][0][i]["chunk_index"],
        }
        for i in range(len(results["ids"][0]))
    ]


def _format_context(chunks: list[dict]) -> str:
    parts = []
    for c in chunks:
        parts.append(f"[source: {c['source']}, chunk {c['chunk_index']}]\n{c['content']}")
    return "\n\n---\n\n".join(parts)


def _extract_score(text: str) -> int:
    match = re.search(r"score[:\s]+(\d+)", text, re.IGNORECASE)
    return min(100, max(0, int(match.group(1)))) if match else 50


def _rag_answer(question: str, provider: str, model: str, embed_provider: str, embed_model: str, collection) -> None:
    chunks = _retrieve_chunks(question, embed_provider, embed_model, collection)
    if not chunks:
        console.print("[dim]No relevant content found in your docs.[/dim]")
        return
    context = _format_context(chunks)
    console.print("\n[bold magenta]■ Answer[/bold magenta]\n")
    call_llm(
        provider,
        model,
        system=(
            "Answer the student's question using ONLY the provided context from their documents. "
            "Be concise. Always cite sources as [source: file, chunk N]."
        ),
        user=f"Question: {question}\n\nContext:\n{context}",
    )


# ── Session ───────────────────────────────────────────────────────────────────

def run_session(
    conn: sqlite3.Connection,
    provider: str,
    model: str,
    embed_provider: str,
    embed_model: str,
    collection,
) -> None:
    from deepowl.graph.builder import build_graph
    from deepowl.teaching.curriculum import next_concept, update_progress, get_progress_summary

    graph = build_graph(conn)
    summary = get_progress_summary(conn)
    total_concepts = sum(summary.values())

    if total_concepts == 0:
        console.print(
            "\n[yellow]No concepts found.[/yellow] "
            "Run [bold]deepowl build[/bold] first to extract concepts from your docs.\n"
        )
        return

    console.print(
        f"\n[bold]deepowl[/bold] — "
        f"[green]{summary['done']}[/green] done  "
        f"[yellow]{summary['in_progress']}[/yellow] in progress  "
        f"[dim]{summary['not_started']}[/dim] not started"
        + (f"  [red]{summary['outdated']} outdated[/red]" if summary["outdated"] else "")
        + f"\n[dim]model: {provider}/{model}[/dim]\n"
    )

    session_start = datetime.now()
    concepts_done = 0

    try:
        while True:
            concept = next_concept(conn, graph)
            if not concept:
                console.print("\n[green]All concepts studied![/green] Add more docs to keep learning.")
                break

            name = concept["name"]
            desc = concept.get("description", "")
            concept_id = concept["id"]

            console.print(Rule(f"[bold]{name}[/bold]", style="cyan"))
            if desc:
                console.print(f"[dim]{desc}[/dim]\n")

            chunks = _retrieve_chunks(name, embed_provider, embed_model, collection)
            context = _format_context(chunks)

            if not chunks:
                console.print("[dim]No relevant chunks found — skipping.[/dim]")
                update_progress(conn, concept_id, 0)
                continue

            # ── Explain ──
            console.print("\n[bold cyan]■ Explanation[/bold cyan]\n")
            explanation = call_llm(
                provider,
                model,
                system=(
                    "You are a concise, clear tutor. Explain the concept using ONLY the provided context. "
                    "Keep it under 150 words. Cite sources as [source: file, chunk N]."
                ),
                user=f"Concept: {name}\n\nContext:\n{context}",
            )

            console.print("\n[dim]Press Enter to continue...[/dim]", end="")
            input()

            # ── Question ──
            console.print("\n[bold yellow]■ Question[/bold yellow]\n")
            question = call_llm(
                provider,
                model,
                system=(
                    "Ask ONE specific question to check the student's understanding of the concept. "
                    "Make it focused — not trivial, not too hard. Ask only the question, nothing else."
                ),
                user=f"Concept: {name}\n\nExplanation:\n{explanation}\n\nContext:\n{context}",
            )

            console.print()
            answer = input("> ").strip()

            if not answer:
                console.print("[dim]Skipped.[/dim]")
                update_progress(conn, concept_id, 0)
                concepts_done += 1
            else:
                # ── Evaluate ──
                console.print("\n[bold green]■ Feedback[/bold green]\n")
                evaluation = call_llm(
                    provider,
                    model,
                    system=(
                        "Evaluate the student's answer briefly (1-2 sentences). "
                        "Be encouraging but honest. End with 'Score: N' where N is 0-100."
                    ),
                    user=(
                        f"Concept: {name}\n\n"
                        f"Context:\n{context}\n\n"
                        f"Question: {question}\n\n"
                        f"Student's answer: {answer}"
                    ),
                )

                score = _extract_score(evaluation)
                update_progress(conn, concept_id, score)
                concepts_done += 1

                bar = "█" * (score // 10) + "░" * (10 - score // 10)
                console.print(f"\n[dim][{bar}] {score}/100[/dim]")

            graph = build_graph(conn)

            while True:
                console.print("\n[dim]Enter to continue · q to quit · ? to ask[/dim]  ", end="")
                cmd = input().strip().lower()
                if cmd == "q":
                    break
                elif cmd.startswith("?"):
                    q = cmd[1:].strip() or input("Your question: ").strip()
                    if q:
                        _rag_answer(q, provider, model, embed_provider, embed_model, collection)
                else:
                    break
            if cmd == "q":
                break

    except KeyboardInterrupt:
        console.print("\n\n[dim]Session interrupted.[/dim]")

    conn.execute(
        "INSERT INTO sessions (started_at, ended_at, concepts_covered) VALUES (?, ?, ?)",
        (session_start.isoformat(), datetime.now().isoformat(), str(concepts_done)),
    )
    conn.commit()

    console.print(
        f"\n[bold]Session done.[/bold] "
        f"Studied [cyan]{concepts_done}[/cyan] concept(s) this session."
    )
