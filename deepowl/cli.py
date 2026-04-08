import typer
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

app = typer.Typer(
    name="deepowl",
    help="Local-first AI learning tool.",
    add_completion=False,
)
console = Console()


@app.command()
def ingest(
    path: Path = typer.Argument(..., help="File or folder to ingest"),
    force: bool = typer.Option(False, "--force", "-f", help="Re-index all files even if unchanged"),
):
    """Index documents into the knowledge base."""
    from deepowl.config import load_config, resolve_paths
    from deepowl.ingest.parser import collect_files, parse_file
    from deepowl.ingest.chunker import chunk_document
    from deepowl.ingest.embedder import get_collection, embed_chunks, delete_file_embeddings
    from deepowl.memory.progress import get_db, hash_file, get_file_record, upsert_file, save_chunks

    if not path.exists():
        console.print(f"[red]Path not found:[/red] {path}")
        raise typer.Exit(1)

    config = resolve_paths(load_config())
    embed_model = config["model"]["embedding"]

    conn = get_db(config["storage"]["db_path"])
    collection = get_collection(config["storage"]["chroma_path"])

    files = collect_files(path)
    if not files:
        console.print("[yellow]No supported files found.[/yellow]")
        console.print(f"Supported formats: .pdf .md .txt .docx .epub")
        raise typer.Exit(0)

    console.print(f"\n[bold]deepowl ingest[/bold] — [cyan]{len(files)}[/cyan] file(s) found\n")

    new_count = updated_count = skipped_count = error_count = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Indexing...", total=len(files))

        for file_path in files:
            progress.update(task, description=f"[dim]{file_path.name}[/dim]")

            file_hash = hash_file(file_path)
            existing = get_file_record(conn, str(file_path))
            is_update = existing is not None

            if is_update and existing["hash"] == file_hash and not force:
                skipped_count += 1
                progress.advance(task)
                continue

            try:
                doc = parse_file(file_path)
                chunks = chunk_document(doc.content, str(file_path))

                if is_update:
                    delete_file_embeddings(collection, str(file_path))

                embedding_ids = embed_chunks(chunks, collection, model=embed_model)
                file_id = upsert_file(conn, str(file_path), file_hash)
                save_chunks(conn, file_id, [
                    {"content": c.content, "embedding_id": embedding_ids[i], "index": c.chunk_index}
                    for i, c in enumerate(chunks)
                ])

                if is_update:
                    updated_count += 1
                else:
                    new_count += 1

            except Exception as e:
                console.print(f"\n[red]Error:[/red] {file_path.name} — {e}")
                error_count += 1

            progress.advance(task)

    conn.close()

    console.print(f"\n[green]Done.[/green]  "
                  f"[green]+{new_count} new[/green]  "
                  f"[yellow]~{updated_count} updated[/yellow]  "
                  f"[dim]{skipped_count} skipped[/dim]"
                  + (f"  [red]{error_count} errors[/red]" if error_count else ""))


@app.command()
def build(
    force: bool = typer.Option(False, "--force", "-f", help="Re-extract even already-processed chunks"),
):
    """Extract concepts from indexed documents and build the knowledge graph."""
    from deepowl.config import load_config, resolve_paths
    from deepowl.memory.progress import get_db
    from deepowl.graph.builder import extract_concepts, save_concepts

    config = resolve_paths(load_config())
    model = config["model"]["name"]
    conn = get_db(config["storage"]["db_path"])

    chunks = conn.execute("""
        SELECT c.id, c.content, f.path
        FROM chunks c
        JOIN files f ON c.file_id = f.id
        ORDER BY f.path, c.chunk_index
    """).fetchall()

    if not chunks:
        console.print("[yellow]No chunks found. Run 'deepowl ingest' first.[/yellow]")
        raise typer.Exit(0)

    console.print(f"\n[bold]deepowl build[/bold] — extracting concepts from [cyan]{len(chunks)}[/cyan] chunks\n")

    total_concepts = 0
    errors = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Building graph...", total=len(chunks))

        for chunk in chunks:
            from pathlib import Path as P
            source_name = P(chunk["path"]).name
            progress.update(task, description=f"[dim]{source_name}[/dim]")

            try:
                concepts = extract_concepts(chunk["content"], chunk["path"], model)
                if concepts:
                    save_concepts(conn, concepts)
                    total_concepts += len(concepts)
            except Exception as e:
                console.print(f"\n[red]Error on chunk {chunk['id']}:[/red] {e}")
                errors += 1

            progress.advance(task)

    conn.close()

    console.print(f"\n[green]Done.[/green]  "
                  f"[cyan]{total_concepts}[/cyan] concept(s) extracted"
                  + (f"  [red]{errors} errors[/red]" if errors else ""))
    console.print("[dim]Run 'deepowl start' to begin studying.[/dim]\n")


@app.command()
def start():
    """Start a learning session."""
    from deepowl.config import load_config, resolve_paths
    from deepowl.memory.progress import get_db
    from deepowl.teaching.tutor import run_session

    from deepowl.ingest.embedder import get_collection

    config = resolve_paths(load_config())
    conn = get_db(config["storage"]["db_path"])
    collection = get_collection(config["storage"]["chroma_path"])
    try:
        run_session(
            conn,
            model=config["model"]["name"],
            embed_model=config["model"]["embedding"],
            collection=collection,
        )
    finally:
        conn.close()


@app.command()
def status():
    """Show knowledge base stats."""
    from deepowl.config import load_config, resolve_paths
    from deepowl.memory.progress import get_db, get_stats

    config = resolve_paths(load_config())
    conn = get_db(config["storage"]["db_path"])
    stats = get_stats(conn)
    conn.close()

    console.print(f"\n[bold]deepowl status[/bold]")
    console.print(f"  Files:    [cyan]{stats['files']}[/cyan]")
    console.print(f"  Chunks:   [cyan]{stats['chunks']}[/cyan]")
    console.print(f"  Concepts: [cyan]{stats['concepts']}[/cyan]")
    if stats["concepts"] > 0:
        console.print(f"    [green]✓ done:        {stats['done']}[/green]")
        console.print(f"    [yellow]~ in progress: {stats['concepts'] - stats['done'] - stats['outdated']}[/yellow]")
        if stats["outdated"]:
            console.print(f"    [red]⚠ outdated:    {stats['outdated']}[/red]")
    console.print()


@app.command()
def watch(
    path: Path = typer.Argument(..., help="Folder to watch for changes"),
):
    """Watch a folder and re-index files when they change."""
    import time
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler

    from deepowl.config import load_config, resolve_paths
    from deepowl.ingest.parser import SUPPORTED_EXTENSIONS, parse_file
    from deepowl.ingest.chunker import chunk_document
    from deepowl.ingest.embedder import get_collection, embed_chunks, delete_file_embeddings
    from deepowl.memory.progress import get_db, hash_file, get_file_record, upsert_file, save_chunks
    from deepowl.teaching.curriculum import mark_outdated

    if not path.exists() or not path.is_dir():
        console.print(f"[red]Not a directory:[/red] {path}")
        raise typer.Exit(1)

    config = resolve_paths(load_config())
    embed_model = config["model"]["embedding"]

    def reindex(file_path: Path) -> None:
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            return

        conn = get_db(config["storage"]["db_path"])
        collection = get_collection(config["storage"]["chroma_path"])

        try:
            new_hash = hash_file(file_path)
            existing = get_file_record(conn, str(file_path))

            if existing and existing["hash"] == new_hash:
                return  # unchanged

            console.print(f"[yellow]Changed:[/yellow] {file_path.name} — re-indexing...")

            doc = parse_file(file_path)
            chunks = chunk_document(doc.content, str(file_path))

            if existing:
                delete_file_embeddings(collection, str(file_path))

            embedding_ids = embed_chunks(chunks, collection, model=embed_model)
            file_id = upsert_file(conn, str(file_path), new_hash)
            save_chunks(conn, file_id, [
                {"content": c.content, "embedding_id": embedding_ids[i], "index": c.chunk_index}
                for i, c in enumerate(chunks)
            ])

            outdated = mark_outdated(conn, file_path.name)
            console.print(
                f"[green]Done:[/green] {file_path.name} — "
                f"{len(chunks)} chunks"
                + (f", [red]{outdated} concepts marked outdated[/red]" if outdated else "")
            )
        except Exception as e:
            console.print(f"[red]Error re-indexing {file_path.name}:[/red] {e}")
        finally:
            conn.close()

    class Handler(FileSystemEventHandler):
        def on_modified(self, event):
            if not event.is_directory:
                reindex(Path(event.src_path))

        def on_created(self, event):
            if not event.is_directory:
                reindex(Path(event.src_path))

    observer = Observer()
    observer.schedule(Handler(), str(path), recursive=True)
    observer.start()

    console.print(f"\n[bold]deepowl watch[/bold] — watching [cyan]{path}[/cyan]\n")
    console.print("[dim]Press Ctrl+C to stop.[/dim]\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        console.print("\n[dim]Stopped.[/dim]")

    observer.join()


def main():
    app()


if __name__ == "__main__":
    main()
