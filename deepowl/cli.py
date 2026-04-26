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
    embed_provider = config["model"]["embedding_provider"]
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

                embedding_ids = embed_chunks(chunks, collection, provider=embed_provider, model=embed_model)
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
    provider = config["model"]["provider"]
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
                concepts = extract_concepts(chunk["content"], chunk["path"], provider, model)
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
            provider=config["model"]["provider"],
            model=config["model"]["name"],
            embed_provider=config["model"]["embedding_provider"],
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
def graph(
    topic: str = typer.Option(None, "--topic", "-t", help="Filter by topic keyword"),
):
    """Show the concept graph as a tree in the terminal."""
    from deepowl.config import load_config, resolve_paths
    from deepowl.memory.progress import get_db

    config = resolve_paths(load_config())
    conn = get_db(config["storage"]["db_path"])

    concepts = conn.execute("""
        SELECT c.id, c.name, p.status, p.confidence
        FROM concepts c
        LEFT JOIN progress p ON p.concept_id = c.id
        ORDER BY c.name
    """).fetchall()

    if not concepts:
        console.print("\n[yellow]No concepts yet.[/yellow] Run [bold]deepowl build[/bold] first.\n")
        conn.close()
        return

    relations = conn.execute("""
        SELECT from_concept_id, to_concept_id FROM concept_relations
    """).fetchall()

    if topic:
        keyword = topic.lower()
        matched_ids = {c["id"] for c in concepts if keyword in c["name"].lower()}
        neighbor_ids = set()
        for r in relations:
            if r["from_concept_id"] in matched_ids:
                neighbor_ids.add(r["to_concept_id"])
            if r["to_concept_id"] in matched_ids:
                neighbor_ids.add(r["from_concept_id"])
        allowed_ids = matched_ids | neighbor_ids
        concepts = [c for c in concepts if c["id"] in allowed_ids]

    concept_map = {c["id"]: c for c in concepts}

    children: dict[int, list[int]] = {c["id"]: [] for c in concepts}
    has_parent: set[int] = set()
    for r in relations:
        fid, tid = r["from_concept_id"], r["to_concept_id"]
        if fid in children and tid in children:
            children[fid].append(tid)
            has_parent.add(tid)

    def style(c) -> tuple[str, str]:
        s = c["status"] or "not_started"
        conf = c["confidence"] or 0
        if s == "done":
            return "green", f"✓ {conf}%"
        elif s == "in_progress":
            return "yellow", f"~ {conf}%"
        elif s == "outdated":
            return "red", "⚠"
        else:
            return "dim", "·"

    def print_tree(concept_id: int, prefix: str, is_last: bool, visited: set) -> None:
        if concept_id in visited:
            return
        visited = visited | {concept_id}
        c = concept_map[concept_id]
        color, badge = style(c)
        connector = "└── " if is_last else "├── "
        console.print(f"{prefix}{connector}[{color}]{c['name']}[/{color}] [dim]{badge}[/dim]")
        kids = children.get(concept_id, [])
        extension = "    " if is_last else "│   "
        for i, kid_id in enumerate(kids):
            print_tree(kid_id, prefix + extension, i == len(kids) - 1, visited)

    roots = [c for c in concepts if c["id"] not in has_parent]
    if not roots:
        roots = list(concepts)

    title = f"[bold]deepowl graph[/bold]"
    if topic:
        title += f" [dim]· filter: {topic}[/dim]"
    console.print(f"\n{title}  [dim]({len(concepts)} concepts)[/dim]\n")
    console.print("[dim]  [green]✓[/green] done  [yellow]~[/yellow] in progress  · not started  [red]⚠[/red] outdated[/dim]\n")

    for i, root in enumerate(roots):
        is_last = i == len(roots) - 1
        c = root
        color, badge = style(c)
        connector = "└── " if is_last else "├── "
        console.print(f"{connector}[{color}]{c['name']}[/{color}] [dim]{badge}[/dim]")
        kids = children.get(c["id"], [])
        extension = "    " if is_last else "│   "
        for j, kid_id in enumerate(kids):
            print_tree(kid_id, extension, j == len(kids) - 1, {c["id"]})

    console.print()
    conn.close()


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
    embed_provider = config["model"]["embedding_provider"]
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
                return

            console.print(f"[yellow]Changed:[/yellow] {file_path.name} — re-indexing...")

            doc = parse_file(file_path)
            chunks = chunk_document(doc.content, str(file_path))

            if existing:
                delete_file_embeddings(collection, str(file_path))

            embedding_ids = embed_chunks(chunks, collection, provider=embed_provider, model=embed_model)
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


# ── Config command ────────────────────────────────────────────────────────────

config_app = typer.Typer(help="View and edit deepowl configuration.")
app.add_typer(config_app, name="config")


@config_app.command("show")
def config_show():
    """Print the current configuration."""
    from deepowl.config import load_config, CONFIG_FILE
    import yaml

    cfg = load_config()
    console.print(f"\n[dim]{CONFIG_FILE}[/dim]\n")
    console.print(yaml.dump(cfg, default_flow_style=False).rstrip())
    console.print()


@config_app.command("set")
def config_set(
    key: str = typer.Argument(..., help="Dot-notation key, e.g. model.provider"),
    value: str = typer.Argument(..., help="New value"),
):
    """Set a configuration value. Use dot notation for nested keys."""
    from deepowl.config import load_config, CONFIG_FILE
    from deepowl.llm import PROVIDER_DEFAULTS
    import yaml

    cfg = load_config()

    # Navigate to parent key
    parts = key.split(".")
    node = cfg
    for part in parts[:-1]:
        if part not in node:
            console.print(f"[red]Key not found:[/red] {key}")
            raise typer.Exit(1)
        node = node[part]

    leaf = parts[-1]
    if leaf not in node:
        console.print(f"[red]Key not found:[/red] {key}")
        raise typer.Exit(1)

    # Coerce type to match existing value
    old = node[leaf]
    if isinstance(old, bool):
        node[leaf] = value.lower() in ("true", "1", "yes")
    elif isinstance(old, int):
        node[leaf] = int(value)
    else:
        node[leaf] = value

    # When changing provider, suggest defaults
    if key == "model.provider" and value in PROVIDER_DEFAULTS:
        defaults = PROVIDER_DEFAULTS[value]
        cfg["model"].setdefault("name", defaults["name"])
        cfg["model"]["name"] = defaults["name"]
        cfg["model"]["embedding"] = defaults["embedding"]
        cfg["model"]["embedding_provider"] = defaults["embedding_provider"]
        console.print(
            f"[dim]Applied defaults for {value}: "
            f"name={defaults['name']}, "
            f"embedding={defaults['embedding']} ({defaults['embedding_provider']})[/dim]"
        )

    CONFIG_FILE.write_text(yaml.dump(cfg, default_flow_style=False))
    console.print(f"[green]Set[/green] {key} = {node[leaf]}")


@config_app.command("get")
def config_get(
    key: str = typer.Argument(..., help="Dot-notation key, e.g. model.provider"),
):
    """Get a single configuration value."""
    from deepowl.config import load_config

    cfg = load_config()
    parts = key.split(".")
    node = cfg
    for part in parts:
        if not isinstance(node, dict) or part not in node:
            console.print(f"[red]Key not found:[/red] {key}")
            raise typer.Exit(1)
        node = node[part]
    console.print(node)


def main():
    app()


if __name__ == "__main__":
    main()
