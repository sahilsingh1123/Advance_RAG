"""CLI for Advance RAG."""

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import typer
from typing import Optional

from advance_rag.core.logging import configure_logging, get_logger

app = typer.Typer(name="advance-rag", help="Advance RAG CLI")
logger = get_logger(__name__)


@app.command()
def generate_data(
    study_id: str = typer.Option("STUDY001", help="Study ID"),
    n_subjects: int = typer.Option(100, help="Number of subjects"),
    output_dir: str = typer.Option("data/dummy", help="Output directory"),
):
    """Generate dummy clinical data."""
    configure_logging()

    from scripts.generate_dummy_data import (
        ClinicalDataGenerator as ClinicalDataGenerator,
    )

    generator = ClinicalDataGenerator(study_id=study_id, n_subjects=n_subjects)
    generator.DATA_DIR = Path(output_dir)
    generator.DATA_DIR.mkdir(parents=True, exist_ok=True)
    generator.save_data()

    typer.echo(f"Generated dummy data for {study_id} with {n_subjects} subjects")


@app.command()
def ingest(
    data_path: str = typer.Argument(..., help="Path to data file or directory"),
    recursive: bool = typer.Option(False, help="Search recursively"),
    verbose: bool = typer.Option(False, help="Verbose logging"),
):
    """Ingest data into the RAG system."""
    import asyncio

    if verbose:
        configure_logging()

    async def _ingest():
        from scripts.ingest_data import main as ingest_main
        import sys

        # Mock sys.argv
        sys.argv = ["ingest_data.py", "--data-dir", data_path]
        if recursive:
            sys.argv.append("--recursive")

        await ingest_main()

    asyncio.run(_ingest())


@app.command()
def init_db():
    """Initialize databases."""
    configure_logging()

    # Initialize PostgreSQL
    typer.echo("Initializing PostgreSQL...")
    import subprocess

    result = subprocess.run(
        ["psql", "-f", "scripts/init_postgres.sql"], capture_output=True, text=True
    )
    if result.returncode == 0:
        typer.echo("✓ PostgreSQL initialized")
    else:
        typer.echo(f"✗ PostgreSQL initialization failed: {result.stderr}")

    # Initialize Neo4j
    typer.echo("Initializing Neo4j...")
    result = subprocess.run(
        ["python", "scripts/init_neo4j.py"], capture_output=True, text=True
    )
    if result.returncode == 0:
        typer.echo("✓ Neo4j initialized")
    else:
        typer.echo(f"✗ Neo4j initialization failed: {result.stderr}")


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Host to bind to"),
    port: int = typer.Option(8000, help="Port to bind to"),
    reload: bool = typer.Option(False, help="Enable auto-reload"),
    workers: int = typer.Option(1, help="Number of worker processes"),
):
    """Start the API server."""
    import uvicorn

    configure_logging()

    uvicorn.run(
        "advance_rag.api.main:app",
        host=host,
        port=port,
        reload=reload,
        workers=workers if not reload else 1,
        log_level="info",
    )


@app.command()
def worker(
    loglevel: str = typer.Option("info", help="Log level"),
    concurrency: int = typer.Option(4, help="Number of concurrent processes"),
):
    """Start Celery worker."""
    from advance_rag.core.celery import celery_app

    configure_logging()

    celery_app.worker_main(
        [
            "worker",
            f"--loglevel={loglevel}",
            f"--concurrency={concurrency}",
            "--without-gossip",
            "--without-mingle",
            "--without-heartbeat",
        ]
    )


@app.command()
def beat():
    """Start Celery beat scheduler."""
    from advance_rag.core.celery import celery_app

    configure_logging()

    celery_app.worker_main(["beat", "--loglevel=info"])


@app.command()
def status():
    """Check system status."""
    configure_logging()

    import asyncio

    async def _check_status():
        from advance_rag.db.vector_store import VectorStore
        from advance_rag.graph.neo4j_service import Neo4jService

        # Check vector store
        try:
            vs = VectorStore()
            await vs.initialize()
            stats = await vs.get_statistics()
            typer.echo(f"✓ Vector Store: {stats['total_chunks']} chunks")
            await vs.close()
        except Exception as e:
            typer.echo(f"✗ Vector Store: {e}")

        # Check Neo4j
        try:
            neo4j = Neo4jService()
            stats = await neo4j.get_graph_statistics()
            typer.echo(
                f"✓ Neo4j: {stats['total_entities']} entities, {stats['total_relations']} relations"
            )
            await neo4j.close()
        except Exception as e:
            typer.echo(f"✗ Neo4j: {e}")

    asyncio.run(_check_status())


if __name__ == "__main__":
    app()
