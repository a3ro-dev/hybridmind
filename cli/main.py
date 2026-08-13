"""
HybridMind CLI - Command line interface for database operations.
"""

import json
import os
import sys
from typing import Optional, List
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

# CLI app
app = typer.Typer(
    name="hybridmind",
    help="HybridMind - Vector + Graph Native Database for AI Retrieval",
    add_completion=False,
)

console = Console()


def get_client():
    """Get HTTP client for API calls."""
    try:
        import httpx

        api_key = os.environ.get("HYBRIDMIND_API_KEY", "").strip()
        headers = {"X-HybridMind-API-Key": api_key} if api_key else None
        return httpx.Client(
            base_url="http://localhost:8000", timeout=30.0, headers=headers
        )
    except ImportError:
        console.print("[red]httpx not installed. Run: pip install httpx[/red]")
        raise typer.Exit(1)


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query text"),
    mode: str = typer.Option(
        "hybrid", "--mode", "-m", help="Search mode: vector, graph, hybrid"
    ),
    top_k: int = typer.Option(10, "--top-k", "-k", help="Number of results"),
    vector_weight: float = typer.Option(
        0.6, "--vector-weight", "-vw", help="Vector weight for hybrid search"
    ),
    graph_weight: float = typer.Option(
        0.4, "--graph-weight", "-gw", help="Graph weight for hybrid search"
    ),
    anchor: Optional[str] = typer.Option(
        None, "--anchor", "-a", help="Anchor node ID for hybrid search"
    ),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
):
    """
    Search the database using vector, graph, or hybrid mode.
    """
    client = get_client()

    try:
        if mode == "vector":
            response = client.post(
                "/search/vector", json={"query_text": query, "top_k": top_k}
            )
        elif mode == "graph":
            if not anchor:
                console.print("[red]Graph search requires --anchor node ID[/red]")
                raise typer.Exit(1)
            response = client.get(
                "/search/graph", params={"start_id": anchor, "depth": 2}
            )
        else:  # hybrid
            payload = {
                "query_text": query,
                "top_k": top_k,
                "vector_weight": vector_weight,
                "graph_weight": graph_weight,
            }
            if anchor:
                payload["anchor_nodes"] = [anchor]
            response = client.post("/search/hybrid", json=payload)

        response.raise_for_status()
        data = response.json()

        if json_output:
            console.print_json(json.dumps(data, indent=2))
            return

        # Display results
        console.print(
            Panel(f"[bold]Query:[/bold] {query}", title=f"{mode.upper()} Search")
        )
        console.print(
            f"[dim]Query time: {data['query_time_ms']:.2f}ms | Candidates: {data['total_candidates']}[/dim]\n"
        )

        if not data["results"]:
            console.print("[yellow]No results found[/yellow]")
            return

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("#", style="dim", width=3)
        table.add_column("Node ID", width=12)
        table.add_column("Text", width=50)
        table.add_column("Score", justify="right", width=10)

        for i, result in enumerate(data["results"], 1):
            score = (
                result.get("combined_score")
                or result.get("vector_score")
                or result.get("graph_score")
                or 0
            )
            text = (
                result["text"][:80] + "..."
                if len(result["text"]) > 80
                else result["text"]
            )
            table.add_row(str(i), result["node_id"][:12], text, f"{score:.3f}")

        console.print(table)

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def compare(
    query: str = typer.Argument(..., help="Search query text"),
    anchor: Optional[str] = typer.Option(None, "--anchor", "-a", help="Anchor node ID"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of results per mode"),
):
    """
    Compare vector, graph, and hybrid search results side-by-side.
    """
    client = get_client()

    try:
        payload = {
            "query_text": query,
            "top_k": top_k,
            "vector_weight": 0.6,
            "graph_weight": 0.4,
        }
        if anchor:
            payload["anchor_nodes"] = [anchor]

        response = client.post("/search/compare", json=payload)
        response.raise_for_status()
        data = response.json()

        console.print(Panel(f"[bold]Query:[/bold] {query}", title="Search Comparison"))

        # Vector results
        console.print("\n[bold cyan]Vector-Only Results:[/bold cyan]")
        for i, r in enumerate(data["vector_only"]["results"][:5], 1):
            console.print(f"  {i}. [{r['score']:.3f}] {r['text'][:60]}...")

        # Graph results
        console.print("\n[bold green]Graph-Only Results:[/bold green]")
        if data["graph_only"]["results"]:
            for i, r in enumerate(data["graph_only"]["results"][:5], 1):
                console.print(f"  {i}. [depth={r.get('depth', 0)}] {r['text'][:60]}...")
        else:
            console.print("  [dim]No anchor provided or no graph connections[/dim]")

        # Hybrid results
        console.print("\n[bold magenta]Hybrid Results:[/bold magenta]")
        for i, r in enumerate(data["hybrid"]["results"][:5], 1):
            console.print(
                f"  {i}. [V:{r['vector_score']:.2f} G:{r['graph_score']:.2f} = {r['combined_score']:.3f}] {r['text'][:50]}..."
            )

        # Analysis
        analysis = data["analysis"]
        console.print("\n[bold]Analysis:[/bold]")
        console.print(f"  Hybrid finds {analysis['hybrid_unique']} unique results")
        console.print(f"  Vector-only unique: {analysis['vector_unique']}")
        console.print(f"  Graph-only unique: {analysis['graph_unique']}")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def stats():
    """
    Show database statistics.
    """
    client = get_client()

    try:
        response = client.get("/search/stats")
        response.raise_for_status()
        data = response.json()

        console.print(Panel("[bold]HybridMind Database Statistics[/bold]"))

        table = Table(show_header=False)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total Nodes", str(data["total_nodes"]))
        table.add_row("Total Edges", str(data["total_edges"]))
        table.add_row("Avg Edges/Node", f"{data['avg_edges_per_node']:.2f}")
        table.add_row("Vector Index Size", str(data["vector_index_size"]))
        table.add_row("Database Size", f"{data['database_size_bytes'] / 1024:.1f} KB")

        console.print(table)

        if data["edge_types"]:
            console.print("\n[bold]Edge Types:[/bold]")
            for edge_type, count in data["edge_types"].items():
                console.print(f"  {edge_type}: {count}")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def add_node(
    text: str = typer.Argument(..., help="Node text content"),
    title: Optional[str] = typer.Option(None, "--title", "-t", help="Node title"),
    tags: Optional[str] = typer.Option(None, "--tags", help="Comma-separated tags"),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
):
    """
    Add a new node to the database.
    """
    client = get_client()

    metadata = {}
    if title:
        metadata["title"] = title
    if tags:
        metadata["tags"] = [t.strip() for t in tags.split(",")]

    try:
        response = client.post("/nodes", json={"text": text, "metadata": metadata})
        response.raise_for_status()
        data = response.json()

        if json_output:
            console.print_json(json.dumps(data, indent=2))
        else:
            console.print(f"[green]✓ Node created: {data['id']}[/green]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def add_edge(
    source: str = typer.Argument(..., help="Source node ID"),
    target: str = typer.Argument(..., help="Target node ID"),
    edge_type: str = typer.Option("related_to", "--type", "-t", help="Edge type"),
    weight: float = typer.Option(1.0, "--weight", "-w", help="Edge weight (0-1)"),
):
    """
    Add an edge between two nodes.
    """
    client = get_client()

    try:
        response = client.post(
            "/edges",
            json={
                "source_id": source,
                "target_id": target,
                "type": edge_type,
                "weight": weight,
            },
        )
        response.raise_for_status()
        data = response.json()

        console.print(f"[green]✓ Edge created: {data['id']}[/green]")
        console.print(f"  {source[:8]}... --[{edge_type}]--> {target[:8]}...")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def get_node(
    node_id: str = typer.Argument(..., help="Node ID"),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
):
    """
    Get node details by ID.
    """
    client = get_client()

    try:
        response = client.get(f"/nodes/{node_id}")
        response.raise_for_status()
        data = response.json()

        if json_output:
            console.print_json(json.dumps(data, indent=2))
            return

        console.print(Panel(f"[bold]Node: {data['id']}[/bold]"))
        console.print(f"[bold]Text:[/bold] {data['text'][:200]}...")
        console.print(
            f"[bold]Metadata:[/bold] {json.dumps(data['metadata'], indent=2)}"
        )

        if data["edges"]:
            console.print(f"\n[bold]Edges ({len(data['edges'])}):[/bold]")
            for edge in data["edges"]:
                console.print(f"  → [{edge['type']}] {edge['target_id'][:12]}...")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Host to bind to"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to bind to"),
    reload: bool = typer.Option(False, "--reload", "-r", help="Enable auto-reload"),
):
    """
    Start the HybridMind API server.
    """
    try:
        import ipaddress
        import uvicorn

        try:
            local_bind = ipaddress.ip_address(host).is_loopback
        except ValueError:
            local_bind = host.lower() == "localhost"
        if not local_bind and not os.environ.get("HYBRIDMIND_API_KEY", "").strip():
            console.print("[red]A non-loopback bind requires HYBRIDMIND_API_KEY.[/red]")
            raise typer.Exit(2)
        console.print(f"[green]Starting HybridMind server on {host}:{port}[/green]")
        console.print(
            f"[dim]API docs: http://{host if host != '0.0.0.0' else 'localhost'}:{port}/docs[/dim]"
        )
        uvicorn.run("main:app", host=host, port=port, reload=reload)
    except ImportError:
        console.print(
            "[red]uvicorn not installed. Run: pip install uvicorn[standard][/red]"
        )
        raise typer.Exit(1)


@app.command()
def load_demo():
    """
    Load demo dataset (research papers).
    """
    import subprocess
    import sys

    console.print("[cyan]Loading demo dataset...[/cyan]")

    try:
        result = subprocess.run(
            [sys.executable, "data/load_demo_data.py"],
            capture_output=True,
            text=True,
            cwd=".",
        )
        console.print(result.stdout)
        if result.stderr:
            console.print(f"[yellow]{result.stderr}[/yellow]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
