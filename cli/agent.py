import argparse
import datetime
import os
import sys
from typing import Any, Dict, List

import openai
from rich.console import Console
from rich.table import Table

# Add the project root to sys.path so we can import sdk
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sdk.memory import HybridMemory

# Auto-load .env file from the root directory so the user doesn't have to set env vars manually
env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
if os.path.isfile(env_path):
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, _, value = line.partition("=")
            if key and value:
                os.environ.setdefault(key.strip(), value.strip().strip('\'"'))


def build_system_prompt(session_mems: List[Dict[str, Any]], cross_mems: List[Dict[str, Any]]) -> str:
    parts = ["You are a helpful assistant with persistent memory.\n"]

    if session_mems:
        parts.append("RECALLED MEMORIES (from this session):")
        for m in session_mems:
            score = m.get("score", m.get("vector_score", 0.0))
            text = (m.get("text") or "")[:200].replace("\n", " ")
            parts.append(f"  [{score:.2f}] {text}")
        parts.append("")

    if cross_mems:
        parts.append("RECALLED MEMORIES (from past sessions):")
        for m in cross_mems:
            score = m.get("score", m.get("vector_score", 0.0))
            text = (m.get("text") or "")[:200].replace("\n", " ")
            parts.append(f"  [{score:.2f}] {text}")
        parts.append("")

    if session_mems or cross_mems:
        parts.append("Use these memories naturally when relevant. Do not announce that you are using memory. Just use it.")
        return "\n".join(parts)
    else:
        return "You are a helpful assistant with persistent memory."


def main():
    parser = argparse.ArgumentParser(description="HybridMind CLI Agent")
    parser.add_argument("--model", default="x-ai/grok-4.3", help="LLM model name")
    parser.add_argument("--memory-url", default="http://localhost:8000", help="HybridMind base URL")
    parser.add_argument("--top-k", type=int, default=5, help="Number of memories to recall per turn")
    parser.add_argument("--no-memory", action="store_true", help="Disable HybridMind entirely")
    parser.add_argument("--session", help="Resume a specific session_id")
    args = parser.parse_args()

    console = Console()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        console.print("[bold red]Error: OPENAI_API_KEY environment variable not set.[/bold red]")
        sys.exit(1)

    base_url = os.environ.get("OPENAI_BASE_URL", "https://ai.hackclub.com/proxy/v1")
    client = openai.OpenAI(api_key=api_key, base_url=base_url)

    memory = None
    session_id = None
    turn_number = 1
    messages = []
    last_recalled = []

    if not args.no_memory:
        try:
            memory = HybridMemory(base_url=args.memory_url)
            stats = memory.stats()
            node_count = stats.get("node_count", 0)
            edge_count = stats.get("edge_count", 0)

            if args.session:
                session_id = args.session
            else:
                now_iso = datetime.datetime.utcnow().isoformat()
                session_data = memory.session.create(name=now_iso)
                session_id = session_data["session_id"]

            console.print("[bold green]HybridMind CLI Agent[/bold green]")
            console.print(f"Session: {session_id}")
            console.print(f"Memory: {node_count} nodes, {edge_count} edges")
            console.print(f"Model: {args.model}")
            console.print("Type /help for commands.\n")

        except Exception as e:
            console.print(f"[bold yellow]Warning: Could not connect to HybridMind at {args.memory_url}. Running with memory disabled.[/bold yellow]")
            console.print(f"Error: {e}\n")
            memory = None
    else:
        console.print("[bold green]HybridMind CLI Agent (Memory Disabled)[/bold green]")
        console.print(f"Model: {args.model}")
        console.print("Type /help for commands.\n")

    while True:
        try:
            user_input = console.input("[bold blue]> [/bold blue]")
        except (KeyboardInterrupt, EOFError):
            console.print()
            break

        if not user_input.strip():
            continue

        if user_input.startswith("/"):
            cmd = user_input.split(" ", 1)
            command = cmd[0].lower()
            arg = cmd[1] if len(cmd) > 1 else ""

            if command in ("/exit", "/quit"):
                break
            elif command == "/help":
                console.print("Commands:")
                console.print("  /memory          → show last recalled memories")
                console.print("  /stats           → show memory stats")
                console.print("  /sessions        → list sessions")
                console.print("  /archive         → archive current session and exit")
                console.print("  /forget <text>   → forget closest memory")
                console.print("  /clear           → clear terminal")
                console.print("  /help            → show commands")
                console.print("  /exit or /quit   → exit")
            elif command == "/clear":
                os.system("cls" if os.name == "nt" else "clear")
            elif command == "/memory":
                if not memory:
                    console.print("[yellow]Memory is disabled.[/yellow]")
                elif not last_recalled:
                    console.print("No memories recalled in the last turn.")
                else:
                    table = Table(show_header=True, header_style="bold magenta")
                    table.add_column("Score", justify="right")
                    table.add_column("Node ID")
                    table.add_column("Preview")
                    for m in last_recalled:
                        score = m.get("score", m.get("vector_score", m.get("hybrid_score", 0.0)))
                        nid = m.get("node_id", m.get("id", "unknown"))
                        text = (m.get("text") or "")[:50].replace("\n", " ")
                        table.add_row(f"{score:.2f}", nid, text)
                    console.print(table)
            elif command == "/stats":
                if not memory:
                    console.print("[yellow]Memory is disabled.[/yellow]")
                else:
                    try:
                        st = memory.stats()
                        console.print(f"Nodes: {st.get('node_count')}")
                        console.print(f"Edges: {st.get('edge_count')}")
                        if "most_connected_nodes" in st:
                            console.print("\nTop Connected Nodes:")
                            for n in st["most_connected_nodes"]:
                                console.print(f"  {n['node_id']} (degree {n['degree']}): {n['text_preview'][:60]}...")
                    except Exception as e:
                        console.print(f"[red]Error fetching stats: {e}[/red]")
            elif command == "/sessions":
                if not memory:
                    console.print("[yellow]Memory is disabled.[/yellow]")
                else:
                    try:
                        sessions = memory.session.list()
                        for s in sessions:
                            console.print(f"Session {s.get('session_id')} ({s.get('status')}): {s.get('node_count')} nodes")
                    except Exception as e:
                        console.print(f"[red]Error fetching sessions: {e}[/red]")
            elif command == "/archive":
                if not memory:
                    console.print("[yellow]Memory is disabled.[/yellow]")
                elif not session_id:
                    console.print("[yellow]No active session to archive.[/yellow]")
                else:
                    try:
                        memory.session.archive(session_id)
                        console.print("Session archived successfully.")
                        break
                    except Exception as e:
                        console.print(f"[red]Error archiving session: {e}[/red]")
            elif command == "/forget":
                if not memory:
                    console.print("[yellow]Memory is disabled.[/yellow]")
                elif not arg:
                    console.print("[yellow]Usage: /forget <text>[/yellow]")
                else:
                    try:
                        res = memory.recall(query=arg, top_k=1, mode="vector")
                        if not res:
                            console.print("No matching memories found.")
                        else:
                            node = res[0]
                            nid = node.get("node_id", node.get("id"))
                            text = (node.get("text") or "")[:100]
                            conf = console.input(f"Forget node {nid}? \n'{text}'\n[y/N]: ")
                            if conf.lower() == "y":
                                memory.forget(nid)
                                console.print("Forgot.")
                    except Exception as e:
                        console.print(f"[red]Error forgetting memory: {e}[/red]")
            else:
                console.print("[yellow]Unknown command. Type /help for options.[/yellow]")
            continue

        session_mems = []
        cross_mems = []
        last_recalled = []

        if memory and session_id:
            try:
                session_mems = memory.session.recall(query=user_input, session_id=session_id, top_k=args.top_k)
            except Exception as e:
                print(f"Memory recall warning (session): {e}", file=sys.stderr)

            try:
                cross_mems_raw = memory.recall(query=user_input, top_k=3, mode="hybrid")
                session_nids = {m.get("node_id", m.get("id")) for m in session_mems}
                cross_mems = [m for m in cross_mems_raw if m.get("node_id", m.get("id")) not in session_nids]
            except Exception as e:
                print(f"Memory recall warning (cross): {e}", file=sys.stderr)

            last_recalled = session_mems + cross_mems

        system_prompt = build_system_prompt(session_mems, cross_mems)
        curr_messages = [{"role": "system", "content": system_prompt}] + messages + [{"role": "user", "content": user_input}]

        try:
            stream = client.chat.completions.create(
                model=args.model,
                messages=curr_messages,
                stream=True,
            )

            assistant_response = ""
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                print(delta, end="", flush=True)
                assistant_response += delta
            print()

            messages.append({"role": "user", "content": user_input})
            messages.append({"role": "assistant", "content": assistant_response})

            if memory and session_id:
                try:
                    memory.store_with_auto_edges(
                        text=user_input,
                        metadata={"role": "user", "turn": turn_number},
                        session_id=session_id
                    )
                    memory.store_with_auto_edges(
                        text=assistant_response,
                        metadata={"role": "assistant", "turn": turn_number},
                        session_id=session_id
                    )
                    console.print(f"[dim][memory: {len(last_recalled)} recalled, 2 stored][/dim]")
                except Exception as e:
                    print(f"Memory store warning: {e}", file=sys.stderr)

            turn_number += 1

        except Exception as e:
            console.print(f"[bold red]Error: {e}[/bold red]")


if __name__ == "__main__":
    main()
