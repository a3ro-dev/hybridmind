"""
HybridMind Native Installer & Configurator.

Automates:
1. Virtual environment setup & dependency installation.
2. Default .env configuration.
3. Automatic integration with AI agents/IDEs:
   - Claude Desktop (Windows/macOS config directories)
   - Claude Code (Global mcp.json)
   - Windsurf IDE (Global mcp_config.json)
4. OpenAPI schema generation for ChatGPT Custom GPT integration.
"""
from __future__ import annotations

import json
import os
import sys
import subprocess
import shutil
from pathlib import Path

# Ensure standard output formatting
def print_step(msg: str):
    print(f"\n[\033[94m*\033[0m] {msg}")

def print_success(msg: str):
    print(f"[\033[92m✓\033[0m] {msg}")

def print_warning(msg: str):
    print(f"[\033[93m!\033[0m] {msg}")

def print_error(msg: str):
    print(f"[\033[91m✗\033[0m] {msg}")

# ─── Config Constants ────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.resolve()
VENV_DIR = PROJECT_ROOT / ".venv"
VENV_PIP = VENV_DIR / ("Scripts" if os.name == "nt" else "bin") / ("pip.exe" if os.name == "nt" else "pip")
VENV_PYTHON = VENV_DIR / ("Scripts" if os.name == "nt" else "bin") / ("python.exe" if os.name == "nt" else "python")

# ─── Step 1: Venv & Requirements ─────────────────────────────────────────────

def setup_venv():
    print_step("Setting up Python virtual environment...")
    if not VENV_DIR.exists():
        print("Creating .venv...")
        subprocess.run([sys.executable, "-m", "venv", str(VENV_DIR)], check=True)
        print_success("Virtual environment created.")
    else:
        print_success(".venv already exists.")

    print("Installing/upgrading requirements...")
    # Run pip install -r requirements.txt
    r = subprocess.run([str(VENV_PIP), "install", "-r", str(PROJECT_ROOT / "requirements.txt")], capture_output=True, text=True)
    if r.returncode != 0:
        print_error("Failed to install requirements:")
        print(r.stderr)
        sys.exit(1)
    print_success("Requirements installed successfully.")

# ─── Step 2: Env Configuration ───────────────────────────────────────────────

def setup_env():
    print_step("Configuring environment variables (.env)...")
    env_file = PROJECT_ROOT / ".env"
    
    default_content = """# HybridMind Config
OSM_API_KEY=your_osm_api_key_here
HC_API_KEY=your_hackclub_ai_key_here
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_BASE_URL="https://ai.hackclub.com/proxy/v1"
HC_EMBEDDING_URL=https://ai.hackclub.com/proxy/v1
FACT_EXTRACTION_ENABLED=true
HYBRIDMIND_AUTO_EDGES_ENABLED=true
HYBRIDMIND_AUTO_EDGE_COSINE_THRESHOLD=0.70
HYBRIDMIND_AUTO_EDGE_MAX_PER_NODE=10
HYBRIDMIND_IMAGE_EMBEDDING_URL=
HYBRIDMIND_IMAGE_RUNPOD_KEY=
"""
    if not env_file.exists():
        env_file.write_text(default_content)
        print_success(".env created with default settings.")
    else:
        print_success(".env already exists.")

# ─── Step 3: MCP Config Merging ───────────────────────────────────────────────

def merge_mcp_config(config_path: Path, server_name: str, server_def: dict):
    """Safely insert or update an MCP server definition in a JSON configuration file."""
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    data = {"mcpServers": {}}
    if config_path.exists():
        try:
            data = json.loads(config_path.read_text())
            if "mcpServers" not in data:
                data["mcpServers"] = {}
        except Exception:
            print_warning(f"Failed to parse existing config at {config_path}. Overwriting.")
            data = {"mcpServers": {}}
            
    data["mcpServers"][server_name] = server_def
    config_path.write_text(json.dumps(data, indent=2))
    print_success(f"Configured {server_name} in {config_path}")

def configure_mcp_clients():
    print_step("Configuring IDEs & AI Agents via MCP...")
    
    mcp_script_path = str(PROJECT_ROOT / "mcp_server" / "main.py")
    
    # Standard MCP server definition
    server_def = {
        "command": str(VENV_PYTHON),
        "args": [mcp_script_path],
        "env": {
            "HYBRIDMIND_API_URL": "http://localhost:8000"
        }
    }
    
    # 1. Claude Desktop
    if os.name == "nt":
        claude_desktop = Path(os.environ.get("APPDATA", "")) / "Claude" / "claude_desktop_config.json"
    else:
        claude_desktop = Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
        
    merge_mcp_config(claude_desktop, "hybridmind", server_def)
    
    # 2. Claude Code CLI
    claude_code = Path.home() / ".claude" / "mcp.json"
    merge_mcp_config(claude_code, "hybridmind", server_def)
    
    # 3. Windsurf IDE
    if os.name == "nt":
        windsurf = Path.home() / ".codeium" / "windsurf" / "mcp_config.json"
    else:
        windsurf = Path.home() / ".codeium" / "windsurf" / "mcp_config.json"
        
    merge_mcp_config(windsurf, "hybridmind", server_def)
    
    # 4. Cursor IDE
    # Provide the snippet to paste in Cursor settings UI
    print("\nTo configure Cursor IDE:")
    print("  Go to Settings -> Features -> MCP -> Add New MCP Server")
    print("  - Name: hybridmind")
    print("  - Type: command")
    print(f"  - Command: {VENV_PYTHON} {mcp_script_path}")

# ─── Step 4: ChatGPT OpenAPI Schema ──────────────────────────────────────────

def generate_openapi_schema():
    print_step("Generating ChatGPT Custom GPT OpenAPI Schema...")
    schema = {
        "openapi": "3.1.0",
        "info": {
            "title": "HybridMind Memory API",
            "version": "1.0.0",
            "description": "API for storing, recalling, and linking agent memories."
        },
        "servers": [
            {
                "url": "https://your-ngrok-or-tunnel-url.ngrok-free.app",
                "description": "Publicly exposed tunnel URL to your local HybridMind instance"
            }
        ],
        "paths": {
            "/nodes": {
                "post": {
                    "summary": "Remember a fact",
                    "operationId": "remember",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "text": {"type": "string", "description": "Fact to remember"},
                                        "metadata": {"type": "object", "description": "Optional attributes"}
                                    },
                                    "required": ["text"]
                                }
                            }
                        }
                    },
                    "responses": {
                        "201": {
                            "description": "Stored fact",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "id": {"type": "string"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/search/hybrid": {
                "post": {
                    "summary": "Recall relevant memories",
                    "operationId": "recall",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "query_text": {"type": "string", "description": "Search term"},
                                        "top_k": {"type": "integer", "default": 10}
                                    },
                                    "required": ["query_text"]
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Matching results",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "results": {
                                                "type": "array",
                                                "items": {
                                                    "type": "object",
                                                    "properties": {
                                                        "text": {"type": "string"},
                                                        "combined_score": {"type": "number"}
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    schema_path = PROJECT_ROOT / "docs" / "chatgpt_openapi_schema.json"
    schema_path.write_text(json.dumps(schema, indent=2))
    print_success(f"OpenAPI Schema generated at {schema_path}")

# ─── Main Execution ──────────────────────────────────────────────────────────

def main():
    print("==================================================")
    print("      HybridMind Native Installer & Configurator   ")
    print("==================================================")
    
    setup_venv()
    setup_env()
    configure_mcp_clients()
    generate_openapi_schema()
    
    print("\n==================================================")
    print("[\033[92m✓\033[0m] Installation & Configuration complete!")
    print("To start the local database server, run:")
    print("  python main.py")
    print("==================================================")

if __name__ == "__main__":
    main()
