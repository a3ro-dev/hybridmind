"""
HybridMind - Vector + Graph Native Database
Research-grade hybrid retrieval system with CRS algorithm
Includes comparison mode for HybridMind vs Neo4j vs ChromaDB
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import requests
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, Dict, List
from datetime import datetime

st.set_page_config(
    page_title="HybridMind",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Scientific dark theme CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Space+Grotesk:wght@400;500;600;700&display=swap');
    
    :root {
        --bg-primary: #0d1117;
        --bg-secondary: #161b22;
        --bg-tertiary: #21262d;
        --border: #30363d;
        --text-primary: #e6edf3;
        --text-secondary: #8b949e;
        --accent-blue: #58a6ff;
        --accent-green: #3fb950;
        --accent-purple: #a371f7;
        --accent-orange: #d29922;
        --accent-red: #f85149;
        --accent-cyan: #39c5cf;
    }
    
    html, body, [class*="css"] {
        font-family: 'Space Grotesk', -apple-system, sans-serif;
        color: var(--text-primary);
    }
    
    code, pre, .stCode, .mono {
        font-family: 'JetBrains Mono', monospace;
    }
    
    .main-header {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 20px 24px;
        margin-bottom: 20px;
    }
    
    .main-title {
        font-size: 1.75rem;
        font-weight: 700;
        color: var(--text-primary);
        margin: 0;
        letter-spacing: -0.02em;
    }
    
    .main-subtitle {
        font-size: 0.875rem;
        color: var(--text-secondary);
        margin-top: 4px;
        font-family: 'JetBrains Mono', monospace;
    }
    
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 500;
        font-family: 'JetBrains Mono', monospace;
    }
    
    .status-ok {
        background: rgba(63, 185, 80, 0.15);
        color: var(--accent-green);
        border: 1px solid rgba(63, 185, 80, 0.3);
    }
    
    .status-error {
        background: rgba(248, 81, 73, 0.15);
        color: var(--accent-red);
        border: 1px solid rgba(248, 81, 73, 0.3);
    }
    
    .status-warning {
        background: rgba(210, 153, 34, 0.15);
        color: var(--accent-orange);
        border: 1px solid rgba(210, 153, 34, 0.3);
    }
    
    .metric-card {
        background: var(--bg-secondary);
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 12px 16px;
    }
    
    .metric-label {
        font-size: 0.65rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-family: 'JetBrains Mono', monospace;
    }
    
    .metric-value {
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--text-primary);
        font-family: 'JetBrains Mono', monospace;
    }
    
    .formula-box {
        background: var(--bg-secondary);
        border: 1px solid var(--border);
        border-left: 3px solid var(--accent-purple);
        border-radius: 4px;
        padding: 12px 16px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 1rem;
        color: var(--accent-purple);
        text-align: center;
    }
    
    .data-table {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
    }
    
    .section-header {
        font-size: 0.7rem;
        font-weight: 600;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin-bottom: 8px;
        font-family: 'JetBrains Mono', monospace;
    }
    
    .result-card {
        background: var(--bg-secondary);
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 12px;
        margin-bottom: 8px;
    }
    
    .score-pill {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        font-weight: 500;
    }
    
    .score-vector { background: rgba(88, 166, 255, 0.2); color: var(--accent-blue); }
    .score-graph { background: rgba(63, 185, 80, 0.2); color: var(--accent-green); }
    .score-combined { background: rgba(163, 113, 247, 0.2); color: var(--accent-purple); }
    
    .system-hybridmind { border-left: 3px solid var(--accent-purple); }
    .system-neo4j { border-left: 3px solid var(--accent-green); }
    .system-chromadb { border-left: 3px solid var(--accent-blue); }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: var(--bg-secondary);
        border-radius: 6px 6px 0 0;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        font-weight: 500;
        font-size: 0.875rem;
    }
    
    .comparison-card {
        background: var(--bg-secondary);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 16px;
        height: 100%;
    }
    
    .winner-badge {
        background: linear-gradient(135deg, rgba(63, 185, 80, 0.2) 0%, rgba(63, 185, 80, 0.1) 100%);
        border: 1px solid rgba(63, 185, 80, 0.4);
        color: var(--accent-green);
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 0.75rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Configuration - use environment variable for Docker, fallback to localhost for local dev
DEFAULT_API_URL = os.environ.get("API_URL", "http://localhost:8000")
API_URL = st.sidebar.text_input("API", value=DEFAULT_API_URL, label_visibility="collapsed")


def api_call(endpoint: str, method: str = "GET", data: dict = None, timeout: int = 30):
    """Execute API request with timing."""
    try:
        url = f"{API_URL}{endpoint}"
        start = time.perf_counter()
        if method == "GET":
            response = requests.get(url, params=data, timeout=timeout)
        else:
            response = requests.post(url, json=data, timeout=timeout)
        elapsed = (time.perf_counter() - start) * 1000
        response.raise_for_status()
        result = response.json()
        # Only add latency to dict responses, not lists
        if isinstance(result, dict):
            result["_client_latency_ms"] = elapsed
        return result
    except requests.exceptions.ConnectionError:
        return None
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return None


def get_system_health() -> tuple[bool, dict]:
    """Get comprehensive system health."""
    health = api_call("/health")
    if health and health.get("status") == "healthy":
        return True, health
    return False, {}


def get_cache_stats() -> dict:
    """Get cache statistics."""
    return api_call("/cache/stats") or {}


# ============================================================================
# SIDEBAR
# ============================================================================

def render_sidebar() -> str:
    """Sidebar with system metrics."""
    st.sidebar.markdown("#### ◆ HybridMind")
    st.sidebar.caption("Vector + Graph Native DB")
    
    # Connection status
    is_ok, health = get_system_health()
    if is_ok:
        st.sidebar.markdown('<span class="status-badge status-ok">● ONLINE</span>', unsafe_allow_html=True)
    else:
        st.sidebar.markdown('<span class="status-badge status-error">● OFFLINE</span>', unsafe_allow_html=True)
        st.sidebar.warning("Start API: `uvicorn main:app`")
    
    st.sidebar.markdown("---")
    
    # Navigation
    page = st.sidebar.radio(
        "Module",
        ["Search", "System", "Explorer"],
        label_visibility="collapsed"
    )
    
    st.sidebar.markdown("---")
    
    # Quick stats
    st.sidebar.markdown('<p class="section-header">Database</p>', unsafe_allow_html=True)
    stats = api_call("/search/stats")
    if stats:
        c1, c2 = st.sidebar.columns(2)
        c1.metric("Nodes", f"{stats.get('total_nodes', 0):,}")
        c2.metric("Edges", f"{stats.get('total_edges', 0):,}")
    
    # Cache stats
    cache = get_cache_stats()
    if cache:
        st.sidebar.markdown('<p class="section-header">Cache</p>', unsafe_allow_html=True)
        hit_rate = cache.get('hit_rate', 0) * 100
        st.sidebar.progress(hit_rate / 100, text=f"Hit Rate: {hit_rate:.1f}%")
        st.sidebar.caption(f"Hits: {cache.get('hits', 0)} | Misses: {cache.get('misses', 0)}")
    
    # CRS Formula
    st.sidebar.markdown("---")
    st.sidebar.markdown('<p class="section-header">CRS Algorithm</p>', unsafe_allow_html=True)
    st.sidebar.markdown('<div class="formula-box">CRS = αV + βG</div>', unsafe_allow_html=True)
    st.sidebar.caption("V: vector similarity | G: graph proximity")
    
    return page


# ============================================================================
# SEARCH PAGE
# ============================================================================

def render_search_page():
    """Main search interface."""
    st.markdown("### Hybrid Search")
    st.caption("Contextual Relevance Scoring with vector-graph fusion")
    
    tabs = st.tabs(["Hybrid", "Vector", "Graph"])
    
    with tabs[0]:
        render_hybrid_search()
    with tabs[1]:
        render_vector_search()
    with tabs[2]:
        render_graph_search()


def render_hybrid_search():
    """CRS-based hybrid search."""
    query = st.text_input("Query", placeholder="semantic search query...", key="h_q")
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        k = st.number_input("k", 1, 100, 10, key="h_k")
    with c2:
        alpha = st.slider("α (vector)", 0.0, 1.0, 0.6, 0.05, key="h_a")
    with c3:
        beta = st.slider("β (graph)", 0.0, 1.0, 0.4, 0.05, key="h_b")
    with c4:
        anchor = st.text_input("anchor", placeholder="node_id", key="h_anc")
    
    if st.button("Execute", key="h_btn", type="primary"):
        if not query:
            st.warning("Query required")
            return
        
        payload = {
            "query_text": query,
            "top_k": k,
            "vector_weight": alpha,
            "graph_weight": beta
        }
        if anchor:
            payload["anchor_nodes"] = [anchor]
        
        with st.spinner("Processing..."):
            start = time.perf_counter()
            results = api_call("/search/hybrid", method="POST", data=payload)
            total_time = (time.perf_counter() - start) * 1000
        
        if results:
            display_search_results(results, total_time, show_breakdown=True)


def render_vector_search():
    """Pure vector similarity search."""
    query = st.text_input("Query", placeholder="semantic search...", key="v_q")
    
    c1, c2 = st.columns([1, 3])
    with c1:
        k = st.number_input("k", 1, 100, 10, key="v_k")
    with c2:
        min_score = st.slider("min_score", 0.0, 1.0, 0.0, 0.05, key="v_min")
    
    if st.button("Execute", key="v_btn", type="primary"):
        if not query:
            st.warning("Query required")
            return
        
        with st.spinner("Embedding & searching..."):
            start = time.perf_counter()
            results = api_call("/search/vector", method="POST", data={
                "query_text": query,
                "top_k": k,
                "min_score": min_score
            })
            total_time = (time.perf_counter() - start) * 1000
        
        if results:
            display_search_results(results, total_time, score_key="vector_score")


def render_graph_search():
    """Graph traversal."""
    start_id = st.text_input("Start Node", placeholder="node_id", key="g_start")
    
    c1, c2 = st.columns(2)
    with c1:
        depth = st.slider("max_depth", 1, 5, 2, key="g_d")
    with c2:
        direction = st.selectbox("direction", ["both", "outgoing", "incoming"], key="g_dir")
    
    if st.button("Traverse", key="g_btn", type="primary"):
        if not start_id:
            st.warning("Start node required")
            return
        
        with st.spinner("Traversing..."):
            start = time.perf_counter()
            results = api_call("/search/graph", data={
                "start_id": start_id,
                "depth": depth,
                "direction": direction
            })
            total_time = (time.perf_counter() - start) * 1000
        
        if results:
            display_search_results(results, total_time, score_key="graph_score", show_path=True)


def display_search_results(results: dict, client_time: float, score_key: str = "combined_score",
                           show_breakdown: bool = False, show_path: bool = False):
    """Render search results with metrics."""
    
    # Performance metrics
    server_time = results.get('query_time_ms', 0)
    candidates = results.get('total_candidates', 0)
    cache_hit = results.get('cache_hit', False)
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Server", f"{server_time:.1f}ms")
    c2.metric("Client", f"{client_time:.1f}ms")
    c3.metric("Candidates", f"{candidates:,}")
    c4.metric("Cache", "HIT" if cache_hit else "MISS")
    
    items = results.get("results", [])
    if not items:
        st.info("No results")
        return
    
    st.markdown(f"**{len(items)} results**")
    
    # Results table
    rows = []
    for i, r in enumerate(items, 1):
        row = {
            "#": i,
            "ID": r["node_id"][:20],
            "Text": r.get("text", "")[:60] + "...",
        }
        
        if show_breakdown:
            row["V"] = f"{r.get('vector_score', 0):.3f}"
            row["G"] = f"{r.get('graph_score', 0):.3f}"
            row["CRS"] = f"{r.get('combined_score', 0):.3f}"
        else:
            row["Score"] = f"{r.get(score_key, 0):.3f}"
        
        if show_path and r.get("depth") is not None:
            row["Depth"] = r.get("depth", 0)
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    st.dataframe(df, width="stretch", hide_index=True)
    
    # Score distribution
    if show_breakdown and len(items) > 1:
        st.markdown("**Score Distribution**")
        score_data = []
        for r in items:
            score_data.append({"Type": "Vector", "Score": r.get("vector_score", 0)})
            score_data.append({"Type": "Graph", "Score": r.get("graph_score", 0)})
            score_data.append({"Type": "CRS", "Score": r.get("combined_score", 0)})
        
        fig = px.box(pd.DataFrame(score_data), x="Type", y="Score", color="Type",
                     color_discrete_map={"Vector": "#58a6ff", "Graph": "#3fb950", "CRS": "#a371f7"})
        fig.update_layout(height=250, showlegend=False, margin=dict(t=10, b=10))
        st.plotly_chart(fig, width="stretch")


# ============================================================================
# COMPARISON PAGE
# ============================================================================

def render_comparison_page():
    """Cross-database comparison interface."""
    st.markdown("### Database Comparison")
    st.caption("HybridMind vs Neo4j vs ChromaDB — Side-by-side analysis")
    
    tabs = st.tabs(["Compare", "Benchmark", "System Status"])
    
    with tabs[0]:
        render_comparison_search()
    with tabs[1]:
        render_benchmark()
    with tabs[2]:
        render_comparison_status()


def render_comparison_status():
    """Show status of all database systems."""
    st.markdown("#### System Availability")
    
    status = api_call("/comparison/status")
    if not status:
        st.error("Could not fetch system status")
        return
    
    c1, c2, c3 = st.columns(3)
    
    # HybridMind
    with c1:
        hm = status.get("hybridmind", {})
        available = hm.get("available", False)
        st.markdown(f"""
        <div class="comparison-card system-hybridmind">
            <h4 style="margin:0; color: #a371f7;">◆ HybridMind</h4>
            <p style="color: #8b949e; font-size: 0.8rem; margin: 4px 0;">Hybrid (Vector + Graph)</p>
            <span class="status-badge {'status-ok' if available else 'status-error'}">
                {'● Online' if available else '● Offline'}
            </span>
        </div>
        """, unsafe_allow_html=True)
        if available:
            st.metric("Nodes", f"{hm.get('nodes', 0):,}")
            st.metric("Edges", f"{hm.get('edges', 0):,}")
            st.metric("Vectors", f"{hm.get('vector_index_size', 0):,}")
    
    # Neo4j
    with c2:
        neo = status.get("neo4j", {})
        available = neo.get("available", False)
        st.markdown(f"""
        <div class="comparison-card system-neo4j">
            <h4 style="margin:0; color: #3fb950;">⬡ Neo4j</h4>
            <p style="color: #8b949e; font-size: 0.8rem; margin: 4px 0;">Graph-Only</p>
            <span class="status-badge {'status-ok' if available else 'status-warning'}">
                {'● Online' if available else '○ Offline'}
            </span>
        </div>
        """, unsafe_allow_html=True)
        if available:
            st.metric("Nodes", f"{neo.get('nodes', 0):,}")
            st.metric("Edges", f"{neo.get('edges', 0):,}")
        else:
            st.caption("Start Neo4j to enable comparison")
            st.code("docker run -p 7687:7687 neo4j", language="bash")
    
    # ChromaDB
    with c3:
        chroma = status.get("chromadb", {})
        available = chroma.get("available", False)
        st.markdown(f"""
        <div class="comparison-card system-chromadb">
            <h4 style="margin:0; color: #58a6ff;">◈ ChromaDB</h4>
            <p style="color: #8b949e; font-size: 0.8rem; margin: 4px 0;">Vector-Only</p>
            <span class="status-badge {'status-ok' if available else 'status-warning'}">
                {'● Online' if available else '○ Offline'}
            </span>
        </div>
        """, unsafe_allow_html=True)
        if available:
            st.metric("Documents", f"{chroma.get('documents', 0):,}")
        else:
            st.caption("Load data to enable comparison")
            st.code("python data/load_all_databases.py", language="bash")


def render_comparison_search():
    """Side-by-side search comparison."""
    st.markdown("#### Search Comparison")
    
    # Initialize selected query state
    if "selected_sample_query" not in st.session_state:
        st.session_state["selected_sample_query"] = ""
    
    # Use selected sample query as default if set
    default_query = st.session_state.get("selected_sample_query", "")
    
    # Query input
    query = st.text_input("Query", value=default_query, placeholder="Enter search query...", key="cmp_q")
    
    # Clear the selected query after it's been used
    if st.session_state["selected_sample_query"]:
        st.session_state["selected_sample_query"] = ""
    
    c1, c2, c3 = st.columns(3)
    with c1:
        top_k = st.number_input("Results per system", 1, 20, 10, key="cmp_k")
    with c2:
        v_weight = st.slider("α (HybridMind vector)", 0.0, 1.0, 0.6, 0.05, key="cmp_v")
    with c3:
        g_weight = st.slider("β (HybridMind graph)", 0.0, 1.0, 0.4, 0.05, key="cmp_g")
    
    # Sample queries
    with st.expander("Sample Queries"):
        samples = api_call("/comparison/sample-queries")
        if samples:
            st.markdown("**Semantic:**")
            for q in samples.get("semantic_queries", [])[:3]:
                if st.button(q, key=f"sq_{q[:10]}"):
                    st.session_state["selected_sample_query"] = q
                    st.rerun()
            
            st.markdown("**Technical:**")
            for q in samples.get("specific_queries", [])[:3]:
                if st.button(q, key=f"tq_{q[:10]}"):
                    st.session_state["selected_sample_query"] = q
                    st.rerun()
    
    if st.button("Compare", key="cmp_btn", type="primary"):
        if not query:
            st.warning("Enter a query")
            return
        
        with st.spinner("Querying all systems..."):
            result = api_call("/comparison/search", method="POST", data={
                "query_text": query,
                "top_k": top_k,
                "vector_weight": v_weight,
                "graph_weight": g_weight
            })
        
        if result:
            display_comparison_results(result)


def display_comparison_results(result: dict):
    """Display comparison results with analysis."""
    
    # Overall metrics
    st.markdown("---")
    st.markdown("#### Results")
    
    analysis = result.get("analysis", {})
    latency = analysis.get("latency_comparison", {})
    
    # Latency comparison chart
    latency_data = pd.DataFrame([
        {"System": "HybridMind", "Latency (ms)": latency.get("hybridmind_ms", 0), "Type": "Hybrid"},
        {"System": "Neo4j", "Latency (ms)": latency.get("neo4j_ms", 0), "Type": "Graph"},
        {"System": "ChromaDB", "Latency (ms)": latency.get("chromadb_ms", 0), "Type": "Vector"},
    ])
    
    c1, c2 = st.columns([2, 1])
    
    with c1:
        fig = px.bar(
            latency_data,
            x="System",
            y="Latency (ms)",
            color="System",
            color_discrete_map={
                "HybridMind": "#a371f7",
                "Neo4j": "#3fb950",
                "ChromaDB": "#58a6ff"
            }
        )
        fig.update_layout(
            height=200,
            margin=dict(t=10, b=10, l=10, r=10),
            showlegend=False,
            xaxis_title="",
            yaxis_title="Latency (ms)"
        )
        st.plotly_chart(fig, width="stretch")
    
    with c2:
        st.metric("Fastest", latency.get("fastest", "-").upper())
        st.metric("Total Unique", analysis.get("total_unique_results", 0))
        st.metric("Common to All", analysis.get("common_to_all", 0))
    
    # Results columns
    st.markdown("#### Side-by-Side Results")
    
    results = result.get("results", {})
    c1, c2, c3 = st.columns(3)
    
    # HybridMind results
    with c1:
        hm = results.get("hybridmind", {})
        st.markdown(f"""
        <div style="border-left: 3px solid #a371f7; padding-left: 12px;">
            <h5 style="color: #a371f7; margin: 0;">◆ HybridMind</h5>
            <span style="color: #8b949e; font-size: 0.8rem;">{hm.get('latency_ms', 0):.1f}ms • {hm.get('count', 0)} results</span>
        </div>
        """, unsafe_allow_html=True)
        
        if hm.get("error"):
            st.error(hm["error"])
        else:
            for i, item in enumerate(hm.get("items", [])[:5], 1):
                with st.container():
                    st.markdown(f"**{i}.** `{item['node_id'][:15]}...`")
                    st.caption(f"Score: {item.get('score', 0):.3f} (V:{item.get('vector_score', 0):.2f} G:{item.get('graph_score', 0):.2f})")
                    st.text(item.get("text", "")[:100] + "...")
    
    # Neo4j results
    with c2:
        neo = results.get("neo4j", {})
        st.markdown(f"""
        <div style="border-left: 3px solid #3fb950; padding-left: 12px;">
            <h5 style="color: #3fb950; margin: 0;">⬡ Neo4j</h5>
            <span style="color: #8b949e; font-size: 0.8rem;">{neo.get('latency_ms', 0):.1f}ms • {neo.get('count', 0)} results</span>
        </div>
        """, unsafe_allow_html=True)
        
        if neo.get("error"):
            st.warning(neo["error"])
        else:
            for i, item in enumerate(neo.get("items", [])[:5], 1):
                with st.container():
                    st.markdown(f"**{i}.** `{item['node_id'][:15]}...`")
                    st.caption(f"Score: {item.get('score', 0):.3f}")
                    st.text(item.get("text", "")[:100] + "...")
    
    # ChromaDB results
    with c3:
        chroma = results.get("chromadb", {})
        st.markdown(f"""
        <div style="border-left: 3px solid #58a6ff; padding-left: 12px;">
            <h5 style="color: #58a6ff; margin: 0;">◈ ChromaDB</h5>
            <span style="color: #8b949e; font-size: 0.8rem;">{chroma.get('latency_ms', 0):.1f}ms • {chroma.get('count', 0)} results</span>
        </div>
        """, unsafe_allow_html=True)
        
        if chroma.get("error"):
            st.warning(chroma["error"])
        else:
            for i, item in enumerate(chroma.get("items", [])[:5], 1):
                with st.container():
                    st.markdown(f"**{i}.** `{item['node_id'][:15]}...`")
                    st.caption(f"Score: {item.get('score', 0):.3f}")
                    st.text(item.get("text", "")[:100] + "...")
    
    # Overlap analysis
    st.markdown("---")
    st.markdown("#### Overlap Analysis")
    
    overlaps = analysis.get("overlaps", {})
    unique = analysis.get("unique_per_system", {})
    
    # Venn-style metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("HM ∩ Neo4j", overlaps.get("hybridmind_neo4j", 0))
    c2.metric("HM ∩ Chroma", overlaps.get("hybridmind_chromadb", 0))
    c3.metric("Neo4j ∩ Chroma", overlaps.get("neo4j_chromadb", 0))
    c4.metric("All Three", analysis.get("common_to_all", 0))
    
    # Unique results
    st.markdown("**Unique Discoveries:**")
    c1, c2, c3 = st.columns(3)
    c1.metric("Only HybridMind", unique.get("hybridmind", 0), help="Results found only by HybridMind")
    c2.metric("Only Neo4j", unique.get("neo4j", 0), help="Results found only by Neo4j")
    c3.metric("Only ChromaDB", unique.get("chromadb", 0), help="Results found only by ChromaDB")


def render_benchmark():
    """Benchmark runner interface."""
    st.markdown("#### Performance Benchmark")
    st.caption("Run comprehensive latency and throughput tests across all systems")
    
    # Benchmark configuration
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("**Configuration**")
        iterations = st.slider("Iterations per query", 1, 10, 3, key="bench_iter")
        top_k = st.number_input("Results per query", 1, 50, 10, key="bench_k")
    
    with c2:
        st.markdown("**Test Queries**")
        query_option = st.radio(
            "Query source",
            ["Default suite", "Custom queries"],
            key="bench_src"
        )
    
    if query_option == "Custom queries":
        custom_queries = st.text_area(
            "Queries (one per line)",
            value="machine learning\ndeep learning neural networks\nnatural language processing",
            key="bench_custom"
        )
        queries = [q.strip() for q in custom_queries.split("\n") if q.strip()]
    else:
        samples = api_call("/comparison/sample-queries")
        queries = samples.get("benchmark_suite", [])[:10] if samples else [
            "machine learning",
            "deep learning",
            "natural language processing"
        ]
        st.info(f"Using {len(queries)} default test queries")
    
    if st.button("Run Benchmark", key="bench_run", type="primary"):
        if not queries:
            st.warning("No queries specified")
            return
        
        progress = st.progress(0, text="Initializing benchmark...")
        
        with st.spinner("Running benchmark... This may take a minute."):
            result = api_call("/comparison/benchmark", method="POST", data={
                "queries": queries,
                "top_k": top_k,
                "iterations": iterations
            }, timeout=120)
        
        progress.progress(100, text="Complete!")
        
        if result:
            display_benchmark_results(result)


def display_benchmark_results(result: dict):
    """Display benchmark results with visualizations."""
    
    st.markdown("---")
    st.markdown("#### Benchmark Results")
    
    config = result.get("benchmark_config", {})
    stats = result.get("statistics", {})
    winner = result.get("winner", {})
    
    # Config summary
    st.caption(f"Queries: {config.get('queries_count', 0)} • Iterations: {config.get('iterations_per_query', 0)} • Top-K: {config.get('top_k', 10)}")
    
    # Winner badges
    c1, c2 = st.columns(2)
    with c1:
        fastest = winner.get("lowest_latency", "-")
        st.markdown(f"""
        <div style="background: rgba(63, 185, 80, 0.1); border: 1px solid rgba(63, 185, 80, 0.3); 
                    border-radius: 8px; padding: 16px; text-align: center;">
            <span style="color: #8b949e; font-size: 0.75rem;">LOWEST LATENCY</span><br>
            <span style="color: #3fb950; font-size: 1.5rem; font-weight: 600;">{fastest.upper()}</span>
        </div>
        """, unsafe_allow_html=True)
    
    with c2:
        throughput = winner.get("highest_throughput", "-")
        st.markdown(f"""
        <div style="background: rgba(88, 166, 255, 0.1); border: 1px solid rgba(88, 166, 255, 0.3); 
                    border-radius: 8px; padding: 16px; text-align: center;">
            <span style="color: #8b949e; font-size: 0.75rem;">HIGHEST THROUGHPUT</span><br>
            <span style="color: #58a6ff; font-size: 1.5rem; font-weight: 600;">{throughput.upper()}</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Latency comparison table
    st.markdown("**Latency Statistics (ms)**")
    
    latency_rows = []
    for system, data in stats.items():
        if data.get("error"):
            latency_rows.append({
                "System": system.upper(),
                "Avg": "-",
                "P50": "-",
                "P95": "-",
                "P99": "-",
                "Min": "-",
                "Max": "-",
                "QPS": "-"
            })
        else:
            latency_rows.append({
                "System": system.upper(),
                "Avg": f"{data.get('avg_latency_ms', 0):.1f}",
                "P50": f"{data.get('p50_latency_ms', 0):.1f}",
                "P95": f"{data.get('p95_latency_ms', 0):.1f}",
                "P99": f"{data.get('p99_latency_ms', 0):.1f}",
                "Min": f"{data.get('min_latency_ms', 0):.1f}",
                "Max": f"{data.get('max_latency_ms', 0):.1f}",
                "QPS": f"{data.get('throughput_qps', 0):.1f}"
            })
    
    df = pd.DataFrame(latency_rows)
    st.dataframe(df, width="stretch", hide_index=True)
    
    # Latency bar chart
    st.markdown("**Average Latency Comparison**")
    
    chart_data = []
    colors = {"hybridmind": "#a371f7", "neo4j": "#3fb950", "chromadb": "#58a6ff"}
    
    for system, data in stats.items():
        if not data.get("error"):
            chart_data.append({
                "System": system.capitalize(),
                "Latency (ms)": data.get("avg_latency_ms", 0),
                "P95 (ms)": data.get("p95_latency_ms", 0)
            })
    
    if chart_data:
        df_chart = pd.DataFrame(chart_data)
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name="Average",
            x=df_chart["System"],
            y=df_chart["Latency (ms)"],
            marker_color=["#a371f7", "#3fb950", "#58a6ff"][:len(chart_data)]
        ))
        
        fig.add_trace(go.Bar(
            name="P95",
            x=df_chart["System"],
            y=df_chart["P95 (ms)"],
            marker_color=["rgba(163,113,247,0.5)", "rgba(63,185,80,0.5)", "rgba(88,166,255,0.5)"][:len(chart_data)]
        ))
        
        fig.update_layout(
            barmode="group",
            height=300,
            margin=dict(t=20, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            yaxis_title="Latency (ms)"
        )
        
        st.plotly_chart(fig, width="stretch")
    
    # Throughput chart
    st.markdown("**Throughput (Queries/Second)**")
    
    throughput_data = []
    for system, data in stats.items():
        if not data.get("error"):
            throughput_data.append({
                "System": system.capitalize(),
                "QPS": data.get("throughput_qps", 0)
            })
    
    if throughput_data:
        df_qps = pd.DataFrame(throughput_data)
        fig = px.bar(
            df_qps,
            x="System",
            y="QPS",
            color="System",
            color_discrete_map={
                "Hybridmind": "#a371f7",
                "Neo4j": "#3fb950",
                "Chromadb": "#58a6ff"
            }
        )
        fig.update_layout(
            height=250,
            margin=dict(t=10, b=10),
            showlegend=False,
            yaxis_title="Queries per Second"
        )
        st.plotly_chart(fig, width="stretch")


# ============================================================================
# SYSTEM PAGE
# ============================================================================

def render_system_page():
    """System monitoring and health."""
    st.markdown("### System Monitor")
    
    tabs = st.tabs(["Health", "Performance", "Components"])
    
    with tabs[0]:
        render_health_tab()
    with tabs[1]:
        render_performance_tab()
    with tabs[2]:
        render_components_tab()


def render_health_tab():
    """Health check details."""
    health = api_call("/health")
    ready = api_call("/ready")
    
    if not health:
        st.error("Cannot connect to API")
        return
    
    # Status overview
    c1, c2, c3 = st.columns(3)
    
    status = health.get("status", "unknown")
    c1.metric("Status", status.upper())
    c2.metric("Uptime", f"{health.get('uptime_seconds', 0):.0f}s")
    c3.metric("Timestamp", datetime.now().strftime("%H:%M:%S"))
    
    # Components
    st.markdown("**Components**")
    components = health.get("components", {})
    comp_data = []
    for name, info in components.items():
        comp_data.append({
            "Component": name,
            "Status": info.get("status", "unknown").upper(),
            "Details": str(info.get("details", "-"))[:50]
        })
    
    if comp_data:
        st.dataframe(pd.DataFrame(comp_data), width="stretch", hide_index=True)
    
    # Readiness
    if ready:
        st.markdown("**Readiness**")
        c1, c2, c3 = st.columns(3)
        c1.metric("Model", "✓" if ready.get("model_loaded") else "✗")
        c2.metric("Nodes", f"{ready.get('nodes_loaded', 0):,}")
        c3.metric("Edges", f"{ready.get('edges_loaded', 0):,}")


def render_performance_tab():
    """Performance metrics."""
    stats = api_call("/search/stats")
    cache = get_cache_stats()
    
    if not stats:
        st.warning("No stats available")
        return
    
    # Database metrics
    st.markdown("**Database**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Nodes", f"{stats.get('total_nodes', 0):,}")
    c2.metric("Edges", f"{stats.get('total_edges', 0):,}")
    c3.metric("Avg Degree", f"{stats.get('avg_edges_per_node', 0):.2f}")
    c4.metric("Vector Index", f"{stats.get('vector_index_size', 0):,}")
    
    # Cache metrics
    if cache:
        st.markdown("**Cache**")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Hits", f"{cache.get('hits', 0):,}")
        c2.metric("Misses", f"{cache.get('misses', 0):,}")
        c3.metric("Hit Rate", f"{cache.get('hit_rate', 0)*100:.1f}%")
        c4.metric("Size", f"{cache.get('size', 0)}/{cache.get('maxsize', 0)}")
        
        # Clear cache button
        if st.button("Clear Cache"):
            result = api_call("/cache/clear", method="POST")
            if result and result.get("status") == "success":
                st.success("Cache cleared")
                st.rerun()
    
    # Edge type distribution
    edge_types = stats.get("edge_types", {})
    if edge_types:
        st.markdown("**Edge Types**")
        fig = px.pie(
            pd.DataFrame([{"Type": k, "Count": v} for k, v in edge_types.items()]),
            values="Count", names="Type", hole=0.4,
            color_discrete_sequence=["#58a6ff", "#3fb950", "#a371f7", "#d29922"]
        )
        fig.update_layout(height=250, margin=dict(t=10, b=10))
        st.plotly_chart(fig, width="stretch")


def render_components_tab():
    """System components info."""
    st.markdown("**Architecture**")
    
    st.code("""
┌─────────────────────────────────────────────────────┐
│                    FastAPI + Uvicorn                │
│              (Rate Limiting, Caching)               │
├─────────────────────────────────────────────────────┤
│                   Hybrid Ranker                     │
│          CRS(q) = αV(q) + βG(q)                     │
├────────────────────┬────────────────────────────────┤
│   Vector Engine    │         Graph Engine           │
│   FAISS + L2       │      NetworkX + BFS            │
├────────────────────┴────────────────────────────────┤
│              Embedding Engine                       │
│         all-MiniLM-L6-v2 (384-dim)                  │
├─────────────────────────────────────────────────────┤
│                    SQLite (WAL)                     │
│              nodes, edges, metadata                 │
└─────────────────────────────────────────────────────┘
    """, language=None)
    
    # Version info
    st.markdown("**Stack**")
    versions = [
        {"Component": "FAISS", "Purpose": "Vector similarity search (L2/IP)"},
        {"Component": "NetworkX", "Purpose": "Graph operations & traversal"},
        {"Component": "SQLite", "Purpose": "Persistent storage (WAL mode)"},
        {"Component": "sentence-transformers", "Purpose": "Embedding generation"},
        {"Component": "FastAPI", "Purpose": "REST API framework"},
    ]
    st.dataframe(pd.DataFrame(versions), width="stretch", hide_index=True)


# ============================================================================
# EXPLORER PAGE
# ============================================================================

def render_explorer_page():
    """Data exploration."""
    st.markdown("### Data Explorer")
    
    tabs = st.tabs(["Nodes", "Edges", "Bulk Import"])
    
    with tabs[0]:
        render_nodes_explorer()
    with tabs[1]:
        render_edges_explorer()
    with tabs[2]:
        render_bulk_import()


def render_nodes_explorer():
    """Node browser."""
    c1, c2 = st.columns([3, 1])
    with c1:
        node_id = st.text_input("Node ID", placeholder="Lookup specific node...")
    with c2:
        limit = st.number_input("Limit", 10, 100, 25)
    
    if node_id:
        node = api_call(f"/nodes/{node_id}")
        if node:
            st.json(node)
        else:
            st.error("Not found")
    else:
        nodes = api_call("/nodes", data={"limit": limit})
        if nodes:
            df = pd.DataFrame([
                {
                    "ID": n["id"][:24],
                    "Title": n.get("metadata", {}).get("title", "-")[:40],
                    "Category": n.get("metadata", {}).get("category", "-"),
                }
                for n in nodes
            ])
            st.dataframe(df, width="stretch", hide_index=True)
        else:
            st.info("No nodes")


def render_edges_explorer():
    """Edge browser."""
    node_id = st.text_input("Node ID for edges", placeholder="Enter node ID...")
    
    if node_id:
        edges = api_call(f"/edges/node/{node_id}")
        if edges:
            df = pd.DataFrame([
                {
                    "Type": e.get("type", "-"),
                    "Source": e.get("source_id", "-")[:20],
                    "Target": e.get("target_id", "-")[:20],
                    "Weight": f"{e.get('weight', 1.0):.2f}",
                }
                for e in edges
            ])
            st.dataframe(df, width="stretch", hide_index=True)
        else:
            st.info("No edges for this node")


def render_bulk_import():
    """Bulk data import including LLM-powered unstructured data processing."""
    
    import_tabs = st.tabs(["📝 Unstructured Data", "📦 JSON Import"])
    
    with import_tabs[0]:
        render_unstructured_import()
    
    with import_tabs[1]:
        render_json_import()


def render_unstructured_import():
    """LLM-powered unstructured data import."""
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(163, 113, 247, 0.1) 0%, rgba(88, 166, 255, 0.1) 100%); 
                border: 1px solid rgba(163, 113, 247, 0.3); border-radius: 8px; padding: 16px; margin-bottom: 16px;">
        <h4 style="margin: 0; color: #a371f7;">Unstructured Data</h4>
    </div>
    """, unsafe_allow_html=True)
    
    # Large text area for unstructured data
    unstructured_text = st.text_area(
        "Paste your text here",
        height=300,
        placeholder="Paste any unstructured text here...",
        key="unstructured_input"
    )
    
    # Advanced options in expander
    with st.expander("Advanced Options"):
        c1, c2 = st.columns(2)
        with c1:
            model = st.selectbox(
                "Model",
                [
                    "qwen3.5-397b-a17b"
                ],
                index=0,
                key="llm_model"
            )
        with c2:
            custom_api_key = st.text_input(
                "Custom API Key (optional)",
                type="password",
                placeholder="Leave empty to use default",
                key="custom_api_key"
            )
    
    # Character count
    char_count = len(unstructured_text) if unstructured_text else 0
    st.caption(f"📊 {char_count:,} characters | Max: 100,000")
    
    # Process button
    col1, col2 = st.columns([1, 3])
    with col1:
        process_btn = st.button(
            "🚀 Extract Knowledge",
            type="primary",
            key="process_unstructured",
            disabled=char_count < 50
        )
    
    if process_btn:
        if char_count < 50:
            st.warning("Please enter at least 50 characters of text")
            return
        
        with st.spinner("🧠 AI is analyzing your text and extracting knowledge..."):
            payload = {
                "text": unstructured_text,
                "model": model
            }
            if custom_api_key:
                payload["api_key"] = custom_api_key
            
            result = api_call("/bulk/unstructured", method="POST", data=payload, timeout=120)
        
        if result:
            # Success/failure status
            if result.get("success"):
                st.success("✅ Knowledge extraction complete!")
            else:
                st.warning("⚠️ Completed with some issues")
            
            # Summary
            if result.get("summary"):
                st.markdown(f"""
                <div style="background: var(--bg-secondary); border-left: 3px solid #58a6ff; 
                            padding: 12px 16px; border-radius: 4px; margin: 12px 0;">
                    <p style="color: #8b949e; font-size: 0.75rem; margin: 0 0 4px 0;">SUMMARY</p>
                    <p style="color: #e6edf3; margin: 0;">{result.get('summary')}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Metrics
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Nodes Created", result.get("nodes_created", 0))
            c2.metric("Edges Created", result.get("edges_created", 0))
            c3.metric("Processing Time", f"{result.get('elapsed_ms', 0):.0f}ms")
            c4.metric("Status", "✓" if result.get("success") else "⚠")
            
            # Extracted entities preview
            entities = result.get("extracted_entities", [])
            if entities:
                st.markdown("**📋 Extracted Knowledge Nodes**")
                for i, entity in enumerate(entities[:10], 1):
                    metadata = entity.get("metadata", {})
                    node_type = metadata.get("type", "unknown")
                    importance = metadata.get("importance", "medium")
                    
                    type_colors = {
                        "fact": "#3fb950",
                        "concept": "#a371f7", 
                        "entity": "#58a6ff",
                        "event": "#d29922",
                        "definition": "#39c5cf"
                    }
                    color = type_colors.get(node_type, "#8b949e")
                    
                    st.markdown(f"""
                    <div style="background: var(--bg-secondary); border: 1px solid var(--border); 
                                border-radius: 6px; padding: 10px 14px; margin-bottom: 8px;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span style="color: #e6edf3; font-weight: 500;">{i}. {entity.get('text_preview', '')[:80]}...</span>
                            <span style="background: {color}22; color: {color}; padding: 2px 8px; 
                                         border-radius: 4px; font-size: 0.7rem; font-weight: 500;">
                                {node_type.upper()}
                            </span>
                        </div>
                        <code style="color: #8b949e; font-size: 0.75rem;">{entity.get('node_id', '')}</code>
                    </div>
                    """, unsafe_allow_html=True)
                
                if len(entities) > 10:
                    st.caption(f"... and {len(entities) - 10} more nodes")
            
            # Errors
            errors = result.get("errors", [])
            if errors:
                with st.expander(f"⚠️ {len(errors)} Warning(s)"):
                    for error in errors:
                        st.warning(error)


def render_json_import():
    """Traditional JSON bulk import."""
    st.caption("Import structured JSON data for nodes and edges")
    
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("**Import Nodes**")
        nodes_json = st.text_area("Nodes JSON", height=150, placeholder='[{"text": "...", "metadata": {}}]')
        
        if st.button("Import Nodes", key="bulk_n"):
            try:
                import json
                nodes = json.loads(nodes_json)
                result = api_call("/bulk/nodes", method="POST", data={
                    "nodes": nodes,
                    "generate_embeddings": True
                })
                if result:
                    st.success(f"Created: {result.get('created', 0)}, Failed: {result.get('failed', 0)}")
            except Exception as e:
                st.error(f"Error: {e}")
    
    with c2:
        st.markdown("**Import Edges**")
        edges_json = st.text_area("Edges JSON", height=150, placeholder='[{"source_id": "...", "target_id": "...", "type": "..."}]')
        
        if st.button("Import Edges", key="bulk_e"):
            try:
                import json
                edges = json.loads(edges_json)
                result = api_call("/bulk/edges", method="POST", data={
                    "edges": edges,
                    "skip_validation": False
                })
                if result:
                    st.success(f"Created: {result.get('created', 0)}, Failed: {result.get('failed', 0)}")
            except Exception as e:
                st.error(f"Error: {e}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    page = render_sidebar()
    
    if page == "Search":
        render_search_page()
    # Comparison disabled
    elif page == "System":
        render_system_page()
    elif page == "Explorer":
        render_explorer_page()


if __name__ == "__main__":
    main()
