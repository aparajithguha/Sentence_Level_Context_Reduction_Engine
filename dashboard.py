import streamlit as st
import os
import json
import re
import pandas as pd
import plotly.graph_objects as go
import networkx as nx
from pathlib import Path

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

try:
    from sentence_transformers import util
except ImportError:
    util = None

# Set page configuration with a premium dark theme layout
st.set_page_config(
    page_title="SCRE V2 Evaluation Suite & Showcase",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium dark-mode styling with Outfit typography and glassmorphism
st.markdown("""
<style>
    /* Main Background & Text Color */
    .stApp {
        background-color: #0B0B0E;
        color: #E2E2E9;
    }
    
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Sleek container panels */
    .panel-card {
        background: rgba(20, 20, 28, 0.7);
        border: 1px solid rgba(44, 44, 62, 0.6);
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.45);
        backdrop-filter: blur(8px);
    }
    
    /* Title Gradient styling */
    .title-gradient {
        background: linear-gradient(135deg, #A855F7 0%, #3B82F6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 2.8rem;
        margin-bottom: 8px;
    }
    
    /* Subtitle styling */
    .subtitle-text {
        font-size: 1.15rem;
        color: #9CA3AF;
        margin-bottom: 24px;
    }
    
    /* Tags inside playground */
    .tag-dec { background-color: rgba(168, 85, 247, 0.15); color: #C084FC; border: 1px solid rgba(168, 85, 247, 0.4); padding: 2px 6px; border-radius: 4px; font-weight: bold; }
    .tag-con { background-color: rgba(249, 115, 22, 0.15); color: #FDBA74; border: 1px solid rgba(249, 115, 22, 0.4); padding: 2px 6px; border-radius: 4px; font-weight: bold; }
    .tag-rea { background-color: rgba(34, 197, 94, 0.15); color: #86EFAC; border: 1px solid rgba(34, 197, 94, 0.4); padding: 2px 6px; border-radius: 4px; font-weight: bold; }
    .tag-fact { background-color: rgba(59, 130, 246, 0.15); color: #93C5FD; border: 1px solid rgba(59, 130, 246, 0.4); padding: 2px 6px; border-radius: 4px; font-weight: bold; }
    .tag-wrk { background-color: rgba(234, 179, 8, 0.15); color: #FDE047; border: 1px solid rgba(234, 179, 8, 0.4); padding: 2px 6px; border-radius: 4px; font-weight: bold; }
    .tag-generic { background-color: rgba(107, 114, 128, 0.15); color: #D1D5DB; border: 1px solid rgba(107, 114, 128, 0.4); padding: 2px 6px; border-radius: 4px; font-weight: bold; }

    /* Custom sidebar header */
    .sidebar-header {
        font-weight: 600;
        font-size: 1.25rem;
        color: #E2E2E9;
        margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)

# Cache model and dataset loading to optimize initialization speed
@st.cache_resource
def load_evaluator():
    from tests.unified_benchmark import UnifiedBenchmark
    return UnifiedBenchmark()

benchmark_engine = load_evaluator()
dataset = benchmark_engine.dataset
scre_instance = benchmark_engine.scre

# Helper to find document path
doc_map = {d["doc_name"]: d for d in dataset}
unique_doc_names = sorted(list(doc_map.keys()))

# Parse and highlight tags in text
def highlight_tags(text: str) -> str:
    """Wrap SCRE output in a scrollable div with controlled font size.
    Escapes raw markdown headings so they don't render as giant H1/H2 in st.markdown."""
    import html as html_lib

    # Highlight tag patterns e.g. [DEC-01], [REA-02] BEFORE escaping
    def repl(match):
        tag = match.group(1)
        tag_type = tag.split('-')[0].upper()
        if tag_type == "DEC":
            return f'##DEC##{tag}##/DEC##'
        elif tag_type == "CON":
            return f'##CON##{tag}##/CON##'
        elif tag_type in ["REA", "REASON"]:
            return f'##REA##{tag}##/REA##'
        elif tag_type in ["FACT", "FCT"]:
            return f'##FACT##{tag}##/FACT##'
        elif tag_type in ["WRK", "WRKFLOW"]:
            return f'##WRK##{tag}##/WRK##'
        else:
            return f'##GEN##{tag}##/GEN##'

    # Substitute tags with placeholders
    text = re.sub(r'\[([A-Z]+-\d+)\]', repl, text)

    # HTML-escape the full text so markdown headers (#, ##) don't blow up
    escaped = html_lib.escape(text)

    # Restore tag spans after escaping
    escaped = re.sub(r'##DEC##(.+?)##/DEC##',  r'<span class="tag-dec">  [\1]</span>', escaped)
    escaped = re.sub(r'##CON##(.+?)##/CON##',  r'<span class="tag-con">  [\1]</span>', escaped)
    escaped = re.sub(r'##REA##(.+?)##/REA##',  r'<span class="tag-rea">  [\1]</span>', escaped)
    escaped = re.sub(r'##FACT##(.+?)##/FACT##', r'<span class="tag-fact">[\1]</span>', escaped)
    escaped = re.sub(r'##WRK##(.+?)##/WRK##',  r'<span class="tag-wrk">  [\1]</span>', escaped)
    escaped = re.sub(r'##GEN##(.+?)##/GEN##',  r'<span class="tag-generic">[\1]</span>', escaped)

    # Wrap in a scrollable, size-controlled container
    lines_html = escaped.replace('\n', '<br>')
    return f"""
    <div style="
        font-family: 'JetBrains Mono', 'Fira Code', 'Courier New', monospace;
        font-size: 0.78rem;
        line-height: 1.6;
        color: #D1D5DB;
        background: rgba(10,10,16,0.6);
        border: 1px solid rgba(168,85,247,0.25);
        border-radius: 8px;
        padding: 12px 14px;
        height: 350px;
        overflow-y: auto;
        white-space: pre-wrap;
        word-break: break-word;
    ">{lines_html}</div>
    """

# Sidebar Tuning parameters
st.sidebar.markdown('<div class="sidebar-header">SCRE Parameters</div>', unsafe_allow_html=True)
max_sentences = st.sidebar.slider("Max Sentences", 3, 12, 6)
context_window = st.sidebar.slider("Context Window", 0, 3, 1)
min_tokens = st.sidebar.number_input("Min Tokens", min_value=100, max_value=1000, value=250, step=50)

st.sidebar.markdown('---')
st.sidebar.markdown('**SCRE Architecture Engine**')
st.sidebar.markdown('A specialized, structure-preserving context reduction engine for technical proposals (KEPs, RFCs).')

# Top header
st.markdown('<div class="title-gradient">SCRE Context Reduction Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-text">Interactive Showcase & Research-Grade Evaluation Dashboard</div>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Performance Dashboard", "🔌 Graph Explorer", " Playground & Comparison"])

# --- TAB 1: PERFORMANCE DASHBOARD ---
with tab1:
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.subheader("Publishable Benchmark Results (75 Docs, 100 Q&As)")
    
    # Load JSON results
    json_path = Path("tests/benchmark_results.json")
    if json_path.exists():
        with open(json_path, "r") as f:
            metrics_data = json.load(f)
            
        # Reformat into DataFrame
        df_rows = []
        for strat, val in metrics_data.items():
            name = strat.split('_', 1)[-1].replace('_', ' ') if '_' in strat else strat
            df_rows.append({
                "Strategy": name,
                "SPS Score (0-100)": val["sps"],
                "Constraint Recall": f"{val['constraint_recall']:.2%}",
                "Decision Traceability": f"{val['decision_traceability']:.2%}",
                "Workflow Integrity": f"{val['workflow_integrity']:.2%}",
                "Reasoning Recall": f"{val['reasoning_recall']:.2%}",
                "Reasoning Graph Recall": f"{val['reasoning_graph_recall']:.2%}",
                "Dependency Recall": f"{val['dependency_recall']:.2%}",
                "SER (Efficiency)": val["ser"],
                "Compression Ratio": f"{val['compression_ratio']:.2%}",
                "Latency (ms)": f"{val['latency_ms']:.1f}ms"
            })
        st.dataframe(pd.DataFrame(df_rows), use_container_width=True)
    else:
        st.warning("No benchmark results found. Run tests/unified_benchmark.py first.")
    st.markdown('</div>', unsafe_allow_html=True)

    # Radar chart comparing strategies
    if json_path.exists():
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown('<div class="panel-card">', unsafe_allow_html=True)
            st.subheader("Evaluation Radar Plot")
            
            categories = ['SPS Score', 'Constraint Recall', 'Decision Traceability', 
                          'Workflow Integrity', 'Reasoning Recall', 'Reasoning Graph Recall', 'Dependency Recall']
            
            fig = go.Figure()
            for strategy in ['B_BM25', 'C_Vector_Search', 'D_SCRE']:
                name = strategy.split('_', 1)[-1].replace('_', ' ')
                metrics = metrics_data[strategy]
                
                values = [
                    metrics['sps'],
                    metrics['constraint_recall'] * 100,
                    metrics['decision_traceability'] * 100,
                    metrics['workflow_integrity'] * 100,
                    metrics['reasoning_recall'] * 100,
                    metrics['reasoning_graph_recall'] * 100,
                    metrics['dependency_recall'] * 100
                ]
                # Wrap polar coordinates
                values.append(values[0])
                
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories + [categories[0]],
                    fill='toself',
                    name=name
                ))
                
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100], gridcolor="#2C2C3E", linecolor="#2C2C3E"),
                    angularaxis=dict(gridcolor="#2C2C3E", linecolor="#2C2C3E", tickfont=dict(color="#A0A0B0")),
                    bgcolor="rgba(0,0,0,0)"
                ),
                showlegend=True,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#E2E2E9")
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col2:
            st.markdown('<div class="panel-card" style="height: 100%;">', unsafe_allow_html=True)
            st.subheader("SPS Ranking Details")
            st.markdown("""
            * **Raw Context (Baseline):** Acts as the absolute semantic baseline. While preserving all structure, it suffers from a size overhead of thousands of tokens, lowering LLM performance and raising operational costs.
            * **BM25:** Strong on keyword retention but lacks semantic sensitivity, resulting in poor Graph Paths and zero Dependency preservation.
            * **Vector Search:** Good at retrieving matching paragraphs but lacks structural and logical coherence, causing reasoning chains to break.
            * **SCRE:** Preserves **84.36% of the semantic properties** of original documents, including workflows and logical reasoning edges, while safely compressing **80.5% of the token size**.
            """)
            st.markdown('</div>', unsafe_allow_html=True)

# --- TAB 2: GRAPH EXPLORER ---
with tab2:
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.subheader("SCRE Reasoning & Dependency Graph Visualizer")
    selected_graph_doc = st.selectbox("Select document to load reasoning graph", unique_doc_names, key="graph_doc")
    
    doc_info = doc_map[selected_graph_doc]
    doc_path = Path(doc_info["path"])
    doc_text = doc_path.read_text(encoding="utf-8")
    
    # Ingest document to ensure graph is in database
    scre_instance.ingest(doc_text, selected_graph_doc)
    
    # Generate the network graph
    c = scre_instance.conn.cursor()
    c.execute("SELECT sentence_index, text, unit_type FROM memory_units WHERE document_id = ?", (selected_graph_doc,))
    nodes_data = c.fetchall()
    
    c.execute("SELECT source_idx, target_idx FROM reasoning_edges WHERE document_id = ?", (selected_graph_doc,))
    edges_data = c.fetchall()
    
    if not edges_data:
        st.info("No reasoning connections found in this document. Select another KEP or RFC to explore reasoning paths.")
    else:
        # Build Networkx Graph
        G = nx.DiGraph()
        active_nodes = set()
        for src, tgt in edges_data:
            G.add_edge(src, tgt)
            active_nodes.add(src)
            active_nodes.add(tgt)
            
        for idx, text, unit_type in nodes_data:
            if idx in active_nodes:
                clean_text = text[:120] + "..." if len(text) > 120 else text
                G.add_node(idx, text=clean_text, unit_type=unit_type)
                
        pos = nx.spring_layout(G, k=0.5, seed=42)
        
        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1.5, color='#4B5563'),
            hoverinfo='none',
            mode='lines'
        )
        
        node_x, node_y, node_hover, node_color, node_text = [], [], [], [], []
        color_map = {
            "decision": "#A855F7",
            "constraint": "#F97316",
            "reason": "#22C55E",
            "fact": "#3B82F6",
            "workflow": "#EAB308",
            "task": "#06B6D4"
        }
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            n_data = G.nodes[node]
            node_hover.append(f"<b>Sentence {node} ({n_data.get('unit_type', 'unknown')})</b><br>{n_data.get('text', '')}")
            node_color.append(color_map.get(n_data.get("unit_type", ""), "#6B7280"))
            node_text.append(str(node))
            
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            textposition="top center",
            hovertext=node_hover,
            marker=dict(
                showscale=False,
                color=node_color,
                size=18,
                line=dict(width=2, color='#1F2937')
            )
        )
        
        fig = go.Figure(data=[edge_trace, node_trace],
                     layout=go.Layout(
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=0, l=0, r=0, t=0),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)"
                     ))
        st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# --- TAB 3: PLAYGROUND & COMPARISON ---
with tab3:
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.subheader("Retrieval Playground")
    
    col_d, col_q = st.columns([1, 2])
    with col_d:
        selected_doc = st.selectbox("Select target document", unique_doc_names, key="play_doc")
    
    # Filter questions for this document
    doc_info = doc_map[selected_doc]
    doc_path = Path(doc_info["path"])
    doc_text = doc_path.read_text(encoding="utf-8")
    
    matching_qs = [q for q in dataset if q["doc_name"] == selected_doc]
    questions_list = [q["question"] for q in matching_qs]
    
    with col_q:
        selected_query = st.selectbox("Select question to test", questions_list)
        
    if st.button("▶ Run Context Reduction", type="primary"):
        with st.spinner("Ingesting document & running retrieval — this may take a few seconds..."):
            # Make sure document is ingested in SCRE database
            scre_instance.ingest(doc_text, selected_doc)

            # 1. Fetch BM25
            sentences = [s.strip() for s in doc_text.split('\n') if s.strip()]
            tokenized_corpus = [s.split() for s in sentences]
            bm25 = BM25Okapi(tokenized_corpus) if BM25Okapi else None
            bm25_context = ""
            if bm25:
                top_bm25 = bm25.get_top_n(selected_query.split(), sentences, n=max_sentences)
                bm25_context = "\n".join(top_bm25)
                
            # 2. Fetch Vector Search
            vector_context = ""
            if benchmark_engine.embedder:
                corpus_embs = benchmark_engine.embedder.encode(sentences, convert_to_tensor=True)
                q_emb = benchmark_engine.embedder.encode(selected_query, convert_to_tensor=True)
                hits = util.semantic_search(q_emb, corpus_embs)[0]
                top_hits = sorted(hits[:max_sentences], key=lambda x: x['corpus_id'])
                vector_context = "\n".join([sentences[h['corpus_id']] for h in top_hits])
                
            # 3. Fetch SCRE
            scre_res = scre_instance.retrieve(
                query=selected_query,
                document_id=selected_doc,
                max_sentences=max_sentences,
                context_window=context_window,
                min_tokens=min_tokens
            )
            scre_context = scre_res["context"]
            scre_meta = scre_res.get("metadata", {})
            
            # Save results in session state for persistence
            st.session_state["play_results"] = {
                "query": selected_query,
                "doc": selected_doc,
                "bm25": bm25_context,
                "vector": vector_context,
                "scre": scre_context,
                "scre_meta": scre_meta
            }
        
    # Render results from session state — show whenever results exist for the current document
    if "play_results" in st.session_state and st.session_state["play_results"].get("doc") == selected_doc:
        res = st.session_state["play_results"]
        bm25_context = res["bm25"]
        vector_context = res["vector"]
        scre_context = res["scre"]
        scre_meta = res.get("scre_meta", {})
        stored_query = res.get("query", "")
        
        # Show which query the results are for
        if stored_query != selected_query:
            st.info(f"Showing results for: *{stored_query}* — click ▶ Run to update for the new query.")
        
        # SCRE compression stats banner
        orig_tok = scre_meta.get('original_estimated_tokens', 0)
        red_tok = scre_meta.get('reduced_estimated_tokens', 0)
        ratio = scre_meta.get('reduction_ratio', 0)
        orig_sents = scre_meta.get('original_sentences', 0)
        red_sents = scre_meta.get('reduced_sentences', 0)
        if orig_tok > 0:
            st.markdown(
                f"""
                <div style="background:rgba(34,197,94,0.1);border:1px solid rgba(34,197,94,0.35);border-radius:10px;padding:12px 18px;margin-bottom:12px;font-size:0.9rem;">
                ✅ <b>SCRE Compression</b> &nbsp;|&nbsp;
                Tokens: <b style="color:#86EFAC">{orig_tok:,} → {red_tok:,}</b> &nbsp;|&nbsp;
                Sentences: <b style="color:#86EFAC">{orig_sents} → {red_sents}</b> &nbsp;|&nbsp;
                Reduction: <b style="color:#4ADE80">{ratio:.1%}</b>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div style="background:rgba(59,130,246,0.1);border:1px solid rgba(59,130,246,0.3);border-radius:10px;padding:10px 16px;margin-bottom:12px;font-size:0.85rem;">ℹ️ Run a query above to see compression statistics.</div>',
                unsafe_allow_html=True
            )
        
        # Display Comparisons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="panel-card" style="min-height: 400px;">', unsafe_allow_html=True)
            st.markdown("### BM25 Keywords (Baseline)")
            token_count = len(bm25_context.split()) if bm25_context else 0
            st.markdown(f"*Tokens: ~{token_count * 1.3:.0f}*")
            if bm25_context:
                st.text_area("BM25 Output", value=bm25_context, height=350, disabled=True, label_visibility="collapsed")
            else:
                st.warning("BM25 module not available.")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col2:
            st.markdown('<div class="panel-card" style="min-height: 400px;">', unsafe_allow_html=True)
            st.markdown("### Vector Similarity (Baseline)")
            token_count = len(vector_context.split()) if vector_context else 0
            st.markdown(f"*Tokens: ~{token_count * 1.3:.0f}*")
            if vector_context:
                st.text_area("Vector Output", value=vector_context, height=350, disabled=True, label_visibility="collapsed")
            else:
                st.warning("Embedding model not available.")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col3:
            st.markdown('<div class="panel-card" style="min-height: 400px; border-color: rgba(168, 85, 247, 0.6);">', unsafe_allow_html=True)
            st.markdown("### ✨ SCRE Output (Structure-Preserved)")
            token_count = len(scre_context.split()) if scre_context else 0
            st.markdown(f"*Tokens: ~{token_count * 1.3:.0f}*")
            if scre_context:
                st.markdown(highlight_tags(scre_context), unsafe_allow_html=True)
            else:
                st.warning("SCRE returned no context. Try selecting a different document or query.")
            st.markdown('</div>', unsafe_allow_html=True)
            
    st.markdown('</div>', unsafe_allow_html=True)
