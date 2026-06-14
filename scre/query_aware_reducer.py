from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import numpy as np
import hashlib
from collections import Counter
from typing import Any

try:
    import spacy
except ImportError:
    spacy = None

try:
    from sentence_transformers import SentenceTransformer, util
except ImportError:
    SentenceTransformer = None
    util = None

try:
    import tiktoken
except ImportError:
    tiktoken = None

# Enforce strict offline mode now that the model is cached.
# This prevents HF Hub update checks and silences token warnings.
os.environ["HF_HUB_OFFLINE"] = "1"

def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def compute_idf(sentences: list) -> dict[str, float]:
    """Compute inverse document frequency for terms across sentences."""
    from collections import Counter
    
    doc_freq = Counter()
    total_docs = len(sentences)
    
    if total_docs == 0:
        return {}
    
    for sent in sentences:
        # Get unique terms in this sentence
        words = set(re.sub(r"\W+", " ", sent.text.lower()).split())
        doc_freq.update(words)
    
    # Calculate IDF: log(total_docs / doc_freq)
    idf = {}
    for term, freq in doc_freq.items():
        if term and len(term) >= 2:  # Skip single chars and empty
            idf[term] = math.log(total_docs / (1 + freq)) + 1
    
    return idf


def jaccard_similarity(a: str, b: str) -> float:
    """Calculate Jaccard similarity between two strings."""
    a_terms = set(re.findall(r"\w+", a.lower()))
    b_terms = set(re.findall(r"\w+", b.lower()))

    if not a_terms or not b_terms:
        return 0.0

    return len(a_terms & b_terms) / len(a_terms | b_terms)


def estimate_tokens(text: str) -> int:
    """Accurate token estimation using tiktoken, fallback to char heuristic."""
    if not text.strip():
        return 0
    if tiktoken is not None:
        # Use cl100k_base which is standard for GPT-4/latest LLMs
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text, disallowed_special=()))
    return max(1, math.ceil(len(text) / 4))


class SemanticUnit:
    def __init__(self, original_sentence_index: int, original_sentence_text: str, unit_type: str,
                 subject: str = "", relation: str = "", obj: str = ""):
        self.original_sentence_index = original_sentence_index
        self.original_sentence_text = original_sentence_text
        self.render_text = original_sentence_text
        self.type = unit_type.lower()
        self.subject = subject
        self.relation = relation
        self.object = obj
        self.score = 0.0
        self.connectivity = 0
        self.context_header = ""

    def __hash__(self):
        return hash((self.type, self.subject.lower(), self.relation.lower(), self.object.lower()))

    def __eq__(self, other):
        if not isinstance(other, SemanticUnit):
            return NotImplemented
        return (self.type == other.type and
                self.subject.lower() == other.subject.lower() and
                self.relation.lower() == other.relation.lower() and
                self.object.lower() == other.object.lower())


class WorkflowUnit(SemanticUnit):
    def __init__(self, original_sentence_index: int, name: str, steps: list[str], original_text: str):
        super().__init__(original_sentence_index, original_text, "workflow")
        self.name = name
        self.steps = steps
        self.render_text = json.dumps({
            "type": "workflow",
            "name": self.name,
            "steps": self.steps
        }, indent=2)
        
    def __hash__(self):
        return hash((super().__hash__(), tuple(self.steps)))

    def __eq__(self, other):
        if not isinstance(other, WorkflowUnit):
            return NotImplemented
        return super().__eq__(other) and self.steps == other.steps

    def get_truncated_text(self) -> str:
        return f"[Workflow '{self.name}' containing {len(self.steps)} steps omitted for brevity]"


CATEGORY_WEIGHTS = {
    "constraint": 1.0,
    "decision": 1.0,
    "reason": 1.0,
    "implementation": 1.0,
    "requirement": 1.0,
    "goal": 0.95,
    "workflow": 0.95,
    "task": 0.90,
    "outcome": 0.85,
    "comparison": 0.80,
    "question": 0.80,
    "fact": 0.70,
}


class UnitExtractorStrategy:
    """Base protocol for semantic text extraction strategies."""
    def extract(self, index: int, sent: Any) -> list[SemanticUnit]:
        raise NotImplementedError


class RegexKeyValueExtractor(UnitExtractorStrategy):
    """Extracts semantic units using configurable regex patterns."""
    def __init__(self, pattern: str, type_map: dict[str, str] = None, default_type: str = "fact"):
        self.pattern = re.compile(pattern, re.IGNORECASE)
        self.type_map = type_map or {}
        self.default_type = default_type

    def extract(self, index: int, sent: Any) -> list[SemanticUnit]:
        text = sent.text.strip()
        match = self.pattern.match(text)
        if match:
            groups = match.groups()
            # Handles SDLC formats e.g. [DEC-01] Title: Description
            if len(groups) == 3:
                doc_type, title, value = groups
                unit_type = self.type_map.get(doc_type.upper(), self.default_type)
                return [SemanticUnit(index, text, unit_type, subject=title.strip(), obj=value.strip())]
            # Handles Markdown Key: Value formats
            elif len(groups) == 2:
                label, value = groups
                unit_type = normalize_text(label).replace(" ", "_").lower()
                
                # Canonicalize memory types
                if unit_type in ["requirement", "rule"]: unit_type = "constraint"
                elif unit_type in ["future_action", "action"]: unit_type = "task"
                elif unit_type in ["open_question"]: unit_type = "question"
                elif unit_type in ["reason"]: unit_type = "fact"
                elif unit_type not in CATEGORY_WEIGHTS:
                    unit_type = "fact"
                    
                if value:
                    return [SemanticUnit(index, text, unit_type, obj=value)]
        return []


class DefaultNLPExtractor(UnitExtractorStrategy):
    """Fallback semantic extraction using SpaCy POS tagging and Dependency Parsing."""
    def extract(self, index: int, sent: Any) -> list[SemanticUnit]:
        units = []
        text = sent.text.strip()
        text_lower = text.lower()

        def get_phrase(token):
            if not token: return ""
            return " ".join([t.text for t in token.subtree]).strip()

        subj_token = next((t for t in sent if "subj" in t.dep_ and t.head == sent.root), None)
        main_subject = get_phrase(subj_token)
        obj_token = next((t for t in sent if t.dep_ in {"dobj", "pobj", "attr", "oprd", "acomp", "xcomp"} and t.head == sent.root), None)
        main_object = get_phrase(obj_token)
        relation = sent.root.text

        unit_type = "fact"
        if any(w in text_lower for w in ["chose", "decided", "selected", "opted", "resolved"]): unit_type = "decision"
        elif any(w in text_lower for w in ["goal", "protect", "preserve", "mission", "purpose", "aim"]): unit_type = "goal"
        elif any(w in text_lower for w in ["because", "resulted", "caused", "benefits", "outcome", "consequence"]): unit_type = "outcome"
        elif any(w in text_lower for w in ["must", "cannot", "required", "limited", "never"]): unit_type = "constraint"
        elif any(w in text_lower for w in ["compare", "vs", "versus", "better than", "faster", "lighter"]): unit_type = "comparison"
        elif any(w in text_lower for w in ["task", "implement", "benchmark", "create", "generate"]): unit_type = "task"
        elif text_lower.strip().endswith("?") or "open question" in text_lower: unit_type = "question"

        if main_subject or relation or main_object:
            units.append(SemanticUnit(index, sent.text, unit_type, main_subject, relation, main_object))
        else:
            units.append(SemanticUnit(index, sent.text, "fact", "", "", sent.text))

        for token in sent:
            if token.dep_ == "advcl" and token.tag_ == "TO":
                p_obj_token = next((c for c in token.children if c.dep_ == "dobj"), None)
                p_obj = get_phrase(p_obj_token)
                if token.text and p_obj:
                    units.append(SemanticUnit(index, sent.text, "goal", main_subject, token.text, p_obj))
            elif token.dep_ == "advcl" and token.text.lower() == "because":
                c_subj = get_phrase(next((c for c in token.head.children if "subj" in c.dep_), None))
                if c_subj:
                    units.append(SemanticUnit(index, sent.text, "outcome", c_subj, token.head.text, get_phrase(token.head)))

        return units


class SCRE:
    """
    Sentence-level Context Reduction Engine.
    A library class to reduce document context based on a query.
    """

    def __init__(self, model: str = "en_core_web_sm", extractors: list[UnitExtractorStrategy] | None = None, db_path: str = ":memory:"):
        """
        Initialize the engine and load the NLP model.
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._init_db()

        if spacy is not None and model:
            self.nlp = spacy.load(model)
        else:
            self.nlp = None
            
        self.content_pos = {"NOUN", "PROPN", "VERB", "ADJ"}
        self.answer_type_boosts = {
            "who": {"PERSON", "ORG"},
            "where": {"GPE", "LOC", "FAC"},
            "when": {"DATE", "TIME"},
        }
        
        if extractors is None:
            self.extractors = [
                # Custom SDLC Benchmark configuration decoupled to a strategy
                RegexKeyValueExtractor(r"^(?:-\s*)?\[([A-Z]+)-\d+\]\s*([^:]+):\s*(.*)", {
                    "FACT": "fact", "DEC": "decision", "CON": "constraint",
                    "TSK": "task", "WRK": "workflow", "REA": "reason", "STC": "state"
                }),
                # Common Markdown Configuration Strategy
                RegexKeyValueExtractor(r"^(?:#+\s*)?([\w\s-]+):\s*(.*)")
            ]
            if self.nlp is not None:
                self.extractors.append(DefaultNLPExtractor())
                
        else:
            self.extractors = extractors
        
        if SentenceTransformer is not None:
            self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        else:
            self.embedder = None

    def _init_db(self):
        c = self.conn.cursor()
        if self.db_path != ":memory:":
            c.execute('PRAGMA journal_mode=WAL;')
            c.execute('PRAGMA synchronous=NORMAL;')
            c.execute('PRAGMA cache_size=-64000;')
        c.execute('''
            CREATE TABLE IF NOT EXISTS memory_units (
                document_id TEXT,
                sentence_index INTEGER,
                text TEXT,
                unit_type TEXT,
                subject TEXT,
                relation TEXT,
                object TEXT,
                context_header TEXT,
                render_text TEXT,
                embedding BLOB
            )
        ''')
        c.execute('''
            CREATE TABLE IF NOT EXISTS knowledge_edges (
                document_id TEXT,
                node TEXT,
                sentence_idx INTEGER
            )
        ''')
        c.execute('''
            CREATE TABLE IF NOT EXISTS reasoning_edges (
                document_id TEXT,
                source_idx INTEGER,
                target_idx INTEGER
            )
        ''')
        c.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                document_id TEXT PRIMARY KEY,
                original_chars INTEGER,
                original_sentences INTEGER,
                original_estimated_tokens INTEGER
            )
        ''')
        self.conn.commit()

    def ingest(self, text: str, document_id: str = "default") -> None:
        """Asynchronous ingestion pipeline: chunk, parse, extract, embed, and store in SQLite."""
        c = self.conn.cursor()
        c.execute('SELECT 1 FROM documents WHERE document_id = ?', (document_id,))
        if c.fetchone():
            return  # Already ingested

        # Pre-process text to handle multi-line "Label: Value" structures,
        # which are common in structured documents but can be broken by sentence tokenizers.
        lines = text.split('\n')
        new_lines = []
        i = 0
        while i < len(lines):
            original_line = lines[i]
            line_stripped = original_line.strip()
            # Heuristic: Check if the line looks like a label (e.g., "Future Action:")
            # It ends with a colon, and only contains one colon.
            if line_stripped.endswith(':') and line_stripped.count(':') == 1:
                # If there is a non-empty next line, merge them.
                if i + 1 < len(lines) and lines[i+1].strip():
                    next_line_stripped = lines[i+1].strip()
                    # Do not merge if the next line is a list item
                    if not re.match(r'^\d+\.\s|^[\*\-]\s', next_line_stripped):
                        leading_whitespace = original_line[:len(original_line) - len(original_line.lstrip())]
                        merged_line = leading_whitespace + line_stripped + ' ' + next_line_stripped
                        new_lines.append(merged_line)
                        i += 2  # Skip the next line as it has been merged
                        continue
            new_lines.append(original_line)
            i += 1
        processed_text = '\n'.join(new_lines)

        class DummySentence:
            """Fallback object to mimic spaCy's sentence spans for RegexExtractors"""
            def __init__(self, text_val: str):
                self.text = text_val

        # 1. Sentence Segmentation
        sentences = []
        for paragraph in processed_text.split('\n\n'):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            lines = paragraph.split('\n')
            has_list_items = any(re.match(r'^\d+\.\s|^[\*\-]\s', l.strip()) for l in lines)
            
            if has_list_items:
                for line in lines:
                    line_str = line.strip()
                    if not line_str:
                        continue
                    # Force list items and potential headers to remain intact as single sentences
                    if re.match(r'^\d+\.\s|^[\*\-]\s', line_str) or line_str.endswith(':'):
                        if self.nlp:
                            doc = self.nlp(line_str)
                            sentences.append(doc[:])
                        else:
                            sentences.append(DummySentence(line_str))
                    else:
                        if self.nlp:
                            doc = self.nlp(line_str)
                            for sent in doc.sents:
                                if normalize_text(sent.text):
                                    sentences.append(sent)
                        else:
                            for sent_text in re.split(r'(?<=[.!?])\s+(?=[A-Z])', line_str):
                                if normalize_text(sent_text):
                                    sentences.append(DummySentence(sent_text))
            else:
                # Normal paragraph, let spaCy do its sentence segmentation
                if self.nlp:
                    doc = self.nlp(paragraph)
                    for sent in doc.sents:
                        if normalize_text(sent.text):
                            sentences.append(sent)
                else:
                    for sent_text in re.split(r'(?<=[.!?])\s+(?=[A-Z])', paragraph):
                        if normalize_text(sent_text):
                            sentences.append(DummySentence(sent_text))
                        
        if not sentences:
            return

        # 2. Semantic Classification & 3. Semantic State Builder
        all_extracted_units = self._build_semantic_state(sentences)

        knowledge_graph = self._build_knowledge_graph(all_extracted_units)
        reasoning_graph = self._build_reasoning_graph(all_extracted_units)

        # Dense Embeddings
        unit_embs = None
        if self.embedder and all_extracted_units:
            unit_texts = [u.original_sentence_text for u in all_extracted_units]
            unit_embs = self.embedder.encode(unit_texts, convert_to_numpy=True)

        for idx, unit in enumerate(all_extracted_units):
            emb_blob = None
            if unit_embs is not None:
                emb_blob = unit_embs[idx].astype(np.float32).tobytes()
            
            c.execute('''
                INSERT INTO memory_units (document_id, sentence_index, text, unit_type, subject, relation, object, context_header, render_text, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                document_id,
                unit.original_sentence_index,
                unit.original_sentence_text,
                unit.type,
                unit.subject,
                unit.relation,
                unit.object,
                getattr(unit, "context_header", ""),
                getattr(unit, "render_text", unit.original_sentence_text),
                emb_blob
            ))

        for node, indices in knowledge_graph.items():
            for idx in indices:
                c.execute('INSERT INTO knowledge_edges (document_id, node, sentence_idx) VALUES (?, ?, ?)', (document_id, node, idx))
                
        for src, targets in reasoning_graph.items():
            for tgt in targets:
                c.execute('INSERT INTO reasoning_edges (document_id, source_idx, target_idx) VALUES (?, ?, ?)', (document_id, src, tgt))
                
        c.execute('''
            INSERT INTO documents (document_id, original_chars, original_sentences, original_estimated_tokens)
            VALUES (?, ?, ?, ?)
        ''', (document_id, len(text), len(sentences), estimate_tokens(text)))
        
        self.conn.commit()

    def retrieve(self, query: str, document_id: str = "default", max_sentences: int = 4, max_tokens: int | None = None, context_window: int = 1, min_tokens: int = 250) -> dict[str, Any]:
        """Phase 2: Hybrid Retrieval Query against SQLite."""
        c = self.conn.cursor()
        c.execute('SELECT original_chars, original_sentences, original_estimated_tokens FROM documents WHERE document_id = ?', (document_id,))
        doc_row = c.fetchone()
        if not doc_row:
            return {"context": "", "metadata": {"reduction_ratio": 0.0, "original_chars": 0, "reduced_chars": 0}}
            
        original_chars, original_sentences, original_estimated_tokens = doc_row

        c.execute('SELECT sentence_index, text, unit_type, subject, relation, object, context_header, render_text, embedding FROM memory_units WHERE document_id = ? ORDER BY sentence_index', (document_id,))
        rows = c.fetchall()
        if not rows:
            return {"context": "", "metadata": {"reduction_ratio": 0.0, "original_chars": 0, "reduced_chars": 0}}

        all_extracted_units = []
        unit_embs_list = []
        sentences_for_idf = []
        has_embeddings = False
        
        class DummySent:
            def __init__(self, t): self.text = t
            
        for r in rows:
            u = SemanticUnit(r[0], r[1], r[2], r[3], r[4], r[5])
            u.context_header = r[6]
            u.render_text = r[7]
            all_extracted_units.append(u)
            sentences_for_idf.append(DummySent(r[1]))
            if r[8]:
                has_embeddings = True
                unit_embs_list.append(np.frombuffer(r[8], dtype=np.float32))

        knowledge_graph = {}
        c.execute('SELECT node, sentence_idx FROM knowledge_edges WHERE document_id = ?', (document_id,))
        for r in c.fetchall():
            knowledge_graph.setdefault(r[0], set()).add(r[1])
            
        reasoning_graph = {}
        c.execute('SELECT source_idx, target_idx FROM reasoning_edges WHERE document_id = ?', (document_id,))
        for r in c.fetchall():
            reasoning_graph.setdefault(r[0], set()).add(r[1])

        self._apply_graph_connectivity(knowledge_graph, all_extracted_units)
        
        query_features = self._get_query_features(query)
        idf = compute_idf(sentences_for_idf)

        dense_scores = []
        if self.embedder and has_embeddings:
            import torch
            query_emb = self.embedder.encode(query, convert_to_tensor=True)
            unit_embs_tensor = torch.tensor(np.stack(unit_embs_list)).to(query_emb.device)
            dense_scores = util.cos_sim(query_emb, unit_embs_tensor)[0].tolist()
            
        for idx, unit in enumerate(all_extracted_units):
            dense_score = dense_scores[idx] if dense_scores else 0.0
            self._score_semantic_unit(unit, query_features, idf, dense_score)
            
        selected_units = select_semantic_units(
            units=all_extracted_units,
            max_sentences=max_sentences,
            max_tokens=max_tokens,
            min_tokens=min_tokens,
        )

        if selected_units:
            if context_window > 0:
                selected_units = self._expand_adjacent_context(selected_units, all_extracted_units, context_window)
            selected_units = self._expand_reasoning_chains(selected_units, all_extracted_units, reasoning_graph)

        reduced_text = build_compressed_context(selected_units)
        return {
            "context": reduced_text,
            "metadata": {
                "original_sentences": original_sentences,
                "reduced_sentences": len(selected_units),
                "selected_sentence_count": len(selected_units),
                "original_chars": original_chars,
                "reduced_chars": len(reduced_text),
                "original_estimated_tokens": original_estimated_tokens,
                "reduced_estimated_tokens": estimate_tokens(reduced_text),
                "reduction_ratio": round(1 - (len(reduced_text) / max(original_chars, 1)), 4)
            }
        }
        
    def reduce(
        self,
        text: str,
        query: str,
        max_sentences: int = 4,
        max_tokens: int | None = None,
        context_window: int = 1,
        min_tokens: int = 250,
    ) -> dict[str, Any]:
        """Legacy API for backwards compatibility. Uses ephemeral document IDs."""
        doc_id = hashlib.md5(text.encode('utf-8')).hexdigest()
        self.ingest(text, doc_id)
        return self.retrieve(query, doc_id, max_sentences, max_tokens, context_window, min_tokens)

    def _build_semantic_state(self, sentences: list[Any]) -> list[SemanticUnit]:
        all_extracted_units = []
        current_h1 = ""
        
        i = 0
        while i < len(sentences):
            sent = sentences[i]
            text = sent.text.strip()
            
            # Hierarchical Context Tracking: Preserve the top-level document subject
            if text.startswith("# "):
                clean_header = text.strip("# \t")
                if i > 0 and sentences[i-1].text.strip().startswith("# "):
                    current_h1 += f" | {clean_header}"
                else:
                    current_h1 = clean_header
            
            is_list_item = lambda t: bool(re.match(r'^\d+\.\s|^[\*\-]\s', t))
            
            # Workflow Detection: Header followed by ordered procedural steps
            if i + 1 < len(sentences) and not is_list_item(text) and is_list_item(sentences[i+1].text.strip()):
                header_text = text.rstrip(':').strip()
                steps = []
                j = i + 1
                while j < len(sentences) and is_list_item(sentences[j].text.strip()):
                    steps.append(sentences[j].text.strip())
                    j += 1
                
                workflow_text = text + "\n" + "\n".join(steps)
                wf_unit = WorkflowUnit(i, header_text, steps, workflow_text)
                wf_unit.context_header = current_h1
                all_extracted_units.append(wf_unit)
                i = j
                continue
            
            # Fallback Workflow Detection: Encountered list items without a valid preceding header
            if is_list_item(text):
                header_text = "Unnamed Workflow"
                steps = [text]
                j = i + 1
                while j < len(sentences) and is_list_item(sentences[j].text.strip()):
                    steps.append(sentences[j].text.strip())
                    j += 1
                
                workflow_text = "\n".join(steps)
                wf_unit = WorkflowUnit(i, header_text, steps, workflow_text)
                wf_unit.context_header = current_h1
                all_extracted_units.append(wf_unit)
                i = j
                continue

            extracted = self._extract_semantic_units_from_sentence(i, sent)
            for unit in extracted:
                unit.context_header = current_h1
            all_extracted_units.extend(extracted)
            i += 1
            
        return all_extracted_units

    def _extract_semantic_units_from_sentence(self, index: int, sent: Any) -> list[SemanticUnit]:
        # Traverse configurable extraction strategies dynamically
        for extractor in self.extractors:
            extracted = extractor.extract(index, sent)
            if extracted:
                return extracted
        return []

    def _apply_graph_connectivity(self, graph: dict[str, set[int]], all_units: list[SemanticUnit]):
        for unit in all_units:
            nodes = {unit.subject.lower(), unit.object.lower()}
            connectivity = 0
            for node in nodes:
                if node and node in graph:
                    connectivity += len(graph[node])
            unit.connectivity = connectivity

    def _score_semantic_unit(self, unit: SemanticUnit, query_features: dict[str, Any], idf: dict[str, float], dense_score: float = 0.0):
        text_lower = unit.original_sentence_text.lower()
        base_weight = CATEGORY_WEIGHTS.get(unit.type, 0.5)
        
        # 1. Semantic Similarity (Term Overlap weighted by IDF)
        matched_terms = [t for t in query_features["terms"] if t in text_lower]
        lexical_similarity = sum(idf.get(t, 1.0) for t in matched_terms)
        
        # Non-linear boost for matching multiple query terms (context density)
        if len(matched_terms) > 1:
            lexical_similarity *= (1.5 ** (len(matched_terms) - 1))
        
        # Fix missing phrase_bonus logic
        phrase_bonus = 0.0
        query_phrase = " ".join(query_features.get("clean_query_words", []))
        if query_phrase and len(query_phrase) > 4 and query_phrase in text_lower:
            phrase_bonus = 5.0

        # Dense embedding bonus to overcome lexical gaps (e.g., "unfinished" vs "pending")
        # Calibrated dense scoring so high semantic matches aren't buried by lexical noise
        dense_bonus = 0.0
        if dense_score > 0.65:
            dense_bonus = dense_score * 12.0  # Scaled down to prevent total override
        elif dense_score > 0.45:
            dense_bonus = dense_score * 6.0
        else:
            dense_bonus = max(0.0, dense_score) * 2.0
        
        # 2. Intent Detection & Type Match
        intent_bonus = 0
        type_terms = set(unit.type.split('_'))
        if type_terms & query_features["terms"]:
            intent_bonus = 3.0  # Intent aligns with node type
            
        # Workflow Relevance: Boost if query asks about pipelines, workflows, or processes
        if unit.type == "workflow" and any(w in query_features["terms"] for w in ["pipeline", "workflow", "process", "steps", "stages"]):
            intent_bonus += 3.0

        # NEW: Hard Identifier / Keyword Boost
        hard_id_boost = 0.0
        for hid in query_features.get("hard_identifiers", set()):
            if hid in unit.original_sentence_text or hid.lower() in text_lower:
                hard_id_boost += 15.0  # Massive boost for exact match of alphanumeric identifiers

        # 3. Entity & Noun Match (Broadened to include noun chunks because NER misses tech terms)
        entity_matches = [e for e in query_features["entities"] if e in text_lower]
        entity_match = sum(3.0 for e in entity_matches)
        
        noun_matches = [n for n in query_features.get("noun_chunks", set()) if n in text_lower]
        noun_match = sum(2.0 for n in noun_matches)

        # Generalized pronoun coreference boost if entities are queried
        if query_features["entities"] and any(p in text_lower.split() for p in ["he", "she", "it", "its", "his", "her", "they", "their"]):
            entity_match += 1.0

        # 4. Relationship Distance (Graph Connectivity)
        relationship_distance = min(unit.connectivity * 0.1, 1.0)
        
        unit.score = (base_weight * 2.0) + lexical_similarity + phrase_bonus + dense_bonus + entity_match + noun_match + relationship_distance + intent_bonus + hard_id_boost

        # Preserve critical nodes if relevant
        high_priority_types = {"decision", "constraint", "goal", "workflow", "task", "question"}
        if unit.type in high_priority_types:
            if lexical_similarity > 0 or entity_match > 0 or noun_match > 0 or dense_score > 0.45 or hard_id_boost > 0:
                unit.score += 2.5 # Increased preservation weight for critical logic nodes

    def _build_reasoning_graph(self, all_units: list[SemanticUnit]) -> dict[int, set[int]]:
        """
        Constructs a graph of reasoning chains (e.g., Decision -> Reason -> Implementation).
        Generates edges for multi-hop retrieval.
        """
        reasoning_graph: dict[int, set[int]] = {u.original_sentence_index: set() for u in all_units}
        last_seen = {}
        
        causal_markers = [
            "because", "due to", "causes", "results in", "enables", 
            "requires", "depends on", "implements", "satisfies", 
            "violates", "fulfills", "supports", "selected because", "chosen because"
        ]
        
        for i, unit in enumerate(all_units):
            def link_if_close(u1, u2, max_dist=15):
                if u1 and u2 and abs(u1.original_sentence_index - u2.original_sentence_index) <= max_dist:
                    reasoning_graph[u1.original_sentence_index].add(u2.original_sentence_index)
                    reasoning_graph[u2.original_sentence_index].add(u1.original_sentence_index)

            # Explicit structural relationships
            if unit.type == "reason":
                link_if_close(unit, last_seen.get("decision"))
            elif unit.type == "decision":
                link_if_close(unit, last_seen.get("constraint"))
                link_if_close(unit, last_seen.get("requirement"))
                link_if_close(unit, last_seen.get("goal"))
            elif unit.type in ["implementation", "requirement", "task"]:
                link_if_close(unit, last_seen.get("decision"))
                link_if_close(unit, last_seen.get("reason"))
                link_if_close(unit, last_seen.get("goal"))

            # Lexical causal markers
            text_lower = unit.original_sentence_text.lower()
            
            if any(marker in text_lower for marker in causal_markers):
                # O(N^2) refactored to O(N) using sliding window boundaries 
                start_idx = max(0, i - 5)
                end_idx = min(len(all_units), i + 6)
                for j in range(start_idx, end_idx):
                    if i != j:
                        other = all_units[j]
                        if abs(other.original_sentence_index - unit.original_sentence_index) <= 2:
                            reasoning_graph[unit.original_sentence_index].add(other.original_sentence_index)
                            reasoning_graph[other.original_sentence_index].add(unit.original_sentence_index)
            
            last_seen[unit.type] = unit
            
        return reasoning_graph

    def _expand_reasoning_chains(self, selected: list[SemanticUnit], all_units: list[SemanticUnit], reasoning_graph: dict[int, set[int]]) -> list[SemanticUnit]:
        """
        Expands the selection by traversing the reasoning graph (multi-hop).
        Ensures isolated nodes pull in their complete reasoning chain.
        """
        selected_indices = {u.original_sentence_index for u in selected}
        expanded_indices = set(selected_indices)
        
        # 2-hop traversal to fetch the complete chain for selected nodes
        for idx in selected_indices:
            hop1 = reasoning_graph.get(idx, set())
            expanded_indices.update(hop1)
            for h1_idx in hop1:
                hop2 = reasoning_graph.get(h1_idx, set())
                expanded_indices.update(hop2)

        expanded = []
        for unit in all_units:
            if unit.original_sentence_index in expanded_indices and unit not in expanded:
                expanded.append(unit)
                
        return sorted(expanded, key=lambda x: x.original_sentence_index)

    def _expand_adjacent_context(self, selected: list[SemanticUnit], all_units: list[SemanticUnit], window: int) -> list[SemanticUnit]:
        """
        Expands the selection to include adjacent sentences based on the context window.
        Crucial for resolving coreferences like pronouns across sentence boundaries.
        """
        selected_indices = {u.original_sentence_index for u in selected}
        target_indices = set(selected_indices)
        
        for idx in selected_indices:
            for w in range(1, window + 1):
                target_indices.add(idx - w)
                target_indices.add(idx + w)
                
        expanded = list(selected)
        existing = {u for u in selected}
        
        for u in all_units:
            if u.original_sentence_index in target_indices and u not in existing:
                expanded.append(u)
                existing.add(u)
                    
        return sorted(expanded, key=lambda x: x.original_sentence_index)

    def _build_knowledge_graph(self, all_units: list[SemanticUnit]) -> dict[str, set[int]]:
        """
        Constructs the Knowledge Graph adjacency list from SemanticUnits.
        """
        graph: dict[str, set[int]] = {}
        for unit in all_units:
            if unit.type == "workflow":
                continue
            nodes = set()
            if unit.subject:
                nodes.add(unit.subject.lower())
            if unit.object:
                nodes.add(unit.object.lower())
            for mention in nodes:
                graph.setdefault(mention, set()).add(unit.original_sentence_index)
        return graph

    def _get_query_features(self, query: str) -> dict[str, Any]:
        query_terms = []
        entities = set()
        noun_chunks = set()
        wh_words = set()
        root_lemma = ""
        
        if self.nlp:
            query_doc = self.nlp(query)
            for token in query_doc:
                if token.is_stop or token.is_punct or token.like_num:
                    continue
                lemma = token.lemma_.lower().strip()
                text_val = token.text.lower().strip()
                if len(lemma) >= 2:
                    query_terms.append(lemma)
                if len(text_val) >= 2 and text_val != lemma:
                    query_terms.append(text_val)
            
            entities = {ent.text.lower().strip() for ent in query_doc.ents if ent.text.strip()}
            noun_chunks = {c.text.lower().strip() for c in query_doc.noun_chunks if len(c.text.strip()) > 2}
            wh_words = {t.lower_ for t in query_doc if t.lower_ in self.answer_type_boosts}
            root_lemma = query_doc[:].root.lemma_.lower().strip()
        else:
            # Lexical Fallback without spaCy
            words = re.findall(r'\b\w+\b', query)
            query_terms = [w.lower() for w in words if len(w) > 2]
            wh_words = {w.lower() for w in words if w.lower() in self.answer_type_boosts}

        # Extract hard identifiers (e.g., AGENT-INT-03, SOC2, [REQ-01], OpenTelemetry, v20)
        regex_pattern = r'\b[A-Z]+-[A-Z0-9\-]+\b|\b[A-Z0-9]{3,}\b|\[[A-Z]+-\d+\]|\b[A-Z][a-z]+[A-Z][a-zA-Z0-9]*\b|\bv\d+(?:\.\d+)*\b'
        hard_identifiers = set(re.findall(regex_pattern, query))
        
        clean_query_words = re.sub(r'\W+', ' ', query.lower()).split()

        return {
            "terms": set(query_terms),
            "entities": entities,
            "noun_chunks": noun_chunks,
            "wh_words": wh_words,
            "root_lemma": root_lemma,
            "hard_identifiers": hard_identifiers,
            "clean_query_words": clean_query_words,
        }


def build_compressed_context(units: list[SemanticUnit]) -> str:
    # Sort units by their original position in the document to maintain narrative flow.
    ordered_units = sorted(units, key=lambda x: x.original_sentence_index)

    # Use a dictionary to ensure each original sentence appears only once,
    # and utilize the updated render_text to support Workflow truncation and JSON representation.
    unique_sentences = {}
    last_header = ""
    for u in ordered_units:
        if u.original_sentence_index not in unique_sentences:
            text_to_render = u.render_text
            
            # Contextual Prefixing: Inject document metadata to prevent context detachment
            if getattr(u, "context_header", "") and not u.original_sentence_text.startswith("#"):
                if u.context_header != last_header:
                    text_to_render = f"[{u.context_header}]\n{text_to_render}"
                    last_header = u.context_header
                    
            unique_sentences[u.original_sentence_index] = text_to_render
            
    return "\n".join(unique_sentences.values()).strip()


def select_semantic_units(units: list[SemanticUnit], max_sentences: int, max_tokens: int | None, min_tokens: int = 250) -> list[SemanticUnit]:
    ranked = sorted([u for u in units if u.score > 0], key=lambda x: x.score, reverse=True)
    selected = []
    tokens = 0
    
    for u in ranked:
        is_duplicate = False
        for s in selected:
            if jaccard_similarity(u.original_sentence_text, s.original_sentence_text) > 0.85:
                is_duplicate = True
                break
            if u == s:
                is_duplicate = True
                break
        
        if is_duplicate:
            continue

        u_tokens = estimate_tokens(u.render_text)
        
        if max_tokens and tokens + u_tokens > max_tokens:
            if getattr(u, "type", "") == "workflow" and hasattr(u, "get_truncated_text"):
                truncated_text = u.get_truncated_text()
                trunc_tokens = estimate_tokens(truncated_text)
                if tokens + trunc_tokens <= max_tokens:
                    u.render_text = truncated_text
                    selected.append(u)
                    tokens += trunc_tokens
                    if len(selected) >= max_sentences: break
                    continue
            break
        
        selected.append(u)
        tokens += u_tokens
        
        # Dynamic Thresholding: Ensure we don't cut off too early if we haven't met the minimum token floor
        if len(selected) >= max_sentences:
            if tokens >= min_tokens or (max_tokens and tokens >= max_tokens):
                break
    return selected
