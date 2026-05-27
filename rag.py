# -*- coding: utf-8 -*-
"""Small local RAG helper using scikit-learn TF-IDF.

This keeps the hackathon project runnable without a vector database. The interface is
simple enough to replace later with FAISS, Milvus, or NVIDIA retrieval components.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Iterable

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:  # pragma: no cover
    TfidfVectorizer = None
    cosine_similarity = None


@dataclass
class EvidenceChunk:
    id: str
    source: str
    category: str
    text: str
    score: float = 0.0


def _read_text(path: Path) -> str:
    return path.read_text(encoding='utf-8', errors='ignore')


def _chunk_text(text: str, max_chars: int = 900) -> List[str]:
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    chunks = []
    cur = ''
    for p in paragraphs:
        if len(cur) + len(p) + 2 > max_chars and cur:
            chunks.append(cur.strip())
            cur = p
        else:
            cur = (cur + '\n\n' + p).strip() if cur else p
    if cur:
        chunks.append(cur.strip())
    return chunks


class SimpleRAG:
    def __init__(self, knowledge_dir: str = 'knowledge_base', rag_sources_dir: str = 'rag_sources') -> None:
        self.knowledge_dir = Path(knowledge_dir)
        self.rag_sources_dir = Path(rag_sources_dir)
        self.chunks: List[EvidenceChunk] = []
        self.vectorizer = None
        self.matrix = None
        self.refresh()

    def refresh(self) -> None:
        self.chunks = []
        idx = 1
        if self.knowledge_dir.exists():
            for path in sorted(self.knowledge_dir.glob('**/*')):
                if not path.is_file() or path.suffix.lower() not in {'.md', '.txt'}:
                    continue
                for chunk in _chunk_text(_read_text(path)):
                    self.chunks.append(EvidenceChunk(f'E{idx}', path.name, path.parent.name, chunk))
                    idx += 1

        if self.rag_sources_dir.exists():
            for path in sorted(self.rag_sources_dir.glob('*.jsonl')):
                with path.open(encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        try:
                            obj = json.loads(line)
                        except Exception:
                            continue
                        text = obj.get('rag_text') or obj.get('abstract') or obj.get('summary') or ''
                        if not text:
                            continue
                        self.chunks.append(EvidenceChunk(
                            obj.get('id') or f'E{idx}',
                            obj.get('source') or path.name,
                            'external_refresh',
                            text[:2000],
                        ))
                        idx += 1

        if TfidfVectorizer is None or not self.chunks:
            self.vectorizer = None
            self.matrix = None
            return
        self.vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=20000)
        self.matrix = self.vectorizer.fit_transform([c.text for c in self.chunks])

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self.chunks:
            return []
        if self.vectorizer is None or self.matrix is None or cosine_similarity is None:
            return [c.__dict__ for c in self.chunks[:top_k]]
        q = self.vectorizer.transform([query or ''])
        sims = cosine_similarity(q, self.matrix).ravel()
        order = sims.argsort()[::-1][:top_k]
        out = []
        for i in order:
            c = self.chunks[int(i)]
            out.append({
                'id': c.id,
                'source': c.source,
                'category': c.category,
                'text': c.text,
                'score': float(sims[int(i)]),
            })
        return out

    def status(self) -> Dict[str, Any]:
        return {'chunks': len(self.chunks), 'knowledge_dir': str(self.knowledge_dir), 'rag_sources_dir': str(self.rag_sources_dir)}
