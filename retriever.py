from __future__ import annotations

import pickle
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from config import get_settings


@dataclass
class HybridRetriever:
    vector_store: FAISS
    documents: list[Document]
    bm25: BM25Okapi
    settings: Any

    def _tokenize(self, text: str) -> list[str]:
        return [token for token in text.lower().split() if token.strip()]

    def semantic_search(self, query: str, k: int | None = None) -> list[tuple[Document, float]]:
        k = k or self.settings.semantic_k
        docs_and_scores = self.vector_store.similarity_search_with_score(query, k=k)
        results: list[tuple[Document, float]] = []
        for doc, score in docs_and_scores:
            similarity_like = 1.0 / (1.0 + float(score))
            results.append((doc, similarity_like))
        return results

    def bm25_search(self, query: str, k: int | None = None) -> list[tuple[Document, float]]:
        k = k or self.settings.bm25_k
        tokenized_query = self._tokenize(query)
        if not tokenized_query:
            return []
        scores = self.bm25.get_scores(tokenized_query)
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[:k]
        return [(self.documents[idx], float(score)) for idx, score in ranked if score > 0]

    def hybrid_search(self, query: str, k: int | None = None) -> list[Document]:
        k = k or self.settings.retrieval_top_k
        semantic = self.semantic_search(query, k=self.settings.semantic_k)
        bm25 = self.bm25_search(query, k=self.settings.bm25_k)

        score_map: dict[str, dict[str, Any]] = {}
        for rank, (doc, score) in enumerate(semantic, start=1):
            key = self._doc_key(doc)
            entry = score_map.setdefault(key, {"doc": doc, "score": 0.0})
            entry["score"] += 0.65 * score + 0.1 / rank

        max_bm25 = max((score for _, score in bm25), default=1.0)
        for rank, (doc, score) in enumerate(bm25, start=1):
            key = self._doc_key(doc)
            entry = score_map.setdefault(key, {"doc": doc, "score": 0.0})
            normalized = score / max_bm25 if max_bm25 else 0.0
            entry["score"] += 0.35 * normalized + 0.1 / rank

        combined = sorted(score_map.values(), key=lambda item: item["score"], reverse=True)
        candidates = [item["doc"] for item in combined[: max(k * 2, self.settings.rerank_top_n)]]
        if not candidates:
            return []

        if len(candidates) <= 3:
            return candidates[:k]

        return self.rerank(query, candidates, top_n=min(k, len(candidates)))

    def rerank(self, query: str, documents: list[Document], top_n: int | None = None) -> list[Document]:
        top_n = top_n or self.settings.rerank_top_n
        try:
            reranker = get_reranker()
            pairs = [(query, doc.page_content) for doc in documents]
            scores = reranker.predict(pairs)
            ranked = sorted(zip(documents, scores), key=lambda item: float(item[1]), reverse=True)
            return [doc for doc, _ in ranked[:top_n]]
        except Exception:
            return documents[:top_n]

    def _doc_key(self, doc: Document) -> str:
        source = str(doc.metadata.get("source", ""))
        page = str(doc.metadata.get("page", ""))
        chunk = str(doc.metadata.get("chunk_id", ""))
        preview = doc.page_content[:80]
        return f"{source}|{page}|{chunk}|{preview}"


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    settings = get_settings()
    return CrossEncoder(settings.reranker_model)


@lru_cache(maxsize=1)
def get_retriever() -> HybridRetriever:
    settings = get_settings()
    index_path = settings.index_path
    faiss_dir = index_path / "faiss_index"
    chunks_path = index_path / "chunks.pkl"
    bm25_tokens_path = index_path / "bm25_tokens.pkl"

    if not faiss_dir.exists() or not chunks_path.exists() or not bm25_tokens_path.exists():
        raise FileNotFoundError(
            "Knowledge index not found. Run `python ingest.py` first to build the local knowledge base."
        )

    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key.get_secret_value(),
    )
    vector_store = FAISS.load_local(
        str(faiss_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )

    with chunks_path.open("rb") as f:
        documents: list[Document] = pickle.load(f)
    with bm25_tokens_path.open("rb") as f:
        tokenized_corpus: list[list[str]] = pickle.load(f)

    bm25 = BM25Okapi(tokenized_corpus)

    return HybridRetriever(
        vector_store=vector_store,
        documents=documents,
        bm25=bm25,
        settings=settings,
    )