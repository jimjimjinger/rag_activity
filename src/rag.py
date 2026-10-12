"""LangChain Documents, local BGE-M3/FAISS retrieval, and a grounded Qwen answer."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import faiss
import httpx
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_ollama import OllamaEmbeddings
from langsmith import tracing_context
from ollama import Client, ResponseError

from llm_client import BASE_URL, CLIENT_KWARGS, chat

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data/rag/corpus.jsonl"
INDEX = ROOT / "data/rag/index.json"
VECTORS = ROOT / "data/rag/index.faiss"
EMBED_MODEL = "bge-m3:567m"
MAX_CONTEXT_CHARS = 8000


class LocalRetriever(BaseRetriever):
    """LangChain retriever backed by a normalized, exact FAISS L2 index."""
    index: Any
    documents: list[Document]
    embedding: Embeddings
    k: int = 3

    def _get_relevant_documents(self, query, *, run_manager):
        vector = np.asarray([self.embedding.embed_query(query)], dtype="float32")
        if vector.shape[1] != self.index.d:
            raise ValueError("Embedding dimensions changed; rebuild the index")
        faiss.normalize_L2(vector)
        distances, positions = self.index.search(vector, min(self.k, len(self.documents)))
        results = []
        for distance, position in zip(distances[0], positions[0]):
            doc = self.documents[int(position)]
            score = max(-1.0, min(1.0, 1.0 - float(distance) / 2.0))
            results.append(Document(id=doc.id, page_content=doc.page_content,
                                    metadata={**doc.metadata, "score": score}))
        return results


def embeddings(keep_alive=0):
    return OllamaEmbeddings(
        model=EMBED_MODEL, base_url=BASE_URL, num_thread=4,
        keep_alive=keep_alive, client_kwargs=CLIENT_KWARGS,
    )


def model_digest():
    client = Client(host=BASE_URL, **CLIENT_KWARGS)
    for model in client.list().models:
        if model.model == EMBED_MODEL:
            return model.digest
    raise ValueError(f"Model missing: ollama pull {EMBED_MODEL}")


def load_documents():
    chunks = [json.loads(line) for line in CORPUS.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not chunks:
        raise ValueError("Corpus is empty; run scripts/build-rag.ps1 first")
    return [Document(id=c["id"], page_content=c["content"],
                     metadata={k: v for k, v in c.items() if k != "content"}) for c in chunks]


def validate_sources(documents):
    sources = {(d.metadata["source_pdf"], d.metadata["source_sha256"]) for d in documents}
    for name, digest in sources:
        source = ROOT / "documents" / Path(name).name
        if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
            raise ValueError("Source PDF changed; rerun parsing and scripts/build-rag.ps1")


def build_index():
    corpus_hash = hashlib.sha256(CORPUS.read_bytes()).hexdigest()
    documents = load_documents()
    validate_sources(documents)
    digest = model_digest()
    embedding = embeddings(keep_alive=120)
    index = None
    for i in range(0, len(documents), 2):
        batch = documents[i:i + 2]
        vectors = np.asarray(embedding.embed_documents([d.page_content for d in batch]), dtype="float32")
        # squared L2 = 2 - 2*cosine for normalized documents and queries.
        faiss.normalize_L2(vectors)
        if index is None:
            index = faiss.IndexFlatL2(vectors.shape[1])
        index.add(vectors)
        print(f"Embedded {min(i + 2, len(documents))}/{len(documents)} chunks", flush=True)
    if corpus_hash != hashlib.sha256(CORPUS.read_bytes()).hexdigest():
        raise ValueError("Corpus changed during indexing; rebuild the index")
    vector_bytes = faiss.serialize_index(index).tobytes()
    manifest = {
        "version": 2, "model": EMBED_MODEL, "model_digest": digest,
        "metric": "normalized_squared_l2", "corpus_sha256": corpus_hash,
        "index_sha256": hashlib.sha256(vector_bytes).hexdigest(),
        "documents": [
            {"id": d.id, "page_content": d.page_content, "metadata": d.metadata}
            for d in documents
        ],
    }
    # Native FAISS bytes + JSON metadata; no pickle deserialization needed.
    # Replacing the manifest last lets the loader detect a mismatched pair.
    vector_tmp = VECTORS.with_suffix(".tmp")
    vector_tmp.write_bytes(vector_bytes)
    vector_tmp.replace(VECTORS)
    manifest_tmp = INDEX.with_suffix(".tmp")
    manifest_tmp.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    manifest_tmp.replace(INDEX)
    print(f"Saved {len(documents)} Documents: {VECTORS}")


def read_manifest():
    if not INDEX.exists():
        raise ValueError("Index missing: run scripts/build-rag.ps1 first")
    manifest = json.loads(INDEX.read_text(encoding="utf-8"))
    if manifest.get("version") != 2 or manifest.get("metric") != "normalized_squared_l2":
        raise ValueError("Old index format: run scripts/build-rag.ps1 first")
    if hashlib.sha256(CORPUS.read_bytes()).hexdigest() != manifest["corpus_sha256"]:
        raise ValueError("Corpus changed; run scripts/build-rag.ps1 first")
    if manifest["model"] != EMBED_MODEL:
        raise ValueError("Embedding model changed; rebuild the index")
    return manifest


def load_retriever(top_k=3):
    manifest = read_manifest()
    documents = [Document(**d) for d in manifest["documents"]]
    validate_sources(documents)
    if model_digest() != manifest["model_digest"]:
        raise ValueError("Embedding model changed; rebuild the index")
    vector_bytes = VECTORS.read_bytes()
    if hashlib.sha256(vector_bytes).hexdigest() != manifest["index_sha256"]:
        raise ValueError("FAISS index and metadata differ; rebuild the index")
    index = faiss.deserialize_index(np.frombuffer(vector_bytes, dtype="uint8").copy())
    if index.ntotal != len(documents) or index.metric_type != faiss.METRIC_L2:
        raise ValueError("Invalid FAISS index; rebuild the index")
    return LocalRetriever(index=index, documents=documents, embedding=embeddings(), k=top_k)


def retrieve(question, top_k=3):
    if not question.strip() or top_k < 1:
        raise ValueError("Provide a question and a positive top-k")
    with tracing_context(enabled=False):
        documents = load_retriever(top_k).invoke(question)
    return [(doc, doc.metadata["score"]) for doc in documents]


def build_context(hits, max_chars=MAX_CONTEXT_CHARS):
    sections, skipped = [], []
    used = 0
    for doc, _ in hits:
        section = f"[p.{doc.metadata['page']}]\n{doc.page_content}"
        if used + len(section) + 2 > max_chars:
            skipped.append(doc.id)
            continue
        sections.append(section)
        used += len(section) + 2
    if not sections:
        raise ValueError("No context fits the input limit; split the Markdown into smaller chunks.")
    return "\n\n".join(sections), skipped


def ask(question, search_only=False, top_k=3):
    hits = retrieve(question, top_k)
    print("\n검색 근거 (코사인 유사도는 정답 확률이 아닙니다):")
    for doc, score in hits:
        print(f"  [p.{doc.metadata['page']}] {doc.id} / {doc.metadata['section']} / {score:.3f}")
    context, skipped = build_context(hits)
    if skipped:
        print("  입력 길이 제한으로 제외된 청크: " + ", ".join(skipped))
    if not search_only:
        print("\n로컬 Qwen이 근거를 읽고 있습니다. CPU에서는 시간이 걸릴 수 있습니다.\n답변:", flush=True)
        return chat(question, context)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--search-only", action="store_true")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    try:
        if args.top_k < 1:
            raise ValueError("--top-k must be positive")
        if args.build:
            build_index()
        elif args.question:
            ask(args.question, args.search_only, args.top_k)
        else:
            manifest = read_manifest()
            pages = sorted({d["metadata"]["page"] for d in manifest["documents"]})
            scope = f"1~{pages[-1]}페이지" if pages == list(range(1, pages[-1] + 1)) else "·".join(map(str, pages)) + "페이지"
            print(f"B2B 매뉴얼 RAG: {scope}, {len(manifest['documents'])}개 청크, top-k={args.top_k}")
            print("질문마다 독립적으로 검색합니다. 종료: /exit")
            while True:
                question = input("\n질문> ").strip()
                if question == "/exit":
                    break
                if question:
                    ask(question, args.search_only, args.top_k)
    except (OSError, ValueError, RuntimeError, KeyError, httpx.HTTPError, ResponseError) as exc:
        print(f"RAG 실행 실패: {exc}", file=sys.stderr)
        return 1
    except (KeyboardInterrupt, EOFError):
        return 0
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    raise SystemExit(main())
