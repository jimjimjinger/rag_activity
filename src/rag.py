"""LangChain FAISS retrieval with local BGE-M3 and a grounded Qwen answer."""
import argparse
import json
from pathlib import Path
import sys

import httpx
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langsmith import tracing_context
from ollama import ResponseError

from llm_client import BASE_URL, CLIENT_KWARGS, chat

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data/rag/corpus.jsonl"
INDEX = ROOT / "data/rag/index.pkl"
PROFILES = {
    "existing": (CORPUS, INDEX),
    "structure": (ROOT / "data/rag/structure/corpus.jsonl", ROOT / "data/rag/structure/index.pkl"),
    "vl": (ROOT / "data/rag/vl/corpus.jsonl", ROOT / "data/rag/vl/index.pkl"),
}
EMBED_MODEL = "bge-m3:567m"
MAX_CONTEXT_CHARS = 8000


def embeddings(keep_alive=0, device="gpu"):
    return OllamaEmbeddings(
        model=EMBED_MODEL, base_url=BASE_URL, num_thread=4,
        num_gpu=0 if device == "cpu" else -1,
        keep_alive=keep_alive, client_kwargs=CLIENT_KWARGS,
    )


def load_documents(corpus=CORPUS):
    chunks = [json.loads(line) for line in corpus.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not chunks:
        raise ValueError("Corpus is empty; run scripts/build-rag.ps1 first")
    return [Document(id=c["id"], page_content=c["content"],
                     metadata={k: v for k, v in c.items() if k != "content"}) for c in chunks]


def build_index(corpus=CORPUS, index=INDEX, device="gpu"):
    documents = load_documents(corpus)
    print(f"Embedding {len(documents)} chunks and building FAISS...", flush=True)
    with tracing_context(enabled=False):
        encoder = embeddings(keep_alive=60, device=device)
        store = FAISS.from_documents(documents[:4], encoder, normalize_L2=True)
        for start in range(4, len(documents), 4):
            store.add_documents(documents[start:start + 4])
            print(f"Embedded {min(start + 4, len(documents))}/{len(documents)} chunks", flush=True)
    # Byte serialization also works in Windows folders containing Korean characters.
    index.parent.mkdir(parents=True, exist_ok=True)
    temporary = index.with_suffix(".tmp")
    temporary.write_bytes(store.serialize_to_bytes())
    temporary.replace(index)
    print(f"Saved {len(documents)} Documents: {index}")


def load_retriever(top_k=3, index=INDEX, device="gpu"):
    if not index.exists():
        raise ValueError("Index missing: run scripts/build-rag.ps1 first")
    # Only load the index generated locally by build_index(); never downloaded pickle files.
    store = FAISS.deserialize_from_bytes(
        index.read_bytes(), embeddings(device=device), normalize_L2=True,
        allow_dangerous_deserialization=True,
    )
    return store.as_retriever(search_kwargs={"k": top_k})


def retrieve(question, top_k=3, index=INDEX, device="gpu"):
    if not question.strip() or top_k < 1:
        raise ValueError("Provide a question and a positive top-k")
    with tracing_context(enabled=False):
        return load_retriever(top_k, index, device).invoke(question)


def build_context(documents, max_chars=MAX_CONTEXT_CHARS):
    sections, skipped = [], []
    used = 0
    for doc in documents:
        section = f"[p.{doc.metadata['page']}]\n{doc.page_content}"
        if used + len(section) + 2 > max_chars:
            skipped.append(doc.id)
            continue
        sections.append(section)
        used += len(section) + 2
    if not sections:
        raise ValueError("No context fits the input limit; split the Markdown into smaller chunks.")
    return "\n\n".join(sections), skipped


def ask(question, search_only=False, top_k=3, index=INDEX, device="gpu"):
    documents = retrieve(question, top_k, index, device)
    print("\n검색 근거:")
    for doc in documents:
        print(f"  [p.{doc.metadata['page']}] {doc.id} / {doc.metadata['section']}")
    context, skipped = build_context(documents)
    if skipped:
        print("  입력 길이 제한으로 제외한 청크: " + ", ".join(skipped))
    sources = {}
    for doc in documents:
        if doc.id not in skipped:
            sources.setdefault(doc.metadata["source_pdf"], []).append(doc.metadata["page"])
    for source, pages in sources.items():
        print(f"\n원본 PDF: {ROOT / 'documents' / Path(source).name}")
        print("근거 페이지: " + " · ".join(map(str, dict.fromkeys(pages))))
    if not search_only:
        print(f"\n로컬 Qwen이 근거를 읽고 있습니다. ({device.upper()})\n답변:", flush=True)
        return chat(question, context, device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--search-only", action="store_true")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--profile", choices=list(PROFILES), default="existing")
    parser.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
    args = parser.parse_args()
    corpus, index = PROFILES[args.profile]
    try:
        if args.top_k < 1:
            raise ValueError("--top-k must be positive")
        if args.build:
            build_index(corpus, index, args.device)
        elif args.question:
            ask(args.question, args.search_only, args.top_k, index, args.device)
        else:
            load_retriever(args.top_k, index, args.device)
            print(f"B2B 매뉴얼 RAG: {args.profile}, top-k={args.top_k}, {args.device.upper()} 설정")
            print("질문마다 독립적으로 검색합니다. 종료: /exit")
            while True:
                question = input("\n질문> ").strip()
                if question == "/exit":
                    break
                if question:
                    ask(question, args.search_only, args.top_k, index, args.device)
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
