"""Local BGE-M3 retrieval followed by a grounded Qwen answer."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import urllib.error
import urllib.request

from llm_client import BASE_URL, chat

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data/rag/corpus.jsonl"
INDEX = ROOT / "data/rag/index.json"
EMBED_MODEL = "bge-m3:567m"
MAX_CONTEXT_CHARS = 8000


def api(path, payload=None):
    request = urllib.request.Request(BASE_URL + path, data=None if payload is None else json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=600) as response:
        result = json.load(response)
    if "error" in result:
        raise RuntimeError(result["error"])
    return result


def embed(texts, keep_alive="2m"):
    result = api("/api/embed", {"model": EMBED_MODEL, "input": texts, "truncate": False,
                               "keep_alive": keep_alive, "options": {"num_thread": 4}})
    vectors = result["embeddings"]
    if len(vectors) != len(texts) or any(not v for v in vectors):
        raise RuntimeError("Unexpected embedding response")
    return vectors


def build_index():
    corpus_bytes = CORPUS.read_bytes()
    chunks = [json.loads(line) for line in corpus_bytes.decode("utf-8").splitlines()]
    model = next(m for m in api("/api/tags")["models"] if m["name"] == EMBED_MODEL)
    cache = {}
    if INDEX.exists():
        old = json.loads(INDEX.read_text(encoding="utf-8"))
        if old.get("model_digest") == model["digest"]:
            cache = {c["content"]: v for c, v in zip(old["chunks"], old["vectors"])}
    missing = list(dict.fromkeys(c["content"] for c in chunks if c["content"] not in cache))
    for i in range(0, len(missing), 2):
        batch = missing[i:i+2]
        cache.update(zip(batch, embed(batch)))
        print(f"Embedded {min(i+2, len(missing))}/{len(missing)} changed chunks", flush=True)
    vectors = [cache[c["content"]] for c in chunks]
    print(f"Indexed {len(chunks)} chunks ({len(chunks)-len(missing)} embeddings reused)", flush=True)
    index = {"model": EMBED_MODEL, "model_digest": model["digest"],
             "scope_pages": sorted({c["page"] for c in chunks}),
             "corpus_sha256": hashlib.sha256(corpus_bytes).hexdigest(),
             "chunks": chunks, "vectors": vectors}
    tmp = INDEX.with_suffix(".tmp")
    tmp.write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
    tmp.replace(INDEX)
    print(f"Index saved: {INDEX}")


def cosine(a, b):
    if len(a) != len(b):
        raise ValueError("Embedding dimensions changed; rebuild the index")
    denominator = math.sqrt(sum(x*x for x in a) * sum(x*x for x in b))
    return sum(x*y for x, y in zip(a, b))/denominator if denominator else 0.0


def select_hits(chunks, vectors, query, top_k=3):
    scored = [dict(chunk, score=cosine(query, vector))
              for chunk, vector in zip(chunks, vectors)]
    return sorted(scored, key=lambda chunk: chunk["score"], reverse=True)[:top_k]

def retrieve(question, top_k=3):
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    if hashlib.sha256(CORPUS.read_bytes()).hexdigest() != index["corpus_sha256"]:
        raise ValueError("Corpus changed; rebuild the index first")
    source = Path(index["chunks"][0]["source_pdf"])
    if hashlib.sha256(source.read_bytes()).hexdigest() != index["chunks"][0]["source_sha256"]:
        raise ValueError("Source PDF changed; rebuild the corpus and index")
    model = next(m for m in api("/api/tags")["models"] if m["name"] == EMBED_MODEL)
    if model["digest"] != index["model_digest"]:
        raise ValueError("Embedding model changed; rebuild the index")
    query = embed([question], keep_alive=0)[0]
    return select_hits(index["chunks"], index["vectors"], query, top_k)


def build_context(hits, max_chars=MAX_CONTEXT_CHARS):
    """Pass retrieved Markdown pages directly, keeping their page citations."""
    sections, skipped = [], []
    used = 0
    for hit in hits:
        section = f"[p.{hit['page']}]\n{hit['content']}"
        if used + len(section) + 2 > max_chars:
            skipped.append(hit["id"])
            continue
        sections.append(section)
        used += len(section) + 2
    if not sections:
        raise ValueError("No context fits the input limit; split the Markdown into smaller chunks.")
    return "\n\n".join(sections), skipped


def ask(question, search_only=False):
    hits = retrieve(question)
    print("\n검색 근거 (유사도는 정답 확률이 아닙니다):")
    for hit in hits:
        print(f"  [p.{hit['page']}] {hit['id']} / {hit['score']:.3f}")
    context, skipped = build_context(hits)
    if skipped:
        print("  입력 길이 제한으로 일부 페이지를 제외했습니다.")
    if not search_only:
        print("\n로컬 Qwen이 근거를 읽고 있습니다. CPU에서는 시간이 걸릴 수 있습니다.\n답변:", flush=True)
        return chat(question, context, system_extra=(
            "이 자료는 사용 매뉴얼이며 실시간 업무 데이터가 아닙니다. "
            "사용법은 제공된 근거의 공통 절차로 설명하고, 없는 점수나 클릭 경로는 만들지 마세요. "
            "Markdown의 이미지 링크는 이미지 내용을 읽었다는 뜻이 아닙니다. "
            "답변은 3문장 또는 4단계 이내로 작성하세요."
        ))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--search-only", action="store_true")
    args = parser.parse_args()
    try:
        if args.build:
            build_index()
        elif args.question:
            ask(args.question, args.search_only)
        else:
            if not INDEX.exists():
                raise ValueError("Index missing: run scripts/build-rag.ps1 first")
            index = json.loads(INDEX.read_text(encoding="utf-8"))
            pages = sorted({c["page"] for c in index["chunks"]})
            scope = f"1~{pages[-1]}페이지" if pages == list(range(1, pages[-1]+1)) else "·".join(map(str, pages)) + "페이지"
            print(f"B2B 매뉴얼 RAG: 현재 검색 범위는 {scope} (총 {len(pages)}페이지)입니다.")
            print("질문마다 독립적으로 검색합니다. 종료: /exit")
            while True:
                question = input("\n질문> ").strip()
                if question == "/exit":
                    break
                if question:
                    ask(question)
    except (OSError, ValueError, RuntimeError, StopIteration, urllib.error.URLError) as exc:
        print(f"RAG 실행 실패: {exc}", file=sys.stderr)
        return 1
    except (KeyboardInterrupt, EOFError):
        return 0
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    raise SystemExit(main())
