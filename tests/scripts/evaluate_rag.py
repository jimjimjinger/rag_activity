"""Compare saved RAG profiles on fixed PDF-reviewed questions; run serially on CPU."""
import argparse
from datetime import datetime
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import httpx
from langchain_community.vectorstores import FAISS
from langsmith import tracing_context
from llm_client import BASE_URL, CLIENT_KWARGS, MODEL, chat
from rag import EMBED_MODEL, PROFILES, build_context, embeddings


def evaluate():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", nargs="+", choices=list(PROFILES), default=list(PROFILES))
    parser.add_argument("--answers", nargs="+", default=[], help="Question IDs to answer; omitted means retrieval only")
    parser.add_argument("--answer-profiles", nargs="+", choices=list(PROFILES), default=["existing", "vl"])
    args = parser.parse_args()
    fixture = json.loads((ROOT / "tests/fixtures/rag_questions.json").read_text(encoding="utf-8"))
    if hashlib.sha256((ROOT / fixture["source_pdf"]).read_bytes()).hexdigest() != fixture["source_sha256"]:
        raise ValueError("Evaluation questions belong to another PDF")
    all_questions = fixture["questions"] + fixture["probes"]
    unknown = set(args.answers) - {item["id"] for item in all_questions}
    if unknown:
        parser.error("Unknown question IDs: " + ", ".join(sorted(unknown)))
    report = {"date": datetime.now().astimezone().isoformat(), "source_sha256": fixture["source_sha256"],
              "device": "cpu", "num_thread": 4, "embedding_model": EMBED_MODEL, "answer_model": MODEL,
              "top_k_answer": 3, "status": "retrieving", "questions": all_questions, "profiles": {},
              "limitations": ["Small fixed question set; page recall is not answer accuracy",
                              "existing includes PDF-reviewed corrections; structure and vl are uncorrected",
                              "Images are metadata references, not model inputs",
                              "Generated answers require manual source review; no automatic semantic grading"]}
    destination = ROOT / "tests/reports/rag_vl_checks.json"
    def save():
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(destination)
    with httpx.Client(base_url=BASE_URL, **CLIENT_KWARGS) as client:
        report["ollama_version"] = client.get("/api/version").json()["version"]
        report["models"] = client.get("/api/tags").json()["models"]
    report["packages"] = {name: version(name) for name in ("langchain-core", "langchain-ollama", "langchain-community", "faiss-cpu", "ollama")}
    save()
    encoder = embeddings(keep_alive=60, device="cpu")
    stores, retrieved = {}, {}
    for profile in args.profiles:
        corpus, index = PROFILES[profile]
        chunks = [json.loads(line) for line in corpus.read_text(encoding="utf-8").splitlines() if line.strip()]
        stores[profile] = FAISS.deserialize_from_bytes(index.read_bytes(), encoder, normalize_L2=True, allow_dangerous_deserialization=True)
        report["profiles"][profile] = {"chunks": len(chunks), "corpus_sha256": hashlib.sha256(corpus.read_bytes()).hexdigest(),
                                        "index_sha256": hashlib.sha256(index.read_bytes()).hexdigest(),
                                        "retrieval": [], "answers": []}
        retrieved[profile] = {}
    with tracing_context(enabled=False):
        for question in all_questions:
            started = time.perf_counter()
            vector = encoder.embed_query(question["question"])
            for profile, store in stores.items():
                docs = store.similarity_search_by_vector(vector, k=5)
                retrieved[profile][question["id"]] = docs
                pages = [doc.metadata["page"] for doc in docs]
                required = set(question["pages"])
                record = {"id": question["id"], "pages": pages,
                          "documents": [{"id": doc.id, "page": doc.metadata["page"], "section": doc.metadata["section"],
                                         "source_markdown": doc.metadata["source_markdown"]} for doc in docs]}
                if required:
                    record["recall"] = {str(k): len(required & set(pages[:k])) / len(required) for k in (1, 3, 5)}
                    record["all_required_pages"] = {str(k): required.issubset(pages[:k]) for k in (1, 3, 5)}
                report["profiles"][profile]["retrieval"].append(record)
            print(f"Retrieved {question['id']} ({time.perf_counter() - started:.2f}s)", flush=True)
            save()
    for profile, result in report["profiles"].items():
        scored = [item for item in result["retrieval"] if item["id"].startswith("Q")]
        result["metrics"] = {str(k): {"all_required_pages": sum(item["all_required_pages"][str(k)] for item in scored),
                                      "questions": len(scored), "mean_page_recall": sum(item["recall"][str(k)] for item in scored) / len(scored)} for k in (1, 3, 5)}
        print(profile, result["metrics"], flush=True)
    report["status"] = "answering" if args.answers else "complete"
    save()
    for question in all_questions:
        if question["id"] not in args.answers:
            continue
        for profile in args.profiles:
            if profile not in args.answer_profiles:
                continue
            docs = retrieved[profile][question["id"]][:3]
            context, skipped = build_context(docs)
            print(f"Answering {profile} {question['id']}", flush=True)
            started = time.perf_counter()
            answer = chat(question["question"], context, device="cpu")
            report["profiles"][profile]["answers"].append({"id": question["id"], "answer": answer,
                "seconds": round(time.perf_counter() - started, 2), "skipped_ids": skipped,
                "included_pages": list(dict.fromkeys(doc.metadata["page"] for doc in docs if doc.id not in skipped)),
                "context": context, "semantic_grade": "not_automatically_graded"})
            save()
    with httpx.Client(base_url=BASE_URL, **CLIENT_KWARGS) as client:
        report["loaded_models_after_run"] = client.get("/api/ps").json()["models"]
    report["status"] = "complete"
    save()
    print("Saved " + str(destination), flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    evaluate()
