"""Generate 60 answers with parent context, using frozen 30-question rankings.

The historical Top-5 answers are the baseline. Gold answers are read only by the
separate review step. This runner makes no embedding or indexing requests.
"""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import lc_qdrant_rag as app
import parent_context as parent

BASELINE = app.PM / "reports/qa30_lc_qdrant_20261102"
DEFAULT_RESULTS = app.PM / "reports/qa30_parent_context_20261108_v2"


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def protected_paths():
    paths = [app.SOURCE / "embedding-cache.sqlite3", app.SOURCE / "chunks.jsonl",
             app.OUTPUT / "qdrant/collection/pm209_hybrid_v1/storage.sqlite",
             app.OUTPUT / "qdrant/meta.json", app.OUTPUT / "index-identity.json"]
    data = app.PM / "datasets/PM209/data"
    for folder in (data, data / "vlm", app.legacy.DEFAULT_INPUT):
        paths.extend(folder / (split + ".jsonl") for split in ("train", "val", "test"))
    paths.extend(p for p in BASELINE.rglob("*") if p.is_file())
    paths.extend([app.PM / "reports/qa30_20261102/questions.json",
                  app.PM / "reports/qa30_20261102/references.json",
                  app.PM / "reports/qa30_hybrid_20261102/translations.json"])
    return paths


def prepare(destination, store_directory):
    destination.mkdir(parents=True, exist_ok=True)
    store = parent.ParentContextStore(store_directory)
    old_protocol = app.legacy.read_json(BASELINE / "protocol.json")
    if app.legacy.sha(Path(app.answering.__file__)) != old_protocol["answering_code_sha256"]:
        raise ValueError("Historical answer prompt/validator changed")
    retrieval_path = BASELINE / "retrieval.json"
    expected = app.legacy.read_json(BASELINE / "summary.json")["files"]["retrieval.json"]
    if parent.file_sha(retrieval_path) != expected:
        raise ValueError("Historical candidate rankings changed")
    protocol = {"date": "2026-11-08", "questions": 30, "new_answers": 60,
                "keyword_percent": [0, 50], "seed_top_k": 5,
                "retrieval": "Exact saved Qdrant rankings and English translations from 2026-11-02",
                "baseline": "60 historical Top-5 responses, reviewed using the frozen original rubric",
                "intervention": "Parent-linked context expansion and corrected section labels",
                "top_k_note": "Top-5 refers to search seeds; expanded source count and input volume are recorded",
                "parent_map": str(store_directory.resolve()),
                "parent_manifest_sha256": parent.file_sha(store_directory / "manifest.json"),
                "parent_module_sha256": parent.file_sha(parent.__file__),
                "answering_code_sha256": old_protocol["answering_code_sha256"],
                "answer_model": app.answering.MODEL, "answer_digest": app.ANSWER_DIGEST,
                "answer_options": app.answering.OPTIONS, "think": False,
                "max_sources": parent.MAX_SOURCES,
                "qa_used_for_mapping_or_generation": False,
                "document_embeddings_generated": 0, "query_embeddings_generated": 0,
                "retrieval_sha256": expected,
                "references_sha256": old_protocol["references_sha256"],
                "runner_sha256": parent.file_sha(__file__)}
    protocol_path = destination / "protocol.json"
    if protocol_path.exists() and app.legacy.read_json(protocol_path) != protocol:
        raise ValueError("Comparison configuration changed; use a new result directory")
    plans = []
    for row in app.legacy.read_json(retrieval_path):
        if row["keyword_percent"] not in (0, 50):
            continue
        hits = row["hits"][:5]
        sources, excluded, expansion = store.prepare_context(row["question"], hits, row["manual_id"])
        plans.append({k: row[k] for k in ("case_id", "manual_id", "question", "english_query", "keyword_percent")}
                     | {"variant": f"parent_kw{row['keyword_percent']:03d}", "top_k": 5,
                        "retrieved_chunk_ids": [h["chunk_id"] for h in hits],
                        "sources": sources, "excluded_sources": excluded, "context_expansion": expansion})
    if len(plans) != 60 or len({(p["case_id"], p["keyword_percent"]) for p in plans}) != 60:
        raise ValueError("Expected exactly 30 cases at each of two weights")
    app.write_json(protocol_path, protocol)
    app.write_json(destination / "frozen-input.json", plans)
    before_path = destination / "preservation-before.json"
    if not before_path.exists():
        app.write_json(before_path, {str(p.resolve()): parent.file_sha(p) for p in protected_paths()})
    baseline = {(r["case_id"], r["keyword_percent"]): r for r in app.legacy.read_json(BASELINE / "results.json")
                if r["top_k"] == 5 and r["keyword_percent"] in (0, 50)}
    for plan in plans:
        old = baseline[(plan["case_id"], plan["keyword_percent"])]
        if (plan["retrieved_chunk_ids"] != old["retrieved_chunk_ids"]
                or plan["english_query"] != old["english_query"]):
            raise ValueError("Search seeds/translation changed")
    return plans


def run(destination, store_directory, prepare_only=False):
    started = time.perf_counter()
    state = {"status": "running", "stage": "prepare", "pid": os.getpid(),
             "started_at": now(), "total": 60, "completed": 0}

    def update(**values):
        state.update(values, updated_at=now(), elapsed_seconds=time.perf_counter()-started)
        app.write_json(destination / "status.json", state)

    update()
    try:
        plans = prepare(destination, store_directory)
        if prepare_only:
            update(status="prepared", stage="ready", completed=0)
            print("Prepared 60 fixed inputs", flush=True)
            return
        chat = app.LocalChat()
        results = []
        fresh = 0
        generation_started = time.perf_counter()
        for plan in plans:
            path = destination / plan["variant"] / (plan["case_id"] + ".json")
            if path.exists():
                result = app.legacy.read_json(path)
                if any(result.get(k) != v for k, v in plan.items()):
                    raise ValueError("Resume input changed")
            else:
                begin = time.perf_counter()
                result = {**plan, "created_at": now(), "status": "failed"}
                try:
                    value, generation = chat.generate(plan["question"], plan["sources"])
                    citations, figures = app.answering.resolve_citations(value, plan["sources"])
                    result.update(value, generation=generation, citations=citations, cited_figures=figures,
                                  status="insufficient_evidence" if value["insufficient_evidence"] else "answered")
                except Exception as error:
                    result["error"] = str(error)
                    if isinstance(error, app.answering.AnswerGenerationError):
                        result["failed_attempts"] = error.attempts
                result.update(elapsed_seconds=time.perf_counter()-begin,
                              gpu_observation=app.legacy.gpu_snapshot())
                app.write_json(path, result)
                fresh += 1
            results.append(result)
            elapsed = time.perf_counter() - generation_started
            update(stage="generation", completed=len(results), saved=len(results), fresh_answers=fresh,
                   case_id=plan["case_id"], keyword_percent=plan["keyword_percent"],
                   failed=sum(r["status"] == "failed" for r in results),
                   eta_seconds=(60-len(results))*elapsed/fresh if fresh else None)
            print(f"Answered {len(results)}/60 {plan['variant']} {plan['case_id']}: {result['status']}", flush=True)
        app.write_json(destination / "results.json", results)
        update(stage="preservation_validation")
        before = app.legacy.read_json(destination / "preservation-before.json")
        changed = [path for path, expected in before.items() if parent.file_sha(path) != expected]
        if changed:
            raise ValueError("Protected file changed: " + str(changed))
        observations = [r.get("gpu_observation") or {} for r in results]
        validation = {"status": "generated_pending_review", "results": len(results),
                      "embedding_calls": 0, "index_writes": 0, "preserved_files": len(before),
                      "changed_files": [], "generation_errors": sum(r["status"] == "failed" for r in results),
                      "elapsed_seconds": time.perf_counter()-started,
                      "max_gpu_used_mib": max(o.get("used_mib", 0) for o in observations),
                      "max_gpu_temperature_c": max(o.get("temperature_c", 0) for o in observations)}
        app.write_json(destination / "execution-validation.json", validation)
        update(status="generated_pending_review", stage="semantic_review", eta_seconds=0)
        print(json.dumps(validation), flush=True)
    except BaseException as error:
        update(status="failed", error=str(error))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--parent-context", type=Path, default=parent.DEFAULT_STORE)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    run(args.results, args.parent_context, args.prepare_only)
