"""Compare the focused prompt on the exact 60 frozen parent-context inputs."""
import argparse
import ast
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import lc_qdrant_rag as app
import parent_context as parent
from run_parent_context_comparison import now, protected_paths

BASELINE = app.PM / "reports/qa30_parent_context_20261108_v2"
DEFAULT_RESULTS = app.PM / "reports/qa30_focused_prompt_20261108"


def implementation_without_prompt(code):
    tree = ast.parse(code)
    names = {"SYSTEM_PROMPT", "SYSTEM_PROMPT_V1", "PROMPT_VERSION"}
    tree.body = [node for node in tree.body if not (
        isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names
                                            for t in node.targets))]
    return ast.dump(tree, include_attributes=False)


def prepare(destination):
    destination.mkdir(parents=True, exist_ok=True)
    old_protocol = app.legacy.read_json(BASELINE / "protocol.json")
    old_code = destination / "answering-before.py"
    if parent.file_sha(old_code) != old_protocol["answering_code_sha256"]:
        raise ValueError("Previous answer implementation snapshot does not match baseline")
    if implementation_without_prompt(old_code.read_text(encoding="utf-8")) != implementation_without_prompt(
            Path(app.answering.__file__).read_text(encoding="utf-8")):
        raise ValueError("Only prompt constants may change in this comparison")
    if (destination / "prompt-before.txt").read_text(encoding="utf-8") != app.answering.SYSTEM_PROMPT_V1:
        raise ValueError("Previous prompt differs from saved baseline")
    if app.answering.OPTIONS != old_protocol["answer_options"] or app.ANSWER_DIGEST != old_protocol["answer_digest"]:
        raise ValueError("Model or generation options changed")
    if parent.file_sha(parent.__file__) != old_protocol["parent_module_sha256"]:
        raise ValueError("Parent linking code changed")
    if parent.file_sha(parent.DEFAULT_STORE / "manifest.json") != old_protocol["parent_manifest_sha256"]:
        raise ValueError("Parent map changed")
    if app.legacy.read_json(BASELINE / "final-validation.json")["status"] != "complete":
        raise ValueError("Baseline has not passed validation")
    old_results = {(r["case_id"], r["keyword_percent"]): r
                   for r in app.legacy.read_json(BASELINE / "results.json")}
    plans = []
    for old in app.legacy.read_json(BASELINE / "frozen-input.json"):
        result = old_results[(old["case_id"], old["keyword_percent"])]
        if any(result.get(k) != v for k, v in old.items()):
            raise ValueError("Baseline answer and frozen input differ")
        bound = app.answering.prompt_upper_bound(app.answering.messages_for(
            old["question"], old["sources"], retry=True))
        if bound + app.answering.OPTIONS["num_predict"] > app.answering.OPTIONS["num_ctx"]:
            raise ValueError("New prompt does not fit unchanged context: " + old["case_id"])
        plans.append({**old, "baseline_variant": old["variant"],
                      "variant": f"focused_kw{old['keyword_percent']:03d}",
                      "new_prompt_token_upper_bound": bound})
    if len(plans) != 60 or len({(p["case_id"], p["keyword_percent"]) for p in plans}) != 60:
        raise ValueError("Expected 30 questions at 0% and 50%")
    protocol = {"date": "2026-11-08", "intervention": "System prompt only",
                "baseline": str(BASELINE), "new_answers": 60, "questions": 30,
                "keyword_percent": [0, 50], "seed_top_k": 5,
                "identical_fields": ["question", "english_query", "sources including order and IDs",
                                     "excluded_sources", "retrieved_chunk_ids", "context_expansion"],
                "prompt_version": app.answering.PROMPT_VERSION,
                "prompt_before_bytes": len(app.answering.SYSTEM_PROMPT_V1.encode()),
                "prompt_after_bytes": len(app.answering.SYSTEM_PROMPT.encode()),
                "prompt_before": app.answering.SYSTEM_PROMPT_V1,
                "prompt_after": app.answering.SYSTEM_PROMPT,
                "answer_model": app.answering.MODEL, "answer_digest": app.ANSWER_DIGEST,
                "answer_options": app.answering.OPTIONS, "think": False,
                "answering_sha256": parent.file_sha(app.answering.__file__),
                "runner_sha256": parent.file_sha(__file__),
                "baseline_inputs_sha256": parent.file_sha(BASELINE / "frozen-input.json"),
                "baseline_results_sha256": parent.file_sha(BASELINE / "results.json"),
                "references_sha256": old_protocol["references_sha256"],
                "qa_used_in_generation": False, "embedding_calls": 0, "index_writes": 0,
                "evaluation_scope": "Same development questions; historical paired baseline; one generation per input"}
    for name, value in (("protocol.json", protocol), ("frozen-input.json", plans)):
        path = destination / name
        if path.exists() and app.legacy.read_json(path) != value:
            raise ValueError("Frozen comparison changed; use a new result directory")
        app.write_json(path, value)
    (destination / "prompt-after.txt").write_text(app.answering.SYSTEM_PROMPT, encoding="utf-8")
    snapshot = destination / "preservation-before.json"
    if not snapshot.exists():
        paths = protected_paths() + [p for p in BASELINE.rglob("*") if p.is_file()]
        paths += [Path(parent.__file__), parent.DEFAULT_STORE / "manifest.json",
                  Path(app.__file__), old_code, destination / "prompt-before.txt"]
        app.write_json(snapshot, {str(p.resolve()): parent.file_sha(p) for p in paths})
    return plans


def run(destination, prepare_only=False):
    started = time.perf_counter()
    state = {"status": "running", "stage": "prepare", "pid": os.getpid(),
             "started_at": now(), "total": 60, "completed": 0}

    def update(**values):
        state.update(values, updated_at=now(), elapsed_seconds=time.perf_counter()-started)
        app.write_json(destination / "status.json", state)

    update()
    try:
        plans = prepare(destination)
        if prepare_only:
            update(status="prepared", stage="ready")
            print("Prepared 60 identical source contexts; only system prompt changed", flush=True)
            return
        chat = app.LocalChat()
        results = []; fresh = 0
        generation_started = time.perf_counter()
        for plan in plans:
            path = destination / plan["variant"] / (plan["case_id"] + ".json")
            if path.exists():
                result = app.legacy.read_json(path)
                if any(result.get(k) != v for k, v in plan.items()):
                    raise ValueError("Saved result input differs")
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
            elapsed = time.perf_counter()-generation_started
            update(stage="generation", completed=len(results), saved=len(results), fresh_answers=fresh,
                   failed=sum(r["status"] == "failed" for r in results),
                   case_id=plan["case_id"], keyword_percent=plan["keyword_percent"],
                   eta_seconds=(60-len(results))*elapsed/fresh if fresh else None)
            print(f"Answered {len(results)}/60 {plan['variant']} {plan['case_id']}: {result['status']}", flush=True)
        app.write_json(destination / "results.json", results)
        update(stage="preservation_validation")
        before = app.legacy.read_json(destination / "preservation-before.json")
        changed = [p for p, expected in before.items() if parent.file_sha(p) != expected]
        if changed:
            raise ValueError("Protected files changed: " + str(changed))
        observations = [r.get("gpu_observation") or {} for r in results]
        validation = {"status": "generated_pending_review", "answers": len(results),
                      "protected_files": len(before), "changed_files": [],
                      "identical_contexts_including_order": 60, "embedding_calls": 0, "index_writes": 0,
                      "generation_errors": sum(r["status"] == "failed" for r in results),
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
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    run(args.results, args.prepare_only)
