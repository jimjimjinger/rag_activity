"""Resume PM209 TableRecognitionPipelineV2 extraction with per-table durable results."""

import argparse
from collections import Counter, deque
import ctypes
import gc
import html
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "PM209/src"))
import paddle_tables as app

RUNTIME = ROOT / ".runtime/pm209-paddle-tables"


def write_review_index(targets, records):
    lines, flags = [], Counter()
    for target in targets:
        record = records.get(target["key"])
        if not record:
            continue
        quality = record["quality"]
        flags.update(quality["flags"])
        base = f'assets/{target["split"]}/{target["key"]}'
        title = html.escape(target["region"]["id"])
        labels = html.escape(", ".join(quality["flags"]) or "unreviewed")
        lines.append(f'<tr><td>{title}</td><td>{labels}</td><td><a href="{base}/image.png">원본</a> · '
                     f'<a href="{base}/table.html">HTML</a> · <a href="{base}/raw.json">OCR</a> · '
                     f'<a href="{base}/complete.json">기록</a></td></tr>')
    header = '''<!doctype html><html lang="ko"><meta charset="utf-8"><title>PM209 Paddle 표 검수 목록</title>
<style>body{font:15px/1.6 system-ui;margin:24px}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;padding:8px;text-align:left;overflow-wrap:anywhere}a{white-space:nowrap}input{font:inherit;width:70%;padding:8px}</style>
<h1>PM209 Paddle 표 검수 목록</h1><p>HTML 추출 결과와 원본 표 이미지입니다. 검수 표시는 우선순위를 정하는 휴리스틱이며 오류 확정이나 정확도 인증이 아닙니다. 아이콘·단위의 모든 누락을 자동으로 발견하지 못합니다. 샘플 3개 외에는 의미 정확도를 수동 검수하지 않았습니다.</p>
<input id="search" placeholder="표 ID 또는 units / icon / confidence 등으로 필터"><table><thead><tr><th>표 ID</th><th>검수 항목</th><th>파일</th></tr></thead><tbody>'''
    footer = '''</tbody></table><script>document.querySelector('#search').oninput=function(){const q=this.value.toLowerCase();for(const r of document.querySelectorAll('tbody tr'))r.hidden=!r.textContent.toLowerCase().includes(q)};</script></html>'''
    app.atomic_bytes(app.OUTPUT / "review_index.html", (header + "\n".join(lines) + footer).encode("utf-8"))
    app.atomic_json(app.OUTPUT / "review_summary.json", dict(completed=len(records), flags=dict(flags),
                    manually_reviewed_with_issues=sum(r["quality"]["manual_review"] != "unreviewed" for r in records.values()),
                    semantic_accuracy_verified=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--all", action="store_true")
    mode.add_argument("--prepare-only", action="store_true", help="Recover samples and validate output without model inference")
    args = parser.parse_args()
    RUNTIME.mkdir(parents=True, exist_ok=True)
    lock = (RUNTIME / "runner.lock").open("a+b")
    import msvcrt
    if lock.seek(0, 2) == 0:
        lock.write(b"0"); lock.flush()
    lock.seek(0)
    try:
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        raise SystemExit("Another Paddle table runner is active")
    if (RUNTIME / "STOP").exists():
        raise SystemExit("STOP is present. Resume only after explicit removal.")
    app.configure_environment()
    started = time.monotonic()
    state = dict(status="preparing", pid=os.getpid(), started_at=app.now(), last_event_at=app.now(),
                 total=2320, completed=0, saved=0, journal_pending=0, processed_this_run=0,
                 generated_this_run=0, failed_this_run=0, reused=0, eta_seconds=None,
                 output=str(app.OUTPUT), pipeline="TableRecognitionPipelineV2", semantic_accuracy_verified=False,
                 max_gpu_used_mib=0, max_gpu_temperature_c=0)
    mutex = threading.Lock()
    stopped = threading.Event()
    records, targets, stamps = {}, [], {}
    before = None
    recovery_ready = False
    times = deque(maxlen=100)
    saved_count = 0
    last_checkpoint = time.monotonic()

    def publish(event=False):
        with mutex:
            state.update(updated_at=app.now(), elapsed_seconds=round(time.monotonic() - started, 1),
                         completed=len(records), saved=saved_count, journal_pending=len(records) - saved_count,
                         unresolved=state["total"] - len(records))
            if event:
                state["last_event_at"] = state["updated_at"]
            app.atomic_json(RUNTIME / "status.json", state)

    def monitor():
        with (RUNTIME / "gpu.jsonl").open("a", encoding="utf-8") as stream:
            while not stopped.is_set():
                try:
                    result = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,temperature.gpu,utilization.gpu",
                                             "--format=csv,noheader,nounits"], capture_output=True, text=True,
                                            timeout=8, check=True, creationflags=subprocess.CREATE_NO_WINDOW)
                    used, temp, util = [int(v.strip()) for v in result.stdout.splitlines()[0].split(",")]
                    with mutex:
                        state.update(gpu_used_mib=used, gpu_temperature_c=temp, gpu_utilization_percent=util)
                        state["max_gpu_used_mib"] = max(state["max_gpu_used_mib"], used)
                        state["max_gpu_temperature_c"] = max(state["max_gpu_temperature_c"], temp)
                    stream.write(json.dumps(dict(at=app.now(), pid=os.getpid(), memory_mib=used, temperature_c=temp, utilization_percent=util)) + "\n")
                    stream.flush()
                    publish()
                except Exception as exc:
                    state["monitor_error"] = str(exc)
                stopped.wait(5)

    def checkpoint():
        nonlocal saved_count, last_checkpoint
        checkpoint_started = time.monotonic()
        state["phase"] = "checkpoint"
        publish(event=True)
        state["saved_by_split"] = app.checkpoint(records)
        saved_count = sum(state["saved_by_split"].values())
        last_checkpoint = time.monotonic()
        write_review_index(targets, records)
        if state["processed_this_run"]:
            state["checkpoint_seconds_total"] = state.get("checkpoint_seconds_total", 0) + time.monotonic() - checkpoint_started
            if len(times) >= 20:
                overhead = state["checkpoint_seconds_total"] / state["processed_this_run"]
                state["eta_seconds"] = round((sum(times) / len(times) + overhead) * state["unprocessed"])
        publish(event=True)
        print(f"CHECKPOINT saved={saved_count}/{state['total']}", flush=True)

    publish(event=True)
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    state["idle_sleep_prevention"] = bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
    exit_code = 0
    try:
        before = app.hashes()
        baseline_file = RUNTIME / "protected-before.json"
        if baseline_file.exists() and app.read_json(baseline_file) != before:
            raise ValueError("Protected dataset changed since the first run")
        if not baseline_file.exists():
            app.atomic_json(baseline_file, before)
        targets, counts = app.inventory()
        if len(targets) != 2320:
            raise ValueError(f"Expected 2320 tables; found {len(targets)}")
        app.atomic_json(RUNTIME / "input-inventory.json", counts)
        samples = {app.key_for(s["split"], s["page_id"], s["region"]["id"]): s
                   for s in app.read_json(app.SAMPLE / "samples.json")}
        image_hashes = {}
        for target in targets:
            filename = target["image_filename"]
            if filename not in image_hashes:
                image_hashes[filename] = app.digest(app.image_path(target))
            stamp = app.source_stamp(target, image_hashes[filename])
            stamps[target["key"]] = stamp
            record = app.load_record(target, stamp)
            if record is None and target["key"] in samples:
                sample = samples[target["key"]]
                if sample["region"] != target["region"] or sample["image_filename"] != filename:
                    raise ValueError("Sample source identity differs")
                content = (app.SAMPLE / f'{sample["kind"]}.html').read_text(encoding="utf-8")
                raw = app.read_json(app.SAMPLE / f'{sample["kind"]}.raw.json')
                if raw["table_res_list"][0]["pred_html"] != content:
                    raise ValueError("Sample HTML differs from its raw prediction")
                image, box = app.crop(target)
                if box != sample["crop_box"]:
                    raise ValueError("Sample crop differs")
                record = app.save_record(target, stamp, image, box, content, raw, None, "sample:" + sample["kind"])
            if record:
                records[target["key"]] = record
        state["reused"] = len(records)
        app.validate_existing(records)
        recovery_ready = True
        pending = [t for t in targets if t["key"] not in records]
        state.update(planned_this_run=len(pending), unprocessed=len(pending))
        checkpoint()
        if args.prepare_only:
            state["phase"] = "validation"
            publish(event=True)
            app.atomic_json(RUNTIME / "prepare-validation.json", app.validate_output(records, before))
            state.update(status="prepared", phase="finished", finished_at=app.now())
            return 0
        if not pending:
            validation = app.validate_output(records, before)
            validation["unresolved"] = []
            app.atomic_json(RUNTIME / "final-validation.json", validation)
            app.atomic_json(app.OUTPUT / "final-validation.json", validation)
            state.update(status="complete", phase="finished", eta_seconds=0, finished_at=app.now())
            return 0

        import numpy as np
        import paddle
        from paddleocr import TableRecognitionPipelineV2
        expected_versions = app.read_json(app.SAMPLE / "environment.json")["versions"]
        for name, expected in expected_versions.items():
            if importlib.metadata.version(name) != expected:
                raise ValueError(f"Environment differs from sample: {name}")
        if not paddle.is_compiled_with_cuda() or paddle.device.cuda.device_count() < 1:
            raise RuntimeError("Required GPU unavailable")
        paddle.set_device("gpu:0")
        state.update(status="loading_models", phase="initialization")
        publish(event=True)
        config = app.settings()
        pipeline = TableRecognitionPipelineV2(**config["init"])
        pipeline.export_paddlex_config_to_yaml(str(RUNTIME / "pipeline.yaml"))
        app.atomic_json(RUNTIME / "settings.json", config)
        state.update(status="running", phase="inference")
        publish(event=True)
        for target in pending:
            if (RUNTIME / "STOP").exists():
                raise KeyboardInterrupt("STOP requested")
            state.update(phase="inference", current_table_id=target["region"]["id"], current_split=target["split"])
            publish(event=True)
            tick = time.monotonic()
            results = raw = pixels = image = None
            failed = False
            try:
                image, box = app.crop(target)
                pixels = np.asarray(image)[:, :, ::-1].copy()
                results = pipeline.predict(pixels, **config["predict"])
                paddle.device.synchronize()
                if len(results) != 1 or len(results[0]["table_res_list"]) != 1:
                    raise ValueError("Pipeline did not return exactly one Table")
                raw = results[0].json["res"]
                content = raw["table_res_list"][0]["pred_html"]
                record = app.save_record(target, stamps[target["key"]], image, box, content, raw,
                                         time.monotonic() - tick, "paddle_full")
                records[target["key"]] = record
                state["generated_this_run"] += 1
                print(f"OK {len(records)}/{len(targets)} {target['region']['id']} {record['seconds']:.2f}s", flush=True)
            except Exception as exc:
                if isinstance(exc, OSError) and (isinstance(exc, PermissionError) or exc.errno in (13, 28)):
                    raise  # Storage permissions/full disk need intervention.
                failure = dict(at=app.now(), **target, error_type=type(exc).__name__, error=str(exc),
                               traceback=traceback.format_exc())
                folder = app.asset_dir(target)
                if image is not None:
                    folder.mkdir(parents=True, exist_ok=True)
                    image.save(folder / "image.png")
                if raw is not None:
                    app.atomic_json(folder / "failed.raw.json", raw)
                app.atomic_json(folder / "failure.json", failure)
                with (RUNTIME / "table-failures.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(failure, ensure_ascii=False) + "\n")
                    stream.flush(); os.fsync(stream.fileno())
                state["failed_this_run"] += 1
                failed = True
                print(f"FAIL {target['region']['id']}: {exc}", flush=True)
                if any(v in str(exc).lower() for v in ("illegal memory access", "driver shutting down", "device-side assert")):
                    raise RuntimeError("GPU context failed; resume from committed records after intervention") from exc
            finally:
                results = raw = pixels = image = None
            if failed:
                gc.collect()
                paddle.device.cuda.empty_cache()
            state["processed_this_run"] += 1
            state["unprocessed"] = len(pending) - state["processed_this_run"]
            times.append(time.monotonic() - tick)
            overhead = state.get("checkpoint_seconds_total", 0) / max(1, state["processed_this_run"])
            state["eta_seconds"] = round((sum(times) / len(times) + overhead) * state["unprocessed"]) if len(times) >= 20 else None
            publish(event=True)
            if state["processed_this_run"] % 100 == 0 or time.monotonic() - last_checkpoint >= 300:
                checkpoint()
                gc.collect()
                paddle.device.cuda.empty_cache()
        checkpoint()
        state.update(status="validating", phase="validation", eta_seconds=0)
        publish(event=True)
        validation = app.validate_output(records, before)
        validation["unresolved"] = [dict(split=t["split"], page_id=t["page_id"], table_id=t["region"]["id"])
                                    for t in targets if t["key"] not in records]
        app.atomic_json(RUNTIME / "final-validation.json", validation)
        app.atomic_json(app.OUTPUT / "final-validation.json", validation)
        state.update(status="complete" if len(records) == len(targets) else "needs_review", phase="finished",
                     finished_at=app.now(), current_table_id=None)
        exit_code = 0 if len(records) == len(targets) else 2
    except BaseException as exc:
        traceback.print_exc()
        exit_code = 130 if isinstance(exc, KeyboardInterrupt) else 1
        state.update(status="stopped" if exit_code == 130 else "failed", error=f"{type(exc).__name__}: {exc}", finished_at=app.now())
        if recovery_ready:
            try:
                checkpoint()
            except Exception as checkpoint_error:
                state["checkpoint_error"] = str(checkpoint_error)
        if before is not None:
            state["protected_files_unchanged"] = app.hashes() == before
    finally:
        stopped.set()
        thread.join(timeout=12)
        state["idle_sleep_prevention"] = False
        publish(event=True)
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        lock.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
