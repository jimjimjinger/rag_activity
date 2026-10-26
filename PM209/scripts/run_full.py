"""Run every PM209 figure with bounded checkpoints and a live status file.

Use the repository Python. To stop cleanly, create .runtime/pm209-full/STOP.
Only the canonical full-page JSONL copies contain generated descriptions.
"""

import argparse
import ctypes
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("describe_images", ROOT / "PM209/src/describe_images.py")
app = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(app)
SPLITS = ("train", "val", "test")
RUNTIME = ROOT / ".runtime/pm209-full"


def now():
    return datetime.now(timezone.utc).isoformat()


def inventory(dataset, output):
    counts = {}
    for split in SPLITS:
        pages = figures = saved = qa = 0
        for original, enriched in app.page_pairs(dataset / "data" / f"{split}.jsonl", output / f"{split}.jsonl"):
            pages += 1
            qa += len(original.get("qa_data", []))
            for region in enriched["bounding_boxes"]:
                if region["structure"] in app.FIGURE_TYPES:
                    figures += 1
                    saved += bool(region.get("description"))
        counts[split] = dict(pages=pages, figures=figures, saved=saved, qa=qa)
    return counts


def process_all(args, counts, refresh, checkpoint, start_pass, should_stop):
    """나머지 전체를 처리한 다음, 미완료 그림만 한 차례 더 처리한다."""
    for pass_number in (1, 2):
        start_pass(pass_number)
        for split in SPLITS:
            args.skip_region_ids = set()
            while counts[split]["saved"] + len(args.skip_region_ids) < counts[split]["figures"]:
                if should_stop():
                    raise KeyboardInterrupt("STOP file requested exit")
                before = counts[split]["saved"] + len(args.skip_region_ids)
                args.split = [split]
                app.run(args)
                counts = refresh()
                if counts[split]["saved"] + len(args.skip_region_ids) <= before:
                    raise RuntimeError(f"No progress saving or recording failed figures for {split}")
                checkpoint()
        if all(value["saved"] == value["figures"] for value in counts.values()):
            break
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=100)
    options = parser.parse_args()
    if options.batch_size < 1:
        parser.error("batch-size must be positive")
    RUNTIME.mkdir(parents=True, exist_ok=True)
    # OS lock is released on exit/crash, so a stale file cannot block a restart.
    lock_file = (RUNTIME / "runner.lock").open("a+b")
    if os.name == "nt":
        import msvcrt
        if lock_file.seek(0, 2) == 0:
            lock_file.write(b"0")
            lock_file.flush()
        lock_file.seek(0)
        try:
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            raise SystemExit("Another PM209 full runner is already active.")
    dataset = ROOT / "PM209/datasets/PM209"
    output = dataset / "data/vlm"
    status_path = RUNTIME / "status.json"
    stop_file = RUNTIME / "STOP"
    if stop_file.exists():
        raise SystemExit(f"Stop requested. Remove {stop_file} only when resuming intentionally.")
    started = time.monotonic()
    if status_path.exists():
        previous = json.loads(status_path.read_text(encoding="utf-8-sig"))
        archive = RUNTIME / f"status-before-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json"
        archive.write_text(json.dumps(previous, ensure_ascii=False, indent=2), encoding="utf-8")
    state = dict(status="starting", pid=os.getpid(), started_at=now(), model="qwen3.5:4b",
                 prompt_version=app.PROMPT_VERSION, retry_policy=app.RETRY_POLICY,
                 failed_attempts=0, deferred=0, pass_number=1,
                 dataset=str(dataset), output=str(output), batch_size=options.batch_size,
                 generated_this_run=0, completed_batches=0, completed=0, saved=0, total=0,
                 max_gpu_temperature_c=None, max_gpu_used_mib=None)
    mutex = threading.Lock()
    stop_monitor = threading.Event()
    pending = set()

    def publish():
        with mutex:
            state["updated_at"] = now()
            state["elapsed_seconds"] = round(time.monotonic() - started, 1)
            generated = state["generated_this_run"]
            state["eta_seconds"] = (round(state["elapsed_seconds"] / generated *
                                          max(0, state["total"] - state["completed"]))
                                    if generated >= 20 and state["status"] == "running" else None)
            snapshot = dict(state)
        temporary = status_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(status_path)

    def monitor():
        while not stop_monitor.is_set():
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used,temperature.gpu,utilization.gpu", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, check=True, timeout=8,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                memory, temperature, utilization = [int(v.strip()) for v in result.stdout.strip().splitlines()[0].split(",")]
                with mutex:
                    state["gpu"] = dict(observed_at=now(), used_mib=memory, temperature_c=temperature, utilization_percent=utilization)
                    state["max_gpu_temperature_c"] = max(temperature, state["max_gpu_temperature_c"] or 0)
                    state["max_gpu_used_mib"] = max(memory, state["max_gpu_used_mib"] or 0)
                with (RUNTIME / "gpu.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(state["gpu"]) + "\n")
            except Exception as error:
                with mutex:
                    state["monitor_error"] = str(error)
            try:
                publish()
            except Exception as error:
                print(f"Status write warning: {error}", flush=True)
            stop_monitor.wait(5)

    def update_inventory():
        counts = inventory(dataset, output)
        with mutex:
            state["splits"] = counts
            state["saved"] = sum(value["saved"] for value in counts.values())
            state["completed"] = state["saved"]
            state["total"] = sum(value["figures"] for value in counts.values())
            state["last_checkpoint_at"] = now()
        return counts

    def progress(event, split, region_id):
        with mutex:
            state.update(current_split=split, current_region_id=region_id, last_event=event, last_event_at=now())
            if event == "generated":
                state["generated_this_run"] += 1
                state["completed"] += 1
                pending.discard((split, region_id))
                state["deferred"] = len(pending)
        if stop_file.exists():
            raise KeyboardInterrupt("STOP file requested a clean checkpoint and exit")

    def record_attempt(failure):
        with mutex:
            state["failed_attempts"] += 1
            record = dict(at=now(), split=state.get("current_split"),
                          region_id=state.get("current_region_id"), **failure)
        with (RUNTIME / "attempt-errors.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        if stop_file.exists():
            raise KeyboardInterrupt("STOP file requested exit after attempt")

    def defer_figure(split, region_id, error):
        with mutex:
            pending.add((split, region_id))
            state["deferred"] = len(pending)
            record = dict(at=now(), split=split, region_id=region_id, pass_number=state["pass_number"],
                          error=str(error), attempts=error.attempts)
        with (RUNTIME / "figure-failures.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        print(f"DEFERRED [{split}] {region_id}: {error}", flush=True)

    def checkpoint():
        with mutex:
            state["completed_batches"] += 1
        print(f"CHECKPOINT {state['saved']}/{state['total']} deferred={state['deferred']}", flush=True)

    def start_pass(number):
        with mutex:
            state["pass_number"] = number
        print(f"PASS {number}/2", flush=True)

    # Prevent automatic idle sleep only for this process's lifetime; screen may turn off.
    if os.name == "nt":
        state["idle_sleep_prevention"] = bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
    publish()
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    exit_code = 0
    try:
        counts = update_inventory()
        with mutex:
            state["initial_saved"] = state["saved"]
            state["status"] = "running"
        args = SimpleNamespace(dataset=dataset, output=output, manual=None, region_ids=None,
                               types=app.FIGURE_TYPES, preview=False, all=False,
                               limit=options.batch_size, model="qwen3.5:4b",
                               url="http://127.0.0.1:11434", max_edge=1536, num_ctx=4096,
                               on_progress=progress, on_attempt_error=record_attempt,
                               on_figure_failure=defer_figure)
        counts = process_all(args, counts, update_inventory, checkpoint, start_pass, stop_file.exists)
        with mutex:
            state["status"] = "complete" if state["saved"] == state["total"] else "needs_review"
            state["unresolved"] = [dict(split=split, region_id=region_id) for split, region_id in sorted(pending)]
            state["finished_at"] = now()
            if state["status"] == "needs_review":
                exit_code = 2
    except BaseException as error:
        exit_code = 130 if isinstance(error, KeyboardInterrupt) else 1
        traceback.print_exc()
        try:
            update_inventory()
        except Exception as validation_error:
            state["validation_error"] = str(validation_error)
        with mutex:
            state["status"] = "stopped" if exit_code == 130 else "failed"
            state["error"] = f"{type(error).__name__}: {error}"
            state["finished_at"] = now()
    finally:
        stop_monitor.set()
        thread.join(timeout=15)
        state["idle_sleep_prevention"] = False
        publish()
        if os.name == "nt":
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        lock_file.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
