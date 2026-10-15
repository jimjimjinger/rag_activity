"""별도 자원 감시와 함께 8GB 환경의 실험을 실행하며 자동으로 재시도하지 않는다."""
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[2]
GPU_LIMIT_MIB = 7168
MIN_FREE_RAM_MIB = 2048
GPU_TEMPERATURE_LIMIT = 80


def stop_parser(process):
    try:
        parent = psutil.Process(process.pid)
        children = parent.children(recursive=True)
        for item in children + [parent]:
            try:
                item.terminate()
            except psutil.NoSuchProcess:
                pass
        _, alive = psutil.wait_procs(children + [parent], timeout=3)
        for item in alive:
            item.kill()
    except psutil.NoSuchProcess:
        pass


def main():
    arguments = sys.argv[1:]
    if "--output" not in arguments:
        raise ValueError("Provide a separate --output folder for this experiment.")
    output = Path(arguments[arguments.index("--output") + 1]).resolve()
    output.relative_to(ROOT)
    output.mkdir(parents=True, exist_ok=True)
    log_folder = ROOT / "tests/logs" / output.name
    log_folder.mkdir(parents=True, exist_ok=True)
    log = log_folder / ("resources_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".jsonl")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen([sys.executable, "-X", "utf8", "-u", "-B", str(Path(__file__).with_name("run_parser.py")), *arguments], env=env)
    reason = None
    print(f"Resource monitor: {log}; stop above {GPU_LIMIT_MIB} MiB GPU, below {MIN_FREE_RAM_MIB} MiB free RAM, or at {GPU_TEMPERATURE_LIMIT} C GPU", flush=True)
    with log.open("w", encoding="utf-8") as stream:
        try:
            while process.poll() is None:
                query = subprocess.run(
                    ["nvidia-smi", "-i", "0", "--query-gpu=memory.used,utilization.gpu,temperature.gpu", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
                if query.returncode:
                    raise RuntimeError("GPU monitor failed: " + query.stderr.strip())
                memory, utilization, temperature = [int(part.strip()) for part in query.stdout.strip().split(",")]
                free_ram = int(psutil.virtual_memory().available / (1024 * 1024))
                record = {"time": datetime.now().astimezone().isoformat(), "gpu_used_mib": memory,
                          "gpu_utilization_percent": utilization, "gpu_temperature_c": temperature,
                          "free_ram_mib": free_ram}
                if memory > GPU_LIMIT_MIB:
                    reason = f"GPU memory {memory} MiB exceeded {GPU_LIMIT_MIB} MiB"
                elif free_ram < MIN_FREE_RAM_MIB:
                    reason = f"Available RAM {free_ram} MiB fell below {MIN_FREE_RAM_MIB} MiB"
                elif temperature >= GPU_TEMPERATURE_LIMIT:
                    reason = f"GPU temperature {temperature} C reached {GPU_TEMPERATURE_LIMIT} C"
                stream.write(json.dumps(record) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                if reason:
                    stop_parser(process)
                    break
                time.sleep(1)
        except (KeyboardInterrupt, Exception) as exc:
            reason = f"Monitor interrupted: {type(exc).__name__}: {exc}"
            stop_parser(process)
        finally:
            if reason:
                path = output / "run_summary.json"
                if path.exists():
                    manifest = json.loads(path.read_text(encoding="utf-8"))
                    manifest["status"] = "stopped_by_resource_guard"
                    manifest["error"] = reason
                    manifest["resource_log"] = log.relative_to(ROOT).as_posix()
                    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
                print(reason, flush=True)
    return 2 if reason else process.wait()


if __name__ == "__main__":
    raise SystemExit(main())
