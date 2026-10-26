"""Qwen 모델 다운로드와 메타데이터 확인만 수행한다. 추론하지 않는다."""

import argparse
import json
from pathlib import Path

import httpx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="qwen3.5:4b")
    args = parser.parse_args()
    with httpx.Client(base_url=args.url, trust_env=False, timeout=1200) as client:
        response = client.get("/api/version")
        response.raise_for_status()
        version = response.json()["version"]
        print(f"Ollama {version}, model {args.model}", flush=True)
        previous = None
        final_status = None
        with client.stream("POST", "/api/pull", json={"model": args.model, "stream": True}) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                record = json.loads(line)
                if "error" in record:
                    raise RuntimeError(record["error"])
                percent = int(record.get("completed", 0) * 100 / record["total"]) if record.get("total") else None
                state = (record.get("status"), percent // 10 if percent is not None else None)
                if state != previous:
                    print(record.get("status"), f"{percent}%" if percent is not None else "", flush=True)
                    previous = state
                final_status = record.get("status")
        if final_status != "success":
            raise RuntimeError("Model download did not finish")
        response = client.post("/api/show", json={"model": args.model})
        response.raise_for_status()
        info = response.json()
        if "vision" not in info.get("capabilities", []):
            raise RuntimeError("Downloaded model does not advertise vision support")
        tags = client.get("/api/tags")
        tags.raise_for_status()
        model = next(m for m in tags.json()["models"] if m["name"] == args.model)
        running = client.get("/api/ps")
        running.raise_for_status()
        summary = {"server_url": args.url, "ollama_version": version,
                   "model": model, "capabilities": info.get("capabilities"),
                   "details": info.get("details"), "loaded_models": running.json().get("models", []),
                   "inference_executed": False}
        destination = Path(__file__).resolve().parents[2] / ".runtime/pm209_environment.json"
        destination.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        print("Environment prepared. No image description or model inference was executed.", flush=True)


if __name__ == "__main__":
    main()
