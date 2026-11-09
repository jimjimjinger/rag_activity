"""Validate 60 explicit judgments and render the paired prompt comparison."""
import argparse
from collections import Counter
from html import escape
import json
from pathlib import Path
import statistics

from run_focused_prompt_comparison import app, parent, now, BASELINE, DEFAULT_RESULTS

LABELS = {"correct": "정답", "partial": "부분정답", "incorrect": "오답",
          "no_answer": "미답변", "error": "출력 오류"}


def metrics(rows):
    counts = Counter(r["verdict"] for r in rows)
    return {"n": len(rows), "counts": dict(counts),
            "correct_percent": round(100 * counts["correct"] / len(rows), 1)}


def render(destination):
    rows = app.legacy.read_json(destination / "results.json")
    judgments = app.legacy.read_json(destination / "judgments.json")
    protocol = app.legacy.read_json(destination / "protocol.json")
    refs_path = app.PM / "reports/qa30_20261102/references.json"
    if parent.file_sha(refs_path) != protocol["references_sha256"]:
        raise ValueError("The fixed evaluation rubric changed")
    refs = {r["case_id"]: r for r in app.legacy.read_json(refs_path)["cases"]}
    old = {(r["case_id"], r["keyword_percent"]): r for r in app.legacy.read_json(BASELINE / "results.json")}
    old_review = {(r["case_id"], r["keyword_percent"]): r
                  for r in app.legacy.read_json(BASELINE / "review.json")["cases"]}
    if len(rows) != 60 or set(judgments) != {r["variant"] + "/" + r["case_id"] for r in rows}:
        raise ValueError("All 60 outputs need explicit judgments")
    reviewed = []
    for row in rows:
        previous = old[(row["case_id"], row["keyword_percent"])]
        for field in ("question", "english_query", "sources", "excluded_sources",
                      "retrieved_chunk_ids", "context_expansion"):
            if row[field] != previous[field]:
                raise ValueError("Inputs differ: " + field)
        judgment = judgments[row["variant"] + "/" + row["case_id"]]
        if judgment["verdict"] not in LABELS or not judgment["note"]:
            raise ValueError("Invalid judgment")
        if row["status"] == "failed" and judgment["verdict"] != "error":
            raise ValueError("Execution errors must remain in the denominator")
        if row["status"] == "insufficient_evidence" and judgment["verdict"] != "no_answer":
            raise ValueError("Abstention is not a correct answer")
        prior_review = old_review[(row["case_id"], row["keyword_percent"])]
        reviewed.append({"case_id": row["case_id"], "keyword_percent": row["keyword_percent"],
                         "category": refs[row["case_id"]]["category"], "variant": row["variant"],
                         "retrieval_sufficiency": prior_review["retrieval_sufficiency"],
                         "retrieval_note": "Unchanged sources; inherited baseline evidence assessment",
                         **judgment})
    summary = {"date": "2026-11-08", "intervention": "System prompt only", "questions": 30,
               "new_answers": 60, "reviewer": "Codex assistant, frozen existing rubric; not independent evaluation",
               "weights": {}}
    for weight in (0, 50):
        current = [r for r in reviewed if r["keyword_percent"] == weight]
        previous = [old_review[(r["case_id"], weight)] for r in current]
        outputs = [r for r in rows if r["keyword_percent"] == weight]
        values = {"before": metrics(previous), "after": metrics(current), "categories": {}}
        values["difference_pp"] = round(100 * (values["after"]["counts"].get("correct", 0)
                                                - values["before"]["counts"].get("correct", 0)) / 30, 1)
        for category in ("text", "table", "figure"):
            values["categories"][category] = {
                "before": metrics([r for r in previous if r["category"] == category]),
                "after": metrics([r for r in current if r["category"] == category])}
        values["improved_to_correct"] = [r["case_id"] for r in current if r["verdict"] == "correct"
                                          and old_review[(r["case_id"], weight)]["verdict"] != "correct"]
        values["lost_correct"] = [r["case_id"] for r in current if r["verdict"] != "correct"
                                  and old_review[(r["case_id"], weight)]["verdict"] == "correct"]
        values["context_sufficiency"] = dict(Counter(r["retrieval_sufficiency"] for r in current))
        values["mean_answer_chars_before"] = round(statistics.mean(len(old[(r["case_id"], weight)].get("answer", "")) for r in current), 1)
        values["mean_answer_chars_after"] = round(statistics.mean(len(r.get("answer", "")) for r in outputs), 1)
        summary["weights"][str(weight)] = values
    app.write_json(destination / "review.json", {"reviewer": summary["reviewer"], "cases": reviewed})
    app.write_json(destination / "summary.json", summary)
    validation = app.legacy.read_json(destination / "execution-validation.json")
    table_rows = [f"| {w}% | {v['before']['correct_percent']:.1f}% | {v['after']['correct_percent']:.1f}% | {v['difference_pp']:+.1f}%p |"
                  for w, v in summary["weights"].items()]
    category_rows = [f"| {w}% | " + " | ".join(
        f"{v['categories'][c]['before']['correct_percent']:.1f}% → {v['categories'][c]['after']['correct_percent']:.1f}%"
        for c in ("text", "table", "figure")) + " |" for w, v in summary["weights"].items()]
    md = ["# PM209 질문 범위에 집중하는 프롬프트 비교", "", "2026-11-08", "",
          "부모 문맥을 연결한 이전 답변을 기준으로, 시스템 프롬프트만 수정해 같은 30문항에 키워드 0%·50%의 답변 60개를 생성했습니다.", "",
          "| 키워드 비중 | 이전 프롬프트 | 새 프롬프트 | 차이 |", "|---|---:|---:|---:|", *table_rows, "",
          "| 키워드 비중 | 본문: 이전 → 이후 | 표: 이전 → 이후 | 그림: 이전 → 이후 |", "|---|---:|---:|---:|", *category_rows, "",
          "전체는 비중별 30문항, 유형별로는 각 10문항의 완전 정답률입니다. 부분정답·미답변·출력 오류를 분모에 포함했습니다. %p는 반올림 전 비율로 계산했습니다.", "",
          "## 바꾼 내용", "",
          "- 질문한 사실·동작을 먼저 답하고, 질문하지 않은 사양·배경·예시를 덧붙이지 않도록 했습니다.",
          "- 간결하게 답하되 필수 조건과 요청한 절차는 생략하지 않도록 했습니다.",
          "- 표의 제목과 셀을 함께 읽고 수치·단위·범위·기술 용어의 의미를 유지하도록 했습니다.",
          "- 대안(or)과 순서(then), 필수와 권장을 구분하고 근거에서 확인되지 않는 아이콘·위치 관계를 만들지 않도록 했습니다.",
          "- 답변 전 근거와 누락 조건을 확인하되 내부 검토나 품질 플래그를 사용자에게 출력하지 않도록 했습니다.",
          "",
          f"실험 프롬프트 버전은 `{protocol['prompt_version']}`입니다. [이전 전문](prompt-before.txt)과 [수정 전문](prompt-after.txt)을 비교할 수 있습니다. 특정 평가 질문이나 정답 예시는 프롬프트에 넣지 않았습니다. **일반 CLI 실행은 기존 v1을 기본으로 유지**하며 `--answer-prompt focused`를 붙이면 이번 수정안을 선택합니다. `--answer-prompt baseline`은 기존 프롬프트를 명시적으로 선택합니다.", "",
          "## 고정한 조건", "",
          "- 기존 문서·질문 임베딩, Qdrant, 영어 번역, 검색 Top-5를 재사용했습니다.",
          "- 부모 문맥까지 확장한 근거의 내용·개수·순서·source_id를 60개 모두 이전과 동일하게 유지했습니다. 문맥을 다시 검색하거나 잘라내지 않았습니다.",
          "- 모델 Qwen3.5:4b, temperature=0, think=false, 문맥 8,192토큰, 출력 1,024토큰, 요청 제한 300초를 유지했습니다.",
          "- 기존 JSON 검증·근거 인용·재시도 처리도 동일합니다. 코드 비교에서 프롬프트 상수 외의 답변 구현이 유지됐는지 확인했습니다.",
          "- 이전 정답·검수 기준을 그대로 적용했습니다. 충분한 근거가 있는지의 판정도 입력이 동일하므로 이전 값을 유지했습니다.",
          "- 같은 문항으로 프롬프트를 개선하고 제가 검수한 결과입니다. 독립된 새 질문 평가, 여러 번 생성한 변동성 검증, 원본 이미지를 직접 입력하는 실험은 별도입니다.", "",
          "## 실행과 검증", "",
          f"- 60개 답변 생성·보존 확인: {validation['elapsed_seconds']/60:.1f}분. 출력 오류: {validation['generation_errors']}개.",
          f"- GPU 관측 최대: {validation['max_gpu_used_mib']}MiB, {validation['max_gpu_temperature_c']}°C.",
          f"- 임베딩 생성 0회, 인덱스 쓰기 0회. 원본·보강본·기존 실험 결과 등 {validation['protected_files']}개 보호 파일의 해시가 유지됐습니다.",
          "- 관련 기존 테스트 22개와 프롬프트 선택·문맥 예산 검사 2개를 통과했습니다. 실행·형식 검증과 의미 정확도는 별도로 평가했습니다. 정답률은 기존 핵심 답변과 중요한 부가 오류 기준이며 문장별 인용의 완전성을 인증하는 지표는 아닙니다.", "",
          "```powershell", r".\.runtime\venv-rag-lc\Scripts\python.exe -X utf8 PM209/scripts/run_focused_prompt_comparison.py",
          r".\.runtime\venv-rag-lc\Scripts\python.exe -X utf8 PM209/scripts/review_focused_prompt.py",
          r".\run-pm209-rag.cmd answer --manual <매뉴얼ID> --query <질문> --keyword-percent 0 --top-k 5 --answer-prompt focused", "```", "",
          "[문항별 이전·새 답변과 근거](report.html) · [집계](summary.json) · [판정 사유](review.json) · [고정 조건](protocol.json) · [최종 검증](final-validation.json)"]
    findings = destination / "findings.md"
    if findings.exists():
        md += ["", "## 문항별 변화와 판단", "", findings.read_text(encoding="utf-8").strip()]
    (destination / "REPORT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    cards = []
    for row, review in zip(rows, reviewed):
        key = (row["case_id"], row["keyword_percent"])
        previous = old[key]; prior_review = old_review[key]; ref = refs[row["case_id"]]
        sources = "".join("<details><summary>" + escape(s["source_id"] + " · " + s["chunk_type"])
                          + "</summary><pre>" + escape(s["text"]) + "</pre></details>" for s in row["sources"])
        cards.append(f'<article data-weight="{row["keyword_percent"]}" data-verdict="{review["verdict"]}"><h2>{row["case_id"]} · 키워드 {row["keyword_percent"]}%</h2>'
                     + "<p><b>질문</b> " + escape(row["question"]) + "</p><p><b>고정 기준</b> " + escape(ref["expected_answer_ko"])
                     + "</p><p><b>이전 · " + LABELS[prior_review["verdict"]] + "</b></p><pre>" + escape(previous.get("answer") or previous.get("error", ""))
                     + "</pre><p><b>새 프롬프트 · " + LABELS[review["verdict"]] + "</b></p><pre>" + escape(row.get("answer") or row.get("error", ""))
                     + "</pre><p><b>검수</b> " + escape(review["note"]) + "</p>" + sources + "</article>")
    table = '<table><tr><th>키워드</th><th>이전</th><th>새 프롬프트</th><th>차이</th></tr>'
    for w, v in summary["weights"].items():
        table += f"<tr><td>{w}%</td><td>{v['before']['correct_percent']:.1f}%</td><td>{v['after']['correct_percent']:.1f}%</td><td>{v['difference_pp']:+.1f}%p</td></tr>"
    html = ('<!doctype html><html lang="ko"><meta charset="utf-8"><title>PM209 프롬프트 비교</title>'
            '<style>body{max-width:1100px;margin:32px auto;font:16px/1.6 system-ui;background:#f4f6f9;color:#162638}article{padding:22px;margin:20px 0;background:white;border:1px solid #ccd4df;border-radius:10px}pre{white-space:pre-wrap;overflow-wrap:anywhere}table{border-collapse:collapse}td,th{padding:10px;border:1px solid #aaa}select{padding:8px}</style>'
            '<h1>질문 범위에 집중하는 프롬프트 · 30문항 비교</h1><p>두 조건 모두 부모 문맥을 사용합니다. 근거 내용·순서·모델·채점 기준을 고정하고 시스템 프롬프트만 바꿨습니다. 같은 개발 문항에 대한 Codex 검수입니다.</p>'
            + table + '</table><p><a href="REPORT.md">설계·결과</a> · <a href="prompt-before.txt">이전 프롬프트</a> · <a href="prompt-after.txt">새 프롬프트</a></p>'
            '<select id="weight"><option value="">모든 비중</option><option value="0">0%</option><option value="50">50%</option></select> '
            '<select id="verdict"><option value="">모든 판정</option>'
            + "".join(f'<option value="{k}">{v}</option>' for k, v in LABELS.items()) + '</select>' + "".join(cards)
            + '<script>function filter(){for(const a of document.querySelectorAll("article")){a.hidden=(weight.value&&a.dataset.weight!==weight.value)||(verdict.value&&a.dataset.verdict!==verdict.value)}}document.querySelectorAll("select").forEach(s=>s.addEventListener("change",filter));</script></html>')
    (destination / "report.html").write_text(html, encoding="utf-8")
    state = app.legacy.read_json(destination / "status.json")
    state.update(status="complete", stage="reviewed", reviewed=60, updated_at=now())
    app.write_json(destination / "status.json", state)
    print(json.dumps(summary["weights"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    render(parser.parse_args().results)
