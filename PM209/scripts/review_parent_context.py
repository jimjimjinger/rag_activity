"""Render parent-context comparison after all 60 answers receive human/agent review."""
import argparse
from collections import Counter
from html import escape
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import lc_qdrant_rag as app
from run_parent_context_comparison import BASELINE, DEFAULT_RESULTS, now


def metrics(rows):
    counts = Counter(row["verdict"] for row in rows)
    return {"n": len(rows), "counts": dict(counts),
            "correct_percent": round(100 * counts["correct"] / len(rows), 1)}


def render(destination):
    results = app.legacy.read_json(destination / "results.json")
    judgments = app.legacy.read_json(destination / "judgments.json")
    keys = {r["variant"] + "/" + r["case_id"] for r in results}
    if len(results) != 60 or keys != set(judgments):
        raise ValueError("Every one of the 60 answers must have a review")
    refs_path = app.PM / "reports/qa30_20261102/references.json"
    protocol = app.legacy.read_json(destination / "protocol.json")
    if app.legacy.sha(refs_path) != protocol["references_sha256"]:
        raise ValueError("Evaluation rubric changed")
    refs = {r["case_id"]: r for r in app.legacy.read_json(refs_path)["cases"]}
    old_reviews = {(r["case_id"], r["keyword_percent"]): r
                   for r in app.legacy.read_json(BASELINE / "review.json")["cases"] if r["top_k"] == 5}
    old_results = {(r["case_id"], r["keyword_percent"]): r
                   for r in app.legacy.read_json(BASELINE / "results.json") if r["top_k"] == 5}
    reviewed = []
    for row in results:
        key = row["variant"] + "/" + row["case_id"]
        judgment = judgments[key]
        if (judgment["verdict"] not in ("correct", "partial", "incorrect", "no_answer", "error")
                or judgment["retrieval_sufficiency"] not in ("sufficient", "partial", "insufficient")
                or not judgment["note"]):
            raise ValueError("Invalid review: " + key)
        if row["status"] == "failed" and judgment["verdict"] != "error":
            raise ValueError("Generation errors must stay in the denominator")
        if row["status"] == "insufficient_evidence" and judgment["verdict"] != "no_answer":
            raise ValueError("Abstention must be recorded")
        ref = refs[row["case_id"]]
        reviewed.append({"variant": row["variant"], "case_id": row["case_id"],
                         "keyword_percent": row["keyword_percent"], "category": ref["category"],
                         "gold_content_region_hit": any(set(s["content_region_ids"]) & set(ref["gold_region_ids"])
                                                        for s in row["sources"]), **judgment})
    summary = {"date": "2026-11-08", "reviewer": "Codex assistant; fixed existing rubric; not independent human review",
               "questions": 30, "new_answers": 60, "seed_top_k": 5, "weights": {}}
    table_rows = []
    for weight in (0, 50):
        current = [r for r in reviewed if r["keyword_percent"] == weight]
        previous = [old_reviews[(r["case_id"], weight)] for r in current]
        new_outputs = [r for r in results if r["keyword_percent"] == weight]
        values = {"before": metrics(previous), "after": metrics(current), "categories": {}}
        values["difference_pp"] = round(100 * (
            values["after"]["counts"].get("correct", 0) / values["after"]["n"]
            - values["before"]["counts"].get("correct", 0) / values["before"]["n"]), 1)
        for category in ("text", "table", "figure"):
            values["categories"][category] = {
                "before": metrics([r for r in previous if r["category"] == category]),
                "after": metrics([r for r in current if r["category"] == category])}
        values["mean_sources_before"] = statistics.mean(len(old_results[(r["case_id"], weight)]["sources"]) for r in current)
        values["mean_sources_after"] = statistics.mean(len(r["sources"]) for r in new_outputs)
        values["mean_chars_before"] = statistics.mean(sum(len(s["text"]) for s in old_results[(r["case_id"], weight)]["sources"]) for r in current)
        values["mean_chars_after"] = statistics.mean(sum(len(s["text"]) for s in r["sources"]) for r in new_outputs)
        values["context_sufficiency"] = dict(Counter(r["retrieval_sufficiency"] for r in current))
        values["improved_to_correct"] = [r["case_id"] for r in current if r["verdict"] == "correct"
                                          and old_reviews[(r["case_id"], weight)]["verdict"] != "correct"]
        values["lost_correct"] = [r["case_id"] for r in current if r["verdict"] != "correct"
                                   and old_reviews[(r["case_id"], weight)]["verdict"] == "correct"]
        summary["weights"][str(weight)] = values
        table_rows.append(f"| {weight}% | {values['before']['correct_percent']:.1f}% | {values['after']['correct_percent']:.1f}% | {values['difference_pp']:+.1f}%p | "
                          + " | ".join(f"{values['categories'][c]['after']['correct_percent']:.1f}%" for c in ("text", "table", "figure")) + " |")
    app.write_json(destination / "review.json", {"reviewer": summary["reviewer"], "cases": reviewed})
    app.write_json(destination / "summary.json", summary)
    validation = app.legacy.read_json(destination / "execution-validation.json")
    md = ["# PM209 부모 문맥 연결과 30문항 비교", "", "2026-11-08", "",
          "기존 500자 청크·임베딩·Qdrant 검색 순위를 재사용하고, 부모로 연결된 문맥을 답변 입력에 추가했습니다. 같은 30문항에 키워드 0%·50%를 적용해 새 답변 60개를 생성했습니다.", "",
          "| 키워드 비중 | 기존 전체 정답률 | 부모 연결 후 정답률 | 차이 | 본문 정답률 | 표 정답률 | 그림 정답률 |",
          "|---|---:|---:|---:|---:|---:|---:|", *table_rows, "",
          "전체는 각 30문항, 유형별로는 각 10문항입니다. 부분정답·미답변·출력 오류도 분모에 포함했습니다. %p 차이는 반올림 전 비율로 계산했습니다. Top-5는 최초 검색 청크 수이며 확장 후 근거 수와 길이는 늘어납니다.", "",
          "## 결과 해석", "",
          "| 키워드 비중 | 본문: 이전 → 이후 | 표: 이전 → 이후 | 그림: 이전 → 이후 |",
          "|---|---:|---:|---:|",
          *[f"| {w}% | " + " | ".join(
              f"{v['categories'][c]['before']['correct_percent']:.1f}% → {v['categories'][c]['after']['correct_percent']:.1f}%"
              for c in ("text", "table", "figure")) + " |" for w, v in summary["weights"].items()], "",
          "- 표 답변은 두 비중 모두 20.0%p 개선됐습니다. 0%에서는 헤드셋 연결 포트, 50%에서는 패키지 구성품처럼 검색 청크 주변에 있던 표 내용이 연결됐습니다.",
          "- 설치 절차(FIG07)는 두 비중 모두 후반 Next·Finish·재시작 권장까지 답하게 됐습니다. 주변 문맥으로 보완할 수 있는 누락에 효과가 있었습니다.",
          "- 문맥이 늘어도 그림의 상대 위치(FIG04), OK 버튼 아이콘(FIG03), 바구니 잠금 해제 방향(FIG10)은 충분히 복원하지 못했습니다. 이 정보가 설명·OCR에 없으면 부모 연결만으로 만들 수 없습니다.",
          "- 새 근거가 있어도 정답이 보장되지는 않았습니다. 전원 공급장치의 두 대안을 연속 절차로 섞었고(FIG05), 기존 정답이던 드릴 중심 맞추기(0%·FIG01)는 누락됐습니다. 50%·TEXT08은 추가 전류 설명에서 정격과 돌입 전류를 혼동해 부분정답으로 내려갔습니다.",
          "- 이번 표본에서는 부모 연결 + 키워드 0%가 가장 높았습니다. 기본 키워드 비중은 변경하지 않았으며 `--keyword-percent 0`으로 선택할 수 있습니다. 다음 비교는 새로운 질문으로 문맥량 제한·불필요한 부연 억제 효과를 확인하는 것이 적절합니다.", "",
          "## 적용한 연결", "",
          "- 같은 페이지의 소제목 아래 본문·표·그림을 연결하고, 들여쓴 표의 제목 범위를 보정했습니다.",
          "- 표·도면 내부의 레이블은 같은 원본 객체에 연결했습니다. 하나의 제목 아래 연속 번호가 확인되는 절차는 양쪽 열을 단계 순서로 연결했습니다.",
          "- 제목 연결이 불확실한 영역은 같은 원본 영역의 청크만 묶었습니다. 페이지를 넘는 절차 연결은 이번 규칙의 범위에 포함되지 않습니다.",
          "- 부모마다 기존 청크를 조회하고, 최초 검색 청크를 우선한 다음 부모별로 번갈아 근거를 추가합니다. 본문·표를 우선하고 중복 청크·동일 텍스트를 제거합니다.",
          "- 원래 프롬프트·8,192토큰 문맥·1,024토큰 출력 설정을 적용했습니다. 입력 예산과 최대 32개 근거 제한에 걸린 청크는 통째로 제외하고 기록했습니다.",
          "- LangChain 체인 안에 부모 조회 단계를 연결했습니다. Qdrant 가중 검색 결과를 받는 전용 확장기를 사용하며, 부모·자식 ID를 별도 페이지별 JSON 저장소에 보관합니다.", "",
          "`ParentDocumentRetriever`의 작은 청크 검색 후 부모 문맥 조회 원리를 적용했습니다. 해당 클래스를 그대로 교체 적용한 구현은 아니며 기존 Qdrant 하이브리드 검색을 유지합니다. 부모 연결은 209개 매뉴얼·10,018페이지·126,938개 청크 전체에 작성했습니다. 제목 접두어의 재할당은 답변 입력용 사본에만 반영했고 원래 임베딩 텍스트는 유지했습니다.", "",
          "## 비교 조건과 확인 범위", "",
          "- 2026-11-02의 영어 번역·Top-5 검색 결과·질문·채점 기준을 고정했습니다. 기존 답변 60개와 이번 새 답변 60개를 비교합니다.",
          "- 원본 QA는 부모 연결과 답변 생성에 사용하지 않았습니다. 검수 단계에서만 고정한 정답과 원본 검수 기록을 참고했습니다.",
          "- 그림은 설명 텍스트와 주변 OCR 문맥으로 전달했습니다. 이번 변화는 문맥 연결과 제목 보정이며 원본 이미지 입력 실험은 별도입니다.",
          "- 같은 30문항을 개발과 비교에 사용한 Codex 검수입니다. 새 질문에 대한 독립 평가와 반복 생성에 따른 변동 측정은 후속 검증 항목입니다.",
          f"- 문서·질문 임베딩 생성 {validation['embedding_calls']}회, 인덱스 쓰기 {validation['index_writes']}회. 보호 파일 {validation['preserved_files']}개의 SHA256이 유지됐습니다.",
          f"- 60개 답변 생성·보존 검사 실행 시간 {validation['elapsed_seconds']/60:.1f}분, 출력 오류 {validation['generation_errors']}개. GPU 관측 최대 {validation['max_gpu_used_mib']}MiB·{validation['max_gpu_temperature_c']}°C입니다. 부모 지도 작성은 약 3.1분이며 구현·검수 시간은 별도입니다.",
          "- 실제 LangChain → Qdrant → 부모 문맥 → 답변 CLI도 별도로 확인했습니다. 같은 영어 질문의 검색 순위가 기존과 일치했고 질문 임베딩 캐시를 재사용했습니다. `cli-smoke.json`은 이 연결 검사이며 60개 평가 집계에 포함하지 않습니다.", "",
          "## 실행", "", "저장소 루트에서 실행합니다.", "", "```powershell",
          r".\run-pm209-rag.cmd answer --manual <매뉴얼ID> --query <질문> --keyword-percent 50 --top-k 5",
          "# 기존 Top-5만 사용하려면 --no-parent-context 옵션을 추가합니다.",
          r".\.runtime\venv-rag-lc\Scripts\python.exe -X utf8 PM209/scripts/run_parent_context_comparison.py",
          "```", "",
          "부모 저장소는 `PM209/datasets/PM209/rag/parent_context_v2`입니다. [문항별 비교](report.html)에서 이전·새 답변과 전달한 근거를 볼 수 있습니다. `frozen-input.json`, `judgments.json`, `summary.json`, `execution-validation.json`에 입력·판정·집계·보존 검증을 기록했습니다. [최종 검증](final-validation.json)에는 실제 CLI 실행 후의 파일 보존, 전체 연결 지도 무결성·청크 보존, 22개 관련 테스트 통과와 60개 답변 검수 완료를 기록했습니다."]
    (destination / "REPORT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    cards = []
    labels = {"correct": "정답", "partial": "부분정답", "incorrect": "오답", "no_answer": "미답변", "error": "출력 오류"}
    for row, verdict in zip(results, reviewed):
        ref = refs[row["case_id"]]
        old = old_results[(row["case_id"], row["keyword_percent"])]
        old_review = old_reviews[(row["case_id"], row["keyword_percent"])]
        sources = "".join("<details><summary>" + escape(s["source_id"] + " · " + s["chunk_type"] + " · " + s["chunk_id"])
                          + "</summary><pre>" + escape(s["text"]) + "</pre></details>" for s in row["sources"])
        cards.append(f'<article data-weight="{row["keyword_percent"]}" data-verdict="{verdict["verdict"]}"><h2>{row["case_id"]} · 키워드 {row["keyword_percent"]}% · {labels[verdict["verdict"]]}</h2>'
                     + "<p><b>질문</b> " + escape(row["question"]) + "</p><p><b>기준 답변</b> " + escape(ref["expected_answer_ko"])
                     + "</p><p><b>이전 답변 · " + labels[old_review["verdict"]] + "</b></p><pre>" + escape(old.get("answer") or old.get("error", ""))
                     + "</pre><p><b>부모 연결 후 답변</b></p><pre>" + escape(row.get("answer") or row.get("error", ""))
                     + "</pre><p><b>검수</b> " + escape(verdict["note"]) + "</p><p>근거 " + str(len(old["sources"])) + " → " + str(len(row["sources"]))
                     + "개 · 추가 " + str(row["context_expansion"]["added_count"]) + "개 · 제외 " + str(len(row["excluded_sources"])) + "개</p>" + sources + "</article>")
    table = '<table><tr><th>키워드</th><th>기존 전체</th><th>연결 후 전체</th><th>본문</th><th>표</th><th>그림</th></tr>'
    for weight, value in summary["weights"].items():
        percents = [value["before"]["correct_percent"], value["after"]["correct_percent"]]
        percents += [value["categories"][c]["after"]["correct_percent"] for c in ("text", "table", "figure")]
        table += "<tr><td>" + weight + "%</td>" + "".join(f"<td>{p:.1f}%</td>" for p in percents) + "</tr>"
    html = ('<!doctype html><html lang="ko"><meta charset="utf-8"><title>PM209 부모 문맥 비교</title>'
            '<style>body{max-width:1100px;margin:32px auto;font:16px/1.6 system-ui;color:#162638;background:#f5f7fa}article{background:white;padding:22px;border:1px solid #ccd4df;border-radius:10px;margin:20px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere}table{border-collapse:collapse}td,th{padding:10px;border:1px solid #aaa}select{padding:8px}</style>'
            '<h1>부모 문맥 연결 · 30문항 비교</h1><p>같은 영어 질문과 검색 Top-5, 새 답변 60개. 기존 답변과 비교합니다. 전체 30문항·유형별 10문항의 완전 정답률이며 출력 오류도 포함합니다.</p>'
            + table + '</table><p>부모 확장 후 근거량이 늘었습니다. 원본 이미지는 답변 입력에 포함하지 않았습니다. 고정한 기존 기준에 대한 Codex 검수이며 독립적인 사람 평가는 아닙니다.</p>'
            '<p><a href="REPORT.md">설계·실행 기록</a> · <a href="summary.json">집계</a> · <a href="review.json">검수</a> · <a href="protocol.json">고정 조건</a></p>'
            '<select id="weight"><option value="">모든 가중치</option><option value="0">0%</option><option value="50">50%</option></select> '
            '<select id="verdict"><option value="">모든 판정</option>'
            + "".join(f'<option value="{k}">{v}</option>' for k, v in labels.items()) + '</select>' + "".join(cards)
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
