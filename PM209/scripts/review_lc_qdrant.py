"""Prepare exact-match review groups, then render only after every group is graded.

No model calls. New semantic judgments must be supplied in judgments.json.
"""
import argparse
from collections import Counter,defaultdict
import hashlib
from html import escape
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import lc_qdrant_rag as app
from run_lc_qdrant_comparison import DEST,WEIGHTS,now


def fingerprint(row):
    value={'case_id':row['case_id'],'messages':app.answering.messages_for(row['question'],row['sources']),
           'status':row['status'],'output':{k:row.get(k) for k in ('answer','source_ids','insufficient_evidence','error')}}
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def prepare():
    old_matches=defaultdict(list)
    for top in (1,5):
        folder=app.PM/'reports'/('qa30_top1_20261102' if top==1 else 'qa30_weights_20261102')
        reviews=app.legacy.read_json(folder/'review.json')['variants']
        for weight in WEIGHTS:
            source=folder/f'keyword_{weight:03d}'
            if top==5 and weight in (0,50):
                source=app.PM/'reports/qa30_hybrid_20261102'/('B_english_dense' if weight==0 else 'C_english_hybrid')
            by_case={x['case_id']:x for x in reviews[str(weight)]}
            for result in app.legacy.read_json(source/'results.json'):
                old_matches[fingerprint(result)].append({'source':str(source/'results.json'),**by_case[result['case_id']]})
    refs={c['case_id']:c for c in app.legacy.read_json(app.PM/'reports/qa30_20261102/references.json')['cases']}
    groups={};numbers=Counter()
    for r in app.legacy.read_json(DEST/'results.json'):
        key=fingerprint(r)
        if key not in groups:
            numbers[r['case_id']]+=1
            reference=refs[r['case_id']]
            group={'group_id':f"{r['case_id']}-{numbers[r['case_id']]:03d}",'fingerprint':key,
                'case_id':r['case_id'],'question':r['question'],'expected':reference['expected_answer_ko'],
                'reference_notes':reference['reference_notes'],'reference_crop':reference['reference_crop'],
                'status':r['status'],'answer':r.get('answer'),'source_ids':r.get('source_ids',[]),'error':r.get('error'),
                'sources':r['sources'],'members':[],'prior_matches':old_matches.get(key,[])}
            groups[key]=group
        groups[key]['members'].append({'variant':r['variant'],'case_id':r['case_id']})
    values=list(groups.values())
    app.write_json(DEST/'review-groups.json',values)
    judgments={}
    for g in values:
        choices={(x['verdict'],x['retrieval_sufficiency']) for x in g['prior_matches']}
        if len(choices)==1:
            previous=g['prior_matches'][0]
            judgments[g['group_id']]={k:previous[k] for k in ('verdict','retrieval_sufficiency','note')}
            judgments[g['group_id']]['origin']='Exact same case, model input and output as prior reviewed record'
    target=DEST/'judgments.json'
    if target.exists():
        existing=app.legacy.read_json(target)
        if not set(existing)<=set(g['group_id'] for g in values):raise ValueError('Judgment group changed')
        judgments.update(existing)
    app.write_json(target,judgments)
    print(json.dumps({'groups':len(values),'exact_previous_matches':len(judgments),'pending':[g['group_id'] for g in values if g['group_id'] not in judgments]}))


def render():
    groups=app.legacy.read_json(DEST/'review-groups.json')
    judgments=app.legacy.read_json(DEST/'judgments.json')
    if set(judgments)!=set(g['group_id'] for g in groups):raise ValueError('Some groups have not been reviewed')
    lookup={}
    for group in groups:
        judgment=judgments[group['group_id']]
        if judgment['verdict'] not in ('correct','partial','incorrect','no_answer','error') or judgment['retrieval_sufficiency'] not in ('sufficient','partial','insufficient') or not judgment['note']:
            raise ValueError('Invalid judgment '+group['group_id'])
        if group['status']=='failed' and judgment['verdict']!='error':raise ValueError('Errors cannot be omitted')
        if group['status']=='insufficient_evidence' and judgment['verdict']!='no_answer':raise ValueError('Abstention mislabeled')
        for member in group['members']:
            lookup[(member['variant'],member['case_id'])]={'group_id':group['group_id'],**judgment}
    results=app.legacy.read_json(DEST/'results.json')
    if len(results)!=300 or len(lookup)!=300:raise ValueError('Expected all 300 conditions')
    protocol=app.legacy.read_json(DEST/'protocol.json')
    if app.legacy.sha(app.PM/'reports/qa30_20261102/references.json')!=protocol['references_sha256']:
        raise ValueError('Original rubric changed')
    refs={c['case_id']:c for c in app.legacy.read_json(app.PM/'reports/qa30_20261102/references.json')['cases']}
    reviewed=[]
    for row in results:
        judgment=lookup[(row['variant'],row['case_id'])]
        hit=any(set(s.get('content_region_ids',[]))&set(refs[row['case_id']]['gold_region_ids']) for s in row['sources'])
        reviewed.append({'variant':row['variant'],'case_id':row['case_id'],'category':refs[row['case_id']]['category'],
                         'keyword_percent':row['keyword_percent'],'top_k':row['top_k'],'gold_content_region_hit':hit,**judgment})
    summary={'date':now(),'questions':30,'answers':300,'review_groups':len(groups),
             'reviewer':'Codex assistant; exact prior review reuse is recorded per group; not independent human grading',
             'weights':{},'files':{name:app.legacy.sha(DEST/name) for name in ('protocol.json','results.json','judgments.json','retrieval.json')}}
    for w in WEIGHTS:
        summary['weights'][str(w)]={}
        for k in (1,5):
            rows=[r for r in reviewed if r['keyword_percent']==w and r['top_k']==k]
            counts=Counter(r['verdict'] for r in rows)
            categories={c:dict(Counter(r['verdict'] for r in rows if r['category']==c)) for c in ('text','table','figure')}
            summary['weights'][str(w)][f'top{k}']={'counts':dict(counts),'correct_percent':round(counts['correct']/30*100,1),
                'by_category':categories,'context_sufficiency':dict(Counter(r['retrieval_sufficiency'] for r in rows)),
                'gold_content_region_hits':sum(r['gold_content_region_hit'] for r in rows)}
    app.write_json(DEST/'review.json',{'reviewer':summary['reviewer'],'cases':reviewed})
    app.write_json(DEST/'summary.json',summary)
    labels={'correct':'정답','partial':'부분정답','incorrect':'오답','no_answer':'미답변','error':'형식·실행 오류'}
    table=['<table><tr><th>키워드 비중</th><th>전체 Top-1 /30</th><th>전체 Top-5 /30</th><th>텍스트 Top-1 /10</th><th>텍스트 Top-5 /10</th></tr>']
    for w in WEIGHTS:
        v=summary['weights'][str(w)]
        values=[v['top1']['counts'].get('correct',0),v['top5']['counts'].get('correct',0),v['top1']['by_category']['text'].get('correct',0),v['top5']['by_category']['text'].get('correct',0)]
        table.append('<tr><td>'+str(w)+'%</td>'+''.join('<td>'+str(n)+'</td>' for n in values)+'</tr>')
    table.append('</table>')
    cards=[]
    for r,v in zip(results,reviewed):
        ref=refs[r['case_id']]
        sources=''.join('<details><summary>'+escape(s['source_id']+' · '+s['chunk_type']+' · '+s['chunk_id'])+'</summary><pre>'+escape(s['text'])+'</pre></details>' for s in r['sources'])
        cards.append(f'<article data-variant="{r["variant"]}" data-verdict="{v["verdict"]}"><h3>{r["variant"]} · {r["case_id"]} · {labels[v["verdict"]]}</h3>'+
            '<p><b>질문</b> '+escape(r['question'])+'</p><p><b>기준 답변</b> '+escape(ref['expected_answer_ko'])+'</p><p><b>실제 답변</b> '+escape(r.get('answer') or r.get('error',''))+'</p>'+
            '<p><b>판정</b> '+escape(v['note'])+' · 검색 근거 '+escape(v['retrieval_sufficiency'])+'</p><p>인용: '+escape(', '.join(r.get('source_ids',[])))+'</p>'+sources+'</article>')
    options=''.join(f'<option value="top{k}_kw{w:03d}">Top-{k} / 키워드 {w}%</option>' for k in (1,5) for w in WEIGHTS)
    html=('<!doctype html><html lang="ko"><meta charset="utf-8"><title>PM209 LangChain·Qdrant 비교</title><style>body{max-width:1100px;margin:35px auto;font:16px/1.6 system-ui;background:#fafafa;color:#17212b}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:8px}article{border:1px solid #ccc;border-radius:8px;padding:18px;margin:18px 0;background:white}pre{white-space:pre-wrap;overflow-wrap:anywhere}select{padding:8px}</style><h1>LangChain + Qdrant 내부 검색: 30문항 비교</h1><p>동일 질문·번역·프롬프트, 가중치 5개 × Top-1/5, 새 답변 300개. Top-k는 한 답변에 전달한 근거 청크 수입니다.</p>'+''.join(table)+
         '<p>정답/부분정답/오답/미답변/형식 오류를 구분하며 오류도 분모에 포함합니다. Codex 검수로 독립 사람 평가가 아닙니다. 기존 SQLite/Python과 BM25 처리·가중 RRF 계산식이 달라 순수 프레임워크 효과로 해석할 수 없습니다. 같은 30문항은 튜닝용 진단 표본입니다. 본문 분류도 검색 대상은 본문·그림·표 전체입니다. TEXT06 질문/정답 범위 차이, TEXT08 공식 QA 오기, FIG05 질문 확장, FIG10 분리 동작 보완은 기존 기준을 그대로 적용했습니다.</p><p><a href="protocol.json">고정 조건</a> · <a href="summary.json">집계</a> · <a href="review.json">검수</a> · <a href="results.json">실제 출력</a> · <a href="retrieval.json">검색·Qdrant 요청</a></p><select id="variant"><option value="">모든 조건</option>'+options+'</select> <select id="verdict"><option value="">모든 판정</option>'+''.join(f'<option value="{key}">{name}</option>' for key,name in labels.items())+'</select>'+''.join(cards)+
         '<script>function filter(){for(const a of document.querySelectorAll("article")){a.hidden=(variant.value&&a.dataset.variant!==variant.value)||(verdict.value&&a.dataset.verdict!==verdict.value)}}document.querySelectorAll("select").forEach(s=>s.addEventListener("change",filter));</script></html>')
    # Keep factual correctness, evidence coverage and citation/wording issues distinct.
    note=('<p>정답 수는 고정한 원본 기준 핵심 내용으로 판정했습니다. 검색 근거의 충분성과 출처 표기·표현 문제는 문항별 주석에서 따로 확인하세요. '
          '출처 ID 검사 통과가 모든 주장의 인용 정확도를 보증하지 않습니다. 동일 입력·출력은 이전 검수 판정을 재사용했고, 나머지는 원본 기준과 검색 근거를 대조했습니다. '
          '<a href="legacy-comparison.json">이전 실험과 수치 비교</a> · <a href="system-validation.json">구현·보존 검증</a></p>')
    (DEST/'report.html').write_text(html.replace('<select id="variant">',note+'<select id="variant">'),encoding='utf-8')
    state=app.legacy.read_json(DEST/'status.json')
    state.update(status='complete',stage='reviewed',updated_at=now(),review_groups=len(groups))
    app.write_json(DEST/'status.json',state)
    print(json.dumps(summary['weights'],ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','render'])
    parser.add_argument('--results',type=Path,default=DEST)
    args=parser.parse_args()
    DEST=args.results
    prepare() if args.stage=='prepare' else render()
