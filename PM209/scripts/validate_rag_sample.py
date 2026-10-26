"""Read-only source coverage checks and three separate Korean retrieval probes."""
import argparse
from collections import defaultdict
import json
import re

from lxml import html
from rag_pipeline import (DEFAULT_OUTPUT, DATA, MODEL, SAMPLE_MANUALS, open_search,
                          read_lines, read_json, write_json, retrieve, check_sources)
from chunking import ocr_text, normalize, manual_id, FIGURES


def main():
    args = argparse.Namespace(output=DEFAULT_OUTPUT, model=MODEL, url='http://127.0.0.1:11434')
    manifest, chunks, model, db = open_search(args)
    try:
        check_sources(manifest)
        grouped=defaultdict(list)
        for chunk in chunks:
            for rid in chunk['content_region_ids'] + chunk['context_region_ids'] + chunk.get('deduplicated_region_ids',[]):
                grouped[rid].append(chunk)
        metadata=read_json(DATA/'meta_data/name2meta.json')
        errors=[]; checked=0; table_cells=0
        for filename in manifest['identity']['source_files']:
            for p in read_lines(filename):
                if manual_id(p, metadata) not in manifest['identity']['manuals']:
                    continue
                for r in p['bounding_boxes']:
                    result=grouped[r['id']]
                    if r['structure'] in {'Text','Title'} | FIGURES:
                        text=ocr_text(r) if r['structure'] in {'Text','Title'} else normalize(r.get('description',''))
                        if not text:
                            continue
                        checked+=1
                        if any(text in normalize(c['text']) for c in result):
                            continue
                        covered=set()
                        for c in result:
                            for span in c['source_spans']:
                                if span.get('region_id')==r['id']:
                                    covered.update(range(span['start'],span['end']))
                        if any(i not in covered and not ch.isspace() for i,ch in enumerate(text)):
                            errors.append({'region':r['id'],'error':'uncovered normalized character'})
                    elif r['structure']=='Table':
                        # Token presence is a loss check, NOT a row/column accuracy score.
                        root=html.fragment_fromstring(r['table_content'],create_parent='div')
                        searchable=' '.join(c['text'] for c in result)
                        for cell in root.xpath('.//td|.//th|.//caption'):
                            table_cells+=1
                            for token in normalize(cell.text_content()).split():
                                if token not in searchable:
                                    errors.append({'region':r['id'],'error':'missing table token','token':token})
        probes=[
            (SAMPLE_MANUALS[1],'스피커를 강제로 다시 시작하려면 전원 버튼을 몇 초 동안 눌러야 하나요?'),
            (SAMPLE_MANUALS[2],'이 노트북의 무게와 크기는 어떻게 되나요?'),
            (SAMPLE_MANUALS[0],'이 헤드폰은 두 살 아이가 사용해도 되나요?'),
        ]
        queries=[]
        for manual,question in probes:
            hits=retrieve(db,model,manual,question,5)
            if any(h['manual_id']!=manual for h in hits):
                errors.append({'error':'manual filter leak'})
            queries.append({'manual':manual,'question':question,'hits':hits})
        points,offset=[],None
        while True:
            batch,offset=db.scroll('pm209_chunks',limit=128,offset=offset,with_vectors=False,with_payload=True)
            points.extend(batch)
            if offset is None:
                break
        expected={c['chunk_id']:c for c in chunks}
        if len(points)!=len(expected) or any(p.payload!=expected.get(str(p.id)) for p in points):
            errors.append({'error':'stored payload differs from chunk source'})
        forbidden={'qa_data','question','answer','relevant'}
        if any(forbidden & set(p.payload) for p in points):
            errors.append({'error':'QA field leaked into index'})
        report={'normalized_text_regions_checked':checked,'table_cells_token_presence_checked':table_cells,
                'persisted_points_checked':len(points),'errors':errors,
                'caveat':'Character/token coverage does not prove original OCR or row/column semantic accuracy.',
                'korean_queries':queries}
        write_json(DEFAULT_OUTPUT/'validation.json',report)
        print(json.dumps({k:v for k,v in report.items() if k!='korean_queries'},ensure_ascii=False))
        for q in queries:
            print(json.dumps({'question':q['question'],'top':[{k:h[k] for k in ('page_id','chunk_type','score','text')} for h in q['hits'][:2]]},ensure_ascii=False))
        if errors:
            raise SystemExit(1)
    finally:
        db.close();model.close()


if __name__=='__main__':
    main()
