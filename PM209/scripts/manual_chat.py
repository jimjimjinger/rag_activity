"""Run one interactive question in the pinned RAG environment. IPC uses UTF-8 JSON lines."""
from contextlib import closing
import json
from pathlib import Path
import re
import subprocess
import sys

PM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PM / 'src'))
import answer_additions as additions
import evidence_answer as evidence
import lc_qdrant_rag as rag
from context_compression import ContextCompressor
from sentence_context import SentenceContextStore
from reranking import BGEReranker
from verification import GemmaVerifier, MODEL as GEMMA, unload

PORT = 11439
URL = f'http://127.0.0.1:{PORT}'


def emit(kind, value):
    print(json.dumps({'event': kind, 'value': value}, ensure_ascii=False), flush=True)


def search(manual, question):
    identity = rag.legacy.read_json(rag.OUTPUT / 'index-identity.json')
    validation = rag.legacy.read_json(rag.OUTPUT / 'final-validation.json')
    with closing(rag.ReusedQwenEmbeddings(url=URL)) as embeddings:
        if (validation['status'] != 'complete' or identity['packages'] != rag.versions()
                or identity['dense_identity'] != embeddings.identity
                or identity['sparse_model'] != rag.BM25_OPTIONS or identity['collection'] != rag.COLLECTION):
            raise RuntimeError('검색 인덱스와 현재 모델 설정이 일치하지 않습니다.')
        with closing(rag.QdrantClient(path=str(rag.OUTPUT / 'qdrant'))) as client:
            if rag.checked_collection(client).points_count != validation['points_verified']:
                raise RuntimeError('검색 인덱스의 청크 수가 검증 기록과 다릅니다.')
            retriever = rag.NativeRetriever(client, embeddings, rag.make_sparse())
            try:
                return retriever.search(question, manual, keyword_percent=50, top_k=rag.CANDIDATE_K)
            finally:
                embeddings.unload()


def linked_draft(chat, question, sources):
    messages = evidence.request(evidence.MAP_PROMPT, question, sources, evidence.MAP_OPTIONS, numbered=True)
    mapped = evidence.call_json(chat, messages, evidence.MAP_OPTIONS, evidence.mapping_schema(sources),
                                lambda value: evidence.validate_mapping_selection(value, sources))
    mapping = mapped['value'] or {'items': []}
    messages = evidence.request(evidence.ANSWER_PROMPT, question, sources, evidence.ANSWER_OPTIONS,
                                evidence_map=mapping, mapping_status=mapped['status'])
    reply = evidence.call_json(chat, messages, evidence.ANSWER_OPTIONS, rag.answering.response_schema(sources),
                               lambda value: evidence.validate_answer(value, sources))
    if reply['status'] != 'ok':
        raise RuntimeError('답변 형식을 확인하지 못했습니다. ' + reply['attempts'][-1]['error'])
    draft = reply['value']
    draft.update(sources=sources, status='insufficient_evidence' if draft['insufficient_evidence'] else 'answered')
    return draft, mapping


class GemmaChat(rag.LocalChat):
    def __init__(self):
        model = GemmaVerifier(URL)
        self.url, self.digest = URL, model.digest
        self.model = model.model.model_copy(update={'reasoning': False})


def review_draft(question, draft, mapping):
    sources = draft['sources']
    chat = GemmaChat()
    short = {key: draft[key] for key in ('answer', 'source_ids', 'status')}
    labelled = dict(short, answer=evidence.numbered_text(short['answer'], 'D')[0])
    messages = evidence.request(evidence.REVIEW_PROMPT, question, sources, evidence.REVIEW_OPTIONS,
                                numbered=True, evidence_map=evidence.compact_mapping(mapping), draft=labelled)
    review = evidence.call_json(chat, messages, evidence.REVIEW_OPTIONS, evidence.review_schema(sources, short),
                                lambda value: evidence.validate_review_selection(value, sources, short))
    if review['status'] != 'ok':
        return draft, '점검 결과를 확인하지 못해 Qwen 초안을 표시합니다.'
    if not additions.eligible_omissions(review):
        return draft, 'Gemma 점검 · 추가 내용 없음'
    messages = additions.model_messages(question, sources, draft, review, mapping)
    result = evidence.call_json(chat, messages, additions.OPTIONS, additions.response_schema(sources),
                                lambda value: additions.validate_additions(value, sources, draft))
    if result['status'] != 'ok':
        return draft, '보완 문장의 근거를 확인하지 못해 Qwen 초안을 표시합니다.'
    extra = result['value']['additions']
    return additions.compose(draft, extra), ('Gemma 점검 · 누락 보완' if extra else 'Gemma 점검 · 추가 내용 없음')


def answer(manual, question):
    rag.answering.SYSTEM_PROMPT = rag.answering.SYSTEM_PROMPT_V1
    rag.answering.prepare_context(question, [], manual)
    catalog = rag.legacy.read_json(PM / 'datasets/PM209/meta_data/name2meta.json')
    if manual not in catalog:
        raise ValueError('등록된 제품 매뉴얼을 선택해 주세요.')
    emit('stage', '로컬 모델을 준비하고 있어요')
    subprocess.run(['powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
                    str(PM / 'scripts/start-ollama.ps1'), '-Port', str(PORT)],
                   stdout=sys.stderr, check=True, timeout=60)
    chat = rag.LocalChat(URL)
    emit('stage', '매뉴얼에서 의미와 키워드로 검색하고 있어요')
    query = chat.translate(question)['english_query'] if re.search('[가-힣]', question) else question
    candidates = search(manual, query)
    emit('stage', '관련성 높은 근거와 주변 문맥을 찾고 있어요')
    selected, _ = BGEReranker(URL).rerank(query, candidates, 5)
    sources, _, _ = SentenceContextStore().prepare_context(question, selected, manual)
    emit('stage', '질문에 필요한 원문을 고르고 있어요')
    sources, _ = ContextCompressor(chat).compress(question, sources)
    if not sources:
        return {'answer': rag.answering.ABSTENTION, 'citations': [], 'note': '답변에 사용할 근거 부족'}
    emit('stage', 'Qwen이 근거를 연결해 답변을 작성하고 있어요')
    draft, mapping = linked_draft(chat, question, sources)
    emit('stage', 'Gemma가 빠진 내용이 있는지 점검하고 있어요')
    try:
        unload(rag.answering.MODEL, URL)
        final, note = review_draft(question, draft, mapping)
    except Exception as exc:
        print(f'Gemma review: {exc}', file=sys.stderr)
        final, note = draft, 'Gemma 점검을 마치지 못해 Qwen 초안을 표시합니다.'
    finally:
        # Release the last GPU model even when the next request hits the translation cache.
        try:
            unload(GEMMA, URL)
        except Exception as exc:
            print(f'Gemma unload: {exc}', file=sys.stderr)
    citations, _ = rag.answering.resolve_citations(final, sources)
    fields = ('source_id', 'page_id', 'image_filename', 'chunk_type', 'text')
    return {'answer': final['answer'], 'note': note,
            'citations': [{key: c[key] for key in fields} for c in citations]}


if __name__ == '__main__':
    try:
        request = json.load(sys.stdin)
        emit('result', answer(request['manual'], request['question']))
    except Exception as exc:
        emit('error', str(exc))
        sys.exit(1)
