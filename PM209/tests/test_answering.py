import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import answering as app
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import rag_pipeline as pipeline


def hit(cid='c1', kind='text', text='The device weighs 4.42 kg.'):
    return {'chunk_id': cid, 'manual_id': 'manual', 'page_id': 'manual_00001',
            'chunk_type': kind, 'text': text, 'quality_flags': [],
            'content_region_ids': ['body'], 'context_region_ids': ['title'],
            'image_filename': 'images/manual/images/page.jpg',
            'source_shapes': [{'id': 'body', 'shape': {'x': 1, 'y': 2, 'width': 3, 'height': 4}},
                              {'id': 'title', 'shape': {'x': 5, 'y': 6, 'width': 7, 'height': 8}}]}


def response(answer='무게는 4.42 kg입니다.', ids=None, insufficient=False, reason='stop'):
    return {'done': True, 'done_reason': reason, 'eval_count': 42, 'prompt_eval_count': 100,
            'message': {'content': json.dumps({'answer': answer, 'source_ids': ['S1'] if ids is None else ids,
                                             'insufficient_evidence': insufficient})}}


class AnswerTests(unittest.TestCase):
    def setUp(self):
        self.sources, _ = app.prepare_context('무게는 얼마인가요?', [hit()], 'manual')

    def test_whitelist_keeps_qa_and_images_out_and_preserves_input(self):
        source = hit(kind='figure')
        source.update(qa_data='SECRET_QA', question='SECRET_GOLD_QUESTION',
                      answer='SECRET_GOLD_ANSWER', images=['SECRET_IMAGE'])
        before = copy.deepcopy(source)
        sources, _ = app.prepare_context('무게는?', [source], 'manual')
        messages = app.messages_for('무게는?', sources)
        serialized = json.dumps(messages, ensure_ascii=False)
        self.assertNotIn('SECRET', serialized)
        self.assertEqual(source, before)
        self.assertNotIn('qa_data', sources[0])
        self.assertEqual(set(messages[1]), {'role', 'content'})

    def test_rejects_cross_manual_and_overlong_question(self):
        with self.assertRaises(ValueError):
            app.prepare_context('무게는?', [hit()], 'other')
        with self.assertRaises(ValueError):
            app.prepare_context('가'*700, [hit()], 'manual')

    def test_budget_drops_whole_chunks_without_silently_truncating(self):
        sources, excluded = app.prepare_context('무게는?', [hit('large', text='X'*10000), hit()], 'manual')
        self.assertEqual(sources[0]['source_id'], 'S2')
        self.assertEqual(sources[0]['text'], hit()['text'])
        self.assertEqual(excluded, [{'chunk_id': 'large', 'reason': 'context_budget'}])
        self.assertLessEqual(app.prompt_upper_bound(app.messages_for('무게는?', sources, True))
                             + app.OPTIONS['num_predict'], app.OPTIONS['num_ctx'])

    def test_rejects_truncation_bad_ids_missing_citations_and_non_korean(self):
        for bad in [response(reason='length'), response(ids=['S99']), response(ids=[]),
                    response(ids=['S1', 'S1']), response(answer='It weighs 4.42 kg.'),
                    response(answer='무게는 4.42 kg입니다. [S99]')]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                app.validate_response(bad, self.sources)
        self.assertEqual(app.validate_response(response(), self.sources)['source_ids'], ['S1'])

    def test_abstention_never_displays_unsupported_prose(self):
        value = app.validate_response(response(answer='근거가 없지만 무게는 999 kg입니다.', ids=[],
                                               insufficient=True), self.sources)
        self.assertEqual(value['answer'], app.ABSTENTION)
        with self.assertRaises(ValueError):
            app.validate_response(response(insufficient=True), self.sources)

    def test_only_cited_figure_content_regions_are_linked(self):
        sources, _ = app.prepare_context('그림?', [hit('a', kind='figure'), hit('b', kind='figure')], 'manual')
        citations, figures = app.resolve_citations({'source_ids': ['S1']}, sources)
        self.assertEqual(len(citations), 1)
        self.assertEqual(len(figures), 1)
        self.assertEqual(figures[0]['region_id'], 'body')
        self.assertEqual(figures[0]['image_filename'], hit()['image_filename'])
        self.assertEqual(figures[0]['shape'], hit()['source_shapes'][0]['shape'])

    def test_no_sources_does_not_call_model(self):
        with patch.object(app.httpx, 'Client') as client:
            value, details = app.generate_answer('질문', [], 'http://unused')
        client.assert_not_called()
        self.assertTrue(value['insufficient_evidence'])
        self.assertFalse(details['called'])

    def run_mock_generation(self, chat_responses):
        requests = []
        def handle(request):
            path = request.url.path
            if path == '/api/tags':
                return httpx.Response(200, json={'models': [{'name': app.MODEL, 'digest': 'digest'}]})
            if path == '/api/show':
                return httpx.Response(200, json={'capabilities': ['completion'], 'model_info': {
                    'general.architecture': 'qwen35', 'tokenizer.ggml.model': 'gpt2'}})
            if path == '/api/ps':
                return httpx.Response(200, json={'models': [{'name': app.MODEL, 'size_vram': 123}]})
            self.assertEqual(path, '/api/chat')
            requests.append(json.loads(request.content))
            return httpx.Response(200, json=chat_responses[len(requests)-1])
        client = httpx.Client(base_url='http://test', transport=httpx.MockTransport(handle))
        with patch.object(app.httpx, 'Client', return_value=client):
            value, details = app.generate_answer('무게는?', self.sources, 'http://test')
        return value, details, requests

    def test_truncated_response_retried_and_only_valid_final_is_returned(self):
        value, details, requests = self.run_mock_generation([response(reason='length'), response()])
        self.assertEqual(value['source_ids'], ['S1'])
        self.assertEqual(len(details['attempts']), 2)
        self.assertIn('error', details['attempts'][0])
        self.assertNotIn('error', details['attempts'][1])
        for request in requests:
            self.assertFalse(request['think'])
            self.assertFalse(request['stream'])
            self.assertEqual(request['options']['num_ctx'], 8192)
            self.assertNotIn('images', request)
            for message in request['messages']:
                self.assertEqual(set(message), {'role', 'content'})

    def test_two_invalid_outputs_fail_instead_of_becoming_an_answer(self):
        with self.assertRaises(app.AnswerGenerationError) as error:
            self.run_mock_generation([response(ids=['S99']), response(reason='length')])
        self.assertEqual(len(error.exception.attempts), 2)


class AnswerPipelineTests(unittest.TestCase):
    def test_answer_stage_defaults_to_full_index_without_changing_sample_default(self):
        argv = ['rag_pipeline.py', 'answer', '--manual', 'manual', '--query', '무게는?']
        with patch.object(sys, 'argv', argv), patch.object(pipeline, 'answer') as answer:
            pipeline.main()
        self.assertEqual(answer.call_args.args[0].output, pipeline.FULL_OUTPUT)
        self.assertNotEqual(pipeline.FULL_OUTPUT, pipeline.DEFAULT_OUTPUT)
        argv[1] = 'search'
        with patch.object(sys, 'argv', argv), patch.object(pipeline, 'search') as search:
            pipeline.main()
        self.assertEqual(search.call_args.args[0].output, pipeline.DEFAULT_OUTPUT)

    def test_latest_failure_cannot_be_mistaken_for_previous_success(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            (output/'index-identity.json').write_text('{}', encoding='utf-8')
            args = SimpleNamespace(query='무게는?', manual=['manual'], output=output,
                                   model=pipeline.MODEL, top_k=5, url='http://test', answer_model=app.MODEL)
            model, db = Mock(), Mock()
            valid = {'answer': '무게는 4.42 kg입니다.', 'source_ids': ['S1'], 'insufficient_evidence': False}
            with patch.object(pipeline, 'open_search', return_value=({}, [hit()], model, db)), \
                 patch.object(pipeline, 'retrieve', return_value=[hit()]), \
                 patch.object(pipeline, 'gpu_snapshot', return_value=None), \
                 patch.object(app, 'generate_answer', return_value=(valid, {})) as generate:
                result = pipeline.answer(args)
                self.assertEqual(result['status'], 'answered')
                self.assertEqual(result['citations'][0]['content_region_ids'], ['body'])
                db.upsert.assert_not_called()
                db.close.assert_called_once()
                model.client.post.assert_called_with('/api/generate', json={'model': pipeline.MODEL, 'keep_alive': 0})
                generate.side_effect = app.AnswerGenerationError([{'error': 'length'}, {'error': 'length'}])
                with self.assertRaises(SystemExit):
                    pipeline.answer(args)
            latest = json.loads((output/'last-answer.json').read_text(encoding='utf-8'))
            self.assertEqual(latest['status'], 'failed')
            self.assertNotIn('answer', latest)
            self.assertEqual(len(latest['failed_attempts']), 2)
            self.assertEqual(len(list((output/'answers').glob('*.json'))), 2)


if __name__ == '__main__':
    unittest.main()
