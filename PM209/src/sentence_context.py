"""Complete split OCR sentences in the ranked seeds before ordinary parent expansion."""
from collections import defaultdict
import re

import answering
from chunking import rect, sentence_spans, x_overlap
from parent_context import ParentContextStore


END = re.compile(r"""[.!?。！？]["'’”)]*$""")
NEW_ITEM = re.compile(r"^(?:[•●▪*–—-]\s*|\(?\d{1,3}[.)]\s|[a-zA-Z][.)]\s|(?:warning|caution|note)\s*:)", re.I)
# Uppercase unit symbols can follow a number broken off on the preceding line.
UNIT = re.compile(r"^(?:[°%]|(?:V|A|W|Hz|MHz|GHz|GB|MB)\b)")


def adjacent_lines(seed, tail, left, right):
    """Require nearby, horizontally aligned OCR boxes in the same reading column."""
    shapes = {s['id']: s for c in (seed, tail) for s in c['source_shapes']}
    if left not in shapes or right not in shapes:
        return False
    a, b = rect(shapes[left]), rect(shapes[right])
    width = min(a[2] - a[0], b[2] - b[0])
    height = min(a[3] - a[1], b[3] - b[1])
    return (width > 0 and height > 0 and -2 <= b[1] - a[3] <= min(14, .75 * height)
            and abs(a[0] - b[0]) <= min(36, .2 * width)
            and x_overlap(a, b) >= .7 * width)


def continuation(page, cid):
    seed = page['chunks'][cid]
    parents = page['child_to_parents'][cid]
    if seed['chunk_type'] != 'text' or len(parents) != 1 or not seed.get('source_spans'):
        return None
    head = seed['text'].rstrip()
    if END.search(head) or head.endswith((':', ';')):
        return None
    siblings = page['parents'][parents[0]]['child_ids']
    pos = siblings.index(cid) + 1
    if pos == len(siblings):
        return None
    tid = siblings[pos]
    tail = page['chunks'][tid]
    if tail['chunk_type'] != 'text' or page['child_to_parents'][tid] != parents or not tail.get('source_spans'):
        return None
    left, right = seed['source_spans'][-1], tail['source_spans'][0]
    if left['field'] != 'ocr_normalized' or right['field'] != 'ocr_normalized':
        return None
    body = tail['text'].removeprefix(tail.get('title', '') + '\n')
    first_region = body[:right['end'] - right['start']]
    offset = 0
    joiner = '' if head.endswith('-') else ' '
    if left['region_id'] == right['region_id']:
        overlap = left['end'] - right['start']
        if right['start'] < left['start'] or not -1 <= overlap < len(first_region):
            return None
        if overlap > 0 and not head.endswith(first_region[:overlap]):
            return None
        offset = max(0, overlap)
        if overlap >= 0:
            joiner = ''
    elif right['start'] != 0 or not adjacent_lines(seed, tail, left['region_id'], right['region_id']):
        return None

    rest = first_region[offset:]
    clean = rest.lstrip()
    if not clean or NEW_ITEM.match(clean):
        return None
    follows_number = head[-1:].isdigit() and UNIT.match(clean)
    if not (clean[0].islower() or clean[0].isdigit() or follows_number):
        return None
    # Keep only the end of this sentence, never the next instruction or paragraph.
    end = sentence_spans(clean)[0][1]
    words = clean[:end].rstrip()
    if not END.search(words) or '\n' in words:
        return None
    start = offset + len(rest) - len(clean)
    span = dict(right, start=right['start'] + start, end=right['start'] + start + len(words))
    if rest != clean:
        joiner = ' '
    return tid, joiner + words, span, start + len(words) == len(body)


def joined_seed(seed, tail, suffix, span):
    region = span['region_id']
    shapes = {s['id']: s for s in seed['source_shapes']}
    shapes.update((s['id'], s) for s in tail['source_shapes'] if s['id'] == region)
    return dict(seed, text=seed['text'].rstrip() + suffix,
                content_region_ids=list(dict.fromkeys([*seed['content_region_ids'], region])),
                source_spans=[*seed['source_spans'], span], source_shapes=list(shapes.values()),
                quality_flags=sorted(set(seed['quality_flags'] + tail['quality_flags'])))


class _QuestionPages(ParentContextStore):
    """Use the ordinary budget, source numbering and expansion on request-local page views."""
    def __init__(self, directory, pages):
        self.directory = directory
        self.pages = pages

    def page(self, manual, page_id):
        return self.pages[manual, page_id]


class SentenceContextStore(ParentContextStore):
    def prepare_context(self, question, hits, manual):
        pages = {}
        seeds = []
        links = []
        covered = defaultdict(set)
        for hit in hits:
            if hit['manual_id'] != manual:
                raise ValueError('Cross-manual parent lookup')
            key = (manual, hit['page_id'])
            if key not in pages:
                page = self.page(*key)
                pages[key] = dict(page, chunks=page['chunks'].copy(), parents=page['parents'].copy())
            seeds.append(pages[key]['chunks'][hit['chunk_id']])
        originals, _ = answering.prepare_context(question, seeds, manual)
        admitted = {s['chunk_id']: s for s in originals}
        seed_ids = {s['chunk_id'] for s in seeds}
        for seed in seeds:
            cid = seed['chunk_id']
            if cid not in admitted:
                continue
            key = (manual, seed['page_id'])
            original = self.page(*key)
            match = continuation(original, cid)
            if not match:
                continue
            tid, suffix, span, complete_tail = match
            merged = joined_seed(seed, original['chunks'][tid], suffix, span)
            # Do not let a continuation displace a ranked seed that previously fit.
            trial = dict(admitted)
            trial[cid] = dict(merged, source_id=admitted[cid]['source_id'])
            messages = answering.messages_for(question, trial.values(), retry=True)
            if answering.prompt_upper_bound(messages) + answering.OPTIONS['num_predict'] > answering.OPTIONS['num_ctx']:
                continue
            admitted = trial
            pages[key]['chunks'][cid] = merged
            links.append({'seed_chunk_id': cid, 'continuation_chunk_id': tid, 'source_span': span})
            if complete_tail and tid not in seed_ids:
                covered[key].add(tid)
        for key, ids in covered.items():
            parents = pages[key]['parents']
            for pid in {p for cid in ids for p in pages[key]['child_to_parents'][cid]}:
                parents[pid] = dict(parents[pid], child_ids=[cid for cid in parents[pid]['child_ids'] if cid not in ids])
        view = _QuestionPages(self.directory, pages)
        sources, excluded, trace = view.prepare_context(question, hits, manual)
        trace['sentence_links'] = links
        return sources, excluded, trace
