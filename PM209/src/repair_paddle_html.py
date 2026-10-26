"""Bounded repair of Paddle row-group tags and literal OCR text, never cell inference.

Rows/cells must already be closed. Their order, attributes and decoded text are
compared before/after repair. Malformed or repetitive cell structures are refused.
"""
import hashlib
import html
from html.parser import HTMLParser
import json
import re

TOKEN = re.compile(r'</?(?:html|body|table|thead|tbody|tfoot|tr|td|th)\b[^>]*>', re.I)
GROUPS = {'thead', 'tbody', 'tfoot'}


class Attributes(HTMLParser):
    def handle_starttag(self, tag, attrs):
        self.attrs = dict(attrs)


def parse(content):
    groups, row, cell = [], None, None
    group = None
    opened = closed = 0
    changes = []
    pos = 0
    for match in TOKEN.finditer(content):
        gap = content[pos:match.start()]
        if cell is not None:
            cell['raw_text'] += gap
        elif gap.strip():
            raise ValueError('Unexpected text outside a cell')
        token = match.group(); pos = match.end()
        tag = re.match(r'</?([a-z]+)', token, re.I)[1].lower()
        ending = token.startswith('</')
        if tag in ('html', 'body'):
            if cell is not None or row is not None:
                raise ValueError('Document ended inside a row/cell')
            continue
        if tag == 'table':
            if row is not None or cell is not None:
                raise ValueError('Table ended with an unfinished row/cell')
            if ending: closed += 1
            else: opened += 1
            continue
        if tag in GROUPS:
            if row is not None or cell is not None:
                raise ValueError('Row-group boundary inside a row/cell')
            if ending:
                if group is None:
                    raise ValueError('Row-group close without rows')
                if group['implicit'] and tag == 'thead':
                    group['tag'] = 'thead'
                    changes.append('inferred_thead_start_from_explicit_end')
                elif group['tag'] != tag:
                    changes.append('matched_group_end_to_explicit_start')
                group = None
            else:
                if group is not None: changes.append('closed_group_before_next_group')
                group = {'tag': tag, 'implicit': False, 'rows': []}
                groups.append(group)
            continue
        if opened != 1 or closed:
            raise ValueError('Rows/cells outside the single table')
        if tag == 'tr':
            if ending:
                if row is None or cell is not None: raise ValueError('Unfinished row/cell')
                row = None
            else:
                if row is not None or cell is not None: raise ValueError('Unclosed previous row')
                if group is None:
                    group = {'tag': 'tbody', 'implicit': True, 'rows': []}
                    groups.append(group); changes.append('made_implicit_tbody_explicit')
                row = []; group['rows'].append(row)
            continue
        if ending:
            if cell is None or cell['tag'] != tag: raise ValueError('Unclosed/mismatched cell')
            cell['text'] = html.unescape(cell['raw_text'])
            cell = None
        else:
            if row is None or cell is not None: raise ValueError('Cell outside row or previous cell unclosed')
            parser = Attributes(); parser.feed(token)
            attrs = parser.attrs
            if set(attrs) - {'rowspan', 'colspan'}: raise ValueError('Unsupported cell attribute')
            if any(v is None or not v.isdigit() or int(v) < 1 for v in attrs.values()):
                raise ValueError('Invalid cell span')
            cell = {'tag': tag, 'attrs': attrs, 'start': token, 'raw_text': ''}
            row.append(cell)
    if content[pos:].strip() or row is not None or cell is not None or opened != 1 or closed != 1:
        raise ValueError('Incomplete table/document')
    if not any(g['rows'] for g in groups): raise ValueError('No rows')
    return groups, changes


def signature(groups):
    return [[(c['tag'], c['attrs'], c['text']) for c in row] for g in groups for row in g['rows']]


def geometry(groups):
    """Audit logical placement. Warnings are not evidence of semantic correctness."""
    rows = [row for g in groups for row in g['rows']]
    occupied, warnings, placements = {}, [], []
    group_ends = []
    n = 0
    for g in groups:
        n += len(g['rows'])
        group_ends.extend([n] * len(g['rows']))
    for y, row in enumerate(rows):
        x = 0
        for cell in row:
            while (y, x) in occupied: x += 1
            rs, cs = (int(cell['attrs'].get(k, 1)) for k in ('rowspan', 'colspan'))
            if rs > 1000 or cs > 1000: raise ValueError('Unreasonable span')
            if y + rs > len(rows): warnings.append('rowspan_beyond_last_row')
            if y + rs > group_ends[y]: warnings.append('rowspan_crosses_row_group')
            for yy in range(y, y + rs):
                for xx in range(x, x + cs):
                    if (yy, xx) in occupied: warnings.append('overlapping_spans')
                    occupied[yy, xx] = bool(cell['text'].strip())
            placements.append((y, x, rs, cs)); x += cs
    trailing = 0
    for y in reversed(range(len(rows))):
        if any(value for (yy, _), value in occupied.items() if yy == y): break
        trailing += 1
    if trailing >= 3: warnings.append('repeated_empty_tail_rows')
    return {'placements': placements, 'warnings': sorted(set(warnings)), 'trailing_empty_rows': trailing}


def normalize(content):
    groups, changes = parse(content)
    original = signature(groups)
    audit = geometry(groups)
    if audit['warnings']:
        raise ValueError('Structure needs recognition/review: ' + ', '.join(audit['warnings']))
    pieces = ['<html><body><table>']
    escaped = 0
    for group in groups:
        if not group['rows']: continue
        pieces.append('<' + group['tag'] + '>')
        for row in group['rows']:
            pieces.append('<tr>')
            for c in row:
                text = html.escape(c['text'], quote=False)
                escaped += text != c['raw_text']
                pieces.extend([c['start'], text, '</' + c['tag'] + '>'])
            pieces.append('</tr>')
        pieces.append('</' + group['tag'] + '>')
    pieces.append('</table></body></html>')
    result = ''.join(pieces)
    after, _ = parse(result)
    if signature(after) != original or geometry(after)['placements'] != audit['placements']:
        raise ValueError('Repair changed a cell, its order, or its span')
    import paddle_tables
    quality = paddle_tables.inspect_html(result)
    return result, dict(row_cell_text_span_preserved=True, logical_positions_preserved=True,
                        signature_sha256=hashlib.sha256(json.dumps(original, ensure_ascii=False).encode()).hexdigest(),
                        row_group_repairs=sorted(set(changes)), escaped_cells=escaped,
                        rows=quality['rows'], cells=quality['cells'], geometry=audit,
                        semantic_accuracy_verified=False)
