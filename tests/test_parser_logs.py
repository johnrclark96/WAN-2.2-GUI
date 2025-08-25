import json
import re
from pathlib import Path


def test_json_progress_done():
    lines = Path('tests/data/sample.log').read_text().splitlines()
    pct = []
    done = None
    for line in lines:
        msg = json.loads(line)
        if msg.get('event') == 'progress':
            pct.append(msg['percent'])
        elif msg.get('event') == 'done':
            done = msg.get('video')
    assert pct == [25]
    assert done == 'out.mp4'


def test_output_result_markers():
    lines = Path('tests/data/official.log').read_text().splitlines()
    out = None
    ok = False
    for line in lines:
        if line.startswith('[OUTPUT]'):
            out = line.split(' ', 1)[1]
        if line.strip() == '[RESULT] OK':
            ok = True
    assert out == 'ok.mp4'
    assert ok


def test_legacy_progress():
    lines = Path('tests/data/legacy.log').read_text().splitlines()
    percents = []
    for line in lines:
        m = re.search(r'percent=(\d+)', line)
        if m:
            percents.append(int(m.group(1)))
    assert percents[-1] == 100


def test_no_completion_pattern():
    lines = Path('tests/data/legacy.log').read_text().splitlines()
    has_output = any(line.startswith('[OUTPUT]') for line in lines)
    has_done = False
    for line in lines:
        if line.startswith('{'):
            try:
                msg = json.loads(line)
            except Exception:
                continue
            if msg.get('event') == 'done':
                has_done = True
                break
    assert not has_output and not has_done
