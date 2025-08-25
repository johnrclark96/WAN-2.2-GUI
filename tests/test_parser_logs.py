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


def test_official_log():
    lines = Path('tests/data/official.log').read_text().splitlines()
    pct = []
    done = None
    for line in lines:
        msg = json.loads(line)
        if msg.get('event') == 'progress':
            pct.append(msg['percent'])
        elif msg.get('event') == 'done':
            done = msg.get('video')
    assert pct[-1] == 50
    assert done == 'ok.mp4'


def test_legacy_progress():
    lines = Path('tests/data/legacy.log').read_text().splitlines()
    percents = []
    for line in lines:
        m = re.search(r'percent=(\d+)', line)
        if m:
            percents.append(int(m.group(1)))
    assert percents[-1] == 100
