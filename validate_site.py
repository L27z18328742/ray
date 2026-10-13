#!/usr/bin/env python3
"""Check built local links/line anchors and snapshot integrity, without network access."""
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parent
WEB = ROOT / 'webapp'
errors = []
count = 0
for page in WEB.rglob('*.html'):
    text = page.read_text()
    refs = re.findall(r'(?:href|src)="([^"]+)"', text)
    if page.parent.name == 'chapters':
        refs += re.findall(r'\]\(([^\s)]+)\)', text)
    for href in refs:
        url = urlparse(href)
        if url.scheme or url.netloc or not url.path:
            continue
        target = (page.parent / unquote(url.path)).resolve()
        count += 1
        if not target.is_file():
            errors.append([str(page.relative_to(WEB)), href, 'missing file'])
        elif url.fragment and '/source/' in str(target) and f'id="{url.fragment}"' not in target.read_text():
            errors.append([str(page.relative_to(WEB)), href, 'missing source anchor'])

manifest = json.loads((ROOT / 'source/manifest.json').read_text())
for path, digest in manifest['files'].items():
    actual = hashlib.sha256((ROOT / 'source' / path).read_bytes()).hexdigest()
    if actual != digest:
        errors.append([path, 'snapshot hash changed'])
chapters = json.loads((ROOT / 'course.json').read_text())
for chapter in chapters:
    for folder, suffix in [('content', '.md'), ('webapp/chapters', '.html')]:
        if not (ROOT / folder / (chapter['id'] + suffix)).is_file():
            errors.append([chapter['id'], 'missing chapter'])
report = {'chapter_count': len(chapters), 'checked_local_targets': count,
          'snapshot_files': len(manifest['files']), 'errors': errors}
(ROOT / 'validation').mkdir(exist_ok=True)
(ROOT / 'validation/links.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(report, ensure_ascii=False))
raise SystemExit(bool(errors))
