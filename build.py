#!/usr/bin/env python3
"""Build the self-contained tutorial. Only the Python standard library is needed."""
import ast
import hashlib
import html
import json
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parent
WEB = ROOT / 'webapp'
SHA = '5bae5bb7928d65906e24e99f5b5d3f99a4daaf93'
GITHUB = f'https://github.com/THUDM/slime/blob/{SHA}/'
COURSE = json.loads((ROOT / 'course.json').read_text())
SYMBOLS = {}


def collect_symbols(tree, prefix=''):
    result = {}
    for node in getattr(tree, 'body', []):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            name = prefix + node.name
            result[name] = node.lineno
            if isinstance(node, ast.ClassDef):
                result.update(collect_symbols(node, name + '.'))
    return result


def source_target(match):
    path, _, symbol = match.group(1).partition('#')
    if path not in SYMBOLS:
        raise ValueError(f'Missing source file: {path}')
    if symbol.startswith('L') and symbol[1:].isdigit():
        line = int(symbol[1:])
        if line > len((ROOT / 'source' / path).read_text().splitlines()):
            raise ValueError(f'Invalid source line: {path} {line}')
    elif symbol:
        if symbol not in SYMBOLS[path]:
            raise ValueError(f'Missing source symbol: {path} {symbol}')
        line = SYMBOLS[path][symbol]
    else:
        line = 1
    return f'../source/{path}.html#L{line}'


def head(title, prefix):
    return f'''<!doctype html><html lang="zh-Hans"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="从 Ray 入门到读懂 slime/ray：基于固定源码的中文交互教程。">
<title>{html.escape(title)} · Ray 与 slime 源码导读</title>
<link rel="icon" href="data:,">
<link rel="stylesheet" href="{prefix}assets/style.css">
<link rel="stylesheet" href="{prefix}assets/tutorial.css"></head>'''


def build_sources():
    manifest = {'repository': 'https://github.com/THUDM/slime', 'commit': SHA,
                'commit_date': '2026-09-19', 'retrieved': '2026-09-20',
                'scope': 'Selected source snapshot, not an executable slime installation', 'files': {}}
    for file in sorted((ROOT / 'source').rglob('*')):
        if not file.is_file() or file.name == 'manifest.json':
            continue
        rel = file.relative_to(ROOT / 'source').as_posix()
        raw = file.read_text()
        SYMBOLS[rel] = collect_symbols(ast.parse(raw)) if file.suffix == '.py' else {}
        manifest['files'][rel] = hashlib.sha256(file.read_bytes()).hexdigest()
        target = WEB / 'source' / (rel + '.html')
        target.parent.mkdir(parents=True, exist_ok=True)
        prefix = '../' * len(target.relative_to(WEB).parts[:-1])
        lines = '\n'.join(f'<span class="source-line" id="L{i}"><a href="#L{i}">{i:4}</a> {html.escape(s)}</span>'
                          for i, s in enumerate(raw.splitlines() or [''], 1))
        target.write_text(head(rel, prefix) + f'''<body class="source-page"><header class="topbar">
<a class="brand" href="{prefix}index.html">Ray 与 slime</a><a href="{GITHUB}{rel}">GitHub 固定版本 ↗</a></header>
<main class="source-wrap"><p><a href="{prefix}chapters/sources.html">← 源码索引</a> · {SHA[:12]} · 只读快照</p>
<h1>{html.escape(rel)}</h1><pre class="source-code"><code>{lines}</code></pre></main></body></html>''')
    (ROOT / 'source' / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')


def build():
    (WEB / 'chapters').mkdir(parents=True, exist_ok=True)
    build_sources()
    combined = ['# Ray 与 slime 源码导读\n', f'源码提交：`{SHA}`。全部课程已展开。\n']
    for i, chapter in enumerate(COURSE):
        cid = chapter['id']
        md = (ROOT / 'content' / (cid + '.md')).read_text()
        combined.append(md)
        md = re.sub(r'source:([^\s)]+)', source_target, md)
        md = md.replace('</script', '<\\/script')
        menu = ''.join(f'<a href="{c["id"]}.html"'+(' aria-current="page"' if c == chapter else '')+f'>{c["title"]}</a>' for c in COURSE)
        links = []
        for idx, direction, cls in [(i-1, '← 上一章', 'prev'), (i+1, '下一章 →', 'next')]:
            if 0 <= idx < len(COURSE):
                c = COURSE[idx]
                links.append(f'<a class="{cls}" href="{c["id"]}.html"><span class="dir">{direction}</span><div class="ttl">{c["title"]}</div></a>')
        label = f'第 {int(cid[2:])} 章' if cid.startswith('ch') else '附录'
        page = head(chapter['title'], '../') + f'''<body data-chapter="{cid}" data-lab="{chapter.get('lab','')}">
<a class="skip" href="#content">跳到正文</a>
<header class="topbar"><a class="brand" href="../index.html">Ray 与 slime<span>源码导读</span></a>
<nav class="chapnav"><a href="../index.html">全部章节</a><a href="sources.html">版本与证据</a></nav></header>
<div class="layout"><aside class="sidebar"><nav class="toc" id="toc"><h2>本章目录</h2><details class="book-nav"><summary>全书章节</summary>{menu}</details></nav></aside>
<main class="main"><p class="chapter-meta">{chapter['part']} / {label} · 约 {chapter['minutes']} 分钟 · 已展开</p>
<details class="mobile-nav"><summary>选择章节</summary>{menu}</details><div id="lab-banner"></div>
<article id="content" class="prose"></article><noscript>请启用 JavaScript 阅读网页，或直接查看 content 中的 Markdown。</noscript>
<section id="lab" class="lab" hidden><h2 class="lab-title"></h2><p class="lab-intro"></p><div id="lab-root"></div></section>
<div class="study-actions"><button id="mark-read" type="button">标记为已读</button><span id="read-status" role="status"></span></div>
<nav class="pager">{''.join(links)}</nav><footer class="pagefoot">独立中文源码教程 · slime {SHA[:12]} · 源码事实、教学简化与实验结果分别标注。<br>视觉样式参考 Ultra-Scale Playbook 中文版；署名见 <a href="sources.html">来源与许可</a>。</footer>
</main></div><script type="text/markdown" id="chapter-md">{md}</script>
<script src="../assets/marked.min.js"></script><script src="../assets/app.js"></script><script src="../assets/labs.js"></script></body></html>'''
        (WEB / 'chapters' / (cid + '.html')).write_text(page)
    sections = []
    for part in dict.fromkeys(c['part'] for c in COURSE):
        cards = []
        for c in [c for c in COURSE if c['part'] == part]:
            label = f'第 {int(c["id"][2:]):02d} 章' if c['id'].startswith('ch') else '附录'
            badge = '<span class="badge">交互实验</span>' if c.get('lab') else ''
            cards.append(f'<a class="card" data-id="{c["id"]}" href="chapters/{c["id"]}.html">{badge}<span class="no">{label}</span><span class="zh">{c["title"]}</span><span class="en">{c["en"]}</span><span class="desc">{c["desc"]}</span><span class="card-time">约 {c["minutes"]} 分钟</span></a>')
        sections.append(f'<section class="course-part"><h2>{part}</h2><div class="chapter-grid">{"".join(cards)}</div></section>')
    page = head('从第一次 remote 到一轮 RL 训练', '') + f'''<body data-chapter="index">
<main><section class="hero"><div class="eyebrow">简体中文 · 源码导学 · 交互版</div>
<h1>从 Ray 入门<br>读懂 slime 的训练编排</h1>
<p class="sub">从一个远程函数开始，沿着真实调用链，走完 GPU 分配、模型启动、生成样本、分发数据、训练和权重同步。为了解过 Ray、还没有实战经验的你而写。</p>
<p class="version">基于 THUDM/slime · <a href="{GITHUB}train.py">{SHA[:12]}</a> · 2026-09-20 读取</p>
<div class="hero-actions"><a class="primary-link" href="chapters/ch00.html">开始阅读 →</a><a href="chapters/ch13.html">先跑 CPU 小实验</a><a id="resume" hidden>继续上次阅读 →</a></div>
<div class="meta"><div><b>14</b>循序渐进的章节</div><div><b>6</b>浏览器交互实验</div><div><b>6</b>CPU 实验脚本</div><div><b>3</b>速查与练习附录</div></div>
<div class="route-strip"><span>Ray 基础</span><span>→</span><span>资源与进程</span><span>→</span><span>数据与训练</span><span>→</span><span>同步与排障</span></div></section>
<div class="grid-wrap"><div class="catalog-tools"><label for="search">查找章节</label><input id="search" type="search" placeholder="试试 ObjectRef、GPU、权重…"><span id="progress"></span></div><p id="no-results" hidden>没有找到匹配章节，换一个关键词试试。</p>{''.join(sections)}</div></main>
<footer class="foot">基于固定版本的独立源码教程，不是 slime 官方文档。保留本地源码快照，正文和交互均可离线阅读。<br>页面样式改编自 <a href="https://github.com/ai-twinkle/ultrascale-playbook-zh-tw">Ultra-Scale Playbook 中文版</a>（Apache 2.0）。<a href="chapters/sources.html">来源、许可与验证边界 →</a></footer><script src="assets/app.js"></script></body></html>'''
    (WEB / 'index.html').write_text(page)
    # A portable, single-file Markdown edition with immutable GitHub links.
    def github_target(m):
        local = source_target(m)
        path, _, _symbol = m.group(1).partition('#')
        return GITHUB + path + '#' + local.split('#')[-1]
    combined_md = '\n\n---\n\n'.join(combined)
    combined_md = re.sub(r'source:([^\s)]+)', github_target, combined_md)
    combined_md = re.sub(r'\]\((ch\d+|glossary|practice|sources)\.html', r'](webapp/chapters/\1.html', combined_md)
    combined_md = combined_md.replace('../../examples/', 'examples/')
    (ROOT / 'tutorial.md').write_text(combined_md)
    shutil.copytree(ROOT / 'examples', WEB / 'examples', dirs_exist_ok=True, ignore=shutil.ignore_patterns('__pycache__'))
    # Chapter URLs point into webapp/examples so the site can be hosted alone.
    for page in (WEB / 'chapters').glob('*.html'):
        page.write_text(page.read_text().replace('../../examples/', '../examples/'))
    print(f'Built {len(COURSE)} chapter pages + index; {len(SYMBOLS)} source files; 0 missing chapters.')


if __name__ == '__main__':
    build()
