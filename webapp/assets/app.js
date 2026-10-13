/* Site UI, no remote services or analytics. */
(() => {
  const safeLoad = (key, fallback) => { try { return JSON.parse(localStorage.getItem(key)) ?? fallback; } catch { return fallback; } };
  const safeSave = (key, value) => { try { localStorage.setItem(key, JSON.stringify(value)); return true; } catch { return false; } };
  const chapter = document.body.dataset.chapter;
  let read = safeLoad('ray-guide-read-v1', []);
  if (!Array.isArray(read)) read = [];
  const md = document.getElementById('chapter-md');
  if (md) {
    const content = document.getElementById('content');
    content.innerHTML = marked.parse(md.textContent.replaceAll('<\\/script', '</script'));
    const heads = [...content.querySelectorAll('h2,h3')];
    const ol = document.createElement('ol');
    heads.forEach((h, i) => {
      h.id = `sec-${i}`;
      const li = document.createElement('li'); li.className = `toc-${h.tagName.toLowerCase()}`;
      const a = document.createElement('a'); a.href = `#${h.id}`; a.textContent = h.textContent;
      li.append(a); ol.append(li);
    });
    document.getElementById('toc').prepend(ol);
    const tocTitle = document.querySelector('#toc h2');
    document.getElementById('toc').prepend(tocTitle);
    const spy = new IntersectionObserver(entries => {
      for (const e of entries) if (e.isIntersecting) for (const a of ol.querySelectorAll('a')) a.classList.toggle('active', a.hash === `#${e.target.id}`);
    }, { rootMargin: '-10% 0px -75% 0px' });
    heads.forEach(h => spy.observe(h));
    content.querySelectorAll('pre').forEach(pre => {
      const button = document.createElement('button'); button.className = 'copy-btn'; button.textContent = '复制代码'; button.type = 'button';
      button.addEventListener('click', async () => {
        try { await navigator.clipboard.writeText(pre.querySelector('code').textContent); button.textContent = '已复制'; }
        catch { button.textContent = '请手动选择复制'; }
      });
      pre.append(button);
    });
    safeSave('ray-guide-last-v1', {id: chapter, title: content.querySelector('h1').textContent});
    const button = document.getElementById('mark-read');
    const updateRead = () => {button.textContent = read.includes(chapter) ? '取消已读标记' : '标记为已读'; document.getElementById('read-status').textContent = read.includes(chapter) ? '本章已读 · 进度保存在当前浏览器' : '';};
    updateRead();
    button.addEventListener('click', () => {
      read = read.includes(chapter) ? read.filter(x => x !== chapter) : [...read, chapter];
      const saved = safeSave('ray-guide-read-v1', read); updateRead();
      if (!saved) document.getElementById('read-status').textContent = '当前浏览器不允许持久保存，本次标记仅在此页面有效';
    });
    if (location.hash) requestAnimationFrame(() => document.getElementById(decodeURIComponent(location.hash.slice(1)))?.scrollIntoView());
  } else {
    const cards = [...document.querySelectorAll('.card')];
    cards.forEach(c => c.classList.toggle('is-read', read.includes(c.dataset.id)));
    document.getElementById('progress').textContent = `${cards.filter(c => read.includes(c.dataset.id)).length} / ${cards.length} 篇已读`;
    const last = safeLoad('ray-guide-last-v1', null);
    if (last && cards.some(c => c.dataset.id === last.id)) {
      const resume = document.getElementById('resume'); resume.href = `chapters/${last.id}.html`; resume.hidden = false;
    }
    document.getElementById('search').addEventListener('input', e => {
      const query = e.target.value.trim().toLowerCase();
      cards.forEach(c => {c.hidden = !c.textContent.toLowerCase().includes(query);});
      document.querySelectorAll('.course-part').forEach(s => {s.hidden = ![...s.querySelectorAll('.card')].some(c => !c.hidden);});
      document.getElementById('no-results').hidden = cards.some(c => !c.hidden);
    });
  }
})();
