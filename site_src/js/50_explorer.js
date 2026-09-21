/* Интерактивная часть: атлас с масштабом и слоями, паспорт МО с сопоставимыми территориями, карточки типов, метод и данные. */
const Explorer = (function () {
  const LAYERS = [['macro', 'Макротипы'], ['sub', 'Подтипы'], ['beh', 'Поведение'], ['wave', 'Прирост маркетплейсов'], ['level', 'Уровень трат']];
  let layer = 'macro', month = 23, hidden = new Set(), atlas, zoomT = d3.zoomIdentity, fitT;
  const levelScale = d3.scaleSequential(d3.interpolateBlues).domain([Math.log(10000), Math.log(80000)]);
  const typeMedianRel = (() => { const cache = {}; return (lens, k) => { const key = lens + k; if (!cache[key]) { const rows = MO.filter(m => modalOf(m, lens) === k); cache[key] = MONTHS.map((_, t) => d3.median(rows, m => m.rl[t])); } return cache[key]; }; })();
  const typeMedianShares = (() => { const cache = {}; return (lens, k) => { const key = lens + k; if (!cache[key]) { const rows = MO.filter(m => modalOf(m, lens) === k && m.sh); cache[key] = SH.map((_, j) => d3.median(rows, m => m.sh[j])); } return cache[key]; }; })();

  /* ---------- атлас ---------- */
  function fillFn(i) {
    const m = MO[i];
    if (layer === 'wave') return m.dmp == null ? null : dmpScale(m.dmp);
    if (layer === 'level') return m.t24 ? levelScale(Math.log(m.t24)) : null;
    const t = typeAt(m, layer, month); if (t == null || isNaN(t) || hidden.has(t)) return null; return colorOf[layer](t);
  }
  function drawAtlas() {
    const t = fitT; atlas.transform = { k: t.k * zoomT.k, x: t.x * zoomT.k + zoomT.x, y: t.y * zoomT.k + zoomT.y }; atlas.pickDirty = true;
    const dark = isDark(); atlas.draw(i => fillFn(i) || (dark ? '#26272d' : '#e8e7e2'), { stroke: dark ? 'rgba(16,17,20,0.85)' : 'rgba(255,255,255,0.85)', lineWidth: 0.5 });
    legend();
  }
  function legend() {
    const el = document.getElementById('atlasLegend');
    if (layer === 'wave') { el.innerHTML = legendDmp().map(l => `<span class="chip"><i style="background:${l.color}"></i>${l.label}</span>`).join(''); return; }
    if (layer === 'level') { el.innerHTML = [10000, 20000, 40000, 80000].map(v => `<span class="chip"><i style="background:${levelScale(Math.log(v))}"></i>${fmt.kilo(v)} ₽</span>`).join(''); return; }
    const counts = {}; MO.forEach(m => { const t = typeAt(m, layer, month); counts[t] = (counts[t] || 0) + 1; });
    el.innerHTML = T[layer].names.map((n, i) => `<button type="button" class="chip${hidden.has(i) ? ' off' : ''}" data-i="${i}" style="cursor:pointer${hidden.has(i) ? ';opacity:.45' : ''}"><i style="background:${colorOf[layer](i)}"></i>${esc(n)} <span class="muted num">${counts[i] || 0}</span></button>`).join('');
    el.querySelectorAll('button').forEach(b => b.onclick = () => { const i = +b.dataset.i; hidden.has(i) ? hidden.delete(i) : hidden.add(i); drawAtlas(); });
  }
  function toolbar() {
    const tb = document.getElementById('atlasToolbar');
    tb.innerHTML = `<div class="seg" id="layerSeg">${LAYERS.map(([k, l]) => `<button type="button" data-l="${k}" class="${k === layer ? 'on' : ''}">${l}</button>`).join('')}</div>
      <label id="monthCtl">Месяц: <b id="atlasMonth" class="num">${fmt.month(MONTHS[month])}</b> <input type="range" id="atlasSlider" min="0" max="23" value="${month}" style="width:200px;vertical-align:middle" aria-label="Месяц"></label>`;
    tb.querySelectorAll('#layerSeg button').forEach(b => b.onclick = () => { layer = b.dataset.l; hidden.clear(); tb.querySelectorAll('#layerSeg button').forEach(x => x.classList.toggle('on', x === b)); document.getElementById('monthCtl').style.visibility = (layer === 'wave' || layer === 'level') ? 'hidden' : 'visible'; drawAtlas(); });
    document.getElementById('atlasSlider').oninput = e => { month = +e.target.value; document.getElementById('atlasMonth').textContent = fmt.month(MONTHS[month]); drawAtlas(); };
  }
  function initAtlas() {
    const box = document.getElementById('atlasBox'); const canvas = document.getElementById('atlasCanvas'); atlas = new MapLayer(canvas);
    const resize = () => { const w = box.clientWidth, h = box.clientHeight; if (!w || !h) return; atlas.resize(w, h); fitT = atlas.fit(DATA.frame.w, DATA.frame.h, 10); drawAtlas(); };
    resize(); new ResizeObserver(resize).observe(box);
    // колесо без Ctrl/⌘ прокручивает страницу, а не карту (щипок на трекпаде приходит как wheel с ctrlKey — работает как масштаб)
    const zoom = d3.zoom().scaleExtent([1, 9]).filter(e => e.type !== 'wheel' || e.ctrlKey || e.metaKey).on('zoom', e => { zoomT = e.transform; drawAtlas(); });
    d3.select(canvas).call(zoom).on('dblclick.zoom', null);
    document.getElementById('zoomIn').onclick = () => d3.select(canvas).transition().duration(300).call(zoom.scaleBy, 1.6);
    document.getElementById('zoomOut').onclick = () => d3.select(canvas).transition().duration(300).call(zoom.scaleBy, 1 / 1.6);
    document.getElementById('zoomReset').onclick = () => d3.select(canvas).transition().duration(400).call(zoom.transform, d3.zoomIdentity);
    let hov = -1;
    canvas.addEventListener('pointermove', e => { const r = canvas.getBoundingClientRect(); const i = atlas.hit(e.clientX - r.left, e.clientY - r.top); if (i !== hov) { hov = i; canvas.style.cursor = i >= 0 ? 'pointer' : 'grab'; } if (i >= 0) { const m = MO[i]; const lens = (layer in colorOf) ? layer : 'macro'; let extra = ''; if (layer === 'wave') extra = `<div class="row"><span>маркетплейсы</span><span>${fmt.pct(m.mp23, 1)} → ${fmt.pct(m.mp24, 1)}</span></div>`; Tip.show(e.clientX, e.clientY, moTip(m, lens, (layer in colorOf) ? month : null) + extra); } else Tip.hide(); });
    canvas.addEventListener('pointerleave', () => Tip.hide());
    canvas.addEventListener('click', e => { const r = canvas.getBoundingClientRect(); const i = atlas.hit(e.clientX - r.left, e.clientY - r.top); if (i >= 0) { openPassport(MO[i].id); document.getElementById('passport').scrollIntoView({ behavior: 'smooth', block: 'start' }); } });
    window.addEventListener('themechange', drawAtlas);
    toolbar();
    document.getElementById('atlasNote').textContent = 'Масштаб — кнопки, щипок на трекпаде или Ctrl + колесо; перетаскивание — панорама. Слои типов считаются для каждого месяца; прирост маркетплейсов и уровень трат — по годовым средним. Нажатие на МО открывает паспорт.';
  }

  /* ---------- паспорт ---------- */
  /* последовательность типов по месяцам → отрезки одинакового типа */
  function runs(seq) { const out = []; let t0 = 0; for (let t = 1; t <= seq.length; t++) { if (t === seq.length || seq[t] !== seq[t0]) { out.push({ k: +seq[t0], a: t0, b: t - 1 }); t0 = t; } } return out; }
  /* лента «тип по месяцам»: подпись линзы, отрезки по длительности, словесная расшифровка справа */
  function timeline(m) {
    const rows = [['macro', 'Макротип', m.ms], ['sub', 'Подтип', m.ss], ['beh', 'Поведение', m.bs]].filter(r => r[2]);
    const body = rows.map(([lens, label, seq]) => {
      const rs = runs(seq);
      const sum = rs.length === 1
        ? `${esc(nameOf[lens](rs[0].k))} <span class="muted">— все 24 месяца</span>`
        : rs.map(r => `${esc(nameOf[lens](r.k))} <span class="muted">(${r.a === r.b ? fmt.month(MONTHS[r.a]) : fmt.month(MONTHS[r.a]) + ' – ' + fmt.month(MONTHS[r.b])})</span>`).join(' <span class="muted">→</span> ');
      const bars = rs.map(r => `<i style="flex:${r.b - r.a + 1};background:${colorOf[lens](r.k)}" title="${esc(nameOf[lens](r.k))}: ${fmt.month(MONTHS[r.a])}${r.a === r.b ? '' : ' – ' + fmt.month(MONTHS[r.b])}"></i>`).join('');
      return `<div class="tl-row"><span class="tl-lab small">${label}</span><div class="strip">${bars}</div><span class="tl-sum small">${sum}</span></div>`;
    }).join('');
    return `<div class="timeline"><div class="tl-head"><b class="small">Тип по месяцам</b><span class="small muted">полоса — 24 месяца 2023–2024; цвет — тип, стык — смена типа</span></div>${body}`
      + `<div class="tl-axis small muted"><span>${fmt.month(MONTHS[0])}</span><span>${fmt.month(MONTHS[MONTHS.length - 1])}</span></div></div>`;
  }
  function passportHtml(m, slot) {
    const rank = (() => { const rows = MO.filter(x => x.su === m.su && x.t24).map(x => x.t24).sort((a, b) => a - b); return rows.findIndex(v => v >= m.t24) / rows.length; })();
    const dev = m.mk ? FEAT.map((f, i) => [f, m.mk[i]]).filter(d => d[1] != null).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])).slice(0, 8) : [];
    const peers = (m.pe || []).map(id => MO[byId.get(id)]).filter(Boolean);
    return `<div class="head"><h3>${esc(m.n)}</h3><span class="small">${esc(m.r)} · ${esc(m.t)}</span></div>
      <div class="toolbar" style="margin:8px 0 4px"><span class="badge" style="background:${MACRO_C[m.ma]}">${esc(nameOf.macro(m.ma))}</span><span class="badge" style="background:${SUB_C[m.su]}">${esc(nameOf.sub(m.su))}</span>${m.be != null ? `<span class="badge" style="background:${BEH_C[m.be]}">${esc(nameOf.beh(m.be))}</span>` : ''}</div>
      ${timeline(m)}
      <div class="kpis"><div class="kpi"><b>${fmt.rub(m.t24)}</b><span>траты на жителя в месяц, 2024 (2023: ${fmt.rub(m.t23)})</span></div><div class="kpi"><b>${fmt.pct(rank)}</b><span>перцентиль трат внутри подтипа</span></div><div class="kpi"><b>${fmt.int(m.pop)}</b><span>население</span></div><div class="kpi"><b>${fmt.rub(m.wage)}</b><span>средняя зарплата</span></div><div class="kpi"><b>${fmt.num(m.acc, 0)}</b><span>доступность рынков (0–1000)</span></div><div class="kpi"><b>${fmt.pct(m.mp23, 1)} → ${fmt.pct(m.mp24, 1)}</b><span>доля маркетплейсов, 2023 → 2024</span></div></div>
      <div class="grid2"><div><b class="small">Относительный уровень трат против медианы подтипа</b><svg class="pp-level" data-slot="${slot}" style="width:100%;height:190px"></svg></div><div><b class="small">Структура трат против медианы подтипа</b><svg class="pp-shares" data-slot="${slot}" style="width:100%;height:190px"></svg></div></div>
      <div class="grid2" style="margin-top:12px"><div><b class="small">Отклонения от среднего по России</b><table>${dev.map(([f, v]) => `<tr><td>${esc(SHORT[f] || f)}</td><td class="num">${fmt.dev(v)}</td></tr>`).join('')}</table></div>
      <div><b class="small">Сопоставимые МО (тот же подтип, ближайшие по профилю)</b>${peers.map(p => `<div><a href="#" class="peer" data-id="${p.id}">${esc(p.n)}</a> <span class="small">${esc(p.r)}</span></div>`).join('') || '<div class="small">—</div>'}<div class="small" style="margin-top:8px">Нажатие на сопоставимое МО открывает его паспорт.</div></div></div>`;
  }
  function drawPassportCharts(el, m) {
    const lvl = d3.select(el.querySelector('svg.pp-level')); const W = el.querySelector('svg.pp-level').clientWidth || 400, H = 190;
    lvl.attr('viewBox', `0 0 ${W} ${H}`); const med = typeMedianRel('sub', m.su);
    Charts.lines(lvl, { x: 0, y: 0, w: W, h: H, series: [m.rl, med], xs: [0, 23], ys: [Math.min(d3.min(m.rl), d3.min(med)) - 0.05, Math.max(d3.max(m.rl), d3.max(med)) + 0.05], colors: [MACRO_C[m.ma], isDark() ? '#6a6b72' : '#b9b8b3'], labels: [shortName(m), 'медиана подтипа'], upto: 23, fmtY: v => fmt.num(v, 1), mr: 120 });
    const sh = d3.select(el.querySelector('svg.pp-shares')); sh.attr('viewBox', `0 0 ${W} ${H}`); if (!m.sh) return;
    const med2 = typeMedianShares('sub', m.su); const { g, iw, ih } = Charts.frame(sh, { x: 0, y: 0, w: W, h: H, ml: 40, mr: 8, mt: 12, mb: 34 });
    const xs = d3.scaleBand().domain(SH).range([0, iw]).padding(0.3), ys = d3.scaleLinear().domain([0, Math.max(d3.max(m.sh), d3.max(med2)) * 1.1]).range([ih, 0]).nice();
    Charts.axisY(g, ys, iw, { fmt: v => Math.round(v * 100) + '%' });
    SH.forEach((k, j) => { const bw = xs.bandwidth() / 2; g.append('rect').attr('x', xs(k)).attr('y', ys(m.sh[j])).attr('width', bw - 1).attr('height', ih - ys(m.sh[j])).attr('rx', 3).attr('fill', MACRO_C[m.ma]); g.append('rect').attr('x', xs(k) + bw).attr('y', ys(med2[j])).attr('width', bw - 1).attr('height', ih - ys(med2[j])).attr('rx', 3).attr('fill', isDark() ? '#6a6b72' : '#c9c8c3'); g.append('text').attr('x', xs(k) + xs.bandwidth() / 2).attr('y', ih + 16).attr('text-anchor', 'middle').attr('font-size', 11).attr('fill', cssVar('--muted')).text(k); });
  }
  const slots = {};
  function openPassport(id) {
    const m = MO[byId.get(id)]; if (!m) return; slots[1] = id; const el = document.getElementById('pp1');
    el.innerHTML = passportHtml(m, 1); drawPassportCharts(el, m);
    el.querySelectorAll('a.peer').forEach(a => a.onclick = ev => { ev.preventDefault(); openPassport(+a.dataset.id); el.scrollIntoView({ behavior: REDUCED ? 'auto' : 'smooth', block: 'start' }); });
  }
  window.openPassport = openPassport;
  /* поиск: нормализация (ё → е, регистр), совпадения в названии и регионе, города вперёд и крупные выше */
  const norm = s => String(s).toLowerCase().replace(/ё/g, 'е');
  const IDX = MO.map((m, i) => ({ i, id: m.id, n: m.n, r: m.r, nn: norm(m.n), ns: norm(m.s || m.n), nr: norm(m.r), pop: m.pop || 0 }));
  function search(q, limit = 8) {
    const nq = norm(q).trim(); if (nq.length < 2) return [];
    const out = [];
    const startsWord = (str, p) => p === 0 || !/[а-яa-z0-9]/.test(str[p - 1]);
    for (const e of IDX) {
      // сначала короткое имя («Ярославль»), потом полное («городской округ город …»), потом регион
      const ps = e.ns.indexOf(nq);
      if (ps >= 0) { out.push({ e, score: (startsWord(e.ns, ps) ? 0 : 600) + ps }); continue; }
      const p = e.nn.indexOf(nq);
      if (p >= 0) { out.push({ e, score: 2000 + (startsWord(e.nn, p) ? 0 : 600) + p }); continue; }
      const pr = e.nr.indexOf(nq);
      if (pr >= 0) out.push({ e, score: 5000 + pr });
    }
    out.sort((a, b) => a.score - b.score || b.e.pop - a.e.pop);
    return out.slice(0, limit);
  }
  const mark = (text, nq) => { const p = norm(text).indexOf(nq); if (p < 0 || !nq) return esc(text); return esc(text.slice(0, p)) + '<mark>' + esc(text.slice(p, p + nq.length)) + '</mark>' + esc(text.slice(p + nq.length)); };
  function initPassport() {
    const inp = document.getElementById('search'), box = document.getElementById('suggest'), found = document.getElementById('found');
    let hits = [], cur = -1;
    const close = () => { box.hidden = true; inp.setAttribute('aria-expanded', 'false'); cur = -1; };
    const render = () => {
      const nq = norm(inp.value).trim();
      hits = search(inp.value);
      if (nq.length < 2) { close(); found.textContent = ''; return; }
      if (!hits.length) { box.innerHTML = '<li class="empty">Ничего не найдено</li>'; box.hidden = false; inp.setAttribute('aria-expanded', 'true'); found.textContent = ''; return; }
      box.innerHTML = hits.map((h, j) => `<li role="option" data-j="${j}" aria-selected="${j === cur}" class="${j === cur ? 'on' : ''}"><span class="nm">${mark(h.e.n, nq)}</span><span class="rg">${mark(h.e.r, nq)}</span></li>`).join('');
      box.hidden = false; inp.setAttribute('aria-expanded', 'true');
      const total = search(inp.value, 1e9).length;
      found.textContent = total > hits.length ? `Показаны ${hits.length} из ${total}` : '';
    };
    const pick = j => { const h = hits[j]; if (!h) return; inp.value = h.e.n; close(); found.textContent = ''; openPassport(h.e.id); };
    const move = d => { if (box.hidden || !hits.length) return; cur = (cur + d + hits.length) % hits.length; [...box.children].forEach((li, j) => { li.classList.toggle('on', j === cur); li.setAttribute('aria-selected', j === cur); }); box.children[cur].scrollIntoView({ block: 'nearest' }); };
    inp.addEventListener('input', () => { cur = -1; render(); });
    inp.addEventListener('focus', () => { if (inp.value.trim().length >= 2) render(); });
    inp.addEventListener('keydown', e => {
      if (e.key === 'ArrowDown') { e.preventDefault(); move(1); }
      else if (e.key === 'ArrowUp') { e.preventDefault(); move(-1); }
      else if (e.key === 'Enter') { e.preventDefault(); pick(cur >= 0 ? cur : 0); }
      else if (e.key === 'Escape') { close(); }
    });
    box.addEventListener('mousedown', e => { const li = e.target.closest('li[data-j]'); if (li) { e.preventDefault(); pick(+li.dataset.j); } });
    document.addEventListener('click', e => { if (!e.target.closest('#searchBox')) close(); });
    const nsk = findMO('Новосибирск'); openPassport(MO[nsk >= 0 ? nsk : 0].id);
  }

  /* ---------- карточки типов ---------- */
  function typeCards(lens) {
    const box = document.getElementById('typeCards'); const K = T[lens].names.length; const rules = (DATA.interp[lens === 'macro' ? 'K4_windowlog' : lens === 'sub' ? 'K6_windowlog' : 'K5_beh_windowlog'] || {}).rules_raw || {};
    box.innerHTML = d3.range(K).map(k => { const ex = MO.filter(m => modalOf(m, lens) === k).sort((a, b) => (b.pop || 0) - (a.pop || 0)).slice(0, 4).map(shortName).join(', '); return `<div class="card typecard" style="border-top-color:${colorOf[lens](k)}"><h3>${esc(nameOf[lens](k))}</h3><div class="small">${T[lens].sizes[k]} МО · ожидаемое пребывание ${Math.round(T[lens].sojourn[k])} мес.</div><div class="fingerprint"><svg data-k="${k}" style="width:100%;height:230px"></svg></div>${rules[k] ? `<div class="small" style="margin-top:6px"><b>Правило:</b> ${rules[k].map(esc).join(' <span class="muted">и</span> ')}</div>` : ''}<div class="small" style="margin-top:6px">Например: ${esc(ex)}</div></div>`; }).join('');
    box.querySelectorAll('svg').forEach(svgEl => { const k = +svgEl.dataset.k; const W = svgEl.clientWidth || 340; const s = d3.select(svgEl).attr('viewBox', `0 0 ${W} 230`); fingerprint(s, null, lens, k, { x: 12, y: 32, w: W - 70, title: false }); });
  }
  function initTypes() { typeCards('macro'); document.querySelectorAll('#typesSeg button').forEach(b => b.onclick = () => { document.querySelectorAll('#typesSeg button').forEach(x => x.classList.toggle('on', x === b)); typeCards(b.dataset.l); }); window.addEventListener('themechange', () => typeCards(document.querySelector('#typesSeg button.on').dataset.l)); }

  /* ---------- метод и данные ---------- */
  function initMethod() {
    const q = checks; const tmp = q.temporal.find(r => r.method === 'kefrin_t_affect_fe' && r.K === 4) || {};
    const cards = [
      ['Данные', `<p>СберИндекс: безналичные расходы жителей по шести категориям для 2 190 МО (2 016 с полной историей за 24 месяца), индекс доступности рынков, автодорожные связи, справочник границ МО (CC BY-SA 4.0). Росстат БД ПМО через «Если быть точным» (CC BY 4.0): население, зарплата, занятость по разделам ОКВЭД2. В панели нет 12 регионов, включая Белгородскую, Курскую, Воронежскую и Ростовскую области, Краснодарский край, Крым и Дагестан.</p>`],
      ['Признаки и сеть', `<p>По месяцам: CLR-структура шести категорий и относительный уровень трат; контекст — 10 признаков Росстата и доступности. Сеть — граф, обученный из гладких сигналов (Kalofolias, 2016) на каждый месяц, средняя степень 15; сравнены 8 правил ребра и 5 разрежений (36 сетей): косинус профиля, корреляции и лаговые корреляции приростов, DTW, дорожная гравитация, SNF.</p>`],
      ['Метод: KEFRiN-T', `<p>KEFRiN (Shalileh & Mirkin, 2022) — K-means для сетей с признаками — расширен во времени: сглаживание срезов с весом истории по shrinkage-оценке AFFECT, фиксированные эффекты узлов, тест инновации (z > 2,5) и второй проход от модального разбиения. Проверка на синтетических динамических сетях с известной истиной; сравнение с 12 методами (K-means, Ward, GMM, спектральная, Leiden двух разрешений, iLouvain, EVA, CANUS, DMoN, KEFRiN на признаках и на сети).</p>`],
      ['Качество', `<p>K = 4: AMI соседних месяцев ${fmt.num(tmp.ami_median, 3)}, смена типа ${fmt.pct(tmp.switch_median, 1)} в месяц, SW ${fmt.num(tmp.SW)}, AVI ${fmt.num(tmp.AVI)}, AVU ${fmt.num(tmp.AVU)}, ANUI ${fmt.num(tmp.ANUI)}, MQ ${fmt.num(tmp.MQ)}, модульность ${fmt.num(tmp.Q)}; bootstrap-ARI ${fmt.num(q.stability.find(r => r.K === 4).ari_mean)}. Реализация AVU/AVI/ANUI/модульности сверена с библиотекой Pattern.</p>`],
      ['Воспроизводимость', `<p>Один автономный файл, никаких серверов. Пайплайн: <code>make env</code>, <code>scripts/download_data.py</code>, <code>make all</code> (около 40 минут). Два контрольных прогона с нуля воспроизвели итоговые метки побайтно. Код, конфигурации и инструкция запуска — <a href="https://github.com/IvanFrolovskiy/sberindex-atlas" target="_blank" rel="noopener">репозиторий на GitHub</a>. Отчёт: <a href="methodology.pdf">методологический отчёт (PDF)</a>. <a href="#" id="csvBtn">Скачать типологию (CSV)</a>.</p>`],
      ['Критерии жюри → где смотреть', `<table><tr><td>Методология</td><td>схема выше, главы 2–3, отчёт §3, §5</td></tr><tr><td>Сеть и атрибуты</td><td>глава 2, отчёт §4</td></tr><tr><td>Сравнение методов</td><td>глава 7, отчёт §5.3–5.5</td></tr><tr><td>ICVI</td><td>карточка «Качество», отчёт §10</td></tr><tr><td>Интерпретация</td><td>главы 3–6, паспорт МО, отчёт §6–9</td></tr><tr><td>Визуализация</td><td>эта страница</td></tr></table>`],
    ];
    document.getElementById('methodCards').innerHTML = cards.map(([t, b]) => `<div class="card"><h3>${t}</h3>${b}</div>`).join('');
    document.getElementById('csvBtn').onclick = e => { e.preventDefault(); const head = ['territory_id', 'name', 'region', 'macro_type', 'sub_type', 'behaviour_type', 'macro_seq', 'switches_macro', 'total_2024', 'population', 'wage', 'market_access', 'mp_share_2023', 'mp_share_2024']; const rows = MO.map(m => [m.id, m.n, m.r, nameOf.macro(m.ma), nameOf.sub(m.su), m.be != null ? nameOf.beh(m.be) : '', m.ms, m.sw[0], m.t24, m.pop, m.wage, m.acc, m.mp23, m.mp24].map(v => `"${String(v ?? '').replace(/"/g, '""')}"`).join(',')); const blob = new Blob(['﻿' + head.join(',') + '\n' + rows.join('\n')], { type: 'text/csv;charset=utf-8' }); const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = 'mo_typology.csv'; a.click(); };
    document.getElementById('footer').innerHTML = `Типы локальных экономик России по данным СберИндекса · онлайн-конкурс СберИндекса 2026, направление «Кластеризация» · Иван Фроловский. Данные: СберИндекс (CC BY-SA 4.0), Росстат БД ПМО через «Если быть точным» (CC BY 4.0). Код — MIT, <a href="https://github.com/IvanFrolovskiy/sberindex-atlas" target="_blank" rel="noopener">github.com/IvanFrolovskiy/sberindex-atlas</a>. Значение расходов — оценка средних безналичных трат жителя МО в месяц.`;
  }

  function init() { initAtlas(); initPassport(); initTypes(); initMethod();
    // смена темы: палитра типов перекрашивается, перерисовать карточки типов и паспорта
    window.addEventListener('themechange', () => { const b = document.querySelector('#typesSeg button.on'); if (b && b.onclick) b.onclick(); if (slots[1] != null) openPassport(slots[1]); });
  }
  return { init, openPassport };
})();
