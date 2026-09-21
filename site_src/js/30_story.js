/* История: главы и шаги. Каждый шаг — текст + функция вида для сцены. */
const SHORT = { 'занятость: сельское хозяйство': 'занятость: с/х', 'занятость: промышленность': 'занятость: пром-сть', 'занятость: рыночные услуги': 'занятость: услуги', 'занятость: бюджетный сектор': 'занятость: бюджет', 'доля городского населения': 'доля горожан', 'траты на жителя': 'траты на жителя', 'доступность рынков': 'доступность рынков' };
const FEAT = T.feature_order; const SH = T.share_order; const iCAFE = SH.indexOf('общепит'), iMP = SH.indexOf('маркетплейсы');
const findMO = q => MO.findIndex(m => m.n.includes(q));
const legendMacro = () => T.macro.names.map((n, i) => ({ color: MACRO_C[i], label: n, count: T.macro.sizes[i] }));
const legendBeh = () => T.beh.names.map((n, i) => ({ color: BEH_C[i], label: n, count: T.beh.sizes[i] }));
const adjacency = (() => { const a = Array.from({ length: N }, () => []); for (const [i, j, w] of DATA.graph.edges) { a[i].push([j, w]); a[j].push([i, w]); } return a; })();
const neighbours = i => new Set(adjacency[i].map(e => e[0]));

/* ---------- SVG-помощники (экранные координаты) ---------- */
function swarmAxis(svg, stage, { label, lanes = null, ticks = [15000, 25000, 40000, 70000] }) {
  const xs = stage.swarmScale; if (!xs) return; const y0 = stage.map.toScreen(0, FRAME.h - 28)[1];
  const g = svg.append('g').attr('class', 'axis').attr('font-family', 'inherit').attr('font-size', 12).attr('fill', cssVar('--muted'));
  g.append('line').attr('x1', stage.map.toScreen(60, 0)[0]).attr('x2', stage.map.toScreen(940, 0)[0]).attr('y1', y0).attr('y2', y0).attr('stroke', cssVar('--line-2'));
  ticks.forEach(v => { const x = stage.map.toScreen(xs(Math.log(v)), 0)[0]; g.append('line').attr('x1', x).attr('x2', x).attr('y1', y0).attr('y2', y0 + 5).attr('stroke', cssVar('--line-2')); g.append('text').attr('x', x).attr('y', y0 + 19).attr('text-anchor', 'middle').text(fmt.kilo(v)); });
  g.append('text').attr('x', stage.map.toScreen(940, 0)[0]).attr('y', y0 + 36).attr('text-anchor', 'end').text(label);
  if (lanes) lanes.forEach((name, gi) => { const laneH = (FRAME.h - 80) / lanes.length; const y = stage.map.toScreen(0, 40 + laneH * (gi + 0.5))[1]; svg.append('text').attr('x', stage.map.toScreen(62, 0)[0]).attr('y', y - laneH * stage.t.k * 0.36).attr('font-size', 12).attr('font-weight', 600).attr('fill', cssVar('--ink-2')).text(name); });
}
function annotate(svg, x, y, text, { dx = 0, dy = -14, anchor = 'middle', color = null } = {}) {
  const g = svg.append('g'); g.append('line').attr('x1', x).attr('y1', y).attr('x2', x + dx).attr('y2', y + dy + 4).attr('stroke', color || cssVar('--ink-2')).attr('stroke-width', 1);
  g.append('text').attr('x', x + dx).attr('y', y + dy).attr('text-anchor', anchor).attr('font-size', 12.5).attr('font-weight', 600).attr('fill', cssVar('--ink')).text(text);
}
/* «отпечаток» типа: самые отклоняющиеся признаки по правилу Миркина (8 на широком экране, 6 на узком) */
function fingerprint(svg, stage, lens, k, { x, y, w = 300, title = true, top = 8 } = {}) {
  const prof = T[lens].profiles.find(p => p.cluster === k); if (!prof) return;
  const items = FEAT.map((f, i) => [f, prof.v[i]]).filter(d => d[1] != null).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])).slice(0, top);
  const maxAbs = Math.max(...items.map(d => Math.abs(d[1])));
  const labW = Math.max(92, Math.min(132, w * 0.4)), valW = 46, barW = Math.max(60, w - labW - 8 - valW);
  const xs = d3.scaleLinear().domain([-maxAbs, maxAbs]).range([0, barW]);
  const g = svg.append('g').attr('transform', `translate(${x},${y})`).attr('font-family', 'inherit');
  g.append('rect').attr('x', -12).attr('y', -30).attr('width', w + 24).attr('height', items.length * 22 + 44).attr('rx', 12).attr('fill', cssVar('--card')).attr('stroke', cssVar('--line'));
  if (title) g.append('text').attr('x', 0).attr('y', -10).attr('font-size', 12.5).attr('font-weight', 650).attr('fill', cssVar('--ink')).text(`${nameOf[lens](k)} · ${prof.size} МО`);
  const col = colorOf[lens](k); const x0b = labW + 8; const zx = x0b + xs(0);
  g.append('line').attr('x1', zx).attr('x2', zx).attr('y1', 0).attr('y2', items.length * 22).attr('stroke', cssVar('--line-2'));
  items.forEach(([f, v], i) => {
    const yy = i * 22 + 4; const x0 = x0b + Math.min(xs(0), xs(v)), ww = Math.abs(xs(v) - xs(0));
    g.append('text').attr('x', labW).attr('y', yy + 10).attr('text-anchor', 'end').attr('font-size', 11.5).attr('fill', cssVar('--ink-2')).text(SHORT[f] || f);
    g.append('rect').attr('x', x0).attr('y', yy).attr('width', Math.max(1, ww)).attr('height', 13).attr('rx', 3).attr('fill', v >= 0 ? col : (isDark() ? '#5a5b62' : '#c9c8c3'));
    g.append('text').attr('x', v >= 0 ? x0 + ww + 5 : zx + 5).attr('y', yy + 10.5).attr('text-anchor', 'start').attr('font-size', 11.5).attr('font-weight', 600).attr('fill', cssVar('--ink')).text(fmt.dev(v));
  });
  Charts.growX(g, 'rect:not(:first-child)', zx, 520, 45);
}
/* подпись строки малых компонент и изолятов под сетевой раскладкой */
function isolatesNote(svg, st) {
  const G = DATA.graph; if (!G.n_small) return;
  const [x, y] = st.map.toScreen(G.w / 2, G.h * 0.945 + 12 + MAP_DY);
  svg.append('text').attr('x', x).attr('y', y + 8).attr('text-anchor', 'middle').attr('font-size', 11).attr('fill', cssVar('--muted')).text(`отдельные компоненты и изоляты · ${G.n_small} МО`);
}
/* компактная версия дерева порогов: три правила подряд (узкие экраны) */
function immList(svg, stage) {
  const W = stage.w, y0 = stage.top + 24, rowH = 62, x = 16, w = W - 32;
  const rows = [['доля общепита больше 6%', 'Столичные агломерации', 2], ['иначе: зарплата больше 93 тыс. ₽', 'Северные и ресурсные', 1],
                ['иначе: траты больше 25,9 тыс. ₽ в месяц', 'Крупные и средние города', 0], ['иначе', 'Сельские и малые', 3]];
  const g = svg.append('g').attr('font-family', 'inherit');
  rows.forEach(([q, name, k], i) => {
    const y = y0 + i * rowH;
    g.append('rect').attr('x', x).attr('y', y).attr('width', w).attr('height', rowH - 10).attr('rx', 12).attr('fill', cssVar('--card')).attr('stroke', cssVar('--line'));
    g.append('rect').attr('x', x).attr('y', y).attr('width', 4).attr('height', rowH - 10).attr('fill', MACRO_C[k]);
    g.append('text').attr('x', x + 16).attr('y', y + 21).attr('font-size', 12).attr('fill', cssVar('--ink-2')).text(q);
    g.append('text').attr('x', x + 16).attr('y', y + 39).attr('font-size', 13).attr('font-weight', 650).attr('fill', cssVar('--ink')).text('→ ' + name);
  });
  Charts.fadeUp(g, 'rect', 10, 460, 80);
  const acc = DATA.interp.K4_windowlog.imm_raw_acc;
  g.append('text').attr('x', W / 2).attr('y', y0 + rows.length * rowH + 16).attr('text-anchor', 'middle').attr('font-size', 11.5).attr('fill', cssVar('--muted')).text(`Три порога объясняют тип ${fmt.pct(acc)} муниципалитетов`);
}
/* дерево порогов IMM для макротипов */
function immTree(svg, stage) {
  const W = stage.w, H = stage.h;
  if (W < 780) return immList(svg, stage); const g = svg.append('g').attr('font-family', 'inherit');
  const box = (x, y, w, h, text, fill, stroke, bold) => { g.append('rect').attr('x', x - w / 2).attr('y', y - h / 2).attr('width', w).attr('height', h).attr('rx', 10).attr('fill', fill).attr('stroke', stroke || 'none'); const lines = text.split('\n'); lines.forEach((ln, i) => g.append('text').attr('x', x).attr('y', y + (i - (lines.length - 1) / 2) * 15 + 5).attr('text-anchor', 'middle').attr('font-size', 12.5).attr('font-weight', bold ? 650 : 500).attr('fill', bold ? '#fff' : cssVar('--ink')).text(ln)); };
  const link = (x1, y1, x2, y2, lab) => { g.append('path').attr('d', `M${x1},${y1} C${(x1 + x2) / 2},${y1} ${(x1 + x2) / 2},${y2} ${x2},${y2}`).attr('fill', 'none').attr('stroke', cssVar('--line-2')).attr('stroke-width', 1.5); g.append('text').attr('x', (x1 + x2) / 2).attr('y', (y1 + y2) / 2 - 6).attr('text-anchor', 'middle').attr('font-size', 11.5).attr('fill', cssVar('--muted')).text(lab); };
  const cx = [W * 0.16, W * 0.42, W * 0.68, W * 0.9]; const qw = Math.min(200, W * 0.24), qh = 46, aw = Math.min(190, W * 0.2), ah = 40; const H0 = stage.top || 0;
  const q = cssVar('--card'), qs = cssVar('--line-2');
  box(cx[0], H * 0.5, qw, qh, 'доля общепита\nбольше 6%?', q, qs); link(cx[0] + qw / 2, H * 0.5, cx[3] - aw / 2, H0 + (H - H0) * 0.12, 'да'); box(cx[3], H0 + (H - H0) * 0.12, aw, ah, 'Столичные агломерации', MACRO_C[2], null, true);
  link(cx[0] + qw / 2, H * 0.5, cx[1] - qw / 2, H * 0.6, 'нет'); box(cx[1], H * 0.6, qw, qh, 'зарплата\nбольше 93 тыс. ₽?', q, qs);
  link(cx[1] + qw / 2, H * 0.6, cx[3] - aw / 2, H * 0.4, 'да'); box(cx[3], H * 0.4, aw, ah, 'Северные и ресурсные', MACRO_C[1], null, true);
  link(cx[1] + qw / 2, H * 0.6, cx[2] - qw / 2, H * 0.72, 'нет'); box(cx[2], H * 0.72, qw, qh, 'траты больше\n25,9 тыс. ₽ в месяц?', q, qs);
  link(cx[2] + qw / 2, H * 0.72, cx[3] - aw / 2, H * 0.64, 'да'); box(cx[3], H * 0.64, aw, ah, 'Крупные и средние города', MACRO_C[0], null, true);
  link(cx[2] + qw / 2, H * 0.72, cx[3] - aw / 2, H * 0.88, 'нет'); box(cx[3], H * 0.88, aw, ah, 'Сельские и малые', MACRO_C[3], null, true);
  Charts.fadeUp(g, 'rect', 10, 480, 70);
  const acc = DATA.interp.K4_windowlog.imm_raw_acc; g.append('text').attr('x', W / 2).attr('y', H * 0.94).attr('text-anchor', 'middle').attr('font-size', 12).attr('fill', cssVar('--muted')).text(`Три порога объясняют тип ${fmt.pct(acc)} муниципалитетов`);
}

/* ---------- главы ---------- */
const iAnchor = findMO('Нижневартовск'), iOrsk = findMO('город Орск');
const cafeScale = d3.scaleSequential(d3.interpolateBlues).domain([0.005, 0.09]);
const CHAPTERS = [
  { id: 'spend', title: 'Как тратят', steps: [
    { text: `<p>Каждая точка — муниципальное образование. Их <b class="num">2 016</b>: все, по которым СберИндекс публикует безналичные расходы жителей за каждый из 24 месяцев 2023–2024 годов.</p><p class="note">В панели нет 12 регионов: Белгородская, Брянская, Курская, Воронежская, Ростовская области, Краснодарский край, Крым, Севастополь, Дагестан, Чечня, Ингушетия, Бурятия — по ним данные не публикуются.</p>`,
      view: s => ({ title: 'Муниципальные образования России', sub: 'каждая точка — МО, положение — на карте', map: { mode: 'grey' }, dots: { layout: 'map', color: () => (isDark() ? '#9a9aa0' : '#6c6c72'), alpha: 0.85 }, legend: [] }) },
    { text: `<p>Первое измерение — <b>уровень</b>: сколько в среднем тратит житель в месяц. Разброс пятикратный: от 14 тысяч рублей в сельских округах Чувашии до 75 тысяч в центре Москвы и в Анадыре.</p>`,
      view: s => ({ title: 'Уровень трат на жителя, 2024', sub: 'логарифмическая шкала', map: { mode: 'none' }, dots: { layout: 'swarm', value: i => MO[i].t24 ? Math.log(MO[i].t24) : null, domain: [Math.log(13000), Math.log(85000)], color: () => (isDark() ? '#9a9aa0' : '#6c6c72') }, legend: [], svg: (svg, st) => { swarmAxis(svg, st, { label: 'траты на жителя в месяц, ₽' }); const m = d3.median(MO, d => d.t24); const x = st.map.toScreen(st.swarmScale(Math.log(m)), 0)[0]; const yTop = st.map.toScreen(0, 40 + (FRAME.h - 80) * 0.5)[1] - 235 * st.t.k; annotate(svg, x, yTop + 8, `медиана ${fmt.kilo(m)} ₽`, { dy: -14 }); } }) },
    { text: `<p>Второе измерение — <b>структура</b>: на что уходят деньги. Здесь показана доля общепита: от половины процента в сельских районах до 12% в центре крупных городов. Уровень и структура связаны, но не совпадают: дорогие северные города тратят много, а в рестораны ходят мало.</p>`,
      view: s => ({ title: 'Доля общепита в тратах, 2024', sub: 'темнее — больше', map: { mode: 'none' }, dots: { layout: 'swarm', value: i => MO[i].t24 ? Math.log(MO[i].t24) : null, domain: [Math.log(13000), Math.log(85000)], color: i => MO[i].sh ? cafeScale(MO[i].sh[iCAFE]) : GREY }, legend: [{ color: cafeScale(0.01), label: 'общепит 1%' }, { color: cafeScale(0.04), label: '4%' }, { color: cafeScale(0.08), label: '8% и больше' }], svg: (svg, st) => swarmAxis(svg, st, { label: 'траты на жителя в месяц, ₽' }) }) },
    { text: `<p>Если добавить к уровню и структуре трат контекст — население, зарплаты, занятость, доступность рынков — и сеть похожих территорий, муниципалитеты собираются в <b>четыре типа</b>. Дальше мы покажем, откуда берётся сеть и как устроен каждый тип.</p>`,
      view: s => ({ title: 'Четыре типа местных экономик', sub: 'те же точки, разложенные по типам', map: { mode: 'none' }, dots: { layout: 'swarm', value: i => MO[i].t24 ? Math.log(MO[i].t24) : null, domain: [Math.log(13000), Math.log(85000)], groups: i => MO[i].ma, K: 4, color: i => MACRO_C[MO[i].ma] }, legend: legendMacro(), svg: (svg, st) => swarmAxis(svg, st, { label: 'траты на жителя в месяц, ₽', lanes: T.macro.names }) }) },
  ] },
  { id: 'network', title: 'Сеть похожих', steps: [
    { text: `<p>Рёбра сети мы не задавали вручную. Граф <b>обучен из данных</b>: два муниципалитета связаны, если их структура и уровень трат близки настолько, что признаки «гладко» лежат на графе. У каждого узла в среднем 15 соседей, всего <b class="num">15 034</b> ребра.</p><p class="note">Метод: learning graphs from smooth signals (Kalofolias, 2016); граф обучается заново для каждого месяца.</p>`,
      view: s => ({ title: 'Обученная сеть сходства потребления', sub: 'силовая раскладка: близкие узлы — похожие траты', map: { mode: 'none' }, dots: { layout: 'network', color: () => (isDark() ? '#9a9aa0' : '#6c6c72'), alpha: 0.8 }, edges: { show: true }, legend: [], svg: isolatesNote }) },
    { text: `<p>Раскрасим узлы типами. Типы оказываются <b>плотными областями сети</b>: столичные агломерации и Север — отдельные сгустки, село и малые города — большое ядро, крупные города — между ними.</p><p class="note">Согласие сообществ, найденных по одной только сети, с типами: AMI ${fmt.num(DATA.checks.network_view.ami_with_macro)}. Сеть сама выделяет столицы и Север; границу между селом и городами проводит контекст.</p>`,
      view: s => ({ title: 'Типы на сети', sub: 'цвет — макротип', map: { mode: 'none' }, dots: { layout: 'network', color: i => MACRO_C[MO[i].ma], alpha: 0.9 }, edges: { show: true }, legend: legendMacro(), svg: isolatesNote }) },
    { text: `<p>Возьмём Нижневартовск, нефтяной город в Югре. Его соседи по сети — не соседи по карте: Мурманск, Архангельск, Северодвинск, Петрозаводск, Комсомольск-на-Амуре, Ногликский район на Сахалине и три соседа по Югре. Всех объединяет похожая корзина трат.</p>`,
      view: s => { const nb = neighbours(iAnchor); nb.add(iAnchor); return { title: 'Соседи Нижневартовска по сети', sub: `${nb.size - 1} ближайших по потреблению`, map: { mode: 'none' }, dots: { layout: 'network', color: i => MACRO_C[MO[i].ma], highlight: nb }, edges: { show: true, only: new Set([iAnchor]) }, legend: legendMacro(), svg: (svg, st) => { const [x, y] = st.map.toScreen(st.tx[iAnchor], st.ty[iAnchor]); annotate(svg, x, y, 'Нижневартовск', { dy: -16 }); } }; } },
    { text: `<p>Вернём те же точки на карту: соседи Нижневартовска разбросаны от Баренцева моря до Сахалина. <b>Сеть строится по поведению, а не по географии</b>, поэтому типы, найденные на ней, — это типы экономик, а не регионы.</p><p class="note">Доля рёбер внутри одного региона — 37% у обученного графа против 78% у графа по дорогам.</p>`,
      view: s => { const nb = neighbours(iAnchor); nb.add(iAnchor); return { title: 'Те же соседи на карте', sub: 'расстояние по дорогам не играет роли', map: { mode: 'grey' }, dots: { layout: 'map', color: i => MACRO_C[MO[i].ma], highlight: nb, alpha: 0.95 }, edges: { show: true, only: new Set([iAnchor]) }, legend: legendMacro(), svg: (svg, st) => { const [x, y] = st.map.toScreen(st.tx[iAnchor], st.ty[iAnchor]); annotate(svg, x, y, 'Нижневартовск', { dy: -16 }); } }; } },
  ] },
  { id: 'types', title: 'Четыре типа', steps: [
    { text: `<p>Четыре макротипа покрывают страну устойчивыми поясами: сельский пояс европейской России и юга Сибири с вкраплениями городов, столичные агломерации и северный ресурсный пояс от Кольского полуострова до Чукотки.</p>`,
      view: s => ({ title: 'Макротипы, декабрь 2024', sub: 'клик по МО откроет паспорт', map: { mode: 'macro', month: 23 }, dots: { layout: 'hidden' }, legend: legendMacro() }) },
    ...[3, 0, 1, 2].map(k => ({
      text: {
        3: `<p><b>Сельские и малые</b> — половина панели. Продовольствие занимает 47% трат, общепит — 1,6%; население, зарплаты и доступность рынков ниже средних, занятость — сельское хозяйство и бюджетный сектор.</p><p class="note">Например: Предгорный, Изобильненский, Балашовский, Моздокский районы.</p>`,
        0: `<p><b>Крупные и средние города</b>: население вдвое выше среднего, промышленная занятость, траты около средних, общепит выше среднего на четверть.</p><p class="note">Например: Новосибирск, Казань, Нижний Новгород, Красноярск.</p>`,
        1: `<p><b>Северные и ресурсные</b>: добыча почти в семь раз выше среднего, зарплаты и траты выше, но маркетплейсы и здоровье ниже — логистика и цены. Доступность рынков вдвое ниже средней.</p><p class="note">Например: Якутск, Норильск, Петропавловск-Камчатский, Сургутский район.</p>`,
        2: `<p><b>Столичные агломерации</b>: общепит в 2,3 раза выше среднего, продовольствие — на четверть ниже; траты, зарплаты и доступность рынков — максимальные. Закон Энгеля в чистом виде.</p><p class="note">Например: Москва и Санкт-Петербург, Подмосковье, Екатеринбург, Тюмень.</p>`,
      }[k],
      view: s => ({ title: nameOf.macro(k), sub: `${T.macro.sizes[k]} МО · отклонения от среднего по России`, map: { mode: 'macro', month: 23, only: new Set(MO.map((m, i) => i).filter(i => MO[i].ma === k)) }, dots: { layout: 'hidden' }, legend: legendMacro(), svg: (svg, st) => { const narrow = st.w < 780;
        fingerprint(svg, st, 'macro', k, narrow ? { x: 22, y: st.top + 42, w: st.w - 56, top: 6 } : { x: st.w - Math.min(330, st.w * 0.36) - 8, y: st.top + 36, w: Math.min(310, st.w * 0.34) }); } }),
    })),
    { text: `<p>Типы объяснимы без модели. Три порога — доля общепита, зарплата и уровень трат — воспроизводят макротип у 80% муниципалитетов. Это дерево можно применить к любому МО «на глаз».</p><p class="note">Пороги подобраны деревом IMM (Dasgupta et al., 2020) на сырых признаках; суррогатное дерево с 8 листьями даёт 89%.</p>`,
      view: s => ({ title: 'Три порога вместо модели', sub: 'дерево решений на сырых признаках', map: { mode: 'none' }, dots: { layout: 'hidden' }, legend: [], svg: (svg, st) => immTree(svg, st) }) },
  ] },
];

/* ---------- движок v2: непрерывный сглаженный прогресс, карточки глав, рейка, полоса прогресса ---------- */
const CH_META = {
  spend:   { lede: 'Уровень и структура трат — два разных измерения одной территории. С них начинается типология.', accent: 'var(--t0)' },
  network: { lede: 'Рёбра сети обучены из данных: соседи по потреблению, а не по карте.', accent: 'var(--t3)' },
  types:   { lede: 'Четыре устойчивых пояса страны и три порога, которые объясняют тип без модели.', accent: 'var(--t2)' },
  time:    { lede: '24 месяца с учётом истории: кто меняет тип, кто нет и почему смены редки.', accent: 'var(--t1)' },
  wave:    { lede: 'Единственный большой сдвиг структуры трат за два года — и быстрее всего он идёт в селе.', accent: 'var(--t0)' },
  lenses:  { lede: 'Экономическая линза отвечает «какая это территория», поведенческая — «как здесь тратят».', accent: 'var(--t5)' },
  checks:  { lede: 'Синтетика с известной истиной, бутстрап по числу типов и два прогона с нуля.', accent: 'var(--t2)' },
};
const Story = (function () {
  let stage, steps = [], cards = [], firstOf = [], active = -1, tops = [], cardTops = [], stepH = [], storyTop = 0, storyH = 1;
  let lastY = -1, pRaw = 0, pS = 0, lastP = -1, dirty = true, debug = false, lastT = performance.now(), warmTimer = 0;
  const rail = document.getElementById('rail'), bar = document.getElementById('progressBar'), story = document.getElementById('story');
  function build() {
    const root = document.getElementById('steps'); let k = 0; const html = [];
    CHAPTERS.forEach((ch, ci) => {
      const meta = CH_META[ch.id] || {}; firstOf[ci] = k;
      html.push(`<section class="chapter-card" id="ch-${esc(ch.id)}" data-first="${k}" style="--ch-accent:${meta.accent || 'var(--t0)'}"><div class="glow"></div><div class="inner"><div class="n">${String(ci + 1).padStart(2, '0')}</div><h2>${esc(ch.title)}</h2><p class="lede">${esc(meta.lede || '')}</p></div></section>`);
      ch.steps.forEach(st => { st.index = k; st.chapter = ch; st.ci = ci; html.push(`<div class="step" data-i="${k}" id="step-${k}"><div class="kicker">Глава ${ci + 1} · ${esc(ch.title)}</div>${st.text}</div>`); steps.push(st); k++; });
    });
    root.innerHTML = html.join('');
    cards = [...root.querySelectorAll('.chapter-card')];
    if (rail) {
      rail.innerHTML = CHAPTERS.map((ch, ci) => `<button type="button" class="dot" data-ch="${ci}" aria-label="Глава ${ci + 1} · ${esc(ch.title)}"><span class="lbl">${ci + 1} · ${esc(ch.title)}</span></button>`).join('');
      rail.querySelectorAll('.dot').forEach(b => b.onclick = () => cards[+b.dataset.ch].scrollIntoView({ behavior: REDUCED ? 'auto' : 'smooth', block: 'start' }));
    }
  }
  const docTop = el => el.getBoundingClientRect().top + window.scrollY;
  function measure() {
    tops = steps.map((s, k) => docTop(document.getElementById('step-' + k))); stepH = steps.map((s, k) => document.getElementById('step-' + k).offsetHeight);
    cardTops = cards.map(docTop); storyTop = docTop(story); storyH = story.offsetHeight; dirty = true;
  }
  function activate(k, instant) {
    if (k === active) return; active = k; document.querySelectorAll('#steps .step').forEach(el => el.classList.toggle('on', +el.dataset.i === k));
    const st = steps[k]; stage.setView(st.view(stage), { instant }); lastP = -1; pS = 0;
    clearTimeout(warmTimer); warmTimer = setTimeout(() => { [steps[k + 1], steps[k - 1]].forEach(s => { if (s) stage.warm(s.view(stage)); }); }, 420);  // прогрев соседних видов
    if (rail) rail.querySelectorAll('.dot').forEach(b => b.classList.toggle('on', +b.dataset.ch === st.ci));
  }
  /* активный шаг — последний, чья верхняя кромка выше линии активации (55% высоты экрана, на узком 82% — ниже липкой сцены);
     карточка главы заранее включает первый вид своей главы */
  function pickActive(y) {
    const line = y + innerHeight * (innerWidth <= 980 ? 0.82 : 0.55); let best = -1;
    for (let k = 0; k < steps.length; k++) { if (tops[k] <= line) best = k; else break; }
    cards.forEach((c, ci) => { const f = firstOf[ci]; if (cardTops[ci] <= line && f > best) best = f; });
    if (best >= 0 && best !== active) activate(best);
  }
  function frame(now) {
    const dt = Math.min(0.064, Math.max(0.001, (now - lastT) / 1000)); lastT = now; const y = window.scrollY;
    if (!debug && (y !== lastY || dirty)) {
      dirty = false; lastY = y; pickActive(y);
      if (active >= 0) { const k = active; const next = tops[k + 1] ?? (tops[k] + stepH[k] + innerHeight * 0.75); const span = Math.max(1, next - tops[k]); pRaw = Math.max(0, Math.min(1, (y + innerHeight * 0.5 - tops[k]) / span)); }
      const mid = y + innerHeight / 2;
      cards.forEach((c, ci) => { const cm = cardTops[ci] + c.offsetHeight / 2; const p = Math.max(0, Math.min(1, 1 - Math.abs(mid - cm) / (innerHeight * 0.62))); if (Math.abs(p - (c._p || 0)) > 0.004) { c._p = p; c.style.setProperty('--p', p.toFixed(3)); } });
      const sp = Math.max(0, Math.min(1, (y - storyTop) / Math.max(1, storyH - innerHeight))); if (bar) bar.style.setProperty('--sp', sp.toFixed(4));
      if (rail) rail.classList.toggle('show', y > storyTop - innerHeight * 0.4 && y < storyTop + storyH - innerHeight * 0.6);
    }
    pS = REDUCED ? pRaw : pS + (pRaw - pS) * (1 - Math.exp(-dt * 10));
    if (Math.abs(pS - lastP) > 0.0015) { lastP = pS; const st = steps[active]; if (st && st.progress) st.progress(pS, stage); }
    requestAnimationFrame(frame);
  }
  function init(s) {
    stage = s; build(); measure();
    window.addEventListener('resize', measure); if (document.fonts && document.fonts.ready) document.fonts.ready.then(measure);
    setInterval(() => { if (Math.abs(story.offsetHeight - storyH) > 2) measure(); }, 1000);
    const dbg = new URLSearchParams(location.search).get('view');
    if (dbg != null && steps[+dbg]) { debug = true; document.body.classList.add('debug-view'); requestAnimationFrame(() => { stage.resize(); activate(+dbg, true); }); requestAnimationFrame(frame); return; }
    const q = new URLSearchParams(location.search).get('step');
    if (q != null && steps[+q]) { const el = document.getElementById('step-' + q); requestAnimationFrame(() => { el.scrollIntoView({ block: 'center', behavior: 'instant' }); measure(); activate(+q, true); }); }
    else activate(0, true);
    requestAnimationFrame(frame);
  }
  return { init, steps: () => steps, activate, activeIndex: () => active, measure };
})();
