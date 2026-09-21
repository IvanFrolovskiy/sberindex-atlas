/* Графики для глав 4–7 и исследователя (SVG, D3). Все — в экранных координатах сцены. */
const Charts = {};
/* Появление графиков. CA.on выключается на перерисовках по прокрутке, чтобы анимация не начиналась заново каждый кадр. */
const CA = { on: true };
const canAnim = () => CA.on && !REDUCED;
/* столбики растут от базовой линии */
Charts.riseY = (g, sel, base, dur = 560, step = 26) => { if (!canAnim()) return; g.selectAll(sel).each(function (d, i) { const r = d3.select(this), y = +r.attr('y'), h = +r.attr('height'); r.attr('y', base).attr('height', 0).transition().delay(i * step).duration(dur).ease(d3.easeCubicOut).attr('y', y).attr('height', h); }); };
/* горизонтальные столбики растут от нулевой линии */
Charts.growX = (g, sel, zero, dur = 560, step = 26) => { if (!canAnim()) return; g.selectAll(sel).each(function (d, i) { const r = d3.select(this), x = +r.attr('x'), w = +r.attr('width'); r.attr('x', zero).attr('width', 0).transition().delay(i * step).duration(dur).ease(d3.easeCubicOut).attr('x', x).attr('width', w); }); };
/* линии прочерчиваются */
Charts.drawPaths = (g, sel, dur = 900, step = 90) => { if (!canAnim()) return; g.selectAll(sel).each(function (d, i) { const p = d3.select(this), L = this.getTotalLength ? this.getTotalLength() : 0; if (!L) return; p.attr('stroke-dasharray', L + ' ' + L).attr('stroke-dashoffset', L).transition().delay(i * step).duration(dur).ease(d3.easeCubicInOut).attr('stroke-dashoffset', 0).on('end', function () { d3.select(this).attr('stroke-dasharray', null); }); }); };
/* точки проявляются с задержкой по порядку */
Charts.pop = (g, sel, dur = 420, total = 700) => { if (!canAnim()) return; const n = g.selectAll(sel).size() || 1; g.selectAll(sel).each(function (d, i) { const c = d3.select(this), r = +c.attr('r'); c.attr('r', 0).transition().delay(i / n * total).duration(dur).ease(d3.easeCubicOut).attr('r', r); }); };
/* общее мягкое проявление с подъёмом */
Charts.fadeUp = (g, sel, dy = 8, dur = 520, step = 40) => { if (!canAnim()) return; g.selectAll(sel).each(function (d, i) { const e = d3.select(this); e.style('opacity', 0).attr('transform', (e.attr('transform') || '') + ` translate(0,${dy})`).transition().delay(i * step).duration(dur).ease(d3.easeCubicOut).style('opacity', 1).attr('transform', e.attr('transform').replace(` translate(0,${dy})`, '')); }); };
/* прямоугольник-маска: график открывается слева направо (ширину двигает прокрутка) */
Charts.clip = (svg, g, { x, y, w, h }) => { const id = 'clip' + Math.random().toString(36).slice(2, 8); const r = svg.append('defs').append('clipPath').attr('id', id).append('rect').attr('x', x).attr('y', y).attr('width', w).attr('height', h); g.attr('clip-path', `url(#${id})`); return r; };
/* общий каркас: группа с полями, шкалы, оси-hairline */
Charts.frame = (svg, { x = 0, y = 0, w, h, ml = 44, mr = 16, mt = 24, mb = 32 } = {}) => {
  const g = svg.append('g').attr('transform', `translate(${x + ml},${y + mt})`).attr('font-family', 'inherit');
  return { g, iw: w - ml - mr, ih: h - mt - mb, ml, mt };
};
Charts.axisX = (g, scale, ih, { ticks = 6, fmt: f = d => d } = {}) => {
  const ax = g.append('g').attr('transform', `translate(0,${ih})`).attr('font-size', 11.5).attr('fill', cssVar('--muted'));
  ax.append('line').attr('x1', scale.range()[0]).attr('x2', scale.range()[1]).attr('stroke', cssVar('--line-2'));
  (scale.ticks ? scale.ticks(ticks) : scale.domain()).forEach(t => { const x = scale(t) + (scale.bandwidth ? scale.bandwidth() / 2 : 0); ax.append('text').attr('x', x).attr('y', 18).attr('text-anchor', 'middle').text(f(t)); });
};
Charts.axisY = (g, scale, iw, { ticks = 5, fmt: f = d => d, grid = true } = {}) => {
  const ax = g.append('g').attr('font-size', 11.5).attr('fill', cssVar('--muted'));
  scale.ticks(ticks).forEach(t => { const y = scale(t); if (grid) ax.append('line').attr('x1', 0).attr('x2', iw).attr('y1', y).attr('y2', y).attr('stroke', cssVar('--grid')); ax.append('text').attr('x', -8).attr('y', y + 4).attr('text-anchor', 'end').text(f(t)); });
};
/* штрих-код историй типа: N строк × 24 месяца, строки сгруппированы по модальному типу */
Charts.barcode = (canvasCtx, { x, y, w, h, lens = 'macro', order = null, dpr = 1 }) => {
  const idx = order || d3.range(N).sort((a, b) => (modalOf(MO[a], lens) - modalOf(MO[b], lens)) || (MO[b].sw[0] - MO[a].sw[0]));
  const rowH = h / idx.length, colW = w / 24; const ctx = canvasCtx;
  ctx.save(); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  for (let r = 0; r < idx.length; r++) { const m = MO[idx[r]]; const s = seqOf(m, lens); if (!s) continue; let t0 = 0; for (let t = 1; t <= 24; t++) { if (t === 24 || s[t] !== s[t0]) { ctx.fillStyle = colorOf[lens](+s[t0]); ctx.fillRect(x + t0 * colW, y + r * rowH, (t - t0) * colW + 0.4, Math.max(rowH, 0.5)); t0 = t; } } }
  ctx.restore(); return idx;
};
/* потоки между типами (Sankey) по выбранным месяцам */
const SHORT_TYPE = { 'Крупные и средние города': 'Города', 'Северные и ресурсные': 'Север', 'Столичные агломерации': 'Столицы', 'Сельские и малые': 'Село' };
Charts.flows = (svg, { x, y, w, h, lens = 'macro', months = null, narrow = false }) => {
  months = months || (narrow ? [0, 11, 23] : [0, 6, 12, 18, 23]);
  const nm = d => narrow ? (SHORT_TYPE[d.name] || d.name) : d.name;
  const K = T[lens].names.length; const nodes = [], links = []; const id = (t, k) => t * K + k;
  months.forEach((mi, t) => { for (let k = 0; k < K; k++) nodes.push({ id: id(t, k), k, t, name: nameOf[lens](k) }); });
  for (let t = 0; t < months.length - 1; t++) { const cnt = new Map(); for (const m of MO) { const a = typeAt(m, lens, months[t]), b = typeAt(m, lens, months[t + 1]); const key = a + '-' + b; cnt.set(key, (cnt.get(key) || 0) + 1); } for (const [key, v] of cnt) { const [a, b] = key.split('-').map(Number); links.push({ source: id(t, a), target: id(t + 1, b), value: v, a, b }); } }
  const sk = d3.sankey().nodeId(d => d.id).nodeWidth(12).nodePadding(10).nodeSort(null).linkSort(null).extent([[x, y + 22], [x + w, y + h - 6]]);
  const graph = sk({ nodes: nodes.map(d => ({ ...d })), links: links.map(d => ({ ...d })) });
  const g = svg.append('g').attr('font-family', 'inherit');
  g.append('g').selectAll('path').data(graph.links).join('path').attr('d', d3.sankeyLinkHorizontal()).attr('fill', 'none').attr('stroke', d => colorOf[lens](d.a)).attr('stroke-opacity', d => d.a === d.b ? 0.16 : 0.6).attr('stroke-width', d => Math.max(1, d.width));
  g.append('g').selectAll('rect').data(graph.nodes).join('rect').attr('x', d => d.x0).attr('y', d => d.y0).attr('width', d => d.x1 - d.x0).attr('height', d => Math.max(1, d.y1 - d.y0)).attr('fill', d => colorOf[lens](d.k)).attr('rx', 2);
  g.append('g').attr('font-size', narrow ? 10 : 11.5).attr('fill', cssVar('--muted')).selectAll('text').data(months).join('text').attr('x', (mi, t) => x + (w) * t / (months.length - 1)).attr('y', y + 12).attr('text-anchor', (d, t) => t === 0 ? 'start' : t === months.length - 1 ? 'end' : 'middle').text(mi => fmt.month(MONTHS[mi]));
  if (canAnim()) g.selectAll('path').style('opacity', 0).transition().delay((d, i) => i * 7).duration(650).ease(d3.easeCubicOut).style('opacity', null);
  if (!narrow) g.append('g').attr('font-size', 11.5).attr('font-weight', 600).attr('fill', cssVar('--ink')).selectAll('text').data(graph.nodes.filter(d => d.t === months.length - 1)).join('text').attr('x', d => d.x0 - 8).attr('y', d => (d.y0 + d.y1) / 2 + 4).attr('text-anchor', 'end').text(d => (d.y1 - d.y0 > 12 ? d.name : ''));
  g.append('g').attr('font-size', narrow ? 10.5 : 11.5).attr('font-weight', 600).attr('fill', cssVar('--ink')).selectAll('text').data(graph.nodes.filter(d => d.t === 0)).join('text').attr('x', d => d.x1 + 8).attr('y', d => (d.y0 + d.y1) / 2 + 4).text(d => (d.y1 - d.y0 > 12 ? nm(d) : ''));
  return graph;
};
/* линии по типам (доля маркетплейсов) с постепенным «дорисовыванием» до месяца upto */
Charts.lines = (svg, { x, y, w, h, series, xs: xsDom, ys: ysDom, colors, labels, upto = 23, fmtY = v => v, title = null, refLine = null, mr = 150, ghost = false }) => {
  const { g, iw, ih } = Charts.frame(svg, { x, y, w, h, ml: 44, mr, mt: title ? 30 : 16, mb: 30 });
  const xs = d3.scaleLinear().domain(xsDom).range([0, iw]), ys = d3.scaleLinear().domain(ysDom).range([ih, 0]).nice();
  Charts.axisY(g, ys, iw, { fmt: fmtY }); Charts.axisX(g, xs, ih, { ticks: 6, fmt: t => MONTHS[Math.round(t)] ? fmt.month(MONTHS[Math.round(t)]) : '' });
  if (title) g.append('text').attr('x', 0).attr('y', -12).attr('font-size', 12.5).attr('font-weight', 650).attr('fill', cssVar('--ink')).text(title);
  const line = d3.line().x(d => xs(d[0])).y(d => ys(d[1])).curve(d3.curveMonotoneX);
  // upto может быть дробным: последний отрезок дорисовывается частично (плавная анимация по прокрутке)
  const cut = s => { const n = Math.min(Math.floor(upto), s.length - 1), f = upto - n; const pts = s.slice(0, n + 1).map((v, i) => [i, v]); if (f > 0.001 && s[n + 1] != null) pts.push([n + f, s[n] + (s[n + 1] - s[n]) * f]); return pts; };
  const endsY = [];
  // весь ряд бледно: читатель сразу видит форму, а прокрутка «проявляет» линию по месяцам
  if (ghost) series.forEach((s, k) => g.append('path').attr('d', line(s.map((v, i) => [i, v]))).attr('fill', 'none').attr('stroke', colors[k]).attr('stroke-opacity', 0.22).attr('stroke-width', 1.5).attr('stroke-linejoin', 'round'));
  series.forEach((s, k) => { const pts = cut(s); g.append('path').attr('d', line(pts)).attr('fill', 'none').attr('stroke', colors[k]).attr('stroke-width', 2).attr('stroke-linejoin', 'round'); const last = pts[pts.length - 1]; endsY[k] = ys(last[1]); g.append('circle').attr('cx', xs(last[0])).attr('cy', ys(last[1])).attr('r', 4).attr('fill', colors[k]).attr('stroke', cssVar('--card')).attr('stroke-width', 2); });
  // подписи справа с раздвижкой
  const ends = series.map((s, k) => ({ k, y: ghost ? ys(s[s.length - 1]) : endsY[k] })).sort((a, b) => a.y - b.y); for (let i = 1; i < ends.length; i++) if (ends[i].y - ends[i - 1].y < 14) ends[i].y = ends[i - 1].y + 14;
  ends.forEach(e => g.append('text').attr('x', iw + 8).attr('y', e.y + 4).attr('font-size', 11.5).attr('font-weight', 600).attr('fill', cssVar('--ink')).text(labels[e.k]));
  if (canAnim() && upto >= 23) Charts.drawPaths(g, 'path', 1100, 140);
  if (refLine) { g.append('line').attr('x1', xs(refLine.x)).attr('x2', xs(refLine.x)).attr('y1', 0).attr('y2', ih).attr('stroke', cssVar('--line-2')).attr('stroke-dasharray', '3,3'); g.append('text').attr('x', xs(refLine.x) + 4).attr('y', 12).attr('font-size', 11).attr('fill', cssVar('--muted')).text(refLine.label); }
  return { g, xs, ys, iw, ih };
};
/* точечная диаграмма с типами */
Charts.scatter = (svg, { x, y, w, h, xv, yv, color, xsDom, ysDom, xLabel, yLabel, fmtX = v => v, fmtY = v => v, logX = false, r = 2.6, annotate: ann = [] }) => {
  const { g, iw, ih } = Charts.frame(svg, { x, y, w, h, ml: 52, mr: 20, mt: 16, mb: 40 });
  const xs = (logX ? d3.scaleLog() : d3.scaleLinear()).domain(xsDom).range([0, iw]), ys = d3.scaleLinear().domain(ysDom).range([ih, 0]).nice();
  Charts.axisY(g, ys, iw, { fmt: fmtY }); const ax = g.append('g').attr('transform', `translate(0,${ih})`).attr('font-size', 11.5).attr('fill', cssVar('--muted'));
  ax.append('line').attr('x1', 0).attr('x2', iw).attr('stroke', cssVar('--line-2')); xs.ticks(6).forEach(t => ax.append('text').attr('x', xs(t)).attr('y', 18).attr('text-anchor', 'middle').text(fmtX(t)));
  g.append('text').attr('x', iw).attr('y', ih + 34).attr('text-anchor', 'end').attr('font-size', 11.5).attr('fill', cssVar('--muted')).text(xLabel);
  g.append('text').attr('x', 0).attr('y', -4).attr('font-size', 11.5).attr('fill', cssVar('--muted')).text(yLabel);
  const pts = g.append('g'); for (let i = 0; i < N; i++) { const xv_ = xv(i), yv_ = yv(i); if (xv_ == null || yv_ == null || isNaN(xv_) || isNaN(yv_)) continue; pts.append('circle').attr('cx', xs(xv_)).attr('cy', ys(yv_)).attr('r', r).attr('fill', color(i)).attr('fill-opacity', 0.75); }
  Charts.pop(pts, 'circle', 380, 900);
  ann.forEach(a => annotate(g, xs(a.x), ys(a.y), a.text, { dy: a.dy ?? -14, dx: a.dx ?? 0, anchor: a.anchor || 'middle' }));
  return { g, xs, ys, iw, ih };
};
/* столбики с усами (медиана + межквартильный размах) по типам */
Charts.bars = (svg, { x, y, w, h, items, fmtV = v => v, yLabel = '' }) => {
  const { g, iw, ih } = Charts.frame(svg, { x, y, w, h, ml: 44, mr: 12, mt: 20, mb: 52 });
  const xs = d3.scaleBand().domain(items.map(d => d.label)).range([0, iw]).padding(0.35); const ys = d3.scaleLinear().domain([0, d3.max(items, d => d.hi || d.v) * 1.15]).range([ih, 0]).nice();
  Charts.axisY(g, ys, iw, { fmt: fmtV });
  items.forEach(d => { const bx = xs(d.label), bw = Math.min(xs.bandwidth(), 60), cx = bx + xs.bandwidth() / 2; g.append('rect').attr('x', cx - bw / 2).attr('y', ys(d.v)).attr('width', bw).attr('height', ih - ys(d.v)).attr('rx', 4).attr('fill', d.color); if (d.lo != null) { g.append('line').attr('x1', cx).attr('x2', cx).attr('y1', ys(d.lo)).attr('y2', ys(d.hi)).attr('stroke', cssVar('--ink-2')).attr('stroke-width', 1.2); [d.lo, d.hi].forEach(v => g.append('line').attr('x1', cx - 5).attr('x2', cx + 5).attr('y1', ys(v)).attr('y2', ys(v)).attr('stroke', cssVar('--ink-2'))); } g.append('text').attr('x', cx).attr('y', ys(d.hi || d.v) - 8).attr('text-anchor', 'middle').attr('font-size', 12).attr('font-weight', 600).attr('fill', cssVar('--ink')).text(fmtV(d.v)); const lines = d.label.split(' и '); lines.forEach((ln, i) => g.append('text').attr('x', cx).attr('y', ih + 16 + i * 14).attr('text-anchor', 'middle').attr('font-size', 11.5).attr('fill', cssVar('--ink-2')).text(i === 0 ? ln + (lines.length > 1 ? ' и' : '') : ln)); });
  g.append('text').attr('x', 0).attr('y', -6).attr('font-size', 11.5).attr('fill', cssVar('--muted')).text(yLabel);
  Charts.riseY(g, 'rect', ih, 620, 70);
  return { g, xs, ys };
};
/* ленты между двумя линзами: экономические типы (слева) → поведенческие (справа) */
Charts.ribbons = (svg, { x, y, w, h }) => {
  const ct = T.lens.crosstab; const rows = T.macro.names, cols = T.beh.names; const nodes = [], links = [];
  rows.forEach((r, i) => nodes.push({ id: 'e' + i, name: r, side: 0, k: i })); cols.forEach((c, j) => nodes.push({ id: 'b' + j, name: c, side: 1, k: j }));
  rows.forEach((r, i) => cols.forEach((c, j) => { const v = ct[r] && ct[r][c]; if (v) links.push({ source: 'e' + i, target: 'b' + j, value: v, a: i, b: j }); }));
  const sk = d3.sankey().nodeId(d => d.id).nodeWidth(14).nodePadding(14).nodeSort(null).linkSort(null).extent([[x + 230, y + 10], [x + w - 250, y + h - 10]]);
  const graph = sk({ nodes: nodes.map(d => ({ ...d })), links: links.map(d => ({ ...d })) });
  const g = svg.append('g').attr('font-family', 'inherit');
  g.append('g').selectAll('path').data(graph.links).join('path').attr('d', d3.sankeyLinkHorizontal()).attr('fill', 'none').attr('stroke', d => MACRO_C[d.a]).attr('stroke-opacity', 0.45).attr('stroke-width', d => Math.max(1, d.width));
  g.append('g').selectAll('rect').data(graph.nodes).join('rect').attr('x', d => d.x0).attr('y', d => d.y0).attr('width', d => d.x1 - d.x0).attr('height', d => Math.max(1, d.y1 - d.y0)).attr('fill', d => d.side === 0 ? MACRO_C[d.k] : BEH_C[d.k]).attr('rx', 3);
  g.append('g').attr('font-size', 12).attr('font-weight', 600).attr('fill', cssVar('--ink')).selectAll('text').data(graph.nodes).join('text').attr('x', d => d.side === 0 ? d.x0 - 10 : d.x1 + 10).attr('y', d => (d.y0 + d.y1) / 2 + 4).attr('text-anchor', d => d.side === 0 ? 'end' : 'start').text(d => d.name + ' · ' + d.value);
  g.append('text').attr('x', x + 220).attr('y', y - 2).attr('text-anchor', 'end').attr('font-size', 11.5).attr('fill', cssVar('--muted')).text('экономическая типология');
  g.append('text').attr('x', x + w - 240).attr('y', y - 2).attr('text-anchor', 'start').attr('font-size', 11.5).attr('fill', cssVar('--muted')).text('поведенческая (только СберИндекс)');
  if (canAnim()) { g.selectAll('path').style('opacity', 0).transition().delay((d, i) => i * 9).duration(700).ease(d3.easeCubicOut).style('opacity', null); Charts.riseY(g, 'rect', 0, 520, 40); }
  return graph;
};
