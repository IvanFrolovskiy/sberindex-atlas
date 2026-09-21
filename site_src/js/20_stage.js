/* Сцена истории v2. Карта — кэшированные битмапы с кроссфейдом (один рендер полигонов на смену вида, а не на кадр);
   точки-МО — экспоненциальное приближение к цели (движение всегда непрерывно и обратимо, без таймерных твинов);
   рёбра и оверлеи — плавное появление; заголовок, подпись и легенда — короткий кроссфейд. */
const FRAME = { w: 1000, h: 620 };            // общий кадр: карта (1000×537) центрируется по вертикали
const MAP_DY = (FRAME.h - DATA.frame.h) / 2;  // смещение карты внутри кадра
const rgbOf = (() => { const cache = new Map(); return s => { let v = cache.get(s); if (!v) { const c = d3.rgb(s); v = [c.r, c.g, c.b, isNaN(c.opacity) ? 1 : c.opacity]; cache.set(s, v); } return v; }; })();
const css = v => `rgba(${v[0] | 0},${v[1] | 0},${v[2] | 0},${v[3]})`;
/* скорости приближения (1/с): чем больше, тем быстрее; k = 1 − e^(−λ·dt) не зависит от частоты кадров */
const LAMBDA = { pos: 6.5, col: 9, mix: 7, alpha: 8, edge: 6 };

class Stage {
  constructor() {
    this.box = document.getElementById('stage');
    this.map = new MapLayer(document.getElementById('mapCanvas'));
    this.dc = document.getElementById('dotsCanvas'); this.dctx = this.dc.getContext('2d');
    this.svg = d3.select('#stageSvg'); this.html = document.getElementById('stageHtml');
    this.titleEl = document.getElementById('stageTitle'); this.subEl = document.getElementById('stageSub'); this.legendEl = document.getElementById('stageLegend');
    // состояние точек: текущее и целевое
    this.x = new Float32Array(N); this.y = new Float32Array(N); this.tx = new Float32Array(N); this.ty = new Float32Array(N);
    this.col = new Float32Array(N * 4); this.tcol = new Float32Array(N * 4);
    this.r = 2.6; this.dotsAlpha = 0; this.dotsAlphaT = 0;
    // карта: две битмапы и доля второй
    this.mapA = null; this.mapB = null; this.mapMix = 1; this.mapFading = false; this.mapStroke = null;
    this.edgeA = 0; this.edgeT = 0; this.ovA = 1; this.paintFn = null; this.paintDirty = true; this.wipeP = 1; this.cache = new Map();  // кэш битмап карты по сигнатуре вида
    this.view = null; this.hover = -1; this.running = false; this.last = 0; this.legendHtml = '';
    this.w = 0; this.h = 0; this.t = { k: 1, x: 0, y: 0 }; this.top = 60; this.bottom = 56; this.chartBottom = 56;
    for (let i = 0; i < N; i++) { const c = MO[i].c; this.x[i] = this.tx[i] = c[0] == null ? -100 : c[0]; this.y[i] = this.ty[i] = c[1] == null ? -100 : c[1] + MAP_DY; }
    this.frame = this.frame.bind(this);
    this.resize();
    new ResizeObserver(() => this.resize()).observe(this.box);
    this.dc.addEventListener('pointermove', e => this.onMove(e)); this.dc.addEventListener('pointerleave', () => { if (this.hover >= 0) { this.hover = -1; Tip.hide(); this.drawMap(); this.drawDots(); } });
    this.dc.addEventListener('click', e => { const i = this.hitAt(e); if (i >= 0 && window.openPassport) { window.openPassport(MO[i].id); document.getElementById('passport').scrollIntoView({ behavior: 'smooth', block: 'start' }); } });
    window.addEventListener('themechange', () => this.rerender());
  }
  resize() {
    const w = this.box.clientWidth, h = this.box.clientHeight;
    if (!w || !h) { this.w = 0; this.h = 0; return; }  // бокс ещё не измерен: ждём ResizeObserver
    this.map.resize(w, h); this.dc.width = Math.round(w * this.map.dpr); this.dc.height = Math.round(h * this.map.dpr); this.svg.attr('viewBox', `0 0 ${w} ${h}`);
    this.w = w; this.h = h; this.cache.clear(); this.paintDirty = true;
    const mobile = w < 700; this.top = mobile ? 44 : 60; this.bottom = mobile ? 70 : 56;  // зоны заголовка сверху и легенды снизу
    this.chartBottom = this.bottom;
    this.t = this.map.fit(FRAME.w, FRAME.h, { top: this.top, bottom: this.bottom, left: 8, right: 8 });
    this.r = Math.max(2.4, Math.min(5.2, 2.3 / this.t.k));
    this.rerender();
  }
  /* пересчёт вида без анимации (ресайз, смена темы) */
  rerender() { if (this.view) this.applyView(this.view, { instant: true }); else { this.mapA = null; this.drawMap(); this.drawDots(); } }
  /* ---- геометрия точек ---- */
  layoutPositions(kind, opts = {}) {
    const px = new Float32Array(N), py = new Float32Array(N);
    if (kind === 'map') { for (let i = 0; i < N; i++) { const c = MO[i].c; px[i] = c[0] == null ? -100 : c[0]; py[i] = c[1] == null ? -100 : c[1] + MAP_DY; } }
    else if (kind === 'network') { const P = DATA.graph.pos; for (let i = 0; i < N; i++) { px[i] = P[i][0] == null ? -100 : P[i][0]; py[i] = P[i][1] == null ? -100 : P[i][1] + MAP_DY; } }
    else if (kind === 'swarm') {
      const val = opts.value, groups = opts.groups || null, K = opts.K || 1, r = this.r * 1.05;
      const xs = d3.scaleLinear().domain(opts.domain).range([70, 930]).clamp(true);
      const lanes = groups ? K : 1; const laneH = (FRAME.h - 80) / lanes;
      for (let g = 0; g < lanes; g++) {
        const idx = []; for (let i = 0; i < N; i++) { const v = val(i); if (v == null || isNaN(v)) { px[i] = -100; py[i] = -100; continue; } if (!groups || groups(i) === g) idx.push(i); }
        idx.sort((a, b) => val(a) - val(b));
        const base = 40 + laneH * (g + 0.5); const placed = [];
        for (const i of idx) {
          const x = xs(val(i)); let y = 0; let best = null;
          const cand = [0]; for (let k = 1; k < 60; k++) { cand.push(k * r * 1.9); cand.push(-k * r * 1.9); }
          for (const cy of cand) { let ok = true; for (let j = placed.length - 1; j >= 0; j--) { const p = placed[j]; if (x - p[0] > 2 * r) break; const dx = x - p[0], dy = cy - p[1]; if (dx * dx + dy * dy < 4 * r * r) { ok = false; break; } } if (ok) { best = cy; break; } }
          y = best == null ? 0 : best; placed.push([x, y]); px[i] = x; py[i] = base + y;
        }
      }
      this.swarmScale = xs;
    }
    return [px, py];
  }
  /* ---- заливка карты для вида ---- */
  mapFill(m) {
    if (!m || m.mode === 'none') return null;
    const dark = isDark(); const base = dark ? 'rgba(40,45,58,1)' : 'rgba(211,210,202,1)', dim = dark ? 'rgba(30,34,44,1)' : 'rgba(231,230,224,1)';
    if (m.mode === 'grey') return () => base;
    if (m.mode === 'custom') return i => { const c = m.color(i); return c || base; };
    const lens = m.mode;
    return i => {
      if (m.only && !m.only.has(i)) return dim;
      const t = m.month == null ? modalOf(MO[i], lens) : typeAt(MO[i], lens, m.month);
      return colorOf[lens](t);
    };
  }
  /* ---- битмапы карты: сигнатура вида → кэш (до 8 штук); warm() греет соседние виды в спокойный момент ---- */
  mapSig(v) { const m = v.map || { mode: 'none' }; if (m.mode === 'none') return null; return [v.title, m.mode, m.month, m.only ? m.only.size : '', isDark() ? 'd' : 'l', this.w, this.h].join('|'); }
  bitmapFor(v) {
    const sig = this.mapSig(v); if (!sig) return null; let e = this.cache.get(sig);
    if (!e) {
      const fill = this.mapFill(v.map); const stroke = isDark() ? 'rgba(7,9,15,0.75)' : 'rgba(255,255,255,0.85)';
      const cols = new Array(N); for (let i = 0; i < N; i++) cols[i] = fill(i);
      // ближайший по содержанию кэшированный вид → перерисовать только отличающиеся полигоны
      let best = null, bestN = N * 0.6;
      for (const c of this.cache.values()) { let n = 0; for (let i = 0; i < N && n < bestN; i++) if (c.cols[i] !== cols[i]) n++; if (n < bestN) { bestN = n; best = c; } }
      let bmp;
      if (best) { const changed = new Set(); for (let i = 0; i < N; i++) if (best.cols[i] !== cols[i]) changed.add(i); bmp = changed.size ? this.map.renderFrom(best.bmp, fill, changed, { stroke, lineWidth: 0.5 }) : best.bmp; }
      else bmp = this.map.render(fill, { stroke, lineWidth: 0.5 });
      e = { bmp, cols }; this.cache.set(sig, e); if (this.cache.size > 8) this.cache.delete(this.cache.keys().next().value);
    } else { this.cache.delete(sig); this.cache.set(sig, e); }
    return e.bmp;
  }
  warm(v) { try { if (v && v.map && v.map.mode !== 'none') this.bitmapFor(v); } catch (e) { /* прогрев не критичен */ } }
  /* ---- установка вида ----
     instant — без анимации (ресайз, отладка); soft — обновление по прокрутке (месяц, дорисовка линий): без кроссфейда оверлея и заголовка */
  setView(v, { duration, instant = false, soft = false } = {}) { this.applyView(v, { instant: instant || duration === 0 && !soft, soft }); }
  applyView(v, { instant = false, soft = false } = {}) {
    this.view = v;
    if (soft || instant) { this.titleEl.textContent = v.title || ''; this.subEl.textContent = v.sub || ''; this.setLegend(v.legend || [], true); }
    else { swapText(this.titleEl, v.title || ''); swapText(this.subEl, v.sub || ''); this.setLegend(v.legend || [], false); }
    // точки: цели
    const d = v.dots || { layout: 'hidden' };
    if (d.layout !== 'hidden') { const [px, py] = this.layoutPositions(d.layout, d); this.tx.set(px); this.ty.set(py); }
    const dotColor = d.color || (() => GREY); const hl = d.highlight || null; const baseA = d.layout === 'hidden' ? 0 : (d.alpha ?? 0.92);
    for (let i = 0; i < N; i++) { const c = rgbOf(dotColor(i)); const a = hl ? (hl.has(i) ? 1 : 0.16) : 1; const o = i * 4; this.tcol[o] = c[0]; this.tcol[o + 1] = c[1]; this.tcol[o + 2] = c[2]; this.tcol[o + 3] = a * baseA; }
    this.dotsAlphaT = d.layout === 'hidden' ? 0 : 1;
    // карта: новая битмапа (один рендер полигонов)
    const bmp = this.bitmapFor(v);
    if (instant || REDUCED) { this.mapA = bmp; this.mapB = null; this.mapMix = 1; this.mapFading = false; }
    else {
      if (this.mapFading) { this.mapA = (this.mapA || this.mapB) ? this.map.composite(this.mapA, this.mapB, this.mapMix) : null; }  // переход не закончен — зафиксировать текущую смесь
      if (bmp !== this.mapA) { this.mapB = bmp; this.mapMix = 0; this.mapFading = true; } else { this.mapB = null; this.mapMix = 1; this.mapFading = false; }
    }
    this.edgesSpec = v.edges || null; this.edgeT = (v.edges && v.edges.show) ? 1 : 0; this.edgeCanvas = null;  // рёбра рисуются в кэш, когда точки остановятся
    // оверлеи
    this.svgFn = v.svg || null; this.htmlContent = v.html || ''; this.paintFn = v.paint || null; this.paintDirty = true; this.tipFn = v.tip || null;
    if (!soft) this.wipeP = v.wipe ? 0.06 : 1;
    this.swapOverlay(instant || soft);
    if (!soft && !instant) this.ovA = 0;
    if (instant || REDUCED) { this.x.set(this.tx); this.y.set(this.ty); this.col.set(this.tcol); this.dotsAlpha = this.dotsAlphaT; this.edgeA = this.edgeT; this.ovA = 1; }
    this.start();
  }
  /* перерисовать только SVG-оверлей текущего вида: дешевле полного setView и не трогает карту, точки и заголовок */
  redrawOverlay() { CA.on = false; this.svg.selectAll('g.ov').remove(); const g = this.svg.append('g').attr('class', 'ov'); if (this.svgFn) this.svgFn(g, this); this.ovA = 1; CA.on = true; }
  /* «шторка» растрового слоя: доля ширины, открытая прокруткой (штрих-код историй) */
  setWipe(p) { const v = Math.max(0, Math.min(1, p)); if (Math.abs(v - this.wipeP) < 0.002) return; this.wipeP = v; this.drawDots(); }
  setLegend(items, immediate) {
    const html = items.map(l => `<span class="chip"><i style="background:${l.color}"></i>${esc(l.label)}${l.count != null ? ` <span class="num">${esc(l.count)}</span>` : ''}</span>`).join('');
    // сколько рядов займёт легенда: графики отступают на эту высоту, масштаб карты при этом не меняется
    const est = items.reduce((a, l) => a + 30 + String(l.label).length * 6.7 + (l.count != null ? 30 : 0), 0);
    this.chartBottom = Math.max(this.bottom, 26 + Math.max(1, Math.ceil(est / Math.max(220, this.w - 20))) * 30);
    if (html === this.legendHtml) return; this.legendHtml = html;
    if (immediate || REDUCED) { this.legendEl.innerHTML = html; return; }
    this.legendEl.classList.add('swap-out'); clearTimeout(this.legendTimer);
    this.legendTimer = setTimeout(() => { this.legendEl.innerHTML = html; this.legendEl.classList.remove('swap-out'); }, 180);
  }
  swapOverlay(immediate) {
    CA.on = !immediate;
    const old = this.svg.selectAll('g.ov');
    if (immediate || REDUCED) old.remove(); else { old.classed('out', true); setTimeout(() => old.remove(), 480); }
    const g = this.svg.append('g').attr('class', 'ov' + (immediate || REDUCED ? '' : ' pre'));
    if (this.svgFn) this.svgFn(g, this);
    if (!(immediate || REDUCED)) requestAnimationFrame(() => requestAnimationFrame(() => g.classed('pre', false)));
    CA.on = true;
    this.html.innerHTML = this.htmlContent ? `<div>${this.htmlContent}</div>` : '';
  }
  /* ---- цикл кадров ---- */
  start() { if (!this.running) { this.running = true; this.last = performance.now(); requestAnimationFrame(this.frame); } }
  frame(now) {
    if (!this.w || !this.h) { this.running = false; return; }
    const dt = Math.min(0.064, Math.max(0.001, (now - this.last) / 1000)); this.last = now;
    const kp = 1 - Math.exp(-dt * LAMBDA.pos), kc = 1 - Math.exp(-dt * LAMBDA.col), km = 1 - Math.exp(-dt * LAMBDA.mix), ka = 1 - Math.exp(-dt * LAMBDA.alpha), ke = 1 - Math.exp(-dt * LAMBDA.edge);
    let moving = false, posMoving = false;
    const x = this.x, y = this.y, tx = this.tx, ty = this.ty, c = this.col, tc = this.tcol;
    for (let i = 0; i < N; i++) {
      const dx = tx[i] - x[i], dy = ty[i] - y[i];
      if (dx > 0.03 || dx < -0.03 || dy > 0.03 || dy < -0.03) { x[i] += dx * kp; y[i] += dy * kp; moving = true; posMoving = true; } else { x[i] = tx[i]; y[i] = ty[i]; }
      const o = i * 4;
      for (let q = 0; q < 4; q++) { const dv = tc[o + q] - c[o + q]; const th = q === 3 ? 0.004 : 0.6; if (dv > th || dv < -th) { c[o + q] += dv * kc; moving = true; } else c[o + q] = tc[o + q]; }
    }
    const da = this.dotsAlphaT - this.dotsAlpha; if (Math.abs(da) > 0.004) { this.dotsAlpha += da * ka; moving = true; } else this.dotsAlpha = this.dotsAlphaT;
    if (this.mapFading) { this.mapMix += (1 - this.mapMix) * km; if (this.mapMix > 0.985) { this.mapA = this.mapB; this.mapB = null; this.mapMix = 1; this.mapFading = false; } moving = true; }
    this.posMoving = posMoving; const eT = posMoving ? 0 : this.edgeT;  // рёбра проявляются, когда узлы встали на место
    const de = eT - this.edgeA; if (Math.abs(de) > 0.005) { this.edgeA += de * ke; moving = true; } else this.edgeA = eT;
    if (this.ovA < 0.995) { this.ovA += (1 - this.ovA) * ka; moving = true; } else this.ovA = 1;
    this.drawMap(); this.drawDots();
    if (moving) requestAnimationFrame(this.frame); else this.running = false;
  }
  /* ---- рисование ---- */
  drawMap() {
    if (!this.w || !this.h) return;
    const m = this.map, ctx = m.ctx; m.clear();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    if (this.mapA) { ctx.globalAlpha = this.mapFading ? 1 - this.mapMix : 1; ctx.drawImage(this.mapA, 0, 0); }
    if (this.mapFading && this.mapB) { ctx.globalAlpha = this.mapMix; ctx.drawImage(this.mapB, 0, 0); }
    ctx.globalAlpha = 1;
    if (this.edgeA > 0.01 && this.edgesSpec) this.drawEdges();
    if (this.hover >= 0 && (this.mapA || this.mapB) && this.dotsAlpha < 0.5) m.strokeSet([this.hover], isDark() ? 'rgba(255,255,255,0.9)' : 'rgba(20,22,30,0.9)', 1.4);
  }
  drawEdges() {
    if (this.posMoving) return;
    if (!this.edgeCanvas) {  // один раз: 15 тыс. отрезков в офф-скрин канвас, дальше drawImage с прозрачностью
      const cv = document.createElement('canvas'); cv.width = this.map.canvas.width; cv.height = this.map.canvas.height; const ctx = cv.getContext('2d'); this.map.apply(ctx);
      const E = DATA.graph.edges; const only = this.edgesSpec.only || null; const dark = isDark();
      ctx.lineWidth = 0.6 / this.t.k; ctx.strokeStyle = this.edgesSpec.color || (dark ? 'rgba(170,190,230,0.11)' : 'rgba(40,40,50,0.09)');
      ctx.beginPath(); for (const [i, j] of E) { if (only && !only.has(i) && !only.has(j)) continue; ctx.moveTo(this.tx[i], this.ty[i]); ctx.lineTo(this.tx[j], this.ty[j]); } ctx.stroke();
      if (only) { ctx.lineWidth = 1.3 / this.t.k; ctx.strokeStyle = this.edgesSpec.hiColor || (dark ? 'rgba(255,150,110,0.85)' : 'rgba(235,104,52,0.8)'); ctx.beginPath(); for (const [i, j] of E) { if (only.has(i) || only.has(j)) { ctx.moveTo(this.tx[i], this.ty[i]); ctx.lineTo(this.tx[j], this.ty[j]); } } ctx.stroke(); }
      this.edgeCanvas = cv;
    }
    const c = this.map.ctx; c.setTransform(1, 0, 0, 1, 0, 0); c.globalAlpha = this.edgeA; c.drawImage(this.edgeCanvas, 0, 0); c.globalAlpha = 1;
  }
  drawDots() {
    if (!this.w || !this.h) return;
    const ctx = this.dctx, d = this.map.dpr, t = this.t; ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.clearRect(0, 0, this.dc.width, this.dc.height);
    if (this.paintFn) {  // растровый слой (штрих-код) рисуется один раз в кэш, дальше — drawImage с прозрачностью
      if (this.paintDirty || !this.paintCanvas) { const pc = this.paintCanvas || (this.paintCanvas = document.createElement('canvas')); pc.width = this.dc.width; pc.height = this.dc.height; const pctx = pc.getContext('2d'); pctx.setTransform(1, 0, 0, 1, 0, 0); pctx.clearRect(0, 0, pc.width, pc.height); this.paintFn(pctx, this); this.paintDirty = false; }
      const wp = (this.view && this.view.wipe) ? this.wipeP : 1;
      ctx.globalAlpha = this.ovA;
      if (wp >= 0.999) ctx.drawImage(this.paintCanvas, 0, 0);
      else if (wp > 0) { const cw = Math.round(this.paintCanvas.width * wp); ctx.drawImage(this.paintCanvas, 0, 0, cw, this.paintCanvas.height, 0, 0, cw, this.paintCanvas.height); }
      ctx.globalAlpha = 1;
    }
    if (this.dotsAlpha <= 0.01) return;
    ctx.setTransform(d * t.k, 0, 0, d * t.k, d * t.x, d * t.y);
    const r = this.r, c = this.col, x = this.x, y = this.y, A = this.dotsAlpha, dark = isDark();
    // ореол под точками (дёшево: один проход крупными полупрозрачными кругами) — только в тёмной теме
    if (dark && !this.posMoving) { for (let i = 0; i < N; i++) { const o = i * 4; if (c[o + 3] <= 0.05 || x[i] < 0) continue; ctx.beginPath(); ctx.arc(x[i], y[i], r * 1.9, 0, 6.2832); ctx.fillStyle = `rgba(${c[o] | 0},${c[o + 1] | 0},${c[o + 2] | 0},${(c[o + 3] * A * 0.16).toFixed(3)})`; ctx.fill(); } }
    for (let i = 0; i < N; i++) { const o = i * 4; if (c[o + 3] <= 0.01 || x[i] < 0) continue; ctx.beginPath(); ctx.arc(x[i], y[i], r, 0, 6.2832); ctx.fillStyle = `rgba(${c[o] | 0},${c[o + 1] | 0},${c[o + 2] | 0},${(c[o + 3] * A).toFixed(3)})`; ctx.fill(); }
    if (this.hover >= 0 && c[this.hover * 4 + 3] > 0.2) { const i = this.hover, o = i * 4; ctx.beginPath(); ctx.arc(x[i], y[i], r * 2, 0, 6.2832); ctx.fillStyle = `rgba(${c[o] | 0},${c[o + 1] | 0},${c[o + 2] | 0},1)`; ctx.fill(); ctx.lineWidth = 2 / t.k; ctx.strokeStyle = dark ? 'rgba(255,255,255,0.9)' : 'rgba(255,255,255,0.95)'; ctx.stroke(); }
  }
  hitAt(e) {
    if (!this.w || !this.h) return -1;
    const rect = this.dc.getBoundingClientRect(); const sx = e.clientX - rect.left, sy = e.clientY - rect.top;
    if (this.dotsAlpha > 0.5) { const [fx, fy] = this.map.toFrame(sx, sy); let best = -1, bd = (10 / this.t.k) ** 2; for (let i = 0; i < N; i++) { if (this.col[i * 4 + 3] < 0.2) continue; const dx = this.x[i] - fx, dy = this.y[i] - fy; const dd = dx * dx + dy * dy; if (dd < bd) { bd = dd; best = i; } } return best; }
    return (this.mapA || this.mapB) ? this.map.hit(sx, sy) : -1;
  }
  onMove(e) {
    const i = this.hitAt(e); if (i !== this.hover) { this.hover = i; if (!this.running) { this.drawMap(); this.drawDots(); } }
    if (i >= 0) { const v = this.view || {}; const lens = (v.map && v.map.mode in colorOf) ? v.map.mode : 'macro'; Tip.show(e.clientX, e.clientY, (this.tipFn ? this.tipFn(i) : moTip(MO[i], lens, v.map && v.map.month))); } else Tip.hide();
  }
}
