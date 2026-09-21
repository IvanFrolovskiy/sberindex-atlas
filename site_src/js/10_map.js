/* Карта на canvas: полигоны в координатах кадра (1000×537), Path2D-кэш, HiDPI, трансформ, picking,
   рендер состояния в офф-скрин битмапу и композит двух битмап — для кроссфейда без перерисовки полигонов. */
class MapLayer {
  constructor(canvas) {
    this.canvas = canvas; this.ctx = canvas.getContext('2d');
    this.paths = new Array(N); this.hasGeo = new Array(N).fill(false);
    for (let i = 0; i < N; i++) {
      const g = DATA.geo[MO[i].id]; if (!g) continue;
      const p = new Path2D();
      for (const poly of g) for (const ring of poly) { p.moveTo(ring[0], ring[1]); for (let k = 2; k < ring.length; k += 2) p.lineTo(ring[k], ring[k + 1]); p.closePath(); }
      this.paths[i] = p; this.hasGeo[i] = true;
    }
    this.transform = { k: 1, x: 0, y: 0 }; this.dpr = Math.max(1, Math.min(2, window.devicePixelRatio || 1));
    this.pick = document.createElement('canvas'); this.pickCtx = this.pick.getContext('2d', { willReadFrequently: true });
    this.pickDirty = true; this.pool = []; this.poolIdx = 0;
  }
  resize(w, h) { const d = this.dpr; this.w = w; this.h = h; this.canvas.width = Math.round(w * d); this.canvas.height = Math.round(h * d); this.pick.width = Math.round(w * d); this.pick.height = Math.round(h * d); this.pickDirty = true; this.pool = []; }
  /* вписать кадр в бокс; pad — число или {top,right,bottom,left} */
  fit(frameW, frameH, pad = 8) {
    const p = typeof pad === 'number' ? { top: pad, right: pad, bottom: pad, left: pad } : { top: 8, right: 8, bottom: 8, left: 8, ...pad };
    const k = Math.min((this.w - p.left - p.right) / frameW, (this.h - p.top - p.bottom) / frameH);
    this.transform = { k, x: p.left + (this.w - p.left - p.right - frameW * k) / 2, y: p.top + (this.h - p.top - p.bottom - frameH * k) / 2 }; this.pickDirty = true; return this.transform;
  }
  apply(ctx) { const d = this.dpr, t = this.transform; ctx.setTransform(d * t.k, 0, 0, d * t.k, d * t.x, d * t.y); }
  clear() { const ctx = this.ctx; ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.clearRect(0, 0, this.canvas.width, this.canvas.height); }
  /* fillFn(i) -> css color | null (не рисовать); рисует на переданном контексте (по умолчанию — основной) */
  draw(fillFn, { stroke = 'rgba(255,255,255,0.75)', lineWidth = 0.5, alpha = 1, only = null, ctx = null, keep = false } = {}) {
    const c = ctx || this.ctx; if (!keep) { if (!ctx) this.clear(); else { c.setTransform(1, 0, 0, 1, 0, 0); c.clearRect(0, 0, c.canvas.width, c.canvas.height); } }
    this.apply(c); c.globalAlpha = alpha;
    c.lineWidth = lineWidth / this.transform.k; c.strokeStyle = stroke; c.lineJoin = 'round';
    for (let i = 0; i < N; i++) { if (!this.hasGeo[i]) continue; if (only && !only.has(i)) continue; const col = fillFn(i); if (!col) continue; c.fillStyle = col; c.fill(this.paths[i]); if (stroke) c.stroke(this.paths[i]); }
    c.globalAlpha = 1;
  }
  /* офф-скрин канвас из пула (две штуки по кругу — для композитов при наложении переходов) */
  scratch() {
    const idx = this.poolIdx++ % 2; let cv = this.pool[idx];
    if (!cv) { cv = document.createElement('canvas'); this.pool[idx] = cv; }
    if (cv.width !== this.canvas.width || cv.height !== this.canvas.height) { cv.width = this.canvas.width; cv.height = this.canvas.height; }
    return cv;
  }
  /* состояние карты → битмапа (свой канвас: живёт в кэше сцены; пул — только для композитов) */
  render(fillFn, opts = {}) { const cv = document.createElement('canvas'); cv.width = this.canvas.width; cv.height = this.canvas.height; this.draw(fillFn, { ...opts, ctx: cv.getContext('2d') }); return cv; }
  /* инкрементально: копия base + перекраска только изменившихся полигонов (месяцы карты отличаются ~30 МО из 2 016) */
  renderFrom(base, fillFn, changed, opts = {}) { const cv = document.createElement('canvas'); cv.width = this.canvas.width; cv.height = this.canvas.height; const c = cv.getContext('2d'); c.setTransform(1, 0, 0, 1, 0, 0); c.drawImage(base, 0, 0); this.draw(fillFn, { ...opts, ctx: c, only: changed, keep: true }); return cv; }
  /* смесь двух битмап (a·(1−mix) + b·mix) → новая битмапа; нужна, когда новый вид приходит посреди кроссфейда */
  composite(a, b, mix) {
    const cv = this.scratch(); const c = cv.getContext('2d'); c.setTransform(1, 0, 0, 1, 0, 0); c.clearRect(0, 0, cv.width, cv.height);
    if (a) { c.globalAlpha = 1 - mix; c.drawImage(a, 0, 0); } if (b) { c.globalAlpha = mix; c.drawImage(b, 0, 0); } c.globalAlpha = 1; return cv;
  }
  strokeSet(set, color, width = 1.5) { const ctx = this.ctx; this.apply(ctx); ctx.lineWidth = width / this.transform.k; ctx.strokeStyle = color; ctx.lineJoin = 'round'; for (const i of set) if (this.hasGeo[i]) ctx.stroke(this.paths[i]); }
  buildPick() { const ctx = this.pickCtx; ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.clearRect(0, 0, this.pick.width, this.pick.height); this.apply(ctx); for (let i = 0; i < N; i++) { if (!this.hasGeo[i]) continue; const id = i + 1; ctx.fillStyle = `rgb(${(id >> 16) & 255},${(id >> 8) & 255},${id & 255})`; ctx.fill(this.paths[i]); } this.pickDirty = false; }
  /* индекс МО под точкой (в css-пикселях канваса) или -1 */
  hit(x, y) { if (this.pickDirty) this.buildPick(); const d = this.dpr; const p = this.pickCtx.getImageData(Math.round(x * d), Math.round(y * d), 1, 1).data; if (p[3] === 0) return -1; return ((p[0] << 16) | (p[1] << 8) | p[2]) - 1; }
  /* кадр → экран */
  toScreen(fx, fy) { const t = this.transform; return [fx * t.k + t.x, fy * t.k + t.y]; }
  toFrame(sx, sy) { const t = this.transform; return [(sx - t.x) / t.k, (sy - t.y) / t.k]; }
}
