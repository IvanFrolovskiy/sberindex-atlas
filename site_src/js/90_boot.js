/* Загрузка: пролог, сцена, история, навигация. */
(function () {
  // пролог: тёмная карта, «загорающаяся» типами
  const hc = document.getElementById('heroCanvas'); const hero = new MapLayer(hc);
  function heroDraw(p) {
    const sec = hc.parentElement; hero.resize(sec.clientWidth, sec.clientHeight);
    const k = Math.min((sec.clientWidth * 0.98) / DATA.frame.w, (sec.clientHeight * 0.92) / DATA.frame.h);
    hero.transform = { k, x: sec.clientWidth - DATA.frame.w * k - sec.clientWidth * 0.01, y: (sec.clientHeight - DATA.frame.h * k) * 0.35 }; hero.pickDirty = true;
    hero.draw(i => { const c = d3.rgb(MACRO_C[MO[i].ma]); const a = 0.25 + 0.6 * p; return `rgba(${c.r},${c.g},${c.b},${a})`; }, { stroke: cssVar('--hero-stroke') || 'rgba(13,26,46,0.85)', lineWidth: 0.5 });
  }
  let hp = 0; const t0 = performance.now(); const heroAnim = () => { hp = REDUCED ? 1 : Math.min(1, (performance.now() - t0) / 2600); heroDraw(d3.easeCubicOut(hp)); if (hp < 1) requestAnimationFrame(heroAnim); };
  heroAnim(); new ResizeObserver(() => heroDraw(hp)).observe(hc.parentElement);
  const heroEl = hc.parentElement; let heroY = -1;
  const heroScroll = () => { const y = Math.min(window.scrollY, innerHeight); if (y !== heroY) { heroY = y; heroEl.style.setProperty('--hy', y.toFixed(0)); } };
  window.addEventListener('scroll', heroScroll, { passive: true }); heroScroll();
  window.addEventListener('themechange', () => heroDraw(hp));  // пролог перекрашивается вместе с темой
  const never = MO.filter(m => m.sw[0] === 0).length / N;
  const int0 = v => String(Math.round(v));
  const KPI = [[N, fmt.int, 'муниципальных образований'], [24, int0, 'месяца, 2023–2024'], [4, int0, 'типа местных экономик'], [never, v => fmt.pct(v), 'не меняли тип за два года']];
  document.getElementById('heroKpis').innerHTML = KPI.map(([v, f, l]) => `<div class="kpi"><b>${f(v)}</b><span>${l}</span></div>`).join('');
  if (!REDUCED) {  // числа пролога набегают от нуля
    const bs = [...document.querySelectorAll('#heroKpis b')];
    const t1 = performance.now(), dur = 1400;
    const tick = now => { const p = Math.min(1, (now - t1) / dur), e = d3.easeCubicOut(p);
      bs.forEach((b, i) => { const [v, f] = KPI[i]; b.textContent = f(v * e); });
      if (p < 1) requestAnimationFrame(tick); };
    requestAnimationFrame(tick);
  }

  // сцена и история
  window.stage = new Stage();
  Story.init(window.stage);

  // навигация: подсветка активного раздела
  const links = [...document.querySelectorAll('#nav a')];
  const secs = links.map(a => document.querySelector(a.getAttribute('href'))).filter(Boolean);
  const io = new IntersectionObserver(es => { es.forEach(e => { if (e.isIntersecting) { links.forEach(a => a.classList.toggle('on', a.getAttribute('href') === '#' + e.target.id)); } }); }, { rootMargin: '-40% 0px -55% 0px' });
  secs.forEach(s => io.observe(s));
  Explorer.init();
  // разделы исследователя: карточки выезжают при появлении в экране
  if (!REDUCED) {
    const mark = () => { document.querySelectorAll('.section .card, .pipeline .pstep, .section h2, .section .lead, .atlas, .toolbar').forEach(el => el.classList.add('rise')); };
    mark();
    const io2 = new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting) { e.target.classList.add('in'); io2.unobserve(e.target); } }), { rootMargin: '0px 0px -12% 0px', threshold: 0.05 });
    document.querySelectorAll('.rise').forEach((el, i) => { el.style.transitionDelay = (i % 6) * 55 + 'ms'; io2.observe(el); });
  }
  // печатный режим: ?only=<id раздела> — показать один раздел исследователя (для скриншотов build_print.py)
  const only = new URLSearchParams(location.search).get('only');
  if (only && document.getElementById(only)) { document.body.classList.add('print-only'); document.getElementById(only).classList.add('print-on'); window.dispatchEvent(new Event('resize'));
  }
})();
