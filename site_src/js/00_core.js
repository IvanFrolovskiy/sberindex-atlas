/* Ядро: данные, форматирование, цвета, тема, тултип. */
'use strict';
const T = DATA.types, MO = DATA.mo, MONTHS = T.months, N = MO.length;
const byId = new Map(MO.map((m, i) => [m.id, i]));
const REDUCED = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
if (new URLSearchParams(location.search).has('nosmooth')) document.documentElement.style.scrollBehavior = 'auto';
if (new URLSearchParams(location.search).has('nofx')) document.body.classList.add('nofx');

const fmt = {
  int: x => x == null ? '—' : Math.round(x).toLocaleString('ru-RU'),
  rub: x => x == null ? '—' : Math.round(x).toLocaleString('ru-RU') + ' ₽',
  pct: (x, d = 0) => x == null ? '—' : (x * 100).toFixed(d).replace('.', ',') + '%',
  pp: (x, d = 1) => x == null ? '—' : (x >= 0 ? '+' : '−') + Math.abs(x).toFixed(d).replace('.', ',') + ' п.п.',
  dev: x => x == null ? '—' : (x >= 0 ? '+' : '−') + Math.abs(x * 100).toFixed(0) + '%',
  num: (x, d = 2) => x == null ? '—' : x.toFixed(d).replace('.', ','),
  month: m => { const [y, mm] = m.split('-'); return ['янв', 'фев', 'мар', 'апр', 'май', 'июн', 'июл', 'авг', 'сен', 'окт', 'ноя', 'дек'][+mm - 1] + ' ' + y; },
  kilo: x => x == null ? '—' : (x >= 1e6 ? (x / 1e6).toFixed(1).replace('.', ',') + ' млн' : x >= 1e3 ? Math.round(x / 1e3) + ' тыс.' : Math.round(x)),
};
const shortName = m => m.s || m.n;

/* цвета типов: макро — палитра темы; подтипы — оттенки родителя; поведение — своя палитра. Массивы живые: при смене темы перекрашиваются на месте. */
const PALETTE = { light: ['#2a78d6', '#eb6834', '#1baf7a', '#4a3aa7', '#eda100', '#e87ba4'], dark: ['#4f8ef0', '#e25a2c', '#1aa070', '#8676ec', '#cf850c', '#d84f8c'] };
const _palIdx = c => PALETTE.light.indexOf(String(c).toLowerCase());
const themePal = () => PALETTE[document.documentElement.dataset.theme === 'light' ? 'light' : 'dark'];
const MACRO_C = T.macro.colors.map(c => { const j = _palIdx(c); return j >= 0 ? themePal()[j] : c; });
const BEH_C = T.beh.colors.map(c => { const j = _palIdx(c); return j >= 0 ? themePal()[j] : c; });
const subParent = [], subShade = [];
(function () {
  const ct = T.crosstab_macro_sub; const seen = {};
  ct.columns.forEach((sj, c) => { let best = -1, bi = 0; ct.index.forEach((mi, r) => { if (ct.data[r][c] > best) { best = ct.data[r][c]; bi = mi; } }); subParent[sj] = bi; seen[bi] = (seen[bi] || 0) + 1; subShade[sj] = seen[bi]; });
})();
function mix(hex, f) { const n = parseInt(hex.slice(1), 16); const r = n >> 16, g = (n >> 8) & 255, b = n & 255; const m = v => Math.round(v + (255 - v) * f); return `rgb(${m(r)},${m(g)},${m(b)})`; }
const subColor = j => subShade[j] <= 1 ? MACRO_C[subParent[j]] : mix(MACRO_C[subParent[j]], subShade[j] === 2 ? 0.42 : 0.66);
const SUB_C = T.sub.names.map((_, j) => subColor(j));
function recolor() { const pal = themePal(); T.macro.colors.forEach((c, i) => { const j = _palIdx(c); if (j >= 0) MACRO_C[i] = pal[j]; }); T.beh.colors.forEach((c, i) => { const j = _palIdx(c); if (j >= 0) BEH_C[i] = pal[j]; }); T.sub.names.forEach((_, j) => { SUB_C[j] = subColor(j); }); }
const colorOf = { macro: i => MACRO_C[i], sub: i => SUB_C[i], beh: i => BEH_C[i] };
const nameOf = { macro: i => T.macro.names[i], sub: i => T.sub.names[i], beh: i => T.beh.names[i] };
const GREY = '#c9c8c3', GREY_DARK = '#4a4b52';
const isDark = () => document.documentElement.dataset.theme !== 'light';
const cssVar = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

/* последовательности типов по месяцам */
const seqOf = (m, lens) => (lens === 'macro' ? m.ms : lens === 'sub' ? m.ss : m.bs) || '';
const typeAt = (m, lens, t) => { const s = seqOf(m, lens); return s ? +s[t] : (lens === 'macro' ? m.ma : lens === 'sub' ? m.su : m.be); };
const modalOf = (m, lens) => lens === 'macro' ? m.ma : lens === 'sub' ? m.su : m.be;

/* тема */
(function () {
  const btn = document.getElementById('themeBtn');
  const qs = new URLSearchParams(location.search).get('theme');
  const saved = qs || (() => { try { return localStorage.getItem('atlas-theme'); } catch (e) { return null; } })();
  document.documentElement.dataset.theme = saved === 'light' ? 'light' : 'dark';
  recolor();
  const label = () => { const t = isDark() ? 'Включить светлую тему' : 'Включить тёмную тему'; btn.title = t; btn.setAttribute('aria-label', t); };
  btn.onclick = () => { const next = isDark() ? 'light' : 'dark'; document.documentElement.dataset.theme = next; try { localStorage.setItem('atlas-theme', next); } catch (e) { } recolor(); label(); window.dispatchEvent(new Event('themechange')); };
  label();
})();

/* тултип */
const Tip = (function () {
  const el = document.getElementById('tip');
  function show(x, y, html) { el.innerHTML = html; el.classList.add('on'); place(x, y); }
  function place(x, y) { const w = el.offsetWidth, h = el.offsetHeight; let L = x + 14, Tt = y + 14; if (L + w > innerWidth - 8) L = x - w - 14; if (Tt + h > innerHeight - 8) Tt = y - h - 14; el.style.left = L + 'px'; el.style.top = Tt + 'px'; }
  function hide() { el.classList.remove('on'); }
  return { show, hide, place };
})();
/* короткий кроссфейд текста: класс .swap-out гасит элемент, через 170 мс текст меняется и элемент проявляется */
function swapText(el, text) { if (el.textContent === text) return; if (REDUCED) { el.textContent = text; return; } el.classList.add('swap-out'); clearTimeout(el._swap); el._swap = setTimeout(() => { el.textContent = text; el.classList.remove('swap-out'); }, 170); }
const esc = s => String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
function moTip(m, lens, month) {
  const t = month == null ? modalOf(m, lens) : typeAt(m, lens, month);
  return `<b>${esc(m.n)}</b><span class="muted">${esc(m.r)}</span><div class="row"><span>${esc(nameOf[lens](t))}</span></div><div class="row"><span>траты 2024</span><span>${fmt.rub(m.t24)}</span></div><div class="row"><span>население</span><span>${fmt.int(m.pop)}</span></div>`;
}
