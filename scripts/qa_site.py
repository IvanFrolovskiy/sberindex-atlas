"""QA лендинга в настоящих браузерах (Playwright): Chromium и WebKit (движок Safari).

Прокручивает историю колесом мыши, как читатель, снимает кадр на каждом шаге (после «успокоения» и в середине
перехода), разделы исследователя, собирает ошибки консоли и статистику длительности кадров при прокрутке.

Запуск: .venv312/bin/python scripts/qa_site.py [--engines chromium,webkit] [--size 1512x982] [--theme dark]
        [--dpr 1] [--out DIR] [--steps 0-27] [--no-mid]
"""
import argparse
import json
import statistics
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site/index.html"

FRAME_PROBE = """
() => { window.__frames = []; let last = performance.now();
  const tick = now => { window.__frames.push(now - last); last = now; requestAnimationFrame(tick); };
  requestAnimationFrame(tick); }
"""


def wheel_to(page, target_y, step=140, pause=0.016):
    """Прокрутка колесом до target_y шагами по step px (как у трекпада), без телепортации."""
    y = page.evaluate("window.scrollY")
    n = 0
    while abs(target_y - y) > 2 and n < 400:
        dy = max(-step, min(step, target_y - y))
        page.mouse.wheel(0, dy)
        time.sleep(pause)
        y = page.evaluate("window.scrollY")
        n += 1
    return y


def run(engine, size, theme, dpr, out, steps, mid, nofx=False):
    out.mkdir(parents=True, exist_ok=True)
    w, h = size
    report = {"engine": engine, "size": f"{w}x{h}", "theme": theme, "console": [], "errors": [], "frames": {}}
    with sync_playwright() as p:
        browser = getattr(p, engine).launch()
        ctx = browser.new_context(viewport={"width": w, "height": h}, device_scale_factor=dpr)
        page = ctx.new_page()
        page.on("console", lambda m: report["console"].append(f"{m.type}: {m.text}") if m.type in ("error", "warning") else None)
        page.on("pageerror", lambda e: report["errors"].append(str(e)))
        page.goto(f"file://{SITE}?theme={theme}" + ("&nofx=1" if nofx else ""), wait_until="load")
        page.evaluate("document.fonts.ready")
        page.wait_for_timeout(1200)
        page.mouse.move(w // 2, h // 2)
        page.screenshot(path=str(out / "hero.png"))
        n_steps = page.evaluate("document.querySelectorAll('#steps .step').length")
        errlog = page.evaluate("document.getElementById('errlog') ? document.getElementById('errlog').textContent : ''")
        if errlog:
            report["errors"].append("errlog: " + errlog)
        page.evaluate(FRAME_PROBE)
        for k in steps:
            if k >= n_steps:
                break
            frac = 0.62 if w <= 980 else 0.45
            target = page.evaluate(f"(() => {{ const el = document.getElementById('step-{k}'); const r = el.getBoundingClientRect(); return r.top + window.scrollY - window.innerHeight * {frac}; }})()")
            wheel_to(page, target)
            if mid:
                page.wait_for_timeout(220)
                page.screenshot(path=str(out / f"step{k:02d}_mid.png"))
            page.wait_for_timeout(1100)
            page.screenshot(path=str(out / f"step{k:02d}.png"))
        n_cards = page.evaluate("document.querySelectorAll('.chapter-card').length")
        for c in range(n_cards):
            target = page.evaluate(f"(() => {{ const el = document.querySelectorAll('.chapter-card')[{c}]; const r = el.getBoundingClientRect(); return r.top + window.scrollY + r.height / 2 - window.innerHeight / 2; }})()")
            wheel_to(page, target, step=400, pause=0.01)
            page.wait_for_timeout(900)
            page.screenshot(path=str(out / f"card{c}.png"))
        frames = page.evaluate("window.__frames.slice(5)")
        if frames:
            frames_sorted = sorted(frames)
            report["frames"] = {"n": len(frames), "p50": round(statistics.median(frames), 1), "p95": round(frames_sorted[int(len(frames) * 0.95) - 1], 1),
                                "max": round(max(frames), 1), "over50ms": sum(1 for f in frames if f > 50), "over100ms": sum(1 for f in frames if f > 100)}
        for sid in ["atlas", "passport", "types", "method"]:
            target = page.evaluate(f"(() => {{ const el = document.getElementById('{sid}'); return el.getBoundingClientRect().top + window.scrollY - 40; }})()")
            wheel_to(page, target, step=400, pause=0.01)
            page.wait_for_timeout(900)
            page.screenshot(path=str(out / f"sec_{sid}.png"))
        errlog = page.evaluate("document.getElementById('errlog') ? document.getElementById('errlog').textContent : ''")
        if errlog:
            report["errors"].append("errlog(end): " + errlog)
        browser.close()
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", default="chromium,webkit")
    ap.add_argument("--size", default="1512x982")
    ap.add_argument("--theme", default="dark")
    ap.add_argument("--dpr", type=float, default=1.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--steps", default="0-27")
    ap.add_argument("--no-mid", action="store_true")
    ap.add_argument("--nofx", action="store_true", help="без размытий и орбов (headless-скриншоты в разы быстрее)")
    a = ap.parse_args()
    w, h = (int(x) for x in a.size.split("x"))
    s0, s1 = (int(x) for x in a.steps.split("-"))
    base = Path(a.out) if a.out else ROOT / "outputs/qa_site"
    for engine in a.engines.split(","):
        out = base / f"{engine}_{a.size}_{a.theme}"
        t0 = time.time()
        rep = run(engine, (w, h), a.theme, a.dpr, out, range(s0, s1 + 1), not a.no_mid, a.nofx)
        print(f"{engine} {a.size} {a.theme}: {time.time() - t0:.0f} с; кадры {rep['frames']}; ошибок {len(rep['errors'])}, консоль {len(rep['console'])} → {out}")
        for e in rep["errors"][:10]:
            print("  !", e[:300])
        for c in rep["console"][:10]:
            print("  ·", c[:300])


if __name__ == "__main__":
    main()
