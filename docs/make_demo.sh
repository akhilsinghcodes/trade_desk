#!/usr/bin/env bash
# TradeDesk demo GIF generator
# Requires: gifski, a running Streamlit server on :8501
# Usage: bash docs/make_demo.sh

set -e

FRAMES_DIR="$(dirname "$0")/demo_frames"
OUT_GIF="$(dirname "$0")/demo.gif"
BASE_URL="http://localhost:8501"
W=1400
H=860
DELAY=0   # gifski handles timing via --fps

mkdir -p "$FRAMES_DIR"
rm -f "$FRAMES_DIR"/*.png

echo "▶ Checking server..."
curl -sf "$BASE_URL" > /dev/null || { echo "ERROR: Streamlit not running on $BASE_URL"; exit 1; }

# Helper: capture browser window via screencapture (finds Chrome/Safari with localhost:8501)
# We use Python + Playwright for headless browser screenshots (more reliable than screencapture)
# Install once: pip install playwright && playwright install chromium

PYTHON=".venv/bin/python"
$PYTHON - << 'PYEOF'
import asyncio, sys, os
from pathlib import Path

try:
    from playwright.async_api import async_playwright
except ImportError:
    print("Installing playwright...")
    os.system(".venv/bin/pip install playwright -q && .venv/bin/playwright install chromium --with-deps -q")
    from playwright.async_api import async_playwright

FRAMES = Path("docs/demo_frames")
BASE   = "http://localhost:8501"
W, H   = 1400, 860
frame  = 0

def pad(n):
    return f"{n:04d}"

async def shot(page, name, count=1):
    global frame
    await page.wait_for_timeout(800)
    for _ in range(count):
        await page.screenshot(path=str(FRAMES / f"{pad(frame)}_{name}.png"))
        frame += 1

async def click_radio(page, label):
    """Click sidebar nav radio by visible label text."""
    await page.get_by_label(label).click()
    await page.wait_for_timeout(2000)

async def main():
    global frame
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        ctx     = await browser.new_context(viewport={"width": W, "height": H})
        page    = await ctx.new_page()

        print("→ Loading app...")
        await page.goto(BASE, wait_until="networkidle")
        await page.wait_for_timeout(3000)

        # ── Disclaimer ────────────────────────────────────────────────────────
        print("→ Disclaimer screen")
        await shot(page, "disclaimer", 2)
        btn = page.get_by_text("I understand")
        if await btn.count() > 0:
            await btn.click()
            await page.wait_for_timeout(2000)

        # ── Analyze: NVDA ─────────────────────────────────────────────────────
        print("→ Analyze — NVDA")
        ticker_box = page.get_by_label("Ticker")
        await ticker_box.click(click_count=3)
        await ticker_box.type("NVDA")
        await ticker_box.press("Enter")
        await page.wait_for_timeout(8000)   # wait for full data load
        await shot(page, "analyze_verdict", 3)

        # scroll to trade strategy + thesis (always visible)
        await page.evaluate("window.scrollTo(0, 500)")
        await page.wait_for_timeout(600)
        await shot(page, "trade_strategy", 2)

        # helper: click a tab by its label text
        async def click_tab(label):
            tab = page.get_by_role("tab", name=label, exact=False).first
            if await tab.count() > 0:
                await tab.click()
                await page.wait_for_timeout(1500)

        # Chart tab (default, scroll to chart)
        await page.evaluate("window.scrollTo(0, 0)")
        await click_tab("Chart")
        await page.evaluate("window.scrollTo(0, 700)")
        await shot(page, "price_chart", 3)

        # Fundamentals tab
        await page.evaluate("window.scrollTo(0, 0)")
        await click_tab("Fundamentals")
        await page.evaluate("window.scrollTo(0, 700)")
        await shot(page, "fundamentals", 2)
        await page.evaluate("window.scrollTo(0, 1400)")
        await shot(page, "piotroski_altman", 2)

        # Analyst & Insider tab
        await page.evaluate("window.scrollTo(0, 0)")
        await click_tab("Analyst")
        await page.evaluate("window.scrollTo(0, 600)")
        await shot(page, "analyst", 2)

        # News tab
        await page.evaluate("window.scrollTo(0, 0)")
        await click_tab("News")
        await page.evaluate("window.scrollTo(0, 600)")
        await shot(page, "news_sentiment", 2)

        # Backtest tab
        await page.evaluate("window.scrollTo(0, 0)")
        await click_tab("Backtest")
        await page.wait_for_timeout(2000)
        await page.evaluate("window.scrollTo(0, 600)")
        await shot(page, "backtest", 2)

        # ── Analyze: JPM (different sector) ───────────────────────────────────
        print("→ Analyze — JPM")
        await page.evaluate("window.scrollTo(0, 0)")
        ticker_box = page.get_by_label("Ticker")
        await ticker_box.click(click_count=3)
        await ticker_box.type("JPM")
        await ticker_box.press("Enter")
        await page.wait_for_timeout(7000)
        await shot(page, "jpm_verdict", 2)

        # helper: navigate sidebar by label text
        # Streamlit radio inputs have overlay divs — use JS to set value + dispatch event
        pages_list = ["📊 Analyze", "⭐ Watchlist", "🔍 Screener", "💼 Portfolio", "🔔 Alerts", "📚 ELI5"]
        async def nav_to(label):
            await page.evaluate("window.scrollTo(0, 0)")
            # Click the div that contains the label text (the styled radio label)
            await page.locator(f"[data-testid='stRadio'] label").filter(has_text=label).first.click(force=True)
            await page.wait_for_timeout(3000)

        # ── Watchlist ──────────────────────────────────────────────────────────
        print("→ Watchlist")
        await nav_to("⭐ Watchlist")
        await shot(page, "watchlist", 3)

        # ── Screener ───────────────────────────────────────────────────────────
        print("→ Screener")
        await nav_to("🔍 Screener")
        await page.wait_for_timeout(6000)  # parallel fetch takes time
        await shot(page, "screener", 3)

        # ── Portfolio ──────────────────────────────────────────────────────────
        print("→ Portfolio")
        await nav_to("💼 Portfolio")
        await page.wait_for_timeout(3000)
        await shot(page, "portfolio_summary", 2)
        await page.evaluate("window.scrollTo(0, 600)")
        await shot(page, "portfolio_positions", 2)
        await page.evaluate("window.scrollTo(0, 9999)")
        await page.wait_for_timeout(2000)
        await shot(page, "correlation_matrix", 3)

        # ── Alerts ────────────────────────────────────────────────────────────
        print("→ Alerts")
        await nav_to("🔔 Alerts")
        await shot(page, "alerts", 2)

        # ── ELI5 ─────────────────────────────────────────────────────────────
        print("→ ELI5")
        await nav_to("📚 ELI5")
        await shot(page, "eli5", 2)
        await page.evaluate("window.scrollTo(0, 600)")
        await shot(page, "eli5_terms", 2)

        await browser.close()
        print(f"✓ {frame} frames saved to docs/demo_frames/")

asyncio.run(main())
PYEOF

echo "▶ Assembling GIF with gifski..."
gifski --fps 4 --width 1400 --output "$OUT_GIF" "$FRAMES_DIR"/*.png
echo "✓ GIF written to $OUT_GIF"
