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

        # scroll to trade strategy
        await page.evaluate("window.scrollTo(0, 600)")
        await page.wait_for_timeout(600)
        await shot(page, "trade_strategy", 2)

        # open AI Thesis
        thesis = page.get_by_text("AI Thesis", exact=False).first
        if await thesis.count() > 0:
            await thesis.click()
            await page.wait_for_timeout(800)
            await shot(page, "thesis", 2)

        # open exit strategy
        exit_exp = page.get_by_text("Exit Strategy", exact=False).first
        if await exit_exp.count() > 0:
            await exit_exp.click()
            await page.wait_for_timeout(600)
            await shot(page, "exit_strategy", 2)

        # open price chart
        await page.evaluate("window.scrollTo(0, 0)")
        chart_exp = page.get_by_text("Price chart", exact=False).first
        if await chart_exp.count() > 0:
            await chart_exp.click()
            await page.wait_for_timeout(1500)
            await page.evaluate("window.scrollTo(0, 700)")
            await shot(page, "price_chart", 3)

        # open fundamentals
        fund_exp = page.get_by_text("Fundamentals", exact=False).first
        if await fund_exp.count() > 0:
            await fund_exp.click()
            await page.wait_for_timeout(800)
            await page.evaluate("window.scrollTo(0, 1200)")
            await shot(page, "fundamentals", 2)
            await page.evaluate("window.scrollTo(0, 2000)")
            await shot(page, "piotroski_altman", 2)

        # analyst section
        analyst_exp = page.get_by_text("Analyst", exact=False).first
        if await analyst_exp.count() > 0:
            await analyst_exp.click()
            await page.wait_for_timeout(800)
            await page.evaluate("window.scrollTo(0, 3000)")
            await shot(page, "analyst", 2)

        # news/sentiment
        news_exp = page.get_by_text("News", exact=False).first
        if await news_exp.count() > 0:
            await news_exp.click()
            await page.wait_for_timeout(800)
            await page.evaluate("window.scrollTo(0, 4000)")
            await shot(page, "news_sentiment", 2)

        # backtest
        await page.evaluate("window.scrollTo(0, 9999)")
        await page.wait_for_timeout(500)
        backtest_exp = page.get_by_text("Backtest", exact=False).first
        if await backtest_exp.count() > 0:
            await backtest_exp.click()
            await page.wait_for_timeout(1500)
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

        # ── Watchlist ──────────────────────────────────────────────────────────
        print("→ Watchlist")
        await page.evaluate("window.scrollTo(0, 0)")
        watchlist_radio = page.get_by_label("⭐ Watchlist")
        if await watchlist_radio.count() > 0:
            await watchlist_radio.click()
        else:
            # fallback: click radio by position
            radios = await page.query_selector_all("input[type=radio]")
            if len(radios) > 1:
                await radios[1].click()
        await page.wait_for_timeout(4000)
        await shot(page, "watchlist", 3)

        # ── Portfolio ──────────────────────────────────────────────────────────
        print("→ Portfolio")
        radios = await page.query_selector_all("input[type=radio]")
        if len(radios) > 2:
            await radios[2].click()
        await page.wait_for_timeout(5000)
        await shot(page, "portfolio_summary", 2)
        await page.evaluate("window.scrollTo(0, 600)")
        await shot(page, "portfolio_positions", 2)
        await page.evaluate("window.scrollTo(0, 9999)")
        await page.wait_for_timeout(2000)
        await shot(page, "correlation_matrix", 3)

        # ── Alerts ────────────────────────────────────────────────────────────
        print("→ Alerts")
        if len(radios) > 3:
            await radios[3].click()
        await page.wait_for_timeout(2000)
        await shot(page, "alerts", 2)

        # ── ELI5 ─────────────────────────────────────────────────────────────
        print("→ ELI5")
        if len(radios) > 4:
            await radios[4].click()
        await page.wait_for_timeout(2000)
        await shot(page, "eli5", 2)
        await page.evaluate("window.scrollTo(0, 600)")
        await shot(page, "eli5_terms", 2)

        await browser.close()
        print(f"✓ {frame} frames saved to docs/demo_frames/")

asyncio.run(main())
PYEOF
