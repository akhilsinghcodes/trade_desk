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

mkdir -p "$FRAMES_DIR"
rm -f "$FRAMES_DIR"/*.png

echo "▶ Checking server..."
curl -sf "$BASE_URL" > /dev/null || { echo "ERROR: Streamlit not running on $BASE_URL"; exit 1; }

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

async def click_tab(page, label):
    tab = page.get_by_role("tab", name=label, exact=False).first
    if await tab.count() > 0:
        await tab.click()
        await page.wait_for_timeout(1500)

async def nav_to(page, label):
    await page.evaluate("window.scrollTo(0, 0)")
    await page.locator("[data-testid='stRadio'] label").filter(has_text=label).first.click(force=True)
    await page.wait_for_timeout(3000)

async def set_ticker(page, ticker, wait_ms=8000):
    await page.evaluate("window.scrollTo(0, 0)")
    ticker_box = page.get_by_label("Ticker")
    await ticker_box.click(click_count=3)
    await ticker_box.type(ticker)
    await ticker_box.press("Enter")
    await page.wait_for_timeout(wait_ms)

async def main():
    global frame
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        ctx     = await browser.new_context(
            viewport={"width": W, "height": H},
            color_scheme="dark",
        )
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
        # Headline: BUY/HOLD/SELL built ONLY from the price signals that have
        # real measured predictive power (IC), weighted by that power, with
        # the resulting rule itself backtested and split-half stability
        # checked — not a hand-picked verdict.
        print("→ Analyze — NVDA — validated verdict")
        await set_ticker(page, "NVDA")
        await shot(page, "validated_verdict", 3)

        # Detected setup + real backtest (win rate / Sharpe / stability gate)
        await page.evaluate("window.scrollTo(0, 420)")
        await shot(page, "setup_backtest", 3)

        # Exploratory correlation check — expand it to show the direction
        # column (normal vs contrarian) that explains the verdict's math
        print("→ Correlation check (expanded)")
        corr_expander = page.get_by_text("Exploratory correlation check", exact=False).first
        if await corr_expander.count() > 0:
            await corr_expander.click()
            await page.wait_for_timeout(800)
            await page.evaluate("window.scrollTo(0, 900)")
            await shot(page, "correlation_check", 3)

        # LLM signal summary — plain-English explainer only, no fake decisions
        await page.evaluate("window.scrollTo(0, 1250)")
        await shot(page, "llm_summary", 2)

        # ── Chart / drill-down tabs ──────────────────────────────────────────
        await page.evaluate("window.scrollTo(0, 0)")
        await click_tab(page, "Chart")
        await page.evaluate("window.scrollTo(0, 700)")
        await shot(page, "price_chart", 3)

        await page.evaluate("window.scrollTo(0, 0)")
        await click_tab(page, "Fundamentals")
        await page.evaluate("window.scrollTo(0, 700)")
        await shot(page, "fundamentals", 2)
        await page.evaluate("window.scrollTo(0, 1400)")
        await shot(page, "piotroski_altman", 2)

        await page.evaluate("window.scrollTo(0, 0)")
        await click_tab(page, "Analyst")
        await page.evaluate("window.scrollTo(0, 600)")
        await shot(page, "analyst_insider", 2)

        await page.evaluate("window.scrollTo(0, 0)")
        await click_tab(page, "News")
        await page.evaluate("window.scrollTo(0, 600)")
        await shot(page, "news_sentiment", 2)

        # Backtest tab (inside Analyze) — signal IC table + strategy backtest
        await page.evaluate("window.scrollTo(0, 0)")
        await click_tab(page, "Backtest")
        await page.wait_for_timeout(2000)
        await page.evaluate("window.scrollTo(0, 600)")
        await shot(page, "analyze_backtest_tab", 2)

        # More tab — exit strategy, AI thesis, momentum signals, etc.
        await page.evaluate("window.scrollTo(0, 0)")
        await click_tab(page, "More")
        await page.wait_for_timeout(1000)
        await page.evaluate("window.scrollTo(0, 400)")
        await shot(page, "more_tab", 2)

        # ── Analyze: a SELL case (different sector, shows the verdict engine
        # calling it the other way, not just always bullish) ────────────────
        print("→ Analyze — ADBE — SELL case")
        await set_ticker(page, "ADBE", wait_ms=7000)
        await shot(page, "adbe_sell_verdict", 2)

        # ── Sidebar nav pages ────────────────────────────────────────────────
        print("→ Watchlist")
        await nav_to(page, "⭐ Watchlist")
        await shot(page, "watchlist", 3)

        print("→ Screener")
        await nav_to(page, "🔍 Screener")
        await page.wait_for_timeout(6000)
        await shot(page, "screener", 3)

        print("→ Portfolio")
        await nav_to(page, "💼 Portfolio")
        await page.wait_for_timeout(3000)
        await shot(page, "portfolio_summary", 2)
        await page.evaluate("window.scrollTo(0, 600)")
        await shot(page, "portfolio_positions", 2)
        await page.evaluate("window.scrollTo(0, 9999)")
        await page.wait_for_timeout(2000)
        await shot(page, "correlation_matrix", 2)

        print("→ Alerts")
        await nav_to(page, "🔔 Alerts")
        await shot(page, "alerts", 2)

        print("→ ELI5")
        await nav_to(page, "📚 ELI5")
        await shot(page, "eli5", 2)
        await page.evaluate("window.scrollTo(0, 600)")
        await shot(page, "eli5_terms", 2)

        # Standalone Backtest page — signal IC factor analysis + full
        # strategy comparison (BREAKOUT/PULLBACK/MEAN_REVERSION/RANGE/BREAKDOWN)
        print("→ Backtest (standalone)")
        await nav_to(page, "🧪 Backtest")
        await page.wait_for_timeout(3000)
        await shot(page, "backtest_signal_ic", 2)
        strategy_tab = page.get_by_role("tab", name="Strategy Backtest", exact=False).first
        if await strategy_tab.count() > 0:
            await strategy_tab.click()
            await page.wait_for_timeout(8000)  # vectorbt run takes a few seconds
            await shot(page, "backtest_strategy_comparison", 3)

        # LLM Usage — real logged cost/token telemetry, not a guess
        print("→ LLM Usage")
        await nav_to(page, "💰 LLM Usage")
        await page.wait_for_timeout(1500)
        await shot(page, "llm_usage", 2)

        await browser.close()
        print(f"✓ {frame} frames saved to docs/demo_frames/")

asyncio.run(main())
PYEOF

echo "▶ Assembling GIF with gifski..."
gifski --fps 4 --width 1400 --output "$OUT_GIF" "$FRAMES_DIR"/*.png
echo "✓ GIF written to $OUT_GIF"
