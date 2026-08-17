"""
Fetch a ticker universe from SEC EDGAR — free, no API key, no payment.

SEC's company_tickers.json lists every registrant with a ticker, roughly
ordered by filer size (largest first). We take the first N, drop
share-class tickers with dots/dashes (BRK.A, BF.B — these trip up
yfinance's default symbol format) and dedupe.

Usage:
  .venv/bin/python fetch_edgar_universe.py                  # top 200 -> tickers.txt
  .venv/bin/python fetch_edgar_universe.py --limit 500
  .venv/bin/python fetch_edgar_universe.py --output my_universe.txt

Then train with it:
  .venv/bin/python train.py --tickers-file tickers.txt
  .venv/bin/python train_return.py --tickers $(cat tickers.txt)   # or edit train_return.py to accept --tickers-file too
"""
import argparse
import json
import urllib.request

EDGAR_URL = "https://www.sec.gov/files/company_tickers.json"
# SEC requires an identifying User-Agent (name + contact) or it will block requests.
USER_AGENT = "trade_ml admin@example.com"


def fetch_edgar_universe(limit: int = 200) -> list[str]:
    req = urllib.request.Request(EDGAR_URL, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())

    tickers = []
    seen = set()
    for entry in data.values():
        t = entry.get("ticker", "").upper().strip()
        if not t or "." in t or "-" in t:
            continue  # skip share-class tickers (BRK.A, BF.B) — yfinance format mismatch
        if t in seen:
            continue
        seen.add(t)
        tickers.append(t)
        if len(tickers) >= limit:
            break
    return tickers


def main():
    parser = argparse.ArgumentParser(description="Fetch a ticker universe from SEC EDGAR")
    parser.add_argument("--limit", type=int, default=200, help="Number of tickers (default 200)")
    parser.add_argument("--output", default="tickers.txt")
    args = parser.parse_args()

    print("Fetching SEC EDGAR company list...")
    tickers = fetch_edgar_universe(limit=args.limit)
    print(f"  Got {len(tickers)} tickers (requested {args.limit})")

    with open(args.output, "w") as f:
        f.write("\n".join(tickers) + "\n")
    print(f"  Saved to {args.output}")
    print(f"\n  Run: .venv/bin/python train.py --tickers-file {args.output}")


if __name__ == "__main__":
    main()
