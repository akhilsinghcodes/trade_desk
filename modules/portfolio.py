"""
Portfolio management module for tracking local stock positions.
Stores positions in JSON without brokerage connection.
"""

import json
from pathlib import Path
from datetime import datetime


def _get_portfolio_path() -> Path:
    """Get the path to the portfolio JSON file."""
    return Path(__file__).parent.parent / "data" / "portfolio.json"


def load_portfolio() -> list[dict]:
    """
    Load positions from JSON.

    Returns:
        list[dict]: List of position dictionaries. Empty list if file doesn't exist.
    """
    portfolio_path = _get_portfolio_path()

    # Handle missing file gracefully
    if not portfolio_path.exists():
        return []

    try:
        with open(portfolio_path, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []


def save_portfolio(positions: list[dict]):
    """
    Save positions to JSON.

    Args:
        positions: List of position dictionaries to save.
    """
    portfolio_path = _get_portfolio_path()

    # Ensure data directory exists
    portfolio_path.parent.mkdir(parents=True, exist_ok=True)

    with open(portfolio_path, 'w') as f:
        json.dump(positions, f, indent=2)


def add_position(ticker: str, shares: float, buy_price: float, buy_date: str = "", note: str = "") -> dict:
    """
    Add a new position to the portfolio.

    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL').
        shares: Number of shares.
        buy_price: Purchase price per share.
        buy_date: Purchase date as YYYY-MM-DD. Defaults to today if empty.
        note: Optional note about the position.

    Returns:
        dict: The newly added position.
    """
    # Default buy_date to today if not provided
    if not buy_date:
        buy_date = datetime.now().strftime('%Y-%m-%d')

    position = {
        "ticker": ticker,
        "shares": shares,
        "buy_price": buy_price,
        "buy_date": buy_date,
        "note": note
    }

    # Load existing positions
    positions = load_portfolio()

    # Remove existing position with same ticker if present
    positions = [p for p in positions if p["ticker"] != ticker]

    # Add new position
    positions.append(position)

    # Save updated portfolio
    save_portfolio(positions)

    return position


def remove_position(ticker: str):
    """
    Remove a position by ticker.

    Args:
        ticker: Stock ticker symbol to remove.
    """
    positions = load_portfolio()
    positions = [p for p in positions if p["ticker"] != ticker]
    save_portfolio(positions)


def get_position(ticker: str) -> dict | None:
    """
    Get a single position by ticker.

    Args:
        ticker: Stock ticker symbol.

    Returns:
        dict or None: The position if found, None otherwise.
    """
    positions = load_portfolio()
    for position in positions:
        if position["ticker"] == ticker:
            return position
    return None


def calc_pnl(position: dict, current_price: float) -> dict:
    """
    Calculate P&L for a position.

    Args:
        position: Position dictionary with ticker, shares, buy_price.
        current_price: Current price per share.

    Returns:
        dict: P&L analysis with:
            - ticker: Stock ticker
            - shares: Number of shares
            - buy_price: Original purchase price
            - current_price: Current price
            - cost_basis: Total cost (shares * buy_price)
            - current_value: Current value (shares * current_price)
            - pnl_dollars: Profit/loss in dollars
            - pnl_pct: Profit/loss as percentage
            - status: "profit", "loss", or "breakeven"
    """
    ticker = position["ticker"]
    shares = position["shares"]
    buy_price = position["buy_price"]

    cost_basis = shares * buy_price
    current_value = shares * current_price
    pnl_dollars = current_value - cost_basis
    pnl_pct = (pnl_dollars / cost_basis * 100) if cost_basis != 0 else 0

    # Determine status
    if pnl_dollars > 0.01:  # Small tolerance for floating point
        status = "profit"
    elif pnl_dollars < -0.01:
        status = "loss"
    else:
        status = "breakeven"

    return {
        "ticker": ticker,
        "shares": shares,
        "buy_price": buy_price,
        "current_price": current_price,
        "cost_basis": cost_basis,
        "current_value": current_value,
        "pnl_dollars": pnl_dollars,
        "pnl_pct": pnl_pct,
        "status": status
    }


def portfolio_summary(positions: list[dict], current_prices: dict[str, float]) -> dict:
    """
    Summarize entire portfolio.

    Args:
        positions: List of position dictionaries.
        current_prices: Dict mapping ticker to current price.

    Returns:
        dict: Portfolio summary with:
            - total_cost: Total cost basis
            - total_value: Total current value
            - total_pnl_dollars: Total P&L in dollars
            - total_pnl_pct: Total P&L as percentage
            - positions: List of calc_pnl results for positions with current prices
    """
    total_cost = 0.0
    total_value = 0.0
    position_details = []

    for position in positions:
        ticker = position["ticker"]

        if ticker not in current_prices:
            continue

        current_price = current_prices[ticker]
        pnl_info = calc_pnl(position, current_price)
        position_details.append(pnl_info)

        total_cost += pnl_info["cost_basis"]
        total_value += pnl_info["current_value"]

    total_pnl_dollars = total_value - total_cost
    total_pnl_pct = (total_pnl_dollars / total_cost * 100) if total_cost != 0 else 0

    return {
        "total_cost": total_cost,
        "total_value": total_value,
        "total_pnl_dollars": total_pnl_dollars,
        "total_pnl_pct": total_pnl_pct,
        "positions": position_details
    }
