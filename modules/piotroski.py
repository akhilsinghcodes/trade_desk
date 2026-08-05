import pandas as pd
import yfinance as yf
from typing import Tuple, Dict, Optional


def _get_row(df: Optional[pd.DataFrame], key: str) -> Optional[pd.Series]:
    """Get a row from a dataframe by substring match in index (case-insensitive)."""
    if df is None or df.empty:
        return None
    matching = [idx for idx in df.index if key.lower() in idx.lower()]
    if matching:
        return df.loc[matching[0]]
    return None


def _safe_get(series: Optional[pd.Series], idx: int) -> Optional[float]:
    """Safely get a value from a series by index."""
    if series is None or len(series) <= idx:
        return None
    val = series.iloc[idx]
    if pd.isna(val):
        return None
    return float(val)


def get_piotroski(ticker: str) -> dict:
    """
    Calculate the Piotroski F-Score (0-9 point financial health score).

    The score measures financial health across 9 criteria:
    - Profitability (4 points): ROA, cash flow, ROA trend, accruals quality
    - Leverage/Liquidity (3 points): debt ratio, current ratio, share dilution
    - Efficiency (2 points): gross margin, asset turnover

    Args:
        ticker: Stock ticker symbol (e.g., "AAPL")

    Returns:
        Dictionary with:
        - score: int (0-9)
        - components: dict of {criterion_name: bool} for all 9 criteria
        - profitability: int (0-4) sub-score
        - leverage: int (0-3) sub-score
        - efficiency: int (0-2) sub-score
        - interpretation: str ("strong" | "neutral" | "weak")
    """
    try:
        t = yf.Ticker(ticker)
        bs = t.balance_sheet
        inc = t.income_stmt
        cf = t.cashflow

        # Normalize indices (convert to lowercase, replace spaces)
        for df in [bs, inc, cf]:
            if df is not None:
                df.index = df.index.str.lower().str.replace(" ", "_")

        components = {}
        profitability_score = 0
        leverage_score = 0
        efficiency_score = 0

        # ============================================================================
        # PROFITABILITY CRITERIA (4 points)
        # ============================================================================

        # 1. ROA > 0 (net income / total assets > 0)
        try:
            net_income = _get_row(inc, "net_income")
            total_assets = _get_row(bs, "total_assets")

            if net_income is not None and total_assets is not None:
                ni_current = _safe_get(net_income, 0)
                ta_current = _safe_get(total_assets, 0)

                if ni_current is not None and ta_current is not None and ta_current != 0:
                    roa = ni_current / ta_current
                    components["roa_positive"] = roa > 0
                    if components["roa_positive"]:
                        profitability_score += 1
                else:
                    components["roa_positive"] = False
            else:
                components["roa_positive"] = False
        except Exception:
            components["roa_positive"] = False

        # 2. Operating Cash Flow > 0
        try:
            ocf = _get_row(cf, "operating_cash_flow")

            if ocf is not None:
                ocf_current = _safe_get(ocf, 0)
                components["ocf_positive"] = ocf_current is not None and ocf_current > 0
                if components["ocf_positive"]:
                    profitability_score += 1
            else:
                components["ocf_positive"] = False
        except Exception:
            components["ocf_positive"] = False

        # 3. ROA increasing YoY (this year ROA > last year ROA)
        try:
            net_income = _get_row(inc, "net_income")
            total_assets = _get_row(bs, "total_assets")

            if net_income is not None and total_assets is not None:
                ni_current = _safe_get(net_income, 0)
                ni_prior = _safe_get(net_income, 1)
                ta_current = _safe_get(total_assets, 0)
                ta_prior = _safe_get(total_assets, 1)

                if (ni_current is not None and ta_current is not None and ta_current != 0 and
                    ni_prior is not None and ta_prior is not None and ta_prior != 0):
                    roa_current = ni_current / ta_current
                    roa_prior = ni_prior / ta_prior
                    components["roa_increasing"] = roa_current > roa_prior
                    if components["roa_increasing"]:
                        profitability_score += 1
                else:
                    components["roa_increasing"] = False
            else:
                components["roa_increasing"] = False
        except Exception:
            components["roa_increasing"] = False

        # 4. Accruals quality: OCF/Assets > ROA (earnings are backed by cash)
        try:
            ocf = _get_row(cf, "operating_cash_flow")
            total_assets = _get_row(bs, "total_assets")
            net_income = _get_row(inc, "net_income")

            if ocf is not None and total_assets is not None and net_income is not None:
                ocf_current = _safe_get(ocf, 0)
                ta_current = _safe_get(total_assets, 0)
                ni_current = _safe_get(net_income, 0)

                if (ocf_current is not None and ta_current is not None and ta_current != 0 and
                    ni_current is not None):
                    ocf_ratio = ocf_current / ta_current
                    roa = ni_current / ta_current
                    components["accruals_quality"] = ocf_ratio > roa
                    if components["accruals_quality"]:
                        profitability_score += 1
                else:
                    components["accruals_quality"] = False
            else:
                components["accruals_quality"] = False
        except Exception:
            components["accruals_quality"] = False

        # ============================================================================
        # LEVERAGE/LIQUIDITY CRITERIA (3 points)
        # ============================================================================

        # 5. Long-term debt ratio decreased YoY
        try:
            lt_debt = _get_row(bs, "long_term_debt")
            total_assets = _get_row(bs, "total_assets")

            if lt_debt is not None and total_assets is not None:
                ltd_current = _safe_get(lt_debt, 0)
                ltd_prior = _safe_get(lt_debt, 1)
                ta_current = _safe_get(total_assets, 0)
                ta_prior = _safe_get(total_assets, 1)

                if (ltd_current is not None and ta_current is not None and ta_current != 0 and
                    ltd_prior is not None and ta_prior is not None and ta_prior != 0):
                    ltd_ratio_current = ltd_current / ta_current
                    ltd_ratio_prior = ltd_prior / ta_prior
                    components["lt_debt_ratio_decreased"] = ltd_ratio_current < ltd_ratio_prior
                    if components["lt_debt_ratio_decreased"]:
                        leverage_score += 1
                else:
                    components["lt_debt_ratio_decreased"] = False
            else:
                components["lt_debt_ratio_decreased"] = False
        except Exception:
            components["lt_debt_ratio_decreased"] = False

        # 6. Current ratio improved YoY (current assets / current liabilities)
        try:
            current_assets = _get_row(bs, "current_assets")
            current_liabilities = _get_row(bs, "current_liabilities")

            if current_assets is not None and current_liabilities is not None:
                ca_current = _safe_get(current_assets, 0)
                ca_prior = _safe_get(current_assets, 1)
                cl_current = _safe_get(current_liabilities, 0)
                cl_prior = _safe_get(current_liabilities, 1)

                if (ca_current is not None and cl_current is not None and cl_current != 0 and
                    ca_prior is not None and cl_prior is not None and cl_prior != 0):
                    cr_current = ca_current / cl_current
                    cr_prior = ca_prior / cl_prior
                    components["current_ratio_improved"] = cr_current > cr_prior
                    if components["current_ratio_improved"]:
                        leverage_score += 1
                else:
                    components["current_ratio_improved"] = False
            else:
                components["current_ratio_improved"] = False
        except Exception:
            components["current_ratio_improved"] = False

        # 7. No new shares issued YoY (shares outstanding not increased)
        try:
            shares = _get_row(bs, "shares_outstanding")

            if shares is not None:
                shares_current = _safe_get(shares, 0)
                shares_prior = _safe_get(shares, 1)

                if shares_current is not None and shares_prior is not None:
                    components["no_new_shares"] = shares_current <= shares_prior
                    if components["no_new_shares"]:
                        leverage_score += 1
                else:
                    components["no_new_shares"] = False
            else:
                components["no_new_shares"] = False
        except Exception:
            components["no_new_shares"] = False

        # ============================================================================
        # EFFICIENCY CRITERIA (2 points)
        # ============================================================================

        # 8. Gross margin improved YoY (gross profit / revenue)
        try:
            gross_profit = _get_row(inc, "gross_profit")
            revenue = _get_row(inc, "total_revenue")

            if gross_profit is not None and revenue is not None:
                gp_current = _safe_get(gross_profit, 0)
                gp_prior = _safe_get(gross_profit, 1)
                rev_current = _safe_get(revenue, 0)
                rev_prior = _safe_get(revenue, 1)

                if (gp_current is not None and rev_current is not None and rev_current != 0 and
                    gp_prior is not None and rev_prior is not None and rev_prior != 0):
                    gm_current = gp_current / rev_current
                    gm_prior = gp_prior / rev_prior
                    components["gross_margin_improved"] = gm_current > gm_prior
                    if components["gross_margin_improved"]:
                        efficiency_score += 1
                else:
                    components["gross_margin_improved"] = False
            else:
                components["gross_margin_improved"] = False
        except Exception:
            components["gross_margin_improved"] = False

        # 9. Asset turnover improved YoY (revenue / total assets)
        try:
            revenue = _get_row(inc, "total_revenue")
            total_assets = _get_row(bs, "total_assets")

            if revenue is not None and total_assets is not None:
                rev_current = _safe_get(revenue, 0)
                rev_prior = _safe_get(revenue, 1)
                ta_current = _safe_get(total_assets, 0)
                ta_prior = _safe_get(total_assets, 1)

                if (rev_current is not None and ta_current is not None and ta_current != 0 and
                    rev_prior is not None and ta_prior is not None and ta_prior != 0):
                    at_current = rev_current / ta_current
                    at_prior = rev_prior / ta_prior
                    components["asset_turnover_improved"] = at_current > at_prior
                    if components["asset_turnover_improved"]:
                        efficiency_score += 1
                else:
                    components["asset_turnover_improved"] = False
            else:
                components["asset_turnover_improved"] = False
        except Exception:
            components["asset_turnover_improved"] = False

        # ============================================================================
        # CALCULATE TOTAL SCORE AND INTERPRETATION
        # ============================================================================

        total_score = profitability_score + leverage_score + efficiency_score

        # Determine interpretation
        if total_score >= 7:
            interpretation = "strong"
        elif total_score <= 3:
            interpretation = "weak"
        else:
            interpretation = "neutral"

        return {
            "score": total_score,
            "components": components,
            "profitability": profitability_score,
            "leverage": leverage_score,
            "efficiency": efficiency_score,
            "interpretation": interpretation,
        }

    except Exception as e:
        # Return zero score on error
        return {
            "score": 0,
            "components": {},
            "profitability": 0,
            "leverage": 0,
            "efficiency": 0,
            "interpretation": "neutral",
        }


def score_piotroski(data: dict) -> Tuple[str, str, str]:
    """
    Format Piotroski F-Score for signals list.

    Args:
        data: Dictionary returned from get_piotroski()

    Returns:
        Tuple of (label, status, text) for signals list
        - label: "Piotroski F-Score"
        - status: "good" if score >= 7, "warning" if score <= 3, "neutral" otherwise
        - text: Descriptive string with score breakdown
    """
    label = "Piotroski F-Score"
    score = data.get("score", 0)

    # Determine status
    if score >= 7:
        status = "good"
    elif score <= 3:
        status = "warning"
    else:
        status = "neutral"

    # Get sub-scores
    prof = data.get("profitability", 0)
    lev = data.get("leverage", 0)
    eff = data.get("efficiency", 0)

    # Build descriptive text
    if score >= 7:
        health_desc = "strong financial health"
    elif score <= 3:
        health_desc = "weak financial health"
    else:
        health_desc = "moderate financial health"

    text = f"Score {score}/9 — {health_desc}. Profitability: {prof}/4, Leverage: {lev}/3, Efficiency: {eff}/2."

    return (label, status, text)
