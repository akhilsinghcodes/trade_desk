"""
Portfolio-construction layer on top of return_model.pkl's predictions.

Everything runs through ONE simulation function (`simulate_portfolio`) with
two switches (hysteresis on/off, inverse-vol weighting on/off) so all 2x2
cells share identical mechanics — no chance of the kind of silent
divergence that produced an implausible 50% annualized vol in an earlier,
separately-coded hysteresis path.

Every day's simulated position carries explicit gross/net exposure and
name-count diagnostics, checked before trusting any Sharpe number — a
long-short book should be dollar-neutral (net ~0) by construction; if it
isn't, you're measuring market beta, not alpha.
"""
import numpy as np
import pandas as pd


def _daily_return_pivot(oos_df: pd.DataFrame) -> pd.DataFrame:
    """date x ticker matrix of next-day close-to-close returns."""
    px = oos_df.pivot_table(index="date", columns="ticker", values="close", aggfunc="last")
    px = px.sort_index()
    return px.pct_change().shift(-1)  # return realized FROM this date TO next date


def compute_trailing_beta(oos_df: pd.DataFrame, spy_returns: pd.Series, window: int = 60) -> pd.DataFrame:
    """
    date x ticker matrix of trailing `window`-day beta vs the benchmark,
    for beta-neutralizing a leg. Uses same-day-realized returns (the
    _daily_return_pivot convention: return realized FROM date d TO d+1)
    so beta at date d uses only information available up to d.
    """
    ret_pivot = oos_df.pivot_table(index="date", columns="ticker", values="close", aggfunc="last").sort_index().pct_change()
    spy = spy_returns.reindex(ret_pivot.index)
    cov = ret_pivot.rolling(window).cov(spy)
    var = spy.rolling(window).var()
    beta = cov.div(var, axis=0)
    return beta


def block_bootstrap_sharpe(daily_returns: pd.Series, block_size: int = 20, n_boot: int = 1000,
                            seed: int = 42) -> dict:
    """
    Block bootstrap (not iid resampling — daily returns here are built from
    overlapping N-day-forward signals, so they're autocorrelated; blocks of
    `block_size` preserve that structure better than resampling single days).
    Returns the 5th/50th/95th percentile of annualized Sharpe across
    `n_boot` resampled paths — an honest interval, not an asymptotic SE.
    """
    rng = np.random.default_rng(seed)
    vals = daily_returns.dropna().values
    n = len(vals)
    if n < block_size * 5:
        return {}

    n_blocks = int(np.ceil(n / block_size))
    sharpes = []
    for _ in range(n_boot):
        starts = rng.integers(0, n - block_size, size=n_blocks)
        sample = np.concatenate([vals[s:s + block_size] for s in starts])[:n]
        mu, sigma = sample.mean() * 252, sample.std() * np.sqrt(252)
        sharpes.append(mu / sigma if sigma > 0 else 0.0)

    sharpes = np.array(sharpes)
    return {
        "p5": float(np.percentile(sharpes, 5)),
        "p50": float(np.percentile(sharpes, 50)),
        "p95": float(np.percentile(sharpes, 95)),
    }


def simulate_portfolio(oos_df: pd.DataFrame, use_hysteresis: bool = False,
                        vol_weight: bool = False, entry_frac: float = 0.1,
                        exit_frac: float = 0.4, cap_mult: float = 1.5,
                        vol_col: str = "std_20", top_n: int | None = None,
                        rebalance_every: int = 1, beta_pivot: pd.DataFrame | None = None) -> dict:
    """
    One simulation path covering every configuration tested this round:
    - top_n overrides entry_frac-based decile sizing with a fixed name count
      (e.g. quintile-equivalent ~20 names instead of decile's ~12) — more
      names lowers idiosyncratic vol without holding anything stale, unlike
      hysteresis which failed that exact test.
    - rebalance_every: only re-select names every N trading days (holding
      prior selection between rebalances) — trades at the model's actual
      signal cadence instead of daily. Weights (vol or equal) still update
      daily using that day's data even when the NAME selection doesn't.
    - beta_pivot: if given, scale the short leg's overall size so realized
      portfolio beta ≈ 0 (using each day's long/short leg average trailing
      beta), trading exact dollar-neutrality for beta-neutrality — reported
      net_exposure will deviate from 0 by design once this is on; net_beta
      is the diagnostic that matters in that mode instead.

    Both legs are weight-normalized to sum to 1 independently by default
    (dollar-neutral, net=0, gross=2) unless beta_pivot rescales the short
    leg for beta-neutrality instead.
    """
    oos_df = oos_df.copy()
    oos_df["date"] = pd.to_datetime(oos_df["date"])
    rank_pivot = oos_df.pivot_table(index="date", columns="ticker", values="pred_return", aggfunc="last").sort_index()
    ret_pivot = _daily_return_pivot(oos_df).reindex(rank_pivot.index)
    vol_pivot = None
    if vol_weight:
        if vol_col not in oos_df.columns:
            raise ValueError(f"{vol_col} not in oos_df")
        vol_pivot = oos_df.pivot_table(index="date", columns="ticker", values=vol_col, aggfunc="last")

    long_held, short_held = set(), set()
    port_rets, diag_rows, turnovers = [], [], []
    dates = rank_pivot.index

    for i, d in enumerate(dates):
        row = rank_pivot.loc[d].dropna()
        if len(row) < 10:
            continue
        n = len(row)
        pct = row.rank(pct=True)
        target_n = top_n if top_n is not None else max(1, int(n * entry_frac))

        prev_long_held, prev_short_held = long_held, short_held
        is_rebalance_day = (i % rebalance_every == 0)

        if is_rebalance_day:
            if use_hysteresis:
                entry_long = set(pct[pct >= 1 - entry_frac].index)
                exit_long_ok = set(pct[pct >= 1 - exit_frac].index)
                entry_short = set(pct[pct <= entry_frac].index)
                exit_short_ok = set(pct[pct <= exit_frac].index)
                long_held = ((long_held & exit_long_ok) | entry_long) & set(row.index)
                short_held = ((short_held & exit_short_ok) | entry_short) & set(row.index)
                cap = int(target_n * cap_mult)
                if len(long_held) > cap:
                    long_held = set(pct.reindex(long_held).sort_values(ascending=False).head(cap).index)
                if len(short_held) > cap:
                    short_held = set(pct.reindex(short_held).sort_values(ascending=True).head(cap).index)
            else:
                ranked = row.sort_values()
                long_held = set(ranked.tail(target_n).index)
                short_held = set(ranked.head(target_n).index)
        else:
            # not a rebalance day: hold prior names, drop any that left the universe
            long_held = prev_long_held & set(row.index)
            short_held = prev_short_held & set(row.index)

        day_rets = ret_pivot.loc[d] if d in ret_pivot.index else pd.Series(dtype=float)

        def _weights(names):
            names = list(names)
            if not names:
                return pd.Series(dtype=float)
            if vol_weight:
                v = vol_pivot.loc[d].reindex(names) if d in vol_pivot.index else pd.Series(dtype=float)
                v = v.replace(0, np.nan)
                inv = (1 / v).dropna()
                if inv.empty:
                    return pd.Series(1.0 / len(names), index=names)  # fallback: equal weight
                return inv / inv.sum()
            return pd.Series(1.0 / len(names), index=names)

        w_long = _weights(long_held)
        w_short = _weights(short_held)

        # Beta-neutralize: scale the short leg's total size so long-leg beta
        # and short-leg beta offset. Long leg stays at gross 1.0 always.
        net_beta = np.nan
        short_scale = 1.0
        if beta_pivot is not None and d in beta_pivot.index and not w_long.empty and not w_short.empty:
            beta_row = beta_pivot.loc[d]
            beta_long_avg = (w_long * beta_row.reindex(w_long.index)).sum()
            beta_short_avg = (w_short * beta_row.reindex(w_short.index)).sum()
            if pd.notna(beta_long_avg) and pd.notna(beta_short_avg) and beta_short_avg != 0:
                short_scale = float(beta_long_avg / beta_short_avg)
                short_scale = float(np.clip(short_scale, 0.2, 5.0))  # sanity bound
                net_beta = beta_long_avg - short_scale * beta_short_avg

        r_long = day_rets.reindex(w_long.index)
        r_short = day_rets.reindex(w_short.index)
        valid_long = r_long.notna()
        valid_short = r_short.notna()

        long_ret = (w_long[valid_long] * r_long[valid_long]).sum() / w_long[valid_long].sum() if valid_long.any() else np.nan
        short_ret_raw = (w_short[valid_short] * r_short[valid_short]).sum() / w_short[valid_short].sum() if valid_short.any() else np.nan

        gross_long = w_long.sum() if not w_long.empty else 0.0
        gross_short = (w_short.sum() * short_scale) if not w_short.empty else 0.0

        if pd.notna(long_ret) and pd.notna(short_ret_raw):
            port_rets.append({"date": d, "ret": long_ret - short_ret_raw * short_scale})

        diag_rows.append({
            "date": d, "n_long": len(long_held), "n_short": len(short_held),
            "gross_exposure": gross_long + gross_short,
            "net_exposure": gross_long - gross_short,
            "net_beta": net_beta,
        })

        if is_rebalance_day:
            new_long = long_held - prev_long_held
            new_short = short_held - prev_short_held
            turnovers.append((len(new_long) / target_n + len(new_short) / target_n) / 2)

    rets = pd.DataFrame(port_rets).set_index("date")["ret"].dropna()
    diag = pd.DataFrame(diag_rows).set_index("date")
    if rets.empty:
        return {}

    ann_return = rets.mean() * 252
    ann_vol = rets.std() * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan

    return {
        "ann_return": float(ann_return),
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "turnover_per_rebalance": float(np.nanmean(turnovers)) if turnovers else np.nan,
        "n_days": len(rets),
        "daily_returns": rets,
        "diagnostics": diag,
    }


def sharpe_standard_error(sharpe: float, n_days: int, n_years: float | None = None) -> dict:
    """
    Two SE estimates:
    - daily-return-based (Lo, 2002 approximation): SE ~= sqrt((1 + 0.5*SR_daily^2) / n_days),
      then annualized by sqrt(252) — standard for a Sharpe computed from daily returns.
    - simple annual-block: sqrt(1/n_years), treating each year as one independent
      observation of the annual Sharpe — cruder, more conservative, matches the
      commonly-cited rule of thumb for judging whether an annualized Sharpe is
      distinguishable from zero at typical backtest lengths.
    """
    sr_daily = sharpe / np.sqrt(252)
    se_daily_based = np.sqrt((1 + 0.5 * sr_daily**2) / n_days) * np.sqrt(252)
    result = {"se_daily_based": float(se_daily_based)}
    if n_years:
        result["se_annual_block"] = float(1 / np.sqrt(n_years))
    return result


def alpha_beta_vs_benchmark(daily_returns: pd.Series, benchmark_returns: pd.Series) -> dict:
    """OLS: portfolio_return ~ alpha + beta * benchmark_return. Reports
    annualized alpha, beta, and the t-stat on alpha (is it distinguishable
    from zero, not just its point estimate)."""
    import statsmodels.api as sm

    aligned = pd.DataFrame({"port": daily_returns, "bench": benchmark_returns}).dropna()
    if len(aligned) < 30:
        return {}
    X = sm.add_constant(aligned["bench"])
    model = sm.OLS(aligned["port"], X).fit()
    alpha_daily = model.params["const"]
    beta = model.params["bench"]
    t_alpha = model.tvalues["const"]
    return {
        "alpha_annualized": float(alpha_daily * 252),
        "beta": float(beta),
        "t_stat_alpha": float(t_alpha),
        "n_obs": len(aligned),
    }


def run_full_analysis(oos_df: pd.DataFrame, spy_returns: pd.Series, n_years: float,
                       bps_sweep: tuple = (10, 20)) -> pd.DataFrame:
    """
    The 2x2: {naive, hysteresis} x {equal, inverse-vol}. Identical code
    path for all 4 cells (only the two switches differ) — for each cell:
    exposure sanity check first, then Sharpe (+ SE), turnover, net-of-cost
    return, and alpha/beta/t-stat vs SPY.
    """
    cells = {
        "A: naive + equal":        dict(use_hysteresis=False, vol_weight=False),
        "B: hysteresis + equal":   dict(use_hysteresis=True,  vol_weight=False),
        "C: naive + inverse-vol":  dict(use_hysteresis=False, vol_weight=True),
        "D: hysteresis + inv-vol": dict(use_hysteresis=True,  vol_weight=True),
    }

    rows = []
    for name, kwargs in cells.items():
        print(f"\n{'='*70}\n  {name}\n{'='*70}")
        res = simulate_portfolio(oos_df, **kwargs)
        if not res:
            print("  No valid days — skipping.")
            continue

        diag = res["diagnostics"]
        print(f"  Exposure check — gross (should be ~2.0): mean={diag['gross_exposure'].mean():.3f}, "
              f"min={diag['gross_exposure'].min():.3f}, max={diag['gross_exposure'].max():.3f}")
        print(f"  Exposure check — net (should be ~0.0)  : mean={diag['net_exposure'].mean():+.3f}, "
              f"std={diag['net_exposure'].std():.3f}, max|net|={diag['net_exposure'].abs().max():.3f}")
        print(f"  Name count — long: mean={diag['n_long'].mean():.1f}, short: mean={diag['n_short'].mean():.1f}")
        neutral_ok = diag['net_exposure'].abs().max() < 0.05
        print(f"  Dollar-neutral: {'OK' if neutral_ok else 'VIOLATED — results below are NOT clean alpha'}")

        se = sharpe_standard_error(res["sharpe"], res["n_days"], n_years)
        ab = alpha_beta_vs_benchmark(res["daily_returns"], spy_returns)

        print(f"  Sharpe: {res['sharpe']:+.3f}  (SE~{se['se_daily_based']:.3f}, annual-block SE~{se.get('se_annual_block', float('nan')):.3f})")
        print(f"  Ann return: {res['ann_return']:+.1%}   Ann vol: {res['ann_vol']:.1%}")
        print(f"  Turnover/rebalance: {res['turnover_per_rebalance']:.1%}")
        if ab:
            print(f"  vs SPY — alpha(ann): {ab['alpha_annualized']:+.1%}  beta: {ab['beta']:+.3f}  t-stat(alpha): {ab['t_stat_alpha']:+.2f}")

        row = {"cell": name, "sharpe": res["sharpe"], "se_daily": se["se_daily_based"],
               "ann_return": res["ann_return"], "ann_vol": res["ann_vol"],
               "turnover": res["turnover_per_rebalance"], "net_exposure_ok": neutral_ok}
        if ab:
            row.update({"alpha_ann": ab["alpha_annualized"], "beta": ab["beta"], "t_stat_alpha": ab["t_stat_alpha"]})
        for bps in bps_sweep:
            cost_drag = res["turnover_per_rebalance"] * (bps / 10000) * 252
            row[f"net_return_{bps}bps"] = res["ann_return"] - cost_drag
        rows.append(row)

    return pd.DataFrame(rows).set_index("cell")


def run_round2_analysis(oos_df: pd.DataFrame, spy_returns: pd.Series, n_years: float,
                         bps_sweep: tuple = (10, 20)) -> pd.DataFrame:
    """
    Cell C (naive + inverse-vol) was the round-1 winner but hysteresis lost
    fairly, not on a technicality — this round tests the things that could
    actually beat it: more names (quintile/fixed-20, the non-stale version
    of "dilution"), less frequent rebalancing (matches the 5-day signal
    horizon instead of trading it 5x too fast), and beta-neutralizing C
    itself (dollar-neutral doesn't imply beta-neutral if the short leg
    systematically loads on lower-beta defensive names, which is common).
    """
    beta_pivot = compute_trailing_beta(oos_df, spy_returns, window=60)
    configs = {
        "C (baseline: decile, daily, vol-wt)":       dict(entry_frac=0.10, rebalance_every=1),
        "quintile (top/bottom 20%, daily)":          dict(entry_frac=0.20, rebalance_every=1),
        "fixed-20 names, daily":                     dict(top_n=20, rebalance_every=1),
        "C, rebalance every 3d":                      dict(entry_frac=0.10, rebalance_every=3),
        "C, rebalance every 5d (signal horizon)":     dict(entry_frac=0.10, rebalance_every=5),
        "C, beta-neutralized":                        dict(entry_frac=0.10, rebalance_every=1, beta_pivot=beta_pivot),
    }

    rows = []
    for name, kwargs in configs.items():
        rebalance_every = kwargs.get("rebalance_every", 1)
        print(f"\n{'='*70}\n  {name}\n{'='*70}")
        res = simulate_portfolio(oos_df, vol_weight=True, **kwargs)
        if not res:
            print("  No valid days — skipping.")
            continue

        diag = res["diagnostics"]
        print(f"  Exposure — gross: {diag['gross_exposure'].mean():.3f}  net: {diag['net_exposure'].mean():+.3f}")
        if diag["net_beta"].notna().any():
            print(f"  Net beta: mean={diag['net_beta'].mean():+.3f}")
        print(f"  Name count — long: {diag['n_long'].mean():.1f}, short: {diag['n_short'].mean():.1f}")

        se = sharpe_standard_error(res["sharpe"], res["n_days"], n_years)
        ab = alpha_beta_vs_benchmark(res["daily_returns"], spy_returns)
        boot = block_bootstrap_sharpe(res["daily_returns"], block_size=20, n_boot=1000)

        print(f"  Sharpe: {res['sharpe']:+.3f}  (SE~{se['se_daily_based']:.3f})")
        if boot:
            print(f"  Bootstrap Sharpe 90% CI: [{boot['p5']:+.3f}, {boot['p95']:+.3f}]  (median {boot['p50']:+.3f})")
        print(f"  Ann return: {res['ann_return']:+.1%}   Ann vol: {res['ann_vol']:.1%}")
        print(f"  Turnover/rebalance: {res['turnover_per_rebalance']:.1%}  (rebalances/yr: {252/rebalance_every:.0f})")
        if ab:
            print(f"  vs SPY — alpha(ann): {ab['alpha_annualized']:+.1%}  beta: {ab['beta']:+.3f}  t-stat(alpha): {ab['t_stat_alpha']:+.2f}")

        row = {"config": name, "sharpe": res["sharpe"], "sharpe_p5": boot.get("p5"), "sharpe_p95": boot.get("p95"),
               "ann_return": res["ann_return"], "ann_vol": res["ann_vol"],
               "turnover": res["turnover_per_rebalance"], "rebalances_per_yr": 252 / rebalance_every}
        if ab:
            row.update({"alpha_ann": ab["alpha_annualized"], "beta": ab["beta"], "t_stat_alpha": ab["t_stat_alpha"]})
        for bps in bps_sweep:
            cost_drag = res["turnover_per_rebalance"] * (bps / 10000) * (252 / rebalance_every)
            row[f"net_return_{bps}bps"] = res["ann_return"] - cost_drag
        rows.append(row)

    result = pd.DataFrame(rows).set_index("config")

    # Cumulative P&L / drawdown check on the baseline (C) — is the edge
    # concentrated in a few episodes, or broadly distributed?
    print(f"\n{'='*70}\n  Cumulative P&L check on baseline (C) — event-concentration risk\n{'='*70}")
    base = simulate_portfolio(oos_df, vol_weight=True, entry_frac=0.10)
    if base:
        cum = (1 + base["daily_returns"]).cumprod()
        running_max = cum.cummax()
        drawdown = (cum / running_max - 1)
        max_dd = drawdown.min()
        max_dd_date = drawdown.idxmin()
        by_year = base["daily_returns"].groupby(base["daily_returns"].index.year).sum()
        print(f"  Max drawdown: {max_dd:.1%} on {max_dd_date.date()}")
        print(f"  Return by year:\n{by_year.apply(lambda x: f'{x:+.1%}').to_string()}")
        best_year = by_year.idxmax()
        frac_from_best_year = by_year[best_year] / by_year.sum() if by_year.sum() != 0 else float("nan")
        print(f"  Best single year ({best_year}) is {frac_from_best_year:.0%} of total summed annual return")

    return result


if __name__ == "__main__":
    """Mechanics-only self-check on a tiny synthetic panel — not a real backtest."""
    np.random.seed(0)
    dates = pd.date_range("2020-01-01", periods=200, freq="B")
    tickers = [f"T{i}" for i in range(30)]
    rows = []
    close = {t: 100.0 for t in tickers}
    for d in dates:
        for t in tickers:
            close[t] *= (1 + np.random.randn() * 0.01)
            rows.append({
                "date": d, "ticker": t, "close": close[t],
                "pred_return": np.random.randn(),
                "std_20": abs(np.random.randn()) * 0.02 + 0.01,
            })
    df = pd.DataFrame(rows)
    spy = pd.Series(np.random.randn(len(dates)) * 0.01, index=dates)

    for use_h in (False, True):
        for use_v in (False, True):
            r = simulate_portfolio(df, use_hysteresis=use_h, vol_weight=use_v)
            assert r and r["n_days"] > 0, f"failed: hysteresis={use_h} vol_weight={use_v}"
            d = r["diagnostics"]
            assert d["gross_exposure"].mean() > 1.5, "gross exposure implausibly low — weighting bug"
            assert d["net_exposure"].abs().mean() < 0.1, "net exposure not ~0 — dollar-neutrality bug"
            print(f"hysteresis={use_h} vol_weight={use_v}: sharpe={r['sharpe']:.3f} "
                  f"gross={d['gross_exposure'].mean():.2f} net={d['net_exposure'].mean():+.3f} OK")

    print("\nAll 4 cells mechanically sound (gross~2, net~0) on synthetic data.")
    print("Run run_full_analysis(oos_df, spy_returns, n_years) on real data next.")
