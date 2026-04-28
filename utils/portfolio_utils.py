"""
portfolio_utils.py
------------------
Portfolio construction and SDF estimation helpers.
"""

import numpy as np
import pandas as pd
from scipy.linalg import solve


# ── SDF estimation ────────────────────────────────────────────────────────────

def estimate_hj_sdf(returns, ridge=1e-3):
    """
    Estimate the Hansen-Jagannathan minimum-variance SDF weights.

    m_{t+1} = 1 - b' R_{t+1}
    b = Sigma^{-1} mu   (ridge-regularised)

    Parameters
    ----------
    returns : pd.DataFrame (T x K) of excess returns
    ridge   : float, ridge regularisation for Sigma

    Returns
    -------
    b    : np.ndarray (K,) SDF weights
    mret : pd.Series  SDF return series  m_t = 1 - b' R_t
    """
    R   = returns.values
    mu  = R.mean(axis=0)
    Sig = np.cov(R, rowvar=False)
    Sig_reg = Sig + ridge * np.eye(Sig.shape[0])
    b   = solve(Sig_reg, mu)
    mret = 1.0 - R @ b
    return b, pd.Series(mret, index=returns.index, name="SDF_return")


def rolling_hj_sdf(returns, window=72, ridge=1e-3):
    """
    Compute rolling out-of-sample SDF returns.

    At each month t, estimates b on returns[t-window:t], then
    computes the SDF return at t+1.

    Returns
    -------
    pd.Series of out-of-sample SDF returns
    """
    T   = len(returns)
    oos = {}
    for t in range(window, T - 1):
        train = returns.iloc[t - window: t]
        b, _  = estimate_hj_sdf(train, ridge=ridge)
        r_next = returns.iloc[t + 1].values
        oos[returns.index[t + 1]] = 1.0 - b @ r_next
    return pd.Series(oos, name="OOS_SDF_return")


def mve_frontier(returns, n_points=200, ridge=1e-3):
    """
    Compute the mean-variance efficient frontier analytically
    via two-fund separation.

    Parameters
    ----------
    returns  : pd.DataFrame (T x K)
    n_points : int, number of frontier points
    ridge    : float

    Returns
    -------
    stds     : np.ndarray of portfolio std devs
    means    : np.ndarray of portfolio mean returns
    tangency_sr : float, Sharpe ratio of tangency portfolio
    """
    R   = returns.values
    mu  = R.mean(axis=0)
    Sig = np.cov(R, rowvar=False) + ridge * np.eye(R.shape[1])

    # Tangency portfolio
    b_tan    = solve(Sig, mu)
    w_tan    = b_tan / b_tan.sum()
    tan_ret  = mu @ w_tan
    tan_std  = np.sqrt(w_tan @ Sig @ w_tan)
    tan_sr   = tan_ret / tan_std * np.sqrt(12)   # annualised

    # Minimum-variance portfolio
    ones   = np.ones(len(mu))
    b_mv   = solve(Sig, ones)
    w_mv   = b_mv / b_mv.sum()
    mv_ret = mu @ w_mv
    mv_std = np.sqrt(w_mv @ Sig @ w_mv)

    # Sweep along the frontier
    alphas = np.linspace(0, 2.5, n_points)
    stds   = []
    means  = []
    for a in alphas:
        w = (1 - a) * w_mv + a * w_tan
        means.append(mu @ w)
        stds.append(np.sqrt(w @ Sig @ w))

    return np.array(stds), np.array(means), tan_sr


# ── H-L portfolio ─────────────────────────────────────────────────────────────

def hl_portfolio(cluster_rets):
    """
    Long highest-ranked cluster (last column), short lowest-ranked (first).

    Parameters
    ----------
    cluster_rets : pd.DataFrame with columns L01..L50 (ranked)

    Returns
    -------
    pd.Series of H-L returns
    """
    high = cluster_rets.iloc[:, -1]
    low  = cluster_rets.iloc[:,  0]
    return (high - low).rename("HL")


def sharpe(returns, annualize=True):
    """Annualised Sharpe ratio."""
    sr = returns.mean() / returns.std()
    return sr * np.sqrt(12) if annualize else sr


def calmar(returns):
    """Calmar ratio: annualised mean / max drawdown."""
    ann_ret  = returns.mean() * 12
    cum      = (1 + returns).cumprod()
    drawdown = (cum / cum.cummax() - 1).min()
    if drawdown == 0:
        return np.nan
    return ann_ret / abs(drawdown)


def max_drawdown(returns):
    """Maximum drawdown of a return series."""
    cum = (1 + returns).cumprod()
    return float((cum / cum.cummax() - 1).min())


def var_95(returns):
    """5th-percentile VaR (negative number)."""
    return float(np.percentile(returns, 5))


# ── Regime splitting ──────────────────────────────────────────────────────────

def rolling_percentile_regime(series, window=120, pct=50):
    """
    Classify each period as 'Top' (above rolling pct-ile) or 'Btm'.

    Parameters
    ----------
    series : pd.Series of macro variable
    window : int, rolling window in months (default 120 = 10 years)
    pct    : float, percentile threshold

    Returns
    -------
    pd.Series of strings: 'Top' or 'Btm'
    """
    rolling_threshold = series.rolling(window, min_periods=window//2).quantile(pct / 100)
    regime = pd.Series(
        np.where(series >= rolling_threshold, "Top", "Btm"),
        index=series.index,
        name=series.name
    )
    return regime
