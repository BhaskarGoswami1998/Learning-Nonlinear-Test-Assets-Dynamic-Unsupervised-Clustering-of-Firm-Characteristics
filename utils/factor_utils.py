"""
factor_utils.py
---------------
Factor model regression helpers: time-series spanning tests,
GRS statistic, alpha diagnostics.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


# ── Factor model definitions ──────────────────────────────────────────────────

FACTOR_MODELS = {
    "CAPM":  ["MKT"],
    "FF3":   ["MKT", "SMB", "HML"],
    "FFC":   ["MKT", "SMB", "HML", "MOM"],
    "FF5":   ["MKT", "SMB", "HML", "RMW", "CMA"],
    "FFPS":  ["MKT", "SMB", "HML", "LIQ"],
    "DHS":   ["MKT", "PEAD", "FIN"],
    "Q5":    ["MKT", "SMB", "IA", "ROE", "EG"],
    "SYM":   ["MKT", "SMB", "MGMT", "PERF"],
}


# ── Single-asset regression ───────────────────────────────────────────────────

def ts_regression(ret, factors, nw_lags=3):
    """
    Time-series OLS regression with Newey-West standard errors.

    Parameters
    ----------
    ret     : pd.Series, excess return of the portfolio
    factors : pd.DataFrame, factor returns (T x K)
    nw_lags : int

    Returns
    -------
    dict with keys: alpha, alpha_se, alpha_t, betas, r2, resid
    """
    common = ret.index.intersection(factors.index).dropna()
    y = ret.loc[common].values
    X = sm.add_constant(factors.loc[common].values)

    res = sm.OLS(y, X).fit(
        cov_type="HAC", cov_kwds={"maxlags": nw_lags}
    )
    return {
        "alpha":    res.params[0],
        "alpha_se": res.bse[0],
        "alpha_t":  res.tvalues[0],
        "betas":    res.params[1:],
        "r2":       res.rsquared,
        "resid":    res.resid,
        "nobs":     int(res.nobs),
    }


# ── Panel diagnostics ─────────────────────────────────────────────────────────

def panel_alpha_diagnostics(cluster_rets, factor_returns, model_name, nw_lags=3):
    """
    Compute cross-sectional average alpha diagnostics across all clusters.

    Parameters
    ----------
    cluster_rets   : pd.DataFrame (T x K) of cluster excess returns
    factor_returns : pd.DataFrame (T x F) of factor returns
    model_name     : str, key in FACTOR_MODELS
    nw_lags        : int

    Returns
    -------
    dict with keys:
        mean_abs_alpha   : mean |alpha_i|
        mean_abs_alpha_r : mean |alpha_i / r_i|
        mean_r2          : mean R²
        sh2_alpha        : Sh²(alpha) = GRS statistic proxy
        alphas           : list of individual alphas
    """
    factors = factor_returns[FACTOR_MODELS[model_name]].dropna()
    alphas, r2s, mean_rets = [], [], []

    for col in cluster_rets.columns:
        res = ts_regression(cluster_rets[col], factors, nw_lags=nw_lags)
        alphas.append(res["alpha"])
        r2s.append(res["r2"])
        mean_rets.append(cluster_rets[col].mean())

    alphas    = np.array(alphas)
    mean_rets = np.array(mean_rets)

    # Sh²(alpha): squared Sharpe ratio of the alpha portfolio
    if np.std(alphas) > 0:
        sh2 = (np.mean(alphas) / np.std(alphas)) ** 2
    else:
        sh2 = 0.0

    return {
        "mean_abs_alpha":   np.mean(np.abs(alphas)),
        "mean_abs_alpha_r": np.mean(np.abs(alphas / (mean_rets + 1e-10))),
        "mean_r2":          np.mean(r2s),
        "sh2_alpha":        sh2,
        "alphas":           alphas.tolist(),
    }


# ── GRS test ──────────────────────────────────────────────────────────────────

def grs_test(cluster_rets, factor_returns, model_name, nw_lags=3):
    """
    Gibbons, Ross & Shanken (1989) F-statistic.

    GRS = (T-N-K)/N * [1 + mu_f' Sig_f^{-1} mu_f]^{-1} * alpha' Sig_e^{-1} alpha

    Parameters
    ----------
    cluster_rets   : pd.DataFrame (T x N)
    factor_returns : pd.DataFrame (T x K)
    model_name     : str

    Returns
    -------
    F_stat : float
    p_val  : float
    """
    factors = factor_returns[FACTOR_MODELS[model_name]].dropna()
    common  = cluster_rets.index.intersection(factors.index)
    R = cluster_rets.loc[common].values
    F = factors.loc[common].values

    T, N = R.shape
    K    = F.shape[1]

    # Run regressions
    X   = np.column_stack([np.ones(T), F])
    B   = np.linalg.lstsq(X, R, rcond=None)[0]       # (K+1, N)
    alp = B[0]                                          # (N,)
    E   = R - X @ B                                     # (T, N)

    Sig_e  = (E.T @ E) / (T - K - 1)
    mu_f   = F.mean(axis=0)
    Sig_f  = np.cov(F, rowvar=False)

    try:
        adj   = 1 + mu_f @ np.linalg.solve(Sig_f, mu_f)
        grs_f = ((T - N - K) / N) * (1 / adj) * (
            alp @ np.linalg.solve(Sig_e, alp)
        )
        p_val = 1 - stats.f.cdf(grs_f, N, T - N - K)
    except np.linalg.LinAlgError:
        grs_f, p_val = np.nan, np.nan

    return float(grs_f), float(p_val)


# ── Summary table builder ─────────────────────────────────────────────────────

def build_alpha_summary_table(cluster_rets, factor_returns, nw_lags=3):
    """
    Build Table 2 style summary: one row per factor model.

    Returns
    -------
    pd.DataFrame with columns: A|alpha|, A|alpha/r|, R2, Sh2(alpha)
    """
    rows = {}
    for model in FACTOR_MODELS:
        diag = panel_alpha_diagnostics(
            cluster_rets, factor_returns, model, nw_lags=nw_lags
        )
        rows[model] = {
            "A|α_i|":           round(diag["mean_abs_alpha"],   4),
            "A|α_i/r_i|":       round(diag["mean_abs_alpha_r"], 4),
            "R²":                round(diag["mean_r2"],          4),
            "Sh²(α)":           round(diag["sh2_alpha"],        4),
        }
    return pd.DataFrame(rows).T
