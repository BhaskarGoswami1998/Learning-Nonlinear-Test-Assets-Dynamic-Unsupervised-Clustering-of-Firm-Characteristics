"""
data_utils.py
=============
Data loading helpers built around the REAL file structure:

  Results/
  ├── cluster_month_panels_K_50/
  │   ├── cluster_month_panel_K_50_lambda_1000000.csv
  │   ├── cluster_month_panel_K_50_lambda_1000.csv
  │   └── ...  (all 13 λ values for K=50)
  └── cluster_month_panels_all_K_except_50/
      ├── cluster_month_panel_K_10_lambda_1000000.csv
      └── ...  (26 other K values × 13 λ values)

CSV columns (per file):
    year_month               : str   'YYYY-MM'
    cluster                  : int   0 … K-1
    mktcap_wt_excess_return  : float (do NOT use — this is a lagged value)
    r_t1_wmean               : float 1-month-ahead VW excess return  ← USE THIS
    r_t2_wmean               : float 2-month-ahead VW excess return
    r_t3_wmean               : float 3-month-ahead VW excess return
    rt2g_wmean               : float 2-month compounded return
    rt3g_wmean               : float 3-month compounded return
    n_permnos                : int   number of firms in cluster
    mktcap_sum               : float total market cap
"""

import os
import numpy as np
import pandas as pd

# ── Repository paths ──────────────────────────────────────────────────────────
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(REPO_ROOT, "data")
FIG_DIR   = os.path.join(REPO_ROOT, "output", "figures")
TAB_DIR   = os.path.join(REPO_ROOT, "output", "tables")

# ── Results folder paths (set these to your actual Results/ location) ─────────
# Default: look for a symlink/copy at  <repo>/data/Results/
# Override by passing data_dir= explicitly to load_cluster_panel()
RESULTS_DIR        = "/ssd1/songjiangliu/shared/asset_clustering/Results"
K50_DIR            = os.path.join(RESULTS_DIR, "cluster_month_panels_K_50")
ALL_K_DIR          = os.path.join(RESULTS_DIR, "cluster_month_panels_all_K_except_50")

# Legacy grid dir (used by notebooks that pre-date this structure)
GRID_DIR = K50_DIR   # backward-compat alias — points to K=50 folder

# ── Grid constants ─────────────────────────────────────────────────────────────
K_GRID = [2,3,4,5,6,7,8,9,10,15,20,25,30,35,40,45,50,
          55,60,65,70,75,80,85,90,95,100]

# Exact lambda strings as they appear in filenames (from Cluster_Ranking.csv columns)
LAMBDA_STRS = [
    "lambda_1e-09", "lambda_1e-08", "lambda_1e-07", "lambda_1e-06",
    "lambda_1e-05", "lambda_0.0001", "lambda_0.001", "lambda_0.01",   # 0.01 not in ranking but keep
    "lambda_0.1",   "lambda_1",      "lambda_10",    "lambda_100",
    "lambda_1000",  "lambda_10000",  "lambda_100000","lambda_1000000",
    "lambda_10000000","lambda_100000000","lambda_1000000000",
]

# Mapping lambda string → float
LAMBDA_STR_TO_FLOAT = {
    "lambda_1e-09":       1e-9,   "lambda_1e-08":       1e-8,
    "lambda_1e-07":       1e-7,   "lambda_1e-06":       1e-6,
    "lambda_1e-05":       1e-5,   "lambda_0.0001":      1e-4,
    "lambda_0.001":       1e-3,   "lambda_0.01":        1e-2,
    "lambda_0.1":         0.1,    "lambda_1":           1.0,
    "lambda_10":          10.0,   "lambda_100":         100.0,
    "lambda_1000":        1e3,    "lambda_10000":       1e4,
    "lambda_100000":      1e5,    "lambda_1000000":     1e6,
    "lambda_10000000":    1e7,    "lambda_100000000":   1e8,
    "lambda_1000000000":  1e9,
}
LAMBDA_FLOAT_TO_STR = {v: k for k, v in LAMBDA_STR_TO_FLOAT.items()}

# Lambda strings that appear in the Cluster_Ranking.csv (confirmed)
RANKING_LAMBDA_STRS = [
    "lambda_1000000", "lambda_1000", "lambda_100",
    "lambda_0.0001",  "lambda_1e-05","lambda_1e-06",
    "lambda_1e-07",   "lambda_1e-08","lambda_1e-09",
    "lambda_1",       "lambda_0.1",  "lambda_10", "lambda_0.001",
]

# 45 firm characteristics
CHARS_45 = [
    "A2ME","AC","AT","ATO","B2M","BETA_d","BETA_m","C2A","CF2B","CF2P",
    "CTO","D2A","D2P","DPI2A","E2P","FC2Y","HIGH52","IdioVol","INV","LEV",
    "ME","NI","NOA","OA","OL","OP","PCM","PM","PROF","Q",
    "R12_2","R12_7","R2_1","R36_13","R60_13","RNA","ROA","ROE","RVAR",
    "S2P","SGA2S","SPREAD","SUV","TURN","VAR",
]


# ── File path resolution ───────────────────────────────────────────────────────

def get_panel_path(K, lam_str, results_dir=None):
    """
    Return the full path to cluster_month_panel_K_{K}_{lam_str}.csv,
    searching in the correct subfolder automatically.

    Parameters
    ----------
    K          : int   number of clusters
    lam_str    : str   lambda string exactly as in filename,
                       e.g. 'lambda_1000000', 'lambda_1e-06'
    results_dir: str or None  — root Results/ directory.
                 Defaults to RESULTS_DIR.

    Returns
    -------
    str  full path  (raises FileNotFoundError if not found)
    """
    if results_dir is None:
        results_dir = RESULTS_DIR

    fname = f"cluster_month_panel_K_{K}_{lam_str}.csv"

    # Check K=50 subfolder first, then the other-K subfolder
    for subdir in ["cluster_month_panels_K_50",
                   "cluster_month_panels_all_K_except_50"]:
        path = os.path.join(results_dir, subdir, fname)
        if os.path.exists(path):
            return path

    # Fallback: check directly inside results_dir (flat structure)
    path = os.path.join(results_dir, fname)
    if os.path.exists(path):
        return path

    raise FileNotFoundError(
        f"Panel file not found: {fname}\n"
        f"Searched in:\n"
        f"  {results_dir}/cluster_month_panels_K_50/\n"
        f"  {results_dir}/cluster_month_panels_all_K_except_50/\n"
        f"  {results_dir}/\n"
        f"Make sure your Results/ folder is symlinked or copied to:\n"
        f"  {RESULTS_DIR}"
    )


def lam_to_str(lam):
    """
    Convert a float lambda to its filename string.
    e.g.  1_000_000 -> 'lambda_1000000',  1e-6 -> 'lambda_1e-06'
    """
    for v, s in LAMBDA_FLOAT_TO_STR.items():
        if abs(v - lam) / max(abs(v), 1e-20) < 1e-6:
            return s
    raise ValueError(f"Unrecognised lambda value: {lam}. "
                     f"Known values: {list(LAMBDA_FLOAT_TO_STR.keys())}")


# ── Panel loader ───────────────────────────────────────────────────────────────

def load_cluster_panel(K=50, lam=1_000_000, results_dir=None, data_dir=None):
    """
    Load and standardise a cluster-month panel.

    Parameters
    ----------
    K           : int    number of clusters
    lam         : float  lambda value
    results_dir : str    path to Results/ root (overrides RESULTS_DIR)
    data_dir    : str    legacy alias for results_dir (for backward compat)

    Returns
    -------
    pd.DataFrame with standardised columns:
        date, year_month, cluster, ret, r_t1, r_t2, r_t3,
        rt2g, rt3g, n_firms, mktcap
    """
    # backward-compat: if data_dir was passed, treat it as results_dir
    if results_dir is None and data_dir is not None:
        results_dir = data_dir

    lam_str = lam_to_str(lam)
    path    = get_panel_path(K, lam_str, results_dir=results_dir)

    df = pd.read_csv(path)
    df = df.rename(columns={
        "mktcap_wt_excess_return": "ret",   # do NOT use — lagged
        "r_t1_wmean":  "r_t1",
        "r_t2_wmean":  "r_t2",
        "r_t3_wmean":  "r_t3",
        "rt2g_wmean":  "rt2g",
        "rt3g_wmean":  "rt3g",
        "n_permnos":   "n_firms",
        "mktcap_sum":  "mktcap",
    })
    df["date"] = pd.to_datetime(df["year_month"], format="%Y-%m")
    return df.sort_values(["date", "cluster"]).reset_index(drop=True)


def pivot_returns(df, col="r_t1", dropna=True):
    """Pivot to wide (T × K) return matrix. Default column: r_t1 (1-month-ahead)."""
    wide = df.pivot(index="date", columns="cluster", values=col)
    return wide.dropna() if dropna else wide


# ── Cluster ranking helpers ────────────────────────────────────────────────────

def load_cluster_ranking(path=None):
    """
    Load Cluster_Ranking.csv.
    Columns: Cluster Name, Inertia, Vanilla, lambda_1000000, lambda_1000, ...
    Returns pd.DataFrame.
    """
    if path is None:
        path = os.path.join(DATA_DIR, "Cluster_Ranking.csv")
    return pd.read_csv(path)


def get_rank_map(lam=1_000_000, lam_str=None, ranking_df=None):
    """
    Build  {raw_cluster_id (int) -> 'L01'…'L50'}  for a given lambda.

    Parameters
    ----------
    lam        : float  lambda value (used to look up lam_str if not given)
    lam_str    : str    e.g. 'lambda_1000000' — overrides lam if given
    ranking_df : pd.DataFrame from load_cluster_ranking(), or None to load

    Returns
    -------
    dict  {int -> 'Lxx'}
    """
    if ranking_df is None:
        ranking_df = load_cluster_ranking()

    if lam_str is None:
        lam_str = lam_to_str(lam)

    if lam_str not in ranking_df.columns:
        raise ValueError(
            f"'{lam_str}' not in Cluster_Ranking.csv.\n"
            f"Available λ columns: {[c for c in ranking_df.columns if c.startswith('lambda')]}"
        )

    raw_to_label = {}
    for _, row in ranking_df.iterrows():
        raw_id = int(row[lam_str])
        label  = row["Cluster Name"]            # e.g. 'L50', 'L9'
        n      = int(label[1:])
        raw_to_label[raw_id] = f"L{n:02d}"     # zero-pad: 'L9' -> 'L09'
    return raw_to_label


def pivot_and_rank(df, lam=1_000_000, lam_str=None, col="r_t1",
                   dropna=True, ranking_df=None):
    """
    One-call: pivot panel → apply official cluster ranking → return (T × 50) matrix.

    Parameters
    ----------
    df         : output of load_cluster_panel()
    lam        : float  lambda (used to find rank_map column)
    lam_str    : str    override lam with explicit lambda string
    col        : return column to pivot (default 'r_t1')
    dropna     : drop rows with any NaN
    ranking_df : pre-loaded Cluster_Ranking DataFrame

    Returns
    -------
    cr_ranked : pd.DataFrame  (T, K), columns = 'L01' … 'L50'
    rank_map  : dict  {raw_id -> 'Lxx'}
    """
    rank_map  = get_rank_map(lam=lam, lam_str=lam_str, ranking_df=ranking_df)
    wide      = pivot_returns(df, col=col, dropna=dropna)
    cr_ranked = wide.rename(columns=rank_map)
    K         = len(rank_map)
    ordered   = [f"L{i:02d}" for i in range(1, K + 1)
                 if f"L{i:02d}" in cr_ranked.columns]
    return cr_ranked[ordered], rank_map


# ── Auxiliary data loaders ─────────────────────────────────────────────────────

def load_factor_data(path=None):
    """Load factor_returns.csv. Columns: date, MKT, SMB, HML, ..."""
    if path is None:
        path = os.path.join(DATA_DIR, "factor_returns.csv")
    return pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()


def load_macro_data(path=None):
    """Load macro_predictors.csv. Columns: date, dp, dy, ep, tbl, ..."""
    if path is None:
        path = os.path.join(DATA_DIR, "macro_predictors.csv")
    return pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()


def load_centroid_chars(path=None):
    """Load centroid_chars.csv. Columns: year_month, cluster, A2ME, ..., VAR"""
    if path is None:
        path = os.path.join(DATA_DIR, "centroid_chars.csv")
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["year_month"], format="%Y-%m")
    return df


def load_firm_panel(path=None):
    """Load firm_panel.csv. Columns: year_month, permno, cluster, ret, me, SIC, ..."""
    if path is None:
        path = os.path.join(DATA_DIR, "firm_panel.csv")
    df = pd.read_csv(path, low_memory=False)
    df["date"] = pd.to_datetime(df["year_month"], format="%Y-%m")
    return df


# ── Statistics helpers ─────────────────────────────────────────────────────────

def nw_tstat(series, lags=3):
    """Return (mean, NW_se, t_stat) with Newey-West HAC standard errors."""
    import statsmodels.api as sm
    s = pd.Series(series).dropna().values
    if len(s) < 10:
        return np.nan, np.nan, np.nan
    res = sm.OLS(s, np.ones(len(s))).fit(
        cov_type="HAC", cov_kwds={"maxlags": lags}
    )
    return float(res.params[0]), float(res.bse[0]), float(res.tvalues[0])


def stars(t):
    if pd.isna(t):      return ""
    if abs(t) >= 2.576: return "***"
    if abs(t) >= 1.960: return "**"
    if abs(t) >= 1.645: return "*"
    return ""


def ann_sharpe(r):
    r = pd.Series(r).dropna()
    return r.mean() / r.std() * np.sqrt(12) if r.std() > 0 else np.nan


def max_dd(r):
    cum = (1 + pd.Series(r).dropna()).cumprod()
    return float((cum / cum.cummax() - 1).min())


def calmar(r):
    r   = pd.Series(r).dropna()
    mdd = abs(max_dd(r))
    return r.mean() * 12 / mdd if mdd > 0 else np.nan


def var95(r):
    return float(np.percentile(pd.Series(r).dropna(), 5))


# ── Output helpers ─────────────────────────────────────────────────────────────

def save_table(df, name, index=True):
    os.makedirs(TAB_DIR, exist_ok=True)
    df.to_csv(os.path.join(TAB_DIR, f"{name}.csv"), index=index)
    try:
        df.to_latex(os.path.join(TAB_DIR, f"{name}.tex"),
                    index=index, float_format="%.4f", na_rep="")
    except Exception:
        pass
    print(f"  Saved → output/tables/{name}.{{csv,tex}}")


def save_figure(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, f"{name}.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG_DIR, f"{name}.png"), dpi=300,
                bbox_inches="tight")
    print(f"  Saved → output/figures/{name}.{{pdf,png}}")
