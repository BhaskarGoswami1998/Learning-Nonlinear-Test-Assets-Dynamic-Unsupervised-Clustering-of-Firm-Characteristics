"""
generate_synthetic_data.py
--------------------------
Generates fully synthetic sample data that mimics the structure of the
original data files.  Run this if you do NOT have the original raw data
and want to test that all notebooks execute end-to-end.

Usage:
    python utils/generate_synthetic_data.py

Outputs (written to data/sample/):
    cluster_month_panel_K_50_lambda_1000000.csv
    factor_returns.csv
    macro_predictors.csv
"""

import os
import numpy as np
import pandas as pd

REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE_DIR = os.path.join(REPO_ROOT, "data", "sample")
os.makedirs(SAMPLE_DIR, exist_ok=True)

SEED   = 42
np.random.seed(SEED)

# ── Time grid ────────────────────────────────────────────────────────────────
dates = pd.date_range("1977-01-01", "2020-12-01", freq="MS")
T     = len(dates)
K     = 50          # clusters
N_PER_CLUSTER = 20  # firms per cluster (synthetic)
N_FIRMS = K * N_PER_CLUSTER

CHARS_45 = [
    "A2ME","AC","AT","ATO","B2M","BETA_d","BETA_m","C2A","CF2B","CF2P",
    "CTO","D2A","D2P","DPI2A","E2P","FC2Y","HIGH52","IdioVol","INV","LEV",
    "ME","NI","NOA","OA","OL","OP","PCM","PM","PROF","Q",
    "R12_2","R12_7","R2_1","R36_13","R60_13","RNA","ROA","ROE","RVAR",
    "S2P","SGA2S","SPREAD","SUV","TURN","VAR"
]

print("Generating synthetic cluster panel …")

# Cluster-level expected return gradient (L01 worst, L50 best)
cluster_mu = np.linspace(-0.018, 0.016, K)

rows = []
for t_idx, dt in enumerate(dates):
    for k in range(1, K + 1):
        mu_k   = cluster_mu[k - 1]
        sig_k  = 0.05 + 0.02 * np.random.rand()
        n_firm = N_PER_CLUSTER + np.random.randint(-5, 6)
        n_firm = max(5, n_firm)

        for firm_i in range(n_firm):
            permno = (k - 1) * N_PER_CLUSTER + firm_i + 1
            ret    = np.random.normal(mu_k, sig_k)
            me     = np.exp(np.random.normal(5 + 0.1 * k, 1.5))

            # Characteristics: centroid + noise, centroid drifts with cluster rank
            char_vals = {}
            for j, char in enumerate(CHARS_45):
                centroid = (k / K - 0.5) * 0.3 + 0.02 * np.sin(j + t_idx * 0.01)
                char_vals[char] = float(np.clip(
                    centroid + np.random.normal(0, 0.08), -0.5, 0.5
                ))

            row = {"date": dt, "permno": permno, "cluster": k,
                   "ret": round(ret, 6), "me": round(me, 2)}
            row.update(char_vals)
            rows.append(row)

panel = pd.DataFrame(rows)
out_path = os.path.join(SAMPLE_DIR, "cluster_month_panel_K_50_lambda_1000000.csv")
panel.to_csv(out_path, index=False)
print(f"  Saved panel: {panel.shape} → {out_path}")

# ── Synthetic factor returns ─────────────────────────────────────────────────
print("Generating synthetic factor returns …")

mkt  = pd.Series(np.random.normal(0.006, 0.045, T), index=dates)
smb  = pd.Series(np.random.normal(0.002, 0.030, T), index=dates)
hml  = pd.Series(np.random.normal(0.003, 0.030, T), index=dates)
rmw  = pd.Series(np.random.normal(0.002, 0.020, T), index=dates)
cma  = pd.Series(np.random.normal(0.001, 0.018, T), index=dates)
mom  = pd.Series(np.random.normal(0.005, 0.055, T), index=dates)
liq  = pd.Series(np.random.normal(0.001, 0.025, T), index=dates)
pead = pd.Series(np.random.normal(0.003, 0.022, T), index=dates)
fin  = pd.Series(np.random.normal(0.001, 0.018, T), index=dates)
ia   = pd.Series(np.random.normal(0.002, 0.018, T), index=dates)
roe  = pd.Series(np.random.normal(0.004, 0.024, T), index=dates)
eg   = pd.Series(np.random.normal(0.002, 0.016, T), index=dates)
mgmt = pd.Series(np.random.normal(0.003, 0.022, T), index=dates)
perf = pd.Series(np.random.normal(0.003, 0.028, T), index=dates)
rf   = pd.Series(np.abs(np.random.normal(0.003, 0.002, T)), index=dates)

# Latent factor proxies for PCA5 / RPPCA5
for i in range(1, 6):
    locals()[f"PC{i}"]  = pd.Series(np.random.normal(0, 0.03, T), index=dates)
    locals()[f"RPC{i}"] = pd.Series(np.random.normal(0, 0.03, T), index=dates)

factor_df = pd.DataFrame({
    "date": dates,
    "MKT": mkt, "SMB": smb, "HML": hml, "RMW": rmw, "CMA": cma,
    "MOM": mom, "LIQ": liq, "PEAD": pead, "FIN": fin,
    "IA": ia, "ROE": roe, "EG": eg, "MGMT": mgmt, "PERF": perf,
    "RF": rf,
    **{f"PC{i}":  locals()[f"PC{i}"]  for i in range(1, 6)},
    **{f"RPC{i}": locals()[f"RPC{i}"] for i in range(1, 6)},
})
fpath = os.path.join(SAMPLE_DIR, "factor_returns.csv")
factor_df.to_csv(fpath, index=False)
print(f"  Saved factor returns: {factor_df.shape} → {fpath}")

# ── Synthetic macro predictors ───────────────────────────────────────────────
print("Generating synthetic macro predictors …")

# Persistent macro variables (AR(1) processes)
def ar1(T, rho=0.97, mu=0.0, sigma=0.01):
    x = np.zeros(T)
    x[0] = mu
    for t in range(1, T):
        x[t] = mu + rho * (x[t-1] - mu) + np.random.normal(0, sigma)
    return x

macro_df = pd.DataFrame({
    "date": dates,
    "dfy":  ar1(T, 0.96, 0.009, 0.002),    # default yield spread
    "dy":   ar1(T, 0.97, 0.034, 0.005),    # dividend yield
    "ep":   ar1(T, 0.96, 0.056, 0.006),    # earnings-price ratio
    "ill":  ar1(T, 0.95, 0.20,  0.05),     # illiquidity
    "infl": ar1(T, 0.94, 0.004, 0.002),    # inflation
    "lev":  ar1(T, 0.96, 0.30,  0.02),     # market leverage
    "ni":   ar1(T, 0.90, 0.01,  0.008),    # net issuance
    "svar": np.abs(ar1(T, 0.85, 0.003, 0.003)),  # stock variance
    "tbl":  ar1(T, 0.98, 0.04,  0.003),    # T-bill rate
    "tms":  ar1(T, 0.95, 0.015, 0.004),    # term spread
})
mpath = os.path.join(SAMPLE_DIR, "macro_predictors.csv")
macro_df.to_csv(mpath, index=False)
print(f"  Saved macro predictors: {macro_df.shape} → {mpath}")

print("\nAll synthetic data files generated successfully.")
print("You can now run notebooks 01–10 without original data.")
print(f"Output directory: {SAMPLE_DIR}")
