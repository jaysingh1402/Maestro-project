"""
06_portfolio_accumulation.py — Cyber Accumulation Risk Module
==============================================================

Models correlated exposure concentration — the key systemic risk driver that
per-policy models ignore.  When many policies share the same cloud provider,
core banking vendor, or geographic region, a single systemic event can trigger
correlated claims far exceeding what independent models predict.

Analyses:
  1. Cloud provider concentration (% TIV, % policies, aggregate PML)
  2. Core banking / fintech vendor concentration
  3. Geographic / regional concentration
  4. Vendor Herfindahl-Hirschman Index (HHI) — market concentration measure
  5. Maximum Foreseeable Loss (MFL) per concentration group
  6. Accumulation Risk Score per policy (for reinsurance flagging)

Run from the project root:
    python code/06_portfolio_accumulation.py

Outputs:
    outputs/model_outputs/accumulation_risk.json
    outputs/eda_visuals/accumulation_heatmap.html
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import warnings
warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
VIS_DIR  = ROOT / "outputs" / "eda_visuals"
MDL_DIR  = ROOT / "outputs" / "model_outputs"
VIS_DIR.mkdir(parents=True, exist_ok=True)
MDL_DIR.mkdir(parents=True, exist_ok=True)

# ── Load portfolio ─────────────────────────────────────────────────────────────
feat_path = DATA_DIR / "09_cyber_pricing_features.csv"
if not feat_path.exists():
    feat_path = DATA_DIR / "07_modeling_dataset.csv"
df = pd.read_csv(feat_path)

# Ensure required columns
for col, default in [
    ("cloud_provider_primary", "Unknown"),
    ("core_banking_vendor",    "Unknown"),
    ("region",                 "Unknown"),
    ("sub_sector",             "Unknown"),
    ("limit_mm",               5.0),
    ("retention_mm",           0.25),
    ("premium_usd",            0.0),
    ("revenue_mm",             50.0),
]:
    if col not in df.columns:
        df[col] = default

df["limit_usd"]     = df["limit_mm"].fillna(5.0)       * 1_000_000
df["retention_usd"] = df["retention_mm"].fillna(0.25)  * 1_000_000
df["revenue_usd"]   = df["revenue_mm"].fillna(50.0)    * 1_000_000
n_policies          = len(df)
total_tiv           = df["limit_usd"].sum()
total_premium       = df["premium_usd"].sum()

print("=" * 60)
print("PORTFOLIO ACCUMULATION RISK MODULE")
print("=" * 60)
print(f"Portfolio: {n_policies} policies")
print(f"Total TIV: ${total_tiv:,.0f}")
print(f"Total Premium: ${total_premium:,.0f}")

# ─────────────────────────────────────────────────────────────────────────────
# HELPER: Compute concentration metrics for any grouping variable
# ─────────────────────────────────────────────────────────────────────────────
def concentration_table(df_: pd.DataFrame, group_col: str) -> pd.DataFrame:
    grp = df_.groupby(group_col, dropna=False).agg(
        n_policies   = ("limit_usd",   "count"),
        total_tiv    = ("limit_usd",   "sum"),
        total_premium= ("premium_usd", "sum"),
        avg_limit    = ("limit_usd",   "mean"),
        avg_revenue  = ("revenue_usd", "mean"),
    ).reset_index()
    grp.rename(columns={group_col: "group"}, inplace=True)
    grp["pct_policies"] = grp["n_policies"] / n_policies
    grp["pct_tiv"]      = grp["total_tiv"]  / total_tiv
    # Maximum Foreseeable Loss: 80% of group TIV (not all will be fully covered)
    grp["mfl_usd"]      = grp["total_tiv"] * 0.80
    grp = grp.sort_values("total_tiv", ascending=False).reset_index(drop=True)
    return grp

# Herfindahl–Hirschman Index (HHI) — sum of squared market shares
# HHI < 0.15: diversified | 0.15–0.25: moderate | >0.25: concentrated
def compute_hhi(shares: np.ndarray) -> float:
    return float(np.sum(shares ** 2))

# ─────────────────────────────────────────────────────────────────────────────
# 1. CLOUD PROVIDER CONCENTRATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1] Cloud Provider Concentration")
cloud_tbl = concentration_table(df, "cloud_provider_primary")
cloud_hhi  = compute_hhi(cloud_tbl["pct_tiv"].values)

# Industry benchmark: Flexera 2024 — AWS ~32%, Azure ~23%, GCP ~11%
cloud_benchmark = {"AWS": 0.32, "Azure": 0.23, "GCP": 0.11, "Hybrid": 0.21, "Private": 0.13}

print(cloud_tbl[["group", "n_policies", "pct_tiv", "mfl_usd"]].to_string(index=False))
print(f"  Cloud HHI: {cloud_hhi:.4f} ({'Concentrated' if cloud_hhi > 0.25 else 'Moderate' if cloud_hhi > 0.15 else 'Diversified'})")

# ─────────────────────────────────────────────────────────────────────────────
# 2. CORE BANKING VENDOR CONCENTRATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2] Core Banking Vendor Concentration")
vendor_tbl = concentration_table(df, "core_banking_vendor")
vendor_hhi  = compute_hhi(vendor_tbl["pct_tiv"].values)

print(vendor_tbl[["group", "n_policies", "pct_tiv", "mfl_usd"]].to_string(index=False))
print(f"  Vendor HHI: {vendor_hhi:.4f} ({'Concentrated' if vendor_hhi > 0.25 else 'Moderate' if vendor_hhi > 0.15 else 'Diversified'})")

# ─────────────────────────────────────────────────────────────────────────────
# 3. GEOGRAPHIC CONCENTRATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3] Geographic Concentration")
geo_tbl   = concentration_table(df, "region")
geo_hhi   = compute_hhi(geo_tbl["pct_tiv"].values)

print(geo_tbl[["group", "n_policies", "pct_tiv", "mfl_usd"]].to_string(index=False))
print(f"  Geographic HHI: {geo_hhi:.4f} ({'Concentrated' if geo_hhi > 0.25 else 'Moderate' if geo_hhi > 0.15 else 'Diversified'})")

# ─────────────────────────────────────────────────────────────────────────────
# 4. SUB-SECTOR CONCENTRATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4] Sub-Sector Concentration")
sector_tbl = concentration_table(df, "sub_sector")
sector_hhi = compute_hhi(sector_tbl["pct_tiv"].values)

print(sector_tbl[["group", "n_policies", "pct_tiv", "mfl_usd"]].to_string(index=False))
print(f"  Sector HHI: {sector_hhi:.4f} ({'Concentrated' if sector_hhi > 0.25 else 'Moderate' if sector_hhi > 0.15 else 'Diversified'})")

# ─────────────────────────────────────────────────────────────────────────────
# 5. PER-POLICY ACCUMULATION RISK SCORE
# ─────────────────────────────────────────────────────────────────────────────
# Score = weighted sum of the % TIV in each group the policy belongs to
# Higher score → policy sits in more concentrated / riskier groups

def group_tiv_pct(df_: pd.DataFrame, tbl: pd.DataFrame, col: str) -> pd.Series:
    mapping = dict(zip(tbl["group"].astype(str), tbl["pct_tiv"]))
    return df_[col].astype(str).map(mapping).fillna(0.0)

df["cloud_conc"]  = group_tiv_pct(df, cloud_tbl,  "cloud_provider_primary")
df["vendor_conc"] = group_tiv_pct(df, vendor_tbl, "core_banking_vendor")
df["geo_conc"]    = group_tiv_pct(df, geo_tbl,    "region")
df["sector_conc"] = group_tiv_pct(df, sector_tbl, "sub_sector")

# Weighted accumulation score (cloud and vendor are highest systemic risk)
df["accumulation_risk_score"] = (
    0.35 * df["cloud_conc"]
  + 0.30 * df["vendor_conc"]
  + 0.20 * df["geo_conc"]
  + 0.15 * df["sector_conc"]
)
df["accumulation_risk_band"] = pd.qcut(
    df["accumulation_risk_score"], q=4,
    labels=["Low", "Moderate", "High", "Extreme"]
)

# Policies flagged for RI accumulation review (top 10% risk score)
ri_flag_threshold = df["accumulation_risk_score"].quantile(0.90)
df["ri_flag"]     = df["accumulation_risk_score"] >= ri_flag_threshold
n_ri_flagged      = int(df["ri_flag"].sum())
print(f"\n[5] Accumulation Risk Scores computed. Policies flagged for RI review: {n_ri_flagged} ({n_ri_flagged/n_policies:.1%})")

# ─────────────────────────────────────────────────────────────────────────────
# 6. PLOTLY DASHBOARD: ACCUMULATION HEATMAP
# ─────────────────────────────────────────────────────────────────────────────
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Cloud Provider — % of Total TIV",
            "Core Banking Vendor — % of Total TIV",
            "Regional Concentration — % of Total TIV",
            "Accumulation Risk Band Distribution",
        ],
        specs=[[{"type": "pie"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "bar"}]],
    )

    # Cloud treemap → pie
    fig.add_trace(go.Pie(
        labels=cloud_tbl["group"],
        values=cloud_tbl["total_tiv"],
        textinfo="label+percent",
        hole=0.35,
        name="Cloud",
    ), row=1, col=1)

    # Vendor bar
    fig.add_trace(go.Bar(
        x=vendor_tbl["group"],
        y=vendor_tbl["total_tiv"] / 1e6,
        text=[f"n={int(v)}" for v in vendor_tbl["n_policies"]],
        textposition="auto",
        name="Vendor TIV ($M)",
        marker_color="#6366f1",
    ), row=1, col=2)

    # Geo bar
    fig.add_trace(go.Bar(
        x=geo_tbl["group"],
        y=geo_tbl["total_tiv"] / 1e6,
        text=[f"{p:.1%}" for p in geo_tbl["pct_tiv"]],
        textposition="auto",
        name="Geo TIV ($M)",
        marker_color="#10b981",
    ), row=2, col=1)

    # Accumulation band distribution
    band_counts = df["accumulation_risk_band"].value_counts().reindex(["Low", "Moderate", "High", "Extreme"])
    band_colors = {"Low": "#10b981", "Moderate": "#f59e0b", "High": "#f97316", "Extreme": "#ef4444"}
    fig.add_trace(go.Bar(
        x=band_counts.index.tolist(),
        y=band_counts.values,
        marker_color=[band_colors.get(b, "#6b7280") for b in band_counts.index],
        text=band_counts.values,
        textposition="auto",
        name="Policies by Accumulation Band",
    ), row=2, col=2)

    fig.update_layout(
        title_text=(
            f"<b>Portfolio Accumulation Risk Dashboard</b><br>"
            f"<sup>{n_policies} policies | Total TIV ${total_tiv/1e9:.2f}B | "
            f"Cloud HHI={cloud_hhi:.3f} | Vendor HHI={vendor_hhi:.3f}</sup>"
        ),
        paper_bgcolor="#0f172a",
        plot_bgcolor="#1e293b",
        font=dict(color="#e2e8f0", family="Inter, sans-serif"),
        showlegend=False,
        height=700,
    )
    for ann in fig.layout.annotations:
        ann.font.color = "#94a3b8"

    out_html = VIS_DIR / "accumulation_heatmap.html"
    fig.write_html(str(out_html))
    print(f"  Accumulation dashboard saved to {out_html}")

except ImportError:
    print("  plotly not available — skipping heatmap HTML generation")

# ─────────────────────────────────────────────────────────────────────────────
# 7. SAVE JSON
# ─────────────────────────────────────────────────────────────────────────────

def tbl_to_list(tbl: pd.DataFrame) -> list:
    """Convert a concentration table to JSON-serialisable list of dicts."""
    rows = []
    for _, r in tbl.iterrows():
        rows.append({
            "group":           str(r["group"]),
            "n_policies":      int(r["n_policies"]),
            "total_tiv_usd":   round(float(r["total_tiv"]), 0),
            "pct_policies":    round(float(r["pct_policies"]), 4),
            "pct_tiv":         round(float(r["pct_tiv"]), 4),
            "mfl_usd":         round(float(r["mfl_usd"]), 0),
        })
    return rows

accumulation_output = {
    "portfolio_summary": {
        "n_policies":    n_policies,
        "total_tiv_usd": round(float(total_tiv), 0),
        "total_premium": round(float(total_premium), 0),
    },
    "hhi_scores": {
        "cloud_provider_hhi": round(cloud_hhi,  4),
        "vendor_hhi":         round(vendor_hhi, 4),
        "geographic_hhi":     round(geo_hhi,    4),
        "sector_hhi":         round(sector_hhi, 4),
        "interpretation": "HHI < 0.15: Diversified | 0.15-0.25: Moderate | >0.25: Concentrated",
    },
    "cloud_concentration":  tbl_to_list(cloud_tbl),
    "vendor_concentration": tbl_to_list(vendor_tbl),
    "geo_concentration":    tbl_to_list(geo_tbl),
    "sector_concentration": tbl_to_list(sector_tbl),
    "accumulation_risk": {
        "ri_flag_threshold":   round(float(ri_flag_threshold), 6),
        "n_policies_flagged":  n_ri_flagged,
        "pct_flagged":         round(n_ri_flagged / n_policies, 4),
        "band_distribution": {
            b: int((df["accumulation_risk_band"] == b).sum())
            for b in ["Low", "Moderate", "High", "Extreme"]
        },
    },
    "cloud_benchmark_vs_portfolio": {
        k: {
            "benchmark_pct": v,
            "portfolio_pct": round(
                float(cloud_tbl.loc[cloud_tbl["group"] == k, "pct_tiv"].values[0])
                if k in cloud_tbl["group"].values else 0.0, 4
            ),
        }
        for k, v in cloud_benchmark.items()
    },
}

out_path = MDL_DIR / "accumulation_risk.json"
with open(out_path, "w") as f:
    json.dump(accumulation_output, f, indent=2)

print(f"  Accumulation risk data saved to {out_path}")
print("\n" + "=" * 60)
print("ACCUMULATION RISK MODULE COMPLETE")
print("=" * 60)
