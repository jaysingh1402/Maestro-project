"""
04_catastrophe_scenarios.py — Named Cyber Catastrophe Scenario Engine
=======================================================================

Runs four deterministic + stochastic stress scenarios calibrated to real
historical cyber events. Unlike the stochastic Hawkes simulation (which
models random claim arrivals), this engine models specific correlated
shock scenarios where a single event affects large swaths of the portfolio
simultaneously.

Scenarios:
  1. Major Cloud Provider Outage   (AWS/Azure-scale — CrowdStrike 2024: $5.4B)
  2. Global Ransomware Campaign    (WannaCry/NotPetya-scale: $4–10B)
  3. Supply Chain Software Attack  (SolarWinds/MOVEit-scale: $90M–$1B+)
  4. Critical Infrastructure Attack (Colonial Pipeline / CISA FS-ISAC scenarios)

Run from the project root:
    python code/04_catastrophe_scenarios.py

Outputs:
    outputs/model_outputs/scenario_results.json
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent
DATA_DIR  = ROOT / "data"
MODEL_DIR = ROOT / "outputs" / "model_outputs"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ── Load portfolio ─────────────────────────────────────────────────────────────
feat_path = DATA_DIR / "09_cyber_pricing_features.csv"
if not feat_path.exists():
    feat_path = DATA_DIR / "07_modeling_dataset.csv"
df = pd.read_csv(feat_path)

# Ensure required columns exist (fallback defaults)
for col, default in [("cloud_provider_primary", "Unknown"),
                      ("core_banking_vendor",    "Unknown"),
                      ("sub_sector",             "Unknown"),
                      ("revenue_mm",             50.0),
                      ("limit_mm",               5.0),
                      ("retention_mm",           0.25),
                      ("cyber_control_score",    0.60)]:
    if col not in df.columns:
        df[col] = default

df["limit_usd"]     = df["limit_mm"].fillna(5.0) * 1_000_000
df["retention_usd"] = df["retention_mm"].fillna(0.25) * 1_000_000
df["revenue_daily"] = df["revenue_mm"].fillna(50.0) * 1_000_000 / 365.0

n_policies = len(df)
total_tiv  = df["limit_usd"].sum()

print("=" * 65)
print("CYBER CATASTROPHE SCENARIO ENGINE")
print("=" * 65)
print(f"Portfolio: {n_policies} policies | Total Insured Value: ${total_tiv:,.0f}")

# ── Simulation parameters ──────────────────────────────────────────────────────
N_SIMS = 10_000   # Monte Carlo iterations per scenario

# ─────────────────────────────────────────────────────────────────────────────
# HELPER: Apply policy structure (retention + limit)
# ─────────────────────────────────────────────────────────────────────────────
def apply_policy_structure(gross_loss: np.ndarray,
                            retention: np.ndarray,
                            limit: np.ndarray) -> np.ndarray:
    """Net insurer loss = max(0, gross - retention), capped at limit."""
    return np.minimum(limit, np.maximum(0.0, gross_loss - retention))

# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO 1: MAJOR CLOUD PROVIDER OUTAGE
# ─────────────────────────────────────────────────────────────────────────────
def run_cloud_outage_scenario(df_: pd.DataFrame, n_sims: int) -> dict:
    """
    Models a severe multi-region cloud provider outage lasting 4-72 hours.
    Real-world basis: CrowdStrike Falcon update (2024, $5.4B insured losses),
    AWS us-east-1 multi-service outage (2021), Cloudflare global outage (2023).

    Affected policies: those whose primary cloud provider is AWS or Azure
    (~35–45% of a typical FS portfolio according to Flexera 2024).
    """
    target_providers = ["AWS", "Azure"]
    affected_mask    = df_["cloud_provider_primary"].isin(target_providers)
    df_aff           = df_[affected_mask].copy()
    n_aff            = len(df_aff)
    pct_aff          = n_aff / len(df_)

    # Stochastic downtime: bimodal — quick recovery (4-8h) vs. extended (24-72h)
    # Bimodal parameters calibrated to CrowdStrike (avg 10h) and AWS (6-48h range)
    portfolio_losses = np.zeros(n_sims)
    for sim in range(n_sims):
        per_policy_gross = np.zeros(n_aff)
        for j, (_, row) in enumerate(df_aff.iterrows()):
            # Each policy: random chance it actually falls in the outage zone
            if np.random.random() > 0.85:   # ~85% of "affected" are actually impacted
                continue
            # Downtime: Gaussian mixture (short vs. extended recovery)
            if np.random.random() < 0.65:   # 65% quick recovery: 4-12h
                downtime_h = np.clip(np.random.normal(8, 3), 2, 14)
            else:                            # 35% extended: 24-72h
                downtime_h = np.clip(np.random.normal(48, 18), 12, 96)

            rev_dep      = np.random.uniform(0.30, 0.90)   # % of revenue dependent on cloud
            time_mult    = np.random.choice([1.0, 1.4, 1.85], p=[0.60, 0.22, 0.18])  # regular/month-end/quarter-end
            lost_rev     = (downtime_h / 24) * row["revenue_daily"] * rev_dep * time_mult
            extra_exp    = lost_rev * 0.30
            forensics    = min(350_000, lost_rev * 0.08)
            notification = 125_000 if downtime_h > 36 else 0
            gross        = lost_rev + extra_exp + forensics + notification

            # Regulatory cost (FINRA/OCC: >36h triggers notification penalties)
            if row.get("primary_regulator", "") in ("OCC", "FRB", "FDIC", "FINRA"):
                if downtime_h > 36:
                    gross += 250_000
            per_policy_gross[j] = gross

        net_losses = apply_policy_structure(per_policy_gross, df_aff["retention_usd"].values, df_aff["limit_usd"].values)
        portfolio_losses[sim] = net_losses.sum()

    return {
        "scenario":             "Major Cloud Provider Outage",
        "real_world_basis":     "CrowdStrike Falcon 2024 ($5.4B), AWS us-east-1 2021, Cloudflare 2023",
        "target_providers":     target_providers,
        "n_affected_policies":  int(n_aff),
        "pct_portfolio":        round(pct_aff, 3),
        "expected_loss_usd":    round(float(np.mean(portfolio_losses)), 0),
        "pml_90_usd":           round(float(np.percentile(portfolio_losses, 90)), 0),
        "pml_99_usd":           round(float(np.percentile(portfolio_losses, 99)), 0),
        "max_scenario_usd":     round(float(np.max(portfolio_losses)), 0),
        "loss_distribution_summary": {
            "p50": round(float(np.percentile(portfolio_losses, 50)), 0),
            "p75": round(float(np.percentile(portfolio_losses, 75)), 0),
            "p90": round(float(np.percentile(portfolio_losses, 90)), 0),
            "p95": round(float(np.percentile(portfolio_losses, 95)), 0),
            "p99": round(float(np.percentile(portfolio_losses, 99)), 0),
        },
    }

# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO 2: GLOBAL RANSOMWARE CAMPAIGN
# ─────────────────────────────────────────────────────────────────────────────
def run_ransomware_campaign_scenario(df_: pd.DataFrame, n_sims: int) -> dict:
    """
    Models a coordinated global ransomware campaign targeting financial services.
    Real-world basis: WannaCry 2017 ($4-8B), NotPetya 2017 ($10B+), REvil/Kaseya 2021.
    ~18% of portfolio hit; higher severity for weaker controls.
    """
    base_hit_rate    = 0.18   # 18% of portfolio affected (DBIR 2024)
    portfolio_losses = np.zeros(n_sims)

    for sim in range(n_sims):
        # Random % affected: triangular around base rate
        hit_rate = np.clip(np.random.triangular(0.10, base_hit_rate, 0.30), 0.01, 0.50)
        hit_mask = np.random.random(len(df_)) < hit_rate
        df_hit   = df_[hit_mask].copy()
        if len(df_hit) == 0:
            continue

        # Severity correlated inversely with cyber control score
        # Weaker controls → longer downtime, larger ransom demand
        control_scores = df_hit["cyber_control_score"].fillna(0.6).values
        downtime_h     = np.clip(np.random.lognormal(np.log(72), 1.2, len(df_hit))
                                  * (1.5 - control_scores), 4, 720)

        rev_daily      = df_hit["revenue_daily"].values
        lost_rev       = (downtime_h / 24) * rev_daily * np.random.uniform(0.5, 1.0, len(df_hit))
        ransom_demand  = rev_daily * 365 * 0.02 * np.random.uniform(0.5, 3.0, len(df_hit))  # 2% of annual revenue
        pct_pay        = 0.46   # industry rate: 46% of victims pay (Coveware 2024)
        ransom_paid    = ransom_demand * (np.random.random(len(df_hit)) < pct_pay)
        extra_expense  = lost_rev * 0.35
        forensics      = np.minimum(2_000_000, lost_rev * 0.12)
        notification   = np.where(downtime_h > 48, 500_000, 150_000)

        # Regulatory fines for FS sector (OCC/FRB >36h rule)
        reg_regulator  = df_hit.get("primary_regulator", pd.Series(["Other"] * len(df_hit))).values
        reg_fine       = np.where(
            np.isin(reg_regulator, ["OCC", "FRB", "FDIC"]) & (downtime_h > 36),
            np.random.uniform(250_000, 2_000_000, len(df_hit)), 0
        )

        gross_loss     = lost_rev + ransom_paid + extra_expense + forensics + notification + reg_fine
        net_loss       = apply_policy_structure(gross_loss, df_hit["retention_usd"].values, df_hit["limit_usd"].values)
        portfolio_losses[sim] = net_loss.sum()

    return {
        "scenario":            "Global Ransomware Campaign",
        "real_world_basis":    "WannaCry 2017 ($4-8B), NotPetya 2017 ($10B+), REvil/Kaseya 2021",
        "avg_pct_affected":    round(base_hit_rate, 3),
        "expected_loss_usd":   round(float(np.mean(portfolio_losses)), 0),
        "pml_90_usd":          round(float(np.percentile(portfolio_losses, 90)), 0),
        "pml_99_usd":          round(float(np.percentile(portfolio_losses, 99)), 0),
        "max_scenario_usd":    round(float(np.max(portfolio_losses)), 0),
        "loss_distribution_summary": {
            "p50": round(float(np.percentile(portfolio_losses, 50)), 0),
            "p75": round(float(np.percentile(portfolio_losses, 75)), 0),
            "p90": round(float(np.percentile(portfolio_losses, 90)), 0),
            "p95": round(float(np.percentile(portfolio_losses, 95)), 0),
            "p99": round(float(np.percentile(portfolio_losses, 99)), 0),
        },
    }

# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO 3: SUPPLY CHAIN SOFTWARE COMPROMISE
# ─────────────────────────────────────────────────────────────────────────────
def run_supply_chain_scenario(df_: pd.DataFrame, n_sims: int) -> dict:
    """
    Models a sophisticated supply chain attack compromising a widely-used
    financial software vendor (core banking / payment platform).
    Real-world basis: SolarWinds 2020, MOVEit 2023 ($1B+, 2,600 orgs),
    XZ Utils 2024.
    Key feature: long discovery lag (avg 212 days) → extended IBNR claims.
    """
    # Affected: policies using the most common core banking vendors
    target_vendors = ["Fiserv", "FIS", "Jack Henry", "Unknown"]
    aff_mask       = df_["core_banking_vendor"].isin(target_vendors)
    df_aff         = df_[aff_mask].copy()
    n_aff          = len(df_aff)
    pct_aff        = n_aff / len(df_)

    portfolio_losses = np.zeros(n_sims)
    for sim in range(n_sims):
        # Not all vendor users are compromised (attacker selectivity)
        compromised = np.random.random(n_aff) < np.random.uniform(0.20, 0.45)
        df_comp     = df_aff[compromised].copy()
        if len(df_comp) == 0:
            continue

        # Discovery lag: lognormal centered at 212 days (SolarWinds average)
        disc_lag_days  = np.clip(np.random.lognormal(np.log(212), 0.8, len(df_comp)), 30, 730)
        # During the dwell period, adversary exfiltrates data → notification costs
        n_records_mm   = np.random.uniform(0.1, 5.0, len(df_comp))   # millions of records
        notification   = n_records_mm * 30_000                         # $30/record notification cost
        forensics      = np.random.uniform(200_000, 2_000_000, len(df_comp))
        # BI loss: operational disruption upon discovery
        downtime_h     = np.clip(np.random.exponential(48, len(df_comp)), 4, 480)
        bi_loss        = (downtime_h / 24) * df_comp["revenue_daily"].values * 0.40

        # Regulatory fines: GDPR-style ($20M or 4% of global revenue, whichever greater)
        rev_annual     = df_comp["revenue_mm"].fillna(50.0).values * 1_000_000
        reg_fine       = np.maximum(
            np.random.uniform(1_000_000, 10_000_000, len(df_comp)),
            rev_annual * np.random.uniform(0.005, 0.04, len(df_comp))
        )

        gross_loss = notification + forensics + bi_loss + reg_fine
        net_loss   = apply_policy_structure(gross_loss, df_comp["retention_usd"].values, df_comp["limit_usd"].values)
        portfolio_losses[sim] = net_loss.sum()

    return {
        "scenario":             "Supply Chain Software Compromise",
        "real_world_basis":     "SolarWinds 2020, MOVEit 2023 ($1B+, 2,600 orgs), XZ Utils 2024",
        "target_vendors":       target_vendors,
        "n_affected_policies":  int(n_aff),
        "pct_portfolio":        round(pct_aff, 3),
        "avg_discovery_lag_days": 212,
        "expected_loss_usd":    round(float(np.mean(portfolio_losses)), 0),
        "pml_90_usd":           round(float(np.percentile(portfolio_losses, 90)), 0),
        "pml_99_usd":           round(float(np.percentile(portfolio_losses, 99)), 0),
        "max_scenario_usd":     round(float(np.max(portfolio_losses)), 0),
        "loss_distribution_summary": {
            "p50": round(float(np.percentile(portfolio_losses, 50)), 0),
            "p75": round(float(np.percentile(portfolio_losses, 75)), 0),
            "p90": round(float(np.percentile(portfolio_losses, 90)), 0),
            "p95": round(float(np.percentile(portfolio_losses, 95)), 0),
            "p99": round(float(np.percentile(portfolio_losses, 99)), 0),
        },
    }

# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO 4: CRITICAL INFRASTRUCTURE ATTACK
# ─────────────────────────────────────────────────────────────────────────────
def run_critical_infra_scenario(df_: pd.DataFrame, n_sims: int) -> dict:
    """
    Nation-state-level attack targeting payment processors and core banking
    in the US financial sector.
    Real-world basis: Colonial Pipeline 2021 ($34M+), CISA FS-ISAC scenario
    planning, US Treasury Section 9 systemic risk exercises.
    Targeted attack on Payment Processor and Retail/Regional Banks.
    """
    target_sectors   = ["Payment Processor", "Retail Bank", "Regional Bank",
                         "Broker-Dealer", "Community Bank"]
    aff_mask         = df_["sub_sector"].isin(target_sectors)
    df_aff           = df_[aff_mask].copy()
    n_aff            = len(df_aff)
    pct_aff          = n_aff / len(df_)

    portfolio_losses = np.zeros(n_sims)
    for sim in range(n_sims):
        # Attack intensity: stochastic (some scenarios hit harder)
        attack_intensity = np.random.triangular(0.5, 1.0, 2.5)

        downtime_h  = np.clip(
            np.random.lognormal(np.log(120), 0.9, n_aff) * attack_intensity, 24, 720
        )
        rev_impact  = np.random.uniform(0.60, 1.00, n_aff)  # full BI during critical infra attack
        lost_rev    = (downtime_h / 24) * df_aff["revenue_daily"].values * rev_impact

        extra_exp   = lost_rev * 0.40
        forensics   = np.minimum(5_000_000, lost_rev * 0.15)

        # Systemic regulatory fines (US Treasury, OFAC, Federal Reserve)
        reg_fine    = np.where(
            downtime_h > 72,
            np.random.uniform(2_000_000, 50_000_000, n_aff),
            np.random.uniform(500_000,    5_000_000,  n_aff)
        )
        # Reputational run-off costs (customer churn, market cap impact — only for public firms)
        reputation  = lost_rev * np.random.uniform(0.10, 0.40, n_aff)

        gross_loss = lost_rev + extra_exp + forensics + reg_fine + reputation
        net_loss   = apply_policy_structure(gross_loss, df_aff["retention_usd"].values, df_aff["limit_usd"].values)
        portfolio_losses[sim] = net_loss.sum()

    return {
        "scenario":             "Critical Infrastructure Attack",
        "real_world_basis":     "Colonial Pipeline 2021, CISA FS-ISAC Scenarios, US Treasury Section 9",
        "target_sectors":       target_sectors,
        "n_affected_policies":  int(n_aff),
        "pct_portfolio":        round(pct_aff, 3),
        "expected_loss_usd":    round(float(np.mean(portfolio_losses)), 0),
        "pml_90_usd":           round(float(np.percentile(portfolio_losses, 90)), 0),
        "pml_99_usd":           round(float(np.percentile(portfolio_losses, 99)), 0),
        "max_scenario_usd":     round(float(np.max(portfolio_losses)), 0),
        "loss_distribution_summary": {
            "p50": round(float(np.percentile(portfolio_losses, 50)), 0),
            "p75": round(float(np.percentile(portfolio_losses, 75)), 0),
            "p90": round(float(np.percentile(portfolio_losses, 90)), 0),
            "p95": round(float(np.percentile(portfolio_losses, 95)), 0),
            "p99": round(float(np.percentile(portfolio_losses, 99)), 0),
        },
    }

# ─────────────────────────────────────────────────────────────────────────────
# REINSURANCE EXHAUSTION ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
def reinsurance_analysis(scenario_results: dict, cat_xs_attachment: float,
                          cat_xs_limit: float) -> dict:
    """
    Determines whether a Cat XL reinsurance treaty absorbs the scenario PML.
    Standard cyber Cat XL: $5M xs $5M (attach $5M, limit up to $50M).
    """
    pml_99     = scenario_results["pml_99_usd"]
    pml_90     = scenario_results["pml_90_usd"]
    # Insurers net of reinsurance (they keep the bottom, RI pays the top)
    ri_recovery_99 = min(max(0, pml_99 - cat_xs_attachment), cat_xs_limit)
    ri_recovery_90 = min(max(0, pml_90 - cat_xs_attachment), cat_xs_limit)
    exhausts_99    = pml_99 > (cat_xs_attachment + cat_xs_limit)
    exhausts_90    = pml_90 > (cat_xs_attachment + cat_xs_limit)

    return {
        "cat_xs_attachment_usd": cat_xs_attachment,
        "cat_xs_limit_usd":      cat_xs_limit,
        "ri_recovery_at_p99":    round(ri_recovery_99, 0),
        "ri_recovery_at_p90":    round(ri_recovery_90, 0),
        "treaty_exhausted_p99":  exhausts_99,
        "treaty_exhausted_p90":  exhausts_90,
        "net_insurer_pml_99":    round(pml_99 - ri_recovery_99, 0),
    }

# ─────────────────────────────────────────────────────────────────────────────
# RUN ALL SCENARIOS
# ─────────────────────────────────────────────────────────────────────────────
CAT_XS_ATTACHMENT = 5_000_000   # $5M attachment
CAT_XS_LIMIT      = 45_000_000  # $45M limit (total tower $50M)

scenarios_output = {}

scenario_runners = [
    ("cloud_outage",          run_cloud_outage_scenario),
    ("ransomware_campaign",   run_ransomware_campaign_scenario),
    ("supply_chain",          run_supply_chain_scenario),
    ("critical_infra",        run_critical_infra_scenario),
]

for key, runner in scenario_runners:
    print(f"\n[Running scenario: {key}]")
    result = runner(df, N_SIMS)
    result["reinsurance"] = reinsurance_analysis(result, CAT_XS_ATTACHMENT, CAT_XS_LIMIT)
    scenarios_output[key] = result

    print(f"  Affected policies : {result.get('n_affected_policies', 'all')} / {n_policies}")
    print(f"  Expected loss     : ${result['expected_loss_usd']:>15,.0f}")
    print(f"  PML 90%           : ${result['pml_90_usd']:>15,.0f}")
    print(f"  PML 99%           : ${result['pml_99_usd']:>15,.0f}")
    print(f"  RI Treaty exhaust?: {result['reinsurance']['treaty_exhausted_p99']} (at 99th pct)")

# ─────────────────────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────────────────────
out = {
    "portfolio_policies":     n_policies,
    "total_tiv_usd":          round(float(total_tiv), 0),
    "num_simulations":        N_SIMS,
    "cat_xs_attachment_usd":  CAT_XS_ATTACHMENT,
    "cat_xs_limit_usd":       CAT_XS_LIMIT,
    "scenarios":              scenarios_output,
}

out_path = MODEL_DIR / "scenario_results.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)

print(f"\n  Scenario results saved to {out_path}")
print("\n" + "=" * 65)
print("CATASTROPHE SCENARIO ENGINE COMPLETE")
print("=" * 65)
