"""
Use Case 2: Ransomware BI Pricing — Advanced Monte Carlo Engine
==================================================================

This engine models BI (Business Interruption) losses using:
  1. Per-system bimodal downtime distributions (GaussianMixture)
  2. Dynamic revenue dependencies per system class
  3. Correlated system outages (ransomware hits multiple systems)
  4. Fully vectorized NumPy Monte Carlo loops for maximum efficiency
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.mixture import GaussianMixture

DATA_DIR = Path(__file__).parent.parent / "data"

# ============================================================
# Load Data
# ============================================================
print("="*60)
print("LOADING DATA & FITTING MODELS")
print("="*60)

# We use 07_modeling_dataset as a proxy for policies since 01_policies is missing
try:
    policies = pd.read_csv(DATA_DIR / "01_policies.csv")
except FileNotFoundError:
    print("01_policies.csv not found, using 07_modeling_dataset.csv instead.")
    policies = pd.read_csv(DATA_DIR / "07_modeling_dataset.csv")
    
outages = pd.read_csv(DATA_DIR / "06_outage_events.csv")
try:
    systems = pd.read_csv(DATA_DIR / "05_system_recovery_profiles.csv")
except FileNotFoundError:
    print("05_system_recovery_profiles.csv not found, generating mock systems data.")
    systems = pd.DataFrame({
        "system_class": ['Customer Portal', 'Trading Platform', 'Email/Productivity', 'Core Banking', 'Multiple Systems', 'Payment Processing'],
        "revenue_dependency_pct": [0.15, 0.85, 0.05, 0.60, 1.0, 0.50],
        "severity_weight": [1.0, 5.0, 0.5, 4.0, 8.0, 3.0]
    })

# ============================================================
# Step 1: Fit Bimodal Gaussian Mixture Models
# ============================================================
downtime_models = {}
print("\nFitting 2-Component Gaussian Mixtures to Downtime Data...")

for sys_class, grp in outages.groupby("systems_affected"):
    # Ensure no zero/negative downtime for log
    downtime = grp["total_downtime_hours"].clip(lower=1).values.reshape(-1, 1)
    log_dt = np.log(downtime)
    
    # Fit 2 components (e.g. quick recovery vs full rebuild)
    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm.fit(log_dt)
    
    downtime_models[sys_class] = gmm
    print(f"  {sys_class:25s} | Means (log-hours): {gmm.means_.flatten().round(2)}")

# ============================================================
# Step 2: Time Multipliers & System Dependencies
# ============================================================
TIME_MULTIPLIERS = {
    "regular": 1.00,
    "month_end": 1.40,
    "quarter_end": 1.85,
    "weekend": 0.55,
}

# Pre-compute revenue dependencies into a dict
rev_deps = dict(zip(systems["system_class"], systems["revenue_dependency_pct"]))

# We define system severity weights to sample correlated primary systems
sys_weights = systems.set_index("system_class")["severity_weight"]
sys_probs = sys_weights / sys_weights.sum()
system_classes = sys_probs.index.values
system_probs = sys_probs.values

# ============================================================
# Step 3: Vectorized Monte Carlo Engine
# ============================================================
def simulate_bi_loss_vectorized(policy_row, n_sims=10_000, seed=None):
    """Fully vectorized Monte Carlo simulation of annual BI loss."""
    if seed is not None:
        np.random.seed(seed)

    rev_daily = (policy_row["revenue_mm"] * 1_000_000) / 252

    # Annual frequency of a material ransomware event
    annual_freq = 0.10 * (5 - policy_row["control_maturity_nist"]) / 2.5
    
    # 1. Simulate Number of Events Per Year (Array of size n_sims)
    n_events_per_year = np.random.poisson(annual_freq, n_sims)
    total_events = n_events_per_year.sum()
    
    if total_events == 0:
        return np.zeros(n_sims)

    # Array mapping each event to its corresponding year index
    event_year_indices = np.repeat(np.arange(n_sims), n_events_per_year)

    # 2. Correlated System Impacts
    # Assume a ransomware event affects 1 to 3 systems simultaneously
    n_systems_hit = np.random.randint(1, 4, size=total_events)
    total_system_impacts = n_systems_hit.sum()
    
    # Map each system impact to its parent event
    impact_event_indices = np.repeat(np.arange(total_events), n_systems_hit)
    
    # Sample which systems are hit for all impacts
    hit_systems = np.random.choice(system_classes, size=total_system_impacts, p=system_probs)

    # 3. Sample Downtime & Dependencies
    downtimes = np.zeros(total_system_impacts)
    impact_rev_deps = np.zeros(total_system_impacts)
    
    for sys_class in np.unique(hit_systems):
        mask = (hit_systems == sys_class)
        count = mask.sum()
        if count > 0:
            gmm = downtime_models[sys_class]
            log_dt, _ = gmm.sample(count)
            downtimes[mask] = np.exp(log_dt.flatten())
            impact_rev_deps[mask] = rev_deps.get(sys_class, 0.35)

    # 4. Calculate Lost Revenue and Expenses per Impact
    r_time = np.random.random(total_events)
    event_time_mults = np.where(r_time < 0.04, TIME_MULTIPLIERS["quarter_end"],
                       np.where(r_time < 0.18, TIME_MULTIPLIERS["month_end"],
                       np.where(r_time < 0.46, TIME_MULTIPLIERS["weekend"], 
                                TIME_MULTIPLIERS["regular"])))
    
    # Map event time mults to individual system impacts
    impact_time_mults = event_time_mults[impact_event_indices]

    lost_rev = (downtimes / 24) * rev_daily * impact_rev_deps * impact_time_mults
    extra_expense = lost_rev * 0.35
    forensics = np.minimum(2_500_000, lost_rev * 0.12)
    restoration = downtimes * 4_000

    impact_losses = lost_rev + extra_expense + forensics + restoration

    # 5. Aggregate Impacts Back to Events
    event_losses = np.bincount(impact_event_indices, weights=impact_losses, minlength=total_events)
    
    # Calculate Regulatory Overlay at the Event Level (based on max downtime per event)
    # Get max downtime per event
    max_downtime_per_event = np.zeros(total_events)
    np.maximum.at(max_downtime_per_event, impact_event_indices, downtimes)
    
    regulator = policy_row["primary_regulator"]
    revenue_mm = policy_row["revenue_mm"]
    reg_cost = np.zeros(total_events)
    
    if regulator in ("OCC", "FRB", "FDIC"):
        reg_cost[max_downtime_per_event > 36] += 250_000
    reg_cost[max_downtime_per_event > 168] += 1_500_000 + (revenue_mm * 200)
    reg_cost[max_downtime_per_event > 336] += 5_000_000
    
    final_event_losses = event_losses + reg_cost

    # 6. Aggregate Events Back to Years
    annual_losses = np.bincount(event_year_indices, weights=final_event_losses, minlength=n_sims)
    
    return annual_losses

# ============================================================
# Step 4: Actuarial Premium Calculation (Cost of Capital)
# ============================================================
def calculate_premium(sim_losses, capital_charge=0.10, expense_ratio=0.25):
    """Calculates risk load dynamically using TVaR 99% Cost of Capital."""
    expected_loss = sim_losses.mean()
    p95_loss = np.percentile(sim_losses, 95)
    p99_loss = np.percentile(sim_losses, 99)
    
    # Tail Value at Risk (average of losses in the worst 1%)
    tail_losses = sim_losses[sim_losses >= p99_loss]
    tvar_99 = tail_losses.mean() if len(tail_losses) > 0 else p99_loss
    
    # Cost of Capital Risk Load
    required_capital = tvar_99 - expected_loss
    risk_load_dollars = max(0, required_capital * capital_charge)
    
    technical_premium = (expected_loss + risk_load_dollars) / (1 - expense_ratio)
    
    return {
        "expected_loss": expected_loss,
        "p95_loss": p95_loss,
        "p99_loss": p99_loss,
        "tvar_99": tvar_99,
        "risk_load_dollars": risk_load_dollars,
        "technical_premium": technical_premium
    }

# ============================================================
# Step 5: Apply to a sample of policies and price
# ============================================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("STEP 5: PRICING SAMPLE INSUREDS (VECTORIZED + CoC)")
    print("="*60)
    
    # Sample a few representative policies — one per sub-sector
    if "sub_sector" in policies.columns:
        sample_pols = (
            policies.sort_values("policy_id")
            .groupby("sub_sector", group_keys=False, as_index=False)
            .head(1)
            .head(6)
        )
    else:
        sample_pols = policies.head(6)

    results = []
    for _, pol in sample_pols.iterrows():
        # We can run 50,000 simulations effortlessly thanks to numpy vectorization
        sim_losses = simulate_bi_loss_vectorized(pol, n_sims=50_000, seed=42)
        
        metrics = calculate_premium(sim_losses, capital_charge=0.10, expense_ratio=0.25)
        
        results.append({
            "policy_id": pol["policy_id"],
            "sub_sector": pol.get("sub_sector", "N/A"),
            "revenue_mm": pol["revenue_mm"],
            "control_maturity": pol["control_maturity_nist"],
            "expected_bi_loss": int(metrics["expected_loss"]),
            "p95_bi_loss": int(metrics["p95_loss"]),
            "p99_bi_loss": int(metrics["p99_loss"]),
            "tvar_99_loss": int(metrics["tvar_99"]),
            "risk_load_usd": int(metrics["risk_load_dollars"]),
            "bi_technical_premium": int(metrics["technical_premium"]),
            "charged_premium": pol.get("premium_usd", 0),
        })

    results_df = pd.DataFrame(results)
    pd.options.display.float_format = '{:,.0f}'.format
    print("\nBI Pricing Output (50,000 Simulations/Policy):")
    print(results_df.to_string(index=False))

    results_df.to_csv(DATA_DIR / "08_bi_pricing_output_sample.csv", index=False)
    print(f"\nWrote {len(results_df)} sample outputs.")
