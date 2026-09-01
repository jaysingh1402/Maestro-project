"""
05_hawkes_process_simulation.py — Enhanced Real-World Contagion Risk Engine
============================================================================

Enhancements over the baseline version:
  1. Reproducible seeding (np.random.seed(42))
  2. Generalized Pareto Distribution (GPD) severity via Peaks-Over-Threshold (POT)
     — the actuarial standard for heavy/fat-tailed risks
  3. Annual frequency trend projection (+14%/yr, calibrated to DBIR 2024 / IBM X-Force)
  4. Sector-specific Hawkes models by cause-of-loss type (Ransomware, Data Breach, BEC, etc.)
  5. Aggregate Exceedance Probability (AEP/OEP) curve for reinsurance pricing
  6. 95% confidence intervals on TVaR via bootstrap resampling (500 samples)
  7. Enriched JSON output consumed by the Streamlit dashboard Tab 4

Run from the project root:
    python code/05_hawkes_process_simulation.py

Outputs:
    outputs/model_outputs/hawkes_results.json
    outputs/model_outputs/oep_curve.csv
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy import stats
from scipy.stats import genpareto
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# ── Reproducibility ────────────────────────────────────────────────────────────
np.random.seed(42)

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "outputs" / "model_outputs"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ── Calibration constants (DBIR 2024, IBM X-Force 2024) ───────────────────────
ANNUAL_FREQ_TREND   = 0.14   # +14% per year compound frequency growth
PROJECTION_YEARS    = 3      # Project 3 years beyond last observed claim
GPD_THRESHOLD_PCT   = 0.75   # POT threshold at 75th percentile of losses
NUM_SIMS            = 50_000
N_BOOTSTRAP         = 500    # Bootstrap samples for TVaR confidence interval

print("=" * 60)
print("HAWKES PROCESS OPTIMIZATION & SIMULATION (Enhanced)")
print("=" * 60)

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
try:
    df_claims = pd.read_csv(DATA_DIR / "02_claims.csv")
    df_claims["loss_date"] = pd.to_datetime(df_claims["loss_date"])
    df_claims = df_claims.sort_values("loss_date").reset_index(drop=True)
    print(f"Loaded {len(df_claims)} claims spanning "
          f"{df_claims['loss_date'].min().date()} to {df_claims['loss_date'].max().date()}")
except Exception as e:
    print(f"Error loading claims data: {e}")
    raise SystemExit(1)

# ─────────────────────────────────────────────────────────────────────────────
# 2. EVENT TIMES (days relative to first event)
# ─────────────────────────────────────────────────────────────────────────────
t_events = (df_claims["loss_date"] - df_claims["loss_date"].min()).dt.days.values.astype(float)
T_max      = float(t_events[-1] + 1)
obs_years  = T_max / 365.25

# Frequency trend: project from end of observation to current + PROJECTION_YEARS
current_year   = 2025
last_data_year = df_claims["loss_date"].max().year
years_to_project = (current_year - last_data_year) + PROJECTION_YEARS
trend_factor     = (1 + ANNUAL_FREQ_TREND) ** years_to_project

# ─────────────────────────────────────────────────────────────────────────────
# 3. SEVERITY DISTRIBUTIONS
# ─────────────────────────────────────────────────────────────────────────────
sev_data = df_claims["gross_incurred_usd"].dropna().values
sev_data = sev_data[sev_data > 0]

# 3a. Gamma (regulatory baseline — kept for backward compatibility)
shape_g, loc_g, scale_g = stats.gamma.fit(sev_data, floc=0)

# 3b. Generalised Pareto Distribution via Peaks-Over-Threshold (POT)
u_threshold         = float(np.percentile(sev_data, GPD_THRESHOLD_PCT * 100))
exceedances         = sev_data[sev_data > u_threshold] - u_threshold
prob_exceed         = float(len(exceedances) / len(sev_data))  # P(X > u)
below_threshold_arr = sev_data[sev_data <= u_threshold]

xi_gpd, loc_gpd, sigma_gpd = genpareto.fit(exceedances, floc=0)

print(f"\n[Severity Models]")
print(f"  Gamma:  shape={shape_g:.4f}, scale={scale_g:,.0f}")
print(f"  GPD POT threshold u = ${u_threshold:,.0f} (75th pct), xi={xi_gpd:.4f}, sigma={sigma_gpd:,.0f}")
print(f"  P(Loss > u) = {prob_exceed:.3f}  ({len(exceedances)} exceedances out of {len(sev_data)})")

# Pre-generate large severity pools for fast simulation
_POOL  = 1_000_000
_gpd_pool   = u_threshold + genpareto.rvs(xi_gpd, scale=sigma_gpd, size=_POOL)
_gpd_idx    = 0

def _refill_gpd_pool():
    global _gpd_pool, _gpd_idx
    _gpd_pool = u_threshold + genpareto.rvs(xi_gpd, scale=sigma_gpd, size=_POOL)
    _gpd_idx  = 0

def sample_severity_gpd(n: int) -> float:
    """Draw n losses from the POT mixed model (below-threshold empirical + GPD above)."""
    global _gpd_idx
    if n <= 0:
        return 0.0
    n_above = int(np.random.binomial(n, prob_exceed))
    n_below = n - n_above
    total   = 0.0
    if n_above > 0:
        end = _gpd_idx + n_above
        if end > _POOL:
            _refill_gpd_pool()
            end = _gpd_idx + n_above
        total += float(np.sum(_gpd_pool[_gpd_idx:end]))
        _gpd_idx = end
    if n_below > 0 and len(below_threshold_arr) > 0:
        total += float(np.sum(np.random.choice(below_threshold_arr, size=n_below, replace=True)))
    return total

def sample_severity_gamma(n: int) -> float:
    if n <= 0:
        return 0.0
    return float(np.sum(np.random.gamma(shape_g, scale_g, n)))

# ─────────────────────────────────────────────────────────────────────────────
# 4. HAWKES NEGATIVE LOG-LIKELIHOOD
# ─────────────────────────────────────────────────────────────────────────────
def hawkes_nll(params, t, T):
    mu, alpha, beta = params
    if mu <= 0 or alpha <= 0 or beta <= 0 or alpha >= beta:
        return np.inf
    n              = len(t)
    integral_term  = mu * T + (alpha / beta) * np.sum(1.0 - np.exp(-beta * (T - t)))
    log_lam_sum    = 0.0
    R              = 0.0
    for i in range(n):
        if i > 0:
            R = np.exp(-beta * (t[i] - t[i - 1])) * (1.0 + R)
        lam_i = mu + alpha * R
        if lam_i <= 0:
            return np.inf
        log_lam_sum += np.log(lam_i)
    return integral_term - log_lam_sum

# ─────────────────────────────────────────────────────────────────────────────
# 5. FIT GLOBAL HAWKES MODEL
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Global Hawkes MLE]")
init_params = [len(t_events) / T_max, 0.05, 0.1]
bounds      = ((1e-4, 5.0), (1e-4, 0.995), (1e-3, 5.0))
res = minimize(hawkes_nll, init_params, args=(t_events, T_max),
               method="L-BFGS-B", bounds=bounds,
               options={"maxiter": 500, "ftol": 1e-10})

mu_opt, alpha_opt, beta_opt = res.x
branching_ratio = alpha_opt / beta_opt
poisson_rate    = mu_opt / (1.0 - branching_ratio)
expected_yr     = 365.0 * poisson_rate

print(f"  Optimization successful: {res.success}")
print(f"  Baseline (mu):      {mu_opt:.4f} events/day")
print(f"  Excitation (alpha): {alpha_opt:.4f}")
print(f"  Decay (beta):       {beta_opt:.4f}")
print(f"  Branching ratio:    {branching_ratio:.4f}  (must be < 1 for stationarity)")
print(f"  Expected events/yr: {expected_yr:.2f}")

# Pre-compute NegBin branching parameters (cluster size distribution)
_mean_cluster = 1.0 / (1.0 - branching_ratio)
_var_cluster  = branching_ratio / (1.0 - branching_ratio) ** 3
_nb_p         = max(min(_mean_cluster / _var_cluster, 0.999), 0.001)
_nb_r         = max(_mean_cluster ** 2 / max(_var_cluster - _mean_cluster, 1e-9), 0.05)

def simulate_hawkes_events(mu_rate: float, br: float) -> int:
    """Sample annual Hawkes event count via branching approximation."""
    n_immigrants = np.random.poisson(365.0 * mu_rate)
    if n_immigrants == 0:
        return 0
    mc    = 1.0 / (1.0 - br)
    vc    = br / (1.0 - br) ** 3
    nb_p_ = max(min(mc / vc, 0.999), 0.001)
    nb_r_ = max(mc ** 2 / max(vc - mc, 1e-9), 0.05)
    try:
        offspring = np.random.negative_binomial(nb_r_, nb_p_, n_immigrants)
        return int(np.sum(offspring + 1))
    except Exception:
        return n_immigrants

# ─────────────────────────────────────────────────────────────────────────────
# 6. SECTOR-SPECIFIC HAWKES MODELS (by cause_of_loss)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Sector-Specific Hawkes Models by Cause of Loss]")
sector_models = {}

for cause in df_claims["cause_of_loss"].dropna().unique():
    sub   = df_claims[df_claims["cause_of_loss"] == cause].copy()
    if len(sub) < 5:
        print(f"  {cause}: too few events ({len(sub)}), skipping")
        continue
    t_sub = (sub["loss_date"] - df_claims["loss_date"].min()).dt.days.values.astype(float)
    T_sub = float(t_sub[-1] + 1)

    res_s = minimize(hawkes_nll, init_params, args=(t_sub, T_sub),
                     method="L-BFGS-B", bounds=bounds,
                     options={"maxiter": 300, "ftol": 1e-9})
    if res_s.success:
        mu_s, alpha_s, beta_s = res_s.x
        br_s = alpha_s / beta_s
        sector_models[cause] = {
            "mu":             round(float(mu_s),    6),
            "alpha":          round(float(alpha_s), 6),
            "beta":           round(float(beta_s),  6),
            "branching_ratio":round(float(br_s),    4),
            "n_events":       int(len(sub)),
            "expected_yr":    round(float(365.0 * mu_s / max(1.0 - br_s, 1e-6)), 2),
        }
        print(f"  {cause:35s} | n={len(sub):3d} | mu={mu_s:.4f} | alpha={alpha_s:.4f} "
              f"| beta={beta_s:.4f} | br={br_s:.4f}")
    else:
        print(f"  {cause}: optimisation did not converge")

# ─────────────────────────────────────────────────────────────────────────────
# 7. 50,000-YEAR STOCHASTIC SIMULATION
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[Running {NUM_SIMS:,}-year Monte Carlo simulation...]")

# Trended parameters
trended_mu          = mu_opt * trend_factor
trended_poisson_rate= trended_mu / (1.0 - branching_ratio)
expected_yr_trended = 365.0 * trended_poisson_rate

annual_losses_poisson     = np.zeros(NUM_SIMS)
annual_losses_hawkes_gamma= np.zeros(NUM_SIMS)
annual_losses_hawkes_gpd  = np.zeros(NUM_SIMS)
annual_losses_trended     = np.zeros(NUM_SIMS)

for i in range(NUM_SIMS):
    # A) Poisson + Gamma (independence baseline)
    n_p = int(np.random.poisson(expected_yr))
    annual_losses_poisson[i]      = sample_severity_gamma(n_p)

    # B) Hawkes + Gamma (contagion, Gamma tail)
    n_hg = simulate_hawkes_events(mu_opt, branching_ratio)
    annual_losses_hawkes_gamma[i] = sample_severity_gamma(n_hg)

    # C) Hawkes + GPD (contagion, realistic fat tail)
    n_hg2 = simulate_hawkes_events(mu_opt, branching_ratio)
    annual_losses_hawkes_gpd[i]   = sample_severity_gpd(n_hg2)

    # D) Trended Hawkes + GPD (3-year projected frequency)
    n_t  = simulate_hawkes_events(trended_mu, branching_ratio)
    annual_losses_trended[i]      = sample_severity_gpd(n_t)

print("  Simulation complete.")

# ─────────────────────────────────────────────────────────────────────────────
# 8. RISK METRICS
# ─────────────────────────────────────────────────────────────────────────────
def compute_tvar(losses: np.ndarray, pct: float = 99.0) -> float:
    threshold = np.percentile(losses, pct)
    tail      = losses[losses >= threshold]
    return float(np.mean(tail)) if len(tail) > 0 else float(threshold)

def compute_var(losses: np.ndarray, pct: float) -> float:
    return float(np.percentile(losses, pct))

# Bootstrap TVaR confidence interval
print(f"  Computing {N_BOOTSTRAP}-sample bootstrap TVaR confidence intervals...")
tvar_gpd_boot = np.array([
    compute_tvar(
        np.random.choice(annual_losses_hawkes_gpd, len(annual_losses_hawkes_gpd), replace=True)
    )
    for _ in range(N_BOOTSTRAP)
])
ci_lo = float(np.percentile(tvar_gpd_boot, 2.5))
ci_hi = float(np.percentile(tvar_gpd_boot, 97.5))

# All TVaR values
tvar_poisson      = compute_tvar(annual_losses_poisson)
tvar_hawkes_gamma = compute_tvar(annual_losses_hawkes_gamma)
tvar_hawkes_gpd   = compute_tvar(annual_losses_hawkes_gpd)
tvar_trended      = compute_tvar(annual_losses_trended)

print(f"\n{'─'*55}")
print(f"  [Poisson + Gamma]    TVaR 99% : ${tvar_poisson:>15,.0f}")
print(f"  [Hawkes + Gamma]     TVaR 99% : ${tvar_hawkes_gamma:>15,.0f}")
print(f"  [Hawkes + GPD]       TVaR 99% : ${tvar_hawkes_gpd:>15,.0f}  ← primary")
print(f"    95% CI: [${ci_lo:,.0f} – ${ci_hi:,.0f}]")
print(f"  [Trended (+{ANNUAL_FREQ_TREND*100:.0f}%/yr x{PROJECTION_YEARS}yr)] TVaR 99% : ${tvar_trended:>12,.0f}")
print(f"{'─'*55}")
print(f"  Contagion premium (Hawkes GPD - Poisson): +${tvar_hawkes_gpd - tvar_poisson:,.0f}")
print(f"  Fat-tail premium (GPD - Gamma):           +${tvar_hawkes_gpd - tvar_hawkes_gamma:,.0f}")
print(f"  Trend premium (+{PROJECTION_YEARS}yr projection):      +${tvar_trended - tvar_hawkes_gpd:,.0f}")

# ─────────────────────────────────────────────────────────────────────────────
# 9. AGGREGATE EXCEEDANCE PROBABILITY (AEP / OEP) CURVE
# ─────────────────────────────────────────────────────────────────────────────
sorted_losses  = np.sort(annual_losses_hawkes_gpd)[::-1]
n_sorted       = len(sorted_losses)
exceed_probs   = np.arange(1, n_sorted + 1) / n_sorted

# Downsample to 250 representative points for JSON / CSV storage
idx_sample     = np.unique(np.round(np.linspace(0, n_sorted - 1, 250)).astype(int))
oep_losses     = [round(float(sorted_losses[i]), 2) for i in idx_sample]
oep_probs      = [round(float(exceed_probs[i]),  6) for i in idx_sample]

oep_df = pd.DataFrame({"loss_usd": oep_losses, "exceedance_prob": oep_probs})
oep_df.to_csv(MODEL_DIR / "oep_curve.csv", index=False)
print(f"\n  AEP/OEP curve saved to outputs/model_outputs/oep_curve.csv")

# ─────────────────────────────────────────────────────────────────────────────
# 10. SAVE RESULTS
# ─────────────────────────────────────────────────────────────────────────────
hawkes_results = {
    # ── Core MLE parameters ───────────────────────────────────────────────
    "mu":                    round(float(mu_opt),     6),
    "alpha":                 round(float(alpha_opt),  6),
    "beta":                  round(float(beta_opt),   6),
    "branching_ratio":       round(float(branching_ratio), 4),
    "expected_events_yr":    round(float(expected_yr), 2),
    "optimization_success":  bool(res.success),

    # ── Severity models ───────────────────────────────────────────────────
    "gamma_params": {
        "shape": round(float(shape_g),  4),
        "scale": round(float(scale_g),  2),
    },
    "gpd_params": {
        "xi":                    round(float(xi_gpd),    6),
        "sigma":                 round(float(sigma_gpd), 2),
        "threshold_u":           round(float(u_threshold), 2),
        "prob_exceed_threshold": round(float(prob_exceed),  4),
        "tail_type": "heavy" if xi_gpd > 0 else ("exponential" if abs(xi_gpd) < 0.01 else "bounded"),
    },

    # ── TVaR comparison ───────────────────────────────────────────────────
    "tvar_poisson":       round(tvar_poisson,      2),
    "tvar_hawkes":        round(tvar_hawkes_gamma,  2),   # backward-compat
    "tvar_hawkes_gamma":  round(tvar_hawkes_gamma,  2),
    "tvar_hawkes_gpd":    round(tvar_hawkes_gpd,    2),
    "tvar_trended":       round(tvar_trended,       2),
    "contagion_premium":  round(tvar_hawkes_gpd - tvar_poisson,      2),
    "fat_tail_premium":   round(tvar_hawkes_gpd - tvar_hawkes_gamma, 2),
    "trend_premium":      round(tvar_trended     - tvar_hawkes_gpd,  2),

    # ── Bootstrap confidence intervals ────────────────────────────────────
    "tvar_gpd_ci_95_low":  round(ci_lo, 2),
    "tvar_gpd_ci_95_high": round(ci_hi, 2),

    # ── Frequency trending ────────────────────────────────────────────────
    "annual_freq_trend_pct":  ANNUAL_FREQ_TREND,
    "projection_years":       PROJECTION_YEARS,
    "trend_factor":           round(trend_factor, 4),
    "trended_mu":             round(float(trended_mu), 6),
    "expected_yr_trended":    round(float(expected_yr_trended), 2),

    # ── VaR exceedance table ──────────────────────────────────────────────
    "var_table": {
        "90_poisson":    round(compute_var(annual_losses_poisson,     90), 2),
        "95_poisson":    round(compute_var(annual_losses_poisson,     95), 2),
        "99_poisson":    round(compute_var(annual_losses_poisson,     99), 2),
        "90_hawkes_gpd": round(compute_var(annual_losses_hawkes_gpd,  90), 2),
        "95_hawkes_gpd": round(compute_var(annual_losses_hawkes_gpd,  95), 2),
        "99_hawkes_gpd": round(compute_var(annual_losses_hawkes_gpd,  99), 2),
        "90_trended":    round(compute_var(annual_losses_trended,      90), 2),
        "99_trended":    round(compute_var(annual_losses_trended,      99), 2),
    },

    # ── OEP curve (250-point downsampled) ─────────────────────────────────
    "oep_curve": {
        "loss_levels":       oep_losses,
        "exceedance_probs":  oep_probs,
    },

    # ── Sector-specific models ────────────────────────────────────────────
    "sector_models": sector_models,

    # ── Simulation metadata ───────────────────────────────────────────────
    "num_simulations": NUM_SIMS,
    "n_bootstrap":     N_BOOTSTRAP,
    "random_seed":     42,
    "gpd_threshold_percentile": GPD_THRESHOLD_PCT,
}

out_path = MODEL_DIR / "hawkes_results.json"
with open(out_path, "w") as f:
    json.dump(hawkes_results, f, indent=2)

print(f"\n  Results saved to {out_path}")
print("\n" + "=" * 60)
print("HAWKES SIMULATION COMPLETE")
print("=" * 60)
