# Maestro Cyber Pricing Engine

> An end-to-end actuarial and data science pipeline for dynamically pricing cyber insurance policies — combining GLMs, XGBoost, DistilBERT NLP, Hawkes Process contagion simulation, and a Google Gemini AI agent into a single interactive Streamlit dashboard.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Real-World Use Case](#2-real-world-use-case)
3. [Complete Pipeline Architecture](#3-complete-pipeline-architecture)
4. [Directory Structure](#4-directory-structure)
5. [Script Reference](#5-script-reference)
6. [Deep Dive: 05\_hawkes\_process\_simulation.py](#6-deep-dive-05_hawkes_process_simulationpy)
7. [Technologies and Dependencies](#7-technologies-and-dependencies)
8. [Installation and Environment Setup](#8-installation-and-environment-setup)
9. [Running the Pipeline](#9-running-the-pipeline)
10. [Running the Streamlit Dashboard](#10-running-the-streamlit-dashboard)
11. [Expected Outputs](#11-expected-outputs)
12. [Troubleshooting](#12-troubleshooting)
13. [Limitations and Known Issues](#13-limitations-and-known-issues)
14. [Data Dictionary Reference](#14-data-dictionary-reference)

---

## 1. Project Overview

The **Maestro Cyber Pricing Engine** is a research-grade actuarial platform that demonstrates how modern insurance companies can price cyber risk with far greater precision than legacy methods allow.

The project answers a fundamental question in insurance: *how do you price a risk that is both unpredictable and contagious?* Unlike a car accident (independent), a ransomware attack on a major cloud vendor can simultaneously trigger claims across hundreds of policyholders. Standard models built on the assumption of independence underestimate this tail risk significantly.

This engine tackles that problem by combining:

| Layer | Technology | Purpose |
|---|---|---|
| **Feature Engineering** | NumPy / pandas | Transform raw policy data into actuarially meaningful predictors |
| **NLP Severity Scoring** | DistilBERT + Random Forest | Convert unstructured regulatory text into quantitative risk scores |
| **Frequency Model** | Poisson GLM (scikit-learn) | Predict the probability a policyholder will file a claim |
| **Severity Model** | Gamma GLM (scikit-learn) | Predict the expected dollar loss if a claim occurs |
| **ML Benchmark** | XGBoost + SHAP | Non-linear benchmark and explainability layer |
| **Contagion Simulation** | Hawkes Process (MLE + Monte Carlo) | Quantify the systemic/contagion risk premium not captured by GLMs |
| **BI Monte Carlo** | Vectorized NumPy Monte Carlo | Simulate Business Interruption losses with correlated system failures |
| **AI Explainability** | Google Gemini ADK (RAG + Function Calling) | Translate complex math into plain-English pricing explanations |
| **Dashboard** | Streamlit + Plotly | Interactive interface for underwriters |

---

## 2. Real-World Use Case

A cyber insurance underwriter at a financial services firm receives thousands of renewal applications. Each application contains:

- **Structured fields:** revenue, employee count, NIST maturity score, MFA coverage, prior incidents.
- **Unstructured text:** regulatory examination findings such as *"The institution's patch management process lacks documented controls..."*
- **Historical claims:** timestamps, loss amounts, affected systems, downtime hours.

**The problem with legacy pricing:**

1. Underwriters manually score text findings — inconsistent and slow.
2. Standard Poisson models assume each claim event is independent — a dangerous assumption in cyber where one supply-chain attack triggers a cascade across shared vendor ecosystems.
3. GLMs produce a "mean" expected loss but cannot quantify how bad the worst 1% of years could actually be.

**What this engine provides:**

1. DistilBERT automatically scores every regulatory finding — consistent, fast, and auditable.
2. The Hawkes Process models **self-exciting** claim arrival: a breach at one firm increases the short-term probability of a breach at similar firms sharing the same vendor ecosystem.
3. TVaR (Tail Value at Risk) at the 99th percentile quantifies how much extra capital the portfolio must hold against systemic catastrophe — expressed as a concrete **Contagion Risk Premium** in dollars.

---

## 3. Complete Pipeline Architecture

```
Raw Data (CSV files)
        |
        v
+--------------------------------------------------------------+
|  STEP 0: NLP Severity Baseline                               |
|  03_nlp_severity_baseline.py                                 |
|  - Load 03_regulatory_findings.csv                           |
|  - Extract DistilBERT [CLS] embeddings (768-dim)             |
|  - Train Random Forest classifier on embeddings              |
|  Output: severity_classifier_baseline.joblib                 |
+------------------------+-------------------------------------+
                         |
                         v
+--------------------------------------------------------------+
|  STEP 1: EDA, Feature Engineering & GLM Pricing              |
|  01_use_case_1_eda.py                                        |
|  - Load all 6 source datasets                                |
|  - Run data quality checks                                   |
|  - Apply NLP model for high_sev_prob inference               |
|  - Engineer 17 numeric + 5 categorical pricing features      |
|  - Generate interactive EDA dashboard (Plotly HTML)          |
|  - Train Poisson GLM (frequency) + Gamma GLM (severity)      |
|  - Compute Pure Premium & Technical Premium per policy       |
|  - Run per-policy Monte Carlo for VaR 95/99 metrics          |
|  - Export models (.joblib) for Streamlit inference           |
|  Output: pure_premium_indications_with_mc.csv                |
|          glm_coefficients.csv, model_diagnostics.txt         |
+------------------------+-------------------------------------+
                         |
                         v
+--------------------------------------------------------------+
|  STEP 2: BI Monte Carlo Simulation (Use Case 2)              |
|  02_use_case_2_bi_simulation.py                              |
|  - Fit bimodal Gaussian Mixture Models per system class      |
|  - Vectorized simulation of ransomware BI scenarios          |
|  - Apply regulatory cost overlays (OCC, FRB, FDIC rules)    |
|  - Calculate TVaR-based Cost of Capital premium              |
|  Output: 08_bi_pricing_output_sample.csv                     |
+------------------------+-------------------------------------+
                         |
                         v
+--------------------------------------------------------------+
|  STEP 3: Hawkes Process Contagion Simulation           *     |
|  05_hawkes_process_simulation.py                             |
|  - Load claim timestamps from 02_claims.csv                  |
|  - Fit Hawkes parameters (mu, alpha, beta) via MLE           |
|  - Run 50,000-year stochastic contagion simulation           |
|  - Fit Gamma severity distribution to historical losses      |
|  - Compare Poisson vs. Hawkes TVaR at 99th percentile        |
|  - Calculate Contagion Risk Premium (dollar delta)           |
|  Output: outputs/model_outputs/hawkes_results.json           |
+------------------------+-------------------------------------+
                         |
                         v
+--------------------------------------------------------------+
|  STEP 4: Streamlit Dashboard                                 |
|  app.py                                                      |
|  - Tab 1: Portfolio Analytics & AI Explainer                 |
|    (Plotly charts, GLM coefficients, Gemini RAG chatbot)     |
|  - Tab 2: Feature Derivation Explainer                       |
|    (Mathematical breakdown of every engineered feature)      |
|  - Tab 3: Interactive Pricing Engine                         |
|    (Real-time premium recalculation via sliders)             |
|  - Tab 4: Advanced Contagion (Hawkes)                        |
|    (Live Hawkes simulation visualization)                    |
+--------------------------------------------------------------+
```

### The Pricing Formula

```
Expected Claim Frequency  =  Poisson GLM predicted probability x Exposure Years
Expected Claim Severity   =  Gamma GLM predicted loss amount
Pure Premium              =  Frequency x Severity
Technical Premium         =  Pure Premium / (1 - Total Load)
```

**Loads applied:**

| Load | Value | Rationale |
|---|---|---|
| Expense load | 25% | Operating costs |
| Profit load | 10% | Required return on equity |
| Cyber catastrophe/systemic load | 12% | Estimated from Hawkes TVaR |
| Reinsurance load | 8% | Ceded reinsurance cost |
| **Total** | **55%** | |

**Risk-Adjusted Premium (Monte Carlo enrichment):**

```
Risk-Adjusted Technical Premium = Technical Premium + (VaR_99 x 5%)
```

---

## 4. Directory Structure

```
Maestro_Cyber-main/
|
+-- README.md                          <- This file
+-- requirements.txt                   <- Python dependencies
+-- app.py                             <- Streamlit Dashboard (main application)
|
+-- code/
|   +-- 01_use_case_1_eda.py           <- EDA, feature engineering, GLM pricing, Monte Carlo
|   +-- 02_use_case_2_bi_simulation.py <- BI ransomware Monte Carlo simulation
|   +-- 03_nlp_severity_baseline.py    <- DistilBERT NLP severity classifier training
|   +-- 05_hawkes_process_simulation.py<- Hawkes Process MLE + contagion simulation (*)
|   +-- models/                        <- Saved trained models (auto-created by scripts)
|       +-- freq_glm_model.joblib      <- Poisson GLM (frequency)
|       +-- sev_glm_model.joblib       <- Gamma GLM (severity)
|       +-- model_scaler.joblib        <- Feature standardization scaler
|       +-- feature_columns.json       <- Ordered list of expected feature columns
|       +-- severity_classifier_baseline.joblib <- DistilBERT + RF NLP model
|       +-- rf_freq_model.joblib       <- Random Forest frequency (legacy)
|       +-- glm_freq_weights.npy       <- GLM weights, NumPy format (legacy)
|       +-- sev_weights.npy            <- Severity weights, NumPy format (legacy)
|
+-- data/
|   +-- DATA_DICTIONARY.md             <- Full schema documentation for all datasets
|   +-- 02_claims.csv                  <- Claim-level loss & timestamp data (*) Hawkes input
|   +-- 03_regulatory_findings.csv     <- Regulatory exam findings (NLP training corpus)
|   +-- 06_outage_events.csv           <- BI-causing outage breakdown
|   +-- 07_modeling_dataset.csv        <- Policy-level merged dataset (pre-features)
|   +-- 07_final_modeling_dataset.csv  <- Final modeling dataset
|   +-- 08_bi_pricing_output_sample.csv<- BI pricing sample output
|   +-- 09_cyber_pricing_features.csv  <- Engineered features (Streamlit input) (*)
|   +-- 09_feature_dictionary.csv      <- Feature engineering notes
|
+-- outputs/
|   +-- eda_visuals/
|   |   +-- cyber_pricing_eda_dashboard.html      <- Interactive Plotly EDA dashboard
|   |   +-- sector_claim_loss_premium_summary.csv
|   |   +-- prior_incident_claim_summary.csv
|   |   +-- control_score_claim_loss_summary.csv
|   |   +-- vendor_pressure_summary.csv
|   +-- model_outputs/
|       +-- pure_premium_indications.csv          <- Per-policy pricing output
|       +-- pure_premium_indications_with_mc.csv  <- Enriched with VaR metrics
|       +-- glm_coefficients.csv                  <- Frequency + Severity GLM coefs
|       +-- model_diagnostics.txt                 <- Model performance report
|       +-- shap_importances.csv                  <- XGBoost SHAP values
|       +-- hawkes_results.json                   <- Hawkes simulation output (*)
|
+-- docs/
    +-- cyber_pricing_feature_engineering_plan.md <- Feature design notes
    +-- Intern_Onboarding_Package.docx            <- Project onboarding guide
```

> `(*)` = Key files directly involved in the Hawkes Process workflow.

---

## 5. Script Reference

### `code/03_nlp_severity_baseline.py` — NLP Severity Classifier

**Must be run first.** Trains the NLP model consumed by `01_use_case_1_eda.py`.

| Aspect | Detail |
|---|---|
| **Input** | `data/03_regulatory_findings.csv` (finding text + severity labels) |
| **Model** | DistilBERT (`distilbert-base-uncased`) → 768-dim CLS embeddings → Random Forest |
| **Target label** | `severity_label`: Low / Medium / High |
| **Data split** | 70% train / 15% val / 15% test (stratified) |
| **Output** | `code/models/severity_classifier_baseline.joblib` |
| **Optional output** | `code/models/questionnaire_quality_classifier.joblib` (if `04_questionnaire_responses.csv` is present) |

---

### `code/01_use_case_1_eda.py` — Core Pricing Pipeline

The **main orchestrator** of the pricing workflow. Runs as a single script from the project root.

**Key functions:**

| Function | Purpose |
|---|---|
| `load_source_data()` | Loads all 6 CSV source datasets with type parsing |
| `run_quality_checks()` | Validates claim consistency and duplicate policy detection |
| `build_modeling_dataset()` | Merges claims, findings, and policies to one row per policy |
| `engineer_pricing_features()` | Creates 17 numeric + 5 categorical features; exports feature dictionary |
| `generate_eda_outputs()` | Produces 6-panel Plotly HTML dashboard and 4 CSV summary tables |
| `run_frequency_severity_template()` | Trains Poisson GLM + Gamma GLM; exports models via joblib |
| `run_monte_carlo_simulation()` | Per-policy Gamma-Poisson Monte Carlo for VaR 95/99 risk metrics |
| `main()` | Orchestrates the full run including live DistilBERT NLP inference |

**Key engineered features:**

| Feature | Source Inputs | Description |
|---|---|---|
| `exposure_size_score` | log_revenue, log_assets, log_employees | Z-score averaged company size proxy |
| `cyber_control_score` | NIST (40%), MFA (25%), EDR (20%), SOC (15%) | Weighted security posture index, range [0, 1] |
| `control_gap_score` | cyber_control_score | Inverted: `1 - control_score` (risk loading signal) |
| `vendor_control_pressure` | n_vendors / (NIST + 0.1) | Supply-chain concentration risk index |
| `vendor_pressure_band` | vendor_control_pressure | Categorical bins: Low / Moderate / High / Extreme |
| `regulatory_findings_pressure` | log(n_findings) x (1 + high_sev_rate + NLP_prob) | Combined finding volume and severity signal |
| `high_sev_rate` | n_high_sev / (n_findings + 1) | Share of high-severity regulatory findings |
| `critical_operations_score` | trading_desk + payments + custodial_AUM flags | Operational criticality sum score |
| `coverage_structure_score` | normalized log(limit) - retention_ratio | Coverage adequacy vs self-insured retention |
| `prior_incident_score` | log1p(prior_incidents_3yr) | Smoothed prior incident count |
| `repeat_offender` | prior_incidents_3yr >= 2 | Binary flag for serial claimants |

---

### `code/02_use_case_2_bi_simulation.py` — BI Monte Carlo Engine

Simulates ransomware Business Interruption scenarios using fully vectorized NumPy.

| Aspect | Detail |
|---|---|
| **Primary inputs** | `data/01_policies.csv` (falls back to `07_modeling_dataset.csv`), `data/06_outage_events.csv`, `data/05_system_recovery_profiles.csv` |
| **Downtime model** | Per system-class bimodal Gaussian Mixture Model fitted to log(downtime_hours) |
| **Simulation count** | 50,000 simulated years per policy using `np.bincount` vectorization |
| **Key features** | Correlated multi-system failures (1-3 simultaneous systems), quarter-end/month-end time multipliers, regulator-specific cost overlays (OCC, FRB, FDIC) |
| **Premium method** | TVaR-99 Cost of Capital: `(Expected Loss + Risk Load) / (1 - Expense Ratio)` |
| **Output** | `data/08_bi_pricing_output_sample.csv` |

---

### `code/05_hawkes_process_simulation.py` — Hawkes Process

See the dedicated [Section 6 deep dive](#6-deep-dive-05_hawkes_process_simulationpy) below.

---

## 6. Deep Dive: 05\_hawkes\_process\_simulation.py

### What Is a Hawkes Process?

A **Hawkes Process** is a self-exciting point process — a stochastic model for event arrival where each past event *temporarily increases* the probability of future events. In cyber insurance this maps directly to real-world dynamics:

- A ransomware breach at one firm spreads across shared vendor ecosystems, increasing short-term attack probability for similarly exposed firms.
- A newly disclosed zero-day vulnerability gets actively exploited in rapid succession.
- Each claim arrival is not independent; it excites a cluster of follow-on claims.

This is fundamentally different from a standard **Poisson Process**, which assumes events are completely independent and arrive at a constant rate.

### Mathematical Foundation

The conditional intensity function (instantaneous claim arrival rate at time *t*) is:

```
lambda(t) = mu + alpha * SUM[ exp(-beta * (t - t_i)) ]   for all past events t_i < t
```

| Parameter | Symbol | Actuarial Meaning |
|---|---|---|
| Baseline intensity | mu | Background claim rate (events/day) — attacks independent of history |
| Excitation coefficient | alpha | How much each new claim spikes the arrival rate |
| Decay rate | beta | How quickly the excitation fades after each event |
| **Branching ratio** | alpha/beta | Must be strictly < 1 for a stationary (non-explosive) process |

**Stationarity constraint:** `alpha < beta` (enforced in the optimizer bounds) ensures the process has a finite long-run mean and does not explode to infinite intensity.

**Unconditional mean rate** (the effective Poisson-equivalent rate):

```
Effective rate = mu / (1 - alpha/beta)
```

### Step-by-Step Code Walkthrough

#### Step 1 — Load and Prepare Event Times

```python
# Load claims and parse dates
df_claims = pd.read_csv('data/02_claims.csv')
df_claims['loss_date'] = pd.to_datetime(df_claims['loss_date'])
df_claims = df_claims.sort_values('loss_date').reset_index(drop=True)

# Convert to relative days from the first event
t_events = (df_claims['loss_date'] - df_claims['loss_date'].min()).dt.days.values
T_max = t_events[-1] + 1  # Total observation window in days
```

The event times are measured in **days relative to the first claim date**. `T_max` is the full observation window used in the likelihood integral.

#### Step 2 — Maximum Likelihood Estimation (MLE)

The **Negative Log-Likelihood** of a Hawkes process observed over `[0, T]`:

```
NLL = mu*T + (alpha/beta) * SUM_i [1 - exp(-beta*(T - t_i))]  -  SUM_i log(lambda(t_i))
```

**Efficient recursive computation** of the log-intensity (reduces from O(n^2) to O(n)):

```python
def hawkes_nll(params, t):
    mu, alpha, beta = params

    # Stationarity check: reject non-physical parameters
    if mu <= 0 or alpha <= 0 or beta <= 0 or alpha >= beta:
        return np.inf

    n = len(t)
    integral_term = mu * T_max + (alpha / beta) * np.sum(1 - np.exp(-beta * (T_max - t)))

    log_intensity_sum = 0
    R = 0  # Running recursive sum: SUM_j<i exp(-beta*(t_i - t_j))
    for i in range(n):
        if i > 0:
            R = np.exp(-beta * (t[i] - t[i-1])) * (1 + R)
        lam_i = mu + alpha * R
        log_intensity_sum += np.log(lam_i)

    return integral_term - log_intensity_sum
```

Optimized using **L-BFGS-B** (Limited-memory Broyden-Fletcher-Goldfarb-Shanno with Bounds):

```python
init_params = [len(t_events)/T_max, 0.05, 0.1]   # Initial guesses
bnds = ((0.01, 5.0), (0.001, 0.9), (0.01, 2.0))   # Parameter bounds

res = minimize(hawkes_nll, init_params, args=(t_events,), method='L-BFGS-B', bounds=bnds)
mu_opt, alpha_opt, beta_opt = res.x
```

#### Step 3 — Severity Distribution Fitting

Historical claim loss amounts are fitted to a **Gamma distribution** via MLE:

```python
sev_data = df_claims['gross_incurred_usd'].dropna().values
shape_g, loc_g, scale_g = stats.gamma.fit(sev_data, floc=0)
```

`floc=0` fixes the location parameter to zero — losses must be non-negative. The resulting `(shape_g, scale_g)` parameters define the severity distribution for the simulation.

#### Step 4 — 50,000-Year Stochastic Simulation

Two parallel simulations run simultaneously for direct comparison:

**Model A — Poisson (Independence Baseline):**

```python
poisson_rate = mu_opt / (1 - alpha_opt / beta_opt)   # Unconditional mean rate
expected_events_yr = 365 * poisson_rate               # Annual event count

for i in range(num_sims):
    n_poisson = np.random.poisson(expected_events_yr)
    if n_poisson > 0:
        annual_losses_poisson[i] = np.sum(np.random.gamma(shape_g, scale_g, n_poisson))
```

**Model B — Hawkes (Contagion), Branching Process Approximation:**

The exact Hawkes simulation (Ogata's thinning algorithm) is computationally intensive for 50,000 years. Instead, a **Negative Binomial branching approximation** is used:

```python
branching_ratio = alpha_opt / beta_opt

# Per-immigrant cluster size statistics
mean_cluster = 1 / (1 - branching_ratio)
var_cluster  = branching_ratio / (1 - branching_ratio)**3

# Match NegBin parameters to cluster mean and variance
p = mean_cluster / var_cluster
r = mean_cluster**2 / (var_cluster - mean_cluster)   # r > 0 required

for i in range(num_sims):
    # 1. Sample number of background (immigrant) events
    n_immigrants = np.random.poisson(365 * mu_opt)

    # 2. Each immigrant spawns a NegBin-distributed cluster of offspring
    if r > 0:
        n_hawkes = np.sum(np.random.negative_binomial(r, p, n_immigrants) + 1)
    else:
        n_hawkes = n_immigrants

    # 3. Severity: Gamma draws for all events
    if n_hawkes > 0:
        annual_losses_hawkes[i] = np.sum(np.random.gamma(shape_g, scale_g, int(n_hawkes)))
```

The **Negative Binomial cluster** captures the heavy-tailed total-event-count distribution that emerges from self-exciting dynamics. A high branching ratio (alpha/beta approaching 1) produces large, rare clusters — the mathematical signature of contagion.

#### Step 5 — TVaR Calculation and Output

```python
# 99th percentile Value at Risk
p99_hawkes = np.percentile(annual_losses_hawkes, 99)

# Tail Value at Risk: expected loss in the worst 1% of years
tvar_hawkes = np.mean(annual_losses_hawkes[annual_losses_hawkes >= p99_hawkes])

# The Contagion Risk Premium is the additional capital needed beyond Poisson
contagion_premium = tvar_hawkes - tvar_poisson
```

Results are saved to `outputs/model_outputs/hawkes_results.json`.

### Script Inputs and Outputs

| Category | Item | Details |
|---|---|---|
| **Input file** | `data/02_claims.csv` | Must contain `loss_date` (date) and `gross_incurred_usd` (numeric) |
| **Output file** | `outputs/model_outputs/hawkes_results.json` | Fitted parameters + TVaR comparison (JSON) |
| **Console output** | Parameter estimates, TVaR values, Contagion Premium | Printed to stdout during execution |

**`hawkes_results.json` schema:**

```json
{
    "mu":               0.0312,
    "alpha":            0.0487,
    "beta":             0.1023,
    "branching_ratio":  0.4762,
    "tvar_poisson":     1234567.89,
    "tvar_hawkes":      1987654.32,
    "contagion_premium": 753086.43
}
```

### Integration with the Rest of the Project

The `hawkes_results.json` output is consumed by two downstream components:

**1. Streamlit Tab 4 — Advanced Contagion (Hawkes):**
The dashboard reads the JSON and renders a TVaR comparison chart showing the Poisson vs. Hawkes loss distributions side-by-side.

**2. Gemini AI Agent (`dynamic_pricing_calculator` tool):**
The agent reads the Hawkes TVaR values to compute a contagion-adjusted technical premium on demand:

```python
hawkes_risk_load = (h_data['tvar_hawkes'] / 5000) * 0.10
final_hawkes = (pure_premium + hawkes_risk_load) / (1 - 0.25)
```

**3. RAG Knowledge Base:**
The chat agent has embedded Hawkes mathematical descriptions, enabling it to answer underwriter questions like *"why is the contagion premium larger than the base premium?"* in plain English.

### Key Functions and Objects

| Symbol | Lines | Description |
|---|---|---|
| `hawkes_nll(params, t)` | 27-45 | Negative log-likelihood; enforces stationarity (alpha < beta) |
| `minimize(...)` | 50 | scipy L-BFGS-B optimizer for bounded MLE |
| `T_max` | 24 | Observation window in days (used in the likelihood integral) |
| `stats.gamma.fit(sev_data, floc=0)` | 62 | MLE Gamma distribution fit to claim severities |
| `branching_ratio` | 69 | alpha/beta — controls cluster size in the branching simulation |
| `np.random.negative_binomial(r, p, n)` | 87 | Samples offspring cluster sizes per immigrant event |
| `np.percentile(arr, 99)` | 93, 96 | 99th percentile VaR calculation |
| `annual_losses_hawkes`, `annual_losses_poisson` | 65-66 | Output arrays holding 50,000 simulated annual loss totals |

---

## 7. Technologies and Dependencies

### Core Scientific Stack

| Library | Version | Role in Project |
|---|---|---|
| `pandas` | >= 2.0 | All CSV loading, date parsing, groupby aggregations |
| `numpy` | >= 1.24 | Vectorized Monte Carlo loops, array math, random sampling |
| `scipy` | >= 1.11 | L-BFGS-B MLE optimizer (`scipy.optimize.minimize`), Gamma MLE (`scipy.stats`) |
| `scikit-learn` | >= 1.3 | Poisson GLM, Gamma GLM, GaussianMixture, RandomForest, train/test split |

### Machine Learning and NLP

| Library | Version | Role in Project |
|---|---|---|
| `xgboost` | >= 2.0 | Gradient boosting benchmark models (frequency + severity) |
| `lightgbm` | >= 4.0 | Alternative gradient boosting (available in env, optional use) |
| `transformers` | >= 4.35 | DistilBERT tokenizer and model for NLP embedding extraction |
| `torch` | >= 2.0 | PyTorch backend for DistilBERT — auto-selects CUDA/MPS/CPU |
| `sentence-transformers` | >= 2.2 | Sentence embedding utilities |
| `shap` | >= 0.43 | SHAP explainability values for XGBoost models |
| `joblib` | >= 1.3 | Save/load trained scikit-learn models |

### Visualization and Dashboard

| Library | Version | Role in Project |
|---|---|---|
| `streamlit` | >= 1.28 | Interactive web dashboard framework |
| `plotly` | >= 5.17 | Interactive charts: histograms, scatter plots, bar charts |
| `matplotlib` | >= 3.7 | SHAP waterfall and beeswarm plots |
| `seaborn` | >= 0.13 | Statistical visualization utilities |

### AI and Cloud

| Library | Version | Role in Project |
|---|---|---|
| `google-genai` | latest | Gemini AI agent with RAG and Function Calling support |
| `jupyter` | >= 1.0 | Notebook exploration (not required for pipeline execution) |

### Dependencies Specific to the Hawkes Simulation

`05_hawkes_process_simulation.py` depends **only** on the core scientific stack — no additional installs are needed beyond `requirements.txt`:

```
pandas  -> data loading and date arithmetic
numpy   -> random sampling, percentile calculation, array operations
scipy   -> scipy.optimize.minimize (L-BFGS-B), scipy.stats.gamma.fit
json    -> writing hawkes_results.json
```

---

## 8. Installation and Environment Setup

### Prerequisites

| Requirement | Minimum | Notes |
|---|---|---|
| Python | 3.10 | 3.11 strongly recommended |
| pip | 23.0 | `python -m pip install --upgrade pip` |
| RAM | 8 GB | 16 GB recommended for DistilBERT embedding extraction |
| VRAM (optional) | 4 GB | GPU optional; CPU fallback is automatic |
| Git | 2.40 | Required only for cloning |

### Step 1 — Clone the Repository

```bash
git clone https://github.com/your-org/Maestro_Cyber-main.git
cd Maestro_Cyber-main
```

### Step 2 — Create a Virtual Environment

**Windows (PowerShell):**

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

> If PowerShell blocks execution, run this first:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

**macOS / Linux:**

```bash
python -m venv venv
source venv/bin/activate
```

### Step 3 — Upgrade pip and Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> The first install may take 5-15 minutes due to `torch` (~2 GB) and `transformers`. Subsequent installs use the pip cache and complete in under a minute.

### Step 4 — Verify Installation

```bash
python -c "import pandas, numpy, scipy, sklearn, torch, transformers, streamlit, xgboost, plotly; print('All imports OK')"
```

Expected output: `All imports OK`

### Step 5 — Check GPU Availability (Optional)

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('MPS:', torch.backends.mps.is_available())"
```

The NLP pipeline selects automatically: CUDA > MPS (Apple Silicon) > CPU.

### Step 6 — Output Directory Setup

The scripts create all required output directories automatically. To create them manually:

**Windows PowerShell:**

```powershell
New-Item -ItemType Directory -Force -Path "outputs\eda_visuals", "outputs\model_outputs", "code\models"
```

**macOS / Linux:**

```bash
mkdir -p outputs/eda_visuals outputs/model_outputs code/models
```

---

## 9. Running the Pipeline

> **Critical:** All commands must be run from the **project root directory** (`Maestro_Cyber-main/`). Relative paths such as `data/`, `outputs/`, and `code/models/` will break if you run scripts from inside the `code/` subdirectory.

### Step A — Train the NLP Severity Model

```bash
python code/03_nlp_severity_baseline.py
```

This step downloads `distilbert-base-uncased` weights (~250 MB) on the first run from HuggingFace Hub, extracts 768-dimensional embeddings, and trains a Random Forest classifier.

**Expected terminal output:**

```
Using device for DistilBERT: cpu
Loading datasets...
Loaded 5000 regulatory findings
Loading DistilBERT model for feature extraction...
Extracting embeddings for findings... This may take a moment.
Extracted embedding matrix shape: (5000, 768)
Training AI-Powered Model (Random Forest on DistilBERT)...
Macro F1-score: 0.8234
Macro AUC-ROC: 0.9101
Saved best model: code/models/severity_classifier_baseline.joblib
```

**Estimated time:** 2-10 minutes on CPU, 30-60 seconds on GPU.

---

### Step B — Run the Core Pricing Pipeline

```bash
python code/01_use_case_1_eda.py
```

**Expected terminal output:**

```
Loading existing modeling dataset...
Loading findings and running NLP inference...
Loading DistilBERT for inference...
--- 3. Feature engineering and pruning ---
Wrote engineered features: data/09_cyber_pricing_features.csv
--- 4. EDA summaries and visualizations ---
Wrote EDA dashboard: outputs/eda_visuals/cyber_pricing_eda_dashboard.html
--- 5. Frequency-severity pure premium template ---
[Actuarial GLM Frequency Model - Poisson]
[Actuarial GLM Severity Model - Gamma]
Average pure premium: 277,206.89
Average technical premium template: 616,015.30
--- 6. Executing Monte Carlo Risk Simulation ---
Monte Carlo metrics calculated successfully.
Enriched pricing file saved to: outputs/model_outputs/pure_premium_indications_with_mc.csv
Exported interactive models to: code/models
```

---

### Step C — Run the BI Monte Carlo Engine

```bash
python code/02_use_case_2_bi_simulation.py
```

**Expected terminal output:**

```
LOADING DATA & FITTING MODELS
Fitting 2-Component Gaussian Mixtures to Downtime Data...
  Core Banking              | Means (log-hours): [2.31 4.78]
  Trading Platform          | Means (log-hours): [1.85 5.12]
  ...
STEP 5: PRICING SAMPLE INSUREDS (VECTORIZED + CoC)
BI Pricing Output (50,000 Simulations/Policy):
 policy_id  sub_sector  revenue_mm  bi_technical_premium
   POL-0001  Retail Bank       450            456,789
Wrote 6 sample outputs.
```

---

### Step D — Run the Hawkes Process Simulation

```bash
python code/05_hawkes_process_simulation.py
```

**Expected terminal output:**

```
============================================================
HAWKES PROCESS OPTIMIZATION & SIMULATION (Enhanced)
============================================================
Loaded 269 claims spanning 2020-01-05 to 2024-12-15

[Severity Models]
  Gamma:  shape=0.4231, scale=2,847,321
  GPD POT threshold u = $185,000 (75th pct), xi=0.3821, sigma=412,000
  P(Loss > u) = 0.250  (67 exceedances out of 269)

[Global Hawkes MLE]
  Optimization successful: True
  Baseline (mu):      0.0312 events/day
  Excitation (alpha): 0.0487
  Decay (beta):       0.1023
  Branching ratio:    0.4762  (must be < 1 for stationarity)

[Sector-Specific Hawkes Models by Cause of Loss]
  Ransomware                         | n= 86 | mu=0.0421 | alpha=0.0631 | beta=0.1205 | br=0.5238
  Data Breach                        | n= 72 | mu=0.0289 | alpha=0.0318 | beta=0.0892 | br=0.3565
  Business Email Compromise          | n= 43 | mu=0.0198 | alpha=0.0187 | beta=0.1024 | br=0.1826
  Insider Threat                     | n= 38 | mu=0.0155 | alpha=0.0094 | beta=0.0998 | br=0.0942

[Running 50,000-year Monte Carlo simulation...]
  Simulation complete.
  Computing 500-sample bootstrap TVaR confidence intervals...

─────────────────────────────────────────────────────────────────
  [Poisson + Gamma]    TVaR 99% :     $1,234,567
  [Hawkes + Gamma]     TVaR 99% :     $1,987,654
  [Hawkes + GPD]       TVaR 99% :     $2,841,239  ← primary
    95% CI: [$2,698,412 – $2,991,087]
  [Trended (+14%/yr x3yr)] TVaR 99% :   $4,213,892
─────────────────────────────────────────────────────────────────
  Contagion premium (Hawkes GPD - Poisson): +$1,606,672
  Fat-tail premium (GPD - Gamma):           +$853,585
  Trend premium (+3yr projection):          +$1,372,653

  AEP/OEP curve saved to outputs/model_outputs/oep_curve.csv
  Results saved to outputs/model_outputs/hawkes_results.json
```

**Estimated time:** 60–120 seconds on CPU (bootstrap adds ~60s). Set `N_BOOTSTRAP=100` in the script to reduce to ~30s.

---

### Run the Complete Pipeline

```bash
# Step A: Train NLP model (run once, or whenever findings data changes)
python code/03_nlp_severity_baseline.py

# Step B: EDA, feature engineering, GLM pricing, Monte Carlo
python code/01_use_case_1_eda.py

# Step C: BI ransomware Monte Carlo
python code/02_use_case_2_bi_simulation.py

# Step D: Hawkes Process contagion simulation (GPD, trend, sector models, OEP)
python code/05_hawkes_process_simulation.py

# Step E: Catastrophe scenario stress testing (Cloud, Ransomware, Supply Chain, Critical Infra)
python code/04_catastrophe_scenarios.py

# Step F: Portfolio accumulation risk analysis (HHI, MFL, vendor/cloud concentration)
python code/06_portfolio_accumulation.py

# Step G: Launch the Streamlit dashboard
streamlit run app.py
```

**Windows PowerShell one-liner:**

```powershell
python code/03_nlp_severity_baseline.py; python code/01_use_case_1_eda.py; python code/02_use_case_2_bi_simulation.py; python code/05_hawkes_process_simulation.py; python code/04_catastrophe_scenarios.py; python code/06_portfolio_accumulation.py; streamlit run app.py
```

---

## 10. Running the Streamlit Dashboard

```bash
streamlit run app.py
```

The dashboard opens automatically at `http://localhost:8501`.

### Dashboard Tabs

| Tab | Icon | Description |
|---|---|---|
| Portfolio Analytics & AI Explainer | Categorical and numeric feature impact charts, GLM coefficient explorer, Gemini AI chatbot with RAG |
| Feature Derivation Explainer | Mathematical walkthrough of every engineered feature with derivation formulas |
| Interactive Pricing Engine | Real-time premium recalculation via sliders (MFA coverage, NIST score, vendor count, revenue) |
| Advanced Contagion (Hawkes) | Hawkes simulation visualization, TVaR comparison, contagion premium display |

### Configuring the AI Agent (Tab 1)

1. Obtain a [Google Gemini API key](https://aistudio.google.com/)
2. Paste the key into the password field in Tab 1
3. The agent auto-generates an actuarial report using GLM coefficients
4. Ask follow-up questions:
   - *"Why is the vendor control pressure coefficient so large?"*
   - *"Price a $500M revenue bank with NIST 3.0 and 60% MFA coverage"*
   - *"Explain the Contagion Risk Premium in plain English"*

> **API Tier Recommendation:** Gemini Pro or higher is strongly recommended. The free tier may exhaust token limits when processing the full portfolio dataset and GLM coefficients in a single session.

---

## 11. Expected Outputs

After running the full pipeline, these files will be present:

### Engineered Data

| File | Size | Description |
|---|---|---|
| `data/09_cyber_pricing_features.csv` | ~460 KB | 1,500 policies with all engineered features |
| `data/09_feature_dictionary.csv` | ~700 B | Feature engineering rationale |
| `data/08_bi_pricing_output_sample.csv` | ~600 B | BI pricing for 6 representative policies |

### Trained Models

| File | Size | Description |
|---|---|---|
| `code/models/freq_glm_model.joblib` | ~1.3 KB | Fitted Poisson GLM |
| `code/models/sev_glm_model.joblib` | ~1.3 KB | Fitted Gamma GLM |
| `code/models/model_scaler.joblib` | ~7 KB | Feature standardization scaler |
| `code/models/feature_columns.json` | ~1.2 KB | Ordered feature column names |
| `code/models/severity_classifier_baseline.joblib` | ~300 KB | DistilBERT + Random Forest NLP model |

### Pricing and Risk Reports

| File | Description |
|---|---|
| `outputs/eda_visuals/cyber_pricing_eda_dashboard.html` | Interactive 6-panel Plotly EDA dashboard |
| `outputs/model_outputs/pure_premium_indications_with_mc.csv` | Per-policy: frequency, severity, pure premium, technical premium, VaR 95/99, risk-adjusted premium |
| `outputs/model_outputs/glm_coefficients.csv` | Poisson and Gamma GLM coefficients for all features |
| `outputs/model_outputs/model_diagnostics.txt` | Accuracy, precision, recall, MAE, and pricing formula summary |
| `outputs/model_outputs/hawkes_results.json` | Fitted Hawkes parameters, Poisson TVaR, Hawkes TVaR, Contagion Premium |

### Sample `model_diagnostics.txt`

```
Cyber Frequency-Severity Pure Premium Template
================================================
Rows: 1,500  |  Train: 1,111  |  Test: 389

Frequency model: Poisson GLM
Train accuracy: 0.6949  |  Test accuracy: 0.7147
Train recall:   0.4541  |  Test recall:   0.4259

Severity model: Gamma GLM on positive-loss claims
Positive-claim severity MAE on test: $2,647,839.68

Portfolio averages
  Average current premium:               $15,998,649.64
  Average pure premium:                     $277,206.89
  Average technical premium template:       $616,015.30
```

---

## 12. Troubleshooting

### FileNotFoundError: `data/02_claims.csv`

```
Cause: Script not run from project root directory.
Fix:   cd Maestro_Cyber-main
       python code/05_hawkes_process_simulation.py
```

### FileNotFoundError: `data/01_policies.csv` in BI simulation

```
Cause: This source file may not be included in all repository distributions.
Fix:   No action needed. The script automatically falls back to
       07_modeling_dataset.csv when 01_policies.csv is missing.
```

### ModuleNotFoundError: `No module named 'google.genai'`

```
Fix:   pip install google-genai
```

### NLP model not found warning in `01_use_case_1_eda.py`

```
Warning: NLP model not found or failed. Using ground truth labels.
Cause:  03_nlp_severity_baseline.py has not been run yet.
Fix:    python code/03_nlp_severity_baseline.py
        Then re-run 01_use_case_1_eda.py
```

### `Optimization Successful: False` in Hawkes simulation

```
Cause: L-BFGS-B failed to converge. Can occur when claims data
       has very few events or unusual temporal spacing.
Fix:   Adjust initial parameters in 05_hawkes_process_simulation.py:
       init_params = [0.01, 0.01, 0.05]   # Lower initial guesses
       Or add: options={'maxiter': 1000} to the minimize() call.
```

### `hawkes_results.json not found` error in Streamlit

```
Cause: 05_hawkes_process_simulation.py has not been run.
Fix:   python code/05_hawkes_process_simulation.py
       The dashboard gracefully falls back to GLM-based risk loads
       if this file is absent.
```

### Streamlit dashboard shows `Data not found`

```
Cause: 09_cyber_pricing_features.csv has not been generated.
Fix:   python code/01_use_case_1_eda.py
```

### CUDA out of memory during DistilBERT inference

```
Cause: GPU memory insufficient for the default batch size of 32.
Fix:   Reduce batch size in 03_nlp_severity_baseline.py (line 44):
       batch_size = 8
       Or allow CPU fallback: the pipeline auto-selects CPU when GPU fails.
```

### Slow embedding extraction on CPU

```
Expected time: 5-15 minutes for 5,000 findings on CPU.
Tip: If severity_classifier_baseline.joblib already exists from a
     previous run, you can skip Step A entirely.
     Step A only needs to be re-run when the findings data changes.
```

### PowerShell execution policy error

```
Error: cannot be loaded because running scripts is disabled on this system
Fix:   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
       Then re-run: venv\Scripts\Activate.ps1
```

---

## 13. Limitations and Known Issues

### Methodological Limitations

1. **Synthetic data only.** All datasets are computer-generated. Results are illustrative and must not be presented as actual portfolio insights or used in production pricing without calibration against real claims data and independent actuarial review.

2. **Hawkes branching approximation.** The 50,000-year simulation uses a Negative Binomial branching approximation rather than exact Hawkes path simulation (Ogata's thinning algorithm). This is computationally efficient but introduces approximation error, especially when the branching ratio (alpha/beta) exceeds 0.5.

3. **Univariate Hawkes model.** The current model uses a single scalar intensity function. A production cyber model should use a **multivariate Hawkes process** to capture cross-excitation between distinct attack types (ransomware, DDoS, data exfiltration) and geographic/sector clusters.

4. **Poisson GLM assumes independence.** The frequency GLM assigns no temporal correlation between claims. This is precisely why the Hawkes model is needed as a complementary tail-risk tool — the GLM quantifies average expected loss while the Hawkes simulation quantifies worst-case systemic scenarios.

5. **Severity-frequency independence.** The GLM treats severity and frequency as independent. In real cyber portfolios, large systemic events typically produce both higher claim counts and higher per-claim losses (positive correlation), causing the GLM to underestimate expected losses during catastrophic scenarios.

### Known Data Issues

6. **Regulator-subsector mismatches.** A small number of records in `01_policies.csv` intentionally contain regulator/subsector inconsistencies (e.g., a Regional Bank with FINRA listed as primary regulator). These are data quality exercises for training purposes, not errors to be fixed.

### Technical Limitations

7. **GLMs retrained on every Streamlit launch.** The models in `app.py` are trained from scratch each time the dashboard starts (they are not loaded from the `.joblib` files). This keeps the interactive pricing engine synchronized with the current feature data at the cost of a 2-3 second startup delay.

8. **Gemini API key not persisted.** The API key entered in the dashboard is held in Streamlit session state only and must be re-entered after every app restart.

9. **Bootstrap TVaR confidence intervals are compute-intensive.** The 500-sample bootstrap in `05_hawkes_process_simulation.py` adds ~30–60 seconds to the Hawkes run time on CPU. The simulation is now seeded (`np.random.seed(42)`) for full reproducibility. To speed up, reduce `N_BOOTSTRAP` in the script from 500 to 100.

---

## 14. Data Dictionary Reference

Full schema documentation for all raw datasets is in [`data/DATA_DICTIONARY.md`](data/DATA_DICTIONARY.md).

### Entity Relationship Summary

```
insured (insured_id)     1 ---- M   policies (policy_id)
policies (policy_id)     1 ---- M   claims (claim_id)              <- Hawkes input
claims (claim_id)        1 ---- 0/1 outage_events (outage_id)      <- BI sim input
policies (policy_id)     1 ---- M   regulatory_findings            <- NLP input
policies (policy_id)     1 ---- M   questionnaire_responses        <- NLP input
policies (policy_id)     1 ---- M   system_recovery_profiles       <- BI sim input
```

### Columns Used by the Hawkes Simulation

| Dataset | Column | Type | Used For |
|---|---|---|---|
| `02_claims.csv` | `loss_date` | date | Event timestamps — converted to relative days for MLE |
| `02_claims.csv` | `gross_incurred_usd` | int | Total loss — fitted to Gamma severity distribution |

---

## Contributing

This is an internship research project at Maestro. For questions, refer to `docs/Intern_Onboarding_Package.docx` or contact the project team lead.

---

## License

For internal research and educational use only. Not for production deployment without independent actuarial review and regulatory compliance validation.

---

*Last updated: September 2026 | Maestro Cyber Actuarial Research*
