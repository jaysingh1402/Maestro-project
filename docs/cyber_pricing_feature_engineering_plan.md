# Cyber Pricing Feature Engineering Plan

## Purpose

This document describes the complete feature engineering and pricing workflow for the Maestro Cyber Pricing Engine. The pipeline covers:

1. NLP-powered severity classification of regulatory findings (DistilBERT).
2. Building a clean, pruned policy-level modeling dataset.
3. Engineering composite pricing features from raw policy, claim, and finding data.
4. Generating EDA visualizations focused on pricing signals.
5. Training transparent Poisson GLM (frequency) and Gamma GLM (severity) models.
6. Calculating pure premium, technical premium, and Monte Carlo VaR metrics.
7. Running a Hawkes Process contagion simulation for systemic tail-risk quantification.

---

## Full Run Order

Run all commands from the **project root directory**.

```bash
# Step 1: Train NLP severity classifier (run once or when findings data changes)
python code/03_nlp_severity_baseline.py

# Step 2: EDA, feature engineering, GLM pricing, Monte Carlo VaR
python code/01_use_case_1_eda.py

# Step 3: BI ransomware Monte Carlo simulation (Use Case 2)
python code/02_use_case_2_bi_simulation.py

# Step 4: Hawkes Process contagion simulation and TVaR comparison
python code/05_hawkes_process_simulation.py

# Step 5: Launch the Streamlit dashboard
streamlit run app.py
```

If your default Python environment lacks the required packages, activate the project virtual environment first:

```bash
# Windows
venv\Scripts\Activate.ps1

# macOS / Linux
source venv/bin/activate
```

---

## Main Outputs

### From `03_nlp_severity_baseline.py`

- `code/models/severity_classifier_baseline.joblib` — DistilBERT + Random Forest NLP model
- `code/models/questionnaire_quality_classifier.joblib` — (optional) questionnaire quality model

### From `01_use_case_1_eda.py`

- `data/09_cyber_pricing_features.csv` — 1,500-row engineered feature dataset
- `data/09_feature_dictionary.csv` — engineering rationale for each feature
- `outputs/eda_visuals/cyber_pricing_eda_dashboard.html` — interactive 6-panel Plotly dashboard
- `outputs/eda_visuals/sector_claim_loss_premium_summary.csv`
- `outputs/eda_visuals/prior_incident_claim_summary.csv`
- `outputs/eda_visuals/control_score_claim_loss_summary.csv`
- `outputs/eda_visuals/vendor_pressure_summary.csv`
- `outputs/model_outputs/pure_premium_indications.csv`
- `outputs/model_outputs/pure_premium_indications_with_mc.csv` — enriched with VaR 95/99
- `outputs/model_outputs/glm_coefficients.csv` — Poisson + Gamma GLM coefficients
- `outputs/model_outputs/model_diagnostics.txt` — accuracy, recall, MAE, pricing formula summary
- `code/models/freq_glm_model.joblib`
- `code/models/sev_glm_model.joblib`
- `code/models/model_scaler.joblib`
- `code/models/feature_columns.json`

### From `02_use_case_2_bi_simulation.py`

- `data/08_bi_pricing_output_sample.csv` — BI pricing for 6 representative policies

### From `05_hawkes_process_simulation.py`

- `outputs/model_outputs/hawkes_results.json` — fitted Hawkes parameters and TVaR comparison

---

## Redundant Variables Merged

The raw dataset contains several correlated or overlapping variables. For model stability and interpretability, the pricing template collapses them into composite features:

| Composite Feature | Source Inputs | Engineering Rationale |
|---|---|---|
| `exposure_size_score` | `log_revenue`, `log_assets`, `log_employees` | Revenue, assets, and employees are all company-size proxies; z-score averaged |
| `cyber_control_score` | `control_maturity_nist` (40%), `mfa_coverage_pct` (25%), `edr_deployed` (20%), `soc_24_7` (15%) | Weighted composite of the four key security posture signals |
| `control_gap_score` | `cyber_control_score` | `1 - cyber_control_score`; inverted for risk loading; higher = weaker controls |
| `vendor_control_pressure` | `n_third_party_vendors`, `control_maturity_nist` | `n_vendors / (NIST + 0.1)`; high vendor count with weak controls = extreme supply-chain risk |
| `vendor_pressure_band` | `vendor_control_pressure` | Categorical bins: Low / Moderate / High / Extreme |
| `regulatory_findings_pressure` | `n_findings`, `n_high_sev`, `n_med_sev`, `high_sev_prob` (NLP) | `log(n_findings) * (1 + high_sev_rate + NLP_prob) * (1 + 0.25 * med_sev_rate)` |
| `high_sev_rate` | `n_high_sev`, `n_findings` | `n_high_sev / (n_findings + 1)`; share of findings classified as High severity |
| `critical_operations_score` | `has_trading_desk`, `processes_payments`, `has_custodial_aum` | Sum of binary operational criticality flags (range 0-3) |
| `coverage_structure_score` | `limit_mm`, `retention_mm`, `revenue_mm` | `minmax(log(limit)) - minmax(retention_ratio)`; captures coverage adequacy |
| `prior_incident_score` | `prior_incidents_3yr` | `log1p(prior_incidents_3yr)`; log-smoothed incident history |
| `repeat_offender` | `prior_incidents_3yr` | Binary flag: 1 if `prior_incidents_3yr >= 2` |

---

## Variables Excluded From Model Predictors

These columns are retained in output files for auditing, comparison, and joins but are **not used as model inputs**:

| Column | Reason Excluded |
|---|---|
| `policy_id`, `insured_id` | Identifier fields; no predictive information |
| `had_claim`, `n_claims` | Target-leakage risk; outcome variables |
| `total_loss`, `bi_loss` | Target variables for severity model |
| `premium_usd`, `loss_ratio` | Direct pricing outputs; would cause target leakage |

---

## Pricing Template

### Core Formula

```
Expected Claim Frequency = predicted_claim_probability * exposure_years
Expected Claim Severity  = Gamma GLM predicted loss (dollars)
Pure Premium             = Expected Claim Frequency * Expected Claim Severity
Technical Premium        = Pure Premium / (1 - total_load)
Risk-Adjusted Premium    = Technical Premium + (VaR_99 * 0.05)
```

### Load Assumptions

| Load Component | Value | Notes |
|---|---|---|
| Expense load | 25% | Operating and distribution costs |
| Profit load | 10% | Required return on equity |
| Cyber catastrophe/systemic load | 12% | Estimated from Hawkes TVaR analysis |
| Reinsurance load | 8% | Estimated ceded reinsurance cost |
| **Total load** | **55%** | |

> **Important:** These loads are placeholders designed for demonstration. They must be calibrated against actual expense studies, reinsurance treaties, and regulatory capital requirements before any commercial use.

---

## NLP Feature Integration

The `regulatory_findings_pressure` feature is enriched by the NLP model trained in `03_nlp_severity_baseline.py`. Specifically:

1. `03_nlp_severity_baseline.py` trains a DistilBERT + Random Forest classifier on `finding_text` → `severity_label` (Low / Medium / High).
2. `01_use_case_1_eda.py` loads this model, extracts DistilBERT embeddings for all findings, and produces `high_sev_prob` — the model's confidence that a finding is High severity.
3. `high_sev_prob` is incorporated into `regulatory_findings_pressure` as an additive term, increasing the composite score for policies with findings that the NLP model classifies as high risk even if the original label was Medium or Low.

---

## Hawkes Process Integration

The Hawkes Process simulation (`05_hawkes_process_simulation.py`) runs independently of the GLM pipeline and does not modify any feature data. Its role is to:

1. Use claim timestamps from `02_claims.csv` to measure **temporal clustering** in historical losses.
2. Fit a self-exciting point process model (Hawkes) via Maximum Likelihood Estimation.
3. Simulate 50,000 portfolio years under both Poisson (independence) and Hawkes (contagion) assumptions.
4. Quantify the **Contagion Risk Premium** — the additional TVaR capital requirement attributable to systemic cyber risk.

The `contagion_premium` value from `hawkes_results.json` is used by the Streamlit dashboard's AI agent to adjust the technical premium upward for policies with high vendor concentration or weak controls.

---

## Next Development Steps

The following improvements are recommended before this template is used for any commercial purposes:

1. Add cross-validation (k-fold) for GLM model selection and stability testing.
2. Add model calibration plots (reliability diagrams) for the Poisson frequency model.
3. Develop separate BI-specific frequency and severity models rather than using aggregate `total_loss`.
4. Upgrade the univariate Hawkes model to a multivariate Hawkes process with cross-excitation between attack types.
5. Calibrate all load assumptions against actual expense studies and reinsurance treaty structures.
6. Add model monitoring infrastructure for detecting distribution shift in incoming policy data.

