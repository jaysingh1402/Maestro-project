# Data Dictionary — Cyber Pricing Models Mock Datasets

All datasets are synthetic. They are designed to mirror the structure and statistical properties of real underwriting and claims data, but no real institution is represented. Use these for development, prototyping, and validation only. Do not present any results from this data to leadership or clients as actual portfolio insights.

---

## 1. `01_policies.csv` — Policy & Exposure Master

Granularity: one row per policy term. Each insured may appear in multiple policy years.

| Column | Type | Description | Notes |
|---|---|---|---|
| `policy_id` | string | Unique policy term identifier | Primary key |
| `insured_id` | string | Stable insured entity identifier | Join key for renewals |
| `sub_sector` | string | Financial sector sub-segment | 12 categories |
| `primary_regulator` | string | Primary prudential or conduct regulator | Used for finding-source weighting |
| `region` | string | US region of HQ | Geographic feature |
| `inception_date` | date | Policy effective date | |
| `expiration_date` | date | Policy expiration date | |
| `policy_year` | int | Underwriting year | 2020–2024 |
| `revenue_mm` | float | Annual revenue in $ millions | Primary exposure base |
| `employees` | int | Employee headcount | |
| `assets_mm` | float | Total assets in $ millions | Secondary exposure base |
| `limit_mm` | float | Aggregate policy limit in $ millions | |
| `retention_mm` | float | Self-insured retention in $ millions | |
| `premium_usd` | int | Charged annual premium in USD | Target for v0 calibration |
| `has_trading_desk` | bool | Operates a trading desk | Operational feature |
| `processes_payments` | bool | Processes payments/payment rails | Operational feature |
| `custodial_aum_bn` | float | Custodial AUM in $ billions | 0 for non-custodial |
| `core_banking_vendor` | string | Primary core banking platform | Concentration risk signal |
| `n_third_party_vendors` | int | Count of material third-party vendors | |
| `cloud_provider_primary` | string | Primary cloud provider | |
| `mfa_coverage_pct` | float | % of users with MFA enforced | |
| `edr_deployed` | bool | EDR tooling deployed | |
| `soc_24_7` | bool | 24/7 SOC monitoring | |
| `control_maturity_nist` | float | Self-reported NIST CSF maturity (1-5) | Composite score |
| `prior_incidents_3yr` | int | Reported cyber incidents in prior 3 years | |

**Known data quality issues** (intentional, for the team to discover): a small number of records have regulator-subsector mismatches (e.g. Regional Bank with FINRA listed as primary regulator). Treat this as a real-world data hygiene task.

---

## 2. `02_claims.csv` — Loss & Claim Detail

Granularity: one row per claim. Links to policies via `policy_id`.

| Column | Type | Description | Notes |
|---|---|---|---|
| `claim_id` | string | Unique claim identifier | |
| `policy_id` | string | Foreign key to policies | |
| `insured_id` | string | Foreign key to insured | |
| `cause_of_loss` | string | Categorical loss cause | 8 categories |
| `loss_date` | date | Date of loss | |
| `report_lag_days` | int | Days from loss to first notice | |
| `gross_incurred_usd` | int | Gross incurred loss | BI + non-BI |
| `bi_loss_usd` | int | Business interruption component | Use Case 2 target |
| `non_bi_loss_usd` | int | Non-BI component (forensics, notification, regulatory, etc.) | |
| `paid_to_date_usd` | int | Paid as of data extract date | < gross for open claims |
| `downtime_hours` | float | Total operational downtime | 0 if no BI |
| `systems_affected` | string | Primary system class impacted | |
| `claim_status` | string | Open / Closed / Reopened | |

---

## 3. `03_regulatory_findings.csv` — Regulatory Examination Findings (Unstructured Text)

Granularity: one row per finding. Multiple findings per insured possible.

| Column | Type | Description | Notes |
|---|---|---|---|
| `finding_id` | string | Unique finding identifier | |
| `insured_id` | string | Foreign key to insured | |
| `policy_id` | string | Policy term in effect at exam | |
| `examining_agency` | string | Regulator that issued finding | |
| `exam_date` | date | Date of examination | |
| `severity_label` | string | **Ground-truth label** for NLP training | Low / Medium / High |
| `control_area` | string | Control domain | |
| `finding_text` | string | **Unstructured finding text — NLP input** | This is the field for severity classification training |

**This is the primary training corpus for the NLP severity classifier.** The `severity_label` column is the supervised target. In real production data this label will not exist — your job is to build a model that predicts it from `finding_text`.

---

## 4. `04_questionnaire_responses.csv` — Underwriting Questionnaire Free-Text Responses

Granularity: one row per question per policy. 10 questions per policy.

| Column | Type | Description | Notes |
|---|---|---|---|
| `response_id` | string | Unique response identifier | |
| `policy_id` | string | Foreign key to policies | |
| `insured_id` | string | Foreign key to insured | |
| `question_id` | string | Question identifier (Q01–Q10) | |
| `question_text` | string | The question asked | |
| `response_text` | string | **Free-text response — NLP input** | |
| `response_quality_label` | string | **Ground-truth label** | high / medium / low |

**This is the second NLP training corpus** — for response-quality scoring. Aggregated quality across all 10 questions feeds into the control-maturity feature for the pricing model.

---

## 5. `05_system_recovery_profiles.csv` — Stated RTO/RPO by System Class (Use Case 2)

Granularity: one row per (insured, system class).

| Column | Type | Description |
|---|---|---|
| `policy_id` | string | Foreign key |
| `insured_id` | string | Foreign key |
| `system_class` | string | One of 8 system classes |
| `stated_rto_hours` | float | Stated recovery time objective |
| `stated_rpo_hours` | float | Stated recovery point objective |
| `revenue_dependency_pct` | float | % of daily revenue depending on this system |
| `has_hot_dr_site` | bool | Hot DR site available |
| `third_party_dependency` | string | None / Single Vendor / Multi-Vendor |
| `last_dr_test_date` | date | Last documented DR test |

This feeds the per-system downtime distributions in the BI Monte Carlo model.

---

## 6. `06_outage_events.csv` — Historical Outage Breakdown (Use Case 2)

Granularity: one row per BI-causing claim. Joins to `02_claims.csv` via `claim_id`.

| Column | Type | Description |
|---|---|---|
| `outage_id` | string | Unique outage identifier |
| `claim_id` | string | Foreign key to claims |
| `policy_id` | string | Foreign key to policies |
| `cause_of_loss` | string | Replicated from claim |
| `outage_start_date` | date | Outage start |
| `total_downtime_hours` | float | Total outage duration |
| `detection_hours` | float | Time from start to detection |
| `containment_hours` | float | Time from detection to containment |
| `recovery_hours` | float | Time from containment to full recovery |
| `systems_affected` | string | Primary system class affected |
| `n_systems_affected` | int | Count of distinct systems affected |
| `is_quarter_end` | bool | Outage occurred near quarter-end |
| `is_month_end` | bool | Outage occurred near month-end |
| `is_weekend_outage` | bool | Outage started on weekend |
| `ransom_demand_usd` | int | Ransom demand if applicable |
| `ransom_paid` | bool | Whether ransom was paid |
| `lost_revenue_usd` | int | Lost revenue component of BI |
| `extra_expense_usd` | int | Extra expense component |
| `forensics_cost_usd` | int | Forensics / IR costs |
| `restoration_cost_usd` | int | System restoration costs |
| `regulatory_notification_triggered` | bool | 36-hour notification rule triggered |

This is the core data for calibrating the BI severity model and the downtime distributions.

---

## Joins & Entity Relationships

```
insured (insured_id)  1 ──── M  policies (policy_id)
policies (policy_id)  1 ──── M  claims (claim_id)
claims (claim_id)     1 ──── 0/1  outage_events (outage_id)
policies (policy_id)  1 ──── M  regulatory_findings (finding_id)
policies (policy_id)  1 ──── M  questionnaire_responses (response_id)
policies (policy_id)  1 ──── M  system_recovery_profiles (one row per system)
```
