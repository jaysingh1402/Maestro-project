"""
Use Case 1: Cyber Risk Pricing Model - EDA, Feature Engineering, and Pure Premium
================================================================================

Run from the project root:
    python code/01_use_case_1_eda.py

This single script performs the first pricing workflow:
  1. Load source cyber insurance datasets
  2. Run data quality checks
  3. Build policy-level modeling data
  4. Engineer and prune pricing features
  5. Generate important EDA visualizations
  6. Train a transparent frequency-severity template
  7. Run Monte Carlo simulation for VaR risk metrics
  8. Output pure premium and technical premium indications
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
VIS_DIR = ROOT / "outputs" / "eda_visuals"
MODEL_DIR = ROOT / "outputs" / "model_outputs"

MODELING_FILE = DATA_DIR / "07_modeling_dataset.csv"
FEATURE_FILE = DATA_DIR / "09_cyber_pricing_features.csv"
FEATURE_DICTIONARY_FILE = DATA_DIR / "09_feature_dictionary.csv"
EDA_DASHBOARD_FILE = VIS_DIR / "cyber_pricing_eda_dashboard.html"
INDICATION_FILE = MODEL_DIR / "pure_premium_indications.csv"
DIAGNOSTICS_FILE = MODEL_DIR / "model_diagnostics.txt"


NUMERIC_FEATURES = [
    "log_revenue",
    "exposure_size_score",
    "cyber_control_score",
    "control_gap_score",
    "vendor_control_pressure",
    "regulatory_findings_pressure",
    "high_sev_rate",
    "critical_operations_score",
    "payment_trading_flag",
    "coverage_structure_score",
    "limit_to_revenue",
    "retention_ratio",
    "prior_incident_score",
    "repeat_offender",
    "hybrid_cloud_flag",
    "core_banking_vendor_missing",
    "exposure_years",
]

CATEGORICAL_FEATURES = [
    "sub_sector",
    "region",
    "primary_regulator",
    "cloud_provider_primary",
    "vendor_pressure_band",
]


def safe_divide(numerator: pd.Series, denominator: pd.Series, fill_value: float = 0.0) -> pd.Series:
    result = numerator / denominator.replace(0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan).fillna(fill_value)


def zscore(series: pd.Series) -> pd.Series:
    clean = series.replace([np.inf, -np.inf], np.nan)
    std = clean.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return ((clean - clean.mean()) / std).fillna(0.0)


def minmax(series: pd.Series) -> pd.Series:
    clean = series.replace([np.inf, -np.inf], np.nan)
    low = clean.min()
    high = clean.max()
    if pd.isna(low) or pd.isna(high) or high == low:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return ((clean - low) / (high - low)).fillna(0.0)


def load_source_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("Loading datasets...")
    policies = pd.read_csv(DATA_DIR / "01_policies.csv", parse_dates=["inception_date", "expiration_date"])
    claims = pd.read_csv(DATA_DIR / "02_claims.csv", parse_dates=["loss_date"])
    findings = pd.read_csv(DATA_DIR / "03_regulatory_findings.csv", parse_dates=["exam_date"])
    responses = pd.read_csv(DATA_DIR / "04_questionnaire_responses.csv")
    systems = pd.read_csv(DATA_DIR / "05_system_recovery_profiles.csv")
    outages = pd.read_csv(DATA_DIR / "06_outage_events.csv", parse_dates=["outage_start_date"])

    print(f"  policies:  {len(policies):>6,} rows")
    print(f"  claims:    {len(claims):>6,} rows")
    print(f"  findings:  {len(findings):>6,} rows")
    print(f"  responses: {len(responses):>6,} rows")
    print(f"  systems:   {len(systems):>6,} rows")
    print(f"  outages:   {len(outages):>6,} rows")
    return policies, claims, findings, responses, systems, outages


def run_quality_checks(policies: pd.DataFrame, claims: pd.DataFrame) -> None:
    print("\n--- 1. Data quality checks ---")
    print(f"Policies missing premium data: {policies['premium_usd'].isna().sum()}")
    print(f"Duplicate policy_id rows: {policies['policy_id'].duplicated().sum()}")

    mismatch = claims[
        claims["bi_loss_usd"].fillna(0)
        + claims["non_bi_loss_usd"].fillna(0)
        != claims["gross_incurred_usd"]
    ]
    print(f"Claims where BI + Non-BI != Gross Loss: {len(mismatch)}")


def build_modeling_dataset(
    policies: pd.DataFrame,
    claims: pd.DataFrame,
    findings: pd.DataFrame,
) -> pd.DataFrame:
    print("\n--- 2. Building policy-level modeling dataset ---")
    
    import joblib
    try:
        nlp_model_path = ROOT / "code" / "models" / "severity_classifier_baseline.joblib"
        nlp_model = joblib.load(nlp_model_path)
        finding_texts = findings["finding_text"].astype(str).tolist()
        findings["severity_label"] = nlp_model.predict(finding_texts)
        probs = nlp_model.predict_proba(finding_texts)
        if "High" in nlp_model.classes_:
            high_idx = list(nlp_model.classes_).index("High")
            findings["high_sev_prob"] = probs[:, high_idx]
        else:
            findings["high_sev_prob"] = 0.0
    except Exception as e:
        print(f"Warning: NLP model not found or failed. Using ground truth labels. Error: {e}")
        findings["high_sev_prob"] = 0.0

    claim_agg = (
        claims.groupby("policy_id")
        .agg(
            n_claims=("claim_id", "count"),
            total_loss=("gross_incurred_usd", "sum"),
            bi_loss=("bi_loss_usd", "sum"),
        )
        .reset_index()
    )

    finding_agg = (
        findings.groupby("policy_id")
        .agg(
            n_findings=("finding_id", "count"),
            n_high_sev=("severity_label", lambda x: (x == "High").sum()),
            n_med_sev=("severity_label", lambda x: (x == "Medium").sum()),
            avg_high_sev_prob=("high_sev_prob", "mean")
        )
        .reset_index()
    )

    model_df = policies.merge(claim_agg, on="policy_id", how="left")
    model_df = model_df.merge(finding_agg, on="policy_id", how="left")
    model_df = model_df.fillna(
        {
            "n_claims": 0,
            "total_loss": 0,
            "bi_loss": 0,
            "n_findings": 0,
            "n_high_sev": 0,
            "n_med_sev": 0,
            "avg_high_sev_prob": 0.0,
        }
    )

    model_df["had_claim"] = (model_df["n_claims"] > 0).astype(int)
    model_df["log_revenue"] = np.log1p(model_df["revenue_mm"].clip(lower=0))

    model_df.to_csv(MODELING_FILE, index=False)
    print(f"Wrote modeling dataset: {MODELING_FILE}")
    print(f"Shape: {model_df.shape}")
    print(f"Claim rate: {model_df['had_claim'].mean():.1%}")
    return model_df


def engineer_pricing_features(model_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("\n--- 3. Feature engineering and pruning ---")
    df = model_df.copy()

    bool_cols = ["has_trading_desk", "processes_payments", "edr_deployed", "soc_24_7"]
    for col in bool_cols:
        df[col] = df[col].fillna(False).astype(int)

    df["core_banking_vendor_missing"] = df["core_banking_vendor"].isna().astype(int)
    df["core_banking_vendor"] = df["core_banking_vendor"].fillna("Unknown")
    df["custodial_aum_bn"] = df["custodial_aum_bn"].fillna(0.0)
    df["has_custodial_aum"] = (df["custodial_aum_bn"] > 0).astype(int)

    df["policy_term_days"] = (
        pd.to_datetime(df["expiration_date"], errors="coerce")
        - pd.to_datetime(df["inception_date"], errors="coerce")
    ).dt.days.fillna(365).clip(lower=1)
    df["exposure_years"] = df["policy_term_days"] / 365.25

    df["log_revenue"] = np.log1p(df["revenue_mm"].clip(lower=0))
    df["log_assets"] = np.log1p(df["assets_mm"].clip(lower=0))
    df["log_employees"] = np.log1p(df["employees"].clip(lower=0))

    # Merge highly correlated size variables into one modeling feature.
    df["exposure_size_score"] = (
        zscore(df["log_revenue"]) + zscore(df["log_assets"]) + zscore(df["log_employees"])
    ) / 3.0

    # Merge overlapping cyber-control variables.
    mfa_score = df["mfa_coverage_pct"].fillna(0).clip(0, 100) / 100.0
    nist_score = df["control_maturity_nist"].fillna(df["control_maturity_nist"].median()).clip(1, 5) / 5.0
    df["cyber_control_score"] = (
        0.40 * nist_score
        + 0.25 * mfa_score
        + 0.20 * df["edr_deployed"]
        + 0.15 * df["soc_24_7"]
    )
    df["control_gap_score"] = 1.0 - df["cyber_control_score"]

    # Combine vendor dependency with control maturity.
    df["vendor_control_pressure"] = safe_divide(
        df["n_third_party_vendors"].astype(float),
        df["control_maturity_nist"].fillna(0) + 0.1,
    )
    df["vendor_pressure_band"] = pd.cut(
        df["vendor_control_pressure"],
        bins=[-np.inf, 8, 14, 22, np.inf],
        labels=["Low", "Moderate", "High", "Extreme"],
    ).astype(str)

    # Combine findings count and severity mix (including NLP probabilities).
    df["high_sev_rate"] = safe_divide(df["n_high_sev"], df["n_findings"] + 1.0)
    df["regulatory_findings_pressure"] = (
        np.log1p(df["n_findings"])
        * (1.0 + df["high_sev_rate"] + df["avg_high_sev_prob"])
        * (1.0 + 0.25 * safe_divide(df["n_med_sev"], df["n_findings"] + 1.0))
    )

    # Combine operational criticality flags.
    df["critical_operations_score"] = (
        df["has_trading_desk"] + df["processes_payments"] + df["has_custodial_aum"]
    )
    df["payment_trading_flag"] = (
        (df["has_trading_desk"] == 1) & (df["processes_payments"] == 1)
    ).astype(int)

    # Coverage structure and prior incident signals.
    df["limit_to_revenue"] = safe_divide(df["limit_mm"], df["revenue_mm"])
    df["retention_ratio"] = safe_divide(df["retention_mm"], df["limit_mm"])
    df["coverage_structure_score"] = minmax(np.log1p(df["limit_mm"])) - minmax(df["retention_ratio"])
    df["prior_incident_score"] = np.log1p(df["prior_incidents_3yr"])
    df["repeat_offender"] = (df["prior_incidents_3yr"] >= 2).astype(int)
    df["hybrid_cloud_flag"] = (df["cloud_provider_primary"].eq("Hybrid")).astype(int)
    df["loss_ratio"] = safe_divide(df["total_loss"], df["premium_usd"])
    df["bi_share_of_loss"] = safe_divide(df["bi_loss"], df["total_loss"])

    retained_cols = [
        "policy_id",
        "insured_id",
        "sub_sector",
        "primary_regulator",
        "region",
        "policy_year",
        "cloud_provider_primary",
        "core_banking_vendor",
        "premium_usd",
        "n_claims",
        "had_claim",
        "total_loss",
        "bi_loss",
        "log_revenue",
        "exposure_size_score",
        "cyber_control_score",
        "control_gap_score",
        "vendor_control_pressure",
        "vendor_pressure_band",
        "regulatory_findings_pressure",
        "high_sev_rate",
        "critical_operations_score",
        "payment_trading_flag",
        "coverage_structure_score",
        "limit_to_revenue",
        "retention_ratio",
        "prior_incident_score",
        "repeat_offender",
        "hybrid_cloud_flag",
        "core_banking_vendor_missing",
        "exposure_years",
        "loss_ratio",
        "bi_share_of_loss",
        "limit_mm",      # Retained for Monte Carlo
        "retention_mm",  # Retained for Monte Carlo
    ]
    engineered = df[retained_cols].copy()

    feature_dictionary = pd.DataFrame(
        [
            ("exposure_size_score", "Merged log_revenue, log_assets, and log_employees."),
            ("cyber_control_score", "Merged NIST maturity, MFA coverage, EDR, and SOC."),
            ("control_gap_score", "Inverted cyber_control_score for risk loading."),
            ("vendor_control_pressure", "Merged vendor count and NIST maturity."),
            ("regulatory_findings_pressure", "Merged n_findings, n_high_sev, and n_med_sev."),
            ("critical_operations_score", "Merged trading desk, payments, and custodial AUM flags."),
            ("coverage_structure_score", "Merged limit and retention structure."),
            ("prior_incident_score", "Log-scaled prior incident count."),
            ("repeat_offender", "Flag for prior_incidents_3yr >= 2."),
            ("hybrid_cloud_flag", "Flags hybrid cloud due to observed loss differentiation."),
        ],
        columns=["feature", "engineering_note"],
    )

    engineered.to_csv(FEATURE_FILE, index=False)
    feature_dictionary.to_csv(FEATURE_DICTIONARY_FILE, index=False)
    print(f"Wrote engineered features: {FEATURE_FILE}")
    print(f"Wrote feature dictionary:  {FEATURE_DICTIONARY_FILE}")
    print(f"Pruned modeling shape: {engineered.shape}")
    return engineered, feature_dictionary


def generate_eda_outputs(df: pd.DataFrame) -> None:
    print("\n--- 4. EDA summaries and visualizations ---")
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    df["positive_total_loss"] = df["total_loss"].where(df["total_loss"] > 0)

    sector_summary = (
        df.groupby("sub_sector")
        .agg(
            policies=("policy_id", "count"),
            claim_rate=("had_claim", "mean"),
            avg_n_claims=("n_claims", "mean"),
            avg_total_loss=("total_loss", "mean"),
            median_positive_loss=("positive_total_loss", "median"),
            avg_premium=("premium_usd", "mean"),
            avg_log_revenue=("log_revenue", "mean"),
            avg_loss_ratio=("loss_ratio", "mean"),
        )
        .reset_index()
    )
    sector_summary.to_csv(VIS_DIR / "sector_claim_loss_premium_summary.csv", index=False)

    prior_summary = (
        df.groupby("repeat_offender")
        .agg(
            policies=("policy_id", "count"),
            claim_rate=("had_claim", "mean"),
            avg_n_claims=("n_claims", "mean"),
            avg_total_loss=("total_loss", "mean"),
            avg_premium=("premium_usd", "mean"),
        )
        .reset_index()
    )
    prior_summary["repeat_offender"] = prior_summary["repeat_offender"].map(
        {0: "0-1 prior incidents", 1: "2+ prior incidents"}
    )
    prior_summary.to_csv(VIS_DIR / "prior_incident_claim_summary.csv", index=False)

    control_summary = (
        df.assign(
            control_band=pd.cut(
                df["cyber_control_score"],
                bins=[0, 0.55, 0.70, 0.85, 1.01],
                labels=["Weak", "Developing", "Strong", "Excellent"],
            )
        )
        .groupby("control_band", observed=False)
        .agg(
            policies=("policy_id", "count"),
            claim_rate=("had_claim", "mean"),
            avg_total_loss=("total_loss", "mean"),
            avg_premium=("premium_usd", "mean"),
        )
        .reset_index()
    )
    control_summary.to_csv(VIS_DIR / "control_score_claim_loss_summary.csv", index=False)

    vendor_summary = (
        df.groupby("vendor_pressure_band")
        .agg(
            policies=("policy_id", "count"),
            claim_rate=("had_claim", "mean"),
            avg_total_loss=("total_loss", "mean"),
            avg_bi_share=("bi_share_of_loss", "mean"),
            avg_premium=("premium_usd", "mean"),
        )
        .reset_index()
    )
    vendor_summary.to_csv(VIS_DIR / "vendor_pressure_summary.csv", index=False)

    fig = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=(
            "Claim Count Distribution",
            "Positive Total Loss Distribution",
            "Claim Rate by Sub-Sector",
            "Average Total Loss by Sub-Sector",
            "Premium vs log_revenue",
            "Total Loss vs log_revenue",
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.10,
    )

    fig.add_trace(go.Histogram(x=df["n_claims"], marker_color="#3f7cac", name="n_claims"), row=1, col=1)

    severity_df = df[df["total_loss"] > 0].copy()
    severity_df["log10_total_loss"] = np.log10(severity_df["total_loss"])
    fig.add_trace(
        go.Histogram(x=severity_df["log10_total_loss"], nbinsx=40, marker_color="#d75f4b", name="total_loss"),
        row=1,
        col=2,
    )

    sector_claims = sector_summary.sort_values("claim_rate", ascending=False)
    fig.add_trace(
        go.Bar(x=sector_claims["sub_sector"], y=sector_claims["claim_rate"], marker_color="#52796f"),
        row=2,
        col=1,
    )

    sector_loss = sector_summary.sort_values("avg_total_loss", ascending=False)
    fig.add_trace(
        go.Bar(x=sector_loss["sub_sector"], y=sector_loss["avg_total_loss"], marker_color="#b85c38"),
        row=2,
        col=2,
    )

    sample_df = df.sample(min(len(df), 600), random_state=42)
    fig.add_trace(
        go.Scatter(
            x=sample_df["log_revenue"],
            y=sample_df["premium_usd"],
            mode="markers",
            marker=dict(
                size=7,
                color=sample_df["had_claim"],
                colorscale=[[0, "#3f7cac"], [1, "#d75f4b"]],
                opacity=0.65,
            ),
            name="premium",
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=sample_df["log_revenue"],
            y=sample_df["total_loss"],
            mode="markers",
            marker=dict(
                size=7,
                color=sample_df["had_claim"],
                colorscale=[[0, "#3f7cac"], [1, "#d75f4b"]],
                opacity=0.65,
            ),
            name="loss",
        ),
        row=3,
        col=2,
    )

    fig.update_xaxes(title_text="n_claims", row=1, col=1)
    fig.update_xaxes(title_text="log10(total_loss)", row=1, col=2)
    fig.update_yaxes(title_text="claim rate", row=2, col=1)
    fig.update_yaxes(title_text="avg total loss", row=2, col=2)
    fig.update_xaxes(title_text="log_revenue", row=3, col=1)
    fig.update_yaxes(title_text="premium_usd", row=3, col=1)
    fig.update_xaxes(title_text="log_revenue", row=3, col=2)
    fig.update_yaxes(title_text="total_loss", row=3, col=2)
    fig.update_layout(
        title_text="Cyber Pricing EDA: Claims, Loss, Premium, and Revenue",
        height=1200,
        showlegend=False,
        template="plotly_white",
    )
    fig.write_html(EDA_DASHBOARD_FILE)
    print(f"Wrote EDA dashboard: {EDA_DASHBOARD_FILE}")


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -35, 35)))


def train_test_split_mask(n_rows: int, test_size: float = 0.25, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random(n_rows) >= test_size


def build_design_matrix(
    df: pd.DataFrame,
    fit_columns: list[str] | None = None,
    scaler: tuple[pd.Series, pd.Series] | None = None,
) -> tuple[np.ndarray, list[str], tuple[pd.Series, pd.Series]]:
    numeric = df[NUMERIC_FEATURES].copy()
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    numeric = numeric.fillna(numeric.median(numeric_only=True))

    categoricals = pd.get_dummies(
        df[CATEGORICAL_FEATURES].fillna("Unknown"),
        columns=CATEGORICAL_FEATURES,
        drop_first=True,
        dtype=float,
    )

    features = pd.concat([numeric, categoricals], axis=1)
    if fit_columns is None:
        fit_columns = list(features.columns)
    else:
        features = features.reindex(columns=fit_columns, fill_value=0.0)

    if scaler is None:
        means = features.mean()
        stds = features.std(ddof=0).replace(0, 1.0)
    else:
        means, stds = scaler

    standardized = (features - means) / stds
    standardized.insert(0, "intercept", 1.0)
    return standardized.to_numpy(dtype=float), ["intercept"] + fit_columns, (means, stds)


def fit_logistic_regression(
    x: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.04,
    l2: float = 0.02,
    n_iter: int = 5_000,
) -> np.ndarray:
    weights = np.zeros(x.shape[1])
    weights[0] = np.log(y.mean() / (1 - y.mean()))

    for _ in range(n_iter):
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            pred = sigmoid(x @ weights)
            gradient = (x.T @ (pred - y)) / len(y)
        gradient[1:] += l2 * weights[1:] / len(y)
        weights -= learning_rate * gradient

    return weights


def fit_lognormal_severity(x: np.ndarray, losses: np.ndarray, ridge: float = 0.10) -> tuple[np.ndarray, float]:
    y = np.log1p(losses)
    penalty = ridge * np.eye(x.shape[1])
    penalty[0, 0] = 0.0
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        beta = np.linalg.solve(x.T @ x + penalty, x.T @ y)
        residuals = y - x @ beta
    sigma = float(np.sqrt(np.mean(residuals**2)))
    return beta, sigma


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.20) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    return {
        "accuracy": (tp + tn) / max(1, len(y_true)),
        "precision": tp / max(1, tp + fp),
        "recall": tp / max(1, tp + fn),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def run_frequency_severity_template(df: pd.DataFrame) -> pd.DataFrame:
    print("\n--- 5. Frequency-severity pure premium template ---")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    train_mask = train_test_split_mask(len(df), test_size=0.25, seed=42)
    train = df.loc[train_mask].reset_index(drop=True)
    test = df.loc[~train_mask].reset_index(drop=True)

    x_train, feature_names, train_scaler = build_design_matrix(train)
    x_test, _, _ = build_design_matrix(test, fit_columns=feature_names[1:], scaler=train_scaler)
    x_all, _, _ = build_design_matrix(df, fit_columns=feature_names[1:], scaler=train_scaler)

    y_train = train["had_claim"].to_numpy(dtype=float)
    y_test = test["had_claim"].to_numpy(dtype=float)

    from sklearn.linear_model import PoissonRegressor, GammaRegressor
    from sklearn.metrics import mean_poisson_deviance, mean_gamma_deviance
    
    # Train Frequency GLM (Poisson)
    print("\n[Actuarial GLM Frequency Model - Poisson]")
    freq_glm = PoissonRegressor(alpha=0.1, max_iter=1000)
    freq_glm.fit(x_train, y_train)
    
    train_freq = freq_glm.predict(x_train)
    test_freq = freq_glm.predict(x_test)
    all_freq = freq_glm.predict(x_all)
    
    # Extract Frequency Coefficients
    freq_coef = pd.DataFrame(list(zip(feature_names, freq_glm.coef_)), columns=['Feature', 'Frequency_Coef'])

    severity_train = train.loc[train["total_loss"] > 0].reset_index(drop=True)
    x_sev_train, sev_feature_names, severity_scaler = build_design_matrix(
        severity_train,
        fit_columns=feature_names[1:],
    )
    
    y_sev_train = severity_train["total_loss"].to_numpy(dtype=float)
    
    print("\n[Actuarial GLM Severity Model - Gamma]")
    sev_glm = GammaRegressor(alpha=0.1, max_iter=1000)
    sev_glm.fit(x_sev_train, y_sev_train)
    
    # Extract Severity Coefficients
    sev_coef = pd.DataFrame(list(zip(sev_feature_names, sev_glm.coef_)), columns=['Feature', 'Severity_Coef'])
    
    # Merge and Save Coefficients for the App
    coef_df = pd.merge(freq_coef, sev_coef, on='Feature', how='outer').fillna(0)
    coef_df.to_csv(MODEL_DIR / "glm_coefficients.csv", index=False)

    x_all_severity, _, _ = build_design_matrix(
        df,
        fit_columns=sev_feature_names[1:],
        scaler=severity_scaler,
    )
    all_expected_severity = sev_glm.predict(x_all_severity)

    test_positive = test["total_loss"] > 0
    if test_positive.any():
        x_sev_test, _, _ = build_design_matrix(
            test.loc[test_positive],
            fit_columns=sev_feature_names[1:],
            scaler=severity_scaler,
        )
        sev_test_pred = sev_glm.predict(x_sev_test)
        sev_mae = float(np.mean(np.abs(test.loc[test_positive, "total_loss"].to_numpy(dtype=float) - sev_test_pred)))
    else:
        sev_mae = np.nan

    indicated = df[
        [
            "policy_id",
            "sub_sector",
            "region",
            "policy_year",
            "premium_usd",
            "had_claim",
            "n_claims",
            "total_loss",
            "bi_loss",
            "log_revenue",
            "cyber_control_score",
            "vendor_control_pressure",
            "prior_incident_score",
            "limit_mm",      # Retained for Monte Carlo
            "retention_mm",  # Retained for Monte Carlo
        ]
    ].copy()
    indicated["predicted_claim_probability"] = all_freq
    indicated["expected_claim_frequency"] = all_freq * df["exposure_years"].clip(lower=0.01)
    indicated["expected_claim_severity"] = all_expected_severity
    indicated["pure_premium"] = indicated["expected_claim_frequency"] * indicated["expected_claim_severity"]

    expense_load = 0.25
    profit_load = 0.10
    cat_load = 0.12
    reinsurance_load = 0.08
    total_load = expense_load + profit_load + cat_load + reinsurance_load

    indicated["technical_premium_template"] = indicated["pure_premium"] / (1.0 - total_load)
    indicated["premium_gap"] = indicated["technical_premium_template"] - indicated["premium_usd"]
    indicated["indicated_to_current_ratio"] = indicated["technical_premium_template"] / indicated[
        "premium_usd"
    ].replace(0, np.nan)
    indicated["actual_loss_ratio"] = indicated["total_loss"] / indicated["premium_usd"].replace(0, np.nan)

    money_cols = [
        "expected_claim_severity",
        "pure_premium",
        "technical_premium_template",
        "premium_gap",
    ]
    for col in money_cols:
        indicated[col] = indicated[col].round(2)
    for col in [
        "predicted_claim_probability",
        "expected_claim_frequency",
        "indicated_to_current_ratio",
        "actual_loss_ratio",
    ]:
        indicated[col] = indicated[col].round(5)

    indicated.to_csv(INDICATION_FILE, index=False)

    action_threshold = 0.20
    train_metrics = binary_metrics(y_train, train_freq, threshold=action_threshold)
    test_metrics = binary_metrics(y_test, test_freq, threshold=action_threshold)

    diagnostics = f"""Cyber Frequency-Severity Pure Premium Template
================================================

Source features: {FEATURE_FILE}
Rows: {len(df):,}
Train rows: {len(train):,}
Test rows: {len(test):,}

Frequency model
---------------
Model: logistic regression implemented in NumPy, calibrated to portfolio base rate.
Target: had_claim
Diagnostic action threshold: {action_threshold:.2f}
Train claim rate: {train['had_claim'].mean():.4f}
Test claim rate: {test['had_claim'].mean():.4f}
Average test predicted probability: {test_freq.mean():.4f}
Train accuracy: {train_metrics['accuracy']:.4f}
Train precision: {train_metrics['precision']:.4f}
Train recall: {train_metrics['recall']:.4f}
Test accuracy: {test_metrics['accuracy']:.4f}
Test precision: {test_metrics['precision']:.4f}
Test recall: {test_metrics['recall']:.4f}

Severity model
--------------
Model: Actuarial Gamma GLM severity model on positive-loss claims.
Target: total_loss where total_loss > 0
Positive-loss train rows: {len(severity_train):,}
Positive-claim severity MAE on test: {sev_mae:,.2f}

Pricing formula
---------------
Expected Claim Frequency = predicted_claim_probability * exposure_years
Pure Premium = Expected Claim Frequency * Expected Claim Severity
Technical Premium Template = Pure Premium / (1 - total_load)

Loads used for template
-----------------------
Expense load: {expense_load:.0%}
Profit load: {profit_load:.0%}
Cyber catastrophe/systemic load: {cat_load:.0%}
Reinsurance load: {reinsurance_load:.0%}
Total load: {total_load:.0%}

Portfolio averages
------------------
Average current premium: {indicated['premium_usd'].mean():,.2f}
Average pure premium: {indicated['pure_premium'].mean():,.2f}
Average technical premium template: {indicated['technical_premium_template'].mean():,.2f}

Important caution
-----------------
This is a first-pass pricing template, not a production actuarial model.
Next iteration should add cross-validation, calibration plots, model monitoring,
and separate BI frequency/severity logic before using indications commercially.
"""
    DIAGNOSTICS_FILE.write_text(diagnostics, encoding="utf-8")

    print(f"Wrote indications: {INDICATION_FILE}")
    print(f"Wrote diagnostics: {DIAGNOSTICS_FILE}")
    print(f"Average pure premium: {indicated['pure_premium'].mean():,.2f}")
    print(f"Average technical premium template: {indicated['technical_premium_template'].mean():,.2f}")
    
    # ---------------------------------------------------------
    # EXPORT MODELS FOR STREAMLIT DASHBOARD INFERENCE
    # ---------------------------------------------------------
    import joblib
    import json
    
    export_dir = ROOT / "code" / "models"
    export_dir.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(freq_glm, export_dir / "freq_glm_model.joblib")
    joblib.dump(sev_glm, export_dir / "sev_glm_model.joblib")
    
    joblib.dump(train_scaler, export_dir / "model_scaler.joblib")
    
    with open(export_dir / "feature_columns.json", "w") as f:
        json.dump(feature_names[1:], f)  # Exclude 'intercept' from the required input columns

    print(f"Exported interactive models to: {export_dir}")
    
    return indicated


def run_monte_carlo_simulation(indicated: pd.DataFrame) -> None:
    print("\n--- 6. Executing Monte Carlo Risk Simulation ---")

    num_simulations = 1000
    np.random.seed(42)
    gamma_shape = 1.5 

    var_95_list = []
    var_99_list = []

    print(f"Simulating {num_simulations} years of cyber risk per policy...")

    for idx, row in indicated.iterrows():
        lam = row['expected_claim_frequency']
        mu = row['expected_claim_severity']
        limit = row['limit_mm'] * 1_000_000
        retention = row['retention_mm']
        
        # 1. Simulate Frequency (Poisson)
        simulated_counts = np.random.poisson(lam=lam, size=num_simulations)
        
        policy_aggregate_losses = np.zeros(num_simulations)
        
        # 2. Simulate Severity (Gamma) and Aggregate
        for i in range(num_simulations):
            n_claims_sim = simulated_counts[i]
            if n_claims_sim > 0:
                # Calculate scale parameter: mu = shape * scale => scale = mu / shape
                gamma_scale = mu / gamma_shape
                
                # Draw individual severities
                individual_losses = np.random.gamma(shape=gamma_shape, scale=gamma_scale, size=n_claims_sim)
                
                # Apply Policy Structure: max(0, Loss - Retention), capped at Limit
                net_payouts = np.minimum(limit, np.maximum(0, individual_losses - retention))
                
                # Sum aggregate loss for the year
                policy_aggregate_losses[i] = np.sum(net_payouts)
                
        # Calculate Risk Metrics
        var_95 = np.percentile(policy_aggregate_losses, 95)
        var_99 = np.percentile(policy_aggregate_losses, 99)
        
        var_95_list.append(var_95)
        var_99_list.append(var_99)

    indicated['simulated_VaR_95'] = var_95_list
    indicated['simulated_VaR_99'] = var_99_list

    # Update the Technical Premium to include a Capital Buffer based on VaR
    indicated['risk_adjusted_technical_premium'] = indicated['technical_premium_template'] + (indicated['simulated_VaR_99'] * 0.05)

    print("Monte Carlo metrics calculated successfully.")

    # Re-export the enriched predictions
    mc_output_path = MODEL_DIR / "pure_premium_indications_with_mc.csv"
    indicated.to_csv(mc_output_path, index=False)
    print(f"Enriched pricing file saved to: {mc_output_path}")


def print_next_steps() -> None:
    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print(
        """
1. Review outputs/eda_visuals/cyber_pricing_eda_dashboard.html.
2. Review data/09_feature_dictionary.csv to confirm feature merges.
3. Review outputs/model_outputs/model_diagnostics.txt for model sanity.
4. Compare current premium vs technical_premium_template in pure_premium_indications_with_mc.csv.
5. Next iteration: add cross-validation, model calibration plots, BI-specific severity, and XGBoost/GLM benchmarks.
"""
    )


def main() -> None:
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading existing modeling dataset...")
    model_df = pd.read_csv(MODELING_FILE)
    
    print("Loading findings and running NLP inference...")
    findings = pd.read_csv(DATA_DIR / "03_regulatory_findings.csv")
    import joblib
    import torch
    from transformers import DistilBertTokenizer, DistilBertModel
    import numpy as np
    
    try:
        nlp_model_path = ROOT / "code" / "models" / "severity_classifier_baseline.joblib"
        nlp_model = joblib.load(nlp_model_path)
        finding_texts = findings["finding_text"].astype(str).tolist()
        
        print("Loading DistilBERT for inference...")
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        bert_model = DistilBertModel.from_pretrained("distilbert-base-uncased").to(device)
        bert_model.eval()
        
        print("Extracting embeddings...")
        all_embeddings = []
        batch_size = 32
        with torch.no_grad():
            for i in range(0, len(finding_texts), batch_size):
                batch = finding_texts[i:i+batch_size]
                encoded = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
                output = bert_model(**encoded)
                cls_embeddings = output.last_hidden_state[:, 0, :].cpu().numpy()
                all_embeddings.append(cls_embeddings)
        finding_emb = np.vstack(all_embeddings)
        
        findings["severity_label"] = nlp_model.predict(finding_emb)
        probs = nlp_model.predict_proba(finding_emb)
        if "High" in nlp_model.classes_:
            high_idx = list(nlp_model.classes_).index("High")
            findings["high_sev_prob"] = probs[:, high_idx]
        else:
            findings["high_sev_prob"] = 0.0
    except Exception as e:
        print(f"Warning: NLP model not found or failed. Error: {e}")
        findings["high_sev_prob"] = 0.0
    
    finding_agg = (
        findings.groupby("policy_id")
        .agg(
            n_findings=("finding_id", "count"),
            n_high_sev=("severity_label", lambda x: (x == "High").sum()),
            n_med_sev=("severity_label", lambda x: (x == "Medium").sum()),
            avg_high_sev_prob=("high_sev_prob", "mean")
        )
        .reset_index()
    )
    
    cols_to_drop = ["n_findings", "n_high_sev", "n_med_sev", "avg_high_sev_prob"]
    model_df = model_df.drop(columns=[c for c in cols_to_drop if c in model_df.columns], errors='ignore')
    
    model_df = model_df.merge(finding_agg, on="policy_id", how="left")
    model_df = model_df.fillna({
        "n_findings": 0, "n_high_sev": 0, "n_med_sev": 0, "avg_high_sev_prob": 0.0
    })

    engineered, _ = engineer_pricing_features(model_df)
    generate_eda_outputs(engineered)
    indicated = run_frequency_severity_template(engineered)
    run_monte_carlo_simulation(indicated)
    print_next_steps()


if __name__ == "__main__":
    main()
    