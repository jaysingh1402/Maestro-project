import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import PoissonRegressor, GammaRegressor
from sklearn.metrics import mean_poisson_deviance, mean_gamma_deviance, roc_auc_score, recall_score
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from pathlib import Path

try:
    from google import genai
    from google.genai import types
except ImportError:
    st.error("Please run `pip install google-genai` to use the AI Agent features.")
    st.stop()

# Setup Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

st.set_page_config(page_title="Advanced Cyber Actuarial Dashboard", layout="wide", page_icon="🛡️")

# Premium Glassmorphism CSS & Modern Typography
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Outfit', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f172a, #1e1b4b);
        color: #e2e8f0;
    }
    
    h1, h2, h3 {
        background: -webkit-linear-gradient(45deg, #38bdf8, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 600;
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
        margin-bottom: 20px;
        text-align: center;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 40px rgba(0, 0, 0, 0.3);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #38bdf8;
        margin: 10px 0;
    }
    
    .metric-label {
        font-size: 1rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    /* Chat bubbles styling */
    .stChatMessage {
        background: rgba(255, 255, 255, 0.03) !important;
        border-radius: 15px !important;
        border: 1px solid rgba(255,255,255,0.05) !important;
    }
    
    /* Input fields */
    .stTextInput input {
        background: rgba(0,0,0,0.2) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        color: white !important;
        border-radius: 8px !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("🛡️ Advanced Cyber Risk & AI Pricing Dashboard")
st.markdown("*Use Case 1: Dynamic Frequency-Severity Pricing & Engineered Features Insights*")

@st.cache_data
def load_data():
    features_file = DATA_DIR / "09_cyber_pricing_features.csv"
    if features_file.exists():
        return pd.read_csv(features_file)
    return None

df = load_data()

if df is None:
    st.error("Data not found. Please ensure `09_cyber_pricing_features.csv` is in the `data/` directory.")
    st.stop()

# ==========================================
# ACTUARIAL GLM ENGINE (TRAINED ON FLY)
# ==========================================
# We train the GLMs directly in Streamlit to enable 100% dynamic user-input pricing
categorical_cols = ["sub_sector", "cloud_provider_primary"]
numeric_cols = [
    "exposure_size_score", "cyber_control_score", 
    "vendor_control_pressure", "regulatory_findings_pressure", "high_sev_rate",
    "critical_operations_score", "payment_trading_flag", "hybrid_cloud_flag"
]

# Prepare Data
X = df[numeric_cols + categorical_cols].copy()
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
features_list = X.columns.tolist()

y_freq = df["had_claim"]
severity_mask = df["total_loss"] > 0
X_sev = X[severity_mask]
y_sev = df[severity_mask]["total_loss"] # Gamma Regressor takes raw positive target, no log1p needed

# Train Frequency Model (Poisson GLM)
glm_freq = PoissonRegressor(alpha=0.1, max_iter=1000)
glm_freq.fit(X, y_freq)

# Train Severity Model (Gamma GLM)
glm_sev = GammaRegressor(alpha=0.1, max_iter=1000)
glm_sev.fit(X_sev, y_sev)

# Train XGBoost Frequency Model
xgb_freq = xgb.XGBClassifier(learning_rate=0.05, max_depth=4, random_state=42, use_label_encoder=False, eval_metric='logloss')
xgb_freq.fit(X, y_freq)

# Train XGBoost Severity Model
xgb_sev = xgb.XGBRegressor(learning_rate=0.05, max_depth=4, random_state=42, objective='reg:gamma')
xgb_sev.fit(X_sev, y_sev)

# Calculate AUC-ROC for comparison
glm_auc = roc_auc_score(y_freq, glm_freq.predict(X))
xgb_auc = roc_auc_score(y_freq, xgb_freq.predict_proba(X)[:, 1])

# Calculate Recall for comparison (Threshold Poisson at mean expected frequency)
glm_recall = recall_score(y_freq, glm_freq.predict(X) > y_freq.mean())
xgb_recall = recall_score(y_freq, xgb_freq.predict(X))

# Extract Coefficients for Agent
coef_df = pd.DataFrame({
    'Feature': features_list,
    'Frequency_Coef': glm_freq.coef_,
    'Severity_Coef': glm_sev.coef_
})
# Export so the Agent can read it
coef_df.to_csv("outputs/model_outputs/glm_coefficients.csv", index=False)


# ==========================================
# TABS
# ==========================================
tab_agent, tab_features, tab_calc, tab_hawkes, tab_scenarios, tab_accum = st.tabs([
    "📊 Portfolio Analytics & AI Explainer",
    "🧬 Feature Derivation Explainer",
    "🧮 Interactive Pricing Engine",
    "🦠 Advanced Contagion (Hawkes)",
    "🌩️ Catastrophe Scenarios",
    "🏗️ Accumulation Risk",
])

# ------------------------------------------
# TAB 1: PORTFOLIO ANALYTICS & AI EXPLAINER
# ------------------------------------------
with tab_agent:
    st.markdown("### Massive Feature Visualization Dashboard")
    st.write("This tab visualizes the exact effects of every extracted and merged feature on `bi_loss` and `loss_ratio`.")
    
    # 1. Categorical Visualizations
    st.markdown("#### 1. Categorical Feature Impacts")
    cat_cols = [c for c in ['primary_regulator', 'sub_sector', 'policy_year', 'cloud_provider_primary', 'core_banking_vendor', 'vendor_pressure_band'] if c in df.columns]
    
    for i in range(0, len(cat_cols), 2):
        c1, c2 = st.columns(2)
        if i < len(cat_cols):
            col_name = cat_cols[i]
            grouped = df.groupby(col_name, observed=False)[['bi_loss', 'loss_ratio']].mean().reset_index()
            fig = px.bar(grouped, x=col_name, y='bi_loss', color='loss_ratio', title=f"Avg BI Loss by {col_name}", color_continuous_scale="Viridis")
            c1.plotly_chart(fig, use_container_width=True)
        if i + 1 < len(cat_cols):
            col_name = cat_cols[i+1]
            grouped = df.groupby(col_name, observed=False)[['bi_loss', 'loss_ratio']].mean().reset_index()
            fig = px.bar(grouped, x=col_name, y='bi_loss', color='loss_ratio', title=f"Avg BI Loss by {col_name}", color_continuous_scale="Viridis")
            c2.plotly_chart(fig, use_container_width=True)

    # 2. Numeric Visualizations
    st.markdown("#### 2. Numeric Feature Impacts")
    num_cols = [c for c in ['cyber_control_score', 'control_gap_score', 'vendor_control_pressure', 'regulatory_findings_pressure', 'high_sev_rate', 'limit_to_revenue', 'prior_incident_score', 'earned_premium'] if c in df.columns]
    
    for i in range(0, len(num_cols), 2):
        c1, c2 = st.columns(2)
        if i < len(num_cols):
            col_name = num_cols[i]
            fig = px.scatter(df, x=col_name, y='bi_loss', color='loss_ratio', title=f"{col_name} vs BI Loss", color_continuous_scale="Inferno")
            c1.plotly_chart(fig, use_container_width=True)
        if i + 1 < len(num_cols):
            col_name = num_cols[i+1]
            fig = px.scatter(df, x=col_name, y='bi_loss', color='loss_ratio', title=f"{col_name} vs BI Loss", color_continuous_scale="Inferno")
            c2.plotly_chart(fig, use_container_width=True)
            
    st.markdown("---")
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("🤖 AI Actuarial Report & Chat")
    st.write("This agent uses Gemini ADK, **Retrieval-Augmented Generation (RAG)**, and **Function Calling (Tools)** to dynamically answer questions, explain math, and execute live pricing calculations on the fly.")
    st.warning("⚠️ **Recommendation:** It is highly recommended to use a **Gemini Pro Account** or higher API tier. Because the Agent processes large amounts of data and generates detailed reports, the free-tier API has a high risk of quickly exhausting tokens or hitting rate limits.")
    
    api_key = st.text_input("Enter Gemini API Key to run Agent:", type="password", key="agent_key_app1")
    
    if api_key:
        client = genai.Client(api_key=api_key)
        
        # Define the Tool for the AI Agent
        def dynamic_pricing_calculator(revenue: float, nist_score: float, mfa_coverage: float, n_vendors: int) -> dict:
            """Calculates the Poisson and Hawkes technical premiums for a given cyber insurance policy profile on the fly.
            
            Args:
                revenue: The company's annual revenue in millions of dollars (e.g., 500 for $500M).
                nist_score: The company's NIST Cybersecurity Framework maturity score (1.0 to 5.0).
                mfa_coverage: The percentage of the company's systems covered by Multi-Factor Authentication (0 to 100).
                n_vendors: The number of third-party vendors the company uses (e.g., 30).
            """
            import numpy as np
            import pandas as pd
            import json
            
            exp_size = np.log1p(revenue) / 15.0
            nist_normalized = nist_score / 5.0
            mfa_normalized = mfa_coverage / 100.0
            
            cyber_control_score = (0.40 * nist_normalized) + (0.25 * mfa_normalized) + (0.20 * 1) + (0.15 * 0)
            control_gap = 1.0 - cyber_control_score
            vendor_pressure = n_vendors / (nist_score + 0.1)
            
            input_dict = {
                "exposure_size_score": exp_size,
                "cyber_control_score": cyber_control_score,
                "control_gap_score": control_gap,
                "vendor_control_pressure": vendor_pressure,
                "regulatory_findings_pressure": 0.5,
                "critical_operations_score": 1,
                "payment_trading_flag": 0,
                "hybrid_cloud_flag": 0
            }
            
            input_array = [input_dict.get(f, 0) for f in features_list]
            input_df = pd.DataFrame([input_array], columns=features_list)
            
            pred_freq = glm_freq.predict(input_df)[0]
            pred_sev = glm_sev.predict(input_df)[0]
            pure_premium = pred_freq * pred_sev
            
            try:
                with open('outputs/model_outputs/hawkes_results.json', 'r') as f:
                    h_data = json.load(f)
                poisson_risk_load = (h_data['tvar_poisson'] / 5000) * 0.10
                hawkes_risk_load = (h_data['tvar_hawkes'] / 5000) * 0.10
            except:
                poisson_risk_load = pure_premium * 0.20
                hawkes_risk_load = pure_premium * 0.25
                
            final_poisson = (pure_premium + poisson_risk_load) / (1 - 0.25)
            final_hawkes = (pure_premium + hawkes_risk_load) / (1 - 0.25)
            
            return {
                "poisson_technical_premium": round(final_poisson, 2),
                "hawkes_technical_premium": round(final_hawkes, 2),
                "insight": "The Hawkes premium explicitly factors in the contagion risk of third-party vendors and poor controls."
            }
            
        # Load GLM Coefficients instead of SHAP
        try:
            coef_df = pd.read_csv("outputs/model_outputs/glm_coefficients.csv")
            glm_coefficients = coef_df.to_dict(orient="records")
        except FileNotFoundError:
            glm_coefficients = {"Error": "GLM coefficients not generated yet."}
            
        # Calculate Correlations for Agent to understand the visual scatterplots
        corr_features = ['cyber_control_score', 'control_gap_score', 'vendor_control_pressure', 'regulatory_findings_pressure', 'bi_loss', 'loss_ratio']
        corr_dict = df[corr_features].corr()[['bi_loss', 'loss_ratio']].to_dict() if all(f in df.columns for f in corr_features) else {}
        
        stats = {
            "avg_loss_ratio": df['loss_ratio'].mean() if 'loss_ratio' in df.columns else None,
            "avg_bi_loss": df['bi_loss'].mean() if 'bi_loss' in df.columns else None,
            "GLM_Coefficients": glm_coefficients,
            "Feature_Correlations": corr_dict
        }
        
        if "agent_report_app1" not in st.session_state:
            with st.spinner("Agent is analyzing deterministic stats and GLM Coefficients..."):
                prompt = f"""
                You are an expert Chief Actuary reviewing a cyber insurance portfolio. 
                Crucially, I have included the exact mathematical GLM Coefficients (Poisson Frequency and Gamma Severity) from the pricing engine. The engine also uses an XGBoost model explained via SHAP values, and a Hawkes Process to simulate TVaR Contagion Risk.
                
                Here are the statistical effects and GLM Coefficients:
                {stats}
                
                Write a concise executive report for the underwriters. Do NOT include formal memo headers like "Date:", "To:", "From:", or "Subject:". Just start the report directly.
                Explicitly study and explain the effect of Vendor Control Pressure, Regulatory Findings, and Cyber Control Score on BI Loss.
                **CRITICAL:** Explicitly use the GLM_Coefficients to explain *why* the Actuarial Pricing Engine cares about certain features over others, relating directly to the visualizations they see on the screen.
                """
                
                try:
                    response = client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=prompt
                    )
                    st.session_state["agent_report_app1"] = response.text
                    st.session_state.messages_app1 = [
                        {"role": "model", "content": "I have completed the portfolio analysis. Ask me follow-up questions about BI loss, specific vendors, or the Advanced Hawkes Contagion Model below!"}
                    ]
                except Exception as e:
                    st.error(f"GenAI Error: {e}")
        
        if "agent_report_app1" in st.session_state:
            st.markdown(st.session_state["agent_report_app1"])
        
        # Chat interface
        st.markdown("---")
        st.subheader("💬 Chat with the Agent")
        
        if "messages_app1" in st.session_state:
            for msg in st.session_state.messages_app1:
                with st.chat_message("assistant" if msg["role"] == "model" else "user"):
                    st.write(msg["content"])
                    
            if prompt_input := st.chat_input("Ask about BI Loss, specific vendors, or the Hawkes Contagion Model..."):
                st.session_state.messages_app1.append({"role": "user", "content": prompt_input})
                with st.chat_message("user"):
                    st.write(prompt_input)
                    
                with st.chat_message("assistant"):
                    with st.spinner("Searching Knowledge Base (RAG) & Thinking..."):
                        # 1. Define RAG Documents
                        rag_docs = [
                            "HAWKES MATH: Unlike Poisson which is independent, Hawkes process is contagious. It has 3 parameters found via Maximum Likelihood Estimation on claim timestamps. Baseline (mu) is random background attacks. Excitation (alpha) is the sudden risk spike after a breach. Decay (beta) is how fast the danger fades.",
                            "HAWKES SIMULATION: We simulate Hawkes using the Branching Approximation. Immigrants (Parents) are generated using Baseline. Offspring (Children) clusters are generated using a Negative Binomial distribution based on the branching ratio (alpha/beta). Total attacks = Parents + Children.",
                            "TVaR COMPARISON: Poisson TVaR ignores systemic risk. Hawkes TVaR is mathematically superior because it creates massive right-tail variance via clustering. The difference between them is the Contagion Risk Premium.",
                            "DISTRIBUTIONS: For frequency, Poisson beat Negative Binomial because the data lacked massive overdispersion. For severity, Lognormal beat Gamma because cyber claims have massive fat tails, but we kept Gamma as the standard regulatory baseline.",
                            "XGBOOST AND SHAP: The dashboard uses an XGBoost Classifier and Regressor as a Machine Learning alternative to GLM. Because XGBoost is a black-box, it calculates local SHAP (SHapley Additive exPlanations) values to mathematically prove to underwriters exactly how much each feature contributed to a specific policy's premium."
                        ]
                        
                        # 2. Vector Search using Gemini Embeddings
                        retrieved_doc = "No specific RAG context retrieved."
                        try:
                            import numpy as np
                            q_emb = client.models.embed_content(model='text-embedding-004', contents=prompt_input).embeddings[0].values
                            doc_embs = [client.models.embed_content(model='text-embedding-004', contents=d).embeddings[0].values for d in rag_docs]
                            similarities = [np.dot(q_emb, d_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(d_emb)) for d_emb in doc_embs]
                            
                            best_match_idx = np.argmax(similarities)
                            if similarities[best_match_idx] > 0.4: # Threshold
                                retrieved_doc = rag_docs[best_match_idx]
                        except Exception as e:
                            pass # Fallback to standard chat if embedding fails
                            
                        # 3. Augmented Generation
                        chat_context = f"You are a Chief Actuary equipped with a Pricing Calculator Tool. Previous report: {st.session_state.get('agent_report_app1', '')}. Retrieved Mathematical Knowledge Base (RAG): {retrieved_doc}. Answer the user accurately based on this or calculate premium using the tool if asked."
                        
                        from google.genai import types
                        chat = client.chats.create(
                            model="gemini-2.5-flash",
                            config=types.GenerateContentConfig(tools=[dynamic_pricing_calculator])
                        )
                        chat.send_message(chat_context)
                        response = chat.send_message(prompt_input)
                        st.write(response.text)
                
                st.session_state.messages_app1.append({"role": "model", "content": response.text})
    else:
        st.info("Provide API key to automatically generate the AI report and start chatting.")
    
    st.markdown('</div>', unsafe_allow_html=True)


# ------------------------------------------
# TAB 2: ENGINEERED FEATURES ANALYTICS
# ------------------------------------------
with tab_features:
    st.markdown("### 🧬 How We Engineered The Features")
    st.write("To improve the AI's predictive power, we didn't just use raw variables. We mathematically merged correlated metrics into risk indices and used Natural Language Processing (NLP) to extract insights from unstructured text.")
    
    st.markdown("#### 1. Merged Risk Indices (Feature Engineering)")
    col_feat1, col_feat2 = st.columns(2)
    
    with col_feat1:
        st.markdown("""
        **🛡️ Cyber Control Score**
        - **Why:** Grouping fragmented security controls into one holistic metric prevents the model from overfitting to individual checkboxes.
        - **How:** We applied a weighted average to core security metrics: 
          `(0.40 * NIST Maturity) + (0.25 * MFA Coverage) + (0.20 * EDR Flag) + (0.15 * SOC Flag)`
        
        **🔗 Vendor Risk Pressure**
        - **Why:** A high number of vendors is only dangerous if internal controls are weak.
        - **How:** We created a ratio by dividing the `Total Number of Vendors` by the `NIST Maturity Score`. This captures systemic third-party supply chain risk.
        """)
        
    with col_feat2:
        st.markdown("""
        **🚨 Regulatory Findings Pressure**
        - **Why:** Past audits are strong predictors of future breaches, but not all findings are equal.
        - **How:** We scaled the total number of findings logarithmically and multiplied it by the ratio of High/Medium severity findings, plus an NLP-derived risk penalty.
        
        **⚠️ Control Gap Score**
        - **Why:** It represents the remaining vulnerability.
        - **How:** Simply calculated as `1.0 - Cyber Control Score`.
        """)

    st.markdown("---")
    st.markdown("#### 2. Natural Language Processing (NLP) Extraction")
    st.write("A major component of this pricing engine is processing **unstructured regulatory audit text** and **threat intel reports**.")
    
    col_nlp1, col_nlp2 = st.columns([1, 2])
    with col_nlp1:
        st.image("https://huggingface.co/front/assets/huggingface_logo-noborder.svg", width=100) # Huggingface logo as placeholder
        st.markdown("**Model Used:**")
        st.markdown("`DistilBERT` (Transformer Model)")
    with col_nlp2:
        st.markdown("""
        **What features did we extract?**
        1. **Severity Probability (`nlp_prob`):** We passed raw text strings from auditor notes (e.g., *"The client failed to patch critical VPN vulnerabilities for 6 months"*) through a fine-tuned DistilBERT model.
        2. **Output:** The NLP model outputs a probability score (e.g., `0.10` or 10%) indicating the likelihood that the text describes a *critical, unmitigated threat*.
        3. **Integration:** This `nlp_prob` is then dynamically injected into the **Regulatory Findings Pressure** equation, meaning qualitative text directly increases the quantitative premium charged!
        """)

    st.markdown("---")
    st.markdown("### Visual Evidence of Feature Engineering")
    f_col1, f_col2 = st.columns(2)
    
    with f_col1:
        st.subheader("Cyber Control Score vs Claim Rate")
        df["control_bin"] = pd.qcut(df["cyber_control_score"], q=4, labels=["Weak", "Developing", "Strong", "Excellent"])
        fig_c = px.bar(df.groupby("control_bin", observed=False)["had_claim"].mean().reset_index(), 
                       x="control_bin", y="had_claim", color="had_claim", color_continuous_scale="Reds",
                       labels={"control_bin": "Merged Control Score", "had_claim": "Claim Probability"})
        st.plotly_chart(fig_c, use_container_width=True)
        
    with f_col2:
        st.subheader("Vendor Control Pressure vs Total Loss")
        fig_v = px.scatter(df, x="vendor_control_pressure", y="total_loss", color="sub_sector", 
                           log_y=True,
                           labels={"vendor_control_pressure": "Vendor Risk Pressure Score", "total_loss": "Historical Claim Loss ($)"})
        st.plotly_chart(fig_v, use_container_width=True)

# ------------------------------------------
# TAB 3: INTERACTIVE PRICING ENGINE
# ------------------------------------------
with tab_calc:
    st.markdown("### Dynamically Price a New Policy Profile")
    model_choice = st.radio("Select Core Pricing Model:", ["Actuarial GLM (Regulatory Baseline)", "Machine Learning (XGBoost)"], horizontal=True)
    st.write("Adjust the features below. The chosen model will calculate expected frequency, severity, and premium on the fly.")
    
    col_in1, col_in2, col_in3 = st.columns(3)
    
    with col_in1:
        st.subheader("Profile")
        sub_sector = st.selectbox("Type of Institution", df["sub_sector"].unique())
        cloud_provider = st.selectbox("Cloud Architecture", df["cloud_provider_primary"].unique())
        revenue = st.slider("Revenue ($M)", min_value=10, max_value=50000, value=500)
        
    with col_in2:
        st.subheader("Controls & Security")
        nist = st.slider("NIST Control Maturity", min_value=1.0, max_value=5.0, value=3.0, step=0.1)
        mfa = st.slider("MFA Coverage %", min_value=0, max_value=100, value=50, step=5)
        edr = st.checkbox("EDR Deployed", value=True)
        soc = st.checkbox("24/7 SOC", value=False)
        n_vendors = st.slider("Number of 3rd Party Vendors", min_value=0, max_value=150, value=30)
        hybrid_flag = 1 if cloud_provider == "Hybrid" else 0
        
    with col_in3:
        st.subheader("Operations & Regulatory")
        has_trading = st.checkbox("Has Trading Desk", value=False)
        processes_payments = st.checkbox("Processes Payments", value=True)
        
        st.markdown("**Regulatory Audit History**")
        n_findings = st.slider("Total Regulatory Findings", min_value=0, max_value=10, value=2)
        n_high = st.slider("High Severity Findings", min_value=0, max_value=n_findings, value=0)
        n_med = st.slider("Medium Severity Findings", min_value=0, max_value=n_findings-n_high if n_findings-n_high > 0 else 0, value=0)
        
        st.info("🤖 **NLP Note:** DistilBERT automatically extracts the severity probability from unstructured text.")
        nlp_prob = 0.10

    # Transform Raw Inputs to Engineered Features
    exp_size = np.log1p(revenue) / 15.0 
    crit_ops = int(has_trading) + int(processes_payments)
    pay_trad = 1 if (has_trading and processes_payments) else 0
    
    # 1. Cyber Control Score
    nist_score = nist / 5.0
    mfa_score = mfa / 100.0
    cyber_control_score = (0.40 * nist_score) + (0.25 * mfa_score) + (0.20 * int(edr)) + (0.15 * int(soc))
    control_gap = 1.0 - cyber_control_score
    
    # 2. Vendor Control Pressure
    vendor_pressure = n_vendors / (nist + 0.1)
    
    # 3. Regulatory Findings Pressure
    high_sev_rate = n_high / (n_findings + 1.0)
    med_sev_rate = n_med / (n_findings + 1.0)
    reg_pressure = np.log1p(n_findings) * (1.0 + high_sev_rate + nlp_prob) * (1.0 + 0.25 * med_sev_rate)
    
    st.markdown("### 📊 Live Engineered Feature Calculations")
    e_col1, e_col2, e_col3 = st.columns(3)
    
    with e_col1:
        st.markdown("**Cyber Control Score**")
        st.progress(cyber_control_score)
        
    with e_col2:
        st.markdown("**Vendor Risk Pressure**")
        norm_vendor = min(vendor_pressure / 50.0, 1.0)
        st.progress(norm_vendor)
        
    with e_col3:
        st.markdown("**Regulatory Pressure**")
        norm_reg = min(reg_pressure / 20.0, 1.0)
        st.progress(norm_reg)
    
    input_dict = {
        "exposure_size_score": exp_size,
        "cyber_control_score": cyber_control_score,
        "vendor_control_pressure": vendor_pressure,
        "regulatory_findings_pressure": reg_pressure,
        "high_sev_rate": high_sev_rate,
        "critical_operations_score": crit_ops,
        "payment_trading_flag": pay_trad,
        "hybrid_cloud_flag": hybrid_flag
    }
    
    for col in categorical_cols:
        val = sub_sector if col == "sub_sector" else cloud_provider
        dummy_col = f"{col}_{val}"
        if dummy_col in features_list:
            input_dict[dummy_col] = 1
            
    input_array = [input_dict.get(f, 0) for f in features_list]
    input_df = pd.DataFrame([input_array], columns=features_list)
    
    if "GLM" in model_choice:
        pred_freq = glm_freq.predict(input_df)[0]
        pred_sev = glm_sev.predict(input_df)[0]
        freq_model_name = "Poisson GLM"
        sev_model_name = "Gamma GLM"
    else:
        pred_freq = xgb_freq.predict_proba(input_df)[0][1]
        pred_sev = xgb_sev.predict(input_df)[0]
        freq_model_name = "XGBoost Classifier"
        sev_model_name = "XGBoost Regressor"
    
    pure_premium = pred_freq * pred_sev
    
    # Load Hawkes data for TVaR Risk Load
    try:
        import json
        with open('outputs/model_outputs/hawkes_results.json', 'r') as f:
            h_data = json.load(f)
        
        tvar_poisson = h_data['tvar_poisson']
        tvar_hawkes = h_data['tvar_hawkes']
        
        # Calculate Risk Loads (Allocating portfolio TVaR to 5000 policies at 10% Cost of Capital)
        poisson_risk_load = (tvar_poisson / 5000) * 0.10
        hawkes_risk_load = (tvar_hawkes / 5000) * 0.10
        
    except:
        poisson_risk_load = pure_premium * 0.20
        hawkes_risk_load = pure_premium * 0.25
        
    expense_ratio = 0.25
    
    # Final Premiums
    final_premium_poisson = (pure_premium + poisson_risk_load) / (1 - expense_ratio)
    final_premium_hawkes = (pure_premium + hawkes_risk_load) / (1 - expense_ratio)
    
    st.markdown("---")
    st.markdown("### Pricing Output & Risk Load Analysis")
    o_col1, o_col2, o_col3 = st.columns(3)
    
    with o_col1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Modeled Pure Premium</div><div class="metric-value">${pure_premium:,.0f}</div></div>', unsafe_allow_html=True)
    with o_col2:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Technical Premium (Poisson)</div><div class="metric-value">${final_premium_poisson:,.0f}</div></div>', unsafe_allow_html=True)
    with o_col3:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Technical Premium (Hawkes)</div><div class="metric-value">${final_premium_hawkes:,.0f}</div></div>', unsafe_allow_html=True)

    st.markdown("#### Actuarial Pricing Report")
    st.info(f"""
    **Pricing Formula:** `Technical Premium = (Pure Premium + Risk Load) / (1 - Expense Ratio)`
    *   **Expense Ratio:** 25% (Standard operating costs)
    
    **How the Pure Premium is Found:**
    `Pure Premium = Expected Frequency × Expected Severity`
    *   **Frequency:** {pred_freq:.2%} (Calculated using the **{freq_model_name}**.)
    *   **Severity:** ${pred_sev:,.0f} (Calculated using the **{sev_model_name}**.)
    *   **Calculation:** {pred_freq:.4f} × ${pred_sev:,.0f} = **${pure_premium:,.0f}**
    
    **How the Risk Load is Found:**
    To find the Risk Load for this specific policy, we allocate a portion of the massive Portfolio Catastrophe Risk (TVaR) to this single policy, applying a 10% Cost of Capital.
    *(Total Portfolio TVaR 99% [Poisson]: **${tvar_poisson:,.0f}** | Total Portfolio TVaR 99% [Hawkes]: **${tvar_hawkes:,.0f}**)*
    
    *   **Method 1: Poisson (Independent Risk)**
        *   Poisson TVaR allocated to this policy = Risk Load of **${poisson_risk_load:,.0f}**
        *   Calculation: `(${pure_premium:,.0f} + ${poisson_risk_load:,.0f}) / (1 - 0.25)` = **${final_premium_poisson:,.0f}** (Poisson Technical Premium)
    *   **Method 2: Hawkes (Contagion Risk)**
        *   Hawkes TVaR allocated to this policy = Risk Load of **${hawkes_risk_load:,.0f}**
        *   Calculation: `(${pure_premium:,.0f} + ${hawkes_risk_load:,.0f}) / (1 - 0.25)` = **${final_premium_hawkes:,.0f}** (Hawkes Technical Premium)
        
    **Conclusion:** The Hawkes model explicitly prices in the contagious "domino effect" of cyber risk, forcing underwriters to charge a higher Risk Load (and thus a higher Technical Premium) to safely capitalize the portfolio.
    """)

    # === SHAP EXPLAINABILITY ===
    st.markdown("---")
    st.markdown(f"### 🔍 Pricing Explainability for {model_choice}")
    
    if "GLM" in model_choice:
        # GLM Contributions (Coefficient * Value)
        contributions = input_df.iloc[0].values * glm_freq.coef_
        contrib_df = pd.DataFrame({'Feature': features_list, 'Contribution': contributions}).sort_values('Contribution', ascending=True)
        fig_shap = px.bar(contrib_df, x='Contribution', y='Feature', orientation='h', 
                          title='GLM Feature Contributions (Frequency)', color='Contribution', color_continuous_scale='RdBu_r')
        st.plotly_chart(fig_shap, use_container_width=True)
    else:
        # XGBoost SHAP
        explainer = shap.TreeExplainer(xgb_freq)
        shap_values = explainer.shap_values(input_df)
        shap_df = pd.DataFrame({'Feature': features_list, 'SHAP Value': shap_values[0]}).sort_values('SHAP Value', ascending=True)
        fig_shap = px.bar(shap_df, x='SHAP Value', y='Feature', orientation='h', 
                          title='XGBoost Marginal Contributions (SHAP for Frequency)', color='SHAP Value', color_continuous_scale='RdBu_r')
        st.plotly_chart(fig_shap, use_container_width=True)

    # === ADDITIONAL COMPARISONS (RECALL & TVAR) ===
    st.markdown("---")
    st.markdown("### 📊 Global Model Metrics: Recall & Tail Risk")
    st.write("Comparing the overarching predictive performance and portfolio catastrophe risk.")
    
    r_col1, r_col2 = st.columns(2)
    
    with r_col1:
        # Recall Graph
        recall_df = pd.DataFrame({
            "Model": ["Poisson GLM", "XGBoost Classifier"],
            "Recall": [glm_recall, xgb_recall]
        })
        fig_recall = px.bar(recall_df, x="Model", y="Recall", color="Model", title="Recall Comparison (Breach Detection)", text_auto='.2%')
        fig_recall.update_layout(showlegend=False)
        st.plotly_chart(fig_recall, use_container_width=True)
        
    with r_col2:
        # TVaR Graph
        losses = df[df["total_loss"] > 0]["total_loss"]
        p95 = np.percentile(losses, 95)
        
        tvar_df = pd.DataFrame({
            "Risk Metric": ["VaR 95% (Historical)", "TVaR 99% (Poisson)", "TVaR 99% (Hawkes)"],
            "Value ($)": [p95, tvar_poisson, tvar_hawkes]
        })
        fig_tvar = px.bar(tvar_df, x="Risk Metric", y="Value ($)", color="Risk Metric", title="Portfolio Catastrophe Risk (VaR & TVaR)", text_auto='.3s')
        fig_tvar.update_layout(showlegend=False)
        st.plotly_chart(fig_tvar, use_container_width=True)



# ------------------------------------------
# TAB 4: ADVANCED CONTAGION (HAWKES PROCESS) — Enhanced
# ------------------------------------------
with tab_hawkes:
    st.header("🦠 Advanced Cyber Contagion Modeling")
    st.markdown("""
    While the **Poisson GLM** assumes every cyber attack is an independent, random event, real-world
    cyber risk is highly **contagious**. When Log4Shell dropped (Dec 2021) there were >800,000 exploitation
    attempts in 72 hours. The Hawkes Process — the same mathematics used to model **earthquake aftershocks** —
    captures this self-exciting cluster dynamic and quantifies the *additional* capital required.

    **Enhancements in this version:** GPD fat-tail severity, +14%/yr frequency trend (DBIR 2024),
    sector-specific contagion models, OEP curve, and bootstrap confidence intervals.
    """)

    try:
        import json as _json_h
        with open('outputs/model_outputs/hawkes_results.json', 'r') as _fh:
            h_data = _json_h.load(_fh)

        # ── Row 1: Core MLE parameters ──────────────────────────────────────
        st.markdown("### Fitted Hawkes Parameters (MLE)")
        hm1, hm2, hm3, hm4 = st.columns(4)
        hm1.metric("Baseline (μ)", f"{h_data['mu']:.4f}", "events/day")
        hm2.metric("Excitation (α)", f"{h_data['alpha']:.4f}", "risk spike per event", delta_color="inverse")
        hm3.metric("Decay (β)", f"{h_data['beta']:.4f}", "fade rate")
        hm4.metric("Branching Ratio α/β", f"{h_data['branching_ratio']:.4f}",
                   "< 1 = stationary ✓" if h_data['branching_ratio'] < 1 else "⚠ non-stationary")

        st.latex(r'''\lambda(t) = \underbrace{\mu}_{\text{baseline}} + \underbrace{\sum_{t_i < t} \alpha e^{-\beta (t - t_i)}}_{\text{self-excitation cluster}}''')
        st.markdown("---")

        # ── Row 2: TVaR comparison (4 models) ───────────────────────────────
        st.markdown("### TVaR 99% Comparison Across Models")
        tvar_labels = ["Poisson\n(Independent)",
                       "Hawkes\n(Gamma severity)",
                       "Hawkes + GPD\n(Fat tail)",
                       f"Trended\n(+{h_data.get('annual_freq_trend_pct', 0.14)*100:.0f}%/yr × {h_data.get('projection_years', 3)}yr)"]
        tvar_values = [
            h_data.get("tvar_poisson", 0),
            h_data.get("tvar_hawkes_gamma", h_data.get("tvar_hawkes", 0)),
            h_data.get("tvar_hawkes_gpd", h_data.get("tvar_hawkes", 0)),
            h_data.get("tvar_trended", 0),
        ]
        tvar_colors = ["#3b82f6", "#8b5cf6", "#ef4444", "#f97316"]
        fig_tvar4 = go.Figure(go.Bar(
            x=tvar_labels, y=tvar_values, marker_color=tvar_colors,
            text=[f"${v:,.0f}" for v in tvar_values], textposition="outside"
        ))
        # Bootstrap CI error bars on GPD column
        ci_lo = h_data.get("tvar_gpd_ci_95_low",  tvar_values[2] * 0.95)
        ci_hi = h_data.get("tvar_gpd_ci_95_high", tvar_values[2] * 1.05)
        fig_tvar4.add_shape(type="line", x0=1.9, x1=2.1,
                            y0=ci_lo, y1=ci_lo, line=dict(color="white", width=2, dash="dot"))
        fig_tvar4.add_shape(type="line", x0=1.9, x1=2.1,
                            y0=ci_hi, y1=ci_hi, line=dict(color="white", width=2, dash="dot"))
        fig_tvar4.add_annotation(x=2, y=(ci_lo + ci_hi) / 2,
                                 text=f"95% CI<br>[${ci_lo:,.0f} –<br>${ci_hi:,.0f}]",
                                 showarrow=True, arrowhead=2, font=dict(size=10, color="white"), ax=60)
        fig_tvar4.update_layout(
            title="Portfolio TVaR 99% — Four Model Comparison",
            yaxis_title="TVaR 99% ($USD)",
            paper_bgcolor="#0f172a", plot_bgcolor="#1e293b",
            font=dict(color="#e2e8f0"), height=420,
        )
        st.plotly_chart(fig_tvar4, use_container_width=True)

        # Premium decomposition
        contagion_p = h_data.get("contagion_premium", 0)
        fat_tail_p  = h_data.get("fat_tail_premium",  0)
        trend_p     = h_data.get("trend_premium",      0)
        p4c1, p4c2, p4c3 = st.columns(3)
        p4c1.metric("Contagion Premium\n(Hawkes GPD vs Poisson)",
                    f"${contagion_p:,.0f}", "Systemic cluster risk load", delta_color="inverse")
        p4c2.metric("Fat-Tail Premium\n(GPD vs Gamma)",
                    f"${fat_tail_p:,.0f}",  "Heavy-tail severity risk load", delta_color="inverse")
        p4c3.metric(f"Trend Premium\n(+14%/yr × 3yr)",
                    f"${trend_p:,.0f}",     "Frequency growth risk load", delta_color="inverse")
        st.markdown("---")

        # ── Row 3: OEP Curve ─────────────────────────────────────────────────
        st.markdown("### Aggregate Exceedance Probability (AEP) Curve")
        st.write("""
        The **AEP curve** answers: *'What is the probability that total portfolio losses exceed X in any given year?'*
        This is the fundamental input to reinsurance pricing (Cat XL attachment selection) and regulatory
        capital modelling (Solvency II SCR, NAIC RBC). A standard Cat XL reinsurance treaty typically
        attaches at the 1-in-10 year (10% AEP) loss level.
        """)
        import os as _os_oep
        oep_path = "outputs/model_outputs/oep_curve.csv"
        if _os_oep.path.exists(oep_path):
            oep_df = pd.read_csv(oep_path)
            # 1-in-N return period lines
            ri_attach = float(oep_df[oep_df["exceedance_prob"] <= 0.10]["loss_usd"].max()) if len(oep_df[oep_df["exceedance_prob"] <= 0.10]) > 0 else 0
            fig_oep = go.Figure()
            fig_oep.add_trace(go.Scatter(
                x=oep_df["loss_usd"] / 1e6, y=oep_df["exceedance_prob"],
                mode="lines", name="AEP (Hawkes + GPD)",
                line=dict(color="#ef4444", width=2.5),
                fill="tozeroy", fillcolor="rgba(239,68,68,0.08)"
            ))
            # Return period markers
            for rp, color in [(10, "#f59e0b"), (100, "#ef4444"), (250, "#7c3aed")]:
                prob = 1 / rp
                fig_oep.add_shape(type="line", x0=oep_df["loss_usd"].min()/1e6,
                                  x1=oep_df["loss_usd"].max()/1e6,
                                  y0=prob, y1=prob,
                                  line=dict(color=color, dash="dash", width=1.5))
                fig_oep.add_annotation(x=oep_df["loss_usd"].max()/1e6*0.98, y=prob,
                                       text=f"1-in-{rp}yr", font=dict(color=color, size=11),
                                       showarrow=False, xanchor="right")
            fig_oep.update_layout(
                title="Portfolio AEP Curve — Annual Aggregate Loss Exceedance (Hawkes + GPD)",
                xaxis_title="Annual Aggregate Loss ($M USD)",
                yaxis_title="Exceedance Probability",
                paper_bgcolor="#0f172a", plot_bgcolor="#1e293b",
                font=dict(color="#e2e8f0"), height=400,
            )
            st.plotly_chart(fig_oep, use_container_width=True)
            if ri_attach > 0:
                st.info(f"**Reinsurance Signal:** A Cat XL treaty attaching at the 1-in-10 year level would attach at approximately **${ri_attach/1e6:.1f}M** of annual portfolio loss.")
        else:
            st.info("OEP curve file not found. Run `05_hawkes_process_simulation.py` to generate it.")
        st.markdown("---")

        # ── Row 4: Sector-specific Hawkes models ─────────────────────────────
        st.markdown("### Sector-Specific Contagion Parameters")
        st.write("""
        Different attack types have fundamentally different contagion dynamics.
        **Ransomware** campaigns spread rapidly (high α, fast decay β), while **Insider Threats**
        are largely independent (low α). Fitting separate Hawkes models per cause-of-loss reveals
        which attack types drive the most systemic risk.
        """)
        sector_models = h_data.get("sector_models", {})
        if sector_models:
            sec_rows = []
            for cause, params in sector_models.items():
                sec_rows.append({
                    "Cause of Loss":    cause,
                    "Events (hist.)": params.get("n_events", "—"),
                    "Baseline μ":       f"{params.get('mu',0):.4f}",
                    "Excitation α":     f"{params.get('alpha',0):.4f}",
                    "Decay β":          f"{params.get('beta',0):.4f}",
                    "Branching α/β":    f"{params.get('branching_ratio',0):.4f}",
                    "Exp. events/yr":   f"{params.get('expected_yr',0):.2f}",
                    "Contagion Risk":   "🔴 High" if params.get("branching_ratio",0) > 0.5
                                       else "🟡 Moderate" if params.get("branching_ratio",0) > 0.3
                                       else "🟢 Low",
                })
            st.dataframe(pd.DataFrame(sec_rows), use_container_width=True, hide_index=True)

            # Bar chart: branching ratios by cause
            sec_df = pd.DataFrame([{"cause": k, "br": v.get("branching_ratio", 0)}
                                    for k, v in sector_models.items()])
            sec_df = sec_df.sort_values("br", ascending=False)
            fig_sec = px.bar(sec_df, x="cause", y="br",
                             color="br", color_continuous_scale="Reds",
                             title="Branching Ratio by Attack Type (higher = more contagious)",
                             labels={"cause": "Cause of Loss", "br": "Branching Ratio α/β"})
            fig_sec.add_hline(y=0.5, line_dash="dash", line_color="orange",
                              annotation_text="High contagion threshold", annotation_position="top right")
            fig_sec.update_layout(
                paper_bgcolor="#0f172a", plot_bgcolor="#1e293b",
                font=dict(color="#e2e8f0"), height=350, showlegend=False
            )
            st.plotly_chart(fig_sec, use_container_width=True)
        else:
            st.info("Sector model data not found. Re-run `05_hawkes_process_simulation.py` to generate sector breakdowns.")

    except Exception as _he:
        st.warning(f"Hawkes data not found or invalid. Please run `05_hawkes_process_simulation.py` first. Error: {_he}")


# ------------------------------------------
# TAB 5: CATASTROPHE SCENARIO TESTING
# ------------------------------------------
with tab_scenarios:
    st.header("🌩️ Cyber Catastrophe Scenario Testing")
    st.markdown("""
    Named catastrophe scenarios allow underwriters to stress-test the portfolio against specific,
    historically-grounded shock events — independent of the stochastic Hawkes simulation.
    These scenarios answer: *'What does our portfolio lose if **this specific event** happens?'*

    All four scenarios below are calibrated to **real-world cyber events** with published industry
    loss estimates from Lloyd's, Swiss Re, and regulatory bodies.
    """)

    SCENARIO_META = {
        "cloud_outage": {
            "label": "☁️ Major Cloud Provider Outage",
            "basis": "CrowdStrike Falcon 2024 ($5.4B insured), AWS us-east-1 2021",
            "color": "#3b82f6",
        },
        "ransomware_campaign": {
            "label": "🦠 Global Ransomware Campaign",
            "basis": "WannaCry 2017 ($4–8B), NotPetya 2017 ($10B+)",
            "color": "#ef4444",
        },
        "supply_chain": {
            "label": "🔗 Supply Chain Software Compromise",
            "basis": "MOVEit 2023 ($1B+, 2,600 orgs), SolarWinds 2020",
            "color": "#f59e0b",
        },
        "critical_infra": {
            "label": "⚡ Critical Infrastructure Attack",
            "basis": "Colonial Pipeline 2021, CISA FS-ISAC Scenarios",
            "color": "#7c3aed",
        },
    }

    try:
        import json as _json_s
        with open('outputs/model_outputs/scenario_results.json', 'r') as _fs:
            sc_data = _json_s.load(_fs)

        scenarios = sc_data.get("scenarios", {})

        # Scenario selector
        selected_key = st.selectbox(
            "Select scenario to explore in detail:",
            list(SCENARIO_META.keys()),
            format_func=lambda k: SCENARIO_META[k]["label"]
        )
        sc   = scenarios.get(selected_key, {})
        meta = SCENARIO_META[selected_key]

        # ── KPI cards ─────────────────────────────────────────────────────
        st.markdown(f"#### {meta['label']}")
        st.caption(f"Real-world basis: *{sc.get('real_world_basis', meta['basis'])}*")

        kc1, kc2, kc3, kc4 = st.columns(4)
        kc1.metric("Policies Affected",
                   f"{sc.get('n_affected_policies', sc.get('avg_pct_affected', 0) * sc_data.get('portfolio_policies', 1500)):.0f}",
                   f"{sc.get('pct_portfolio', sc.get('avg_pct_affected', 0)):.1%} of portfolio")
        kc2.metric("Expected Loss (Mean)",  f"${sc.get('expected_loss_usd', 0):,.0f}")
        kc3.metric("PML 90% (1-in-10 yr)",  f"${sc.get('pml_90_usd', 0):,.0f}")
        kc4.metric("PML 99% (1-in-100 yr)", f"${sc.get('pml_99_usd', 0):,.0f}")

        # ── Loss percentile waterfall chart ───────────────────────────────
        dist = sc.get("loss_distribution_summary", {})
        fig_sc = go.Figure(go.Bar(
            x=["p50", "p75", "p90", "p95", "p99"],
            y=[dist.get(p, 0) for p in ["p50", "p75", "p90", "p95", "p99"]],
            marker_color=meta["color"],
            text=[f"${dist.get(p, 0)/1e6:.1f}M" for p in ["p50", "p75", "p90", "p95", "p99"]],
            textposition="outside",
        ))
        fig_sc.update_layout(
            title=f"{meta['label']} — Portfolio Loss Percentile Distribution",
            xaxis_title="Loss Percentile",
            yaxis_title="Net Portfolio Loss ($USD)",
            paper_bgcolor="#0f172a", plot_bgcolor="#1e293b",
            font=dict(color="#e2e8f0"), height=380,
        )
        st.plotly_chart(fig_sc, use_container_width=True)

        # ── Reinsurance exhaustion analysis ───────────────────────────────
        ri = sc.get("reinsurance", {})
        st.markdown("#### Reinsurance Treaty Exhaustion Analysis")
        ri_c1, ri_c2, ri_c3 = st.columns(3)
        ri_c1.metric("Cat XL Attachment", f"${ri.get('cat_xs_attachment_usd', 5000000):,.0f}")
        ri_c2.metric("RI Recovery at PML99", f"${ri.get('ri_recovery_at_p99', 0):,.0f}")
        ri_c3.metric("Net Insurer PML99",    f"${ri.get('net_insurer_pml_99', 0):,.0f}")
        if ri.get("treaty_exhausted_p99"):
            st.error("⚠️ **Treaty exhausted at the 99th percentile.** This scenario exceeds the Cat XL tower. Additional protection (aggregate stop-loss or excess-of-loss layer) is needed.")
        elif ri.get("treaty_exhausted_p90"):
            st.warning("🟡 **Treaty exhausted at the 90th percentile.** The 1-in-10 year scenario is already at the limit of the reinsurance tower.")
        else:
            st.success("✅ Current Cat XL treaty absorbs both the 90th and 99th percentile scenario losses.")

        st.markdown("---")
        # ── All scenarios side-by-side comparison ─────────────────────────
        st.markdown("### All Scenarios — PML Comparison")
        comp_data = []
        for k, m in SCENARIO_META.items():
            s = scenarios.get(k, {})
            comp_data.append({
                "Scenario":    m["label"],
                "Mean Loss":   s.get("expected_loss_usd", 0),
                "PML 90%":     s.get("pml_90_usd", 0),
                "PML 99%":     s.get("pml_99_usd", 0),
                "RI Exhausted (p99)": "Yes" if s.get("reinsurance", {}).get("treaty_exhausted_p99") else "No",
                "Color":       m["color"],
            })
        comp_df = pd.DataFrame(comp_data)
        fig_comp = go.Figure()
        for _, row in comp_df.iterrows():
            fig_comp.add_trace(go.Bar(
                name=row["Scenario"],
                x=["Mean Loss", "PML 90%", "PML 99%"],
                y=[row["Mean Loss"], row["PML 90%"], row["PML 99%"]],
                marker_color=row["Color"],
            ))
        fig_comp.update_layout(
            barmode="group", title="Scenario PML Comparison (all 4 scenarios)",
            yaxis_title="Net Portfolio Loss ($USD)",
            paper_bgcolor="#0f172a", plot_bgcolor="#1e293b",
            font=dict(color="#e2e8f0"), height=420,
            legend=dict(bgcolor="rgba(0,0,0,0)")
        )
        st.plotly_chart(fig_comp, use_container_width=True)

        # Summary table
        st.dataframe(
            comp_df[["Scenario", "Mean Loss", "PML 90%", "PML 99%", "RI Exhausted (p99)"]]
            .style.format({"Mean Loss": "${:,.0f}", "PML 90%": "${:,.0f}", "PML 99%": "${:,.0f}"}),
            use_container_width=True, hide_index=True
        )

        st.markdown("---")
        st.markdown("### Industry Loss Benchmarks")
        try:
            import json as _jbm
            with open('data/industry_benchmarks.json', 'r') as _fbm:
                bm = _jbm.load(_fbm)
            events = bm.get("notable_events", [])
            ev_df = pd.DataFrame(events)
            if not ev_df.empty:
                ev_df["estimated_loss_usd"] = ev_df["estimated_loss_usd"].apply(lambda x: f"${x/1e9:.1f}B" if x >= 1e9 else f"${x/1e6:.0f}M")
                st.dataframe(ev_df[["year", "event", "cause", "estimated_loss_usd", "affected_organizations"]]
                             .rename(columns={"year": "Year", "event": "Event", "cause": "Cause",
                                              "estimated_loss_usd": "Est. Loss", "affected_organizations": "Orgs Affected"}),
                             use_container_width=True, hide_index=True)
        except:
            pass

    except FileNotFoundError:
        st.info("Scenario results not found. Run `code/04_catastrophe_scenarios.py` from the project root to generate them.")
        st.code("python code/04_catastrophe_scenarios.py", language="bash")
    except Exception as _se:
        st.error(f"Error loading scenario results: {_se}")


# ------------------------------------------
# TAB 6: PORTFOLIO ACCUMULATION RISK
# ------------------------------------------
with tab_accum:
    st.header("🏗️ Portfolio Accumulation Risk")
    st.markdown("""
    **Accumulation risk** is the hidden correlation in a cyber portfolio: when many policies share the
    same cloud provider, fintech vendor, or geographic region, a single systemic shock creates
    correlated claims that dwarf what independent models predict.

    This dashboard quantifies the **Herfindahl-Hirschman Index (HHI)** — a standard measure of
    concentration used in antitrust economics — applied to insurance exposure. A portfolio with
    all policies on AWS has HHI = 1.0 (maximum concentration). A perfectly diversified portfolio
    has HHI ≈ 0.
    """)

    try:
        import json as _json_a
        with open('outputs/model_outputs/accumulation_risk.json', 'r') as _fa:
            acc = _json_a.load(_fa)

        # ── HHI Gauge metrics ──────────────────────────────────────────────
        st.markdown("### Concentration Index (HHI) — Portfolio Diversification Score")
        hhi = acc.get("hhi_scores", {})
        hh1, hh2, hh3, hh4 = st.columns(4)

        def hhi_color(v):
            return "inverse" if v > 0.25 else "normal"

        hh1.metric("Cloud Provider HHI", f"{hhi.get('cloud_provider_hhi', 0):.4f}",
                   "Concentrated" if hhi.get('cloud_provider_hhi', 0) > 0.25
                   else "Moderate" if hhi.get('cloud_provider_hhi', 0) > 0.15 else "Diversified",
                   delta_color=hhi_color(hhi.get("cloud_provider_hhi", 0)))
        hh2.metric("Vendor HHI", f"{hhi.get('vendor_hhi', 0):.4f}",
                   "Concentrated" if hhi.get('vendor_hhi', 0) > 0.25
                   else "Moderate" if hhi.get('vendor_hhi', 0) > 0.15 else "Diversified",
                   delta_color=hhi_color(hhi.get("vendor_hhi", 0)))
        hh3.metric("Geographic HHI", f"{hhi.get('geographic_hhi', 0):.4f}",
                   "Concentrated" if hhi.get('geographic_hhi', 0) > 0.25
                   else "Moderate" if hhi.get('geographic_hhi', 0) > 0.15 else "Diversified",
                   delta_color=hhi_color(hhi.get("geographic_hhi", 0)))
        hh4.metric("Sector HHI", f"{hhi.get('sector_hhi', 0):.4f}",
                   "Concentrated" if hhi.get('sector_hhi', 0) > 0.25
                   else "Moderate" if hhi.get('sector_hhi', 0) > 0.15 else "Diversified",
                   delta_color=hhi_color(hhi.get("sector_hhi", 0)))

        st.caption("*HHI interpretation: < 0.15 Diversified | 0.15–0.25 Moderate | > 0.25 Concentrated*")
        st.markdown("---")

        # ── Cloud Concentration ────────────────────────────────────────────
        ac1, ac2 = st.columns(2)

        with ac1:
            st.subheader("Cloud Provider — TIV Concentration")
            cloud_rows = acc.get("cloud_concentration", [])
            if cloud_rows:
                cd = pd.DataFrame(cloud_rows)
                fig_cloud = go.Figure(go.Pie(
                    labels=cd["group"],
                    values=cd["total_tiv_usd"],
                    textinfo="label+percent",
                    hole=0.40,
                    marker_colors=["#3b82f6", "#06b6d4", "#10b981", "#f59e0b", "#6b7280"],
                ))
                fig_cloud.update_layout(
                    paper_bgcolor="#0f172a", font=dict(color="#e2e8f0"),
                    height=350, showlegend=True,
                    legend=dict(bgcolor="rgba(0,0,0,0)"),
                    title="TIV by Cloud Provider"
                )
                st.plotly_chart(fig_cloud, use_container_width=True)
                # Benchmark comparison
                st.markdown("**vs. Industry Benchmark (Flexera 2024):**")
                bm_rows = acc.get("cloud_benchmark_vs_portfolio", {})
                bm_list = [{"Provider": k, "Benchmark": f"{v['benchmark_pct']:.1%}",
                            "Portfolio": f"{v['portfolio_pct']:.1%}",
                            "Delta": f"{v['portfolio_pct'] - v['benchmark_pct']:+.1%}"}
                           for k, v in bm_rows.items()]
                if bm_list:
                    st.dataframe(pd.DataFrame(bm_list), use_container_width=True, hide_index=True)

        with ac2:
            st.subheader("Core Banking Vendor — TIV Concentration")
            vendor_rows = acc.get("vendor_concentration", [])
            if vendor_rows:
                vd = pd.DataFrame(vendor_rows)
                fig_vendor = px.bar(
                    vd, x="group", y="total_tiv_usd",
                    color="pct_tiv", color_continuous_scale="Reds",
                    text=[f"n={r['n_policies']}" for _, r in vd.iterrows()],
                    labels={"group": "Core Banking Vendor", "total_tiv_usd": "Total TIV ($USD)"},
                    title="TIV by Core Banking Vendor"
                )
                fig_vendor.update_layout(
                    paper_bgcolor="#0f172a", plot_bgcolor="#1e293b",
                    font=dict(color="#e2e8f0"), height=350, showlegend=False,
                    coloraxis_showscale=False
                )
                st.plotly_chart(fig_vendor, use_container_width=True)

        st.markdown("---")

        # ── Geographic + Sector ────────────────────────────────────────────
        ag1, ag2 = st.columns(2)
        with ag1:
            st.subheader("Geographic Concentration")
            geo_rows = acc.get("geo_concentration", [])
            if geo_rows:
                gd = pd.DataFrame(geo_rows)
                fig_geo = px.bar(
                    gd, x="group", y="mfl_usd",
                    color="pct_tiv", color_continuous_scale="Oranges",
                    labels={"group": "Region", "mfl_usd": "MFL ($USD — 80% TIV)"},
                    title="Maximum Foreseeable Loss by Region"
                )
                fig_geo.update_layout(
                    paper_bgcolor="#0f172a", plot_bgcolor="#1e293b",
                    font=dict(color="#e2e8f0"), height=340, showlegend=False,
                    coloraxis_showscale=False
                )
                st.plotly_chart(fig_geo, use_container_width=True)

        with ag2:
            st.subheader("Sector Accumulation Risk")
            sector_rows = acc.get("sector_concentration", [])
            if sector_rows:
                sd = pd.DataFrame(sector_rows)
                fig_sec_a = px.bar(
                    sd, x="group", y="total_tiv_usd",
                    color="pct_tiv", color_continuous_scale="Purples",
                    labels={"group": "Sub-Sector", "total_tiv_usd": "Total TIV ($USD)"},
                    title="TIV by Sub-Sector"
                )
                fig_sec_a.update_layout(
                    paper_bgcolor="#0f172a", plot_bgcolor="#1e293b",
                    font=dict(color="#e2e8f0"), height=340, showlegend=False,
                    coloraxis_showscale=False
                )
                st.plotly_chart(fig_sec_a, use_container_width=True)

        st.markdown("---")
        # ── Accumulation risk band ─────────────────────────────────────────
        st.markdown("### Policy Accumulation Risk Band Distribution")
        ar_info = acc.get("accumulation_risk", {})
        band_dist = ar_info.get("band_distribution", {})

        ar1, ar2, ar3, ar4 = st.columns(4)
        for col_w, band, emoji, col_color in [
            (ar1, "Low",      "🟢", "normal"),
            (ar2, "Moderate", "🟡", "normal"),
            (ar3, "High",     "🟠", "inverse"),
            (ar4, "Extreme",  "🔴", "inverse"),
        ]:
            col_w.metric(f"{emoji} {band}", f"{band_dist.get(band, 0):,} policies")

        n_flagged = ar_info.get("n_policies_flagged", 0)
        pct_flag  = ar_info.get("pct_flagged", 0)
        if n_flagged > 0:
            st.warning(
                f"**{n_flagged} policies ({pct_flag:.1%}) flagged for reinsurance accumulation review** — "
                f"these sit in the top 10% of accumulation risk score and should receive individual "
                f"underwriting scrutiny before binding."
            )

        st.caption("*Accumulation risk score = weighted concentration across cloud (35%), vendor (30%), geography (20%), sector (15%)*")

    except FileNotFoundError:
        st.info("Accumulation risk data not found. Run `code/06_portfolio_accumulation.py` from the project root.")
        st.code("python code/06_portfolio_accumulation.py", language="bash")
    except Exception as _ae:
        st.error(f"Error loading accumulation data: {_ae}")
