import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, roc_auc_score, recall_score
from sklearn.preprocessing import LabelBinarizer
import joblib
import torch
from transformers import DistilBertTokenizer, DistilBertModel

# ============================================================
# Paths
# ============================================================
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "code" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Device Configuration
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device for DistilBERT: {device}")

# ============================================================
# 1. Load Data
# ============================================================
print("\nLoading datasets...")
findings = pd.read_csv(DATA_DIR / "03_regulatory_findings.csv")
print(f"Loaded {len(findings)} regulatory findings")

# ============================================================
# 2. Extract DistilBERT Embeddings
# ============================================================
print("\nLoading DistilBERT model for feature extraction...")
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
bert_model = DistilBertModel.from_pretrained("distilbert-base-uncased").to(device)
bert_model.eval()

def get_embeddings(texts, batch_size=32):
    all_embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            encoded = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            output = bert_model(**encoded)
            # Use the [CLS] token representation (0th token) as sentence embedding
            cls_embeddings = output.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(cls_embeddings)
    return np.vstack(all_embeddings)

print("Extracting embeddings for findings... This may take a moment.")
X_texts = findings["finding_text"].astype(str).tolist()
y = findings["severity_label"].astype(str).tolist()

X_emb = get_embeddings(X_texts)
print(f"Extracted embedding matrix shape: {X_emb.shape}")

# ============================================================
# 3. Train/val/test split
# ============================================================
X_train, X_temp, y_train, y_temp = train_test_split(
    X_emb, y, test_size=0.30, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
)

# ============================================================
# 4. Evaluation Helper
# ============================================================
def evaluate_model(name, y_true, y_pred, y_prob):
    print(f"\n{'='*60}")
    print(f"{name} - TEST METRICS")
    print(f"{'='*60}")
    print(classification_report(y_true, y_pred, zero_division=0))
    
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    
    lb = LabelBinarizer()
    y_true_bin = lb.fit_transform(y_true)
    if y_true_bin.shape[1] == 1:
        y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])
    macro_auc = roc_auc_score(y_true_bin, y_prob, average="macro", multi_class="ovr")
    
    print(f"Macro F1-score: {macro_f1:.4f}")
    print(f"Macro Recall:   {macro_recall:.4f}")
    print(f"Macro AUC-ROC:  {macro_auc:.4f}\n")
    return macro_f1

# ============================================================
# 5. Train Models (Random Forest over Embeddings)
# ============================================================
print("Training AI-Powered Model (Random Forest on DistilBERT)...")
rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
rf.fit(X_train, y_train)

rf_preds = rf.predict(X_test)
rf_probs = rf.predict_proba(X_test)
rf_f1 = evaluate_model("AI-POWERED (DistilBERT + RF)", y_test, rf_preds, rf_probs)

# Save the model
joblib.dump(rf, MODEL_DIR / "severity_classifier_baseline.joblib")
print(f"Saved best model: {MODEL_DIR / 'severity_classifier_baseline.joblib'}")

# ============================================================
# 6. Optional: Apply to Questionnaire Responses
# ============================================================
print("\n" + "=" * 60)
print("APPLYING TO QUESTIONNAIRE RESPONSES")
print("=" * 60)

try:
    responses = pd.read_csv(DATA_DIR / "04_questionnaire_responses.csv")
    if not responses.empty:
        Xq_texts = responses["response_text"].astype(str).tolist()
        yq = responses["response_quality_label"].astype(str).tolist()
        
        print("Extracting embeddings for questionnaire...")
        Xq_emb = get_embeddings(Xq_texts)
        
        Xq_tr, Xq_te, yq_tr, yq_te = train_test_split(Xq_emb, yq, test_size=0.20, stratify=yq, random_state=42)
        
        rf_q = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
        rf_q.fit(Xq_tr, yq_tr)
        
        rf_q_preds = rf_q.predict(Xq_te)
        rf_q_probs = rf_q.predict_proba(Xq_te)
        evaluate_model("AI-POWERED (Questionnaire DistilBERT)", yq_te, rf_q_preds, rf_q_probs)
        
        joblib.dump(rf_q, MODEL_DIR / "questionnaire_quality_classifier.joblib")
        print(f"Saved best questionnaire model: {MODEL_DIR / 'questionnaire_quality_classifier.joblib'}")

except FileNotFoundError:
    print("Warning: 04_questionnaire_responses.csv not found. Skipping questionnaire scoring.")
