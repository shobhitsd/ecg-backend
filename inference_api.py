import numpy as np
import pandas as pd
import joblib
import pickle
import json
import io
import shap
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
app = FastAPI(title="CardioVision AI API", version="1.0")

# Enable CORS (Critical for Lovable frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for hackathon (restrict in production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
model = None
scaler = None
explainer = None
feature_names = []

# ==========================================
# 2. FEATURE EXTRACTOR (Copy from Pipeline)
# ==========================================
class ECGFeatureExtractor:
    def __init__(self, sampling_rate: int = 100):
        self.fs = sampling_rate
        
    def extract_features(self, signal: np.ndarray) -> np.ndarray:
        features = []
        for lead_idx in range(signal.shape[1]):
            lead = signal[:, lead_idx]
            # Time-domain
            features.append(np.mean(lead))
            features.append(np.std(lead))
            features.append(np.max(lead))
            features.append(np.min(lead))
            features.append(np.max(lead) - np.min(lead))
            features.append(np.percentile(lead, 75) - np.percentile(lead, 25))
            
            # Zero-crossing
            zero_crossings = np.sum(np.abs(np.diff(np.sign(lead - np.mean(lead)))) > 0)
            features.append(zero_crossings / len(lead))
            
            # RMS
            rms = np.sqrt(np.mean(lead ** 2))
            features.append(rms)
            
            # Skew/Kurtosis
            if np.std(lead) > 0:
                features.append(self._skewness(lead))
                features.append(self._kurtosis(lead))
            else:
                features.extend([0, 0])
                
            # First derivative
            diff = np.diff(lead)
            features.append(np.mean(np.abs(diff)))
            features.append(np.max(np.abs(diff)))
            
        # Cross-lead features
        for i in range(min(3, signal.shape[1])):
            for j in range(i + 1, min(4, signal.shape[1])):
                corr = np.corrcoef(signal[:, i], signal[:, j])[0, 1]
                features.append(corr if not np.isnan(corr) else 0)
        
        return np.array(features, dtype=np.float32)
    
    def _skewness(self, x):
        n = len(x)
        mean = np.mean(x)
        std = np.std(x)
        return np.sum((x - mean) ** 3) / (n * std ** 3) if std != 0 else 0
    
    def _kurtosis(self, x):
        n = len(x)
        mean = np.mean(x)
        std = np.std(x)
        return (np.sum((x - mean) ** 4) / (n * std ** 4) - 3) if std != 0 else 0

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def generate_layman_explanation(feature_name, impact):
    """Translates technical features to plain English"""
    lead_map = {
        'V1': 'Right Ventricle', 'V2': 'Septum', 'V3': 'Anterior Wall', 'V4': 'Anterior Wall',
        'V5': 'Lateral Wall', 'V6': 'Lateral Wall', 'I': 'Lateral Wall', 'aVL': 'Lateral Wall',
        'II': 'Inferior Wall', 'III': 'Inferior Wall', 'aVF': 'Inferior Wall', 'aVR': 'Right Atrium'
    }
    feat_map = {
        'mean': 'voltage level', 'std': 'activity variability', 'max': 'peak height',
        'min': 'trough depth', 'p2p': 'signal amplitude', 'diff_mean': 'beat-to-beat stability',
        'skew': 'wave asymmetry', 'zcr': 'rhythm irregularity', 'rms': 'overall energy'
    }
    
    lead = next((l for l in lead_map if feature_name.startswith(l)), None)
    metric = next((m for m in feat_map if m in feature_name), 'pattern')
    region = lead_map.get(lead, 'General Heart')
    concept = feat_map.get(metric, 'pattern')
    direction = "Higher" if impact > 0 else "Lower"
    
    if lead:
        return f"{direction} {concept} in {lead} ({region})"
    else:
        return f"{direction} {concept} across leads"

def generate_feature_names():
    names = []
    lead_names = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']
    for lead in lead_names:
        names.extend([
            f'{lead}_mean', f'{lead}_std', f'{lead}_max', f'{lead}_min',
            f'{lead}_p2p', f'{lead}_iqr', f'{lead}_zcr', f'{lead}_rms',
            f'{lead}_skew', f'{lead}_kurt', f'{lead}_diff_mean', f'{lead}_diff_max'
        ])
    names.extend(['I_II_corr','I_III_corr','I_aVR_corr','II_III_corr','II_aVR_corr','III_aVR_corr'])
    return names

# ==========================================
# 4. LIFECYCLE EVENTS
# ==========================================
@app.on_event("startup")
async def load_models():
    global model, scaler, explainer, feature_names
    try:
        # Load artifacts
        scaler = joblib.load("scaler.pkl")
        model = joblib.load("xgboost_model.pkl") # Using XGBoost for speed/explainability
        
        # Initialize Explainer
        explainer = shap.TreeExplainer(model)
        feature_names = generate_feature_names()
        
        print("✅ Models loaded successfully")
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        print("Ensure 'scaler.pkl' and 'xgboost_model.pkl' are in the same directory.")

# ==========================================
# 5. API ENDPOINTS
# ==========================================
@app.get("/")
def health_check():
    return {"status": "online", "model": "ECG-XGBoost-v1"}

@app.post("/analyze-ecg")
async def analyze_ecg(file: UploadFile = File(...)):
    try:
        # 1. Read File
        contents = await file.read()
        
        # Determine format (CSVs usually)
        try:
            # decode bytes to string
            s = str(contents, 'utf-8')
            data = io.StringIO(s)
            ecg_signal = pd.read_csv(data, header=None).values
        except:
            return {"error": "Invalid CSV format. Expected 1000 rows x 12 columns."}

        # 2. Validation
        if ecg_signal.shape != (1000, 12):
            return {
                "error": f"Incorrect shape {ecg_signal.shape}. Expected (1000, 12).",
                "risk_score": 0, "prediction": "ERROR", "explanation": "Invalid Data",
                "shap_values": {}
            }

        # 3. Extract Features
        extractor = ECGFeatureExtractor()
        feats = extractor.extract_features(ecg_signal)
        
        # 4. Scale
        feats_scaled = scaler.transform(feats.reshape(1, -1))
        
        # 5. Predict
        prob = model.predict_proba(feats_scaled)[0, 1]
        
        # 6. Explain
        shap_vals = explainer.shap_values(feats_scaled)[0]
        top_indices = np.argsort(np.abs(shap_vals))[-5:][::-1]
        
        layman_explanations = []
        shap_dict = {}
        
        for i in top_indices:
            feat_name = feature_names[i]
            impact = float(shap_vals[i])
            layman_explanations.append(generate_layman_explanation(feat_name, impact))
            shap_dict[feat_name] = impact

        return {
            "risk_score": float(prob),
            "prediction": "ABNORMAL" if prob >= 0.5 else "NORMAL",
            "confidence": "HIGH" if (prob < 0.2 or prob > 0.8) else "MODERATE",
            "explanation": "Top risk factors: " + ", ".join(layman_explanations[:3]),
            "shap_values": shap_dict
        }

    except Exception as e:
        return {"error": str(e)}
