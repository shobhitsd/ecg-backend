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

app = FastAPI(title="CardioVision AI API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
scaler = None
explainer = None
feature_names = []

# ==========================================
# ROBUST FEATURE EXTRACTOR
# ==========================================
class ECGFeatureExtractor:
    def __init__(self, sampling_rate: int = 100):
        self.fs = sampling_rate
        
    def extract_features(self, signal: np.ndarray) -> np.ndarray:
        features = []
        for lead_idx in range(signal.shape[1]):
            lead = signal[:, lead_idx]
            features.append(np.mean(lead))
            features.append(np.std(lead))
            features.append(np.max(lead))
            features.append(np.min(lead))
            features.append(np.max(lead) - np.min(lead))
            features.append(np.percentile(lead, 75) - np.percentile(lead, 25))
            
            zero_crossings = np.sum(np.abs(np.diff(np.sign(lead - np.mean(lead)))) > 0)
            features.append(zero_crossings / len(lead))
            
            rms = np.sqrt(np.mean(lead ** 2))
            features.append(rms)
            
            if np.std(lead) > 0:
                features.append(self._skewness(lead))
                features.append(self._kurtosis(lead))
            else:
                features.extend([0, 0])
                
            diff = np.diff(lead)
            features.append(np.mean(np.abs(diff)))
            features.append(np.max(np.abs(diff)))
            
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

def generate_layman_explanation(feature_name, impact):
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
# STARTUP - WITH ERROR HANDLING
# ==========================================
@app.on_event("startup")
async def load_models():
    global model, scaler, explainer, feature_names
    print("🔄 Loading models...")
    
    try:
        scaler = joblib.load("scaler.pkl")
        print("✅ Scaler loaded")
    except Exception as e:
        print(f"❌ Scaler Error: {e}")

    try:
        # Warning: XGBoost on CPU might warn about GPU params, that's fine.
        model = joblib.load("xgboost_model.pkl")
        print("✅ XGBoost loaded")
        
        # Force CPU helper if needed (sometimes helps with version mismatches)
        if hasattr(model, 'set_params'):
            try:
                model.set_params(device='cpu')
            except:
                pass
                
    except Exception as e:
        print(f"❌ XGBoost Error: {e}")

    try:
        explainer = shap.TreeExplainer(model)
        feature_names = generate_feature_names()
        print("✅ Explainer initialized")
    except Exception as e:
        print(f"⚠️ Explainer Error (Skipping SHAP): {e}")

@app.get("/")
def health_check():
    return {"status": "online", "model": "ECG-XGBoost-v1"}

@app.post("/analyze-ecg")
async def analyze_ecg(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        try:
            s = str(contents, 'utf-8')
            data = io.StringIO(s)
            ecg_signal = pd.read_csv(data, header=None).values
        except:
            return {"error": "Invalid CSV format."}

        # Handling inputs of different sizes for robustness
        if ecg_signal.shape[0] != 1000:
             # simple crop or pad
             if ecg_signal.shape[0] > 1000:
                 ecg_signal = ecg_signal[:1000, :]
             else:
                 pad_len = 1000 - ecg_signal.shape[0]
                 ecg_signal = np.pad(ecg_signal, ((0, pad_len), (0,0)))

        extractor = ECGFeatureExtractor()
        feats = extractor.extract_features(ecg_signal)
        
        if scaler:
            feats_scaled = scaler.transform(feats.reshape(1, -1))
        else:
            feats_scaled = feats.reshape(1, -1) # Fallback
        
        if model:
            prob = float(model.predict_proba(feats_scaled)[0, 1])
        else:
            prob = 0.5 # Default fallback
        
        layman_explanations = []
        shap_dict = {}
        
        if explainer:
            try:
                shap_vals = explainer.shap_values(feats_scaled)[0]
                top_indices = np.argsort(np.abs(shap_vals))[-5:][::-1]
                for i in top_indices:
                    feat_name = feature_names[i]
                    impact = float(shap_vals[i])
                    layman_explanations.append(generate_layman_explanation(feat_name, impact))
                    shap_dict[feat_name] = impact
            except:
                pass # SHAP fail shouldn't crash app

        return {
            "risk_score": prob,
            "prediction": "ABNORMAL" if prob >= 0.5 else "NORMAL",
            "confidence": "HIGH" if (prob < 0.2 or prob > 0.8) else "MODERATE",
            "explanation": "Top risk factors: " + ", ".join(layman_explanations[:3]),
            "shap_values": shap_dict
        }

    except Exception as e:
        return {"error": str(e)}
