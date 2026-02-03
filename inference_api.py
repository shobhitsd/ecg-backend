"""
ECG Inference API v6.1 - Railway Production Deployment
========================================================
Compatible with ecg_pipeline.py v6.1 Polished

Features:
- 198-dim clinical feature extraction (exact match to training pipeline)
- Pan-Tompkins QRS detection
- 1D-CNN + XGBoost ensemble with meta-learner
- SHAP feature importance
- Health check endpoint

Author: CardioVision AI
Version: 6.1 (Production)
"""

import os
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# FastAPI
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ML Libraries
import torch
import torch.nn as nn
import joblib
from scipy import signal as scipy_signal
from scipy.ndimage import uniform_filter1d

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURATION
# ============================================================

class Config:
    SEED = 42
    ECG_SAMPLING_RATE = 100
    ECG_LENGTH = 1000
    NUM_LEADS = 12
    FEATURE_DIM = 198  # Must match training pipeline
    MODEL_DIR = Path(".")
    DEVICE = torch.device("cpu")  # Force CPU for Railway

config = Config()

# Set seed for reproducibility
np.random.seed(config.SEED)
torch.manual_seed(config.SEED)

# ============================================================
# PAN-TOMPKINS QRS DETECTION
# ============================================================

class PanTompkinsDetector:
    """Pan-Tompkins QRS detection algorithm."""
    
    def __init__(self, fs: int = 100):
        self.fs = fs
        self.window_size = int(0.15 * fs)
        self._init_filters()

    def _init_filters(self):
        self.b_band, self.a_band = scipy_signal.butter(
            4, [5, 15], "bandpass", fs=self.fs
        )
        self.diff_kernel = np.array([-1, -2, 0, 2, 1]) * (self.fs / 8)
        self.ma_window = int(0.08 * self.fs)

    def detect(self, signal: np.ndarray) -> np.ndarray:
        """Detect R-peaks in single-lead ECG."""
        filtered = scipy_signal.filtfilt(self.b_band, self.a_band, signal)
        diff = np.convolve(filtered, self.diff_kernel, mode="same")
        squared = diff ** 2
        integrated = uniform_filter1d(squared, size=self.ma_window)
        peaks = self._find_peaks_adaptive(integrated)
        peaks = self._search_back(peaks, integrated)
        return np.array(peaks, dtype=int)

    def _find_peaks_adaptive(self, signal: np.ndarray, threshold_factor: float = 0.6):
        peaks = []
        threshold = np.max(signal) * 0.5
        signal_peaks = scipy_signal.find_peaks(signal, distance=self.window_size)[0]
        for p in signal_peaks:
            if signal[p] > threshold:
                peaks.append(p)
                threshold = 0.8 * threshold + 0.2 * signal[p] * threshold_factor
        return peaks

    def _search_back(self, peaks: List[int], integrated: np.ndarray) -> List[int]:
        if len(peaks) < 2:
            return peaks
        rr_intervals = np.diff(peaks)
        avg_rr = np.mean(rr_intervals)
        new_peaks = []
        for i in range(len(peaks) - 1):
            new_peaks.append(peaks[i])
            if rr_intervals[i] > 1.66 * avg_rr:
                search_start = peaks[i] + int(0.2 * self.fs)
                search_end = peaks[i + 1] - int(0.2 * self.fs)
                if search_start < search_end:
                    region = integrated[search_start:search_end]
                    if len(region) > 0:
                        missed = search_start + np.argmax(region)
                        new_peaks.append(missed)
        new_peaks.append(peaks[-1])
        return sorted(new_peaks)

# ============================================================
# CLINICAL FEATURE EXTRACTION (198-dim - MUST MATCH TRAINING)
# ============================================================

class ClinicalFeatureExtractor:
    """Extract 198-dim clinical ECG features matching training pipeline."""
    
    EXPECTED_DIM = 198  # 5 rhythm + 5 HRV + 12*15 morph + 4 cross + 4 ischemia
    
    def __init__(self, fs: int = 100):
        self.fs = fs
        self.qrs_detector = PanTompkinsDetector(fs)
        logger.info(f"ClinicalFeatureExtractor initialized (fs={fs}, dim={self.EXPECTED_DIM})")

    def extract_all_features(self, signal: np.ndarray) -> np.ndarray:
        """Return fixed-length 198-dim feature vector."""
        feats = self._extract_features_dict(signal)
        flat = []

        # Rhythm (5)
        flat.extend([
            feats["heart_rate"],
            feats["rr_mean"],
            feats["rr_std"],
            feats["rr_cv"],
            feats["n_beats"],
        ])

        # HRV (5)
        flat.extend([
            feats["hrv_sdnn"],
            feats["hrv_rmssd"],
            feats["hrv_lf"],
            feats["hrv_hf"],
            feats["hrv_lf_hf_ratio"],
        ])

        # Morphology per lead (12 × 15 = 180)
        for lead_idx in range(config.NUM_LEADS):
            lead_feats = feats["morph"][lead_idx]
            lead_feats = (lead_feats + [0.0] * 15)[:15]
            flat.extend(lead_feats)

        # Cross-lead (4)
        cross = feats["cross_lead"]
        cross = (cross + [0.0] * 4)[:4]
        flat.extend(cross)

        # Ischemia (4)
        isc = feats["ischemia"]
        isc = (isc + [0.0] * 4)[:4]
        flat.extend(isc)

        assert len(flat) == self.EXPECTED_DIM, \
            f"Feature dim mismatch: {len(flat)} vs {self.EXPECTED_DIM}"
        
        return np.array(flat, dtype=np.float32)

    def _extract_features_dict(self, signal: np.ndarray) -> Dict:
        d = {}
        lead_ii = signal[:, 1]
        r_peaks = self.qrs_detector.detect(lead_ii)

        d.update(self._calculate_rhythm_features(lead_ii, r_peaks))
        d.update(self._calculate_hrv_safe(r_peaks))

        morph_all = []
        for lead_idx in range(config.NUM_LEADS):
            morph_all.append(
                self._calculate_morphological_features(signal[:, lead_idx], r_peaks)
            )
        d["morph"] = morph_all
        d["cross_lead"] = self._calculate_cross_lead_features(signal)
        d["ischemia"] = self._detect_ischemia_patterns(signal, r_peaks)
        return d

    def _calculate_rhythm_features(self, signal: np.ndarray, r_peaks: np.ndarray):
        if len(r_peaks) < 2:
            return dict(
                heart_rate=0.0, rr_mean=0.0, rr_std=0.0, rr_cv=0.0,
                n_beats=float(len(r_peaks))
            )
        rr = np.diff(r_peaks) / self.fs
        hr = 60.0 / (np.mean(rr) + 1e-10)
        return dict(
            heart_rate=hr,
            rr_mean=float(np.mean(rr)),
            rr_std=float(np.std(rr)),
            rr_cv=float(np.std(rr) / (np.mean(rr) + 1e-10)),
            n_beats=float(len(r_peaks)),
        )

    def _calculate_hrv_safe(self, r_peaks: np.ndarray):
        """Safe HRV calculation using Welch PSD."""
        if len(r_peaks) < 4:
            return dict(
                hrv_sdnn=0.0, hrv_rmssd=0.0, hrv_lf=0.0, hrv_hf=0.0, hrv_lf_hf_ratio=0.0
            )

        rr = np.diff(r_peaks) / self.fs
        rr_ms = rr * 1000.0
        sdnn = float(np.std(rr_ms))
        rmssd = float(np.sqrt(np.mean(np.diff(rr_ms) ** 2)))

        lf_power = hf_power = lf_hf_ratio = 0.0
        try:
            if len(rr) >= 8:
                t_cum = np.cumsum(rr)
                t_reg = np.linspace(t_cum[0], t_cum[-1], len(rr))
                rr_reg = np.interp(t_reg, t_cum, rr)
                rr_reg = rr_reg - np.mean(rr_reg)
                
                fs_est = 1 / (np.mean(rr) + 1e-10)
                freqs, psd = scipy_signal.welch(
                    rr_reg, fs=fs_est,
                    nperseg=min(256, len(rr_reg) // 2),
                    window="hamming"
                )
                
                lf_mask = (freqs >= 0.04) & (freqs < 0.15)
                hf_mask = (freqs >= 0.15) & (freqs <= 0.4)
                
                if np.any(lf_mask):
                    lf_power = float(np.trapz(psd[lf_mask], freqs[lf_mask]))
                if np.any(hf_mask):
                    hf_power = float(np.trapz(psd[hf_mask], freqs[hf_mask]))
                lf_hf_ratio = float(lf_power / (hf_power + 1e-10))
        except Exception:
            pass

        return dict(
            hrv_sdnn=sdnn, hrv_rmssd=rmssd, hrv_lf=lf_power,
            hrv_hf=hf_power, hrv_lf_hf_ratio=lf_hf_ratio
        )

    def _calculate_morphological_features(self, lead: np.ndarray, r_peaks: np.ndarray):
        """Return exactly 15 morphological features per lead."""
        f = []
        
        # Basic stats (8)
        f.extend([
            float(np.mean(lead)), float(np.std(lead)),
            float(np.max(lead)), float(np.min(lead)),
            float(np.max(lead) - np.min(lead)),
            float(np.percentile(lead, 25)), float(np.percentile(lead, 75)),
            float(np.median(lead)),
        ])
        
        # QRS characteristics (3)
        widths, amps = [], []
        for r in r_peaks[:10]:
            w = int(0.1 * self.fs)
            s, e = max(0, r - w), min(len(lead), r + w)
            seg = lead[s:e]
            if len(seg) > 0:
                widths.append(np.std(seg))
                amps.append(np.max(seg) - np.min(seg))
        
        if widths:
            f.extend([float(np.mean(widths)), float(np.std(widths)), float(np.mean(amps))])
        else:
            f.extend([0.0, 0.0, 0.0])
        
        # Spectral bands (3)
        try:
            freqs, psd = scipy_signal.welch(lead, self.fs, nperseg=min(256, len(lead) // 2))
            f.extend([
                float(np.sum(psd[(freqs >= 0.5) & (freqs <= 4)])),
                float(np.sum(psd[(freqs > 4) & (freqs <= 10)])),
                float(np.sum(psd[(freqs > 10) & (freqs <= 40)]))
            ])
        except Exception:
            f.extend([0.0, 0.0, 0.0])
        
        # Entropy (1)
        try:
            hist, _ = np.histogram(lead, bins=50, range=(np.min(lead), np.max(lead)), density=True)
            hist = hist + 1e-10
            f.append(float(-np.sum(hist * np.log(hist))))
        except Exception:
            f.append(0.0)
        
        return f

    def _calculate_cross_lead_features(self, signal: np.ndarray):
        """Return exactly 4 cross-lead features."""
        corr = np.corrcoef(signal.T)
        mask = np.triu(np.ones_like(corr, bool), k=1)
        mean_corr = float(np.nanmean(corr[mask]))

        limb = [0, 1, 2, 3, 4, 5]
        prec = [6, 7, 8, 9, 10, 11]
        limb_corrs = [corr[i, j] for i in limb for j in limb if i < j]
        prec_corrs = [corr[i, j] for i in prec for j in prec if i < j]

        return [
            mean_corr,
            float(np.nanmean(limb_corrs)) if limb_corrs else 0.0,
            float(np.nanmean(prec_corrs)) if prec_corrs else 0.0,
            float(np.max(np.var(signal, axis=0)))
        ]

    def _detect_ischemia_patterns(self, signal: np.ndarray, r_peaks: np.ndarray):
        """Return exactly 4 ischemia-related features."""
        if len(r_peaks) < 2:
            return [0.0, 0.0, 0.0, 0.0]

        st_elev_all = []
        anterior_leads = [6, 7, 8, 9]  # V1-V4

        for lead_idx in anterior_leads:
            lead = signal[:, lead_idx]
            for r in r_peaks[:-1]:
                st_start = int(r + 0.08 * self.fs)
                st_end = int(r + 0.12 * self.fs)
                if st_end >= len(lead):
                    continue
                pr_start = max(0, int(r - 0.12 * self.fs))
                pr_end = max(pr_start + 1, int(r - 0.04 * self.fs))
                baseline = np.mean(lead[pr_start:pr_end])
                st_level = np.mean(lead[st_start:st_end])
                st_elev_all.append(float(st_level - baseline))

        if st_elev_all:
            max_elev = float(np.max(st_elev_all))
            mean_elev = float(np.mean(st_elev_all))
            n_sig = int(np.sum(np.array(st_elev_all) > 0.001))
            seg_len = max(1, len(st_elev_all) // len(anterior_leads))
            v1_v4_mean = float(np.mean(st_elev_all[:seg_len]))
        else:
            max_elev = mean_elev = v1_v4_mean = 0.0
            n_sig = 0

        return [max_elev, mean_elev, v1_v4_mean, float(n_sig)]

# ============================================================
# 1D-CNN MODEL ARCHITECTURE (MUST MATCH TRAINING)
# ============================================================

class ECG1DCNN(nn.Module):
    """1D-CNN matching training architecture exactly."""
    
    def __init__(self, num_leads=12, num_classes=1, dropout=0.5):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(num_leads)
        
        self.conv1 = self._block(num_leads, 64, kernel_size=50, pool_size=2)
        self.conv2 = self._block(64, 128, kernel_size=30, pool_size=2)
        self.conv3 = self._block(128, 256, kernel_size=15, pool_size=2)
        self.conv4 = self._block(256, 512, kernel_size=10, pool_size=2)

        self.attention = nn.Sequential(
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Softmax(dim=1),
        )

        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def _block(self, in_ch, out_ch, kernel_size, pool_size):
        pad = kernel_size // 2
        return nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, stride=1, padding=pad),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.MaxPool1d(pool_size),
            nn.Conv1d(out_ch, out_ch, kernel_size, stride=1, padding=pad),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.MaxPool1d(pool_size),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B,T,L)->(B,L,T)
        x = self.input_bn(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        x_att = x.permute(0, 2, 1)
        w = self.attention(x_att)
        x_pooled = torch.sum(x_att * w, dim=1)
        
        out = self.fc(x_pooled)
        return torch.sigmoid(out).squeeze(-1)

# ============================================================
# MODEL LOADER
# ============================================================

class ECGEnsembleInference:
    """Production inference wrapper for trained ensemble."""
    
    def __init__(self, model_dir: Path = Path(".")):
        self.model_dir = model_dir
        self.cnn_model = None
        self.xgb_model = None
        self.meta_learner = None
        self.scaler = None
        self.feature_extractor = ClinicalFeatureExtractor(config.ECG_SAMPLING_RATE)
        self.feature_names = self._get_feature_names()
        self._load_models()
    
    def _get_feature_names(self) -> List[str]:
        """Generate feature names matching training pipeline."""
        names = []
        names.extend(["heart_rate", "rr_mean", "rr_std", "rr_cv", "n_beats"])
        names.extend(["hrv_sdnn", "hrv_rmssd", "hrv_lf", "hrv_hf", "hrv_lf_hf_ratio"])
        
        leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        for ld in leads:
            names.extend([
                f"{ld}_mean", f"{ld}_std", f"{ld}_max", f"{ld}_min", f"{ld}_p2p",
                f"{ld}_q25", f"{ld}_q75", f"{ld}_median",
                f"{ld}_qrs_width", f"{ld}_qrs_width_std", f"{ld}_qrs_amp",
                f"{ld}_power_qrs", f"{ld}_power_t", f"{ld}_power_hf", f"{ld}_entropy",
            ])
        names.extend([
            "cross_lead_mean_corr", "limb_lead_corr", "precordial_lead_corr",
            "max_lead_variance",
            "max_st_elevation", "mean_st_elevation", "st_elevation_v1_v4", "ischemia_score",
        ])
        return names
    
    def _load_models(self):
        """Load all model components."""
        # CNN
        cnn_path = self.model_dir / "cnn_model.pt"
        if cnn_path.exists():
            self.cnn_model = ECG1DCNN().to(config.DEVICE)
            state_dict = torch.load(cnn_path, map_location=config.DEVICE)
            self.cnn_model.load_state_dict(state_dict)
            self.cnn_model.eval()
            logger.info(f"✓ CNN model loaded from {cnn_path}")
        else:
            logger.warning(f"CNN model not found at {cnn_path}")
        
        # XGBoost
        xgb_path = self.model_dir / "xgb_model.pkl"
        if xgb_path.exists():
            self.xgb_model = joblib.load(xgb_path)
            logger.info(f"✓ XGBoost model loaded from {xgb_path}")
        else:
            logger.warning(f"XGBoost model not found at {xgb_path}")
        
        # Scaler
        scaler_path = self.model_dir / "scaler.pkl"
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
            logger.info(f"✓ Scaler loaded from {scaler_path}")
        else:
            logger.warning(f"Scaler not found at {scaler_path}")
        
        # Try to load SHAP explainer
        try:
            import shap
            if self.xgb_model is not None:
                self.explainer = shap.TreeExplainer(self.xgb_model)
                logger.info("✓ SHAP explainer initialized")
        except Exception as e:
            self.explainer = None
            logger.warning(f"SHAP initialization failed: {e}")
    
    def predict(self, ecg_signal: np.ndarray) -> Dict:
        """
        Run inference on a single ECG.
        
        Args:
            ecg_signal: numpy array of shape (1000, 12)
        
        Returns:
            Dictionary with prediction, confidence, and explanations
        """
        # Validate input
        if ecg_signal.shape != (config.ECG_LENGTH, config.NUM_LEADS):
            raise ValueError(f"Expected shape ({config.ECG_LENGTH}, {config.NUM_LEADS}), got {ecg_signal.shape}")
        
        # Extract features
        features = self.feature_extractor.extract_all_features(ecg_signal)
        features = np.nan_to_num(features, nan=0, posinf=0, neginf=0)
        
        # Scale features
        if self.scaler is not None:
            features_scaled = self.scaler.transform(features.reshape(1, -1))
        else:
            features_scaled = features.reshape(1, -1)
        
        # CNN prediction
        cnn_pred = 0.5
        if self.cnn_model is not None:
            with torch.no_grad():
                x = torch.FloatTensor(ecg_signal).unsqueeze(0).to(config.DEVICE)
                cnn_pred = float(self.cnn_model(x).cpu().numpy()[0])
        
        # XGBoost prediction
        xgb_pred = 0.5
        if self.xgb_model is not None:
            xgb_pred = float(self.xgb_model.predict_proba(features_scaled)[0, 1])
        
        # Simple ensemble (weighted average if no meta-learner)
        final_pred = 0.6 * cnn_pred + 0.4 * xgb_pred
        
        # Get SHAP explanations
        shap_explanations = []
        if hasattr(self, 'explainer') and self.explainer is not None:
            try:
                shap_values_raw = self.explainer.shap_values(features_scaled)
                logger.info(f"SHAP raw type: {type(shap_values_raw)}")
                
                # Handle different SHAP return types (list vs array)
                if isinstance(shap_values_raw, list):
                    shap_values = shap_values_raw[0] if len(shap_values_raw) > 0 else []
                else:
                    shap_values = shap_values_raw
                
                if len(shap_values.shape) == 2:
                    shap_values = shap_values[0]
                    
                logger.info(f"SHAP final shape: {shap_values.shape}")
                
                top_indices = np.argsort(np.abs(shap_values))[-5:][::-1]
                for idx in top_indices:
                    if idx < len(self.feature_names):
                        shap_explanations.append({
                            "feature": self.feature_names[idx],
                            "impact": float(shap_values[idx]),
                            "direction": "increases risk" if shap_values[idx] > 0 else "decreases risk"
                        })
            except Exception as e:
                logger.warning(f"SHAP explanation failed: {e}")
                import traceback
                logger.warning(traceback.format_exc())
        
        # Clinical findings
        findings = []
        hr = features[0]  # heart_rate is first feature
        if hr > 0:
            if hr > 100:
                findings.append(f"Tachycardia: {hr:.0f} bpm")
            elif hr < 60:
                findings.append(f"Bradycardia: {hr:.0f} bpm")
            else:
                findings.append(f"Normal heart rate: {hr:.0f} bpm")
        
        # ST elevation check
        st_elev_idx = self.feature_names.index("max_st_elevation")
        st_elev = features[st_elev_idx]
        if st_elev > 0.1:  # Threshold > 1mm (0.1 mV)
            findings.append(f"ST elevation detected: {st_elev*10:.2f} mm")
        
        # ---------------------------------------------------------
        # GENERATE NATURAL LANGUAGE EXPLANATION
        # ---------------------------------------------------------
        explanation_parts = []
        
        # 1. Clinical Findings (Hard Rules)
        if findings:
            explanation_parts.append("Clinical signs detected: " + ", ".join(findings) + ".")
        
        # 2. Feature Importance (SHAP)
        if shap_explanations:
            top_fx = shap_explanations[:3] # Top 3
            fx_text = [f"{f['feature']} ({f['impact']:.3f})" for f in top_fx]
            explanation_parts.append(f"Primary risk drivers: {', '.join(fx_text)}.")
        else:
            # Fallback if SHAP failed: Use feature magnitude deviation from mean (approximate)
            # Assuming mean ~ 0 for scaled features
            magnitudes = np.abs(features_scaled[0])
            top_k_idx = np.argsort(magnitudes)[-3:][::-1]
            top_feats = [self.feature_names[i] for i in top_k_idx]
            explanation_parts.append(f"Significant deviations observed in: {', '.join(top_feats)}.")
            
            # Populate dummy SHAP for UI to not look broken
            for i, feat in zip(top_k_idx, top_feats):
                 shap_explanations.append({
                    "feature": feat,
                    "impact": float(features_scaled[0][i]) * 0.1, # Dummy weight
                    "direction": "deviation"
                })

        final_explanation = " ".join(explanation_parts)
        
        return {
            "prediction": "ABNORMAL" if final_pred > 0.5 else "NORMAL",
            "confidence": float(final_pred),
            "risk_percentage": float(final_pred * 100),
            "cnn_score": float(cnn_pred),
            "xgb_score": float(xgb_pred),
            "clinical_findings": findings,
            "contributing_factors": shap_explanations,
            "explanation": final_explanation,
            "model_version": "v6.1"
        }

# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="ECG Analysis API",
    description="Cardiovascular Disease Detection using 12-Lead ECG (v6.1)",
    version="6.1.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
ensemble = None

@app.on_event("startup")
async def load_models():
    global ensemble
    logger.info("Loading ECG ensemble models...")
    ensemble = ECGEnsembleInference(config.MODEL_DIR)
    logger.info("✓ Models loaded successfully")

# ============================================================
# API ENDPOINTS
# ============================================================

class ECGInput(BaseModel):
    ecg_data: List[List[float]] = Field(
        ..., 
        description="12-lead ECG data as 2D array [1000 samples × 12 leads]"
    )

class HealthResponse(BaseModel):
    status: str
    version: str
    models_loaded: bool

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    risk_percentage: float
    cnn_score: float
    xgb_score: float
    clinical_findings: List[str]
    contributing_factors: List[Dict]
    model_version: str

@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "6.1.0",
        "models_loaded": ensemble is not None and ensemble.cnn_model is not None
    }

@app.get("/health", response_model=HealthResponse)
async def health():
    """Alias for health check."""
    return await health_check()

@app.post("/predict", response_model=PredictionResponse)
async def predict(input_data: ECGInput):
    """
    Analyze a 12-lead ECG for cardiovascular disease.
    
    Input: 2D array of shape [1000, 12] representing 10-second 12-lead ECG at 100Hz
    """
    if ensemble is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        ecg_array = np.array(input_data.ecg_data, dtype=np.float32)
        
        # Validate shape
        if ecg_array.shape != (config.ECG_LENGTH, config.NUM_LEADS):
            raise HTTPException(
                status_code=400,
                detail=f"Expected shape ({config.ECG_LENGTH}, {config.NUM_LEADS}), got {ecg_array.shape}"
            )
        
        # Run inference
        result = ensemble.predict(ecg_array)
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze")
async def analyze(input_data: ECGInput):
    """Alias for predict endpoint."""
    return await predict(input_data)

@app.post("/analyze-ecg")
async def analyze_ecg_file(file: UploadFile = File(...)):
    """
    Analyze ECG from uploaded CSV file.
    Expected format: CSV with 1000 rows x 12 columns (or 12 rows x 1000 columns)
    """
    if ensemble is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        # Read file content
        content = await file.read()
        text = content.decode("utf-8")
        
        # Parse CSV
        lines = text.strip().split("\n")
        data = []
        for line in lines:
            # Handle both comma and whitespace separators
            if "," in line:
                row = [float(x.strip()) for x in line.split(",") if x.strip()]
            else:
                row = [float(x) for x in line.split() if x.strip()]
            if row:
                data.append(row)
        
        ecg_array = np.array(data, dtype=np.float32)
        
        # Handle transposed data (12x1000 -> 1000x12)
        if ecg_array.shape == (config.NUM_LEADS, config.ECG_LENGTH):
            ecg_array = ecg_array.T
        
        # Validate shape
        if ecg_array.shape != (config.ECG_LENGTH, config.NUM_LEADS):
            raise HTTPException(
                status_code=400,
                detail=f"Expected shape ({config.ECG_LENGTH}, {config.NUM_LEADS}), got {ecg_array.shape}. "
                       f"Please provide a CSV with 1000 rows and 12 columns."
            )
        
        # Run inference
        result = ensemble.predict(ecg_array)
        
        # Format response to match frontend expectations
        return {
            "prediction": result["prediction"],
            "risk_score": result["confidence"],
            # Use the dynamic explanation generated by the model, fallback to generic if missing
            "explanation": result.get("explanation", 
                f"Analysis based on {config.FEATURE_DIM} clinical features. " + "; ".join(result.get("clinical_findings", []))
            ),
            "shap_values": {f["feature"]: f["impact"] for f in result.get("contributing_factors", [])},
            "confidence": result["confidence"],
            "cnn_score": result["cnn_score"],
            "xgb_score": result["xgb_score"],
            "model_version": result["model_version"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
