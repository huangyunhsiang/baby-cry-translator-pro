# -*- coding: utf-8 -*-
"""
推論層：把 predict.py 的核心邏輯抽成可被 app.py 呼叫的函式。

唯一正本原則：特徵抽取一律 import features.py，不在此重寫。
輸入接受任意 bytes-like 音訊資料（麥克風錄音或上傳檔案），內部用
soundfile 讀成波形陣列，再交給 features.extract_features 處理。
"""
from __future__ import annotations

import io
import json
import os

import numpy as np
import onnxruntime as ort
import soundfile as sf

try:
    from .features import extract_features
except ImportError:  # 允許以腳本方式直接執行本檔（無套件上下文）
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from features import extract_features  # type: ignore  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
ONNX_PATH = os.path.join(MODELS_DIR, "cry_cnn.onnx")
LABELS_PATH = os.path.join(MODELS_DIR, "labels.json")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()


def load_labels() -> list[str]:
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_metrics() -> dict:
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_session() -> ort.InferenceSession:
    """建立新的 onnxruntime session。是否快取交由呼叫端（例如 app.py 用
    st.cache_resource）決定，本函式本身不做全域快取，避免匯入本模組時
    產生副作用。"""
    return ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])


def classify_waveform(wav: np.ndarray, sr: int,
                       session: ort.InferenceSession | None = None,
                       class_names: list[str] | None = None) -> list[tuple[str, float]]:
    """給定已載入的波形與取樣率，回傳 5 類機率（降冪排序）。"""
    if session is None:
        session = get_session()
    if class_names is None:
        class_names = load_labels()

    input_name = session.get_inputs()[0].name

    feat = extract_features(wav, sr=sr)          # (1, N_MELS, T)
    feat_batch = feat[np.newaxis, :, :, :]        # (1, 1, N_MELS, T)

    outputs = session.run(None, {input_name: feat_batch})
    logits = outputs[0][0]
    probs = softmax(logits)

    ranked = sorted(zip(class_names, (float(p) for p in probs)),
                     key=lambda x: x[1], reverse=True)
    return ranked


def classify_audio_bytes(data: bytes,
                          session: ort.InferenceSession | None = None,
                          class_names: list[str] | None = None) -> list[tuple[str, float]]:
    """
    給定任意音訊檔案的原始 bytes（wav/mp3/ogg/flac 等 soundfile 可解的格式），
    回傳 5 類機率，由高到低排序的 (類別名, 機率) list。
    """
    wav, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim > 1:
        # 多聲道先轉單聲道（平均），features.py 的 extract_features 只接受 1D
        wav = wav.mean(axis=1).astype(np.float32)

    return classify_waveform(wav, sr, session=session, class_names=class_names)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        with open(sys.argv[1], "rb") as f:
            raw = f.read()
        result = classify_audio_bytes(raw)
        for name, prob in result:
            print(f"{name:12s}: {prob:.4f}")
        print("sum:", sum(p for _, p in result))
