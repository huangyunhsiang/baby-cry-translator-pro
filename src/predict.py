# -*- coding: utf-8 -*-
"""
CLI 推論工具，同時作為 ONNX 匯出的 sanity check。

用法：
    PYTHONUTF8=1 .venv/Scripts/python.exe src/predict.py <音檔路徑>
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import extract_features  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
ONNX_PATH = os.path.join(MODELS_DIR, "cry_cnn.onnx")
LABELS_PATH = os.path.join(MODELS_DIR, "labels.json")


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()


def load_labels():
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def predict(audio_path: str):
    class_names = load_labels()
    session = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    feat = extract_features(audio_path)          # (1, N_MELS, T)
    feat_batch = feat[np.newaxis, :, :, :]        # (1, 1, N_MELS, T)

    outputs = session.run(None, {input_name: feat_batch})
    logits = outputs[0][0]
    probs = softmax(logits)

    ranked = sorted(zip(class_names, probs), key=lambda x: x[1], reverse=True)
    return ranked


def main():
    parser = argparse.ArgumentParser(description="嬰兒哭聲分類推論 (ONNX)")
    parser.add_argument("audio_path", help="音檔絕對或相對路徑 (.wav)")
    args = parser.parse_args()

    if not os.path.isfile(args.audio_path):
        print(f"錯誤：找不到檔案 {args.audio_path}")
        sys.exit(1)

    ranked = predict(args.audio_path)

    print(f"音檔: {args.audio_path}")
    print("類別機率（由高到低）：")
    total = 0.0
    for cname, prob in ranked:
        print(f"  {cname:12s}: {prob:.4f}")
        total += float(prob)
    print(f"機率總和: {total:.6f}")


if __name__ == "__main__":
    main()
