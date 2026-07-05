# -*- coding: utf-8 -*-
"""
特徵抽取單一正本（Single Source of Truth）。

train.py 與未來的 app.py / predict.py 都必須只經由本模組取得模型輸入特徵，
不得各自重寫 log-mel 抽取邏輯，避免訓練與推論端特徵不一致（train/serve skew）。

介面約定：
- 輸入：檔案路徑（str）或 已載入的波形 (np.ndarray, sr:int)
- 輸出：shape (1, N_MELS, T) 的 float32 陣列，已做 per-sample 標準化
"""
from __future__ import annotations

import numpy as np
import librosa

# ---- 常數（訓練與推論共用，不可分別定義）----
SR = 16000              # 統一取樣率 (Hz)
DURATION = 5.0          # 統一裁切/補齊長度 (秒)
N_MELS = 64             # mel filterbank 數
N_FFT = 1024
HOP_LENGTH = 256
TARGET_LEN = int(SR * DURATION)  # 樣本點數 = 80000

# repeat-pad / 取中段所需的期望 frame 數（供 shape 檢查與文件用途）
EXPECTED_T = 1 + TARGET_LEN // HOP_LENGTH


def load_waveform(path: str, sr: int = SR) -> np.ndarray:
    """讀取音檔，統一轉為單聲道、指定取樣率。"""
    wav, _ = librosa.load(path, sr=sr, mono=True)
    return wav.astype(np.float32)


def fix_length(wav: np.ndarray, target_len: int = TARGET_LEN) -> np.ndarray:
    """
    統一波形長度：
    - 不足 target_len：repeat-pad（重複波形直到補滿，非零值補零，避免靜音填充造成
      特徵分布偏移）
    - 超過 target_len：取中段（比取開頭更能涵蓋哭聲主要能量段）
    """
    n = len(wav)
    if n == 0:
        # 空音檔防呆：回傳全零，避免下游除以零
        return np.zeros(target_len, dtype=np.float32)
    if n < target_len:
        n_repeat = int(np.ceil(target_len / n))
        wav = np.tile(wav, n_repeat)
        n = len(wav)
    if n > target_len:
        start = (n - target_len) // 2
        wav = wav[start:start + target_len]
    else:
        wav = wav[:target_len]
    return wav.astype(np.float32)


def waveform_to_logmel(wav: np.ndarray, sr: int = SR) -> np.ndarray:
    """
    波形 -> log-mel 頻譜圖 -> per-sample 標準化。
    輸出 shape: (1, N_MELS, T)
    """
    wav = fix_length(wav, TARGET_LEN)
    mel = librosa.feature.melspectrogram(
        y=wav,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)

    # per-sample 標準化（每個樣本自己的 mean/std，不是跨資料集統計）
    mean = log_mel.mean()
    std = log_mel.std()
    if std < 1e-6:
        std = 1e-6
    log_mel = (log_mel - mean) / std

    log_mel = log_mel.astype(np.float32)
    return log_mel[np.newaxis, :, :]  # (1, N_MELS, T)


def extract_features(source, sr: int = SR) -> np.ndarray:
    """
    統一入口。
    source: str（檔案路徑）或 np.ndarray（已載入的波形，需另外提供 sr）
    回傳 shape (1, N_MELS, T) 的 float32 陣列。
    """
    if isinstance(source, str):
        wav = load_waveform(source, sr=sr)
    elif isinstance(source, np.ndarray):
        wav = source.astype(np.float32)
        if sr != SR:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
    else:
        raise TypeError(f"不支援的輸入型別: {type(source)}")

    return waveform_to_logmel(wav, sr=SR)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        feat = extract_features(sys.argv[1])
        print("feature shape:", feat.shape, "dtype:", feat.dtype)
        print("mean:", feat.mean(), "std:", feat.std())
