# -*- coding: utf-8 -*-
"""
嬰兒哭聲分類訓練管線。
log-mel 特徵（見 features.py）-> 小型 CNN（PyTorch）-> 匯出 ONNX + 誠實評估報告。

執行：
    PYTHONUTF8=1 .venv/Scripts/python.exe src/train.py
"""
from __future__ import annotations

import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import extract_features, load_waveform, fix_length, SR, TARGET_LEN, N_MELS  # noqa: E402

# ---- 路徑常數 ----
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(
    PROJECT_ROOT, "data", "donateacry-corpus",
    "donateacry_corpus_cleaned_and_updated_data",
)
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")

SEED = 42
BATCH_SIZE = 16
MAX_EPOCHS = 60
PATIENCE = 10          # val macro-F1 早停耐心值
LR = 1e-3
WEIGHT_DECAY = 1e-4
VAL_RATIO = 0.2


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def list_samples():
    """回傳 (filepaths, labels, class_names)。class_names 依資料夾名稱字母序。"""
    class_names = sorted(
        d for d in os.listdir(DATA_ROOT)
        if os.path.isdir(os.path.join(DATA_ROOT, d))
    )
    filepaths = []
    labels = []
    for idx, cname in enumerate(class_names):
        cdir = os.path.join(DATA_ROOT, cname)
        for fn in sorted(os.listdir(cdir)):
            if fn.lower().endswith(".wav"):
                filepaths.append(os.path.join(cdir, fn))
                labels.append(idx)
    return filepaths, labels, class_names


# ---------------- 波形層資料增強（僅訓練集）----------------

def augment_waveform(wav: np.ndarray, sr: int = SR) -> np.ndarray:
    """隨機 time-shift + 高斯噪音 + 隨機增益。"""
    wav = wav.copy()

    # 隨機 time-shift（最多 ±0.2 秒）
    max_shift = int(0.2 * sr)
    if max_shift > 0:
        shift = random.randint(-max_shift, max_shift)
        wav = np.roll(wav, shift)

    # 隨機增益（0.8x ~ 1.2x）
    gain = random.uniform(0.8, 1.2)
    wav = wav * gain

    # 加高斯噪音（訊噪比隨機）
    if random.random() < 0.8:
        noise_level = random.uniform(0.001, 0.01)
        noise = np.random.randn(*wav.shape).astype(np.float32) * noise_level
        wav = wav + noise

    return wav.astype(np.float32)


class CryDataset(Dataset):
    def __init__(self, filepaths, labels, train: bool):
        self.filepaths = filepaths
        self.labels = labels
        self.train = train
        # 快取原始波形（資料量小，457 筆 x 5 秒 x 16kHz 記憶體可負擔），
        # 避免每個 epoch 重複做 I/O + resample。
        self._wav_cache = {}

    def __len__(self):
        return len(self.filepaths)

    def _get_wav(self, path):
        if path not in self._wav_cache:
            wav = load_waveform(path, sr=SR)
            wav = fix_length(wav, TARGET_LEN)
            self._wav_cache[path] = wav
        return self._wav_cache[path].copy()

    def __getitem__(self, idx):
        path = self.filepaths[idx]
        label = self.labels[idx]
        wav = self._get_wav(path)

        if self.train:
            wav = augment_waveform(wav, sr=SR)

        feat = extract_features(wav, sr=SR)  # (1, N_MELS, T)
        return torch.from_numpy(feat), label


class CryCNN(nn.Module):
    """小型 CNN：3~4 個 conv block + adaptive pooling + FC，參數量 < 1M。"""

    def __init__(self, n_classes: int, n_mels: int = N_MELS):
        super().__init__()

        def block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.conv1 = block(1, 16)
        self.conv2 = block(16, 32)
        self.conv3 = block(32, 64)
        self.conv4 = block(64, 64)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for feats, labels in loader:
            feats = feats.to(device)
            logits = model(feats)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())
    return np.array(all_labels), np.array(all_preds)


def main():
    t_start = time.time()
    set_seed(SEED)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    device = torch.device("cpu")

    filepaths, labels, class_names = list_samples()
    n_classes = len(class_names)
    print(f"總樣本數: {len(filepaths)}, 類別: {class_names}")
    counts = {c: labels.count(i) for i, c in enumerate(class_names)}
    print(f"各類別樣本數: {counts}")

    # 分層 80/20 split
    train_idx, val_idx = train_test_split(
        list(range(len(filepaths))),
        test_size=VAL_RATIO,
        stratify=labels,
        random_state=SEED,
    )
    train_paths = [filepaths[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_paths = [filepaths[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]

    print(f"train: {len(train_paths)}, val: {len(val_paths)}")
    train_val_counts = {
        c: (train_labels.count(i), val_labels.count(i))
        for i, c in enumerate(class_names)
    }
    print(f"各類別 train/val 分布: {train_val_counts}")

    train_ds = CryDataset(train_paths, train_labels, train=True)
    val_ds = CryDataset(val_paths, val_labels, train=False)

    # WeightedRandomSampler 過採樣少數類（僅訓練集）。
    # 注意（實測踩坑）：本資料集類別比例極端（hungry 305 筆 vs burping 僅 6 筆訓練樣本，
    # 比例約 51:1）。若用「完全反頻率」權重讓 sampler 把各類別拉到每 epoch 均等曝光，
    # hungry 每 epoch 的曝光量會被壓到跟 burping 一樣少，且 burping 少量樣本被
    # 大量重複抽樣＋高強度波形增強，訓練出的模型會學到「多數類幾乎不出現」的錯誤先驗，
    # 導致 val 上 hungry recall 崩至 0（已實測驗證，非理論推測）。
    # 因此對 sampler 權重做 sqrt 弱化：仍過採樣少數類，但不完全拉平到 1/N，
    # 讓多數類在訓練中仍保有與其真實佔比相稱的曝光量。
    class_sample_count = np.array(
        [train_labels.count(i) for i in range(n_classes)]
    )
    class_sample_count = np.clip(class_sample_count, 1, None)
    weight_per_class = np.sqrt(1.0 / class_sample_count)
    sample_weights = np.array([weight_per_class[l] for l in train_labels])
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    # CrossEntropyLoss 加 class weights（用整體訓練分布的反頻率，同樣 sqrt 弱化）。
    # sampler 與 loss weight 是兩個獨立機制，皆對少數類做補償；兩者都用 sqrt 弱化
    # 而非完全反頻率，避免補償量疊加後過度矯正（見上方 sampler 註解的實測結果）。
    inv_freq = (1.0 / class_sample_count) * class_sample_count.sum() / n_classes
    class_weights = torch.FloatTensor(np.sqrt(inv_freq))
    print(f"class weights (sqrt-dampened): {class_weights.tolist()}")

    model = CryCNN(n_classes=n_classes).to(device)
    n_params = count_params(model)
    print(f"模型參數量: {n_params}")
    if n_params >= 1_000_000:
        print("[WARN] 參數量超過 1M，超出規格上限，但仍繼續訓練。")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_macro_f1 = -1.0
    best_state = None
    best_epoch = -1
    epochs_no_improve = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for feats, batch_labels in train_loader:
            feats = feats.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            logits = model(feats)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        y_true, y_pred = evaluate(model, val_loader, device)
        val_acc = accuracy_score(y_true, y_pred)
        val_macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

        elapsed = time.time() - t_start
        print(
            f"epoch {epoch:02d}/{MAX_EPOCHS} | loss={avg_loss:.4f} | "
            f"val_acc={val_acc:.4f} | val_macro_f1={val_macro_f1:.4f} | "
            f"elapsed={elapsed:.1f}s"
        )

        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= PATIENCE:
            print(f"早停於 epoch {epoch}（val macro-F1 連續 {PATIENCE} 輪未提升）")
            break

        if elapsed > 20 * 60:
            print("[WARN] 已超過 20 分鐘時間預算，強制停止訓練。")
            break

    print(f"最佳 epoch: {best_epoch}, 最佳 val macro-F1: {best_macro_f1:.4f}")
    model.load_state_dict(best_state)

    # ---- 最終評估（用最佳權重）----
    y_true, y_pred = evaluate(model, val_loader, device)
    final_acc = accuracy_score(y_true, y_pred)
    final_macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(n_classes)), zero_division=0
    )

    per_class = {}
    for i, cname in enumerate(class_names):
        per_class[cname] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }

    metrics = {
        "accuracy": float(final_acc),
        "macro_f1": float(final_macro_f1),
        "per_class": per_class,
        "best_epoch": int(best_epoch),
        "n_train": len(train_paths),
        "n_val": len(val_paths),
        "class_names": class_names,
        "model_params": int(n_params),
    }

    metrics_path = os.path.join(MODELS_DIR, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"已寫入 {metrics_path}")

    labels_path = os.path.join(MODELS_DIR, "labels.json")
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)
    print(f"已寫入 {labels_path}")

    # ---- 混淆矩陣圖 ----
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix (val, acc={final_acc:.3f}, macro-F1={final_macro_f1:.3f})")
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="black" if cm[i, j] < cm.max() / 2 else "white")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    cm_path = os.path.join(REPORTS_DIR, "confusion_matrix.png")
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"已寫入 {cm_path}")

    # ---- 匯出 ONNX（dynamic batch）----
    model.eval()
    # 用實際特徵抽取結果決定 dummy input 的 T 維度，確保與推論端一致
    sample_feat = extract_features(train_paths[0])
    dummy_input = torch.from_numpy(sample_feat).unsqueeze(0)  # (1, 1, N_MELS, T)

    onnx_path = os.path.join(MODELS_DIR, "cry_cnn.onnx")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch", 3: "time"},
            "logits": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,  # torch 2.12 預設走 dynamo exporter，需要未安裝的 onnxscript；
                        # 本專案禁止裝新套件，改用穩定的 legacy TorchScript-based exporter。
    )
    print(f"已寫入 {onnx_path}")

    total_elapsed = time.time() - t_start
    print(f"訓練總耗時: {total_elapsed:.1f}s")
    print(f"最終 val accuracy: {final_acc:.4f}")
    print(f"最終 val macro-F1: {final_macro_f1:.4f}")
    print(f"各類別 support (val): { {c: per_class[c]['support'] for c in class_names} }")


if __name__ == "__main__":
    main()
