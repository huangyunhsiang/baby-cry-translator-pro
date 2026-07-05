# 智慧嬰語翻譯機 2.0（Baby Cry Translator Pro）

以真實機器學習模型判讀 0–6 個月嬰兒哭聲需求類別的研究級原型。
使用公開資料集訓練聲學分類模型（log-mel 頻譜圖 + CNN），本機 ONNX 推論，Streamlit 網頁介面，可部署至 Streamlit Community Cloud。

> 前代版本：`033.smart-baby-cry-translator`（前端啟發式規則 PWA）。本版差異：改用真實訓練的 ML 模型，並附完整評估報告。

## 重要邊界（請先閱讀）

- 本工具是**照護輔助參考**，不是醫療診斷，不取代兒科醫師、護理師或急診判斷。
- 適用對象為 **0–6 個月**的反射性哭聲階段；月齡越大，哭聲越個體化、意圖化，分類參考價值越低。
- 若嬰兒出現發燒、呼吸困難、膚色發青／發灰、難以喚醒、脫水、持續無法安撫等警訊，**應立即尋求醫療協助**。

## 安裝與啟動

```powershell
cd C:\Users\USER\Projects\064.baby-cry-translator-pro
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt   # 推論（App）用
.venv\Scripts\python -m streamlit run app.py
```

訓練（重現模型）需另裝訓練依賴：

```powershell
.venv\Scripts\python -m pip install -r requirements-train.txt
.venv\Scripts\python src\train.py
```

## 模型概要

- **資料集**：[donateacry-corpus](https://github.com/gveres/donateacry-corpus)（公開嬰兒哭聲語料，5 類：hungry / tired / discomfort / belly_pain / burping）
- **特徵**：log-mel spectrogram
- **模型**：小型 CNN，類別加權 + 資料增強處理類別不平衡
- **推論**：匯出 ONNX，App 端僅需 onnxruntime（無需 PyTorch）
- **誠實限制**：語料類別極度不平衡（hungry 佔絕大多數）、標注由捐贈者自報未經臨床驗證；實際辨識力以 `models/metrics.json` 與 `reports/` 的混淆矩陣為準，請勿高估

## 理論依據

- Barr, R. G. (1990). The normal crying curve: What do we know? *Developmental Medicine & Child Neurology, 32*(4), 356–362.
- LaGasse, L. L., Neal, A. R., & Lester, B. M. (2005). Assessment of infant cry: Acoustic cry analysis and parental perception. *Mental Retardation and Developmental Disabilities Research Reviews, 11*(1), 83–93.
- Ji, C., Mudiyanselage, T. B., Gao, Y., & Pan, Y. (2021). A review of infant cry analysis and classification. *EURASIP Journal on Audio, Speech, and Music Processing, 2021*, Article 8.

## 部署（Streamlit Community Cloud）

1. Push 本 repo 至 GitHub（`data/` 與 `.venv/` 已在 .gitignore，模型檔 `models/*.onnx` 需入版控）
2. share.streamlit.io 連結 repo，主檔 `app.py`，依賴檔 `requirements.txt`
