# -*- coding: utf-8 -*-
"""
研究級智慧嬰語翻譯機 2.0 — Streamlit 推論介面。

只負責：音訊輸入 → 呼叫 src/infer.py → 呈現結果與安全資訊。
不重寫任何特徵抽取或推論邏輯（唯一正本在 src/features.py、src/infer.py）。
"""
from __future__ import annotations

import os
import sys

import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from infer import (  # noqa: E402
    classify_audio_bytes,
    get_session,
    load_labels,
    load_metrics,
)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CONFUSION_MATRIX_PATH = os.path.join(PROJECT_ROOT, "reports", "confusion_matrix.png")

LOW_CONFIDENCE_THRESHOLD = 0.4

CLASS_ZH = {
    "hungry": "肚子餓",
    "tired": "想睡／累了",
    "discomfort": "身體不舒服",
    "belly_pain": "肚子痛／脹氣",
    "burping": "需要拍嗝",
}

CARE_TIPS = {
    "hungry": [
        "評估距上次餵奶的時間，是否已到下一餐的時間",
        "觀察是否有覓食反射（轉頭找奶、吸吮動作）",
        "先安撫情緒再餵食，避免嗆奶",
    ],
    "tired": [
        "檢查是否已到平常的睡眠時間或清醒時間過長",
        "移至安靜、光線較暗的環境，減少刺激",
        "嘗試包巾、輕拍或白噪音等固定的入睡儀式",
    ],
    "discomfort": [
        "檢查尿布是否濕了或髒了",
        "檢查衣物、包巾是否過緊、標籤摩擦或太熱／太冷",
        "檢查是否有異物、頭髮纏繞手指腳趾等狀況",
    ],
    "belly_pain": [
        "觀察是否伴隨縮腿、脹氣、排便狀況異常",
        "可嘗試飛機抱、順時針按摩腹部（如照護者熟悉手法）",
        "留意脹氣是否與餵食方式（奶嘴洞大小、餵奶姿勢）有關",
    ],
    "burping": [
        "直立抱起，輕拍或摩擦背部協助排氣",
        "檢查餵奶後是否已充分拍嗝",
        "若拍嗝後仍持續哭鬧，觀察是否合併吐奶、脹氣",
    ],
}

RED_FLAGS = [
    "發燒（尤其 3 個月以下嬰兒體溫 ≥ 38°C）",
    "呼吸困難、呼吸急促或有雜音",
    "膚色發青、發灰或蒼白",
    "難以喚醒、嗜睡或活動力明顯下降",
    "疑似脫水（尿量明顯減少、囟門凹陷、哭泣無淚）",
    "持續哭鬧且用盡方法仍完全無法安撫",
]

st.set_page_config(
    page_title="智慧嬰語翻譯機 2.0",
    page_icon="👶",
    layout="centered",
)


def inject_theme_css() -> None:
    st.markdown(
        """
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Noto+Serif+TC:wght@500;700&display=swap" rel="stylesheet">
        <style>
        html, body, [class*="css"] {
            font-family: "Noto Sans TC", "Microsoft JhengHei", sans-serif;
        }
        h1, h2, h3 {
            font-family: "Noto Serif TC", serif;
        }
        .result-card {
            background-color: #FFFFFF;
            border: 1px solid #E7E0D6;
            border-radius: 12px;
            padding: 2rem;
            margin: 1rem 0 1.5rem 0;
            text-align: center;
        }
        .result-card .label {
            font-family: "Noto Serif TC", serif;
            font-size: 2rem;
            font-weight: 700;
            color: #2B2620;
        }
        .result-card .prob {
            font-size: 1.1rem;
            color: #6B6459;
            margin-top: 0.3rem;
        }
        .low-confidence-banner {
            background-color: #FBEFE0;
            border: 1px solid #E8C79A;
            border-radius: 8px;
            padding: 0.9rem 1.1rem;
            color: #6B4A1E;
            margin-bottom: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def cached_session():
    return get_session()


@st.cache_resource
def cached_labels():
    return load_labels()


@st.cache_data
def cached_metrics():
    return load_metrics()


def render_result_card(top_name: str, top_prob: float) -> None:
    zh = CLASS_ZH.get(top_name, top_name)
    st.markdown(
        f"""
        <div class="result-card">
            <div class="label">{zh}</div>
            <div class="prob">信心度：{top_prob * 100:.1f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_prob_bars(ranked: list[tuple[str, float]]) -> None:
    st.markdown("#### 各類別機率")
    for name, prob in ranked:
        zh = CLASS_ZH.get(name, name)
        st.write(f"{zh}（{name}）")
        st.progress(min(max(prob, 0.0), 1.0), text=f"{prob * 100:.1f}%")


def render_care_tips(top_name: str) -> None:
    tips = CARE_TIPS.get(top_name, [])
    if not tips:
        return
    st.markdown("#### 照護建議（非醫療處置）")
    for tip in tips:
        st.markdown(f"- {tip}")


def render_safety_block() -> None:
    st.markdown("---")
    st.markdown("### 安全資訊")

    st.warning(
        "**醫療免責聲明**：本工具僅為照護輔助參考，"
        "不是醫療診斷工具，判讀結果不能取代專業醫療人員的評估。"
        "如對嬰兒健康狀況有疑慮，請諮詢兒科醫師。"
    )

    with st.expander("紅旗警訊（出現任一項請立即就醫）", expanded=True):
        for flag in RED_FLAGS:
            st.markdown(f"- {flag}")

    st.info("**適用月齡邊界**：本工具僅針對 0–6 個月嬰兒的哭聲設計與訓練，"
            "超過此月齡的幼兒哭聲型態已不同，判讀結果不具參考價值。")


def render_model_card() -> None:
    with st.expander("模型卡（Model Card）— 如實揭露訓練與評估資訊"):
        try:
            metrics = cached_metrics()
        except (OSError, ValueError) as exc:
            st.error(f"無法讀取 metrics.json：{exc}")
            return

        st.markdown(
            f"""
            - **整體 Accuracy**：{metrics.get('accuracy', float('nan')):.4f}
            - **Macro-F1**：{metrics.get('macro_f1', float('nan')):.4f}
            - **訓練樣本數**：{metrics.get('n_train', '未知')}
            - **驗證樣本數**：{metrics.get('n_val', '未知')}
            """
        )

        per_class = metrics.get("per_class", {})
        if per_class:
            st.markdown("**各類別詳細指標**")
            rows = []
            for cname, stats in per_class.items():
                rows.append(
                    {
                        "類別": f"{CLASS_ZH.get(cname, cname)}（{cname}）",
                        "precision": round(stats.get("precision", 0.0), 4),
                        "recall": round(stats.get("recall", 0.0), 4),
                        "f1": round(stats.get("f1", 0.0), 4),
                        "support": stats.get("support", 0),
                    }
                )
            st.dataframe(rows, use_container_width=True, hide_index=True)

        if os.path.isfile(CONFUSION_MATRIX_PATH):
            st.image(CONFUSION_MATRIX_PATH, caption="混淆矩陣（驗證集）")
        else:
            st.write("（找不到混淆矩陣圖檔）")

        st.markdown(
            "**資料集與限制說明**：模型以公開資料集 "
            "[donateacry-corpus](https://github.com/gveres/donateacry-corpus) 訓練，"
            "共約 457 筆錄音，五類樣本數嚴重不平衡（`hungry` 佔絕大多數，"
            "`burping`、`tired`、`belly_pain` 樣本極少）。"
            "整體 accuracy 偏高主要來自多數類別 `hungry` 的貢獻，"
            "少數類別的 precision／recall 偏低、可信度有限，"
            "macro-F1 更能反映模型在少數類別上的實際表現不足。"
            "本模型為研究與展示用途，**不宣稱**能「精準辨識」或「讀懂」嬰兒需求，"
            "使用時請務必搭配現場觀察與照護者的判斷。"
        )


def main() -> None:
    inject_theme_css()

    st.title("智慧嬰語翻譯機 2.0")
    st.caption("研究級嬰兒哭聲判讀輔助工具（0–6 個月適用）")

    st.markdown(
        "錄音或上傳一段嬰兒哭聲音檔，模型會嘗試判讀最可能的需求。"
        "**本工具僅供照護參考，不是醫療診斷。**"
    )

    audio_bytes: bytes | None = None

    recorded = st.audio_input("錄音（使用麥克風）")
    if recorded is not None:
        audio_bytes = recorded.getvalue()

    uploaded = st.file_uploader(
        "或上傳音檔", type=["wav", "mp3", "ogg", "flac"]
    )
    if uploaded is not None:
        audio_bytes = uploaded.getvalue()

    if audio_bytes is None:
        st.info("請先錄音或上傳音檔以進行判讀。")
        render_safety_block()
        render_model_card()
        return

    st.audio(audio_bytes)

    with st.spinner("判讀中…"):
        try:
            session = cached_session()
            class_names = cached_labels()
            ranked = classify_audio_bytes(
                audio_bytes, session=session, class_names=class_names
            )
        except Exception as exc:  # noqa: BLE001 — 對使用者顯示可讀錯誤
            st.error(f"判讀失敗，請確認音檔格式是否正確。錯誤訊息：{exc}")
            render_safety_block()
            render_model_card()
            return

    top_name, top_prob = ranked[0]

    if top_prob < LOW_CONFIDENCE_THRESHOLD:
        st.markdown(
            '<div class="low-confidence-banner">'
            "判讀信心不足，請以現場觀察為準。"
            "以下結果僅供參考，不建議單獨依賴此判讀做照護決策。"
            "</div>",
            unsafe_allow_html=True,
        )

    render_result_card(top_name, top_prob)
    render_prob_bars(ranked)
    render_care_tips(top_name)
    render_safety_block()
    render_model_card()


if __name__ == "__main__":
    main()
