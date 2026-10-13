"""恶意网络流量检测 —— Gradio Web 界面（UNSW-NB15 真实 42 维特征）"""
import os, json
import gradio as gr
import pandas as pd
import numpy as np
import joblib

MODEL_DIR = "models"

model = joblib.load(os.path.join(MODEL_DIR, "best_model.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))
encoders = joblib.load(os.path.join(MODEL_DIR, "encoders.pkl"))
with open(os.path.join(MODEL_DIR, "ui_meta.json"), encoding="utf-8") as f:
    ui_meta = json.load(f)

DEFAULTS = ui_meta["defaults"]
CAT_VALUES = ui_meta["cat_values"]
NORMAL_EX = ui_meta["normal_example"]
ATTACK_EX = ui_meta["attack_example"]

FLOW_FEATURES = feature_names[:30]
CONTENT_FEATURES = feature_names[30:]


def safe_transform(le, vals):
    classes = set(le.classes_)
    return [le.transform([v])[0] if v in classes else -1 for v in vals]


def encode_and_predict(df):
    df = df.copy()
    for col, le in encoders.items():
        if col in df.columns:
            df[col] = safe_transform(le, df[col].astype(str).tolist())
    df = df.reindex(columns=feature_names, fill_value=0)
    X = scaler.transform(df.fillna(0).astype(float))
    probs = model.predict_proba(X)[:, 1]
    preds = (probs > 0.5).astype(int)
    return preds, probs


def predict_single(*vals):
    df = pd.DataFrame([dict(zip(feature_names, vals))])
    _, probs = encode_and_predict(df)
    p = float(probs[0])
    if p > 0.5:
        return f"⚠️ 恶意流量（置信度 {p:.1%}）", {"正常": 1 - p, "恶意": p}
    return f"✅ 正常流量（置信度 {1 - p:.1%}）", {"正常": 1 - p, "恶意": p}


def batch_predict(file):
    if file is None:
        return "请上传 CSV 文件", None
    df = pd.read_csv(file.name)
    original = df.copy()
    preds, probs = encode_and_predict(df)
    result_df = original.copy()
    result_df["预测结果"] = ["⚠️ 恶意" if p == 1 else "✅ 正常" for p in preds]
    result_df["恶意概率"] = np.round(probs, 4)
    m = int(preds.sum())
    return f"共 {len(preds)} 条，检测出 {m} 条恶意流量（{m / len(preds):.1%}）", result_df


def make_component(col):
    if col in encoders:
        return gr.Dropdown(choices=CAT_VALUES[col], value=DEFAULTS[col], label=col)
    return gr.Number(label=col, value=float(DEFAULTS[col]))


with gr.Blocks(title="恶意网络流量检测系统（UNSW-NB15）") as app:
    gr.Markdown("""
    # 🔍 基于机器学习的恶意网络流量检测系统

    基于 **UNSW-NB15** 真实入侵检测数据集（82,332 条流记录、42 维特征），对比 **Random Forest** 与 **XGBoost**，
    自动选择 F1 更优者（XGBoost，测试集 Acc ≈ 97.4% / F1 ≈ 0.976）。
    """)

    with gr.Tabs():
        with gr.Tab("📊 单条检测"):
            gr.Markdown("输入一条网络流的 42 维特征，即时判断是否为恶意流量。也可一键填入真实样本。")
            with gr.Row():
                fill_normal = gr.Button("填入正常流示例", variant="secondary")
                fill_attack = gr.Button("填入恶意流示例", variant="secondary")
                clear_btn = gr.Button("清空")

            flow_comps, content_comps = [], []
            with gr.Accordion("基础流特征", open=True):
                for i in range(0, len(FLOW_FEATURES), 3):
                    with gr.Row():
                        for c in FLOW_FEATURES[i:i + 3]:
                            flow_comps.append(make_component(c))
            with gr.Accordion("连接 / 内容特征", open=False):
                for i in range(0, len(CONTENT_FEATURES), 3):
                    with gr.Row():
                        for c in CONTENT_FEATURES[i:i + 3]:
                            content_comps.append(make_component(c))
            all_comps = flow_comps + content_comps

            btn = gr.Button("🔍 检测", variant="primary", size="lg")
            with gr.Row():
                result_text = gr.Textbox(label="检测结果", interactive=False)
                prob_chart = gr.Label(label="概率分布")

            btn.click(predict_single, all_comps, [result_text, prob_chart])
            fill_normal.click(lambda: [NORMAL_EX[c] for c in feature_names], None, all_comps)
            fill_attack.click(lambda: [ATTACK_EX[c] for c in feature_names], None, all_comps)
            clear_btn.click(lambda: [DEFAULTS[c] for c in feature_names], None, all_comps)

        with gr.Tab("📁 批量检测"):
            gr.Markdown("上传 CSV（需包含 UNSW-NB15 的 42 维特征列），逐条预测恶意概率。")
            file_input = gr.File(label="上传 CSV")
            batch_btn = gr.Button("批量检测", variant="primary")
            batch_summary = gr.Textbox(label="检测摘要")
            batch_result = gr.Dataframe(label="检测结果")
            batch_btn.click(batch_predict, [file_input], [batch_summary, batch_result])

        with gr.Tab("📈 模型信息"):
            gr.Markdown(f"""
            ### 模型
            - **算法**：Random Forest + XGBoost，自动选择 F1 更优者（当前为 XGBoost）
            - **特征数**：{len(feature_names)} 维（39 数值 + 3 类别：proto / service / state）
            - **预处理**：Label Encoding（类别特征）+ StandardScaler（数值特征）

            ### 数据集
            - **UNSW-NB15** 训练集 82,332 条（正常 37,000 / 恶意 45,332，恶意占比 55.1%）
            - 9 类攻击：Generic / Exploits / Fuzzers / DoS / Reconnaissance / Analysis / Backdoor / Shellcode / Worms
            - 切分：训练集内 80% / 20% 分层切分（train 65,865 / test 16,467）

            ### 测试集结果（test 16,467 条）
            | 模型 | Accuracy | Precision | Recall | F1 |
            |---|---|---|---|---|
            | Random Forest | 0.9727 | 0.9838 | 0.9663 | 0.9750 |
            | **XGBoost（选用）** | **0.9735** | **0.9834** | **0.9682** | **0.9758** |

            > 复现：`python train.py`（自动下载 UNSW-NB15 训练集并重训）。
            """)

if __name__ == "__main__":
    app.launch(theme=gr.themes.Soft(), server_name="0.0.0.0", server_port=7861)
