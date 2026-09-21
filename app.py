"""恶意网络流量检测 —— Gradio Web 界面"""
import gradio as gr
import pandas as pd
import numpy as np
import joblib
import os

MODEL_DIR = "models"

# 加载模型
model = joblib.load(os.path.join(MODEL_DIR, 'best_model.pkl'))
scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
feature_names = joblib.load(os.path.join(MODEL_DIR, 'feature_names.pkl'))


def predict(
    duration, protocol_type, src_bytes, dst_bytes,
    src_packets, dst_packets, port_src, port_dst,
    tcp_flags, packet_size_mean, packet_size_std,
    flow_rate, syn_count, ack_count, fin_count,
    rst_count, urg_count, http_method,
    dns_query_len, dns_response_len,
):
    """单条预测"""
    # 构建 DataFrame
    data = {
        'duration': [float(duration)],
        'protocol_type': [protocol_type.lower()],
        'src_bytes': [float(src_bytes)],
        'dst_bytes': [float(dst_bytes)],
        'src_packets': [int(src_packets)],
        'dst_packets': [int(dst_packets)],
        'port_src': [int(port_src)],
        'port_dst': [int(port_dst)],
        'tcp_flags': [int(tcp_flags)],
        'packet_size_mean': [float(packet_size_mean)],
        'packet_size_std': [float(packet_size_std)],
        'flow_rate': [float(flow_rate)],
        'syn_count': [int(syn_count)],
        'ack_count': [int(ack_count)],
        'fin_count': [int(fin_count)],
        'rst_count': [int(rst_count)],
        'urg_count': [int(urg_count)],
        'http_method': [http_method.upper()],
        'dns_query_len': [float(dns_query_len)],
        'dns_response_len': [float(dns_response_len)],
    }
    df = pd.DataFrame(data)

    # 编码类别特征
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col] = pd.factorize(df[col])[0]

    # 对齐特征
    df = df.reindex(columns=feature_names, fill_value=0)
    X = scaler.transform(df.fillna(0))

    # 预测
    prob = model.predict_proba(X)[0]
    pred = int(prob[1] > 0.5)

    if pred == 1:
        result = f"⚠️ 恶意流量（置信度 {prob[1]:.1%}）"
    else:
        result = f"✅ 正常流量（置信度 {prob[0]:.1%}）"

    return result, {"正常": float(prob[0]), "恶意": float(prob[1])}


def batch_predict(file):
    """批量 CSV 预测"""
    if file is None:
        return "请上传 CSV 文件", None

    df = pd.read_csv(file.name)
    original = df.copy()

    # 同 predict 流程
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col] = pd.factorize(df[col].astype(str))[0]

    df = df.reindex(columns=feature_names, fill_value=0)
    X = scaler.transform(df.fillna(0))
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]

    result_df = original.copy()
    result_df['预测结果'] = ['⚠️ 恶意' if p == 1 else '✅ 正常' for p in preds]
    result_df['恶意概率'] = probs

    malicious_count = int(sum(preds))
    total = len(preds)
    summary = f"共 {total} 条，检测出 {malicious_count} 条恶意流量（{malicious_count/total:.1%}）"

    return summary, result_df


# ── Gradio 界面 ──
with gr.Blocks(title="恶意网络流量检测系统") as app:
    gr.Markdown("""
    # 🔍 基于机器学习的恶意网络流量检测系统

    使用 **Random Forest** 与 **XGBoost** 模型，对网络流量特征进行分类，识别恶意流量。
    """)

    with gr.Tabs():
        with gr.Tab("📊 单条检测"):
            gr.Markdown("输入网络流量的特征值，即时判断是否为恶意流量。")

            with gr.Row():
                with gr.Column():
                    duration = gr.Number(label="持续时间 (ms)", value=50)
                    protocol_type = gr.Dropdown(['tcp', 'udp', 'icmp'], label="协议类型", value='tcp')
                    src_bytes = gr.Number(label="源字节数", value=200)
                    dst_bytes = gr.Number(label="目标字节数", value=300)
                    src_packets = gr.Number(label="源包数", value=10)
                    dst_packets = gr.Number(label="目标包数", value=8)
                with gr.Column():
                    port_src = gr.Number(label="源端口", value=52000)
                    port_dst = gr.Number(label="目标端口", value=80)
                    tcp_flags = gr.Number(label="TCP 标志位", value=2)
                    packet_size_mean = gr.Number(label="平均包大小", value=200)
                    packet_size_std = gr.Number(label="包大小标准差", value=50)
                    flow_rate = gr.Number(label="流速率", value=0.5)

            with gr.Accordion("高级特征", open=False):
                with gr.Row():
                    syn_count = gr.Number(label="SYN 计数", value=2)
                    ack_count = gr.Number(label="ACK 计数", value=5)
                    fin_count = gr.Number(label="FIN 计数", value=1)
                    rst_count = gr.Number(label="RST 计数", value=0)
                    urg_count = gr.Number(label="URG 计数", value=0)
                with gr.Row():
                    http_method = gr.Dropdown(['GET', 'POST', 'PUT', 'HEAD'], label="HTTP 方法", value='GET')
                    dns_query_len = gr.Number(label="DNS 查询长度", value=20)
                    dns_response_len = gr.Number(label="DNS 响应长度", value=50)

            with gr.Row():
                btn = gr.Button("🔍 检测", variant="primary", size="lg")

            with gr.Row():
                result_text = gr.Textbox(label="检测结果", interactive=False)
                prob_chart = gr.Label(label="概率分布")

            btn.click(
                predict,
                [duration, protocol_type, src_bytes, dst_bytes,
                 src_packets, dst_packets, port_src, port_dst,
                 tcp_flags, packet_size_mean, packet_size_std,
                 flow_rate, syn_count, ack_count, fin_count,
                 rst_count, urg_count, http_method,
                 dns_query_len, dns_response_len],
                [result_text, prob_chart],
            )

        with gr.Tab("📁 批量检测"):
            gr.Markdown("上传 CSV 文件进行批量检测（需包含与训练数据相同的特征列）。")
            file_input = gr.File(label="上传 CSV")
            batch_btn = gr.Button("批量检测")
            batch_summary = gr.Textbox(label="检测摘要")
            batch_result = gr.Dataframe(label="检测结果")

            batch_btn.click(batch_predict, [file_input], [batch_summary, batch_result])

        with gr.Tab("📈 模型信息"):
            gr.Markdown("""
            ### 模型架构
            - **算法**：Random Forest + XGBoost，自动选择 F1 更优者
            - **特征数**：{} 个
            - **预处理**：Label Encoding（类别特征）+ StandardScaler（数值特征）

            ### 数据集
            - 使用合成网络流量数据（基于真实攻击模式生成）
            - 训练集 / 测试集：80% / 20%

            ### 特征列表
            {}、port_src、port_dst、tcp_flags 等共 {} 个特征
            """.format(len(feature_names), ', '.join(feature_names[:8]), len(feature_names)))

if __name__ == '__main__':
    app.launch(theme=gr.themes.Soft(), server_name='0.0.0.0', server_port=7861)
