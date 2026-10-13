# 基于机器学习的恶意网络流量检测

一个完整的机器学习二分类流水线：**数据加载 → 特征工程 → 模型训练（Random Forest vs XGBoost）→ 评估 → joblib 持久化 → Gradio Web 部署**，基于 UNSW-NB15 真实入侵检测数据集识别恶意网络流量。

## 系统链路

```
UNSW-NB15 流量数据 ─▶ 特征工程（Label Encoding + StandardScaler）─▶ 训练对比（RF vs XGBoost）
        ─▶ 评估（Acc / P / R / F1）─▶ joblib 持久化 ─▶ Gradio 单条 / 批量预测
```

## 核心亮点

1. **双模型对比选优**：Random Forest（100 棵、max_depth 15）与 XGBoost（100 棵、max_depth 6）在同一测试集上对比 accuracy / precision / recall / F1，自动按 F1 选优保存。
2. **完整预处理链**：类别特征（proto / service / state）Label Encoding、数值特征 StandardScaler，编码器与特征列名随模型一起持久化，推理时严格对齐，避免编码错位。
3. **模型持久化 + Web 部署**：joblib 保存 `best_model.pkl / scaler.pkl / encoders.pkl / feature_names.pkl`，Gradio 提供「单条检测 / 批量 CSV 检测 / 模型信息」三个面板，单条检测支持一键填入真实正常 / 恶意样本。

## 技术栈

| 层 | 选型 |
|---|---|
| 数据 | UNSW-NB15（官方：research.unsw.edu.au；镜像自动下载，82,332 条训练流记录） |
| 特征工程 | pandas · numpy · LabelEncoder · StandardScaler（42 维特征） |
| 模型 | scikit-learn RandomForest · XGBoost |
| 评估 | accuracy / precision / recall / F1 · 分层切分（stratify） |
| 部署 | joblib 持久化 · Gradio |

## 量化结果（真实复现）

UNSW-NB15 训练集 82,332 条（正常 37,000 / 恶意 45,332），训练集内 80/20 分层切分（train 65,865 / test 16,467）。

| 模型 | Accuracy | Precision | Recall | F1 |
|---|---|---|---|
| Random Forest | 0.9727 | 0.9838 | 0.9663 | 0.9750 |
| **XGBoost（选用）** | **0.9735** | **0.9834** | **0.9682** | **0.9758** |

> 复现：`python train.py`（自动下载 UNSW-NB15 训练集 → 特征工程 → 训练对比 → 保存最优模型）。

## 目录结构

```
traffic-detection/
├── train.py      # 数据下载 + 特征工程 + RF vs XGBoost 训练对比
├── app.py        # Gradio Web（单条检测 / 批量检测 / 模型信息）
├── data/         # UNSW_NB15_training-set.csv（自动下载，已 gitignore）
├── models/       # best_model.pkl / scaler.pkl / encoders.pkl / feature_names.pkl / ui_meta.json
└── .gitignore
```

## 快速开始

```bash
# 1. 安装依赖
pip install pandas numpy scikit-learn xgboost joblib gradio

# 2. 训练（自动下载 UNSW-NB15，约 82K 样本）
python train.py

# 3. 启动 Web 界面
python app.py
# 访问 http://127.0.0.1:7861
```

## 数据集

[UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset) 是 UNSW 发布的入侵检测基准数据集，含 9 类攻击（Generic / Exploits / Fuzzers / DoS / Reconnaissance / Analysis / Backdoor / Shellcode / Worms）与正常流量。本项目使用训练集 82,332 条做二分类（正常 / 恶意）。
