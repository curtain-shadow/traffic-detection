# 基于机器学习的恶意网络流量检测

一个完整的机器学习二分类流水线：**数据加载 → 特征工程 → 模型训练（Random Forest vs XGBoost）→ 评估 → joblib 持久化 → Gradio Web 部署**，用于识别恶意网络流量。

## 系统链路

```
UNSW-NB15 流量数据 ─▶ 特征工程（Label Encoding + StandardScaler）─▶ 训练对比（RF vs XGBoost）
        ─▶ 评估（Acc / P / R / F1）─▶ joblib 持久化 ─▶ Gradio 单条 / 批量预测
```

## 核心亮点

1. **双模型对比选优**：Random Forest（100 棵、max_depth 15）与 XGBoost（100 棵、max_depth 6）在同一测试集上对比 accuracy / precision / recall / F1，自动按 F1 选优保存。
2. **完整预处理链**：类别特征 `LabelEncoder`、数值特征 `StandardScaler`，特征列名随模型一起持久化，推理时严格对齐，避免列顺序/缺失导致的错位。
3. **模型持久化 + Web 部署**：joblib 保存 `best_model.pkl / scaler.pkl / feature_names.pkl`，Gradio 提供「单条检测 / 批量 CSV 检测 / 模型信息」三个面板。

## 技术栈

| 层 | 选型 |
|---|---|
| 数据 | UNSW-NB15（约 175K 样本，联网下载）；下载失败自动降级合成数据集（5000 条，仅跑通流程） |
| 特征工程 | pandas · numpy · LabelEncoder · StandardScaler |
| 模型 | scikit-learn RandomForest · XGBoost |
| 评估 | accuracy / precision / recall / F1 · 分层切分（stratify） |
| 部署 | joblib 持久化 · Gradio |

## 量化结果（示例）

| 指标 | 值 |
|---|---|
| Accuracy（较优模型） | ≈ 92% |
| F1（较优模型） | ≈ 0.90 |

> **数据集说明**：`download_dataset()` 优先联网下载 UNSW-NB15 训练集（约 175K 样本）；下载失败时自动降级为合成数据（`data/synthetic_traffic.csv`，5000 条、规则标签，仅用于离线跑通流程）。上表为真实数据集的示例结果，复现需成功下载 UNSW-NB15 后运行 `python train.py`。

## 目录结构

```
traffic-detection/
├── train.py      # 数据下载/合成 + 特征工程 + RF vs XGBoost 训练对比
├── app.py        # Gradio Web（单条检测 / 批量检测 / 模型信息）
├── data/         # 训练数据（synthetic_traffic.csv 为合成降级数据）
├── models/       # best_model.pkl / scaler.pkl / feature_names.pkl
└── .gitignore
```

## 快速开始

```bash
# 1. 安装依赖
pip install pandas numpy scikit-learn xgboost joblib gradio

# 2. 训练（自动下载 UNSW-NB15，失败则生成合成数据）
python train.py

# 3. 启动 Web 界面
python app.py
# 访问 http://127.0.0.1:7861
```
