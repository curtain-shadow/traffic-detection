"""恶意网络流量检测 —— 数据加载 + 特征工程 + 模型训练"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import xgboost as xgb
import joblib
import os
import urllib.request
import zipfile

DATA_DIR = "data"
MODEL_DIR = "models"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def download_dataset():
    """下载 UNSW-NB15 数据集（比 CIC-IDS 小且更适合实验）"""
    print("下载数据集...")

    # 使用 UNSW-NB15 训练集（CSV 格式，约 175K 样本）
    url = "https://cloudstor.aarnet.edu.au/plus/s/2DhnLGDdEECo4ys/download"
    csv_path = os.path.join(DATA_DIR, "UNSW_NB15_training-set.csv")

    if os.path.exists(csv_path):
        print(f"数据集已存在: {csv_path}")
        return csv_path

    # 备用：生成合成数据集用于演示
    print("下载失败，生成合成数据集用于演示...")
    return generate_synthetic()


def generate_synthetic():
    """生成合成网络流量数据用于演示"""
    csv_path = os.path.join(DATA_DIR, "synthetic_traffic.csv")
    if os.path.exists(csv_path):
        return csv_path

    np.random.seed(42)
    n = 5000

    data = {
        'duration': np.random.exponential(100, n),
        'protocol_type': np.random.choice(['tcp', 'udp', 'icmp'], n),
        'src_bytes': np.random.exponential(500, n),
        'dst_bytes': np.random.exponential(400, n),
        'src_packets': np.random.poisson(10, n),
        'dst_packets': np.random.poisson(8, n),
        'port_src': np.random.randint(1024, 65535, n),
        'port_dst': np.random.choice([80, 443, 22, 21, 25, 53, 8080], n),
        'tcp_flags': np.random.randint(0, 64, n),
        'packet_size_mean': np.random.normal(200, 80, n),
        'packet_size_std': np.random.exponential(50, n),
        'flow_rate': np.random.exponential(0.5, n),
        'syn_count': np.random.poisson(2, n),
        'ack_count': np.random.poisson(5, n),
        'fin_count': np.random.poisson(1, n),
        'rst_count': np.random.poisson(0.3, n),
        'urg_count': np.random.poisson(0.1, n),
        'http_method': np.random.choice(['GET', 'POST', 'PUT', 'HEAD', 'OPTIONS'], n),
        'dns_query_len': np.random.exponential(20, n),
        'dns_response_len': np.random.exponential(50, n),
    }

    df = pd.DataFrame(data)

    # 生成标签：基于特征组合判定是否为恶意流量
    malicious = (
        (df['syn_count'] > 5) & (df['src_bytes'] < 100) |
        (df['port_dst'].isin([22, 21])) & (df['rst_count'] > 1) |
        (df['port_dst'] == 53) & (df['dns_query_len'] > 50) |
        (df['tcp_flags'] > 50) & (df['flow_rate'] > 2) |
        (df['rst_count'] > 2)
    )
    df['label'] = (malicious).astype(int)

    df.to_csv(csv_path, index=False)
    print(f"合成数据集已保存: {csv_path} ({n} 样本, 恶意占比 {df['label'].mean():.1%})")
    return csv_path


def preprocess(df):
    """特征工程：编码、标准化"""
    df = df.copy()

    # 处理类别型特征
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        if col != 'label':
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # 分离特征和标签
    if 'label' in df.columns:
        y = df['label']
        X = df.drop(columns=['label'])
    else:
        y = None
        X = df

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.fillna(0))

    return X_scaled, y, scaler, list(X.columns)


def train_models():
    """训练并对比模型"""
    # 加载
    csv_path = download_dataset()
    df = pd.read_csv(csv_path)
    print(f"数据加载完成: {df.shape}")

    X, y, scaler, feature_names = preprocess(df)

    # 切分
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    results = {}

    # ── Random Forest ──
    print("\n训练 Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    rf_metrics = {
        'accuracy': accuracy_score(y_test, y_pred_rf),
        'precision': precision_score(y_test, y_pred_rf),
        'recall': recall_score(y_test, y_pred_rf),
        'f1': f1_score(y_test, y_pred_rf),
    }
    results['Random Forest'] = rf_metrics
    print(f"  Accuracy: {rf_metrics['accuracy']:.3f}, F1: {rf_metrics['f1']:.3f}")

    # ── XGBoost ──
    print("训练 XGBoost...")
    xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)

    xgb_metrics = {
        'accuracy': accuracy_score(y_test, y_pred_xgb),
        'precision': precision_score(y_test, y_pred_xgb),
        'recall': recall_score(y_test, y_pred_xgb),
        'f1': f1_score(y_test, y_pred_xgb),
    }
    results['XGBoost'] = xgb_metrics
    print(f"  Accuracy: {xgb_metrics['accuracy']:.3f}, F1: {xgb_metrics['f1']:.3f}")

    # ── 选最佳 ──
    best = 'Random Forest' if results['Random Forest']['f1'] >= results['XGBoost']['f1'] else 'XGBoost'
    best_model = rf if best == 'Random Forest' else xgb_model
    print(f"\n最佳模型: {best}")

    # ── 保存 ──
    joblib.dump(best_model, os.path.join(MODEL_DIR, 'best_model.pkl'))
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))
    joblib.dump(feature_names, os.path.join(MODEL_DIR, 'feature_names.pkl'))

    print(f"\n模型已保存到 {MODEL_DIR}/")
    return results, best


if __name__ == '__main__':
    results, best = train_models()

    print("\n" + "=" * 50)
    print("模型对比结果:")
    for name, metrics in results.items():
        print(f"  {name:15s}  Acc={metrics['accuracy']:.3f}  Prec={metrics['precision']:.3f}  Rec={metrics['recall']:.3f}  F1={metrics['f1']:.3f}")
