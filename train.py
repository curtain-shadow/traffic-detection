"""恶意网络流量检测 —— UNSW-NB15 真实数据训练（RF vs XGBoost 对比）"""
import os, json, urllib.request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import joblib

DATA_DIR = "data"
MODEL_DIR = "models"
CSV_PATH = os.path.join(DATA_DIR, "UNSW_NB15_training-set.csv")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

URLS = [
    "https://hf-mirror.com/datasets/jharrrry/UNSW-NB15/resolve/main/train.csv",
    "https://huggingface.co/datasets/jharrrry/UNSW-NB15/resolve/main/train.csv",
]


def download_dataset():
    if os.path.exists(CSV_PATH):
        print(f"数据集已存在: {CSV_PATH}")
        return CSV_PATH
    for url in URLS:
        try:
            print(f"下载数据集: {url}")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=180) as r, open(CSV_PATH, "wb") as f:
                f.write(r.read())
            print(f"下载完成 -> {CSV_PATH}")
            return CSV_PATH
        except Exception as e:
            print(f"  [失败] {e}")
    raise SystemExit("UNSW-NB15 下载失败：请手动下载 train.csv 放到 data/UNSW_NB15_training-set.csv 后重试")


def preprocess(df):
    df = df.copy()
    df = df.drop(columns=[c for c in ("id", "attack_cat") if c in df.columns])
    y = df["label"].values
    X = df.drop(columns=["label"])
    encoders = {}
    for col in X.select_dtypes(exclude="number").columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.fillna(0))
    return X_scaled, y, scaler, list(X.columns), encoders


def build_ui_meta(df, encoders):
    df = df.drop(columns=[c for c in ("id", "attack_cat") if c in df.columns])
    X = df.drop(columns=["label"])
    defaults, cat_values = {}, {}
    for col in X.columns:
        if col in encoders:
            defaults[col] = X[col].mode()[0]
            cat_values[col] = encoders[col].classes_.tolist()
        else:
            defaults[col] = float(np.median(X[col]))
    # 正常 / 恶意 各取一条示例（保留类别原始字符串）
    normal = df[df["label"] == 0].drop(columns=["label"]).iloc[0].to_dict()
    attack = df[df["label"] == 1].drop(columns=["label"]).iloc[0].to_dict()
    return {
        "defaults": defaults,
        "cat_values": cat_values,
        "normal_example": {k: (v.item() if hasattr(v, "item") else v) for k, v in normal.items()},
        "attack_example": {k: (v.item() if hasattr(v, "item") else v) for k, v in attack.items()},
    }


def train_models():
    csv_path = download_dataset()
    df = pd.read_csv(csv_path, encoding="utf-8-sig", low_memory=False)
    print(f"数据加载完成: {df.shape}  (label 恶意占比 {df['label'].mean():.4f})")

    X, y, scaler, feature_names, encoders = preprocess(df)
    print(f"特征数: {len(feature_names)}  类别特征: {list(encoders.keys())}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"切分: train={X_train.shape[0]}  test={X_test.shape[0]}")

    results = {}

    print("\n训练 Random Forest (100 trees, max_depth=15)...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    yp = rf.predict(X_test)
    results["Random Forest"] = {
        "accuracy": accuracy_score(y_test, yp), "precision": precision_score(y_test, yp),
        "recall": recall_score(y_test, yp), "f1": f1_score(y_test, yp),
    }
    print(f"  Acc={results['Random Forest']['accuracy']:.4f}  P={results['Random Forest']['precision']:.4f}  R={results['Random Forest']['recall']:.4f}  F1={results['Random Forest']['f1']:.4f}")

    print("训练 XGBoost (100 trees, max_depth=6)...")
    xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                                  random_state=42, n_jobs=-1, eval_metric="logloss")
    xgb_model.fit(X_train, y_train)
    yp = xgb_model.predict(X_test)
    results["XGBoost"] = {
        "accuracy": accuracy_score(y_test, yp), "precision": precision_score(y_test, yp),
        "recall": recall_score(y_test, yp), "f1": f1_score(y_test, yp),
    }
    print(f"  Acc={results['XGBoost']['accuracy']:.4f}  P={results['XGBoost']['precision']:.4f}  R={results['XGBoost']['recall']:.4f}  F1={results['XGBoost']['f1']:.4f}")

    best = "Random Forest" if results["Random Forest"]["f1"] >= results["XGBoost"]["f1"] else "XGBoost"
    best_model = rf if best == "Random Forest" else xgb_model
    print(f"\n最佳模型: {best}")

    joblib.dump(best_model, os.path.join(MODEL_DIR, "best_model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(feature_names, os.path.join(MODEL_DIR, "feature_names.pkl"))
    joblib.dump(encoders, os.path.join(MODEL_DIR, "encoders.pkl"))
    print(f"模型已保存到 {MODEL_DIR}/")

    ui_meta = build_ui_meta(df, encoders)
    with open(os.path.join(MODEL_DIR, "ui_meta.json"), "w", encoding="utf-8") as f:
        json.dump(ui_meta, f, ensure_ascii=False, indent=2)

    meta = {
        "samples": int(df.shape[0]), "features": len(feature_names),
        "test_size": int(X_test.shape[0]), "results": results, "best": best,
    }
    with open("metrics.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return results, best


if __name__ == "__main__":
    train_models()
