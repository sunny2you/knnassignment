# src/cifar10_knn.py
import os, json, argparse
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import matplotlib.pyplot as plt

# 고정 설정
IMAGE_SIZE = (32, 32)
RANDOM_STATE = 42

# ===== 이미지 → 피처 =====
def extract_features(img_path, mode="rgb"):
    """
    mode="rgb": 32x32x3을 평탄화(권장: CIFAR-10 원형 유지)
    mode="gray": 흑백 변환 후 평탄화
    """
    try:
        img = Image.open(img_path)
        if mode == "gray":
            img = img.convert("L").resize(IMAGE_SIZE)
            arr = np.array(img, dtype=np.float32) / 255.0
            return arr.flatten()
        else:
            img = img.convert("RGB").resize(IMAGE_SIZE)
            arr = np.array(img, dtype=np.float32) / 255.0   # (H,W,3)
            return arr.reshape(-1)                          # flatten
    except Exception:
        dim = IMAGE_SIZE[0]*IMAGE_SIZE[1]*(1 if mode=="gray" else 3)
        return np.zeros(dim, dtype=np.float32)

# ===== 데이터 로드 =====
def load_train(data_dir="data", feature_mode="rgb"):
    df = pd.read_csv(os.path.join(data_dir, "trainLabels.csv"))
    le = LabelEncoder()
    df["label_encoded"] = le.fit_transform(df["label"])
    X, y = [], []
    for _, row in df.iterrows():
        p = os.path.join(data_dir, "train", f"{row['id']}.png")
        X.append(extract_features(p, feature_mode))
        y.append(row["label_encoded"])
    return np.stack(X), np.array(y), le

def load_test_ids(data_dir="data", feature_mode="rgb"):
    sub_df = pd.read_csv(os.path.join(data_dir, "sampleSubmission.csv"))
    ids = sub_df["id"].tolist()
    Xtest = [extract_features(os.path.join(data_dir, "test", f"{_id}.png"), feature_mode) for _id in ids]
    return ids, np.stack(Xtest)

# ===== 모델/지표 =====
def fit_predict_knn(Xtr, ytr, Xte, k):
    clf = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    clf.fit(Xtr, ytr)
    return clf, clf.predict(Xte)

def compute_metrics(y_true, y_pred, le=None):
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    rep = classification_report(y_true, y_pred, target_names=(le.classes_ if le else None), zero_division=0)
    return {"accuracy": acc, "precision_macro": p, "recall_macro": r, "f1_macro": f1, "report": rep}

# ===== 1) Train/Test split =====
def run_train_test(X, y, le, k=5, test_size=0.2, out_dir="results"):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE)
    _, yhat = fit_predict_knn(Xtr, ytr, Xte, k)
    m = compute_metrics(yte, yhat, le)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "metrics_train_test.json"), "w") as f:
        json.dump({**{k:v for k,v in m.items() if k!="report"}, "k": k}, f, indent=2)
    print("=== Train/Test ===")
    print(f"k={k}")
    print(m["report"])
    print({k:v for k,v in m.items() if k!="report"})

# ===== 2) Train/Val/Test (검증으로 k 선택) =====
def run_train_val_test(X, y, le, k_candidates, val_size=0.2, test_size=0.2, out_dir="results"):
    X_rem, X_te, y_rem, y_te = train_test_split(X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE)
    X_tr, X_val, y_tr, y_val = train_test_split(X_rem, y_rem, test_size=val_size, stratify=y_rem, random_state=RANDOM_STATE)

    scores = []
    for k in k_candidates:
        _, y_val_pred = fit_predict_knn(X_tr, y_tr, X_val, k)
        scores.append((k, accuracy_score(y_val, y_val_pred)))
    scores.sort(key=lambda x: x[1], reverse=True)
    best_k = scores[0][0]

    # train+val로 재학습 → test 평가
    X_tv = np.vstack([X_tr, X_val]); y_tv = np.concatenate([y_tr, y_val])
    _, y_te_pred = fit_predict_knn(X_tv, y_tv, X_te, best_k)
    m = compute_metrics(y_te, y_te_pred, le)

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "metrics_train_val_test.json"), "w") as f:
        json.dump({"best_k": best_k, "val_scores": scores,
                   **{k:v for k,v in m.items() if k!="report"}}, f, indent=2)

    print("=== Train/Val/Test ===")
    print("Validation accuracy per k:", scores)
    print(f"Chosen k: {best_k}")
    print(m["report"])
    print({k:v for k,v in m.items() if k!="report"})

# ===== 3) 5-fold CV + k-스윕 그래프 =====
def run_cv_k_sweep(X, y, k_list, out_png="results/cv_accuracy_vs_k.png", out_csv="results/cv_accuracy_vs_k.csv"):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    means, stds = [], []
    rows = []
    for k in k_list:
        fold_acc = []
        for tr_idx, te_idx in skf.split(X, y):
            Xtr, Xte = X[tr_idx], X[te_idx]; ytr, yte = y[tr_idx], y[te_idx]
            _, yhat = fit_predict_knn(Xtr, ytr, Xte, k)
            fold_acc.append(accuracy_score(yte, yhat))
        mu, sd = float(np.mean(fold_acc)), float(np.std(fold_acc, ddof=1))
        means.append(mu); stds.append(sd)
        rows.append({"k": k, "mean_acc": mu, "std_acc": sd})
        print(f"k={k}: {mu:.4f} ± {sd:.4f}")

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    plt.figure()
    plt.errorbar(k_list, means, yerr=stds, fmt='-o')
    plt.xlabel("k"); plt.ylabel("Accuracy (5-fold mean ± std)")
    plt.title("KNN 5-fold CV Accuracy vs. k")
    plt.tight_layout(); plt.savefig(out_png, dpi=200)
    print(f"Saved plot: {out_png}\nSaved table: {out_csv}")

# ===== 4) 제출파일 생성 =====
def run_submission(le, k=5, data_dir="data", out_csv="submission.csv", feature_mode="rgb"):
    X, y, _ = load_train(data_dir, feature_mode)
    ids, Xtest = load_test_ids(data_dir, feature_mode)
    _, yhat = fit_predict_knn(X, y, Xtest, k)
    labels = le.inverse_transform(yhat)
    pd.DataFrame({"id": ids, "label": labels}).to_csv(out_csv, index=False)
    print(f"Saved {out_csv}")

# ===== CLI =====
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--mode", choices=["train_test","train_val_test","cv","submit"], required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--k_min", type=int, default=1)
    parser.add_argument("--k_max", type=int, default=29)
    parser.add_argument("--k_step", type=int, default=2)
    parser.add_argument("--feature_mode", choices=["rgb","gray"], default="rgb")
    args = parser.parse_args()

    if args.mode == "submit":
        # submission은 라벨 인코더가 필요하므로 train 로드해서 얻는다
        X, y, le = load_train(args.data_dir, args.feature_mode)
        run_submission(le, k=args.k, data_dir=args.data_dir, feature_mode=args.feature_mode)
        return

    # 나머지 모드: 학습셋 로드
    X, y, le = load_train(args.data_dir, args.feature_mode)

    if args.mode == "train_test":
        run_train_test(X, y, le, k=args.k)

    elif args.mode == "train_val_test":
        k_list = list(range(args.k_min, args.k_max+1, args.k_step))
        run_train_val_test(X, y, le, k_candidates=k_list)

    elif args.mode == "cv":
        k_list = list(range(args.k_min, args.k_max+1, args.k_step))
        run_cv_k_sweep(X, y, k_list)

if __name__ == "__main__":
    main()
