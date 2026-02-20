"""
モデル学習モジュール
LightGBM 二値分類 + ランキング（LambdaRank）
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score

from config.settings import FEATURE_COLUMNS, CATEGORICAL_FEATURES, MODEL_DIR, PROCESSED_DIR


# ============================================================
# データ分割
# ============================================================

def time_based_split(df: pd.DataFrame, val_start: str = "2025-01-01",
                     test_start: str = None) -> tuple:
    """時系列ベースの学習・検証（・テスト）分割

    Returns:
        test_start指定時: (train, val, test) の3つ
        test_start未指定: (train, val) の2つ（後方互換性）
    """
    df["race_date"] = pd.to_datetime(df["race_date"])
    val_date = pd.Timestamp(val_start)
    train = df[df["race_date"] < val_date].copy()
    if test_start:
        test_date = pd.Timestamp(test_start)
        val = df[(df["race_date"] >= val_date) & (df["race_date"] < test_date)].copy()
        test = df[df["race_date"] >= test_date].copy()
        return train, val, test
    else:
        val = df[df["race_date"] >= val_date].copy()
        return train, val


def prepare_data(df: pd.DataFrame) -> tuple:
    """DataFrame から X, y, groups, feature_names を抽出"""
    available = [c for c in FEATURE_COLUMNS if c in df.columns]
    X = df[available].values.astype(np.float32)
    y = df["top3"].values if "top3" in df.columns else None
    groups = df["race_id"].values if "race_id" in df.columns else None
    return X, y, groups, available


def _get_categorical_indices(feature_names: list) -> list:
    """カテゴリカル特徴量のインデックスを返す"""
    return [i for i, f in enumerate(feature_names) if f in CATEGORICAL_FEATURES]


# ============================================================
# 二値分類モデル
# ============================================================

DEFAULT_BINARY_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": 7,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 0.1,
    "lambda_l2": 1.0,
    "verbose": -1,
    "seed": 42,
}


def train_binary_model(train_df: pd.DataFrame, val_df: pd.DataFrame,
                       params: dict = None) -> tuple:
    """二値分類モデル（3着以内予測）を学習"""
    params = params or DEFAULT_BINARY_PARAMS.copy()

    X_train, y_train, _, feature_names = prepare_data(train_df)
    X_val, y_val, _, _ = prepare_data(val_df)

    cat_indices = _get_categorical_indices(feature_names)

    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feature_names,
                         categorical_feature=cat_indices if cat_indices else "auto")
    dval = lgb.Dataset(X_val, label=y_val, feature_name=feature_names, reference=dtrain)

    callbacks = [lgb.log_evaluation(100), lgb.early_stopping(50)]

    model = lgb.train(
        params, dtrain, num_boost_round=1000,
        valid_sets=[dtrain, dval], valid_names=["train", "val"],
        callbacks=callbacks,
    )

    y_pred = model.predict(X_val)
    auc = roc_auc_score(y_val, y_pred)
    logloss = log_loss(y_val, y_pred)
    acc = accuracy_score(y_val, (y_pred >= 0.5).astype(int))

    val_df = val_df.copy()
    val_df["pred_prob"] = y_pred
    top3_hit = _calc_top3_hit_rate(val_df)

    metrics = {
        "auc": round(auc, 4),
        "logloss": round(logloss, 4),
        "accuracy": round(acc, 4),
        "top3_hit_rate": round(top3_hit, 4),
        "num_iterations": model.best_iteration,
        "train_size": len(X_train),
        "val_size": len(X_val),
    }
    return model, metrics


def _calc_top3_hit_rate(val_df: pd.DataFrame) -> float:
    """レースごとに予測上位3頭のうち実際に3着以内だった馬の割合"""
    hits = total = 0
    for _, race_group in val_df.groupby("race_id"):
        if len(race_group) < 3:
            continue
        top3_pred = race_group.nlargest(3, "pred_prob")
        hits += top3_pred["top3"].sum()
        total += 3
    return hits / total if total > 0 else 0.0


# ============================================================
# ランキングモデル
# ============================================================

DEFAULT_RANK_PARAMS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [3, 5],
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": 7,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 0.1,
    "lambda_l2": 1.0,
    "verbose": -1,
    "seed": 42,
    "label_gain": [0, 1, 2, 3, 4],
}


def _make_rank_labels(finish_positions: np.ndarray, num_entries: int) -> np.ndarray:
    """着順をランキング用ラベルに変換（固定スケール、レース間で比較可能）"""
    labels = np.zeros(len(finish_positions), dtype=np.float32)
    labels[finish_positions == 1] = 4
    labels[finish_positions == 2] = 3
    labels[finish_positions == 3] = 2
    labels[(finish_positions >= 4) & (finish_positions <= 5)] = 1
    return labels


def train_ranking_model(train_df: pd.DataFrame, val_df: pd.DataFrame,
                        params: dict = None) -> tuple:
    """ランキングモデル（LambdaRank）を学習"""
    params = params or DEFAULT_RANK_PARAMS.copy()

    # LambdaRankはデータがグループ(race_id)順にソートされている必要がある
    train_df = train_df.sort_values("race_id").reset_index(drop=True)
    val_df = val_df.sort_values("race_id").reset_index(drop=True)

    X_train, _, _, feature_names = prepare_data(train_df)
    X_val, _, _, _ = prepare_data(val_df)

    cat_indices = _get_categorical_indices(feature_names)

    y_train_rank, train_group_sizes = [], []
    for _, group in train_df.groupby("race_id", sort=True):
        n = len(group)
        labels = _make_rank_labels(group["finish_position"].values, n)
        y_train_rank.extend(labels)
        train_group_sizes.append(n)

    y_val_rank, val_group_sizes = [], []
    for _, group in val_df.groupby("race_id", sort=True):
        n = len(group)
        labels = _make_rank_labels(group["finish_position"].values, n)
        y_val_rank.extend(labels)
        val_group_sizes.append(n)

    dtrain = lgb.Dataset(
        X_train, label=np.array(y_train_rank, dtype=np.float32),
        group=train_group_sizes, feature_name=feature_names,
        categorical_feature=cat_indices if cat_indices else "auto",
    )
    dval = lgb.Dataset(
        X_val, label=np.array(y_val_rank, dtype=np.float32),
        group=val_group_sizes, feature_name=feature_names, reference=dtrain,
    )

    callbacks = [lgb.log_evaluation(100), lgb.early_stopping(50)]

    model = lgb.train(
        params, dtrain, num_boost_round=1000,
        valid_sets=[dtrain, dval], valid_names=["train", "val"],
        callbacks=callbacks,
    )

    val_df = val_df.copy()
    val_df["pred_score"] = model.predict(X_val)
    top3_hit = _calc_top3_hit_rate_by_score(val_df)

    metrics = {
        "top3_hit_rate": round(top3_hit, 4),
        "num_iterations": model.best_iteration,
        "train_size": len(X_train),
        "val_size": len(X_val),
        "train_groups": len(train_group_sizes),
        "val_groups": len(val_group_sizes),
    }
    return model, metrics


def _calc_top3_hit_rate_by_score(val_df: pd.DataFrame) -> float:
    """ランキングモデルの3着以内的中率"""
    hits = total = 0
    for _, race_group in val_df.groupby("race_id"):
        if len(race_group) < 3:
            continue
        top3_pred = race_group.nlargest(3, "pred_score")
        hits += (top3_pred["finish_position"] <= 3).sum()
        total += 3
    return hits / total if total > 0 else 0.0


# ============================================================
# 特徴量重要度
# ============================================================

def get_feature_importance(model, feature_names: list) -> pd.DataFrame:
    """特徴量重要度を取得"""
    importance = model.feature_importance(importance_type="gain")
    fi = pd.DataFrame({
        "feature": feature_names,
        "importance": importance,
    }).sort_values("importance", ascending=False)
    fi["importance_pct"] = (fi["importance"] / fi["importance"].sum() * 100).round(2)
    return fi


# ============================================================
# Optuna ハイパーパラメータ最適化
# ============================================================

def optimize_hyperparams(train_df: pd.DataFrame, val_df: pd.DataFrame,
                         n_trials: int = 50) -> dict:
    """Optuna でハイパーパラメータ最適化"""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    X_train, y_train, _, feature_names = prepare_data(train_df)
    X_val, y_val, _, _ = prepare_data(val_df)

    cat_indices = _get_categorical_indices(feature_names)

    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feature_names,
                         categorical_feature=cat_indices if cat_indices else "auto")
    dval = lgb.Dataset(X_val, label=y_val, feature_name=feature_names, reference=dtrain)

    def objective(trial):
        params = {
            "objective": "binary",
            "metric": "auc",
            "verbose": -1,
            "seed": 42,
            "feature_pre_filter": False,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-4, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-4, 10.0, log=True),
        }
        model = lgb.train(
            params, dtrain, num_boost_round=500,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
        )
        y_pred = model.predict(X_val)
        return roc_auc_score(y_val, y_pred)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    print(f"\n  最良AUC: {study.best_value:.4f}")
    print(f"  最良パラメータ: {study.best_params}")

    best_params = DEFAULT_BINARY_PARAMS.copy()
    best_params.update(study.best_params)
    return best_params


# ============================================================
# モデル保存・読み込み
# ============================================================

def save_model(model, metrics: dict, feature_names: list,
               model_type: str, output_dir: Path = None):
    """モデルと付随情報を保存"""
    output_dir = output_dir or MODEL_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / f"{model_type}_model.txt"
    model.save_model(str(model_path))

    meta = {
        "model_type": model_type,
        "feature_names": feature_names,
        "metrics": metrics,
    }
    meta_path = output_dir / f"{model_type}_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    fi = get_feature_importance(model, feature_names)
    fi_path = output_dir / f"{model_type}_feature_importance.csv"
    fi.to_csv(fi_path, index=False)

    print(f"  モデル保存: {model_path}")
    print(f"  メタ情報: {meta_path}")
    print(f"  特徴量重要度: {fi_path}")
    return model_path


def load_model(model_type: str = "binary", model_dir: Path = None) -> tuple:
    """保存済みモデルを読み込み"""
    model_dir = model_dir or MODEL_DIR
    model_path = model_dir / f"{model_type}_model.txt"
    meta_path = model_dir / f"{model_type}_meta.json"

    model = lgb.Booster(model_file=str(model_path))
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return model, meta


# ============================================================
# 学習実行
# ============================================================

def train_all(input_path: str = None, val_start: str = "2025-01-01",
              tune: bool = False):
    """全モデルを学習"""
    input_path = input_path or str(PROCESSED_DIR / "features.csv")

    print(f"\n[データ読み込み] {input_path}")
    df = pd.read_csv(input_path, dtype={"race_id": str, "horse_id": str})
    df["race_date"] = pd.to_datetime(df["race_date"])
    print(f"  全体: {len(df)}行, {df['race_id'].nunique()}レース")

    train_df, val_df = time_based_split(df, val_start)
    print(f"  学習: {len(train_df)}行 (~{val_start})")
    print(f"  検証: {len(val_df)}行 ({val_start}~)")

    if len(train_df) == 0 or len(val_df) == 0:
        print("[ERROR] 学習/検証データが空です")
        return

    available_features = [c for c in FEATURE_COLUMNS if c in df.columns]

    binary_params = DEFAULT_BINARY_PARAMS.copy()
    if tune:
        print(f"\n[Optuna最適化] 50試行")
        binary_params = optimize_hyperparams(train_df, val_df)

    # 二値分類モデル
    print(f"\n{'─' * 40}")
    print(f"[二値分類モデル]")
    binary_model, binary_metrics = train_binary_model(train_df, val_df, binary_params)
    print(f"  AUC:        {binary_metrics['auc']}")
    print(f"  LogLoss:    {binary_metrics['logloss']}")
    print(f"  Accuracy:   {binary_metrics['accuracy']}")
    print(f"  Top3的中率: {binary_metrics['top3_hit_rate']}")
    save_model(binary_model, binary_metrics, available_features, "binary")

    fi = get_feature_importance(binary_model, available_features)
    print(f"\n  [特徴量重要度 Top10]")
    for _, row in fi.head(10).iterrows():
        print(f"    {row['feature']:30s} {row['importance_pct']:5.1f}%")

    # ランキングモデル
    print(f"\n{'─' * 40}")
    print(f"[ランキングモデル]")
    rank_model, rank_metrics = train_ranking_model(train_df, val_df)
    print(f"  Top3的中率: {rank_metrics['top3_hit_rate']}")
    save_model(rank_model, rank_metrics, available_features, "ranking")

    print(f"\n  [OK] 学習完了")
    print(f"  モデル出力: {MODEL_DIR}")
