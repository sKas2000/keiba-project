#!/usr/bin/env python3
"""
モデル学習パイプライン v1.0
============================
特徴量CSVからLightGBMモデルを学習する。
二値分類（3着以内予測）とランキング（LambdaRank）の2モデルを構築。

使い方:
  python model_trainer.py [--input data/ml/processed/features.csv]
  python model_trainer.py --tune  # Optunaハイパーパラメータ最適化
"""

VERSION = "1.0"

import json
import sys
from pathlib import Path
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "ml"
DEFAULT_INPUT = DATA_DIR / "processed" / "features.csv"
MODEL_DIR = DATA_DIR / "models"

# feature_engineer.py で定義した特徴量カラム
FEATURE_COLUMNS = [
    "frame_number", "horse_number", "sex_code", "age",
    "weight_carried", "horse_weight", "horse_weight_change",
    "num_entries",
    "surface_code", "distance", "track_condition_code",
    "race_class_code", "distance_cat", "course_id_code",
    "prev_finish_1", "prev_finish_2", "prev_finish_3",
    "avg_finish_last5", "best_finish_last5",
    "win_rate_last5", "place_rate_last5",
    "avg_last3f_last5", "days_since_last_race",
    "total_races", "career_win_rate", "career_place_rate",
    "surface_win_rate", "surface_place_rate",
    "distance_cat_win_rate",
    "jockey_win_rate_365d", "jockey_place_rate_365d",
    "jockey_ride_count_365d",
]


# ============================================================
# データ分割
# ============================================================

def time_based_split(df: pd.DataFrame, val_start: str = "2025-01-01") -> tuple:
    """
    時系列ベースの学習・検証分割
    Args:
        df: 特徴量DataFrame
        val_start: 検証期間の開始日
    Returns:
        (train_df, val_df)
    """
    df["race_date"] = pd.to_datetime(df["race_date"])
    val_date = pd.Timestamp(val_start)

    train = df[df["race_date"] < val_date].copy()
    val = df[df["race_date"] >= val_date].copy()

    return train, val


def prepare_data(df: pd.DataFrame) -> tuple:
    """DataFrameからX, y, groupを抽出"""
    available_features = [c for c in FEATURE_COLUMNS if c in df.columns]
    X = df[available_features].values.astype(np.float32)
    y = df["top3"].values if "top3" in df.columns else None
    groups = df["race_id"].values if "race_id" in df.columns else None
    return X, y, groups, available_features


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
    """
    二値分類モデル(3着以内予測)を学習
    Returns:
        (model, metrics_dict)
    """
    params = params or DEFAULT_BINARY_PARAMS.copy()

    X_train, y_train, _, feature_names = prepare_data(train_df)
    X_val, y_val, _, _ = prepare_data(val_df)

    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
    dval = lgb.Dataset(X_val, label=y_val, feature_name=feature_names, reference=dtrain)

    callbacks = [
        lgb.log_evaluation(100),
        lgb.early_stopping(50),
    ]

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=callbacks,
    )

    # 評価
    y_pred = model.predict(X_val)
    auc = roc_auc_score(y_val, y_pred)
    logloss = log_loss(y_val, y_pred)
    acc = accuracy_score(y_val, (y_pred >= 0.5).astype(int))

    # 3着以内の的中率（上位3頭を選んだ場合）
    val_df = val_df.copy()
    val_df["pred_prob"] = y_pred
    top3_hit_rate = _calc_top3_hit_rate(val_df)

    metrics = {
        "auc": round(auc, 4),
        "logloss": round(logloss, 4),
        "accuracy": round(acc, 4),
        "top3_hit_rate": round(top3_hit_rate, 4),
        "num_iterations": model.best_iteration,
        "train_size": len(X_train),
        "val_size": len(X_val),
    }

    return model, metrics


def _calc_top3_hit_rate(val_df: pd.DataFrame) -> float:
    """レースごとに予測上位3頭のうち実際に3着以内だった馬の割合"""
    hits = 0
    total = 0
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
}


def _make_rank_labels(finish_positions: np.ndarray, num_entries: int) -> np.ndarray:
    """着順をランキング用ラベルに変換（高い=良い）"""
    return np.maximum(num_entries + 1 - finish_positions, 0).astype(np.float32)


def train_ranking_model(train_df: pd.DataFrame, val_df: pd.DataFrame,
                        params: dict = None) -> tuple:
    """
    ランキングモデル(LambdaRank)を学習
    Returns:
        (model, metrics_dict)
    """
    params = params or DEFAULT_RANK_PARAMS.copy()

    X_train, _, train_groups, feature_names = prepare_data(train_df)
    X_val, _, val_groups, _ = prepare_data(val_df)

    # ランキングラベル作成
    y_train_rank = []
    train_group_sizes = []
    for race_id, group in train_df.groupby("race_id"):
        n = len(group)
        labels = _make_rank_labels(group["finish_position"].values, n)
        y_train_rank.extend(labels)
        train_group_sizes.append(n)

    y_val_rank = []
    val_group_sizes = []
    for race_id, group in val_df.groupby("race_id"):
        n = len(group)
        labels = _make_rank_labels(group["finish_position"].values, n)
        y_val_rank.extend(labels)
        val_group_sizes.append(n)

    dtrain = lgb.Dataset(
        X_train, label=np.array(y_train_rank, dtype=np.float32),
        group=train_group_sizes, feature_name=feature_names,
    )
    dval = lgb.Dataset(
        X_val, label=np.array(y_val_rank, dtype=np.float32),
        group=val_group_sizes, feature_name=feature_names,
        reference=dtrain,
    )

    callbacks = [
        lgb.log_evaluation(100),
        lgb.early_stopping(50),
    ]

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=callbacks,
    )

    # 評価: 予測スコアで順位付けし、実際の3着以内と比較
    val_df = val_df.copy()
    val_df["pred_score"] = model.predict(X_val)
    top3_hit_rate = _calc_top3_hit_rate_by_score(val_df)

    metrics = {
        "top3_hit_rate": round(top3_hit_rate, 4),
        "num_iterations": model.best_iteration,
        "train_size": len(X_train),
        "val_size": len(X_val),
        "train_groups": len(train_group_sizes),
        "val_groups": len(val_group_sizes),
    }

    return model, metrics


def _calc_top3_hit_rate_by_score(val_df: pd.DataFrame) -> float:
    """ランキングモデルの3着以内的中率"""
    hits = 0
    total = 0
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
# Optunaハイパーパラメータ最適化
# ============================================================

def optimize_hyperparams(train_df: pd.DataFrame, val_df: pd.DataFrame,
                         n_trials: int = 50) -> dict:
    """Optunaでハイパーパラメータ最適化"""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    X_train, y_train, _, feature_names = prepare_data(train_df)
    X_val, y_val, _, _ = prepare_data(val_df)

    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
    dval = lgb.Dataset(X_val, label=y_val, feature_name=feature_names, reference=dtrain)

    def objective(trial):
        params = {
            "objective": "binary",
            "metric": "auc",
            "verbose": -1,
            "seed": 42,
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
            valid_sets=[dval], callbacks=[
                lgb.early_stopping(30),
                lgb.log_evaluation(0),
            ],
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
               model_type: str, output_dir: Path = MODEL_DIR):
    """モデルと付随情報を保存"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # モデル本体
    model_path = output_dir / f"{model_type}_model.txt"
    model.save_model(str(model_path))

    # メタ情報
    meta = {
        "version": VERSION,
        "model_type": model_type,
        "feature_names": feature_names,
        "metrics": metrics,
    }
    meta_path = output_dir / f"{model_type}_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # 特徴量重要度
    fi = get_feature_importance(model, feature_names)
    fi_path = output_dir / f"{model_type}_feature_importance.csv"
    fi.to_csv(fi_path, index=False)

    print(f"  モデル保存: {model_path}")
    print(f"  メタ情報: {meta_path}")
    print(f"  特徴量重要度: {fi_path}")

    return model_path


def load_model(model_type: str = "binary", model_dir: Path = MODEL_DIR) -> tuple:
    """保存済みモデルを読み込み"""
    model_path = model_dir / f"{model_type}_model.txt"
    meta_path = model_dir / f"{model_type}_meta.json"

    model = lgb.Booster(model_file=str(model_path))

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    return model, meta


# ============================================================
# メイン処理
# ============================================================

def train_all(input_path: str, val_start: str = "2025-01-01", tune: bool = False):
    """全モデルを学習"""
    print(f"\n{'=' * 60}")
    print(f"  モデル学習パイプライン v{VERSION}")
    print(f"{'=' * 60}")

    # データ読み込み
    print(f"\n[データ読み込み] {input_path}")
    df = pd.read_csv(input_path, dtype={"race_id": str, "horse_id": str})
    df["race_date"] = pd.to_datetime(df["race_date"])
    print(f"  全体: {len(df)}行, {df['race_id'].nunique()}レース")

    # 分割
    train_df, val_df = time_based_split(df, val_start)
    print(f"  学習: {len(train_df)}行 (~{val_start})")
    print(f"  検証: {len(val_df)}行 ({val_start}~)")

    if len(train_df) == 0 or len(val_df) == 0:
        print("[ERROR] 学習/検証データが空です")
        return

    available_features = [c for c in FEATURE_COLUMNS if c in df.columns]

    # ハイパーパラメータ最適化
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

    # 特徴量重要度トップ10
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

    print(f"\n{'=' * 60}")
    print(f"  [OK] 学習完了")
    print(f"  モデル出力: {MODEL_DIR}")
    print(f"{'=' * 60}\n")


def main():
    parser = ArgumentParser(description="モデル学習パイプライン")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="特徴量CSV")
    parser.add_argument("--val-start", default="2025-01-01", help="検証開始日")
    parser.add_argument("--tune", action="store_true", help="Optuna最適化を実行")
    args = parser.parse_args()

    train_all(args.input, args.val_start, args.tune)


if __name__ == "__main__":
    main()
