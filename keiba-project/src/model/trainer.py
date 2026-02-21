"""
モデル学習モジュール
LightGBM 二値分類 + 勝率直接推定 + ランキング（LambdaRank）
+ Platt Scaling / Isotonic Regression キャリブレーション
"""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

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
# 勝率直接推定モデル（1着予測）
# ============================================================

def train_win_model(train_df: pd.DataFrame, val_df: pd.DataFrame,
                    params: dict = None) -> tuple:
    """勝率直接推定モデル（1着 vs others）を学習"""
    params = params or DEFAULT_BINARY_PARAMS.copy()
    params["is_unbalance"] = True  # クラス不均衡対応（陽性率 ~7%）

    available = [c for c in FEATURE_COLUMNS if c in train_df.columns]
    cat_indices = _get_categorical_indices(available)

    X_train = train_df[available].values.astype(np.float32)
    X_val = val_df[available].values.astype(np.float32)
    y_train = (train_df["finish_position"] == 1).astype(int).values
    y_val = (val_df["finish_position"] == 1).astype(int).values

    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=available,
                         categorical_feature=cat_indices if cat_indices else "auto")
    dval = lgb.Dataset(X_val, label=y_val, feature_name=available, reference=dtrain)

    callbacks = [lgb.log_evaluation(100), lgb.early_stopping(50)]

    model = lgb.train(
        params, dtrain, num_boost_round=1000,
        valid_sets=[dtrain, dval], valid_names=["train", "val"],
        callbacks=callbacks,
    )

    y_pred = model.predict(X_val)
    auc = roc_auc_score(y_val, y_pred)
    logloss = log_loss(y_val, y_pred)

    # Top1的中率
    val_copy = val_df.copy()
    val_copy["pred_win"] = y_pred
    top1_hits = total_races = 0
    for _, grp in val_copy.groupby("race_id"):
        if len(grp) < 3:
            continue
        if grp.nlargest(1, "pred_win").iloc[0]["finish_position"] == 1:
            top1_hits += 1
        total_races += 1

    metrics = {
        "auc": round(auc, 4),
        "logloss": round(logloss, 4),
        "top1_hit_rate": round(top1_hits / total_races, 4) if total_races > 0 else 0,
        "num_iterations": model.best_iteration,
        "train_size": len(X_train),
        "val_size": len(X_val),
        "positive_rate": round(float(y_val.mean()), 4),
    }
    return model, metrics


# ============================================================
# Platt Scaling キャリブレーション
# ============================================================

def fit_calibrator(model, val_df: pd.DataFrame,
                   model_dir: Path = None) -> LogisticRegression:
    """Platt Scaling: バリデーションセットで確率キャリブレーション

    LightGBMの生予測確率をlogit変換し、LogisticRegressionで再キャリブレーション。
    温度パラメータを不要にし、確率の信頼性を向上させる。
    """
    model_dir = model_dir or MODEL_DIR
    model_dir.mkdir(parents=True, exist_ok=True)

    available = [c for c in FEATURE_COLUMNS if c in val_df.columns]
    X_val = val_df[available].values.astype(np.float32)
    y_val = val_df["top3"].values

    raw_probs = model.predict(X_val)
    raw_probs_clipped = np.clip(raw_probs, 1e-6, 1 - 1e-6)
    logits = np.log(raw_probs_clipped / (1 - raw_probs_clipped)).reshape(-1, 1)

    calibrator = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
    calibrator.fit(logits, y_val)

    cal_path = model_dir / "calibrator.pkl"
    with open(cal_path, "wb") as f:
        pickle.dump(calibrator, f)

    cal_probs = calibrator.predict_proba(logits)[:, 1]
    raw_brier = brier_score_loss(y_val, raw_probs)
    cal_brier = brier_score_loss(y_val, cal_probs)

    print(f"  [Platt Scaling]")
    print(f"    Brier Score (raw):        {raw_brier:.4f}")
    print(f"    Brier Score (calibrated): {cal_brier:.4f}")
    if raw_brier > 0:
        print(f"    改善: {(raw_brier - cal_brier) / raw_brier * 100:.1f}%")
    print(f"    係数: a={calibrator.coef_[0][0]:.4f}, b={calibrator.intercept_[0]:.4f}")
    print(f"    保存: {cal_path}")

    return calibrator


def load_calibrator(model_dir: Path = None) -> LogisticRegression | None:
    """保存済みキャリブレーターを読み込み"""
    model_dir = model_dir or MODEL_DIR
    cal_path = model_dir / "calibrator.pkl"
    if not cal_path.exists():
        return None
    with open(cal_path, "rb") as f:
        return pickle.load(f)


def calibrate_probs(raw_probs: np.ndarray, calibrator: LogisticRegression) -> np.ndarray:
    """生確率をキャリブレーション済み確率に変換"""
    clipped = np.clip(raw_probs, 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    return calibrator.predict_proba(logits)[:, 1]


# ============================================================
# Isotonic Regression キャリブレーション
# ============================================================

def fit_isotonic_calibrator(model, val_df: pd.DataFrame, target: str = "top3",
                            save_name: str = "isotonic",
                            model_dir: Path = None) -> IsotonicRegression:
    """Isotonic Regression: 非パラメトリック確率キャリブレーション

    Platt Scalingより柔軟。任意の単調歪みを補正可能。
    """
    model_dir = model_dir or MODEL_DIR
    model_dir.mkdir(parents=True, exist_ok=True)

    available = [c for c in FEATURE_COLUMNS if c in val_df.columns]
    X_val = val_df[available].values.astype(np.float32)

    if target == "win":
        y_val = (val_df["finish_position"] == 1).astype(int).values
    elif target in val_df.columns:
        y_val = val_df[target].values
    else:
        raise ValueError(f"Unknown target: {target}")

    raw_probs = model.predict(X_val)

    iso = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds='clip')
    iso.fit(raw_probs, y_val)

    cal_path = model_dir / f"{save_name}.pkl"
    with open(cal_path, "wb") as f:
        pickle.dump(iso, f)

    cal_probs = iso.predict(raw_probs)
    raw_brier = brier_score_loss(y_val, np.clip(raw_probs, 0, 1))
    cal_brier = brier_score_loss(y_val, cal_probs)

    print(f"  [{save_name}]")
    print(f"    Brier Score (raw):        {raw_brier:.6f}")
    print(f"    Brier Score (calibrated): {cal_brier:.6f}")
    improvement = (raw_brier - cal_brier) / raw_brier * 100 if raw_brier > 0 else 0
    print(f"    改善: {improvement:.1f}%")
    print(f"    保存: {cal_path}")

    return iso


def fit_isotonic_cv(train_df: pd.DataFrame, params: dict,
                    target: str = "top3", save_name: str = "isotonic_cv",
                    n_folds: int = 5, model_dir: Path = None) -> IsotonicRegression:
    """CV-Isotonic: Train期間のK-fold OOF予測でIsotonicを学習（リーク完全解消）

    1. Train期間を時系列でK分割
    2. 各foldでモデル学習 → held-outに予測
    3. OOF予測全体でIsotonic Regressionを学習
    4. val/test期間はリーク無しでキャリブレーション可能
    """
    model_dir = model_dir or MODEL_DIR
    model_dir.mkdir(parents=True, exist_ok=True)

    available = [c for c in FEATURE_COLUMNS if c in train_df.columns]
    cat_indices = _get_categorical_indices(available)

    # 時系列K-fold: race_dateでソートして均等分割
    train_sorted = train_df.sort_values("race_date").reset_index(drop=True)
    dates = train_sorted["race_date"].unique()
    fold_size = len(dates) // n_folds

    oof_preds = np.zeros(len(train_sorted))
    if target == "win":
        y_all = (train_sorted["finish_position"] == 1).astype(int).values
    elif target in train_sorted.columns:
        y_all = train_sorted[target].values
    else:
        raise ValueError(f"Unknown target: {target}")

    print(f"  [CV-Isotonic] {n_folds}-fold, target={target}")

    for fold in range(n_folds):
        # 時系列分割: fold番目の日付範囲をheld-outに
        start_idx = fold * fold_size
        if fold == n_folds - 1:
            fold_dates = dates[start_idx:]
        else:
            fold_dates = dates[start_idx:start_idx + fold_size]

        fold_date_set = set(fold_dates)
        held_out_mask = train_sorted["race_date"].isin(fold_date_set)
        train_mask = ~held_out_mask

        fold_train = train_sorted[train_mask]
        fold_val = train_sorted[held_out_mask]

        if len(fold_train) == 0 or len(fold_val) == 0:
            continue

        X_tr = fold_train[available].values.astype(np.float32)
        X_ho = fold_val[available].values.astype(np.float32)

        if target == "win":
            y_tr = (fold_train["finish_position"] == 1).astype(int).values
        else:
            y_tr = fold_train[target].values

        fold_params = params.copy()
        if target == "win":
            fold_params["is_unbalance"] = True

        dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=available,
                             categorical_feature=cat_indices if cat_indices else "auto")
        dval_fold = lgb.Dataset(X_ho, label=y_all[held_out_mask], feature_name=available,
                                reference=dtrain)

        fold_model = lgb.train(
            fold_params, dtrain, num_boost_round=1000,
            valid_sets=[dval_fold], valid_names=["val"],
            callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)],
        )

        oof_preds[held_out_mask] = fold_model.predict(X_ho)
        print(f"    Fold {fold+1}/{n_folds}: train={len(fold_train)}, "
              f"held_out={len(fold_val)}, iters={fold_model.best_iteration}")

    # OOF予測でIsotonic Regressionを学習
    valid_mask = oof_preds > 0  # fold漏れがないか確認
    iso = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds='clip')
    iso.fit(oof_preds[valid_mask], y_all[valid_mask])

    cal_path = model_dir / f"{save_name}.pkl"
    with open(cal_path, "wb") as f:
        pickle.dump(iso, f)

    # 評価
    cal_preds = iso.predict(oof_preds[valid_mask])
    raw_brier = brier_score_loss(y_all[valid_mask], np.clip(oof_preds[valid_mask], 0, 1))
    cal_brier = brier_score_loss(y_all[valid_mask], cal_preds)
    improvement = (raw_brier - cal_brier) / raw_brier * 100 if raw_brier > 0 else 0

    print(f"    OOF Brier (raw):        {raw_brier:.6f}")
    print(f"    OOF Brier (calibrated): {cal_brier:.6f}")
    print(f"    改善: {improvement:.1f}%")
    print(f"    保存: {cal_path}")

    return iso


def load_isotonic_calibrator(save_name: str = "isotonic",
                             model_dir: Path = None) -> IsotonicRegression | None:
    """保存済みIsotonicキャリブレーターを読み込み"""
    model_dir = model_dir or MODEL_DIR
    cal_path = model_dir / f"{save_name}.pkl"
    if not cal_path.exists():
        return None
    with open(cal_path, "rb") as f:
        return pickle.load(f)


def calibrate_isotonic(raw_probs: np.ndarray,
                       calibrator: IsotonicRegression) -> np.ndarray:
    """Isotonic Regressionでキャリブレーション"""
    return calibrator.predict(raw_probs)


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

    # Platt Scaling キャリブレーション
    print(f"\n{'─' * 40}")
    print(f"[Platt Scaling キャリブレーション]")
    fit_calibrator(binary_model, val_df)

    # 勝率直接推定モデル
    print(f"\n{'─' * 40}")
    print(f"[勝率直接推定モデル（1着予測）]")
    win_params = binary_params.copy()
    win_model, win_metrics = train_win_model(train_df, val_df, win_params)
    print(f"  AUC:        {win_metrics['auc']}")
    print(f"  LogLoss:    {win_metrics['logloss']}")
    print(f"  Top1的中率: {win_metrics['top1_hit_rate']}")
    print(f"  陽性率:     {win_metrics['positive_rate']}")
    save_model(win_model, win_metrics, available_features, "win")

    fi_win = get_feature_importance(win_model, available_features)
    print(f"\n  [特徴量重要度 Top10 (Win)]")
    for _, row in fi_win.head(10).iterrows():
        print(f"    {row['feature']:30s} {row['importance_pct']:5.1f}%")

    # CV-Isotonic Regression キャリブレーション（リーク無し）
    print(f"\n{'─' * 40}")
    print(f"[CV-Isotonic Regression キャリブレーション（Train OOF）]")
    fit_isotonic_cv(train_df, binary_params, target="top3",
                    save_name="binary_isotonic", model_dir=MODEL_DIR)
    win_cv_params = binary_params.copy()
    fit_isotonic_cv(train_df, win_cv_params, target="win",
                    save_name="win_isotonic", model_dir=MODEL_DIR)

    # ランキングモデル
    print(f"\n{'─' * 40}")
    print(f"[ランキングモデル]")
    rank_model, rank_metrics = train_ranking_model(train_df, val_df)
    print(f"  Top3的中率: {rank_metrics['top3_hit_rate']}")
    save_model(rank_model, rank_metrics, available_features, "ranking")

    print(f"\n  [OK] 学習完了")
    print(f"  モデル出力: {MODEL_DIR}")
