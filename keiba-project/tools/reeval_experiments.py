"""
モデル判断の再精査実験スクリプト
softmax修正・Isotonic導入後のコードで、パラメータ/特徴量選択を体系的に再検証

Usage:
    python tools/reeval_experiments.py              # 全実験
    python tools/reeval_experiments.py --group a    # パラメータスイープのみ
    python tools/reeval_experiments.py --group b    # 特徴量実験のみ
"""
import argparse
import json
import sys
from datetime import datetime
from itertools import combinations
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from dateutil.relativedelta import relativedelta

from config.settings import (
    FEATURE_COLUMNS, CATEGORICAL_FEATURES,
    EXPANDING_BEST_PARAMS, PROCESSED_DIR, RAW_DIR,
    setup_encoding,
)
from src.model.evaluator import (
    BET_TYPES, _empty_bet_stats, _load_returns, simulate_bets,
)
from src.model.trainer import DEFAULT_BINARY_PARAMS, DEFAULT_RANK_PARAMS

# ============================================================
# Val/Test 期間定義
# ============================================================
VAL_START = pd.Timestamp("2025-01-01")
TEST_START = pd.Timestamp("2025-07-01")

# 主要券種（最適化対象）
KEY_BETS = ["quinella", "wide", "trio"]
BET_JA = {"win": "単勝", "place": "複勝", "quinella": "馬連",
           "wide": "ワイド", "trio": "3連複"}


# ============================================================
# データ読み込み（共通）
# ============================================================
def load_data():
    """features.csv + returns.csv を読み込み"""
    input_path = str(PROCESSED_DIR / "features.csv")
    returns_path = RAW_DIR / "returns.csv"

    df = pd.read_csv(input_path, dtype={"race_id": str, "horse_id": str})
    df["race_date"] = pd.to_datetime(df["race_date"])
    returns = _load_returns(returns_path)

    # 基本フィルタ（新馬・障害・少頭数）
    pre = len(df)
    df = df[df["race_class_code"] != 1]  # 新馬除外
    df = df[df["surface_code"] != 2]      # 障害除外
    df = df[df["num_entries"] >= 6]       # 少頭数除外
    print(f"  データ: {pre}行 → {len(df)}行（{pre - len(df)}行除外）")
    return df, returns


# ============================================================
# Expanding Window + Multi-Config Backtest
# ============================================================
def expanding_multi_backtest(
    df: pd.DataFrame,
    returns: dict,
    feature_columns: list,
    param_configs: list,
    use_ranking: bool = False,
    calibration_pct: float = 0.10,
    window_months: int = 3,
    initial_train_end: str = "2024-01-01",
    quiet: bool = False,
    lgbm_binary_params: dict = None,
    lgbm_rank_params: dict = None,
) -> dict:
    """Expanding Windowで複数パラメータ構成を同時評価

    Args:
        df: フィルタ済みデータ
        returns: 払い戻しデータ
        feature_columns: 使用する特徴量リスト
        param_configs: [{"name": "xxx", "kwargs": {simulate_betsのキーワード引数}}, ...]
        use_ranking: Rankingモデルを使用するか
        calibration_pct: キャリブレーション用データ割合
        window_months: テスト期間（月数）
        lgbm_binary_params: LightGBMの二値分類パラメータ（Noneなら DEFAULT_BINARY_PARAMS）
        lgbm_rank_params: LightGBMのランキングパラメータ（Noneなら DEFAULT_RANK_PARAMS）

    Returns:
        {config_name: {"windows": [...], "val_rois": {...}, "test_rois": {...}, "total_rois": {...}}}
    """
    from src.model.trainer import _get_categorical_indices, _make_rank_labels

    available = [c for c in feature_columns if c in df.columns]
    cat_indices = [i for i, f in enumerate(available) if f in CATEGORICAL_FEATURES]

    train_end = pd.Timestamp(initial_train_end)
    max_date = df["race_date"].max() + pd.Timedelta(days=1)

    # 結果格納
    config_results = {}
    for cfg in param_configs:
        config_results[cfg["name"]] = {
            "windows": [],
            "val_bets": {bt: _empty_bet_stats() for bt in BET_TYPES},
            "test_bets": {bt: _empty_bet_stats() for bt in BET_TYPES},
            "total_bets": {bt: _empty_bet_stats() for bt in BET_TYPES},
        }

    # LightGBMパラメータ（カスタム or デフォルト）
    _binary_params = lgbm_binary_params or DEFAULT_BINARY_PARAMS
    _rank_params = lgbm_rank_params or DEFAULT_RANK_PARAMS

    quiet_cb = [lgb.log_evaluation(0), lgb.early_stopping(50)]
    window_idx = 0

    while train_end < max_date:
        test_start = train_end
        test_end = train_end + relativedelta(months=window_months)
        if test_end > max_date:
            test_end = max_date

        full_train = df[df["race_date"] < train_end].copy()
        test_df = df[(df["race_date"] >= test_start) &
                     (df["race_date"] < test_end)].copy()

        if len(full_train) < 1000 or len(test_df) < 100:
            train_end = test_end
            continue

        # Train / Calibration split
        train_dates = np.sort(full_train["race_date"].unique())
        cal_idx = int(len(train_dates) * (1 - calibration_pct))
        cal_split = train_dates[cal_idx]
        train_inner = full_train[full_train["race_date"] < cal_split].copy()
        cal_set = full_train[full_train["race_date"] >= cal_split].copy()

        if len(train_inner) < 500 or len(cal_set) < 200:
            train_inner = full_train
            cal_set = None

        X_inner = train_inner[available].values.astype(np.float32)
        y_inner_top3 = train_inner["top3"].values
        y_inner_win = (train_inner["finish_position"] == 1).astype(int).values

        # --- Binary model ---
        bp = _binary_params.copy()
        if cal_set is not None:
            X_cal = cal_set[available].values.astype(np.float32)
            y_cal_top3 = cal_set["top3"].values
            d_tr = lgb.Dataset(X_inner, label=y_inner_top3, feature_name=available,
                               categorical_feature=cat_indices if cat_indices else "auto")
            d_cal = lgb.Dataset(X_cal, label=y_cal_top3, feature_name=available, reference=d_tr)
            binary_model = lgb.train(bp, d_tr, num_boost_round=1000,
                                     valid_sets=[d_cal], valid_names=["val"],
                                     callbacks=quiet_cb)
        else:
            X_test_tmp = test_df[available].values.astype(np.float32)
            d_tr = lgb.Dataset(X_inner, label=y_inner_top3, feature_name=available,
                               categorical_feature=cat_indices if cat_indices else "auto")
            d_te = lgb.Dataset(X_test_tmp, label=test_df["top3"].values,
                               feature_name=available, reference=d_tr)
            binary_model = lgb.train(bp, d_tr, num_boost_round=1000,
                                     valid_sets=[d_te], valid_names=["val"],
                                     callbacks=quiet_cb)

        # --- Win model ---
        wp = _binary_params.copy()
        wp["is_unbalance"] = True
        if cal_set is not None:
            y_cal_win = (cal_set["finish_position"] == 1).astype(int).values
            d_tr_w = lgb.Dataset(X_inner, label=y_inner_win, feature_name=available,
                                 categorical_feature=cat_indices if cat_indices else "auto")
            d_cal_w = lgb.Dataset(X_cal, label=y_cal_win, feature_name=available, reference=d_tr_w)
            win_model = lgb.train(wp, d_tr_w, num_boost_round=1000,
                                  valid_sets=[d_cal_w], valid_names=["val"],
                                  callbacks=quiet_cb)
        else:
            y_test_win = (test_df["finish_position"] == 1).astype(int).values
            d_tr_w = lgb.Dataset(X_inner, label=y_inner_win, feature_name=available,
                                 categorical_feature=cat_indices if cat_indices else "auto")
            d_te_w = lgb.Dataset(X_test_tmp, label=y_test_win,
                                 feature_name=available, reference=d_tr_w)
            win_model = lgb.train(wp, d_tr_w, num_boost_round=1000,
                                  valid_sets=[d_te_w], valid_names=["val"],
                                  callbacks=quiet_cb)

        # --- Ranking model (optional) ---
        has_ranking = False
        if use_ranking:
            rp = _rank_params.copy()
            train_rank = train_inner.sort_values("race_id").reset_index(drop=True)
            X_rank = train_rank[available].values.astype(np.float32)
            y_rank, grp_sizes = [], []
            for _, grp in train_rank.groupby("race_id", sort=True):
                labels = _make_rank_labels(grp["finish_position"].values, len(grp))
                y_rank.extend(labels)
                grp_sizes.append(len(grp))

            if len(grp_sizes) > 10:
                d_tr_r = lgb.Dataset(X_rank, label=np.array(y_rank, dtype=np.float32),
                                     group=grp_sizes, feature_name=available,
                                     categorical_feature=cat_indices if cat_indices else "auto")
                if cal_set is not None:
                    cal_rank = cal_set.sort_values("race_id").reset_index(drop=True)
                else:
                    cal_rank = test_df.sort_values("race_id").reset_index(drop=True)
                X_cal_r = cal_rank[available].values.astype(np.float32)
                y_cal_r, cal_grp = [], []
                for _, grp in cal_rank.groupby("race_id", sort=True):
                    labels = _make_rank_labels(grp["finish_position"].values, len(grp))
                    y_cal_r.extend(labels)
                    cal_grp.append(len(grp))
                d_cal_r = lgb.Dataset(X_cal_r, label=np.array(y_cal_r, dtype=np.float32),
                                      group=cal_grp, feature_name=available, reference=d_tr_r)
                try:
                    ranking_model = lgb.train(rp, d_tr_r, num_boost_round=1000,
                                              valid_sets=[d_cal_r], valid_names=["val"],
                                              callbacks=quiet_cb)
                    has_ranking = True
                except Exception:
                    has_ranking = False

        # --- Prediction + Isotonic Calibration ---
        X_test = test_df[available].values.astype(np.float32)
        raw_binary = binary_model.predict(X_test)
        raw_win = win_model.predict(X_test)

        if cal_set is not None:
            cal_binary_raw = binary_model.predict(X_cal)
            cal_win_raw = win_model.predict(X_cal)

            iso_binary = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds='clip')
            iso_binary.fit(cal_binary_raw, y_cal_top3)
            test_df["pred_prob"] = iso_binary.predict(raw_binary)

            iso_win = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds='clip')
            iso_win.fit(cal_win_raw, y_cal_win)
            test_df["win_prob_direct"] = iso_win.predict(raw_win)
        else:
            test_df["pred_prob"] = raw_binary
            test_df["win_prob_direct"] = raw_win

        if has_ranking:
            test_df["rank_score"] = ranking_model.predict(X_test)

        test_df = test_df.sort_values(
            ["race_id", "pred_prob"], ascending=[True, False]
        )

        # --- 各パラメータ構成で simulate_bets ---
        prepared = {
            "val_df": test_df,
            "full_df": df,
            "returns": returns,
            "has_win_model": True,
            "has_ranking": has_ranking,
            "calibrator": "isotonic" if cal_set is not None else None,
            "val_start": str(test_start.date()),
            "val_end": str(test_end.date()),
        }

        # 期間判定
        is_val = VAL_START <= test_start < TEST_START
        is_test = test_start >= TEST_START
        period = "val" if is_val else ("test" if is_test else "train")

        for cfg in param_configs:
            res = simulate_bets(prepared, **cfg["kwargs"])
            n_races = res.get("races", 0)

            window_rois = {}
            window_bets = {}
            for bt in BET_TYPES:
                b = res.get(f"bets_{bt}", {})
                window_bets[bt] = {k: b.get(k, 0) for k in ["count", "invested", "returned", "hits"]}
                window_rois[bt] = (b["returned"] / b["invested"] * 100) if b.get("invested", 0) > 0 else 0

            config_results[cfg["name"]]["windows"].append({
                "window": window_idx,
                "period": period,
                "test_start": str(test_start.date()),
                "test_end": str(test_end.date()),
                "races": n_races,
                "rois": {bt: round(window_rois.get(bt, 0), 1) for bt in BET_TYPES},
                "bets": window_bets,
            })

            # 期間別・合計に加算
            for bt in BET_TYPES:
                for k in ["count", "invested", "returned", "hits"]:
                    config_results[cfg["name"]]["total_bets"][bt][k] += window_bets[bt].get(k, 0)
                    if is_val:
                        config_results[cfg["name"]]["val_bets"][bt][k] += window_bets[bt].get(k, 0)
                    elif is_test:
                        config_results[cfg["name"]]["test_bets"][bt][k] += window_bets[bt].get(k, 0)

        if not quiet:
            # 1行目のconfigだけ進捗表示
            first_cfg = param_configs[0]["name"]
            rois = config_results[first_cfg]["windows"][-1]["rois"]
            n = config_results[first_cfg]["windows"][-1]["races"]
            train_label = f"~{train_end.strftime('%Y-%m')}"
            test_label = f"{test_start.strftime('%Y-%m')}~{test_end.strftime('%Y-%m')}"
            print(f"  W{window_idx:02d} {train_label:>12s} {test_label:>18s} {n:>5d}R"
                  f"  Q:{rois.get('quinella', 0):>6.1f}%"
                  f"  W:{rois.get('wide', 0):>6.1f}%"
                  f"  T:{rois.get('trio', 0):>6.1f}%"
                  f"  [{period}]")

        window_idx += 1
        train_end = test_end

    # ROI集計
    for cfg_name, data in config_results.items():
        for period_key in ["val", "test", "total"]:
            bets = data[f"{period_key}_bets"]
            data[f"{period_key}_rois"] = {}
            for bt in BET_TYPES:
                b = bets[bt]
                data[f"{period_key}_rois"][bt] = (
                    round(b["returned"] / b["invested"] * 100, 1)
                    if b.get("invested", 0) > 0 else 0.0
                )

    return config_results


# ============================================================
# 結果表示
# ============================================================
def print_comparison(title: str, config_results: dict, current_name: str = None):
    """実験結果の比較テーブルを表示"""
    print(f"\n{'=' * 100}")
    print(f"  {title}")
    print(f"{'=' * 100}")

    header = f"  {'Config':<25s}"
    for bt in KEY_BETS:
        header += f" {'Val_' + BET_JA[bt]:>10s} {'Test_' + BET_JA[bt]:>10s}"
    header += f" {'Val平均':>8s} {'Test平均':>8s}"
    print(header)
    print(f"  {'-' * 95}")

    best_val_avg = -999
    best_config = None

    for cfg_name, data in config_results.items():
        val_rois = data.get("val_rois", {})
        test_rois = data.get("test_rois", {})

        line = f"  {cfg_name:<25s}"
        val_key_rois = []
        test_key_rois = []
        for bt in KEY_BETS:
            v = val_rois.get(bt, 0)
            t = test_rois.get(bt, 0)
            line += f" {v:>10.1f}% {t:>10.1f}%"
            if v > 0:
                val_key_rois.append(v)
            if t > 0:
                test_key_rois.append(t)

        val_avg = np.mean(val_key_rois) if val_key_rois else 0
        test_avg = np.mean(test_key_rois) if test_key_rois else 0
        marker = " ★" if cfg_name == current_name else ""
        line += f" {val_avg:>7.1f}% {test_avg:>7.1f}%{marker}"
        print(line)

        if val_avg > best_val_avg:
            best_val_avg = val_avg
            best_config = cfg_name

    print(f"  {'-' * 95}")
    if best_config:
        print(f"  → Val最良: {best_config} (平均 {best_val_avg:.1f}%)")
    print()
    return best_config


# ============================================================
# Group A: パラメータスイープ
# ============================================================
def run_group_a(df, returns):
    """パラメータスイープ実験（現行特徴量、再学習はウィンドウ内のみ）"""
    print(f"\n{'#' * 60}")
    print(f"# Group A: パラメータスイープ")
    print(f"{'#' * 60}")

    base_kwargs = {
        "kelly_fraction": 0,
        "confidence_min": 0.04,
        "quinella_top_n": 2,
        "wide_top_n": 2,
        "skip_classes": [4, 6],
        "top_n": 3,
    }

    # --- A1: skip_classes ---
    print(f"\n--- A1: skip_classes ---")
    configs_a1 = []
    for name, val in [
        ("skip_none", []),
        ("skip_4", [4]),
        ("skip_6", [6]),
        ("skip_4_6★", [4, 6]),
        ("skip_1_4_6", [1, 4, 6]),
    ]:
        kw = base_kwargs.copy()
        kw["skip_classes"] = val
        configs_a1.append({"name": name, "kwargs": kw})

    results_a1 = expanding_multi_backtest(
        df, returns, FEATURE_COLUMNS, configs_a1,
        use_ranking=False, calibration_pct=0.10, window_months=3,
    )
    best_skip = print_comparison("A1: skip_classes", results_a1, "skip_4_6★")

    # A1の最良skip_classesを取得
    best_skip_val = next(c["kwargs"]["skip_classes"] for c in configs_a1 if c["name"] == best_skip)

    # --- A2: confidence_min ---
    print(f"\n--- A2: confidence_min ---")
    configs_a2 = []
    for val in [0.0, 0.02, 0.04, 0.06, 0.08]:
        name = f"conf_{val:.2f}" + ("★" if val == 0.04 else "")
        kw = base_kwargs.copy()
        kw["skip_classes"] = best_skip_val
        kw["confidence_min"] = val
        configs_a2.append({"name": name, "kwargs": kw})

    results_a2 = expanding_multi_backtest(
        df, returns, FEATURE_COLUMNS, configs_a2,
        use_ranking=False, calibration_pct=0.10, window_months=3,
    )
    best_conf = print_comparison("A2: confidence_min", results_a2, "conf_0.04★")
    best_conf_val = next(c["kwargs"]["confidence_min"] for c in configs_a2 if c["name"] == best_conf)

    # --- A3: quinella_top_n ---
    print(f"\n--- A3: quinella_top_n ---")
    configs_a3 = []
    for val in [2, 3, 4]:
        name = f"q_top_{val}" + ("★" if val == 2 else "")
        kw = base_kwargs.copy()
        kw["skip_classes"] = best_skip_val
        kw["confidence_min"] = best_conf_val
        kw["quinella_top_n"] = val
        configs_a3.append({"name": name, "kwargs": kw})

    results_a3 = expanding_multi_backtest(
        df, returns, FEATURE_COLUMNS, configs_a3,
        use_ranking=False, calibration_pct=0.10, window_months=3,
    )
    best_q = print_comparison("A3: quinella_top_n", results_a3, "q_top_2★")
    best_q_val = next(c["kwargs"]["quinella_top_n"] for c in configs_a3 if c["name"] == best_q)

    # --- A4: wide_top_n ---
    print(f"\n--- A4: wide_top_n ---")
    configs_a4 = []
    for val in [2, 3, 4]:
        name = f"w_top_{val}" + ("★" if val == 2 else "")
        kw = base_kwargs.copy()
        kw["skip_classes"] = best_skip_val
        kw["confidence_min"] = best_conf_val
        kw["quinella_top_n"] = best_q_val
        kw["wide_top_n"] = val
        configs_a4.append({"name": name, "kwargs": kw})

    results_a4 = expanding_multi_backtest(
        df, returns, FEATURE_COLUMNS, configs_a4,
        use_ranking=False, calibration_pct=0.10, window_months=3,
    )
    best_w = print_comparison("A4: wide_top_n", results_a4, "w_top_2★")
    best_w_val = next(c["kwargs"]["wide_top_n"] for c in configs_a4 if c["name"] == best_w)

    # --- A5: top_n ---
    print(f"\n--- A5: top_n ---")
    configs_a5 = []
    for val in [3, 4, 5]:
        name = f"top_{val}" + ("★" if val == 3 else "")
        kw = base_kwargs.copy()
        kw["skip_classes"] = best_skip_val
        kw["confidence_min"] = best_conf_val
        kw["quinella_top_n"] = best_q_val
        kw["wide_top_n"] = best_w_val
        kw["top_n"] = val
        configs_a5.append({"name": name, "kwargs": kw})

    results_a5 = expanding_multi_backtest(
        df, returns, FEATURE_COLUMNS, configs_a5,
        use_ranking=False, calibration_pct=0.10, window_months=3,
    )
    best_t = print_comparison("A5: top_n", results_a5, "top_3★")
    best_t_val = next(c["kwargs"]["top_n"] for c in configs_a5 if c["name"] == best_t)

    # --- A6: use_ranking ---
    print(f"\n--- A6: use_ranking ---")
    best_kw = {
        "kelly_fraction": 0,
        "skip_classes": best_skip_val,
        "confidence_min": best_conf_val,
        "quinella_top_n": best_q_val,
        "wide_top_n": best_w_val,
        "top_n": best_t_val,
    }

    # ranking=False（既にA5で計算済みだが、best paramsで再実行）
    configs_a6_norank = [{"name": "no_ranking★", "kwargs": best_kw}]
    results_a6_norank = expanding_multi_backtest(
        df, returns, FEATURE_COLUMNS, configs_a6_norank,
        use_ranking=False, calibration_pct=0.10, window_months=3,
    )

    # ranking=True
    configs_a6_rank = [{"name": "with_ranking", "kwargs": best_kw}]
    results_a6_rank = expanding_multi_backtest(
        df, returns, FEATURE_COLUMNS, configs_a6_rank,
        use_ranking=True, calibration_pct=0.10, window_months=3,
    )

    combined_a6 = {**results_a6_norank, **results_a6_rank}
    best_rank = print_comparison("A6: use_ranking", combined_a6, "no_ranking★")
    use_ranking_best = best_rank == "with_ranking"

    # --- 最終サマリ ---
    best_params = {
        "skip_classes": best_skip_val,
        "confidence_min": best_conf_val,
        "quinella_top_n": best_q_val,
        "wide_top_n": best_w_val,
        "top_n": best_t_val,
        "use_ranking": use_ranking_best,
        "kelly_fraction": 0,
        "use_calibration": True,
        "calibration_pct": 0.10,
        "window_months": 3,
    }

    print(f"\n{'=' * 60}")
    print(f"  Group A 最良パラメータ:")
    for k, v in best_params.items():
        current = EXPANDING_BEST_PARAMS.get(k, "N/A")
        changed = " ← CHANGED" if v != current else ""
        print(f"    {k}: {v} (was: {current}){changed}")
    print(f"{'=' * 60}")

    return {
        "best_params": best_params,
        "experiments": {
            "A1_skip_classes": results_a1,
            "A2_confidence_min": results_a2,
            "A3_quinella_top_n": results_a3,
            "A4_wide_top_n": results_a4,
            "A5_top_n": results_a5,
            "A6_use_ranking": combined_a6,
        },
    }


# ============================================================
# Group B: 特徴量実験
# ============================================================

# 除外中の特徴量セット
EXTRA_FEATURES = {
    "B1_odds": ["win_odds"],
    "B2_popularity": ["popularity"],
    "B3_odds_pop": ["win_odds", "popularity"],
    "B4_zscore": [
        "z_surface_place_rate", "z_jockey_place_rate_365d",
        "z_avg_finish_last5", "z_career_place_rate",
        "z_trainer_place_rate_365d",
    ],
    "B5_jockey_horse": ["same_jockey_rides", "same_jockey_win_rate"],
    "B6_course_apt": ["course_dist_win_rate", "course_dist_place_rate"],
    "B7_all_extra": [
        "z_surface_place_rate", "z_jockey_place_rate_365d",
        "z_avg_finish_last5", "z_career_place_rate",
        "z_trainer_place_rate_365d",
        "same_jockey_rides", "same_jockey_win_rate",
        "course_dist_win_rate", "course_dist_place_rate",
    ],
}


def run_group_b(df, returns, best_kwargs: dict = None):
    """特徴量実験（再学習あり）"""
    print(f"\n{'#' * 60}")
    print(f"# Group B: 特徴量実験（再学習あり）")
    print(f"{'#' * 60}")

    # bet_kwargsはGroup Aの最良値を使う（なければ現行値）
    if best_kwargs is None:
        best_kwargs = {
            "kelly_fraction": EXPANDING_BEST_PARAMS.get("kelly_fraction", 0),
            "confidence_min": EXPANDING_BEST_PARAMS.get("confidence_min", 0.04),
            "quinella_top_n": EXPANDING_BEST_PARAMS.get("quinella_top_n", 2),
            "wide_top_n": EXPANDING_BEST_PARAMS.get("wide_top_n", 2),
            "skip_classes": EXPANDING_BEST_PARAMS.get("skip_classes", [4, 6]),
            "top_n": EXPANDING_BEST_PARAMS.get("top_n", 3),
        }

    use_ranking = best_kwargs.pop("use_ranking", False) if "use_ranking" in best_kwargs else False

    all_results = {}

    # ベースライン（現行特徴量）
    print(f"\n--- Baseline（現行特徴量） ---")
    baseline_configs = [{"name": "baseline★", "kwargs": best_kwargs}]
    baseline_result = expanding_multi_backtest(
        df, returns, FEATURE_COLUMNS, baseline_configs,
        use_ranking=use_ranking, calibration_pct=0.10, window_months=3,
    )
    all_results.update(baseline_result)

    # 各特徴量セット
    for exp_name, extra_cols in EXTRA_FEATURES.items():
        # 列がデータに存在するか確認
        available_extra = [c for c in extra_cols if c in df.columns]
        if len(available_extra) != len(extra_cols):
            missing = set(extra_cols) - set(available_extra)
            print(f"\n--- {exp_name}: 欠損列あり {missing}（スキップ） ---")
            continue

        feat_cols = FEATURE_COLUMNS + available_extra
        print(f"\n--- {exp_name}: +{len(available_extra)}特徴量 ({', '.join(available_extra)}) ---")

        configs = [{"name": exp_name, "kwargs": best_kwargs}]
        result = expanding_multi_backtest(
            df, returns, feat_cols, configs,
            use_ranking=use_ranking, calibration_pct=0.10, window_months=3,
        )
        all_results.update(result)

    best_feat = print_comparison("Group B: 特徴量実験", all_results, "baseline★")

    return {
        "best_feature_set": best_feat,
        "experiments": all_results,
    }


# ============================================================
# メイン
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="モデル判断の再精査実験")
    parser.add_argument("--group", choices=["a", "b", "all"], default="all",
                        help="実行グループ (a=パラメータ, b=特徴量, all=全て)")
    parser.add_argument("--output", type=str, default=None,
                        help="結果保存パス (JSON)")
    args = parser.parse_args()

    setup_encoding()

    print(f"{'=' * 60}")
    print(f"  モデル判断の再精査実験")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Val期間: {VAL_START.date()} ~ {TEST_START.date()}")
    print(f"  Test期間: {TEST_START.date()} ~")
    print(f"{'=' * 60}")

    df, returns = load_data()

    results = {"timestamp": datetime.now().isoformat()}
    results["current_params"] = EXPANDING_BEST_PARAMS
    results["current_features"] = FEATURE_COLUMNS

    group_a_result = None
    group_b_result = None

    if args.group in ("a", "all"):
        group_a_result = run_group_a(df, returns)
        results["group_a"] = {
            "best_params": group_a_result["best_params"],
        }

    if args.group in ("b", "all"):
        best_kwargs = None
        if group_a_result:
            bp = group_a_result["best_params"]
            best_kwargs = {
                "kelly_fraction": bp.get("kelly_fraction", 0),
                "confidence_min": bp.get("confidence_min", 0.04),
                "quinella_top_n": bp.get("quinella_top_n", 2),
                "wide_top_n": bp.get("wide_top_n", 2),
                "skip_classes": bp.get("skip_classes", [4, 6]),
                "top_n": bp.get("top_n", 3),
                "use_ranking": bp.get("use_ranking", False),
            }
        group_b_result = run_group_b(df, returns, best_kwargs)
        results["group_b"] = {
            "best_feature_set": group_b_result["best_feature_set"],
        }

    # 最終サマリ
    print(f"\n{'#' * 60}")
    print(f"# 最終サマリ")
    print(f"{'#' * 60}")
    if group_a_result:
        bp = group_a_result["best_params"]
        print(f"\n  Group A 最良パラメータ:")
        for k, v in bp.items():
            current = EXPANDING_BEST_PARAMS.get(k, "N/A")
            marker = " ← CHANGED" if v != current else ""
            print(f"    {k}: {v}{marker}")

    if group_b_result:
        bf = group_b_result["best_feature_set"]
        print(f"\n  Group B 最良特徴量セット: {bf}")
        if bf != "baseline★":
            extra = EXTRA_FEATURES.get(bf, [])
            print(f"    追加特徴量: {extra}")

    # 結果保存
    output_path = args.output or str(PROCESSED_DIR / "reeval_results.json")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  結果保存: {output_path}")


if __name__ == "__main__":
    main()
