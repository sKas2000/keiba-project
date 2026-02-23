"""
LightGBM ハイパーパラメータチューニング実験
B7特徴量 + 6ヶ月ウィンドウの構成で、LightGBMモデルパラメータを最適化

実験:
1. learning_rate スイープ（0.01, 0.03, 0.05, 0.08, 0.1）
2. num_leaves / max_depth スイープ
3. 正則化パラメータ（lambda_l1, lambda_l2）スイープ
4. feature_fraction / bagging_fraction スイープ
5. min_child_samples スイープ
6. 最良組合せ vs デフォルト vs Ranking有効化

Usage:
    python tools/reeval_lgbm_tuning.py
"""
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
from config.settings import FEATURE_COLUMNS, setup_encoding
from src.model.trainer import DEFAULT_BINARY_PARAMS, DEFAULT_RANK_PARAMS
from tools.reeval_experiments import (
    load_data, expanding_multi_backtest, print_comparison,
    EXTRA_FEATURES,
)

# B7全特徴量
B7_FEATURES = FEATURE_COLUMNS + EXTRA_FEATURES["B7_all_extra"]

# B7最適ベットパラメータ
B7_BET_KWARGS = {
    "kelly_fraction": 0,
    "confidence_min": 0.02,
    "skip_classes": [4, 6],
    "quinella_top_n": 2,
    "wide_top_n": 2,
    "top_n": 3,
}

# デフォルトLGBMパラメータ（変更前の基準）
DEFAULT_PARAMS = DEFAULT_BINARY_PARAMS.copy()


def _make_params(**overrides):
    """デフォルトパラメータに上書きして返す"""
    p = DEFAULT_PARAMS.copy()
    p.update(overrides)
    return p


def run_learning_rate_sweep(df, returns):
    """実験1: learning_rate スイープ"""
    print(f"\n{'#' * 60}")
    print(f"# 実験1: learning_rate スイープ")
    print(f"{'#' * 60}")

    all_results = {}
    for lr in [0.01, 0.03, 0.05, 0.08, 0.1]:
        name = f"lr_{lr}"
        params = _make_params(learning_rate=lr)
        configs = [{"name": name, "kwargs": B7_BET_KWARGS}]
        result = expanding_multi_backtest(
            df, returns, B7_FEATURES, configs,
            use_ranking=False, calibration_pct=0.10, window_months=6,
            lgbm_binary_params=params,
        )
        all_results.update(result)

    best = print_comparison("learning_rate スイープ", all_results)
    return all_results, best


def run_tree_structure_sweep(df, returns, best_lr=None):
    """実験2: num_leaves / max_depth スイープ"""
    print(f"\n{'#' * 60}")
    print(f"# 実験2: num_leaves / max_depth スイープ")
    print(f"{'#' * 60}")

    lr = best_lr or DEFAULT_PARAMS["learning_rate"]
    all_results = {}

    combos = [
        ("nl31_d5", 31, 5),
        ("nl63_d7", 63, 7),   # current default
        ("nl63_d9", 63, 9),
        ("nl127_d7", 127, 7),
        ("nl127_d9", 127, 9),
        ("nl31_d-1", 31, -1),  # no depth limit
        ("nl63_d-1", 63, -1),
    ]

    for name, nl, md in combos:
        params = _make_params(learning_rate=lr, num_leaves=nl, max_depth=md)
        configs = [{"name": name, "kwargs": B7_BET_KWARGS}]
        result = expanding_multi_backtest(
            df, returns, B7_FEATURES, configs,
            use_ranking=False, calibration_pct=0.10, window_months=6,
            lgbm_binary_params=params,
        )
        all_results.update(result)

    best = print_comparison("num_leaves / max_depth スイープ", all_results)
    return all_results, best


def run_regularization_sweep(df, returns, best_lr=None, best_nl=None, best_md=None):
    """実験3: 正則化パラメータ（lambda_l1, lambda_l2）スイープ"""
    print(f"\n{'#' * 60}")
    print(f"# 実験3: 正則化パラメータスイープ")
    print(f"{'#' * 60}")

    lr = best_lr or DEFAULT_PARAMS["learning_rate"]
    nl = best_nl or DEFAULT_PARAMS["num_leaves"]
    md = best_md or DEFAULT_PARAMS["max_depth"]

    all_results = {}
    combos = [
        ("l1_0.01_l2_0.1", 0.01, 0.1),
        ("l1_0.1_l2_1.0", 0.1, 1.0),    # current default
        ("l1_0.1_l2_5.0", 0.1, 5.0),
        ("l1_1.0_l2_1.0", 1.0, 1.0),
        ("l1_1.0_l2_5.0", 1.0, 5.0),
        ("l1_0.0_l2_0.0", 0.0, 0.0),    # no regularization
        ("l1_5.0_l2_10.0", 5.0, 10.0),   # strong regularization
    ]

    for name, l1, l2 in combos:
        params = _make_params(learning_rate=lr, num_leaves=nl, max_depth=md,
                              lambda_l1=l1, lambda_l2=l2)
        configs = [{"name": name, "kwargs": B7_BET_KWARGS}]
        result = expanding_multi_backtest(
            df, returns, B7_FEATURES, configs,
            use_ranking=False, calibration_pct=0.10, window_months=6,
            lgbm_binary_params=params,
        )
        all_results.update(result)

    best = print_comparison("正則化パラメータスイープ", all_results)
    return all_results, best


def run_sampling_sweep(df, returns, best_lr=None, best_nl=None, best_md=None,
                       best_l1=None, best_l2=None):
    """実験4: feature_fraction / bagging_fraction スイープ"""
    print(f"\n{'#' * 60}")
    print(f"# 実験4: サンプリングパラメータスイープ")
    print(f"{'#' * 60}")

    lr = best_lr or DEFAULT_PARAMS["learning_rate"]
    nl = best_nl or DEFAULT_PARAMS["num_leaves"]
    md = best_md or DEFAULT_PARAMS["max_depth"]
    l1 = best_l1 or DEFAULT_PARAMS["lambda_l1"]
    l2 = best_l2 or DEFAULT_PARAMS["lambda_l2"]

    all_results = {}
    combos = [
        ("ff0.6_bf0.6", 0.6, 0.6),
        ("ff0.7_bf0.7", 0.7, 0.7),
        ("ff0.8_bf0.8", 0.8, 0.8),  # current default
        ("ff0.9_bf0.9", 0.9, 0.9),
        ("ff1.0_bf1.0", 1.0, 1.0),  # no sampling
        ("ff0.7_bf0.9", 0.7, 0.9),
        ("ff0.9_bf0.7", 0.9, 0.7),
    ]

    for name, ff, bf in combos:
        params = _make_params(learning_rate=lr, num_leaves=nl, max_depth=md,
                              lambda_l1=l1, lambda_l2=l2,
                              feature_fraction=ff, bagging_fraction=bf)
        configs = [{"name": name, "kwargs": B7_BET_KWARGS}]
        result = expanding_multi_backtest(
            df, returns, B7_FEATURES, configs,
            use_ranking=False, calibration_pct=0.10, window_months=6,
            lgbm_binary_params=params,
        )
        all_results.update(result)

    best = print_comparison("サンプリングパラメータスイープ", all_results)
    return all_results, best


def run_min_child_sweep(df, returns, best_lr=None, best_nl=None, best_md=None,
                        best_l1=None, best_l2=None, best_ff=None, best_bf=None):
    """実験5: min_child_samples スイープ"""
    print(f"\n{'#' * 60}")
    print(f"# 実験5: min_child_samples スイープ")
    print(f"{'#' * 60}")

    lr = best_lr or DEFAULT_PARAMS["learning_rate"]
    nl = best_nl or DEFAULT_PARAMS["num_leaves"]
    md = best_md or DEFAULT_PARAMS["max_depth"]
    l1 = best_l1 or DEFAULT_PARAMS["lambda_l1"]
    l2 = best_l2 or DEFAULT_PARAMS["lambda_l2"]
    ff = best_ff or DEFAULT_PARAMS["feature_fraction"]
    bf = best_bf or DEFAULT_PARAMS["bagging_fraction"]

    all_results = {}
    for mcs in [5, 10, 20, 30, 50, 80]:
        name = f"mcs_{mcs}"
        params = _make_params(learning_rate=lr, num_leaves=nl, max_depth=md,
                              lambda_l1=l1, lambda_l2=l2,
                              feature_fraction=ff, bagging_fraction=bf,
                              min_child_samples=mcs)
        configs = [{"name": name, "kwargs": B7_BET_KWARGS}]
        result = expanding_multi_backtest(
            df, returns, B7_FEATURES, configs,
            use_ranking=False, calibration_pct=0.10, window_months=6,
            lgbm_binary_params=params,
        )
        all_results.update(result)

    best = print_comparison("min_child_samples スイープ", all_results)
    return all_results, best


def run_final_comparison(df, returns, best_lgbm_params):
    """実験6: 最良LGBMパラメータ vs デフォルト vs Ranking有効化"""
    print(f"\n{'#' * 60}")
    print(f"# 実験6: 最終比較（LGBM最良 vs デフォルト vs Ranking）")
    print(f"{'#' * 60}")

    # デフォルトLGBMパラメータ
    print("\n  [Default LGBM params]")
    configs_default = [{"name": "default_lgbm", "kwargs": B7_BET_KWARGS}]
    result_default = expanding_multi_backtest(
        df, returns, B7_FEATURES, configs_default,
        use_ranking=False, calibration_pct=0.10, window_months=6,
    )

    # 最良LGBMパラメータ
    print("\n  [Best LGBM params]")
    configs_best = [{"name": "best_lgbm", "kwargs": B7_BET_KWARGS}]
    result_best = expanding_multi_backtest(
        df, returns, B7_FEATURES, configs_best,
        use_ranking=False, calibration_pct=0.10, window_months=6,
        lgbm_binary_params=best_lgbm_params,
    )

    # 最良LGBM + Ranking
    print("\n  [Best LGBM + Ranking]")
    configs_rank = [{"name": "best_lgbm+rank", "kwargs": B7_BET_KWARGS}]
    result_rank = expanding_multi_backtest(
        df, returns, B7_FEATURES, configs_rank,
        use_ranking=True, calibration_pct=0.10, window_months=6,
        lgbm_binary_params=best_lgbm_params,
    )

    all_results = {**result_default, **result_best, **result_rank}
    print_comparison("最終比較: LGBM最良 vs デフォルト vs Ranking", all_results)
    return all_results


def _extract_param(config_name, param_name, combos_map):
    """config名から対応するパラメータ値を取得"""
    return combos_map.get(config_name, {}).get(param_name)


def main():
    setup_encoding()
    print(f"{'=' * 60}")
    print(f"  LightGBM ハイパーパラメータチューニング")
    print(f"  B7特徴量 + 6ヶ月ウィンドウ")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 60}")
    print(f"  デフォルトパラメータ:")
    for k, v in DEFAULT_PARAMS.items():
        if k not in ("objective", "metric", "verbose", "seed"):
            print(f"    {k}: {v}")

    df, returns = load_data()

    # 各実験の最良パラメータを追跡
    best_lr = None
    best_nl = None
    best_md = None
    best_l1 = None
    best_l2 = None
    best_ff = None
    best_bf = None
    best_mcs = None

    # 実験1: learning_rate
    lr_results, lr_best_name = run_learning_rate_sweep(df, returns)
    lr_map = {f"lr_{lr}": lr for lr in [0.01, 0.03, 0.05, 0.08, 0.1]}
    if lr_best_name in lr_map:
        best_lr = lr_map[lr_best_name]
    print(f"  → Best learning_rate: {best_lr}")

    # 実験2: num_leaves / max_depth
    tree_results, tree_best_name = run_tree_structure_sweep(df, returns, best_lr)
    tree_map = {
        "nl31_d5": (31, 5), "nl63_d7": (63, 7), "nl63_d9": (63, 9),
        "nl127_d7": (127, 7), "nl127_d9": (127, 9),
        "nl31_d-1": (31, -1), "nl63_d-1": (63, -1),
    }
    if tree_best_name in tree_map:
        best_nl, best_md = tree_map[tree_best_name]
    print(f"  → Best num_leaves: {best_nl}, max_depth: {best_md}")

    # 実験3: 正則化
    reg_results, reg_best_name = run_regularization_sweep(
        df, returns, best_lr, best_nl, best_md)
    reg_map = {
        "l1_0.01_l2_0.1": (0.01, 0.1), "l1_0.1_l2_1.0": (0.1, 1.0),
        "l1_0.1_l2_5.0": (0.1, 5.0), "l1_1.0_l2_1.0": (1.0, 1.0),
        "l1_1.0_l2_5.0": (1.0, 5.0), "l1_0.0_l2_0.0": (0.0, 0.0),
        "l1_5.0_l2_10.0": (5.0, 10.0),
    }
    if reg_best_name in reg_map:
        best_l1, best_l2 = reg_map[reg_best_name]
    print(f"  → Best lambda_l1: {best_l1}, lambda_l2: {best_l2}")

    # 実験4: サンプリング
    samp_results, samp_best_name = run_sampling_sweep(
        df, returns, best_lr, best_nl, best_md, best_l1, best_l2)
    samp_map = {
        "ff0.6_bf0.6": (0.6, 0.6), "ff0.7_bf0.7": (0.7, 0.7),
        "ff0.8_bf0.8": (0.8, 0.8), "ff0.9_bf0.9": (0.9, 0.9),
        "ff1.0_bf1.0": (1.0, 1.0), "ff0.7_bf0.9": (0.7, 0.9),
        "ff0.9_bf0.7": (0.9, 0.7),
    }
    if samp_best_name in samp_map:
        best_ff, best_bf = samp_map[samp_best_name]
    print(f"  → Best feature_fraction: {best_ff}, bagging_fraction: {best_bf}")

    # 実験5: min_child_samples
    mcs_results, mcs_best_name = run_min_child_sweep(
        df, returns, best_lr, best_nl, best_md, best_l1, best_l2, best_ff, best_bf)
    mcs_map = {f"mcs_{v}": v for v in [5, 10, 20, 30, 50, 80]}
    if mcs_best_name in mcs_map:
        best_mcs = mcs_map[mcs_best_name]
    print(f"  → Best min_child_samples: {best_mcs}")

    # 最良パラメータ構築
    best_lgbm_params = _make_params(
        learning_rate=best_lr or DEFAULT_PARAMS["learning_rate"],
        num_leaves=best_nl or DEFAULT_PARAMS["num_leaves"],
        max_depth=best_md or DEFAULT_PARAMS["max_depth"],
        lambda_l1=best_l1 if best_l1 is not None else DEFAULT_PARAMS["lambda_l1"],
        lambda_l2=best_l2 if best_l2 is not None else DEFAULT_PARAMS["lambda_l2"],
        feature_fraction=best_ff or DEFAULT_PARAMS["feature_fraction"],
        bagging_fraction=best_bf or DEFAULT_PARAMS["bagging_fraction"],
        min_child_samples=best_mcs or DEFAULT_PARAMS["min_child_samples"],
    )

    print(f"\n{'=' * 60}")
    print(f"  最良LGBMパラメータ:")
    for k, v in best_lgbm_params.items():
        if k not in ("objective", "metric", "verbose", "seed"):
            default_v = DEFAULT_PARAMS.get(k)
            marker = " ← CHANGED" if v != default_v else ""
            print(f"    {k}: {v}{marker}")
    print(f"{'=' * 60}")

    # 実験6: 最終比較
    final_results = run_final_comparison(df, returns, best_lgbm_params)

    # 結果保存
    out_path = Path("data/processed/reeval_lgbm_tuning_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "best_lgbm_params": {k: v for k, v in best_lgbm_params.items()
                                  if k not in ("objective", "metric", "verbose", "seed")},
            "default_params": {k: v for k, v in DEFAULT_PARAMS.items()
                                if k not in ("objective", "metric", "verbose", "seed")},
        }, f, ensure_ascii=False, indent=2)
    print(f"\n  結果保存: {out_path}")


if __name__ == "__main__":
    main()
