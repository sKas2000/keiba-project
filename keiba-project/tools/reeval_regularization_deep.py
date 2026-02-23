"""
正則化パラメータ l1_0.01_l2_0.1 の深掘り検証 + シード感度分析

LightGBMチューニングでTest ROI 121.7%を記録した l1=0.01, l2=0.1 について
複数の観点から再現性と頑健性を検証する。

実験:
1. シード感度分析: l1_0.01_l2_0.1 vs default を5シードで比較
2. 正則化の微細グリッド: l1_0.01付近のfine-grained search
3. l1_0.01_l2_0.1 × confidence_min スイープ
4. l1_0.01_l2_0.1 × calibration_pct スイープ
5. l1_0.01_l2_0.1 × window_months スイープ

Usage:
    python tools/reeval_regularization_deep.py
"""
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
from config.settings import FEATURE_COLUMNS, setup_encoding
from src.model.trainer import DEFAULT_BINARY_PARAMS
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

DEFAULT_PARAMS = DEFAULT_BINARY_PARAMS.copy()

# 検証対象の正則化パラメータ
LIGHT_REG = {"lambda_l1": 0.01, "lambda_l2": 0.1}
CURRENT_REG = {"lambda_l1": 0.1, "lambda_l2": 1.0}


def _make_params(**overrides):
    p = DEFAULT_PARAMS.copy()
    p.update(overrides)
    return p


def run_seed_sensitivity(df, returns):
    """実験1: シード感度分析 — 5シードで default vs l1_0.01_l2_0.1 を比較"""
    print(f"\n{'#' * 60}")
    print(f"# 実験1: シード感度分析 (5シード × 2パラメータ)")
    print(f"{'#' * 60}")

    all_results = {}
    seeds = [42, 123, 256, 777, 2024]

    for seed in seeds:
        # default params with seed
        params_default = _make_params(seed=seed)
        name_default = f"default_s{seed}"
        configs = [{"name": name_default, "kwargs": B7_BET_KWARGS}]
        result = expanding_multi_backtest(
            df, returns, B7_FEATURES, configs,
            use_ranking=False, calibration_pct=0.10, window_months=6,
            lgbm_binary_params=params_default,
        )
        all_results.update(result)

        # light reg params with seed
        params_light = _make_params(seed=seed, **LIGHT_REG)
        name_light = f"light_s{seed}"
        configs = [{"name": name_light, "kwargs": B7_BET_KWARGS}]
        result = expanding_multi_backtest(
            df, returns, B7_FEATURES, configs,
            use_ranking=False, calibration_pct=0.10, window_months=6,
            lgbm_binary_params=params_light,
        )
        all_results.update(result)

    print_comparison("シード感度分析", all_results)

    # シード別の平均を計算して表示
    print("\n  --- シード別サマリ ---")
    for group, prefix in [("default", "default_s"), ("light_reg", "light_s")]:
        vals, tests = [], []
        for seed in seeds:
            name = f"{prefix}{seed}"
            if name in all_results:
                r = all_results[name]
                vals.append(r.get("val_avg", 0))
                tests.append(r.get("test_avg", 0))
        if vals:
            print(f"  {group:15s}  Val平均: {sum(vals)/len(vals):.1f}%  "
                  f"Test平均: {sum(tests)/len(tests):.1f}%  "
                  f"Val std: {_std(vals):.1f}  Test std: {_std(tests):.1f}")

    return all_results


def _std(values):
    """標準偏差を計算"""
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return (sum((v - mean) ** 2 for v in values) / (n - 1)) ** 0.5


def run_fine_grid(df, returns):
    """実験2: l1_0.01付近の微細グリッドサーチ"""
    print(f"\n{'#' * 60}")
    print(f"# 実験2: 正則化パラメータ微細グリッド")
    print(f"{'#' * 60}")

    all_results = {}
    combos = [
        ("l1_0.001_l2_0.01", 0.001, 0.01),
        ("l1_0.005_l2_0.05", 0.005, 0.05),
        ("l1_0.01_l2_0.05", 0.01, 0.05),
        ("l1_0.01_l2_0.1", 0.01, 0.1),     # 前回Test最高
        ("l1_0.01_l2_0.2", 0.01, 0.2),
        ("l1_0.01_l2_0.5", 0.01, 0.5),
        ("l1_0.02_l2_0.1", 0.02, 0.1),
        ("l1_0.05_l2_0.1", 0.05, 0.1),
        ("l1_0.05_l2_0.5", 0.05, 0.5),
        ("l1_0.1_l2_1.0", 0.1, 1.0),       # 現行デフォルト
    ]

    for name, l1, l2 in combos:
        params = _make_params(lambda_l1=l1, lambda_l2=l2)
        configs = [{"name": name, "kwargs": B7_BET_KWARGS}]
        result = expanding_multi_backtest(
            df, returns, B7_FEATURES, configs,
            use_ranking=False, calibration_pct=0.10, window_months=6,
            lgbm_binary_params=params,
        )
        all_results.update(result)

    print_comparison("正則化微細グリッド", all_results)
    return all_results


def run_conf_sweep_with_light_reg(df, returns):
    """実験3: l1_0.01_l2_0.1 × confidence_min スイープ"""
    print(f"\n{'#' * 60}")
    print(f"# 実験3: light_reg × confidence_min スイープ")
    print(f"{'#' * 60}")

    params = _make_params(**LIGHT_REG)
    configs = []
    for conf in [0.00, 0.01, 0.02, 0.03, 0.04, 0.06]:
        kwargs = B7_BET_KWARGS.copy()
        kwargs["confidence_min"] = conf
        configs.append({"name": f"conf_{conf:.2f}", "kwargs": kwargs})

    all_results = expanding_multi_backtest(
        df, returns, B7_FEATURES, configs,
        use_ranking=False, calibration_pct=0.10, window_months=6,
        lgbm_binary_params=params,
    )

    print_comparison("light_reg × confidence_min", all_results)
    return all_results


def run_cal_sweep_with_light_reg(df, returns):
    """実験4: l1_0.01_l2_0.1 × calibration_pct スイープ"""
    print(f"\n{'#' * 60}")
    print(f"# 実験4: light_reg × calibration_pct スイープ")
    print(f"{'#' * 60}")

    params = _make_params(**LIGHT_REG)
    all_results = {}
    for cal in [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]:
        name = f"cal_{cal:.2f}"
        configs = [{"name": name, "kwargs": B7_BET_KWARGS}]
        result = expanding_multi_backtest(
            df, returns, B7_FEATURES, configs,
            use_ranking=False, calibration_pct=cal, window_months=6,
            lgbm_binary_params=params,
        )
        all_results.update(result)

    print_comparison("light_reg × calibration_pct", all_results)
    return all_results


def run_window_sweep_with_light_reg(df, returns):
    """実験5: l1_0.01_l2_0.1 × window_months スイープ"""
    print(f"\n{'#' * 60}")
    print(f"# 実験5: light_reg × window_months スイープ")
    print(f"{'#' * 60}")

    params = _make_params(**LIGHT_REG)
    all_results = {}
    for wm in [3, 4, 6, 9, 12]:
        name = f"win_{wm}mo"
        configs = [{"name": name, "kwargs": B7_BET_KWARGS}]
        result = expanding_multi_backtest(
            df, returns, B7_FEATURES, configs,
            use_ranking=False, calibration_pct=0.10, window_months=wm,
            lgbm_binary_params=params,
        )
        all_results.update(result)

    print_comparison("light_reg × window_months", all_results)
    return all_results


def main():
    setup_encoding()
    log_path = Path("data/processed/reeval_reg_deep_log.txt")

    print(f"{'=' * 60}")
    print(f"  正則化パラメータ深掘り検証 + シード感度分析")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 60}")
    print(f"  検証対象: lambda_l1=0.01, lambda_l2=0.1")
    print(f"  比較対象: lambda_l1=0.1, lambda_l2=1.0 (現行)")
    print(f"  ベース: B7特徴量 + 6ヶ月ウィンドウ")
    print(f"  ログ: {log_path}")

    df, returns = load_data()

    all_experiment_results = {}

    # 実験1: シード感度
    seed_results = run_seed_sensitivity(df, returns)
    all_experiment_results["seed_sensitivity"] = seed_results

    # 実験2: 微細グリッド
    grid_results = run_fine_grid(df, returns)
    all_experiment_results["fine_grid"] = grid_results

    # 実験3: confidence_min
    conf_results = run_conf_sweep_with_light_reg(df, returns)
    all_experiment_results["conf_sweep"] = conf_results

    # 実験4: calibration_pct
    cal_results = run_cal_sweep_with_light_reg(df, returns)
    all_experiment_results["cal_sweep"] = cal_results

    # 実験5: window_months
    win_results = run_window_sweep_with_light_reg(df, returns)
    all_experiment_results["window_sweep"] = win_results

    # 結果保存
    out_path = Path("data/processed/reeval_reg_deep_results.json")
    serializable = {}
    for exp_name, results in all_experiment_results.items():
        serializable[exp_name] = {
            name: {k: v for k, v in r.items() if isinstance(v, (int, float, str))}
            for name, r in results.items()
        }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "light_reg": LIGHT_REG,
            "current_reg": CURRENT_REG,
            "experiments": serializable,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n  結果保存: {out_path}")

    # 最終サマリ
    print(f"\n{'=' * 60}")
    print(f"  全実験完了")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
