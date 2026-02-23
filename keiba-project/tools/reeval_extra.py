"""
追加検証: B4_zscore / B7_all_extra のパラメータ最適化
"""
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
from config.settings import FEATURE_COLUMNS, setup_encoding
from tools.reeval_experiments import (
    load_data, expanding_multi_backtest, print_comparison,
    EXTRA_FEATURES, KEY_BETS, BET_JA,
)

# 特徴量セット定義
FEAT_BASELINE = FEATURE_COLUMNS
FEAT_B4 = FEATURE_COLUMNS + EXTRA_FEATURES["B4_zscore"]
FEAT_B7 = FEATURE_COLUMNS + EXTRA_FEATURES["B7_all_extra"]

# 共通パラメータ
BASE_KWARGS = {
    "kelly_fraction": 0,
    "skip_classes": [4, 6],
    "quinella_top_n": 2,
    "wide_top_n": 2,
    "top_n": 3,
}


def run_confidence_sweep(df, returns, feat_cols, feat_name):
    """confidence_minスイープ"""
    print(f"\n--- {feat_name}: confidence_min sweep ---")
    configs = []
    for val in [0.0, 0.02, 0.04, 0.06, 0.08]:
        kw = BASE_KWARGS.copy()
        kw["confidence_min"] = val
        name = f"conf_{val:.2f}"
        configs.append({"name": name, "kwargs": kw})

    results = expanding_multi_backtest(
        df, returns, feat_cols, configs,
        use_ranking=False, calibration_pct=0.10, window_months=3,
    )
    best = print_comparison(f"{feat_name}: confidence_min", results)
    return results, best


def run_skip_sweep(df, returns, feat_cols, feat_name, best_conf):
    """skip_classesスイープ"""
    print(f"\n--- {feat_name}: skip_classes sweep (conf={best_conf}) ---")
    configs = []
    for name, val in [
        ("skip_none", []),
        ("skip_4", [4]),
        ("skip_6", [6]),
        ("skip_4_6", [4, 6]),
    ]:
        kw = BASE_KWARGS.copy()
        kw["confidence_min"] = best_conf
        kw["skip_classes"] = val
        configs.append({"name": name, "kwargs": kw})

    results = expanding_multi_backtest(
        df, returns, feat_cols, configs,
        use_ranking=False, calibration_pct=0.10, window_months=3,
    )
    best = print_comparison(f"{feat_name}: skip_classes", results)
    return results, best


def run_top_n_sweep(df, returns, feat_cols, feat_name, best_conf, best_skip):
    """top_nスイープ"""
    print(f"\n--- {feat_name}: top_n sweep ---")
    configs = []
    for top_n in [3, 4, 5]:
        kw = BASE_KWARGS.copy()
        kw["confidence_min"] = best_conf
        kw["skip_classes"] = best_skip
        kw["top_n"] = top_n
        configs.append({"name": f"top_{top_n}", "kwargs": kw})

    results = expanding_multi_backtest(
        df, returns, feat_cols, configs,
        use_ranking=False, calibration_pct=0.10, window_months=3,
    )
    best = print_comparison(f"{feat_name}: top_n", results)
    return results, best


def run_kelly_test(df, returns, feat_cols, feat_name, best_conf, best_skip, best_top_n):
    """Kelly基準のテスト"""
    print(f"\n--- {feat_name}: Kelly criterion test ---")
    configs = []
    for kf in [0, 0.125, 0.25, 0.5]:
        kw = BASE_KWARGS.copy()
        kw["confidence_min"] = best_conf
        kw["skip_classes"] = best_skip
        kw["top_n"] = best_top_n
        kw["kelly_fraction"] = kf
        name = f"kelly_{kf:.3f}" if kf > 0 else "flat★"
        configs.append({"name": name, "kwargs": kw})

    results = expanding_multi_backtest(
        df, returns, feat_cols, configs,
        use_ranking=False, calibration_pct=0.10, window_months=3,
    )
    best = print_comparison(f"{feat_name}: Kelly", results)
    return results, best


def run_calibration_pct_sweep(df, returns, feat_cols, feat_name, best_kwargs):
    """キャリブレーション割合スイープ"""
    print(f"\n--- {feat_name}: calibration_pct sweep ---")
    all_results = {}
    for pct in [0.05, 0.10, 0.15, 0.20]:
        name = f"cal_{pct:.2f}"
        configs = [{"name": name, "kwargs": best_kwargs}]
        result = expanding_multi_backtest(
            df, returns, feat_cols, configs,
            use_ranking=False, calibration_pct=pct, window_months=3,
        )
        all_results.update(result)

    best = print_comparison(f"{feat_name}: calibration_pct", all_results)
    return all_results, best


def analyze_feature_set(df, returns, feat_cols, feat_name):
    """特徴量セットの包括的分析"""
    print(f"\n{'#' * 60}")
    print(f"# {feat_name} 包括的パラメータ探索")
    print(f"{'#' * 60}")

    # Step 1: confidence_min
    _, best_conf_name = run_confidence_sweep(df, returns, feat_cols, feat_name)
    best_conf = float(best_conf_name.replace("conf_", ""))

    # Step 2: skip_classes
    _, best_skip_name = run_skip_sweep(df, returns, feat_cols, feat_name, best_conf)
    skip_map = {"skip_none": [], "skip_4": [4], "skip_6": [6], "skip_4_6": [4, 6]}
    best_skip = skip_map.get(best_skip_name, [4, 6])

    # Step 3: top_n
    _, best_top_name = run_top_n_sweep(df, returns, feat_cols, feat_name, best_conf, best_skip)
    best_top_n = int(best_top_name.replace("top_", ""))

    # Step 4: Kelly criterion
    run_kelly_test(df, returns, feat_cols, feat_name, best_conf, best_skip, best_top_n)

    # Step 5: calibration_pct
    best_kw = {
        "kelly_fraction": 0,
        "confidence_min": best_conf,
        "skip_classes": best_skip,
        "quinella_top_n": 2,
        "wide_top_n": 2,
        "top_n": best_top_n,
    }
    run_calibration_pct_sweep(df, returns, feat_cols, feat_name, best_kw)

    print(f"\n  {feat_name} 最良パラメータ:")
    print(f"    confidence_min: {best_conf}")
    print(f"    skip_classes: {best_skip}")
    print(f"    top_n: {best_top_n}")
    return best_kw


def main():
    setup_encoding()
    print(f"{'=' * 60}")
    print(f"  追加検証: B4_zscore / B7_all_extra パラメータ最適化")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 60}")

    df, returns = load_data()

    # Baseline 包括分析
    baseline_best = analyze_feature_set(df, returns, FEAT_BASELINE, "Baseline")

    # B4 包括分析
    b4_best = analyze_feature_set(df, returns, FEAT_B4, "B4_zscore")

    # B7 包括分析
    b7_best = analyze_feature_set(df, returns, FEAT_B7, "B7_all_extra")

    # 最終比較
    print(f"\n{'#' * 60}")
    print(f"# 最終比較: 各特徴量セットの最適パラメータで比較")
    print(f"{'#' * 60}")

    configs = [
        {"name": "Baseline_opt", "kwargs": baseline_best},
        {"name": "B4_zscore_opt", "kwargs": b4_best},
        {"name": "B7_all_opt", "kwargs": b7_best},
    ]

    final_results = {}
    for feat_name, feat_cols in [
        ("Baseline_opt", FEAT_BASELINE),
        ("B4_zscore_opt", FEAT_B4),
        ("B7_all_opt", FEAT_B7),
    ]:
        cfg = [c for c in configs if c["name"] == feat_name][0]
        result = expanding_multi_backtest(
            df, returns, feat_cols, [cfg],
            use_ranking=False, calibration_pct=0.10, window_months=3,
        )
        final_results.update(result)

    print_comparison("最終比較: 特徴量×最適パラメータ", final_results)

    # 結果保存
    output = {
        "timestamp": datetime.now().isoformat(),
        "baseline_best": baseline_best,
        "b4_best": b4_best,
        "b7_best": b7_best,
    }
    out_path = Path("data/processed/reeval_extra_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  結果保存: {out_path}")


if __name__ == "__main__":
    main()
