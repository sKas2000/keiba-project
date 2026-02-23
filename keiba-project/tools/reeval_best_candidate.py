"""
最有望候補の検証: B7から same_jockey_rides 除外 + 6ヶ月ウィンドウ
アブレーション結果を踏まえた最終検証実験

Usage:
    python tools/reeval_best_candidate.py
"""
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
from config.settings import FEATURE_COLUMNS, setup_encoding
from tools.reeval_experiments import (
    load_data, expanding_multi_backtest, print_comparison,
    EXTRA_FEATURES,
)

# B7からsame_jockey_ridesを除外した8特徴量
B7_MINUS_RIDES = [f for f in EXTRA_FEATURES["B7_all_extra"] if f != "same_jockey_rides"]
FEAT_B7_MINUS = FEATURE_COLUMNS + B7_MINUS_RIDES
FEAT_B7_FULL = FEATURE_COLUMNS + EXTRA_FEATURES["B7_all_extra"]
FEAT_BASELINE = FEATURE_COLUMNS

# パラメータ候補
B7_KWARGS = {
    "kelly_fraction": 0,
    "confidence_min": 0.02,
    "skip_classes": [4, 6],
    "quinella_top_n": 2,
    "wide_top_n": 2,
    "top_n": 3,
}

BASELINE_KWARGS = {
    "kelly_fraction": 0,
    "confidence_min": 0.04,
    "skip_classes": [4, 6],
    "quinella_top_n": 2,
    "wide_top_n": 2,
    "top_n": 3,
}


def run_main_comparison(df, returns):
    """メイン比較: Baseline vs B7 vs B7-rides, 各ウィンドウサイズ"""
    print(f"\n{'#' * 60}")
    print(f"# メイン比較: 特徴量×ウィンドウサイズ")
    print(f"{'#' * 60}")

    all_results = {}

    configs_map = {
        "Baseline": (FEAT_BASELINE, BASELINE_KWARGS),
        "B7_full": (FEAT_B7_FULL, B7_KWARGS),
        "B7_minus_rides": (FEAT_B7_MINUS, B7_KWARGS),
    }

    for window in [3, 6]:
        for name, (feat_cols, kw) in configs_map.items():
            config_name = f"{name}_w{window}mo"
            configs = [{"name": config_name, "kwargs": kw}]
            result = expanding_multi_backtest(
                df, returns, feat_cols, configs,
                use_ranking=False, calibration_pct=0.10, window_months=window,
            )
            all_results.update(result)

    print_comparison("特徴量×ウィンドウサイズ 比較", all_results)
    return all_results


def run_confidence_sweep_b7minus(df, returns):
    """B7-rides で confidence_min を再スイープ（6ヶ月ウィンドウ）"""
    print(f"\n{'#' * 60}")
    print(f"# B7-rides 6ヶ月: confidence_min スイープ")
    print(f"{'#' * 60}")

    configs = []
    for val in [0.00, 0.02, 0.04, 0.06]:
        kw = B7_KWARGS.copy()
        kw["confidence_min"] = val
        configs.append({"name": f"conf_{val:.2f}", "kwargs": kw})

    results = expanding_multi_backtest(
        df, returns, FEAT_B7_MINUS, configs,
        use_ranking=False, calibration_pct=0.10, window_months=6,
    )
    print_comparison("B7-rides 6mo: confidence_min", results)
    return results


def run_calibration_sweep_b7minus(df, returns):
    """B7-rides で calibration_pct を再スイープ（6ヶ月ウィンドウ）"""
    print(f"\n{'#' * 60}")
    print(f"# B7-rides 6ヶ月: calibration_pct スイープ")
    print(f"{'#' * 60}")

    all_results = {}
    for pct in [0.05, 0.08, 0.10, 0.12, 0.15]:
        name = f"cal_{pct:.2f}"
        configs = [{"name": name, "kwargs": B7_KWARGS}]
        result = expanding_multi_backtest(
            df, returns, FEAT_B7_MINUS, configs,
            use_ranking=False, calibration_pct=pct, window_months=6,
        )
        all_results.update(result)

    print_comparison("B7-rides 6mo: calibration_pct", all_results)
    return all_results


def run_skip_classes_sweep_b7minus(df, returns):
    """B7-rides で skip_classes を再スイープ（6ヶ月ウィンドウ）"""
    print(f"\n{'#' * 60}")
    print(f"# B7-rides 6ヶ月: skip_classes スイープ")
    print(f"{'#' * 60}")

    configs = []
    for name, val in [
        ("skip_none", []),
        ("skip_4", [4]),
        ("skip_6", [6]),
        ("skip_4_6", [4, 6]),
    ]:
        kw = B7_KWARGS.copy()
        kw["skip_classes"] = val
        configs.append({"name": name, "kwargs": kw})

    results = expanding_multi_backtest(
        df, returns, FEAT_B7_MINUS, configs,
        use_ranking=False, calibration_pct=0.10, window_months=6,
    )
    print_comparison("B7-rides 6mo: skip_classes", results)
    return results


def main():
    setup_encoding()
    print(f"{'=' * 60}")
    print(f"  最有望候補の検証: B7-rides + 6ヶ月ウィンドウ")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 60}")
    print(f"  B7-rides = B7全9特徴量からsame_jockey_ridesを除外した8特徴量")
    print(f"  除外特徴量: {B7_MINUS_RIDES}")

    df, returns = load_data()

    # 実験1: メイン比較
    main_results = run_main_comparison(df, returns)

    # 実験2: confidence_min再スイープ
    conf_results = run_confidence_sweep_b7minus(df, returns)

    # 実験3: skip_classes再スイープ
    skip_results = run_skip_classes_sweep_b7minus(df, returns)

    # 実験4: calibration_pct再スイープ
    cal_results = run_calibration_sweep_b7minus(df, returns)

    # 結果保存
    out_path = Path("data/processed/reeval_best_candidate_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "features_used": B7_MINUS_RIDES,
            "features_excluded": ["same_jockey_rides"],
        }, f, ensure_ascii=False, indent=2)
    print(f"\n  結果保存: {out_path}")


if __name__ == "__main__":
    main()
