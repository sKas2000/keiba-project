"""
B7アブレーション分析: 9特徴量のうちどれが効いているかを検証

方法:
1. B7全部入り（9特徴量追加）をベースラインとする
2. 1特徴量ずつ抜く（leave-one-out）→ 9実験
3. 各特徴量グループ（Z-score, 騎手×馬, コース適性）単独の効果も確認

Usage:
    python tools/reeval_ablation.py
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

# B7の最適パラメータ（reeval_extra.pyの結果）
B7_BEST_KWARGS = {
    "kelly_fraction": 0,
    "confidence_min": 0.02,
    "skip_classes": [4, 6],
    "quinella_top_n": 2,
    "wide_top_n": 2,
    "top_n": 3,
}

# Baselineの最適パラメータ
BASELINE_KWARGS = {
    "kelly_fraction": 0,
    "confidence_min": 0.04,
    "skip_classes": [4, 6],
    "quinella_top_n": 2,
    "wide_top_n": 2,
    "top_n": 3,
}

# B7の9特徴量
B7_EXTRAS = EXTRA_FEATURES["B7_all_extra"]

# グループ分け
ZSCORE_FEATURES = [
    "z_surface_place_rate", "z_jockey_place_rate_365d",
    "z_avg_finish_last5", "z_career_place_rate",
    "z_trainer_place_rate_365d",
]
JOCKEY_HORSE_FEATURES = ["same_jockey_rides", "same_jockey_win_rate"]
COURSE_APT_FEATURES = ["course_dist_win_rate", "course_dist_place_rate"]


def run_leave_one_out(df, returns):
    """Leave-one-out: 1特徴量ずつ抜いて効果を検証"""
    print(f"\n{'#' * 60}")
    print(f"# Leave-One-Out 分析: 各特徴量の貢献度")
    print(f"{'#' * 60}")
    print(f"  B7全体 = {len(B7_EXTRAS)}特徴量")
    print(f"  各特徴量を1つ抜いた時のROI変化を測定\n")

    # B7全部入りベースライン
    b7_full = FEATURE_COLUMNS + B7_EXTRAS
    configs = [{"name": "B7_full", "kwargs": B7_BEST_KWARGS}]
    full_result = expanding_multi_backtest(
        df, returns, b7_full, configs,
        use_ranking=False, calibration_pct=0.10, window_months=3,
    )

    # Leave-one-out
    loo_results = dict(full_result)
    for feat in B7_EXTRAS:
        reduced_extras = [f for f in B7_EXTRAS if f != feat]
        feat_cols = FEATURE_COLUMNS + reduced_extras
        configs = [{"name": f"w/o_{feat}", "kwargs": B7_BEST_KWARGS}]
        result = expanding_multi_backtest(
            df, returns, feat_cols, configs,
            use_ranking=False, calibration_pct=0.10, window_months=3,
        )
        loo_results.update(result)

    print_comparison("Leave-One-Out 分析", loo_results)
    return loo_results


def run_leave_group_out(df, returns):
    """グループ単位でLeave-out: Z-score群, 騎手×馬群, コース適性群"""
    print(f"\n{'#' * 60}")
    print(f"# Leave-Group-Out 分析: 特徴量グループの貢献度")
    print(f"{'#' * 60}")

    b7_full = FEATURE_COLUMNS + B7_EXTRAS
    configs_base = [{"name": "B7_full", "kwargs": B7_BEST_KWARGS}]
    base_result = expanding_multi_backtest(
        df, returns, b7_full, configs_base,
        use_ranking=False, calibration_pct=0.10, window_months=3,
    )

    group_results = dict(base_result)
    groups = {
        "w/o_zscore_5feat": ZSCORE_FEATURES,
        "w/o_jockey_horse_2feat": JOCKEY_HORSE_FEATURES,
        "w/o_course_apt_2feat": COURSE_APT_FEATURES,
    }

    for group_name, group_feats in groups.items():
        reduced = [f for f in B7_EXTRAS if f not in group_feats]
        feat_cols = FEATURE_COLUMNS + reduced
        configs = [{"name": group_name, "kwargs": B7_BEST_KWARGS}]
        result = expanding_multi_backtest(
            df, returns, feat_cols, configs,
            use_ranking=False, calibration_pct=0.10, window_months=3,
        )
        group_results.update(result)

    print_comparison("Leave-Group-Out 分析", group_results)
    return group_results


def run_additive_analysis(df, returns):
    """加算的分析: Baseline → +Z-score → +騎手×馬 → +コース適性"""
    print(f"\n{'#' * 60}")
    print(f"# 加算的分析: 特徴量を順に追加")
    print(f"{'#' * 60}")

    all_results = {}

    steps = [
        ("step0_baseline", []),
        ("step1_+zscore", ZSCORE_FEATURES),
        ("step2_+jockey_horse", ZSCORE_FEATURES + JOCKEY_HORSE_FEATURES),
        ("step3_+course_apt(=B7)", ZSCORE_FEATURES + JOCKEY_HORSE_FEATURES + COURSE_APT_FEATURES),
    ]

    for name, extra_feats in steps:
        feat_cols = FEATURE_COLUMNS + extra_feats
        # step0はBaselineパラメータ、それ以降はB7パラメータ
        kw = BASELINE_KWARGS if len(extra_feats) == 0 else B7_BEST_KWARGS
        configs = [{"name": name, "kwargs": kw}]
        result = expanding_multi_backtest(
            df, returns, feat_cols, configs,
            use_ranking=False, calibration_pct=0.10, window_months=3,
        )
        all_results.update(result)

    print_comparison("加算的分析: 特徴量追加の効果", all_results)
    return all_results


def run_additive_reverse(df, returns):
    """逆順の加算的分析: Baseline → +コース適性 → +騎手×馬 → +Z-score"""
    print(f"\n{'#' * 60}")
    print(f"# 逆順加算的分析: 特徴量を逆順に追加")
    print(f"{'#' * 60}")

    all_results = {}

    steps = [
        ("step0_baseline", []),
        ("step1_+course_apt", COURSE_APT_FEATURES),
        ("step2_+jockey_horse", COURSE_APT_FEATURES + JOCKEY_HORSE_FEATURES),
        ("step3_+zscore(=B7)", COURSE_APT_FEATURES + JOCKEY_HORSE_FEATURES + ZSCORE_FEATURES),
    ]

    for name, extra_feats in steps:
        feat_cols = FEATURE_COLUMNS + extra_feats
        kw = BASELINE_KWARGS if len(extra_feats) == 0 else B7_BEST_KWARGS
        configs = [{"name": name, "kwargs": kw}]
        result = expanding_multi_backtest(
            df, returns, feat_cols, configs,
            use_ranking=False, calibration_pct=0.10, window_months=3,
        )
        all_results.update(result)

    print_comparison("逆順加算的分析: 特徴量追加の効果", all_results)
    return all_results


def run_window_size_experiment(df, returns):
    """ウィンドウサイズの効果検証（B7特徴量で）"""
    print(f"\n{'#' * 60}")
    print(f"# ウィンドウサイズ検証（B7特徴量）")
    print(f"{'#' * 60}")

    b7_full = FEATURE_COLUMNS + B7_EXTRAS
    all_results = {}

    for months in [2, 3, 4, 6]:
        name = f"window_{months}mo"
        configs = [{"name": name, "kwargs": B7_BEST_KWARGS}]
        result = expanding_multi_backtest(
            df, returns, b7_full, configs,
            use_ranking=False, calibration_pct=0.10, window_months=months,
        )
        all_results.update(result)

    print_comparison("ウィンドウサイズ検証", all_results)
    return all_results


def main():
    setup_encoding()
    print(f"{'=' * 60}")
    print(f"  B7特徴量アブレーション・追加実験")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 60}")

    df, returns = load_data()

    all_experiment_results = {}

    # 実験1: Leave-One-Out
    loo = run_leave_one_out(df, returns)
    all_experiment_results["leave_one_out"] = {k: str(v) for k, v in loo.items()}

    # 実験2: Leave-Group-Out
    lgo = run_leave_group_out(df, returns)
    all_experiment_results["leave_group_out"] = {k: str(v) for k, v in lgo.items()}

    # 実験3: 加算的分析（正順）
    add = run_additive_analysis(df, returns)
    all_experiment_results["additive"] = {k: str(v) for k, v in add.items()}

    # 実験4: 加算的分析（逆順）
    add_rev = run_additive_reverse(df, returns)
    all_experiment_results["additive_reverse"] = {k: str(v) for k, v in add_rev.items()}

    # 実験5: ウィンドウサイズ検証
    ws = run_window_size_experiment(df, returns)
    all_experiment_results["window_size"] = {k: str(v) for k, v in ws.items()}

    # 結果保存
    out_path = Path("data/processed/reeval_ablation_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_experiment_results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  結果保存: {out_path}")


if __name__ == "__main__":
    main()
