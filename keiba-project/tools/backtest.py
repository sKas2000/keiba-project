#!/usr/bin/env python3
"""
バックテスト v1.0
==================
学習済みモデルの予測結果を使って、過去レースでの回収率をシミュレーションする。

使い方:
  python backtest.py [--input data/ml/processed/features.csv]
  python backtest.py --val-start 2025-01-01  # 検証期間指定
"""

VERSION = "1.0"

import json
import sys
from pathlib import Path
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import lightgbm as lgb

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "ml"
DEFAULT_INPUT = DATA_DIR / "processed" / "features.csv"
MODEL_DIR = DATA_DIR / "models"

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
# バックテストロジック
# ============================================================

def run_backtest(input_path: str, model_dir: str = str(MODEL_DIR),
                 val_start: str = "2025-01-01",
                 bet_threshold: float = 0.35,
                 top_n: int = 3) -> dict:
    """
    バックテストを実行
    Args:
        input_path: 特徴量CSV
        model_dir: モデルディレクトリ
        val_start: 検証開始日
        bet_threshold: 馬券購入閾値（予測確率がこれ以上で購入）
        top_n: 各レースで上位何頭を購入対象とするか
    Returns:
        バックテスト結果辞書
    """
    model_dir = Path(model_dir)

    # データ読み込み
    df = pd.read_csv(input_path, dtype={"race_id": str, "horse_id": str})
    df["race_date"] = pd.to_datetime(df["race_date"])

    # 検証期間のみ
    val_df = df[df["race_date"] >= pd.Timestamp(val_start)].copy()
    if len(val_df) == 0:
        print("[ERROR] 検証期間のデータがありません")
        return {}

    # モデル読み込み
    binary_model_path = model_dir / "binary_model.txt"
    if not binary_model_path.exists():
        print(f"[ERROR] モデルが見つかりません: {binary_model_path}")
        return {}

    model = lgb.Booster(model_file=str(binary_model_path))

    # 予測
    available = [c for c in FEATURE_COLUMNS if c in val_df.columns]
    X_val = val_df[available].values.astype(np.float32)
    val_df["pred_prob"] = model.predict(X_val)

    # 各レースごとにシミュレーション
    results = {
        "races": 0,
        "bets_win": {"count": 0, "invested": 0, "returned": 0, "hits": 0},
        "bets_place": {"count": 0, "invested": 0, "returned": 0, "hits": 0},
        "prediction_accuracy": {"top1_hit": 0, "top3_hit": 0, "total": 0},
        "race_details": [],
    }

    for race_id, race_group in val_df.groupby("race_id"):
        if len(race_group) < 4:
            continue

        race_group = race_group.sort_values("pred_prob", ascending=False)
        results["races"] += 1

        # 予測精度
        top1_pred = race_group.iloc[0]
        results["prediction_accuracy"]["total"] += 1
        if top1_pred.get("finish_position", 99) == 1:
            results["prediction_accuracy"]["top1_hit"] += 1

        top3_pred = race_group.head(3)
        top3_actual = (top3_pred["finish_position"] <= 3).sum()
        results["prediction_accuracy"]["top3_hit"] += top3_actual

        # 単勝シミュレーション: 予測1位で閾値以上なら購入
        if top1_pred["pred_prob"] >= bet_threshold and "win_odds" in race_group.columns:
            results["bets_win"]["count"] += 1
            results["bets_win"]["invested"] += 100  # 1口100円
            if top1_pred.get("finish_position", 99) == 1:
                payout = top1_pred.get("win_odds", 0) * 100
                results["bets_win"]["returned"] += payout
                results["bets_win"]["hits"] += 1

        # 複勝シミュレーション: 予測上位top_n頭で閾値以上なら購入
        for _, horse in race_group.head(top_n).iterrows():
            if horse["pred_prob"] >= bet_threshold * 0.8 and "win_odds" in race_group.columns:
                results["bets_place"]["count"] += 1
                results["bets_place"]["invested"] += 100
                if horse.get("finish_position", 99) <= 3:
                    # 複勝オッズ概算（単勝オッズの30%）
                    est_place_odds = max(horse.get("win_odds", 1) * 0.3, 1.1)
                    payout = est_place_odds * 100
                    results["bets_place"]["returned"] += payout
                    results["bets_place"]["hits"] += 1

        # レース詳細（上位5頭）
        detail = {
            "race_id": race_id,
            "date": str(race_group.iloc[0].get("race_date", "")),
            "predictions": [],
        }
        for _, horse in race_group.head(5).iterrows():
            detail["predictions"].append({
                "horse": horse.get("horse_name", ""),
                "pred_prob": round(float(horse["pred_prob"]), 3),
                "actual_finish": int(horse.get("finish_position", 0)),
                "odds": float(horse.get("win_odds", 0)),
            })
        results["race_details"].append(detail)

    return results


def print_backtest_report(results: dict):
    """バックテスト結果を表示"""
    if not results:
        return

    print(f"\n{'=' * 60}")
    print(f"  バックテスト結果レポート v{VERSION}")
    print(f"{'=' * 60}")

    races = results["races"]
    print(f"\n  対象レース数: {races}")

    # 予測精度
    pa = results["prediction_accuracy"]
    total = pa["total"]
    if total > 0:
        print(f"\n  [予測精度]")
        print(f"    1着的中率:   {pa['top1_hit']}/{total} = {pa['top1_hit']/total:.1%}")
        print(f"    3着内的中数: {pa['top3_hit']}/{total*3} = {pa['top3_hit']/(total*3):.1%}")

    # 単勝
    bw = results["bets_win"]
    if bw["count"] > 0:
        roi_win = bw["returned"] / bw["invested"] * 100 if bw["invested"] > 0 else 0
        print(f"\n  [単勝シミュレーション]")
        print(f"    購入回数: {bw['count']}")
        print(f"    的中回数: {bw['hits']} ({bw['hits']/bw['count']:.1%})")
        print(f"    投資額:   ¥{bw['invested']:,}")
        print(f"    回収額:   ¥{bw['returned']:,.0f}")
        print(f"    回収率:   {roi_win:.1f}%")

    # 複勝
    bp = results["bets_place"]
    if bp["count"] > 0:
        roi_place = bp["returned"] / bp["invested"] * 100 if bp["invested"] > 0 else 0
        print(f"\n  [複勝シミュレーション]")
        print(f"    購入回数: {bp['count']}")
        print(f"    的中回数: {bp['hits']} ({bp['hits']/bp['count']:.1%})")
        print(f"    投資額:   ¥{bp['invested']:,}")
        print(f"    回収額:   ¥{bp['returned']:,.0f}")
        print(f"    回収率:   {roi_place:.1f}%")

    # サンプルレース
    details = results.get("race_details", [])
    if details:
        print(f"\n  [サンプルレース（最新5件）]")
        for detail in details[-5:]:
            print(f"\n    {detail['race_id']} ({detail['date']})")
            for pred in detail["predictions"][:3]:
                hit = "◎" if pred["actual_finish"] <= 3 else "×"
                print(f"      {hit} {pred['horse']:12s} "
                      f"予測{pred['pred_prob']:.1%} "
                      f"実際{pred['actual_finish']}着 "
                      f"オッズ{pred['odds']:.1f}")

    print(f"\n{'=' * 60}\n")


# ============================================================
# メイン処理
# ============================================================

def main():
    parser = ArgumentParser(description="バックテスト")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="特徴量CSV")
    parser.add_argument("--model-dir", default=str(MODEL_DIR), help="モデルディレクトリ")
    parser.add_argument("--val-start", default="2025-01-01", help="検証開始日")
    parser.add_argument("--threshold", type=float, default=0.35, help="購入閾値")
    parser.add_argument("--top-n", type=int, default=3, help="上位何頭を対象")
    parser.add_argument("--save", action="store_true", help="結果をJSONで保存")
    args = parser.parse_args()

    results = run_backtest(
        args.input, args.model_dir, args.val_start,
        args.threshold, args.top_n,
    )

    print_backtest_report(results)

    if args.save and results:
        output_path = DATA_DIR / "backtest_report.json"
        # race_detailsは大きいので最新20件のみ保存
        save_results = results.copy()
        save_results["race_details"] = results["race_details"][-20:]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(save_results, f, ensure_ascii=False, indent=2, default=str)
        print(f"  [OK] 保存: {output_path}")


if __name__ == "__main__":
    main()
