"""
バックテスト・評価モジュール
学習済みモデルの予測結果で過去レースの回収率をシミュレーション
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

from config.settings import FEATURE_COLUMNS, MODEL_DIR, PROCESSED_DIR


# ============================================================
# バックテスト
# ============================================================

def run_backtest(input_path: str = None, model_dir: Path = None,
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
    input_path = input_path or str(PROCESSED_DIR / "features.csv")
    model_dir = model_dir or MODEL_DIR

    df = pd.read_csv(input_path, dtype={"race_id": str, "horse_id": str})
    df["race_date"] = pd.to_datetime(df["race_date"])

    val_df = df[df["race_date"] >= pd.Timestamp(val_start)].copy()
    if len(val_df) == 0:
        print("[ERROR] 検証期間のデータがありません")
        return {}

    binary_model_path = model_dir / "binary_model.txt"
    if not binary_model_path.exists():
        print(f"[ERROR] モデルが見つかりません: {binary_model_path}")
        return {}

    model = lgb.Booster(model_file=str(binary_model_path))

    available = [c for c in FEATURE_COLUMNS if c in val_df.columns]
    X_val = val_df[available].values.astype(np.float32)
    val_df["pred_prob"] = model.predict(X_val)

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

        # 単勝シミュレーション
        if top1_pred["pred_prob"] >= bet_threshold and "win_odds" in race_group.columns:
            results["bets_win"]["count"] += 1
            results["bets_win"]["invested"] += 100
            if top1_pred.get("finish_position", 99) == 1:
                payout = top1_pred.get("win_odds", 0) * 100
                results["bets_win"]["returned"] += payout
                results["bets_win"]["hits"] += 1

        # 複勝シミュレーション
        for _, horse in race_group.head(top_n).iterrows():
            if horse["pred_prob"] >= bet_threshold * 0.8 and "win_odds" in race_group.columns:
                results["bets_place"]["count"] += 1
                results["bets_place"]["invested"] += 100
                if horse.get("finish_position", 99) <= 3:
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


# ============================================================
# レポート表示
# ============================================================

def print_backtest_report(results: dict):
    """バックテスト結果を表示"""
    if not results:
        return

    races = results["races"]
    print(f"\n  対象レース数: {races}")

    pa = results["prediction_accuracy"]
    total = pa["total"]
    if total > 0:
        print(f"\n  [予測精度]")
        print(f"    1着的中率:   {pa['top1_hit']}/{total} = {pa['top1_hit']/total:.1%}")
        print(f"    3着内的中数: {pa['top3_hit']}/{total*3} = {pa['top3_hit']/(total*3):.1%}")

    bw = results["bets_win"]
    if bw["count"] > 0:
        roi_win = bw["returned"] / bw["invested"] * 100 if bw["invested"] > 0 else 0
        print(f"\n  [単勝シミュレーション]")
        print(f"    購入回数: {bw['count']}")
        print(f"    的中回数: {bw['hits']} ({bw['hits']/bw['count']:.1%})")
        print(f"    投資額:   ¥{bw['invested']:,}")
        print(f"    回収額:   ¥{bw['returned']:,.0f}")
        print(f"    回収率:   {roi_win:.1f}%")

    bp = results["bets_place"]
    if bp["count"] > 0:
        roi_place = bp["returned"] / bp["invested"] * 100 if bp["invested"] > 0 else 0
        print(f"\n  [複勝シミュレーション]")
        print(f"    購入回数: {bp['count']}")
        print(f"    的中回数: {bp['hits']} ({bp['hits']/bp['count']:.1%})")
        print(f"    投資額:   ¥{bp['invested']:,}")
        print(f"    回収額:   ¥{bp['returned']:,.0f}")
        print(f"    回収率:   {roi_place:.1f}%")

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


def save_backtest_report(results: dict, output_path: Path = None):
    """バックテスト結果をJSONで保存"""
    output_path = output_path or (PROCESSED_DIR / "backtest_report.json")
    save_results = results.copy()
    save_results["race_details"] = results.get("race_details", [])[-20:]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(save_results, f, ensure_ascii=False, indent=2, default=str)
    print(f"  [OK] 保存: {output_path}")
