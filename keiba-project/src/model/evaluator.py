"""
バックテスト・評価モジュール
学習済みモデルの予測結果で過去レースの回収率をシミュレーション
EVフィルタリング + 実払い戻しデータ対応
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

from config.settings import FEATURE_COLUMNS, MODEL_DIR, PROCESSED_DIR, RAW_DIR


# ============================================================
# 払い戻しデータ読み込み
# ============================================================

def _load_returns(returns_path: Path) -> dict:
    """returns.csv を {race_id -> {bet_type -> [{combination, payout}]}} に変換"""
    if not returns_path.exists():
        return {}
    df = pd.read_csv(returns_path, dtype={"race_id": str})
    lookup = {}
    for _, row in df.iterrows():
        rid = row["race_id"]
        if rid not in lookup:
            lookup[rid] = {}
        bt = row["bet_type"]
        if bt not in lookup[rid]:
            lookup[rid][bt] = []
        lookup[rid][bt].append({
            "combination": str(row["combination"]),
            "payout": float(row["payout"]),
        })
    return lookup


def _get_place_payout(returns: dict, race_id: str, horse_number: int) -> float:
    """実際の複勝払い戻しを取得（100円あたり）"""
    race_returns = returns.get(race_id, {})
    place_entries = race_returns.get("place", [])
    horse_str = str(int(horse_number))
    for entry in place_entries:
        if entry["combination"] == horse_str:
            return entry["payout"]
    return 0.0


# ============================================================
# バックテスト
# ============================================================

def run_backtest(input_path: str = None, model_dir: Path = None,
                 returns_path: str = None,
                 val_start: str = "2025-01-01",
                 val_end: str = None,
                 ev_threshold: float = 0.0,
                 bet_threshold: float = 0.0,
                 top_n: int = 3,
                 temperature: float = 1.0) -> dict:
    """
    EVフィルタ付きバックテスト

    Args:
        ev_threshold: EV(予測勝率×オッズ)がこの値以上で購入（0=フィルタなし）
        bet_threshold: 予測確率がこの値以上で購入対象
        returns_path: returns.csv パス（複勝の実オッズ使用）
        val_end: 検証終了日（指定しない場合はデータ末尾まで）
        temperature: ソフトマックス温度パラメータ（logitスケール、低いほど鋭い分布）
    """
    input_path = input_path or str(PROCESSED_DIR / "features.csv")
    model_dir = model_dir or MODEL_DIR
    returns_path = Path(returns_path) if returns_path else RAW_DIR / "returns.csv"

    df = pd.read_csv(input_path, dtype={"race_id": str, "horse_id": str})
    df["race_date"] = pd.to_datetime(df["race_date"])

    val_df = df[df["race_date"] >= pd.Timestamp(val_start)].copy()
    if val_end:
        val_df = val_df[val_df["race_date"] < pd.Timestamp(val_end)].copy()
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

    # 払い戻しデータ読み込み
    returns = _load_returns(returns_path)

    results = {
        "val_start": val_start,
        "ev_threshold": ev_threshold,
        "races": 0,
        "bets_win": {"count": 0, "invested": 0, "returned": 0, "hits": 0},
        "bets_place": {"count": 0, "invested": 0, "returned": 0, "hits": 0},
        "prediction_accuracy": {"top1_hit": 0, "top3_hit": 0, "total": 0},
        "race_details": [],
        "monthly": {},
    }

    for race_id, race_group in val_df.groupby("race_id"):
        if len(race_group) < 4:
            continue

        race_group = race_group.sort_values("pred_prob", ascending=False)
        results["races"] += 1

        # 月別集計用
        month_key = str(race_group.iloc[0]["race_date"])[:7]
        if month_key not in results["monthly"]:
            results["monthly"][month_key] = {
                "win": {"count": 0, "invested": 0, "returned": 0, "hits": 0},
                "place": {"count": 0, "invested": 0, "returned": 0, "hits": 0},
            }

        # レース内のsoftmax勝率を計算（logit変換: sigmoid逆変換で元のスコアに復元）
        pred_probs = np.clip(race_group["pred_prob"].values, 1e-6, 1 - 1e-6)
        logits = np.log(pred_probs / (1 - pred_probs))
        scores = logits / temperature
        exp_s = np.exp(scores - scores.max())
        win_probs = exp_s / exp_s.sum()
        race_group = race_group.copy()
        race_group["win_prob"] = win_probs

        # 予測精度
        top1_pred = race_group.iloc[0]
        results["prediction_accuracy"]["total"] += 1
        if top1_pred.get("finish_position", 99) == 1:
            results["prediction_accuracy"]["top1_hit"] += 1

        top3_pred = race_group.head(3)
        top3_actual = (top3_pred["finish_position"] <= 3).sum()
        results["prediction_accuracy"]["top3_hit"] += top3_actual

        # --- 単勝シミュレーション ---
        for _, horse in race_group.head(top_n).iterrows():
            odds = horse.get("win_odds", 0)
            if odds <= 0:
                continue
            ev_win = horse["win_prob"] * odds
            if ev_threshold > 0 and ev_win < ev_threshold:
                continue
            if horse["pred_prob"] < bet_threshold:
                continue

            results["bets_win"]["count"] += 1
            results["bets_win"]["invested"] += 100
            results["monthly"][month_key]["win"]["count"] += 1
            results["monthly"][month_key]["win"]["invested"] += 100

            if horse.get("finish_position", 99) == 1:
                payout = odds * 100
                results["bets_win"]["returned"] += payout
                results["bets_win"]["hits"] += 1
                results["monthly"][month_key]["win"]["returned"] += payout
                results["monthly"][month_key]["win"]["hits"] += 1

        # --- 複勝シミュレーション ---
        for _, horse in race_group.head(top_n).iterrows():
            pred_place = horse["pred_prob"]
            odds = horse.get("win_odds", 0)
            if odds <= 0:
                continue
            est_place_odds = max(odds * 0.3, 1.1)
            ev_place = pred_place * est_place_odds
            if ev_threshold > 0 and ev_place < ev_threshold:
                continue
            if pred_place < bet_threshold:
                continue

            results["bets_place"]["count"] += 1
            results["bets_place"]["invested"] += 100
            results["monthly"][month_key]["place"]["count"] += 1
            results["monthly"][month_key]["place"]["invested"] += 100

            if horse.get("finish_position", 99) <= 3:
                actual_payout = _get_place_payout(
                    returns, race_id, horse.get("horse_number", 0))
                if actual_payout > 0:
                    payout = actual_payout
                else:
                    payout = est_place_odds * 100
                results["bets_place"]["returned"] += payout
                results["bets_place"]["hits"] += 1
                results["monthly"][month_key]["place"]["returned"] += payout
                results["monthly"][month_key]["place"]["hits"] += 1

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
                "win_prob": round(float(horse["win_prob"]), 3),
                "actual_finish": int(horse.get("finish_position", 0)),
                "odds": float(horse.get("win_odds", 0)),
                "ev_win": round(float(horse["win_prob"] * horse.get("win_odds", 0)), 2),
            })
        results["race_details"].append(detail)

    return results


# ============================================================
# EV閾値比較
# ============================================================

def compare_ev_thresholds(input_path: str = None, model_dir: Path = None,
                          returns_path: str = None,
                          val_start: str = "2025-01-01",
                          val_end: str = None,
                          thresholds: list = None,
                          temperature: float = 1.0) -> list:
    """複数のEV閾値でバックテストを実行し比較"""
    thresholds = thresholds or [0.0, 0.8, 1.0, 1.2, 1.5, 2.0]
    comparison = []

    for t in thresholds:
        res = run_backtest(
            input_path=input_path, model_dir=model_dir,
            returns_path=returns_path, val_start=val_start,
            val_end=val_end,
            ev_threshold=t, top_n=3, temperature=temperature,
        )
        if not res:
            continue

        bw = res["bets_win"]
        bp = res["bets_place"]
        roi_win = (bw["returned"] / bw["invested"] * 100) if bw["invested"] > 0 else 0
        roi_place = (bp["returned"] / bp["invested"] * 100) if bp["invested"] > 0 else 0

        comparison.append({
            "ev_threshold": t,
            "win_bets": bw["count"],
            "win_hits": bw["hits"],
            "win_hit_rate": round(bw["hits"] / bw["count"] * 100, 1) if bw["count"] > 0 else 0,
            "win_roi": round(roi_win, 1),
            "place_bets": bp["count"],
            "place_hits": bp["hits"],
            "place_hit_rate": round(bp["hits"] / bp["count"] * 100, 1) if bp["count"] > 0 else 0,
            "place_roi": round(roi_place, 1),
        })

    return comparison


def optimize_temperature(input_path: str = None, model_dir: Path = None,
                         returns_path: str = None,
                         val_start: str = "2025-01-01",
                         val_end: str = None,
                         ev_threshold: float = 1.0) -> dict:
    """ソフトマックス温度パラメータの最適化（グリッドサーチ、logitスケール）"""
    temperatures = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0]
    best = {"temperature": 1.0, "win_roi": 0, "place_roi": 0}

    period = f"{val_start}〜{val_end}" if val_end else f"{val_start}〜"
    print(f"\n  [温度パラメータ最適化] EV閾値={ev_threshold} 期間={period}")
    print(f"    {'温度':>6s} {'単勝回数':>8s} {'単勝的中率':>10s} {'単勝回収率':>10s} "
          f"{'複勝回数':>8s} {'複勝的中率':>10s} {'複勝回収率':>10s}")
    print(f"    {'─' * 70}")

    for temp in temperatures:
        res = run_backtest(
            input_path=input_path, model_dir=model_dir,
            returns_path=returns_path, val_start=val_start,
            val_end=val_end,
            ev_threshold=ev_threshold, top_n=3, temperature=temp,
        )
        if not res:
            continue
        bw = res["bets_win"]
        bp = res["bets_place"]
        roi_win = (bw["returned"] / bw["invested"] * 100) if bw["invested"] > 0 else 0
        roi_place = (bp["returned"] / bp["invested"] * 100) if bp["invested"] > 0 else 0
        hit_w = bw["hits"] / bw["count"] * 100 if bw["count"] > 0 else 0
        hit_p = bp["hits"] / bp["count"] * 100 if bp["count"] > 0 else 0

        mark = ""
        if roi_win > best["win_roi"]:
            best = {"temperature": temp, "win_roi": roi_win, "place_roi": roi_place,
                    "win_bets": bw["count"], "place_bets": bp["count"]}
            mark = " ★"

        print(f"    {temp:>6.1f} {bw['count']:>8d} {hit_w:>9.1f}% {roi_win:>9.1f}%{mark}"
              f" {bp['count']:>8d} {hit_p:>9.1f}% {roi_place:>9.1f}%")

    print(f"\n  最適温度: {best['temperature']} (単勝回収率: {best['win_roi']:.1f}%)")
    return best


# ============================================================
# レポート表示
# ============================================================

def print_backtest_report(results: dict):
    """バックテスト結果を表示"""
    if not results:
        return

    races = results["races"]
    ev_t = results.get("ev_threshold", 0)
    print(f"\n  対象レース数: {races}")
    if ev_t > 0:
        print(f"  EVフィルタ: >= {ev_t}")

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
    else:
        print(f"\n  [単勝] 購入対象なし")

    bp = results["bets_place"]
    if bp["count"] > 0:
        roi_place = bp["returned"] / bp["invested"] * 100 if bp["invested"] > 0 else 0
        print(f"\n  [複勝シミュレーション]")
        print(f"    購入回数: {bp['count']}")
        print(f"    的中回数: {bp['hits']} ({bp['hits']/bp['count']:.1%})")
        print(f"    投資額:   ¥{bp['invested']:,}")
        print(f"    回収額:   ¥{bp['returned']:,.0f}")
        print(f"    回収率:   {roi_place:.1f}%")
    else:
        print(f"\n  [複勝] 購入対象なし")

    # 月別
    monthly = results.get("monthly", {})
    if monthly:
        print(f"\n  [月別回収率]")
        print(f"    {'月':8s} {'単勝':>10s} {'複勝':>10s}")
        for month in sorted(monthly.keys()):
            m = monthly[month]
            w_roi = (m["win"]["returned"] / m["win"]["invested"] * 100) if m["win"]["invested"] > 0 else 0
            p_roi = (m["place"]["returned"] / m["place"]["invested"] * 100) if m["place"]["invested"] > 0 else 0
            w_str = f"{w_roi:.0f}%" if m["win"]["count"] > 0 else "-"
            p_str = f"{p_roi:.0f}%" if m["place"]["count"] > 0 else "-"
            print(f"    {month:8s} {w_str:>10s} {p_str:>10s}")

    details = results.get("race_details", [])
    if details:
        print(f"\n  [サンプルレース（最新5件）]")
        for detail in details[-5:]:
            print(f"\n    {detail['race_id']} ({detail['date'][:10]})")
            for pred in detail["predictions"][:3]:
                hit = "◎" if pred["actual_finish"] <= 3 else "×"
                print(f"      {hit} {pred['horse']:12s} "
                      f"top3={pred['pred_prob']:.1%} "
                      f"win={pred['win_prob']:.1%} "
                      f"EV={pred['ev_win']:.2f} "
                      f"実際{pred['actual_finish']}着 "
                      f"オッズ{pred['odds']:.1f}")


def print_ev_comparison(comparison: list):
    """EV閾値比較を表示"""
    if not comparison:
        return
    print(f"\n  [EV閾値比較]")
    print(f"    {'EV閾値':>7s} {'単勝回数':>8s} {'単勝的中':>8s} {'単勝回収率':>10s} "
          f"{'複勝回数':>8s} {'複勝的中':>8s} {'複勝回収率':>10s}")
    print(f"    {'─' * 65}")
    for c in comparison:
        t_str = "なし" if c["ev_threshold"] == 0 else f">={c['ev_threshold']}"
        w_roi_str = f"{c['win_roi']:.1f}%"
        p_roi_str = f"{c['place_roi']:.1f}%"
        mark_w = " ★" if c["win_roi"] >= 100 else ""
        mark_p = " ★" if c["place_roi"] >= 100 else ""
        print(f"    {t_str:>7s} {c['win_bets']:>8d} {c['win_hit_rate']:>7.1f}% "
              f"{w_roi_str:>10s}{mark_w}  "
              f"{c['place_bets']:>7d} {c['place_hit_rate']:>7.1f}% "
              f"{p_roi_str:>10s}{mark_p}")


def save_backtest_report(results: dict, output_path: Path = None):
    """バックテスト結果をJSONで保存"""
    output_path = output_path or (PROCESSED_DIR / "backtest_report.json")
    save_results = results.copy()
    save_results["race_details"] = results.get("race_details", [])[-20:]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(save_results, f, ensure_ascii=False, indent=2, default=str)
    print(f"  [OK] 保存: {output_path}")
