"""
バックテスト結果のレポート表示・保存
"""
import json
from pathlib import Path

from config.settings import PROCESSED_DIR
from src.model.evaluator import BET_TYPES, BET_LABELS


def print_backtest_report(results: dict):
    """バックテスト結果を表示（全券種対応）"""
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

    for bt in BET_TYPES:
        b = results.get(f"bets_{bt}", {})
        label = BET_LABELS.get(bt, bt)
        if b.get("count", 0) > 0:
            roi = b["returned"] / b["invested"] * 100 if b["invested"] > 0 else 0
            mark = " ★" if roi >= 100 else ""
            print(f"\n  [{label}シミュレーション]")
            print(f"    購入回数: {b['count']}")
            print(f"    的中回数: {b['hits']} ({b['hits']/b['count']:.1%})")
            print(f"    投資額:   ¥{b['invested']:,}")
            print(f"    回収額:   ¥{b['returned']:,.0f}")
            print(f"    回収率:   {roi:.1f}%{mark}")
        else:
            print(f"\n  [{label}] 購入対象なし")

    monthly = results.get("monthly", {})
    if monthly:
        print(f"\n  [月別回収率]")
        header = f"    {'月':8s}"
        for bt in BET_TYPES:
            header += f" {BET_LABELS[bt]:>6s}"
        print(header)
        for month in sorted(monthly.keys()):
            m = monthly[month]
            row = f"    {month:8s}"
            for bt in BET_TYPES:
                b = m.get(bt, {})
                if b.get("count", 0) > 0 and b.get("invested", 0) > 0:
                    roi = b["returned"] / b["invested"] * 100
                    row += f" {roi:>5.0f}%"
                else:
                    row += f" {'---':>6s}"
            print(row)

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
    """EV閾値比較を表示（全券種対応）"""
    if not comparison:
        return
    print(f"\n  [EV閾値比較 — 全券種]")

    print(f"\n  ■ 単勝・複勝（EVフィルタ対象）")
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

    base = None
    for c in comparison:
        if c["ev_threshold"] == 0:
            base = c
            break
    if base is None and comparison:
        base = comparison[0]
    if base:
        multi_types = ["quinella", "wide", "exacta", "trio", "trifecta"]
        print(f"\n  ■ 組合せ券種（Top3全組合せ）")
        print(f"    {'券種':8s} {'購入点数':>8s} {'的中率':>8s} {'回収率':>10s}")
        print(f"    {'─' * 40}")
        for bt in multi_types:
            label = BET_LABELS[bt]
            bets = base.get(f"{bt}_bets", 0)
            roi = base.get(f"{bt}_roi", 0)
            hit_rate = base.get(f"{bt}_hit_rate", 0)
            if bets > 0:
                mark = " ★" if roi >= 100 else ""
                print(f"    {label:8s} {bets:>8d} {hit_rate:>7.1f}% {roi:>9.1f}%{mark}")
            else:
                print(f"    {label:8s} {'---':>8s} {'---':>8s} {'---':>10s}")


def save_backtest_report(results: dict, output_path: Path = None):
    """バックテスト結果をJSONで保存"""
    output_path = output_path or (PROCESSED_DIR / "backtest_report.json")
    save_results = results.copy()
    save_results["race_details"] = results.get("race_details", [])[-20:]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(save_results, f, ensure_ascii=False, indent=2, default=str)
    print(f"  [OK] 保存: {output_path}")
