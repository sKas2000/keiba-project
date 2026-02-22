"""
モニターデータの評価サマリー生成＋クリーンアップ
大量のモニターJSONから要点だけを抽出して保存し、元ファイルを削除
"""
import json
from pathlib import Path

from config.settings import RACES_DIR


def summarize_monitor_day(date_dir: Path) -> dict:
    """指定日のモニターデータからサマリーを生成

    Args:
        date_dir: data/races/monitor/YYYYMMDD/

    Returns:
        dict: 日次サマリー
    """
    races = {}

    # _predicted.json からレース情報 + 予測データを抽出
    for f in sorted(date_dir.glob("*_predicted.json")):
        race_key = f.stem.replace("_predicted", "")
        try:
            with open(f, "r", encoding="utf-8") as fp:
                data = json.load(fp)
        except Exception:
            continue

        race_info = data.get("race", {})
        horses = data.get("horses", [])
        ev = data.get("ev_results", {})

        # 上位3頭の要約
        sorted_horses = sorted(horses, key=lambda h: h.get("score", 0), reverse=True)
        top3 = []
        for h in sorted_horses[:3]:
            top3.append({
                "num": h.get("num"),
                "name": h.get("name"),
                "score": h.get("score"),
                "ml_top3_prob": h.get("ml_top3_prob"),
                "ml_win_prob": h.get("ml_win_prob"),
                "win_prob": round(h.get("win_prob", 0), 4),
                "odds_win": h.get("odds_win"),
            })

        # EV>=1.0の推奨買い目
        recommendations = []
        for bt, key in [("馬連", "quinella"), ("ワイド", "wide")]:
            for bet in ev.get(key, []):
                if bet.get("ev", 0) >= 1.0:
                    recommendations.append({
                        "type": bt,
                        "combo": bet.get("combo"),
                        "odds": bet.get("odds"),
                        "ev": round(bet.get("ev", 0), 3),
                        "prob": round(bet.get("prob", 0), 4),
                    })

        races[race_key] = {
            "venue": race_info.get("venue"),
            "race_number": race_info.get("race_number"),
            "name": race_info.get("name"),
            "date": race_info.get("date"),
            "surface": race_info.get("surface"),
            "distance": race_info.get("distance"),
            "grade": race_info.get("grade"),
            "num_horses": len(horses),
            "confidence": round(ev.get("confidence", 0), 3),
            "confidence_gap": round(ev.get("confidence_gap", 0), 4),
            "low_confidence": ev.get("low_confidence", False),
            "top3_predictions": top3,
            "recommendations": recommendations,
        }

    # _results.json からベット結果を統合
    total_invested = 0
    total_returned = 0
    bet_count = 0
    win_count = 0
    by_type = {}

    for f in sorted(date_dir.glob("*_results.json")):
        race_key = f.stem.replace("_results", "")
        try:
            with open(f, "r", encoding="utf-8") as fp:
                result_data = json.load(fp)
        except Exception:
            continue

        if race_key in races:
            races[race_key]["actual_top3"] = result_data.get("top3", [])
            races[race_key]["bet_results"] = result_data.get("bet_results", {})

        br = result_data.get("bet_results", {})
        for b in br.get("bets", []):
            bet_count += 1
            total_invested += b.get("invested", 0)
            total_returned += b.get("returned", 0)
            if b.get("won"):
                win_count += 1

            bt = b.get("type", "")
            st = by_type.setdefault(bt, {
                "count": 0, "wins": 0, "invested": 0, "returned": 0,
            })
            st["count"] += 1
            st["invested"] += b.get("invested", 0)
            st["returned"] += b.get("returned", 0)
            if b.get("won"):
                st["wins"] += 1

    roi = round(total_returned / total_invested * 100, 1) if total_invested > 0 else 0

    summary = {
        "date": date_dir.name,
        "total_races": len(races),
        "total_bets": bet_count,
        "total_wins": win_count,
        "total_invested": total_invested,
        "total_returned": total_returned,
        "roi": roi,
        "by_type": by_type,
        "races": races,
    }

    return summary


def run_monitor_summary(date: str = None, cleanup: bool = False, all_dates: bool = False):
    """モニターデータのサマリー生成＋クリーンアップ

    Args:
        date: 対象日（YYYYMMDD）。Noneで最新日
        cleanup: Trueで元ファイルを削除（サマリーと_results.jsonは残す）
        all_dates: Trueで全日付を処理
    """
    monitor_base = RACES_DIR / "monitor"
    if not monitor_base.exists():
        print("[ERROR] モニターデータが見つかりません")
        return

    if all_dates:
        date_dirs = sorted([d for d in monitor_base.iterdir() if d.is_dir()])
    elif date:
        target = monitor_base / date
        if not target.exists():
            print(f"[ERROR] ディレクトリが見つかりません: {target}")
            return
        date_dirs = [target]
    else:
        # 最新日
        date_dirs = sorted([d for d in monitor_base.iterdir() if d.is_dir()])
        if not date_dirs:
            print("[ERROR] モニターデータが見つかりません")
            return
        date_dirs = [date_dirs[-1]]

    for date_dir in date_dirs:
        print(f"\n[サマリー生成] {date_dir.name}")

        summary = summarize_monitor_day(date_dir)
        if not summary["races"]:
            print(f"  対象レースなし（スキップ）")
            continue

        # サマリー保存
        summary_path = date_dir / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        n_races = summary["total_races"]
        n_bets = summary["total_bets"]
        roi = summary["roi"]
        invested = summary["total_invested"]
        returned = summary["total_returned"]

        print(f"  レース数: {n_races}")
        print(f"  購入数: {n_bets}")
        if invested > 0:
            print(f"  投資: ¥{invested:,}  回収: ¥{returned:,.0f}  回収率: {roi}%")

        # 券種別
        for bt, st in summary.get("by_type", {}).items():
            bt_roi = round(st["returned"] / st["invested"] * 100, 1) if st["invested"] > 0 else 0
            mark = " *" if bt_roi >= 100 else ""
            print(f"    {bt}: {st['count']}件 的中{st['wins']} "
                  f"¥{st['invested']:,}→¥{st['returned']:,.0f} ({bt_roi}%{mark})")

        print(f"  -> {summary_path}")

        # クリーンアップ
        if cleanup:
            deleted = 0
            freed = 0
            for pattern in ["*_input.json", "*_enriched.json", "*_predicted.json"]:
                for f in date_dir.glob(pattern):
                    size = f.stat().st_size
                    f.unlink()
                    deleted += 1
                    freed += size

            freed_mb = freed / 1024 / 1024
            print(f"  [クリーンアップ] {deleted}ファイル削除 ({freed_mb:.1f}MB解放)")
            print(f"  残存: summary.json + *_results.json")
