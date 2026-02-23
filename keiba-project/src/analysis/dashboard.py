"""
ライブ検証ダッシュボード
モニター予測の月次ROI自動集計 + モデルバージョン追跡
"""
import json
from collections import defaultdict
from pathlib import Path

from config.settings import DATA_DIR, MODEL_DIR


MONITOR_DIR = DATA_DIR / "races" / "monitor"

BET_TYPE_LABELS = {
    "馬連": "quinella",
    "ワイド": "wide",
    "単勝": "win",
    "複勝": "place",
    "3連複": "trio",
}


def _load_all_summaries(monitor_dir: Path = None) -> list:
    """全日付のsummary.jsonを読み込み"""
    monitor_dir = monitor_dir or MONITOR_DIR
    summaries = []
    if not monitor_dir.exists():
        return summaries

    for date_dir in sorted(monitor_dir.iterdir()):
        if not date_dir.is_dir():
            continue
        summary_path = date_dir / "summary.json"
        if not summary_path.exists():
            continue
        with open(summary_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["_date_dir"] = date_dir.name
        summaries.append(data)
    return summaries


def aggregate_monthly(monitor_dir: Path = None) -> dict:
    """月次ROI集計

    Returns:
        {
            "months": {
                "2025-01": {
                    "dates": [...],
                    "races": int,
                    "by_type": {
                        "馬連": {"count": int, "wins": int, "invested": int, "returned": int},
                        ...
                    },
                    "total_invested": int,
                    "total_returned": int,
                    "roi": float,
                },
                ...
            },
            "cumulative": {
                "total_invested": int,
                "total_returned": int,
                "roi": float,
                "total_bets": int,
                "total_wins": int,
            },
        }
    """
    summaries = _load_all_summaries(monitor_dir)

    months = defaultdict(lambda: {
        "dates": [],
        "races": 0,
        "by_type": defaultdict(lambda: {"count": 0, "wins": 0, "invested": 0, "returned": 0}),
        "total_invested": 0,
        "total_returned": 0,
    })

    for summary in summaries:
        date_str = summary.get("date", summary.get("_date_dir", ""))
        if len(date_str) == 8:
            month_key = f"{date_str[:4]}-{date_str[4:6]}"
        elif len(date_str) >= 7:
            month_key = date_str[:7]
        else:
            continue

        m = months[month_key]
        m["dates"].append(date_str)
        m["races"] += summary.get("total_races", 0)

        by_type = summary.get("by_type", {})
        for bt_label, bt_data in by_type.items():
            if isinstance(bt_data, dict):
                for k in ["count", "wins", "invested", "returned"]:
                    m["by_type"][bt_label][k] += bt_data.get(k, 0)
                m["total_invested"] += bt_data.get("invested", 0)
                m["total_returned"] += bt_data.get("returned", 0)

    # ROI計算
    for month_key, m in months.items():
        m["roi"] = (m["total_returned"] / m["total_invested"] * 100) if m["total_invested"] > 0 else 0

    # 累計
    cum_inv = sum(m["total_invested"] for m in months.values())
    cum_ret = sum(m["total_returned"] for m in months.values())
    cum_bets = sum(
        sum(bt["count"] for bt in m["by_type"].values())
        for m in months.values()
    )
    cum_wins = sum(
        sum(bt["wins"] for bt in m["by_type"].values())
        for m in months.values()
    )

    return {
        "months": dict(sorted(months.items())),
        "cumulative": {
            "total_invested": cum_inv,
            "total_returned": cum_ret,
            "roi": (cum_ret / cum_inv * 100) if cum_inv > 0 else 0,
            "total_bets": cum_bets,
            "total_wins": cum_wins,
        },
    }


def list_model_versions(model_dir: Path = None) -> list:
    """モデルバージョン一覧をメタデータ付きで返す"""
    model_dir = model_dir or MODEL_DIR
    versions = []

    # active version
    active_path = model_dir / "active_version.txt"
    active_version = ""
    if active_path.exists():
        active_version = active_path.read_text(encoding="utf-8").strip()

    for d in sorted(model_dir.iterdir()):
        if not d.is_dir() or not d.name.startswith("v"):
            continue
        meta_path = d / "version_meta.json"
        if not meta_path.exists():
            continue
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        meta["is_active"] = d.name == active_version
        versions.append(meta)

    return versions


def print_dashboard(monitor_dir: Path = None, model_dir: Path = None):
    """ダッシュボード表示"""
    result = aggregate_monthly(monitor_dir)
    months = result["months"]
    cumulative = result["cumulative"]

    print(f"\n{'═' * 70}")
    print(f"  ライブ検証ダッシュボード")
    print(f"{'═' * 70}")

    if not months:
        print(f"\n  データがありません。")
        print(f"  python main.py monitor で予測を開始し、")
        print(f"  python main.py verify-monitor で検証してください。")
        print(f"{'═' * 70}")
        return result

    # 月次ROI一覧
    print(f"\n  [月次ROI]")
    print(f"  {'月':10s} {'開催日数':>8s} {'レース':>6s} {'投資':>10s} {'回収':>10s} {'ROI':>8s}")
    print(f"  {'─' * 60}")

    for month_key, m in sorted(months.items()):
        roi_mark = "*" if m["roi"] >= 100 else ""
        print(f"  {month_key:10s} {len(m['dates']):>8d} {m['races']:>6d} "
              f"{m['total_invested']:>10,d} {m['total_returned']:>10,.0f} "
              f"{m['roi']:>6.1f}%{roi_mark}")

    print(f"  {'─' * 60}")
    cum_mark = "*" if cumulative["roi"] >= 100 else ""
    print(f"  {'累計':10s} {'':>8s} {'':>6s} "
          f"{cumulative['total_invested']:>10,d} {cumulative['total_returned']:>10,.0f} "
          f"{cumulative['roi']:>6.1f}%{cum_mark}")
    print(f"  (* = ROI >= 100%)")

    # 券種別ROI
    type_totals = defaultdict(lambda: {"count": 0, "wins": 0, "invested": 0, "returned": 0})
    for m in months.values():
        for bt_label, bt_data in m["by_type"].items():
            for k in ["count", "wins", "invested", "returned"]:
                type_totals[bt_label][k] += bt_data.get(k, 0)

    if type_totals:
        print(f"\n  [券種別ROI（累計）]")
        print(f"  {'券種':8s} {'賭数':>6s} {'的中':>6s} {'的中率':>8s} {'投資':>10s} {'回収':>10s} {'ROI':>8s}")
        print(f"  {'─' * 62}")
        for bt_label, bt_data in sorted(type_totals.items()):
            if bt_data["count"] == 0:
                continue
            hit_rate = bt_data["wins"] / bt_data["count"] * 100
            roi = (bt_data["returned"] / bt_data["invested"] * 100) if bt_data["invested"] > 0 else 0
            roi_mark = "*" if roi >= 100 else ""
            print(f"  {bt_label:8s} {bt_data['count']:>6d} {bt_data['wins']:>6d} "
                  f"{hit_rate:>6.1f}% {bt_data['invested']:>10,d} {bt_data['returned']:>10,.0f} "
                  f"{roi:>6.1f}%{roi_mark}")

    # モデルバージョン
    versions = list_model_versions(model_dir)
    if versions:
        print(f"\n  [モデルバージョン]")
        print(f"  {'バージョン':12s} {'作成日':18s} {'Binary AUC':>10s} {'Win AUC':>10s} {'学習行数':>10s} {'状態':>6s}")
        print(f"  {'─' * 70}")
        for v in versions:
            status = "★ 稼働" if v.get("is_active") else ""
            created = v.get("created_at", "")[:19]
            print(f"  {v['version']:12s} {created:18s} "
                  f"{v.get('binary_auc', 0):>10.4f} {v.get('win_auc', 0):>10.4f} "
                  f"{v.get('train_rows', 0):>10,d} {status:>6s}")

    print(f"\n{'═' * 70}")
    return result


def run_dashboard(monitor_dir: Path = None, model_dir: Path = None):
    """ダッシュボードCLIエントリポイント"""
    return print_dashboard(monitor_dir, model_dir)
