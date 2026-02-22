"""
ライブ予測の事後検証
monitor/_predicted.json と実際のレース結果を照合してROIを計算
バックテスト期待値との乖離を可視化
"""
import asyncio
import json
import re
from pathlib import Path

import httpx

from config.settings import RACES_DIR, COURSE_CODES


async def _fetch_race_ids_for_date(client: httpx.AsyncClient,
                                   date_str: str) -> list:
    """netkeibaから日付のレースID一覧を取得"""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/131.0.0.0 Safari/537.36"
        ),
    }

    # race.netkeiba.com
    url = f"https://race.netkeiba.com/top/race_list_sub.html?kaisai_date={date_str}"
    try:
        r = await client.get(url, headers=headers)
        if r.status_code == 200:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(r.content, "lxml")
            race_ids = []
            for a in soup.find_all("a", href=re.compile(r"race_id=\d{12}")):
                m = re.search(r"race_id=(\d{12})", a["href"])
                if m and m.group(1) not in race_ids:
                    race_ids.append(m.group(1))
            if race_ids:
                return race_ids
    except Exception as e:
        print(f"  [WARN] race.netkeiba.com エラー: {e}")

    # フォールバック: db.netkeiba.com
    url = f"https://db.netkeiba.com/race/list/{date_str}/"
    try:
        r = await client.get(url, headers=headers)
        if r.status_code == 200:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(r.content, "lxml")
            race_ids = []
            for a in soup.find_all("a", href=re.compile(r"/race/\d{12}")):
                m = re.search(r"/race/(\d{12})", a["href"])
                if m and m.group(1) not in race_ids:
                    race_ids.append(m.group(1))
            return race_ids
    except Exception as e:
        print(f"  [WARN] db.netkeiba.com エラー: {e}")

    return []


def _race_id_to_venue_race(race_id: str) -> tuple:
    """race_id → (会場名, レース番号)"""
    course_id = race_id[4:6]
    race_num = int(race_id[10:12]) if race_id[10:12].isdigit() else 0
    venue = COURSE_CODES.get(course_id, "")
    return venue, race_num


async def _fetch_result_and_payoffs(client: httpx.AsyncClient,
                                    race_id: str) -> tuple:
    """1レースの結果と払戻を取得"""
    from src.scraping.race import parse_race_html, parse_return_html

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/131.0.0.0 Safari/537.36"
        ),
    }

    urls = [
        f"https://race.netkeiba.com/race/result.html?race_id={race_id}",
        f"https://db.netkeiba.com/race/{race_id}/",
    ]

    for url in urls:
        try:
            r = await client.get(url, headers=headers)
            if r.status_code != 200:
                continue
            results = parse_race_html(r.content, race_id)
            if not results:
                continue
            payoffs = parse_return_html(r.content, race_id)
            return results, payoffs
        except Exception:
            continue

    return [], []


def _match_bets_with_payoffs(ev_results: dict, payoffs: list) -> dict:
    """予測買い目と実際の払戻を照合"""
    payoff_map = {}
    for p in payoffs:
        bt = p["bet_type"]
        combo = p["combination"]
        nums = sorted(re.findall(r"\d+", combo))
        key = "-".join(nums)
        payoff_map.setdefault(bt, {})[key] = p["payout"]

    bets = []
    total_invested = 0
    total_returned = 0

    if ev_results.get("low_confidence"):
        return {
            "skipped": True, "reason": "見送り",
            "bets": [], "total_invested": 0, "total_returned": 0,
        }

    for bt_label, bt_key, payoff_type in [
        ("馬連", "quinella", "quinella"),
        ("ワイド", "wide", "wide"),
    ]:
        for bet in ev_results.get(bt_key, []):
            if bet["ev"] < 1.0:
                continue
            combo = bet.get("combo", "")
            parts = sorted(re.findall(r"\d+", str(combo)))
            norm = "-".join(parts)

            invested = 100
            total_invested += invested
            returned = payoff_map.get(payoff_type, {}).get(norm, 0)
            total_returned += returned

            bets.append({
                "type": bt_label, "combo": combo,
                "odds": bet["odds"], "ev": bet["ev"],
                "invested": invested, "returned": returned,
                "won": returned > 0,
            })

    return {
        "bets": bets,
        "total_invested": total_invested,
        "total_returned": total_returned,
    }


async def verify_monitor_day(date: str) -> dict:
    """モニター予測の事後検証

    Args:
        date: YYYYMMDD形式の日付

    Returns:
        検証結果dict
    """
    monitor_dir = RACES_DIR / "monitor" / date

    if not monitor_dir.exists():
        print(f"[ERROR] モニターデータなし: {monitor_dir}")
        return {}

    # 1. _predicted.json を読み込み
    predictions = {}
    for f in sorted(monitor_dir.glob("*_predicted.json")):
        race_key = f.stem.replace("_predicted", "")
        with open(f, "r", encoding="utf-8") as fp:
            predictions[race_key] = json.load(fp)

    if not predictions:
        print("[ERROR] 予測データなし")
        return {}

    print(f"\n[検証開始] {date} — {len(predictions)}レースの予測")

    # 2. netkeibaからレースID一覧を取得
    formatted_date = f"{date[:4]}-{date[4:6]}-{date[6:8]}"
    print(f"  netkeibaからレースID取得中...")

    async with httpx.AsyncClient(
        timeout=15, follow_redirects=True,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/131.0.0.0 Safari/537.36"
            ),
        },
    ) as client:
        race_ids = await _fetch_race_ids_for_date(client, date)

        if not race_ids:
            print("  [ERROR] レースIDを取得できません")
            return {}

        print(f"  {len(race_ids)}件のレースID取得")

        # race_id → (venue, race_num) マッピング
        id_map = {}
        for rid in race_ids:
            venue, race_num = _race_id_to_venue_race(rid)
            key = f"{venue}{race_num}R"
            id_map[key] = rid

        # 3. 各レースの結果を取得して照合
        total_invested = 0
        total_returned = 0
        total_bets = 0
        total_wins = 0
        by_type = {}
        race_results = {}
        skipped = 0

        for race_key, pred in predictions.items():
            race_id = id_map.get(race_key)
            if not race_id:
                print(f"  {race_key}: race_idマッチなし（スキップ）")
                skipped += 1
                continue

            ev_results = pred.get("ev_results", {})

            # _results.json が既にあればそれを使う
            existing_result = monitor_dir / f"{race_key}_results.json"
            if existing_result.exists():
                with open(existing_result, "r", encoding="utf-8") as fp:
                    cached = json.load(fp)
                bet_results = cached.get("bet_results", {})
            else:
                # netkeibaから結果取得
                results, payoffs = await _fetch_result_and_payoffs(client, race_id)
                if not results:
                    print(f"  {race_key}: 結果未確定")
                    skipped += 1
                    continue

                bet_results = _match_bets_with_payoffs(ev_results, payoffs)

                # 結果を保存
                top3 = sorted(results, key=lambda r: r.get("finish_position", 99))[:3]
                result_data = {
                    "race_id": race_id,
                    "top3": [{"num": r["horse_number"], "name": r["horse_name"],
                              "pos": r["finish_position"]} for r in top3],
                    "payoffs": payoffs,
                    "bet_results": bet_results,
                }
                with open(existing_result, "w", encoding="utf-8") as fp:
                    json.dump(result_data, fp, ensure_ascii=False, indent=2)

                await asyncio.sleep(1.0)

            # 集計
            if bet_results.get("skipped"):
                mark = "見送り"
            elif not bet_results.get("bets"):
                mark = "推奨なし"
            else:
                for b in bet_results["bets"]:
                    total_bets += 1
                    total_invested += b["invested"]
                    total_returned += b["returned"]
                    if b["won"]:
                        total_wins += 1

                    bt = b["type"]
                    st = by_type.setdefault(bt, {
                        "count": 0, "wins": 0, "invested": 0, "returned": 0,
                    })
                    st["count"] += 1
                    st["invested"] += b["invested"]
                    st["returned"] += b["returned"]
                    if b["won"]:
                        st["wins"] += 1

                won = sum(1 for b in bet_results["bets"] if b["won"])
                inv = bet_results["total_invested"]
                ret = bet_results["total_returned"]
                mark = f"{won}/{len(bet_results['bets'])}的中 ¥{inv}→¥{ret}"

            race_results[race_key] = bet_results
            print(f"  {race_key}: {mark}")

    # 4. サマリー
    roi = round(total_returned / total_invested * 100, 1) if total_invested else 0

    summary = {
        "date": date,
        "races_predicted": len(predictions),
        "races_verified": len(race_results),
        "races_skipped": skipped,
        "total_bets": total_bets,
        "total_wins": total_wins,
        "total_invested": total_invested,
        "total_returned": total_returned,
        "roi": roi,
        "by_type": by_type,
    }

    # 表示
    print(f"\n{'=' * 60}")
    print(f"  ライブ検証結果: {date}")
    print(f"{'=' * 60}")
    print(f"  予測レース: {len(predictions)}")
    print(f"  検証完了: {len(race_results)} (スキップ: {skipped})")
    print(f"  購入数: {total_bets} (的中: {total_wins})")
    if total_invested > 0:
        print(f"  投資: ¥{total_invested:,}")
        print(f"  回収: ¥{total_returned:,.0f}")
        print(f"  回収率: {roi}%")
        print()
        for bt, st in by_type.items():
            bt_roi = round(st["returned"] / st["invested"] * 100, 1) if st["invested"] else 0
            mark = " ★" if bt_roi >= 100 else ""
            print(f"    {bt}: {st['count']}件 的中{st['wins']} "
                  f"¥{st['invested']:,}→¥{st['returned']:,.0f} ({bt_roi}%{mark})")
    else:
        print(f"  購入なし（全レース見送りまたは推奨なし）")
    print(f"{'=' * 60}")

    # サマリー更新
    summary_path = monitor_dir / "summary.json"
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            existing_summary = json.load(f)
        existing_summary["verification"] = summary
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(existing_summary, f, ensure_ascii=False, indent=2)
        print(f"\n  → summary.json に検証結果を追記")

    return summary


def run_verify_monitor(date: str = None):
    """CLIエントリポイント"""
    if not date:
        # 最新日を検索
        monitor_base = RACES_DIR / "monitor"
        if not monitor_base.exists():
            print("[ERROR] モニターデータなし")
            return
        date_dirs = sorted([d.name for d in monitor_base.iterdir() if d.is_dir()])
        if not date_dirs:
            print("[ERROR] モニターデータなし")
            return
        date = date_dirs[-1]

    asyncio.run(verify_monitor_day(date))
