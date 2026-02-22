"""
結果確認 + 回収率計算 + 日次サマリー
"""
import re
from datetime import datetime

from src.data.storage import write_json
from src.notify import (
    send_notify, format_result_notification, format_daily_summary,
)

from .helpers import construct_race_id


class ResultsMixin:
    """結果確認・回収率計算"""

    async def _check_results_batch(self, race_keys: list):
        """レースグループの結果を確認"""
        self.log(f"\n{'=' * 50}")
        self.log(f"[結果確認] {', '.join(race_keys)}")
        self.log(f"{'=' * 50}")

        for race_key in race_keys:
            race = self.races.get(race_key)
            if not race:
                continue

            ev = race.get("predicted_ev")
            if not ev:
                self.log(f"  {race_key}: 予測なし（スキップ）")
                continue

            race_id = construct_race_id(race, self.meeting_info)
            if not race_id:
                self.log(f"  {race_key}: race_id構築失敗")
                continue

            results, payoffs = await self._fetch_race_results(race_id)
            if not results:
                self.log(f"  {race_key}: 結果未確定 ({race_id})")
                continue

            top3 = sorted(
                results, key=lambda r: r.get("finish_position", 99)
            )[:3]

            bet_results = _calculate_bet_results(ev, top3, payoffs)
            race["bet_results"] = bet_results

            result_path = self.monitor_dir / f"{race_key}_results.json"
            write_json({
                "race_id": race_id,
                "top3": [{"num": r["horse_number"], "name": r["horse_name"],
                          "pos": r["finish_position"]} for r in top3],
                "payoffs": payoffs,
                "bet_results": bet_results,
            }, result_path)

            text = format_result_notification(race_key, race, bet_results)
            send_notify(text, self.webhook_url)

            if bet_results.get("bets"):
                won = sum(1 for b in bet_results["bets"] if b["won"])
                total = len(bet_results["bets"])
                self.log(f"  {race_key}: {won}/{total}的中 "
                         f"(投資{bet_results['total_invested']}円 "
                         f"→ 回収{bet_results['total_returned']}円)")
            else:
                self.log(f"  {race_key}: 推奨買い目なし or 見送り")

    async def _fetch_race_results(self, race_id: str):
        """netkeibaからレース結果と払戻金を取得"""
        import httpx
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
                async with httpx.AsyncClient(
                    timeout=15, headers=headers, follow_redirects=True,
                ) as client:
                    r = await client.get(url)
                    if r.status_code != 200:
                        continue
                    results = parse_race_html(r.content, race_id)
                    if not results:
                        continue
                    payoffs = parse_return_html(r.content, race_id)
                    return results, payoffs
            except Exception as e:
                self.log(f"    結果取得エラー ({url}): {e}")
                continue

        return [], []

    def _send_daily_summary(self):
        """日次の回収率サマリを送信"""
        total_invested = 0
        total_returned = 0
        bet_count = 0
        win_count = 0
        by_type = {}

        for race_key, race in self.races.items():
            br = race.get("bet_results")
            if not br or br.get("skipped"):
                continue
            for b in br.get("bets", []):
                bt = b["type"]
                bet_count += 1
                total_invested += b["invested"]
                total_returned += b["returned"]
                if b["won"]:
                    win_count += 1

                st = by_type.setdefault(bt, {
                    "count": 0, "wins": 0,
                    "invested": 0, "returned": 0,
                })
                st["count"] += 1
                st["invested"] += b["invested"]
                st["returned"] += b["returned"]
                if b["won"]:
                    st["wins"] += 1

        stats = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "bet_count": bet_count,
            "win_count": win_count,
            "total_invested": total_invested,
            "total_returned": total_returned,
            "by_type": by_type,
        }

        text = format_daily_summary(stats)
        send_notify(text, self.webhook_url)
        self.log(f"\n{text}")


def _calculate_bet_results(ev: dict, top3: list, payoffs: list) -> dict:
    """予測と実際の結果を比較してベット結果を計算"""
    if not ev or ev.get("low_confidence"):
        return {
            "skipped": True, "reason": "見送り", "bets": [],
            "total_invested": 0, "total_returned": 0,
            "top3": [{"num": r["horse_number"], "name": r["horse_name"]}
                     for r in top3],
        }

    # payoffをdict化
    payoff_map = {}
    for p in payoffs:
        bt = p["bet_type"]
        combo = p["combination"]
        nums = sorted(re.findall(r"\d+", combo))
        key = "-".join(nums)
        payoff_map.setdefault(bt, {})[key] = p["payout"]

    results = {
        "bets": [],
        "total_invested": 0,
        "total_returned": 0,
        "top3": [{"num": r["horse_number"], "name": r["horse_name"]}
                 for r in top3],
    }

    for bt_label, bt_key, payoff_type in [
        ("馬連", "quinella", "quinella"),
        ("ワイド", "wide", "wide"),
    ]:
        for bet in ev.get(bt_key, []):
            if bet["ev"] < 1.0:
                continue
            combo = bet.get("combo", "")
            parts = sorted(re.findall(r"\d+", str(combo)))
            norm = "-".join(parts)

            invested = 100
            results["total_invested"] += invested

            returned = payoff_map.get(payoff_type, {}).get(norm, 0)
            results["total_returned"] += returned
            results["bets"].append({
                "type": bt_label, "combo": combo,
                "odds": bet["odds"], "ev": bet["ev"],
                "invested": invested, "returned": returned,
                "won": returned > 0,
            })

    return results
