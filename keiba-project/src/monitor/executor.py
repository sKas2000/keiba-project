"""
Phase 3: スケジュール実行（予測 + Discord通知）
"""
import asyncio
import copy
from datetime import datetime, timedelta

from config.settings import MODEL_DIR
from src.data.storage import write_json
from src.notify import format_race_notification, send_notify

from .constants import RESULT_CHECK_DELAY
from .helpers import collect_recommendations


class ExecutorMixin:
    """スケジュール実行・予測"""

    async def _run_schedule(self):
        """発走時刻に合わせてバッチ実行 + 結果確認"""
        # 同一トリガー時刻のレースをグループ化
        time_groups = []
        current_trigger = None
        for trigger_dt, post_dt, race_key in self.schedule:
            if current_trigger != trigger_dt:
                time_groups.append((trigger_dt, post_dt, []))
                current_trigger = trigger_dt
            time_groups[-1][2].append(race_key)

        # 結果確認キュー
        result_queue = []

        for g_idx, (trigger_dt, post_dt, race_keys) in enumerate(time_groups):
            remaining = sum(len(g[2]) for g in time_groups[g_idx:])
            await self._wait_with_result_checks(
                trigger_dt, result_queue,
                next_label=f"{race_keys[0]}他 発走{post_dt.strftime('%H:%M')}",
                remaining=remaining,
            )

            for race_key in race_keys:
                race = self.races.get(race_key)
                if not race or not race.get("enriched"):
                    self.log(f"  {race_key}: enrichedデータなし（スキップ）")
                    continue

                self.log(f"\n{'=' * 50}")
                self.log(f"[実行] {race_key} (発走{post_dt.strftime('%H:%M')})")
                self.log(f"{'=' * 50}")
                await self._scrape_predict_notify(race_key)

            check_dt = post_dt + timedelta(minutes=RESULT_CHECK_DELAY)
            result_queue.append((check_dt, post_dt, race_keys))

        # 残りの結果確認
        result_queue.sort(key=lambda x: x[0])
        for check_dt, post_dt, race_keys in result_queue:
            wait = (check_dt - datetime.now()).total_seconds()
            if wait > 0:
                self.log(f"\n--- 結果確認待ち: {', '.join(race_keys)} "
                         f"(あと{int(wait // 60)}分{int(wait % 60)}秒) ---")
                await asyncio.sleep(wait)
            await self._check_results_batch(race_keys)

    async def _wait_with_result_checks(self, until_dt, result_queue,
                                       next_label="", remaining=0):
        """指定時刻まで待機しつつ、結果確認を実行"""
        while datetime.now() < until_dt:
            ready = [i for i, q in enumerate(result_queue)
                     if q[0] <= datetime.now()]
            if ready:
                for i in reversed(ready):
                    _, _, race_keys = result_queue.pop(i)
                    await self._check_results_batch(race_keys)
                continue

            next_events = [until_dt]
            next_events += [q[0] for q in result_queue]
            next_event = min(next_events)
            wait_secs = (next_event - datetime.now()).total_seconds()

            if wait_secs > 0:
                if next_event == until_dt and next_label:
                    self.log(f"\n--- 次: {next_label} "
                             f"(あと{int(wait_secs // 60)}分"
                             f"{int(wait_secs % 60)}秒待機, "
                             f"残り{remaining}レース) ---")
                    await asyncio.sleep(wait_secs)
                else:
                    await asyncio.sleep(min(wait_secs, 30))

    async def _scrape_predict_notify(self, race_key: str):
        """単一レースのオッズ再取得→ML予測→Discord通知"""
        from src.scraping.odds import OddsScraper
        from src.model.predictor import score_ml, calculate_ev

        race = self.races[race_key]
        race_info = race["race_info"]
        venue = race_info.get("venue", "")
        race_num = race_info.get("race_number", 0)
        m_idx = race["meeting_idx"]

        # 最新オッズ取得
        scraper = OddsScraper(headless=self.headless, debug=False)
        await scraper.start()
        try:
            meetings = await scraper.goto_odds_top()
            if m_idx < len(meetings):
                await scraper.select_meeting(meetings[m_idx]["element"])
                await scraper.select_race(race_num)

                horses_new = await scraper.parse_win_place()
                if horses_new:
                    quinella = await scraper.parse_triangle_odds("馬連")
                    wide = await scraper.parse_triangle_odds(
                        "ワイド", is_range=True
                    )
                    trio = await scraper.parse_trio()

                    enriched = race["enriched"]
                    for h_new in horses_new:
                        for h_old in enriched.get("horses", []):
                            if h_old.get("num") == h_new["num"]:
                                h_old["odds_win"] = h_new["odds_win"]
                                h_old["odds_place"] = h_new["odds_place"]
                                break
                    enriched["combo_odds"] = {
                        "quinella": quinella,
                        "wide": wide,
                        "trio": trio,
                    }
                    self.log(f"  最新オッズ取得完了")
                else:
                    self.log(f"  [WARN] オッズ取得失敗 — 既存データで予測")
        except Exception as e:
            self.log(f"  [WARN] オッズ取得エラー ({e}) — 既存データで予測")
        finally:
            await scraper.close()

        # ML予測
        data = copy.deepcopy(race["enriched"])
        result = score_ml(data, model_dir=MODEL_DIR)
        if result is None:
            self.log(f"  [WARN] MLモデル未学習")
            send_notify(
                f"\n{venue}{race_num}R: MLモデル未学習のため予測不可",
                self.webhook_url,
            )
            return

        ev_results = calculate_ev(result)
        race["predicted_ev"] = ev_results

        result["ev_results"] = ev_results
        path = self.monitor_dir / f"{race_key}_predicted.json"
        write_json(result, path)

        text = format_race_notification(race_info, ev_results)
        send_notify(text, self.webhook_url)

        recs = collect_recommendations(ev_results)
        if recs:
            self.log(f"  {len(recs)} 件の推奨買い目を通知")
        elif ev_results.get("low_confidence"):
            self.log(f"  確信度不足 → 見送り推奨")
        else:
            self.log(f"  推奨買い目なし")
