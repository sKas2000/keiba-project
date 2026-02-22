"""
Phase 1: レース一覧取得 + スケジュール作成
"""
from datetime import datetime, timedelta

from .constants import DEFAULT_POST_TIMES
from .helpers import extract_venue


class ScannerMixin:
    """レーススキャン・スケジュール構築"""

    async def _scan_race_schedule(self):
        """全レースの情報と発走時刻を取得（オッズはまだ取らない）"""
        from src.scraping.odds import OddsScraper

        scraper = OddsScraper(headless=self.headless, debug=False)
        await scraper.start()
        try:
            meetings = await scraper.goto_odds_top()
            if not meetings:
                self.log("[WARN] 開催情報が見つかりません")
                return

            # JRAサイトは土日2日分を同時表示するため、今日の開催のみ処理
            all_meetings = []
            for i, m in enumerate(meetings):
                venue = extract_venue(m["text"])
                all_meetings.append((i, m["text"], venue))
                self.log(f"  開催{i+1}: {m['text']} ({venue})")

            # 同じ会場が2回あるか判定
            venue_meetings = {}
            for idx, text, venue in all_meetings:
                venue_meetings.setdefault(venue, []).append((idx, text, venue))

            is_sunday = datetime.now().weekday() == 6  # 0=Mon, 6=Sun
            self.meeting_info = []
            for venue, entries in venue_meetings.items():
                if len(entries) >= 2:
                    pick = entries[-1] if is_sunday else entries[0]
                    skip = entries[0] if is_sunday else entries[-1]
                    self.log(f"  → {pick[1]} を本日分として採用"
                             f"（{skip[1]} はスキップ）")
                    self.meeting_info.append(pick)
                else:
                    self.meeting_info.append(entries[0])

            for m_idx, m_text, venue in self.meeting_info:
                if self.venue_filter and venue not in self.venue_filter:
                    self.log(f"  [{venue}] スキップ（フィルタ対象外）")
                    continue

                meetings = await scraper.goto_odds_top()
                if m_idx >= len(meetings):
                    continue
                await scraper.select_meeting(meetings[m_idx]["element"])

                for race_num in range(1, 13):
                    race_key = f"{venue}{race_num}R"
                    try:
                        await scraper.select_race(race_num)
                        race_info = await scraper.scrape_race_info(m_text, race_num)

                        post_time_str = race_info.get("post_time", "")
                        if not post_time_str:
                            post_time_str = DEFAULT_POST_TIMES.get(race_num, "12:00")
                            self.log(f"  {race_key}: 発走時刻未検出 → デフォルト {post_time_str}")

                        self.races[race_key] = {
                            "data": None,
                            "enriched": None,
                            "race_info": race_info,
                            "meeting_idx": m_idx,
                            "meeting_text": m_text,
                            "post_time": post_time_str,
                            "predicted_ev": None,
                            "bet_results": None,
                        }

                        name = race_info.get("name", "")
                        self.log(f"  {race_key}: {name} 発走{post_time_str}")

                    except Exception as e:
                        self.log(f"  {race_key}: エラー ({e})")
                        continue
        finally:
            await scraper.close()

    def _build_schedule(self):
        """発走時刻からスケジュールを作成"""
        today = datetime.now().date()
        now = datetime.now()
        self.schedule = []

        for race_key, race in self.races.items():
            post_str = race["post_time"]
            try:
                h, m = map(int, post_str.split(":"))
                post_dt = datetime.combine(
                    today, datetime.min.time().replace(hour=h, minute=m)
                )
                trigger_dt = post_dt - timedelta(minutes=self.before)

                if trigger_dt > now:
                    self.schedule.append((trigger_dt, post_dt, race_key))
                else:
                    self.log(f"  {race_key}: 発走{post_str} → 既に通過（スキップ）")
            except (ValueError, TypeError):
                self.log(f"  {race_key}: 時刻パースエラー '{post_str}'")

        self.schedule.sort(key=lambda x: x[0])
        self.log(f"\n  スケジュール: {len(self.schedule)} レース")
        for trigger_dt, post_dt, race_key in self.schedule:
            self.log(f"    {trigger_dt.strftime('%H:%M')} 実行 → "
                     f"{race_key} (発走{post_dt.strftime('%H:%M')})")

    def _format_schedule(self) -> str:
        """スケジュールを通知用にフォーマット"""
        lines = [
            f"\n[スケジュール] {len(self.schedule)}レース "
            f"(発走{self.before}分前に通知)"
        ]
        for trigger_dt, post_dt, race_key in self.schedule:
            race = self.races[race_key]
            name = race["race_info"].get("name", "")
            lines.append(f"{post_dt.strftime('%H:%M')} {race_key} {name}")
        return "\n".join(lines)
