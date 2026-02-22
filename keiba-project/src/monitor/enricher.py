"""
Phase 2: 馬データ補完（netkeiba出馬表 + 過去走 + 騎手成績）
"""
import asyncio
import copy
import re
import traceback

from src.data.storage import write_json

from .helpers import construct_race_id


class EnricherMixin:
    """馬データ補完"""

    async def _enrich_all(self):
        """全レースの馬データを補完（スケジュール対象のみ）"""
        from src.scraping.horse import HorseScraper
        from src.scraping.odds import OddsScraper, build_input_json

        scheduled_keys = {rk for _, _, rk in self.schedule}
        self.log(f"  オッズ取得対象: {len(scheduled_keys)}レース")

        # オッズ取得
        odds_scraper = OddsScraper(headless=self.headless, debug=False)
        await odds_scraper.start()
        try:
            for m_idx, m_text, venue in self.meeting_info:
                if self.venue_filter and venue not in self.venue_filter:
                    continue

                meetings = await odds_scraper.goto_odds_top()
                if m_idx >= len(meetings):
                    continue
                await odds_scraper.select_meeting(meetings[m_idx]["element"])

                for race_num in range(1, 13):
                    race_key = f"{venue}{race_num}R"
                    if race_key not in scheduled_keys:
                        continue
                    race = self.races.get(race_key)
                    if not race:
                        continue

                    try:
                        await odds_scraper.select_race(race_num)
                        horses = await odds_scraper.parse_win_place()
                        if not horses:
                            self.log(f"  {race_key}: 出走馬なし")
                            continue

                        quinella = await odds_scraper.parse_triangle_odds("馬連")
                        wide = await odds_scraper.parse_triangle_odds(
                            "ワイド", is_range=True
                        )
                        trio = await odds_scraper.parse_trio()

                        scraped = {
                            "horses": horses, "quinella": quinella,
                            "wide": wide, "trio": trio,
                        }
                        data = build_input_json(scraped, race["race_info"])
                        race["data"] = data

                        path = self.monitor_dir / f"{race_key}_input.json"
                        write_json(data, path)
                        self.log(f"  {race_key}: オッズ取得完了 ({len(horses)}頭)")

                    except Exception as e:
                        self.log(f"  {race_key}: オッズ取得エラー ({e})")
        finally:
            await odds_scraper.close()

        # netkeiba出馬表から馬体重・馬場状態・調教師名を補完
        self.log("\n  netkeiba出馬表データ取得中...")
        shutuba_cache = {}
        for race_key, race in self.races.items():
            if race_key not in scheduled_keys or not race.get("data"):
                continue
            race_id = construct_race_id(race, self.meeting_info)
            if not race_id:
                continue
            try:
                from src.scraping.race import fetch_shutuba
                shutuba = await fetch_shutuba(race_id)
                horses_s = shutuba.get("horses", [])
                if horses_s:
                    shutuba_cache[race_key] = shutuba
                    self.log(f"    {race_key}: 出馬表取得OK ({len(horses_s)}頭)")
                    meta = shutuba.get("race_meta", {})
                    race_data = race["data"].get("race", {})
                    if meta.get("track_condition") and not race_data.get("track_condition"):
                        race_data["track_condition"] = meta["track_condition"]
                    if meta.get("weather") and not race_data.get("weather"):
                        race_data["weather"] = meta["weather"]
                else:
                    self.log(f"    {race_key}: 出馬表データなし")
            except Exception as e:
                self.log(f"    {race_key}: 出馬表取得エラー ({e})")
            await asyncio.sleep(1.0)

        # 馬データ補完（過去走 + 騎手成績）
        horse_scraper = HorseScraper(headless=self.headless, debug=False)
        try:
            await horse_scraper.start()

            enriched_count = 0
            targets = [(k, r) for k, r in self.races.items()
                       if k in scheduled_keys and r.get("data")]
            total = len(targets)

            for r_idx, (race_key, race) in enumerate(targets, 1):
                data = copy.deepcopy(race["data"])
                horses = data.get("horses", [])
                self.log(f"\n[{r_idx}/{total}] {race_key}: "
                         f"{len(horses)}頭を補完中...")

                # 出馬表データをマージ
                shutuba = shutuba_cache.get(race_key)
                if shutuba:
                    self._merge_shutuba(horses, shutuba)

                for i, horse in enumerate(horses):
                    horse_name = horse.get("name", "")
                    if not horse_name:
                        horse["past_races"] = []
                        horse["jockey_stats"] = {
                            "win_rate": 0.0, "place_rate": 0.0,
                            "wins": 0, "races": 0,
                        }
                        continue

                    if i > 0 and i % 5 == 0:
                        await horse_scraper.restart()

                    birth_year = self._calc_birth_year(horse, data)

                    try:
                        horse_url = await horse_scraper.search_horse(
                            horse_name, birth_year=birth_year
                        )
                        if horse_url:
                            past_races = await horse_scraper.scrape_horse_past_races(
                                horse_url
                            )
                            horse["past_races"] = past_races
                            trainer_name = await horse_scraper.get_trainer_name()
                            if trainer_name:
                                horse["trainer_name"] = trainer_name
                        else:
                            horse["past_races"] = []
                    except Exception as e:
                        self.log(f"    {horse_name}: 馬データエラー ({e})")
                        horse["past_races"] = []

                    await self._enrich_jockey_stats(horse, horse_scraper)
                    await asyncio.sleep(2.0)

                race["enriched"] = data
                path = self.monitor_dir / f"{race_key}_enriched.json"
                write_json(data, path)
                enriched_count += 1
                self.log(f"  {race_key}: 補完完了 ({enriched_count}/{total})")

        except KeyboardInterrupt:
            self.log("[!] 補完中断 — 完了済みレースのみ対象とします")
        except Exception as e:
            self.log(f"[ERROR] 補完エラー: {e}")
            traceback.print_exc()
        finally:
            await horse_scraper.close()

        self.log(f"\n  補完完了: {enriched_count}/{total} レース")

    @staticmethod
    def _merge_shutuba(horses: list, shutuba: dict):
        """出馬表データを馬リストにマージ"""
        shutuba_horses = {
            h["horse_number"]: h for h in shutuba.get("horses", [])
        }
        for horse in horses:
            sh = shutuba_horses.get(horse.get("num", 0))
            if not sh:
                continue
            if sh.get("horse_weight") and sh["horse_weight"] > 200:
                horse["weight"] = (
                    f"{sh['horse_weight']}({sh['horse_weight_change']:+d})"
                    if sh["horse_weight_change"] else str(sh["horse_weight"])
                )
            if sh.get("trainer_name"):
                horse["trainer_name"] = sh["trainer_name"]
            if sh.get("horse_id") and not horse.get("horse_id"):
                horse["horse_id"] = sh["horse_id"]

    @staticmethod
    def _calc_birth_year(horse: dict, data: dict) -> int:
        """馬の生年を推定"""
        sex_age = horse.get("sex_age", "")
        race_year = data.get("race", {}).get("date", "")[:4]
        if sex_age and race_year:
            age_match = re.search(r'(\d+)', sex_age)
            if age_match:
                return int(race_year) - int(age_match.group(1))
        return 0

    @staticmethod
    async def _enrich_jockey_stats(horse: dict, horse_scraper):
        """騎手成績を取得してhorseに追加"""
        jockey_name = horse.get("jockey", "")
        if not jockey_name:
            horse["jockey_stats"] = {
                "win_rate": 0.0, "place_rate": 0.0,
                "wins": 0, "races": 0,
            }
            return

        try:
            past = horse.get("past_races", [])
            jockey_id = horse_scraper.extract_jockey_id_from_past_races(
                jockey_name, past
            )
            if jockey_id:
                stats = await horse_scraper.get_jockey_stats_by_id(
                    jockey_id, jockey_name
                )
            else:
                stats = await horse_scraper.search_jockey(jockey_name)
            horse["jockey_stats"] = stats
        except Exception:
            horse["jockey_stats"] = {
                "win_rate": 0.0, "place_rate": 0.0,
                "wins": 0, "races": 0,
            }
