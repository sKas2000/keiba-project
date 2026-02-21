"""
オッズ監視サーバー（スケジュール方式）
各レースの発走時刻N分前にオッズ取得→ML予測→Discord通知
"""
import asyncio
import copy
import re
import traceback
from datetime import datetime, timedelta
from pathlib import Path

from config.settings import (
    RACES_DIR, MODEL_DIR, setup_encoding,
)
from src.data.storage import write_json
from src.notify import (
    send_notify, format_race_notification, get_webhook_url,
)


# JRA標準発走時刻（post_timeが取得できない場合のフォールバック）
# 開催場・季節で多少ズレるが概算として十分
DEFAULT_POST_TIMES = {
    1: "10:00", 2: "10:30", 3: "11:00", 4: "11:30",
    5: "12:25", 6: "12:55", 7: "13:25", 8: "13:55",
    9: "14:25", 10: "14:55", 11: "15:25", 12: "15:40",
}


class RaceMonitor:
    """レース発走時刻に合わせたスケジュール実行モニター"""

    def __init__(self, before: int = 5, token: str = None,
                 headless: bool = True, venue_filter: str = None):
        """
        Args:
            before: 発走何分前にバッチ実行するか（デフォルト5分）
            token: Discord Webhook URL
            headless: ブラウザ非表示
            venue_filter: 会場フィルタ（例: "東京", "中山,東京"）
        """
        self.before = before
        self.webhook_url = token or get_webhook_url()
        self.headless = headless
        self.venue_filter = set(venue_filter.split(",")) if venue_filter else None

        # データ
        self.monitor_dir = RACES_DIR / "monitor" / datetime.now().strftime("%Y%m%d")
        self.races = {}          # race_key -> {data, enriched, race_info, post_time}
        self.meeting_info = []   # [(index, text, venue), ...]
        self.schedule = []       # [(datetime, race_key), ...] sorted

    def log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] {msg}")

    # ==========================================================
    # メインエントリ
    # ==========================================================

    async def run(self):
        """メインループ"""
        setup_encoding()

        if not self.webhook_url:
            print("=" * 60)
            print("[ERROR] Discord Webhook URLが未設定")
            print()
            print("設定方法:")
            print("  1. Discordサーバーのチャンネル設定→連携サービス→ウェブフック作成")
            print("  2. Webhook URLをコピー")
            print("  3. 以下のいずれかで設定:")
            print("     a) .envファイルに DISCORD_WEBHOOK_URL=https://... を記載")
            print("     b) コマンド引数: python main.py monitor --webhook URL")
            print("=" * 60)
            return

        self.monitor_dir.mkdir(parents=True, exist_ok=True)
        self.log("オッズ監視サーバー開始")
        self.log(f"  発走 {self.before}分前 にバッチ実行")
        self.log(f"  会場フィルタ: {self.venue_filter or '全会場'}")

        # Discord疎通テスト
        if send_notify("[監視開始] オッズ監視サーバーが起動しました", self.webhook_url):
            self.log("Discord通知テスト: OK")
        else:
            self.log("[WARN] Discord通知テスト失敗 — Webhook URLを確認してください")
            return

        try:
            # Phase 1: レース一覧 + 発走時刻取得
            self.log("=" * 50)
            self.log("Phase 1: レース一覧・発走時刻取得")
            self.log("=" * 50)
            await self._scan_race_schedule()

            if not self.races:
                self.log("[END] 本日のレースがありません")
                send_notify("\n本日のJRAレースはありません", self.webhook_url)
                return

            # スケジュール作成
            self._build_schedule()

            if not self.schedule:
                self.log("[END] 実行予定のレースがありません（全て発走済み）")
                send_notify("\n全レース発走済みです", self.webhook_url)
                return

            # Phase 2: 馬データ補完（netkeiba）
            self.log("=" * 50)
            self.log("Phase 2: 馬データ補完（netkeiba）")
            self.log("  ※ 1レースあたり約3分かかります")
            self.log("=" * 50)
            await self._enrich_all()

            # スケジュール通知
            schedule_text = self._format_schedule()
            send_notify(schedule_text, self.webhook_url)

            # Phase 3: スケジュール実行
            self.log("=" * 50)
            self.log("Phase 3: スケジュール実行開始")
            self.log("=" * 50)
            await self._run_schedule()

            self.log("[完了] 全レースの処理が終了しました")
            send_notify("\n[完了] 本日の全レース処理が終了しました", self.webhook_url)

        except KeyboardInterrupt:
            self.log("\n[停止] Ctrl+Cで停止しました")
            send_notify("\n[停止] オッズ監視サーバーを停止しました", self.webhook_url)

    # ==========================================================
    # Phase 1: レース一覧 + 発走時刻
    # ==========================================================

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

            self.meeting_info = []
            for i, m in enumerate(meetings):
                venue = self._extract_venue(m["text"])
                self.meeting_info.append((i, m["text"], venue))
                self.log(f"  開催{i+1}: {m['text']} ({venue})")

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

                        # 発走時刻
                        post_time_str = race_info.get("post_time", "")
                        if not post_time_str:
                            post_time_str = DEFAULT_POST_TIMES.get(race_num, "12:00")
                            self.log(f"  {race_key}: 発走時刻未検出 → デフォルト {post_time_str}")

                        self.races[race_key] = {
                            "data": None,
                            "enriched": None,
                            "race_info": race_info,
                            "meeting_idx": m_idx,
                            "post_time": post_time_str,
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
                post_dt = datetime.combine(today, datetime.min.time().replace(hour=h, minute=m))
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
            self.log(f"    {trigger_dt.strftime('%H:%M')} 実行 → {race_key} (発走{post_dt.strftime('%H:%M')})")

    def _format_schedule(self) -> str:
        """スケジュールをLINE通知用にフォーマット"""
        lines = [f"\n[スケジュール] {len(self.schedule)}レース (発走{self.before}分前に通知)"]
        for trigger_dt, post_dt, race_key in self.schedule:
            race = self.races[race_key]
            name = race["race_info"].get("name", "")
            lines.append(f"{post_dt.strftime('%H:%M')} {race_key} {name}")
        return "\n".join(lines)

    # ==========================================================
    # Phase 2: 馬データ補完
    # ==========================================================

    async def _enrich_all(self):
        """全レースの馬データを補完（スケジュール対象のみ）"""
        from src.scraping.horse import HorseScraper
        from src.scraping.odds import OddsScraper, build_input_json

        # まずスケジュール対象レースのオッズを取得
        scheduled_keys = {rk for _, _, rk in self.schedule}
        self.log(f"  オッズ取得対象: {len(scheduled_keys)}レース")

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
                        wide = await odds_scraper.parse_triangle_odds("ワイド", is_range=True)
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

        # 馬データ補完
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
                self.log(f"\n[{r_idx}/{total}] {race_key}: {len(horses)}頭を補完中...")

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

                    birth_year = 0
                    sex_age = horse.get("sex_age", "")
                    race_year = data.get("race", {}).get("date", "")[:4]
                    if sex_age and race_year:
                        age_match = re.search(r'(\d+)', sex_age)
                        if age_match:
                            birth_year = int(race_year) - int(age_match.group(1))

                    try:
                        horse_url = await horse_scraper.search_horse(
                            horse_name, birth_year=birth_year
                        )
                        if horse_url:
                            past_races = await horse_scraper.scrape_horse_past_races(horse_url)
                            horse["past_races"] = past_races
                            trainer_name = await horse_scraper.get_trainer_name()
                            if trainer_name:
                                horse["trainer_name"] = trainer_name
                        else:
                            horse["past_races"] = []
                    except Exception as e:
                        self.log(f"    {horse_name}: 馬データエラー ({e})")
                        horse["past_races"] = []

                    jockey_name = horse.get("jockey", "")
                    if jockey_name:
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
                    else:
                        horse["jockey_stats"] = {
                            "win_rate": 0.0, "place_rate": 0.0,
                            "wins": 0, "races": 0,
                        }

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

    # ==========================================================
    # Phase 3: スケジュール実行
    # ==========================================================

    async def _run_schedule(self):
        """発走時刻に合わせてバッチ実行"""
        for i, (trigger_dt, post_dt, race_key) in enumerate(self.schedule):
            race = self.races.get(race_key)
            if not race or not race.get("enriched"):
                self.log(f"  {race_key}: enrichedデータなし（スキップ）")
                continue

            now = datetime.now()
            wait_seconds = (trigger_dt - now).total_seconds()

            if wait_seconds > 0:
                remaining = len(self.schedule) - i
                self.log(f"\n--- 次: {race_key} 発走{post_dt.strftime('%H:%M')} "
                         f"(あと{int(wait_seconds//60)}分{int(wait_seconds%60)}秒待機, "
                         f"残り{remaining}レース) ---")
                await asyncio.sleep(wait_seconds)

            self.log(f"\n{'=' * 50}")
            self.log(f"[実行] {race_key} (発走{post_dt.strftime('%H:%M')})")
            self.log(f"{'=' * 50}")

            # 最新オッズ取得 + ML予測 + 通知
            await self._scrape_predict_notify(race_key)

    async def _scrape_predict_notify(self, race_key: str):
        """単一レースのオッズ再取得→ML予測→LINE通知"""
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
                    wide = await scraper.parse_triangle_odds("ワイド", is_range=True)
                    trio = await scraper.parse_trio()

                    # enriched データのオッズだけ更新
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
                f"\n{venue}{race_num}R: MLモデル未学習のため予測不可", self.webhook_url
            )
            return

        ev_results = calculate_ev(result)

        # ファイル保存
        result["ev_results"] = ev_results
        path = self.monitor_dir / f"{race_key}_predicted.json"
        write_json(result, path)

        # LINE通知
        text = format_race_notification(race_info, ev_results)
        send_notify(text, self.webhook_url)

        # コンソールにも表示
        recs = self._collect_recommendations(ev_results)
        if recs:
            self.log(f"  {len(recs)} 件の推奨買い目を通知")
        elif ev_results.get("low_confidence"):
            self.log(f"  確信度不足 → 見送り推奨")
        else:
            self.log(f"  推奨買い目なし")

    # ==========================================================
    # ヘルパー
    # ==========================================================

    @staticmethod
    def _collect_recommendations(ev_results: dict) -> list:
        """推奨買い目を収集 → [(bet_type, bet_dict), ...]"""
        recs = []
        if ev_results.get("low_confidence"):
            return recs
        for bt, key in [("馬連", "quinella"), ("ワイド", "wide")]:
            for bet in ev_results.get(key, []):
                if bet["ev"] >= 1.0:
                    recs.append((bt, bet))
        return recs

    @staticmethod
    def _extract_venue(meeting_text: str) -> str:
        """開催テキストから会場名を抽出"""
        from src.scraping.parsers import extract_venue
        return extract_venue(meeting_text)
