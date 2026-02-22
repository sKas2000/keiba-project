"""
オッズ監視サーバー（スケジュール方式）
各レースの発走時刻N分前にオッズ取得→ML予測→Discord通知
発走15分後に結果取得→回収率計算→Discord通知
"""
import asyncio
import copy
import re
import traceback
from datetime import datetime, timedelta
from pathlib import Path

from config.settings import (
    RACES_DIR, MODEL_DIR, COURSE_NAME_TO_ID,
)
from src.data.storage import write_json
from src.notify import (
    send_notify, format_race_notification, format_result_notification,
    format_daily_summary, get_webhook_url,
)


# JRA標準発走時刻（post_timeが取得できない場合のフォールバック）
# 開催場・季節で多少ズレるが概算として十分
DEFAULT_POST_TIMES = {
    1: "10:00", 2: "10:30", 3: "11:00", 4: "11:30",
    5: "12:25", 6: "12:55", 7: "13:25", 8: "13:55",
    9: "14:25", 10: "14:55", 11: "15:25", 12: "15:40",
}

# 結果確認の待機時間（発走後N分）
RESULT_CHECK_DELAY = 15


class RaceMonitor:
    """レース発走時刻に合わせたスケジュール実行モニター"""

    def __init__(self, before: int = 5, token: str = None,
                 headless: bool = True, venue_filter: str = None):
        self.before = before
        self.webhook_url = token or get_webhook_url()
        self.headless = headless
        self.venue_filter = set(venue_filter.split(",")) if venue_filter else None

        # データ
        self.monitor_dir = RACES_DIR / "monitor" / datetime.now().strftime("%Y%m%d")
        self.races = {}          # race_key -> {data, enriched, race_info, ...}
        self.meeting_info = []   # [(index, text, venue), ...]
        self.schedule = []       # [(trigger_dt, post_dt, race_key), ...] sorted

    def log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] {msg}")

    # ==========================================================
    # メインエントリ
    # ==========================================================

    async def run(self):
        """メインループ"""
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
        self.log(f"  発走 {RESULT_CHECK_DELAY}分後 に結果確認")
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

            # Phase 3: スケジュール実行（予測 + 結果確認）
            self.log("=" * 50)
            self.log("Phase 3: スケジュール実行開始")
            self.log("=" * 50)
            await self._run_schedule()

            # 日次サマリ
            self._send_daily_summary()

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

            # JRAサイトは土日2日分を同時表示するため、今日の開催のみ処理
            # 土曜=先の開催（日数小）、日曜=後の開催（日数大）
            all_meetings = []
            for i, m in enumerate(meetings):
                venue = self._extract_venue(m["text"])
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
                    # 土日2日分: 土曜なら先(0)、日曜なら後(-1)
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
        """スケジュールを通知用にフォーマット"""
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
    # Phase 3: スケジュール実行（予測 + 結果確認）
    # ==========================================================

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

        # 結果確認キュー: [(check_dt, post_dt, [race_keys])]
        result_queue = []

        for g_idx, (trigger_dt, post_dt, race_keys) in enumerate(time_groups):
            # 待機（結果確認を挟む）
            remaining = sum(len(g[2]) for g in time_groups[g_idx:])
            await self._wait_with_result_checks(
                trigger_dt, result_queue,
                next_label=f"{race_keys[0]}他 発走{post_dt.strftime('%H:%M')}",
                remaining=remaining,
            )

            # 予測実行
            for race_key in race_keys:
                race = self.races.get(race_key)
                if not race or not race.get("enriched"):
                    self.log(f"  {race_key}: enrichedデータなし（スキップ）")
                    continue

                self.log(f"\n{'=' * 50}")
                self.log(f"[実行] {race_key} (発走{post_dt.strftime('%H:%M')})")
                self.log(f"{'=' * 50}")
                await self._scrape_predict_notify(race_key)

            # 結果確認をスケジュール
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
            # 結果確認が必要なレースがあるか
            ready = [i for i, q in enumerate(result_queue)
                     if q[0] <= datetime.now()]
            if ready:
                for i in reversed(ready):
                    _, _, race_keys = result_queue.pop(i)
                    await self._check_results_batch(race_keys)
                continue

            # 次のイベントまで待機
            next_events = [until_dt]
            next_events += [q[0] for q in result_queue]
            next_event = min(next_events)
            wait_secs = (next_event - datetime.now()).total_seconds()

            if wait_secs > 0:
                if next_event == until_dt and next_label:
                    self.log(f"\n--- 次: {next_label} "
                             f"(あと{int(wait_secs // 60)}分{int(wait_secs % 60)}秒待機, "
                             f"残り{remaining}レース) ---")
                    # 長い待機は一気にsleep
                    await asyncio.sleep(wait_secs)
                else:
                    # 結果確認までの短い待機
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
        race["predicted_ev"] = ev_results

        # ファイル保存
        result["ev_results"] = ev_results
        path = self.monitor_dir / f"{race_key}_predicted.json"
        write_json(result, path)

        # Discord通知
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
    # 結果確認 + 回収率計算
    # ==========================================================

    async def _check_results_batch(self, race_keys: list):
        """レースグループの結果を確認"""
        self.log(f"\n{'=' * 50}")
        self.log(f"[結果確認] {', '.join(race_keys)}")
        self.log(f"{'=' * 50}")

        for race_key in race_keys:
            race = self.races.get(race_key)
            if not race:
                continue

            # 予測がないレースはスキップ
            ev = race.get("predicted_ev")
            if not ev:
                self.log(f"  {race_key}: 予測なし（スキップ）")
                continue

            # netkeiba race_id 構築
            race_id = self._construct_race_id(race)
            if not race_id:
                self.log(f"  {race_key}: race_id構築失敗")
                continue

            # 結果取得
            results, payoffs = await self._fetch_race_results(race_id)
            if not results:
                self.log(f"  {race_key}: 結果未確定 ({race_id})")
                continue

            # 着順（上位3頭）
            top3 = sorted(results, key=lambda r: r.get("finish_position", 99))[:3]

            # 賭け結果を計算
            bet_results = self._calculate_bet_results(race_key, payoffs, top3)
            race["bet_results"] = bet_results

            # 結果保存
            result_path = self.monitor_dir / f"{race_key}_results.json"
            write_json({
                "race_id": race_id,
                "top3": [{"num": r["horse_number"], "name": r["horse_name"],
                          "pos": r["finish_position"]} for r in top3],
                "payoffs": payoffs,
                "bet_results": bet_results,
            }, result_path)

            # Discord通知
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

    def _construct_race_id(self, race: dict) -> str:
        """meeting情報からnetkeiba race_idを構築
        形式: YYYYCCKK DDNN (12桁)
        """
        m_text = race.get("meeting_text", "")
        if not m_text:
            # meeting_infoから検索
            m_idx = race.get("meeting_idx")
            for idx, text, venue in self.meeting_info:
                if idx == m_idx:
                    m_text = text
                    break

        if not m_text:
            return ""

        # "1回東京8日" → kai=1, day=8
        m = re.match(r"(\d+)回(.+?)(\d+)日", m_text)
        if not m:
            return ""

        kai = int(m.group(1))
        day = int(m.group(3))

        venue_name = race["race_info"].get("venue", "")
        course_id = COURSE_NAME_TO_ID.get(venue_name, 0)
        if not course_id:
            return ""

        race_num = race["race_info"].get("race_number", 0)
        year = datetime.now().year

        return f"{year:04d}{course_id:02d}{kai:02d}{day:02d}{race_num:02d}"

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

        # race.netkeiba.com → db.netkeiba.com の順で試行
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

    def _calculate_bet_results(self, race_key: str, payoffs: list,
                               top3: list) -> dict:
        """予測と実際の結果を比較してベット結果を計算"""
        race = self.races[race_key]
        ev = race.get("predicted_ev", {})

        if not ev or ev.get("low_confidence"):
            return {"skipped": True, "reason": "見送り", "bets": [],
                    "total_invested": 0, "total_returned": 0,
                    "top3": [{"num": r["horse_number"], "name": r["horse_name"]}
                             for r in top3]}

        # payoffをdict化: {bet_type: {normalized_combo: payout}}
        payoff_map = {}
        for p in payoffs:
            bt = p["bet_type"]
            combo = p["combination"]
            # 組合せを正規化（数字のみ、ソート）
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

        # 馬連チェック
        for bet in ev.get("quinella", []):
            if bet["ev"] < 1.0:
                continue
            combo = bet.get("combo", "")
            parts = sorted(re.findall(r"\d+", str(combo)))
            norm = "-".join(parts)

            invested = 100
            results["total_invested"] += invested

            returned = payoff_map.get("quinella", {}).get(norm, 0)
            results["total_returned"] += returned
            results["bets"].append({
                "type": "馬連", "combo": combo,
                "odds": bet["odds"], "ev": bet["ev"],
                "invested": invested, "returned": returned,
                "won": returned > 0,
            })

        # ワイドチェック
        for bet in ev.get("wide", []):
            if bet["ev"] < 1.0:
                continue
            combo = bet.get("combo", "")
            parts = sorted(re.findall(r"\d+", str(combo)))
            norm = "-".join(parts)

            invested = 100
            results["total_invested"] += invested

            returned = payoff_map.get("wide", {}).get(norm, 0)
            results["total_returned"] += returned
            results["bets"].append({
                "type": "ワイド", "combo": combo,
                "odds": bet["odds"], "ev": bet["ev"],
                "invested": invested, "returned": returned,
                "won": returned > 0,
            })

        return results

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
