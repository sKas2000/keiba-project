"""
オッズ監視サーバー
JRAオッズを定期的に再取得し、ML予測結果をLINE通知
"""
import asyncio
import copy
import json
import re
import traceback
from datetime import datetime
from pathlib import Path

from config.settings import (
    RACES_DIR, MODEL_DIR, EXPANDING_BEST_PARAMS, setup_encoding,
)
from src.data.storage import write_json, read_json
from src.notify import (
    send_line_notify, format_race_notification,
    format_odds_change_notification, get_token,
)


class RaceMonitor:
    """JRAオッズ監視 + ML予測 + LINE通知サーバー"""

    def __init__(self, interval: int = 30, token: str = None,
                 headless: bool = True, venue_filter: str = None):
        """
        Args:
            interval: 再スキャン間隔（分）
            token: LINE Notify トークン
            headless: ブラウザ非表示
            venue_filter: 会場フィルタ（例: "東京", "中山,東京"）
        """
        self.interval = interval
        self.token = token or get_token()
        self.headless = headless
        self.venue_filter = set(venue_filter.split(",")) if venue_filter else None

        # 監視データ
        self.monitor_dir = RACES_DIR / "monitor" / datetime.now().strftime("%Y%m%d")
        self.races = {}          # race_key -> {data, enriched, predicted, race_info}
        self.prev_recs = {}      # race_key -> [(bet_type, bet_dict), ...]
        self.meeting_info = []   # [(index, text, venue), ...]

    def log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] {msg}")

    # ==========================================================
    # メインエントリ
    # ==========================================================

    async def run(self):
        """メインループ"""
        setup_encoding()

        if not self.token:
            print("=" * 60)
            print("[ERROR] LINE Notifyトークンが未設定")
            print()
            print("設定方法:")
            print("  1. https://notify-bot.line.me/ でトークン取得")
            print("  2. 以下のいずれかで設定:")
            print("     a) 環境変数: set LINE_NOTIFY_TOKEN=xxxxx")
            print("     b) .envファイルに LINE_NOTIFY_TOKEN=xxxxx を記載")
            print("     c) コマンド引数: python main.py monitor --token xxxxx")
            print("=" * 60)
            return

        self.monitor_dir.mkdir(parents=True, exist_ok=True)
        self.log("オッズ監視サーバー開始")
        self.log(f"  監視間隔: {self.interval}分")
        self.log(f"  会場フィルタ: {self.venue_filter or '全会場'}")
        self.log(f"  データ保存先: {self.monitor_dir}")

        # LINE疎通テスト
        if send_line_notify("\n[監視開始] オッズ監視サーバーが起動しました", self.token):
            self.log("LINE通知テスト: OK")
        else:
            self.log("[WARN] LINE通知テスト失敗 — トークンを確認してください")
            return

        try:
            # Phase 1: 初期スキャン（オッズ取得）
            self.log("=" * 50)
            self.log("Phase 1: オッズ取得")
            self.log("=" * 50)
            await self._scan_all_odds()

            if not self.races:
                self.log("[END] 本日のレースがありません")
                send_line_notify("\n本日のJRAレースはありません", self.token)
                return

            self.log(f"  {len(self.races)} レース検出")

            # Phase 2: 馬データ補完（netkeiba）
            self.log("=" * 50)
            self.log("Phase 2: 馬データ補完（netkeiba）")
            self.log("  ※ 1レースあたり約3分、全レースで30-60分かかります")
            self.log("=" * 50)
            await self._enrich_all()

            # Phase 3: 初回ML予測 + 通知
            self.log("=" * 50)
            self.log("Phase 3: ML予測 + 初回通知")
            self.log("=" * 50)
            self._predict_and_notify_all()

            # Phase 4: 監視ループ
            cycle = 1
            while True:
                self.log(f"\n--- {self.interval}分待機中 (Ctrl+Cで停止) ---")
                await asyncio.sleep(self.interval * 60)

                self.log("=" * 50)
                self.log(f"再スキャン #{cycle}")
                self.log("=" * 50)
                await self._rescrape_odds()
                self._predict_and_notify_updates()
                cycle += 1

        except KeyboardInterrupt:
            self.log("\n[停止] Ctrl+Cで停止しました")
            send_line_notify("\n[停止] オッズ監視サーバーを停止しました", self.token)

    # ==========================================================
    # Phase 1: 全レースのオッズ取得
    # ==========================================================

    async def _scan_all_odds(self):
        from src.scraping.odds import OddsScraper, build_input_json

        scraper = OddsScraper(headless=self.headless, debug=False)
        await scraper.start()
        try:
            meetings = await scraper.goto_odds_top()
            if not meetings:
                self.log("[WARN] 開催情報が見つかりません")
                return

            # 会場情報を保存
            self.meeting_info = []
            for i, m in enumerate(meetings):
                venue = self._extract_venue(m["text"])
                self.meeting_info.append((i, m["text"], venue))
                self.log(f"  開催{i+1}: {m['text']} ({venue})")

            for m_idx, m_text, venue in self.meeting_info:
                if self.venue_filter and venue not in self.venue_filter:
                    self.log(f"  [{venue}] スキップ（フィルタ対象外）")
                    continue

                # 開催を再取得して選択
                meetings = await scraper.goto_odds_top()
                if m_idx >= len(meetings):
                    continue
                await scraper.select_meeting(meetings[m_idx]["element"])

                for race_num in range(1, 13):
                    race_key = f"{venue}{race_num}R"
                    try:
                        await scraper.select_race(race_num)
                        race_info = await scraper.scrape_race_info(m_text, race_num)
                        horses = await scraper.parse_win_place()

                        if not horses:
                            self.log(f"  {race_key}: 出走馬なし（スキップ）")
                            continue

                        quinella = await scraper.parse_triangle_odds("馬連")
                        wide = await scraper.parse_triangle_odds("ワイド", is_range=True)
                        trio = await scraper.parse_trio()

                        scraped = {
                            "horses": horses, "quinella": quinella,
                            "wide": wide, "trio": trio,
                        }
                        data = build_input_json(scraped, race_info)

                        self.races[race_key] = {
                            "data": data,
                            "enriched": None,
                            "race_info": race_info,
                            "meeting_idx": m_idx,
                        }

                        # ファイル保存
                        path = self.monitor_dir / f"{race_key}_input.json"
                        write_json(data, path)

                        n_horses = len(horses)
                        self.log(f"  {race_key}: {race_info.get('name', '')} "
                                 f"({n_horses}頭, 馬連{len(quinella)}組)")

                    except Exception as e:
                        self.log(f"  {race_key}: エラー ({e})")
                        continue
        finally:
            await scraper.close()

    # ==========================================================
    # Phase 2: 馬データ補完
    # ==========================================================

    async def _enrich_all(self):
        from src.scraping.horse import HorseScraper

        scraper = HorseScraper(headless=self.headless, debug=False)
        try:
            await scraper.start()
            total = len(self.races)

            for r_idx, (race_key, race) in enumerate(self.races.items(), 1):
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
                        await scraper.restart()

                    # 生年計算
                    birth_year = 0
                    sex_age = horse.get("sex_age", "")
                    race_year = data.get("race", {}).get("date", "")[:4]
                    if sex_age and race_year:
                        age_match = re.search(r'(\d+)', sex_age)
                        if age_match:
                            birth_year = int(race_year) - int(age_match.group(1))

                    try:
                        horse_url = await scraper.search_horse(
                            horse_name, birth_year=birth_year
                        )
                        if horse_url:
                            past_races = await scraper.scrape_horse_past_races(horse_url)
                            horse["past_races"] = past_races
                            trainer_name = await scraper.get_trainer_name()
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
                            jockey_id = scraper.extract_jockey_id_from_past_races(
                                jockey_name, past
                            )
                            if jockey_id:
                                stats = await scraper.get_jockey_stats_by_id(
                                    jockey_id, jockey_name
                                )
                            else:
                                stats = await scraper.search_jockey(jockey_name)
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

                # enriched データを保存
                race["enriched"] = data
                path = self.monitor_dir / f"{race_key}_enriched.json"
                write_json(data, path)
                self.log(f"  {race_key}: 補完完了")

        except KeyboardInterrupt:
            self.log("[!] 補完中断 — 完了済みレースのみ監視します")
        except Exception as e:
            self.log(f"[ERROR] 補完エラー: {e}")
            traceback.print_exc()
        finally:
            await scraper.close()

    # ==========================================================
    # Phase 3: ML予測 + 初回通知
    # ==========================================================

    def _predict_and_notify_all(self):
        from src.model.predictor import score_ml, calculate_ev

        summary_lines = ["\n[初回予測結果]"]
        has_recs = False

        for race_key, race in self.races.items():
            enriched = race.get("enriched")
            if enriched is None:
                continue

            data = copy.deepcopy(enriched)
            result = score_ml(data, model_dir=MODEL_DIR)
            if result is None:
                self.log(f"  {race_key}: MLモデル未学習（スキップ）")
                continue

            ev_results = calculate_ev(result)
            race_info = data.get("race", {})

            # 推奨買い目を収集
            recs = self._collect_recommendations(ev_results)
            self.prev_recs[race_key] = recs

            # ファイル保存
            result["ev_results"] = ev_results
            path = self.monitor_dir / f"{race_key}_predicted.json"
            write_json(result, path)

            # 通知テキスト作成
            if recs and not ev_results.get("low_confidence"):
                has_recs = True
                text = format_race_notification(race_info, ev_results)
                summary_lines.append(text)
                self.log(f"  {race_key}: {len(recs)} 件の推奨買い目")
            else:
                reason = "確信度不足" if ev_results.get("low_confidence") else "推奨なし"
                self.log(f"  {race_key}: {reason}")

        # LINE通知送信（1000文字制限のため分割送信）
        if has_recs:
            self._send_chunked(summary_lines)
        else:
            send_line_notify("\n[初回結果] 推奨買い目のあるレースはありません", self.token)

    # ==========================================================
    # Phase 4: オッズ再取得 + 変動通知
    # ==========================================================

    async def _rescrape_odds(self):
        """オッズだけ再取得し、enriched データに反映"""
        from src.scraping.odds import OddsScraper

        scraper = OddsScraper(headless=self.headless, debug=False)
        await scraper.start()
        try:
            for m_idx, m_text, venue in self.meeting_info:
                if self.venue_filter and venue not in self.venue_filter:
                    continue

                meetings = await scraper.goto_odds_top()
                if m_idx >= len(meetings):
                    continue
                await scraper.select_meeting(meetings[m_idx]["element"])

                for race_num in range(1, 13):
                    race_key = f"{venue}{race_num}R"
                    race = self.races.get(race_key)
                    if not race or not race.get("enriched"):
                        continue

                    try:
                        await scraper.select_race(race_num)
                        horses = await scraper.parse_win_place()

                        if not horses:
                            continue

                        quinella = await scraper.parse_triangle_odds("馬連")
                        wide = await scraper.parse_triangle_odds("ワイド", is_range=True)
                        trio = await scraper.parse_trio()

                        # enriched データのオッズだけ更新
                        enriched = race["enriched"]
                        for h_new in horses:
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
                        self.log(f"  {race_key}: オッズ更新完了")

                    except Exception as e:
                        self.log(f"  {race_key}: 再取得エラー ({e})")
                        continue
        finally:
            await scraper.close()

    def _predict_and_notify_updates(self):
        """再予測して変動を通知"""
        from src.model.predictor import score_ml, calculate_ev

        changes = []

        for race_key, race in self.races.items():
            enriched = race.get("enriched")
            if enriched is None:
                continue

            data = copy.deepcopy(enriched)
            result = score_ml(data, model_dir=MODEL_DIR)
            if result is None:
                continue

            ev_results = calculate_ev(result)
            race_info = data.get("race", {})
            curr_recs = self._collect_recommendations(ev_results)
            prev_recs = self.prev_recs.get(race_key, [])

            # 変動検知
            change_text = format_odds_change_notification(
                race_info, prev_recs, curr_recs
            )
            if change_text:
                changes.append(change_text)
                self.log(f"  {race_key}: 推奨変動あり")
            else:
                self.log(f"  {race_key}: 変動なし")

            self.prev_recs[race_key] = curr_recs

            # ファイル更新
            result["ev_results"] = ev_results
            path = self.monitor_dir / f"{race_key}_predicted.json"
            write_json(result, path)

        if changes:
            self._send_chunked(changes)
        else:
            self.log("  全レース変動なし")

    # ==========================================================
    # ヘルパー
    # ==========================================================

    def _collect_recommendations(self, ev_results: dict) -> list:
        """推奨買い目を収集 → [(bet_type, bet_dict), ...]"""
        recs = []
        if ev_results.get("low_confidence"):
            return recs
        for bt, key in [("馬連", "quinella"), ("ワイド", "wide")]:
            for bet in ev_results.get(key, []):
                if bet["ev"] >= 1.0:
                    recs.append((bt, bet))
        return recs

    def _send_chunked(self, lines: list):
        """1000文字制限でチャンク分割送信"""
        chunk = ""
        for line in lines:
            if len(chunk) + len(line) + 1 > 950:
                if chunk:
                    send_line_notify(chunk, self.token)
                chunk = line
            else:
                chunk += ("\n" if chunk else "") + line
        if chunk:
            send_line_notify(chunk, self.token)

    @staticmethod
    def _extract_venue(meeting_text: str) -> str:
        """開催テキストから会場名を抽出"""
        from src.scraping.parsers import extract_venue
        return extract_venue(meeting_text)
