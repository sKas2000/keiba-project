"""
馬の過去成績・騎手成績取得
netkeiba.com から馬詳細ページ・騎手ページをスクレイピング
"""
import asyncio
import json
import logging
import re
from pathlib import Path

from src.scraping.base import BaseScraper
from src.scraping.parsers import safe_int, safe_float, normalize_jockey_name

logger = logging.getLogger("keiba.scraping.horse")


class HorseScraper(BaseScraper):
    """netkeiba.com 馬・騎手スクレイパー"""

    def __init__(self, headless=True, debug=True):
        super().__init__(headless, debug)
        self.jockey_cache = {}

    # ----------------------------------------------------------
    # 馬名検索
    # ----------------------------------------------------------

    async def search_horse(self, horse_name: str, birth_year: int = 0) -> str:
        self.log(f"馬名検索: {horse_name}" + (f" (生年{birth_year})" if birth_year else ""))

        for attempt in range(self.max_retries):
            try:
                await self.page.goto(
                    "https://db.netkeiba.com/?pid=horse_search_detail",
                    wait_until="domcontentloaded", timeout=30000,
                )
                await asyncio.sleep(2)

                input_found = False
                try:
                    input_field = self.page.locator('input[name="horse_name"]')
                    if await input_field.count() > 0:
                        await input_field.fill(horse_name)
                        input_found = True
                except Exception as e:
                    logger.debug("input_field検索フォールバック: %s", e)

                if not input_found:
                    try:
                        text_inputs = await self.page.locator('input[type="text"]').all()
                        if text_inputs:
                            await text_inputs[0].fill(horse_name)
                            input_found = True
                    except Exception as e:
                        logger.debug("text_inputs検索フォールバック: %s", e)

                if not input_found:
                    self.log("  [!] 検索フォームが見つかりません")
                    return ""

                await asyncio.sleep(1)
                submit_clicked = False
                try:
                    submit_button = self.page.locator('input[type="submit"], button[type="submit"]')
                    if await submit_button.count() > 0:
                        await submit_button.first.click()
                        submit_clicked = True
                except Exception as e:
                    logger.debug("submit_button検索フォールバック: %s", e)
                if not submit_clicked:
                    try:
                        await self.page.keyboard.press("Enter")
                    except Exception as e:
                        logger.debug("Enter送信フォールバック: %s", e)

                await asyncio.sleep(3)

                links = await self.page.locator("table a[href*='/horse/']").all()
                if not links:
                    links = await self.page.locator("a[href*='/horse/']").all()

                candidates = []
                for link in links:
                    href = await link.get_attribute("href")
                    if href and "/horse/" in href:
                        if any(x in href for x in ["search_detail", "top.html", "sire/", "bms_", "leading"]):
                            continue
                        m = re.search(r'/horse/(\d{10})', href)
                        if m:
                            horse_id = m.group(1)
                            id_birth_year = int(horse_id[:4])
                            if href.startswith("/"):
                                href = f"https://db.netkeiba.com{href}"
                            candidates.append((href, id_birth_year))

                if not candidates:
                    self.log(f"  [!] 有効な馬ページが見つかりません: {horse_name}")
                    return ""

                if birth_year > 0:
                    matched = [c for c in candidates if c[1] == birth_year]
                    if matched:
                        return matched[0][0]
                    close = [c for c in candidates if abs(c[1] - birth_year) <= 1]
                    if close:
                        return close[0][0]

                candidates.sort(key=lambda c: c[1], reverse=True)
                return candidates[0][0]

            except Exception as e:
                self.log(f"  [ERROR] 馬検索エラー (試行{attempt+1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))

        return ""

    # ----------------------------------------------------------
    # 過去走データ
    # ----------------------------------------------------------

    async def scrape_horse_past_races(self, horse_url: str) -> list:
        self.log(f"過去走データ取得中: {horse_url}")
        past_races = []

        try:
            await self.page.goto(horse_url, wait_until="domcontentloaded")
            await asyncio.sleep(1.5)

            tables = await self.page.locator("table").all()
            result_table = None

            for table in tables:
                ths = await table.locator("th").all()
                headers = []
                for th in ths:
                    text = await th.text_content()
                    headers.append(text.strip() if text else "")

                if any("着順" in h for h in headers):
                    result_table = table
                    header_map = {}
                    keyword_map = {
                        # 既存フィールド
                        "日付": "date", "開催": "venue", "競馬場": "venue",
                        "レース名": "race", "距離": "distance", "馬場": "surface",
                        "着順": "finish", "騎手": "jockey", "タイム": "time",
                        "着差": "margin", "上がり": "last3f", "上り": "last3f",
                        "通過": "position",
                        "タイム指数": "time_index",  # プレミアム会員時のみ表示

                        # スーパープレミアム追加フィールド
                        "ペース": "pace",           # 37.4-39.6形式
                        "オッズ": "odds",
                        "人気": "popularity",
                        "斤量": "weight_carried",
                        "馬体重": "horse_weight",    # 465(-1)形式
                        "頭数": "horse_count",
                        "賞金": "prize_money",
                        "天気": "weather",
                        "枠番": "gate_number",
                        "馬番": "horse_number",
                        "R": "race_number",
                    }
                    for i, h in enumerate(headers):
                        for keyword, key in keyword_map.items():
                            if keyword in h and key not in header_map:
                                header_map[key] = i
                    break

            if not result_table:
                self.log("  [!] 競走成績テーブルが見つかりません")
                return past_races

            rows = await result_table.locator("tbody tr").all()
            if not rows:
                rows = await result_table.locator("tr").all()
                rows = [r for r in rows if await r.locator("th").count() == 0]

            for row_idx, row in enumerate(rows[:10]):  # 4走 → 10走に拡大
                cells = await row.locator("td").all()
                if len(cells) < 5:
                    continue

                cell_texts = []
                for cell in cells:
                    text = await cell.text_content()
                    cell_texts.append(text.strip() if text else "")

                race_data = {
                    # 既存フィールド
                    "date": "", "venue": "", "race": "",
                    "distance": "", "surface": "", "finish": 0,
                    "jockey_name": "", "jockey_id": "",
                    "margin": "", "time": "", "last3f": "", "position": "",
                    "time_index": "",  # タイム指数（プレミアム会員時のみ）

                    # スーパープレミアム追加フィールド
                    "pace_front": "", "pace_back": "",  # ペースを分割
                    "odds": "", "popularity": "",
                    "weight_carried": "",
                    "weight": "", "weight_change": "",  # 馬体重を分割
                    "horse_count": "",
                    "prize_money": "",
                    "weather": "",
                    "gate_number": "", "horse_number": "",
                    "race_number": "",
                }

                if "date" in header_map and header_map["date"] < len(cell_texts):
                    race_data["date"] = cell_texts[header_map["date"]]
                if "venue" in header_map and header_map["venue"] < len(cell_texts):
                    venue_text = cell_texts[header_map["venue"]]
                    venue_match = re.search(r'\d+回(.+?)\d+日', venue_text)
                    race_data["venue"] = venue_match.group(1) if venue_match else venue_text
                if "race" in header_map and header_map["race"] < len(cell_texts):
                    race_data["race"] = cell_texts[header_map["race"]]
                if "distance" in header_map and header_map["distance"] < len(cell_texts):
                    dist_text = cell_texts[header_map["distance"]]
                    dist_match = re.match(r'([芝ダ障])(\d+)', dist_text)
                    if dist_match:
                        race_data["surface"] = dist_match.group(1)
                        race_data["distance"] = dist_match.group(2)
                    else:
                        race_data["distance"] = re.sub(r'[^\d]', '', dist_text)
                if "surface" in header_map and header_map["surface"] < len(cell_texts):
                    if not race_data["surface"]:
                        race_data["surface"] = cell_texts[header_map["surface"]]
                if "finish" in header_map and header_map["finish"] < len(cell_texts):
                    race_data["finish"] = safe_int(cell_texts[header_map["finish"]])

                # 騎手情報
                if "jockey" in header_map and header_map["jockey"] < len(cells):
                    jockey_cell = cells[header_map["jockey"]]
                    jockey_text = await jockey_cell.text_content()
                    race_data["jockey_name"] = jockey_text.strip() if jockey_text else ""
                    jockey_links = await jockey_cell.locator("a[href*='/jockey/']").all()
                    if jockey_links:
                        href = await jockey_links[0].get_attribute("href")
                        jockey_id_match = re.search(r'/jockey/(?:result/recent/)?(\d{5})', href)
                        if jockey_id_match:
                            race_data["jockey_id"] = jockey_id_match.group(1)

                if "time" in header_map and header_map["time"] < len(cell_texts):
                    race_data["time"] = cell_texts[header_map["time"]]
                if "margin" in header_map and header_map["margin"] < len(cell_texts):
                    race_data["margin"] = cell_texts[header_map["margin"]]
                if "last3f" in header_map and header_map["last3f"] < len(cell_texts):
                    race_data["last3f"] = cell_texts[header_map["last3f"]]
                if "position" in header_map and header_map["position"] < len(cell_texts):
                    race_data["position"] = cell_texts[header_map["position"]]
                if "time_index" in header_map and header_map["time_index"] < len(cell_texts):
                    race_data["time_index"] = cell_texts[header_map["time_index"]]

                # スーパープレミアム追加フィールドのパース
                # ペース（37.4-39.6形式を分割）
                if "pace" in header_map and header_map["pace"] < len(cell_texts):
                    pace_text = cell_texts[header_map["pace"]]
                    if "-" in pace_text:
                        pace_parts = pace_text.split("-")
                        race_data["pace_front"] = pace_parts[0].strip()
                        race_data["pace_back"] = pace_parts[1].strip() if len(pace_parts) > 1 else ""

                # オッズ・人気
                if "odds" in header_map and header_map["odds"] < len(cell_texts):
                    race_data["odds"] = cell_texts[header_map["odds"]]
                if "popularity" in header_map and header_map["popularity"] < len(cell_texts):
                    race_data["popularity"] = cell_texts[header_map["popularity"]]

                # 斤量
                if "weight_carried" in header_map and header_map["weight_carried"] < len(cell_texts):
                    race_data["weight_carried"] = cell_texts[header_map["weight_carried"]]

                # 馬体重（465(-1)形式を分割）
                if "horse_weight" in header_map and header_map["horse_weight"] < len(cell_texts):
                    weight_text = cell_texts[header_map["horse_weight"]]
                    weight_match = re.match(r'(\d+)\(([+-]?\d+)\)', weight_text)
                    if weight_match:
                        race_data["weight"] = weight_match.group(1)
                        race_data["weight_change"] = weight_match.group(2)
                    else:
                        # マッチしない場合は数値部分だけ抽出
                        race_data["weight"] = re.sub(r'[^\d]', '', weight_text)

                # 頭数・賞金
                if "horse_count" in header_map and header_map["horse_count"] < len(cell_texts):
                    race_data["horse_count"] = cell_texts[header_map["horse_count"]]
                if "prize_money" in header_map and header_map["prize_money"] < len(cell_texts):
                    race_data["prize_money"] = cell_texts[header_map["prize_money"]]

                # 天気・枠番・馬番・R
                if "weather" in header_map and header_map["weather"] < len(cell_texts):
                    race_data["weather"] = cell_texts[header_map["weather"]]
                if "gate_number" in header_map and header_map["gate_number"] < len(cell_texts):
                    race_data["gate_number"] = cell_texts[header_map["gate_number"]]
                if "horse_number" in header_map and header_map["horse_number"] < len(cell_texts):
                    race_data["horse_number"] = cell_texts[header_map["horse_number"]]
                if "race_number" in header_map and header_map["race_number"] < len(cell_texts):
                    race_data["race_number"] = cell_texts[header_map["race_number"]]

                if race_data["finish"] > 0 or race_data["date"]:
                    past_races.append(race_data)

            self.log(f"  -> 過去走 {len(past_races)}件取得")
        except Exception as e:
            self.log(f"  [ERROR] 過去走取得エラー: {e}")
            import traceback
            traceback.print_exc()

        return past_races

    # ----------------------------------------------------------
    # 調教師名抽出（馬詳細ページから）
    # ----------------------------------------------------------

    async def get_trainer_name(self) -> str:
        """現在表示中の馬ページから調教師名を抽出"""
        try:
            # 馬プロフィールテーブルから「調教師」行を探す
            profile = self.page.locator("table.db_prof_table, table.horse_prof_table, div.db_prof_area_02 table")
            if await profile.count() > 0:
                rows = await profile.first.locator("tr").all()
                for row in rows:
                    th = await row.locator("th").first.text_content()
                    if th and "調教師" in th.strip():
                        td = await row.locator("td").first.text_content()
                        if td:
                            name = td.strip()
                            # "（栗東）田中克典" → "田中克典"
                            name = re.sub(r'[（\(][^）\)]*[）\)]', '', name).strip()
                            self.log(f"  調教師: {name}")
                            return name

            # フォールバック: リンクから調教師を探す
            trainer_links = await self.page.locator("a[href*='/trainer/']").all()
            for link in trainer_links:
                text = await link.text_content()
                if text and text.strip():
                    name = text.strip()
                    self.log(f"  調教師: {name}")
                    return name
        except Exception as e:
            self.log(f"  [!] 調教師名取得失敗: {e}")
        return ""

    # ----------------------------------------------------------
    # 血統・生産者取得
    # ----------------------------------------------------------

    async def get_pedigree_info(self) -> dict:
        """現在表示中の馬ページから血統(父・母父)・生産者を抽出

        Returns:
            {"sire": "父名", "bms": "母父名", "breeder": "生産者名"}
        """
        result = {"sire": "", "bms": "", "breeder": ""}
        try:
            # blood_table: td[0]=父, td[4]=母父(BMS)
            bt = self.page.locator("table.blood_table")
            if await bt.count() > 0:
                tds = await bt.locator("td").all()
                if len(tds) >= 1:
                    result["sire"] = (
                        await tds[0].text_content() or ""
                    ).strip()
                if len(tds) >= 5:
                    result["bms"] = (
                        await tds[4].text_content() or ""
                    ).strip()

            # db_prof_table: 「生産者」行
            prof = self.page.locator(
                "table.db_prof_table, div.db_prof_area_02 table"
            )
            if await prof.count() > 0:
                rows = await prof.first.locator("tr").all()
                for row in rows:
                    th_el = row.locator("th")
                    if await th_el.count() == 0:
                        continue
                    th = (await th_el.first.text_content() or "").strip()
                    if "生産者" in th:
                        td = (
                            await row.locator("td").first.text_content()
                            or ""
                        ).strip()
                        result["breeder"] = td
                        break

            if result["sire"]:
                self.log(
                    f"  血統: 父={result['sire']}, "
                    f"母父={result['bms']}, "
                    f"生産者={result['breeder']}"
                )
        except Exception as e:
            self.log(f"  [!] 血統取得失敗: {e}")

        return result

    # ----------------------------------------------------------
    # 騎手ID抽出（過去走データから）
    # ----------------------------------------------------------

    def extract_jockey_id_from_past_races(self, jockey_name: str, past_races: list) -> str:
        if not past_races:
            return ""
        normalized_search = normalize_jockey_name(jockey_name)

        for race in past_races:
            race_jockey_name = race.get("jockey_name", "")
            race_jockey_id = race.get("jockey_id", "")
            if race_jockey_name and race_jockey_id:
                if normalize_jockey_name(race_jockey_name) == normalized_search:
                    self.log(f"  [OK] 過去走で騎手発見: {race_jockey_name} (ID: {race_jockey_id})")
                    return race_jockey_id
        return ""

    # ----------------------------------------------------------
    # 騎手成績パース
    # ----------------------------------------------------------

    async def _parse_jockey_stats_from_page(self) -> dict:
        result = {"win_rate": 0.0, "place_rate": 0.0, "wins": 0, "races": 0}
        found_2026 = False

        tables = await self.page.locator("table").all()
        for table in tables:
            ths = await table.locator("th").all()
            if not ths:
                continue

            header_texts = []
            for th in ths:
                t = await th.text_content()
                header_texts.append(t.strip() if t else "")

            win_rate_idx = place_rate_idx = wins_idx = races_idx = -1
            for i, ht in enumerate(header_texts):
                if ht == "勝率":
                    win_rate_idx = i
                elif ht == "複勝率":
                    place_rate_idx = i
                elif ht == "1着":
                    wins_idx = i
                elif "騎乗回数" in ht or ht == "回数":
                    races_idx = i

            if win_rate_idx == -1 or races_idx == -1:
                continue

            rows = await table.locator("tr").all()
            for row in rows:
                cells = await row.locator("td").all()
                if not cells or len(cells) <= win_rate_idx:
                    continue

                first_cell_text = await cells[0].text_content()
                first_cell_text = first_cell_text.strip() if first_cell_text else ""

                if first_cell_text == "2026":
                    found_2026 = True
                    win_text = await cells[win_rate_idx].text_content()
                    win_text = win_text.strip().replace("％", "").replace("%", "") if win_text else "0"
                    result["win_rate"] = safe_float(win_text) / 100.0

                    if place_rate_idx >= 0 and len(cells) > place_rate_idx:
                        place_text = await cells[place_rate_idx].text_content()
                        place_text = place_text.strip().replace("％", "").replace("%", "") if place_text else "0"
                        result["place_rate"] = safe_float(place_text) / 100.0

                    if wins_idx >= 0 and len(cells) > wins_idx:
                        wins_text = await cells[wins_idx].text_content()
                        result["wins"] = safe_int(wins_text.strip() if wins_text else "0")

                    if races_idx >= 0 and len(cells) > races_idx:
                        races_text = await cells[races_idx].text_content()
                        races_text = races_text.strip().replace(",", "") if races_text else "0"
                        result["races"] = safe_int(races_text)
                    break

            if found_2026:
                break
        return result

    # ----------------------------------------------------------
    # 騎手ID -> 成績
    # ----------------------------------------------------------

    async def get_jockey_stats_by_id(self, jockey_id: str, jockey_name: str = "") -> dict:
        if jockey_name and jockey_name in self.jockey_cache:
            return self.jockey_cache[jockey_name]

        result = {"win_rate": 0.0, "place_rate": 0.0, "wins": 0, "races": 0}
        try:
            jockey_url = f"https://db.netkeiba.com/jockey/{jockey_id}"
            await self.page.goto(jockey_url, wait_until="domcontentloaded", timeout=30000)
            await asyncio.sleep(1.5)
            result = await self._parse_jockey_stats_from_page()
            if jockey_name:
                self.jockey_cache[jockey_name] = result
        except Exception as e:
            self.log(f"  [ERROR] 騎手成績取得エラー: {e}")
        return result

    # ----------------------------------------------------------
    # 調教データ取得（oikiri ページ, type=2: 最終追切）
    # ----------------------------------------------------------

    async def scrape_training_data(self, race_id: str) -> list:
        """レースの調教データを取得

        Args:
            race_id: 12桁のrace_id (e.g. "202609010211")

        Returns:
            馬ごとの調教データリスト
        """
        url = f"https://race.netkeiba.com/race/oikiri.html?race_id={race_id}&type=2"
        results = []

        try:
            await self.page.goto(url, wait_until="domcontentloaded", timeout=30000)
            await asyncio.sleep(3)

            # 繰り返しスクロールして全馬を読み込む（遅延読み込み対応）
            for _ in range(5):
                await self.page.evaluate(
                    "window.scrollTo(0, document.body.scrollHeight)"
                )
                await asyncio.sleep(1)

            # ページ先頭に戻る
            await self.page.evaluate("window.scrollTo(0, 0)")
            await asyncio.sleep(1)

            table = self.page.locator("table.OikiriTable").first
            if await table.count() == 0:
                self.log(f"  [!] 調教テーブルが見つかりません (race_id={race_id})")
                return results

            rows = await table.locator("tr.HorseList").all()
            if not rows:
                return results

            # 行はペアで構成: 情報行(馬番+コメント) + データ行(タイム等)
            i = 0
            while i < len(rows):
                info_row = rows[i]

                # 馬番を確認して情報行かどうか判定
                umaban_el = info_row.locator("td.Umaban")
                if await umaban_el.count() == 0:
                    i += 1
                    continue

                horse_num = safe_int((await umaban_el.text_content() or "").strip())
                if horse_num <= 0:
                    i += 1
                    continue

                # 馬名・horse_id
                horse_name = ""
                horse_id = ""
                name_links = await info_row.locator(
                    ".Horse_Name a, td.Horse_Info a[href*='/horse/']"
                ).all()
                for link in name_links:
                    href = await link.get_attribute("href") or ""
                    if "/horse/" in href and "training" not in href:
                        horse_name = (await link.text_content() or "").strip()
                        m = re.search(r'/horse/(\d{10})', href)
                        if m:
                            horse_id = m.group(1)
                        break

                # 調教コメント（TrainingReview_Cell）
                review = ""
                review_el = info_row.locator(".TrainingReview_Cell")
                if await review_el.count() > 0:
                    review = (await review_el.text_content() or "").strip()

                # データ行（次の行）
                training_date = ""
                training_course = ""
                track_condition = ""
                rider = ""
                training_time = ""
                training_load = ""
                evaluation = ""
                grade = ""
                overall_time = ""
                final_3f = ""
                final_1f = ""
                lap_times = ""
                sparring_info = ""

                if i + 1 < len(rows):
                    data_row = rows[i + 1]
                    # データ行かどうか確認（馬番セルがなければデータ行）
                    data_umaban = data_row.locator("td.Umaban")
                    if await data_umaban.count() == 0:
                        # 日付・コース・馬場・騎手はクラスセレクタで取得
                        day_el = data_row.locator("td.Training_Day")
                        if await day_el.count() > 0:
                            training_date = (
                                await day_el.text_content() or ""
                            ).strip()

                        cells = await data_row.locator("td").all()
                        cell_texts = []
                        for cell in cells:
                            t = await cell.text_content()
                            cell_texts.append(t.strip() if t else "")
                        if len(cell_texts) >= 4:
                            training_course = cell_texts[1]
                            track_condition = cell_texts[2]
                            rider = cell_texts[3]

                        # タイムデータを TrainingTimeDataList から構造化取得
                        time_list = data_row.locator(
                            ".TrainingTimeDataList li"
                        )
                        time_count = await time_list.count()
                        if time_count > 0:
                            times = []
                            laps = []
                            for ti in range(time_count):
                                li_el = time_list.nth(ti)
                                # <li>83.8<span>(17.3)</span></li>
                                # li全体のテキストから span テキストを引く
                                full = (
                                    await li_el.text_content() or ""
                                ).strip()
                                rap_el = li_el.locator(".RapTime")
                                rap = ""
                                if await rap_el.count() > 0:
                                    rap = (
                                        await rap_el.text_content() or ""
                                    ).strip()
                                # 通過タイム = 全テキスト - ラップ部分
                                t_val = full.replace(rap, "").strip()
                                # ラップタイムから括弧を除去
                                lap_val = rap.strip("()")
                                times.append(t_val)
                                laps.append(lap_val)

                            training_time = "-".join(times)
                            lap_times = "-".join(laps)
                            if times:
                                overall_time = times[0]
                            if len(times) >= 2:
                                final_3f = times[-2]
                            if times:
                                final_1f = times[-1]
                        else:
                            # TrainingTimeDataList がない場合は
                            # テキスト全体をフォールバック
                            if len(cell_texts) >= 5:
                                training_time = cell_texts[4]

                        # 併走情報（TrainingTimeData内のul後テキスト）
                        time_td = data_row.locator("td.TrainingTimeData")
                        if await time_td.count() > 0:
                            sp = await time_td.evaluate("""
                                el => {
                                    const ul = el.querySelector(
                                        '.TrainingTimeDataList'
                                    );
                                    if (!ul) return '';
                                    let text = '';
                                    let node = ul.nextSibling;
                                    while (node) {
                                        if (node.nodeType === 3)
                                            text += node.textContent;
                                        else if (node.nodeType === 1)
                                            text += node.textContent;
                                        node = node.nextSibling;
                                    }
                                    return text.trim();
                                }
                            """)
                            sparring_info = sp.replace("\n", " ").strip()

                        # CSS class で特定フィールドを取得
                        load_el = data_row.locator(".TrainingLoad")
                        if await load_el.count() > 0:
                            training_load = (
                                await load_el.text_content() or ""
                            ).strip()

                        critic_el = data_row.locator(".Training_Critic")
                        if await critic_el.count() > 0:
                            evaluation = (
                                await critic_el.text_content() or ""
                            ).strip()

                        rank_el = data_row.locator("[class*='Rank_']")
                        if await rank_el.count() > 0:
                            grade = (
                                await rank_el.text_content() or ""
                            ).strip()

                        i += 2  # 情報行 + データ行をスキップ
                    else:
                        # データ行なし → 情報行から評価だけ取得
                        critic_el = info_row.locator(".Training_Critic")
                        if await critic_el.count() > 0:
                            evaluation = (
                                await critic_el.text_content() or ""
                            ).strip()
                        rank_el = info_row.locator("[class*='Rank_']")
                        if await rank_el.count() > 0:
                            grade = (
                                await rank_el.text_content() or ""
                            ).strip()
                        i += 1
                else:
                    i += 1

                results.append({
                    "horse_number": horse_num,
                    "horse_name": horse_name,
                    "horse_id": horse_id,
                    "training_date": training_date,
                    "training_course": training_course,
                    "track_condition": track_condition,
                    "rider": rider,
                    "training_time": training_time,
                    "lap_times": lap_times,
                    "overall_time": overall_time,
                    "final_3f": final_3f,
                    "final_1f": final_1f,
                    "sparring_info": sparring_info,
                    "training_load": training_load,
                    "evaluation": evaluation,
                    "grade": grade,
                    "review": review,
                })

            self.log(f"  調教データ: {len(results)}頭取得 (race_id={race_id})")

        except Exception as e:
            self.log(f"  [!] 調教データ取得エラー: {e}")
            logger.debug("調教データ取得エラー (race_id=%s): %s", race_id, e)

        return results

    # ----------------------------------------------------------
    # 騎手名検索 -> 成績
    # ----------------------------------------------------------

    async def search_jockey(self, jockey_name: str) -> dict:
        if jockey_name in self.jockey_cache:
            return self.jockey_cache[jockey_name]

        result = {"win_rate": 0.0, "place_rate": 0.0, "wins": 0, "races": 0}

        for attempt in range(self.max_retries):
            try:
                await self.page.goto(
                    "https://db.netkeiba.com/?pid=jockey_search_detail",
                    wait_until="domcontentloaded", timeout=30000,
                )
                await asyncio.sleep(2)

                forms = await self.page.locator('form').all()
                jockey_form = None
                for form in forms:
                    hidden_inputs = await form.locator('input[type="hidden"]').all()
                    for h in hidden_inputs:
                        name = await h.get_attribute('name')
                        value = await h.get_attribute('value')
                        if name == 'pid' and value == 'jockey_list':
                            jockey_form = form
                            break
                    if jockey_form:
                        break

                if not jockey_form:
                    self.jockey_cache[jockey_name] = result
                    return result

                search_name = re.sub(r'[★▲△☆◇]', '', jockey_name)
                search_name = search_name.replace(" ", "").replace("\u3000", "")
                search_name = re.sub(
                    r'^([A-Za-z])\.',
                    lambda m: chr(ord(m.group(1).upper()) - ord('A') + ord('\uff21')) + '\uff0e',
                    search_name,
                )
                word_input = jockey_form.locator('input[name="word"]')
                await word_input.fill(search_name)

                await asyncio.sleep(1)
                submit_button = jockey_form.locator('input[type="submit"]')
                await submit_button.click()
                await asyncio.sleep(3)

                current_url = self.page.url
                jockey_redirect = re.search(r'/jockey/(\d{5})/?$', current_url)

                if jockey_redirect:
                    result = await self._parse_jockey_stats_from_page()
                else:
                    links = await self.page.locator("a[href*='/jockey/']").all()
                    normalized_search_name = normalize_jockey_name(jockey_name)
                    jockey_page_url = None

                    for link in links:
                        href = await link.get_attribute("href")
                        if href and "/jockey/" in href:
                            if any(x in href for x in ["search_detail", "top.html", "leading"]):
                                continue
                            if re.search(r'/jockey/\d{5}/?$', href):
                                link_text = await link.text_content()
                                if link_text and normalize_jockey_name(link_text.strip()) == normalized_search_name:
                                    if href.startswith("/"):
                                        href = f"https://db.netkeiba.com{href}"
                                    jockey_page_url = href
                                    break

                    if not jockey_page_url:
                        self.jockey_cache[jockey_name] = result
                        return result

                    await self.page.goto(jockey_page_url, wait_until="domcontentloaded")
                    await asyncio.sleep(1.5)
                    result = await self._parse_jockey_stats_from_page()

                self.jockey_cache[jockey_name] = result
                return result

            except Exception as e:
                self.log(f"  [ERROR] 騎手検索エラー (試行{attempt+1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))

        self.jockey_cache[jockey_name] = result
        return result


# ============================================================
# enriched_input.json 生成
# ============================================================

def _extract_horse_id_from_url(url: str) -> str:
    """馬ページURLからhorse_idを抽出"""
    m = re.search(r'/horse/(\d{10})', url)
    return m.group(1) if m else ""


def _construct_race_id_from_data(data: dict) -> str:
    """enriched data から netkeiba race_id を構築"""
    from src.monitor.helpers import build_race_id
    race = data.get("race", {})
    meeting_text = race.get("meeting_text", "")
    venue = race.get("venue", "")
    race_number = race.get("race_number", 0)
    return build_race_id(meeting_text, venue, race_number)


async def _enrich_horses(scraper: HorseScraper, data: dict) -> dict:
    """馬データのスクレイピング + キャッシュ統合コア処理"""
    from src.scraping.cache import HorseCache

    cache = HorseCache()
    cache_stats = {"horse_hit": 0, "horse_miss": 0, "jockey_hit": 0, "jockey_miss": 0}

    horses = data.get("horses", [])
    print(f"[対象馬] {len(horses)}頭")

    for i, horse in enumerate(horses):
        horse_name = horse.get("name", "")
        print(f"\n[{i+1}/{len(horses)}] {horse_name}")

        if i > 0 and i % 5 == 0:
            print("\n[メモリリフレッシュ] ブラウザを再起動します...")
            await scraper.restart()

        if not horse_name:
            horse["past_races"] = []
            horse["jockey_stats"] = {"win_rate": 0.0, "place_rate": 0.0, "wins": 0, "races": 0}
            continue

        # 生年計算
        birth_year = 0
        sex_age = horse.get("sex_age", "")
        race_year_str = data.get("race", {}).get("date", "")[:4]
        if sex_age and race_year_str:
            age_match = re.search(r'(\d+)', sex_age)
            if age_match:
                birth_year = int(race_year_str) - int(age_match.group(1))

        # キャッシュから馬データを検索（horse_idが判明している場合）
        horse_id = horse.get("horse_id", "")
        cached_horse = cache.get_horse(horse_id) if horse_id else None

        if cached_horse:
            # キャッシュヒット
            horse["past_races"] = cached_horse["past_races"]
            if cached_horse["trainer_name"]:
                horse["trainer_name"] = cached_horse["trainer_name"]
            # 血統データ復元
            pedigree = cached_horse.get("pedigree", {})
            if pedigree:
                horse["sire"] = pedigree.get("sire", "")
                horse["bms"] = pedigree.get("bms", "")
                horse["breeder"] = pedigree.get("breeder", "")
            cache_stats["horse_hit"] += 1
            print(f"  [cache] 馬データ取得済み (horse_id={horse_id})")
        else:
            # キャッシュミス → スクレイピング
            horse_url = await scraper.search_horse(horse_name, birth_year=birth_year)
            if horse_url:
                past_races = await scraper.scrape_horse_past_races(horse_url)
                horse["past_races"] = past_races
                trainer_name = await scraper.get_trainer_name()
                if trainer_name:
                    horse["trainer_name"] = trainer_name
                # 血統・生産者取得（馬ページ表示中に取得）
                pedigree = await scraper.get_pedigree_info()
                horse["sire"] = pedigree.get("sire", "")
                horse["bms"] = pedigree.get("bms", "")
                horse["breeder"] = pedigree.get("breeder", "")
                # キャッシュに保存
                scraped_id = _extract_horse_id_from_url(horse_url)
                if scraped_id:
                    horse["horse_id"] = scraped_id
                    cache.set_horse(scraped_id, horse_name, past_races,
                                    horse.get("trainer_name", ""),
                                    pedigree=pedigree)
            else:
                horse["past_races"] = []
            cache_stats["horse_miss"] += 1

        # 騎手成績
        jockey_name = horse.get("jockey", "")
        if jockey_name:
            cached_jockey = cache.get_jockey(jockey_name)
            if cached_jockey:
                horse["jockey_stats"] = cached_jockey
                cache_stats["jockey_hit"] += 1
                print(f"  [cache] 騎手成績取得済み ({jockey_name})")
            else:
                past_races = horse.get("past_races", [])
                jockey_id = scraper.extract_jockey_id_from_past_races(jockey_name, past_races)
                if jockey_id:
                    jockey_stats = await scraper.get_jockey_stats_by_id(jockey_id, jockey_name)
                else:
                    jockey_stats = await scraper.search_jockey(jockey_name)
                horse["jockey_stats"] = jockey_stats
                cache.set_jockey(jockey_name, jockey_stats)
                cache_stats["jockey_miss"] += 1
        else:
            horse["jockey_stats"] = {"win_rate": 0.0, "place_rate": 0.0, "wins": 0, "races": 0}

        await asyncio.sleep(2.0)

    # キャッシュ統計
    db_stats = cache.stats()
    print(f"\n[キャッシュ] 馬: hit={cache_stats['horse_hit']}, miss={cache_stats['horse_miss']} "
          f"| 騎手: hit={cache_stats['jockey_hit']}, miss={cache_stats['jockey_miss']} "
          f"| DB: 馬{db_stats['horses']}件, 騎手{db_stats['jockeys']}件")
    cache.close()

    # 調教データ取得（プレミアム会員 + ログイン済みの場合のみ）
    if scraper._netkeiba_logged_in:
        race_id = _construct_race_id_from_data(data)
        if race_id:
            print(f"\n[調教データ] race_id={race_id}")
            training = await scraper.scrape_training_data(race_id)
            if training:
                data.setdefault("premium", {})["training"] = training
                # 各馬にマージ
                training_map = {t["horse_number"]: t for t in training}
                for horse in horses:
                    t = training_map.get(horse.get("num", 0))
                    if t:
                        horse["training"] = {
                            "date": t["training_date"],
                            "course": t["training_course"],
                            "time": t["training_time"],
                            "lap_times": t.get("lap_times", ""),
                            "overall_time": t.get("overall_time", ""),
                            "final_3f": t.get("final_3f", ""),
                            "final_1f": t.get("final_1f", ""),
                            "sparring_info": t.get("sparring_info", ""),
                            "load": t["training_load"],
                            "evaluation": t["evaluation"],
                            "grade": t["grade"],
                            "review": t["review"],
                        }
                print(f"  [OK] 調教データ {len(training)}頭マージ完了")
            else:
                print("  [!] 調教データなし（非プレミアム or レース未登録）")
        else:
            print("\n[調教データ] race_id構築不可（meeting_text不足）")

    return data


async def enrich_race_data(scraper_or_path, data=None):
    """input.json を読み込み、netkeiba から追加データを取得して enriched_input.json に保存

    呼び出しパターン:
      1. enrich_race_data(scraper, data)  — パイプラインから（外部管理のscraper使用）
      2. enrich_race_data("path/to/input.json")  — CLI直接実行（内部でscraper作成）
    """
    # パターン1: パイプラインからの呼び出し（scraper + data）
    if data is not None and isinstance(scraper_or_path, HorseScraper):
        return await _enrich_horses(scraper_or_path, data)

    # パターン2: CLI直接実行（ファイルパスのみ）
    input_path = str(scraper_or_path)
    input_file = Path(input_path)
    if not input_file.exists():
        print(f"[ERROR] ファイルが見つかりません: {input_path}")
        return

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    scraper = HorseScraper(headless=True, debug=True)
    try:
        await scraper.start()
        await scraper.login_netkeiba()
        data = await _enrich_horses(scraper, data)

        output_file = input_file.parent / input_file.name.replace("_input.json", "_enriched_input.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"\n[OK] 保存完了: {output_file}")

    except KeyboardInterrupt:
        print("\n[!] 中断")
    except Exception as e:
        print(f"\n[ERROR] エラー: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await scraper.close()
