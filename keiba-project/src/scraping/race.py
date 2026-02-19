"""
レース結果取得
netkeiba.com から過去レース結果をスクレイピングし CSV で保存
"""
import asyncio
import csv
import re
from datetime import datetime, timedelta
from pathlib import Path

from src.scraping.base import BaseScraper
from src.scraping.parsers import (
    safe_int, safe_float, time_to_seconds, parse_horse_weight, parse_sex_age,
)
from config.settings import COURSE_CODES, CSV_COLUMNS, RAW_DIR


def generate_date_range(start_date: str, end_date: str) -> list:
    """日付範囲を生成（土日のみ）"""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    dates = []
    current = start
    while current <= end:
        if current.weekday() in (5, 6):
            dates.append(current.strftime("%Y%m%d"))
        current += timedelta(days=1)
    return dates


class RaceResultScraper(BaseScraper):
    """netkeiba 過去レース結果スクレイパー"""

    async def get_race_ids_for_date(self, date_str: str) -> list:
        url = f"https://db.netkeiba.com/race/list/{date_str}/"
        race_ids = []

        for attempt in range(self.max_retries):
            try:
                await self.page.goto(url, wait_until="domcontentloaded", timeout=30000)
                await asyncio.sleep(1.5)

                links = await self.page.locator("a[href*='/race/']").all()
                for link in links:
                    href = await link.get_attribute("href")
                    if href:
                        m = re.search(r'/race/(\d{12})/?', href)
                        if m and m.group(1) not in race_ids:
                            race_ids.append(m.group(1))

                self.log(f"  {date_str}: {len(race_ids)}レース発見")
                return race_ids
            except Exception as e:
                self.log(f"  [ERROR] レースID取得エラー (試行{attempt+1}): {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))
        return race_ids

    async def scrape_race_result(self, race_id: str) -> list:
        url = f"https://db.netkeiba.com/race/{race_id}/"
        results = []

        for attempt in range(self.max_retries):
            try:
                await self.page.goto(url, wait_until="domcontentloaded", timeout=30000)
                await asyncio.sleep(1.5)

                race_meta = await self._parse_race_meta(race_id)

                table = self.page.locator("table.race_table_01, table.nk_tb_common").first
                if await table.count() == 0:
                    tables = await self.page.locator("table").all()
                    found = False
                    for t in tables:
                        ths = await t.locator("th").all()
                        for th in ths:
                            text = await th.text_content()
                            if text and "着順" in text:
                                table = t
                                found = True
                                break
                        if found:
                            break
                    if not found:
                        return results

                ths = await table.locator("th").all()
                headers = []
                for th in ths:
                    text = await th.text_content()
                    headers.append(text.strip() if text else "")

                header_map = self._build_header_map(headers)

                rows = await table.locator("tbody tr").all()
                if not rows:
                    rows = await table.locator("tr").all()
                    rows = [r for r in rows if await r.locator("td").count() > 3]

                for row in rows:
                    cells = await row.locator("td").all()
                    if len(cells) < 5:
                        continue
                    cell_texts = []
                    for cell in cells:
                        text = await cell.text_content()
                        cell_texts.append(text.strip() if text else "")

                    horse_data = self._parse_row(cell_texts, cells, header_map, race_meta)
                    if horse_data and horse_data["finish_position"] > 0:
                        results.append(horse_data)
                        await self._extract_ids(cells, header_map, horse_data)

                race_meta["num_entries"] = len(results)
                for r in results:
                    r["num_entries"] = len(results)

                self.log(f"  {race_id}: {len(results)}頭パース完了")
                return results
            except Exception as e:
                self.log(f"  [ERROR] レース結果取得エラー (試行{attempt+1}): {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))
        return results

    async def _parse_race_meta(self, race_id: str) -> dict:
        meta = {
            "race_id": race_id,
            "race_date": "",
            "course_id": race_id[4:6],
            "course_name": COURSE_CODES.get(race_id[4:6], ""),
            "race_number": int(race_id[10:12]) if race_id[10:12].isdigit() else 0,
            "race_name": "", "race_class": "", "surface": "",
            "distance": 0, "track_condition": "", "weather": "",
            "num_entries": 0,
        }
        try:
            title_el = self.page.locator("h1, .racedata dt, dl.racedata h1").first
            if await title_el.count() > 0:
                title_text = await title_el.text_content()
                meta["race_name"] = title_text.strip() if title_text else ""

            info_el = self.page.locator(".racedata, dl.racedata, .data_intro, span.smalltxt").first
            if await info_el.count() > 0:
                info_text = await info_el.text_content()
                if info_text:
                    info_text = info_text.strip()
                    dist_match = re.search(r'([芝ダ障])(\d{3,4})m', info_text)
                    if dist_match:
                        meta["surface"] = dist_match.group(1)
                        meta["distance"] = int(dist_match.group(2))
                    weather_match = re.search(r'天候\s*[:：]?\s*([^\s/]+)', info_text)
                    if weather_match:
                        meta["weather"] = weather_match.group(1)
                    cond_match = re.search(r'[芝ダート]*\s*[:：]?\s*(良|稍重|重|不良)', info_text)
                    if cond_match:
                        meta["track_condition"] = cond_match.group(1)
                    date_match = re.search(r'(\d{4})年(\d{1,2})月(\d{1,2})日', info_text)
                    if date_match:
                        meta["race_date"] = f"{date_match.group(1)}-{int(date_match.group(2)):02d}-{int(date_match.group(3)):02d}"
                    for cls in ["G1", "G2", "G3", "GI", "GII", "GIII",
                                "オープン", "OP", "3勝", "2勝", "1勝",
                                "未勝利", "新馬", "リステッド"]:
                        if cls in info_text:
                            meta["race_class"] = cls.replace("GI", "G1").replace("GII", "G2").replace("GIII", "G3")
                            break

            if not meta["race_date"]:
                date_els = await self.page.locator("p.smalltxt, .race_otherdata, dd").all()
                for el in date_els:
                    text = await el.text_content()
                    if text:
                        dm = re.search(r'(\d{4})年(\d{1,2})月(\d{1,2})日', text)
                        if dm:
                            meta["race_date"] = f"{dm.group(1)}-{int(dm.group(2)):02d}-{int(dm.group(3)):02d}"
                            break
        except Exception as e:
            self.log(f"  [!] メタ情報パースエラー: {e}")
        return meta

    def _build_header_map(self, headers: list) -> dict:
        hmap = {}
        for i, h in enumerate(headers):
            if "着順" in h:
                hmap["finish"] = i
            elif "枠" in h and "番" in h:
                hmap["frame"] = i
            elif "馬番" in h:
                hmap["horse_num"] = i
            elif "馬名" in h:
                hmap["horse_name"] = i
            elif "性齢" in h or ("性" in h and "齢" in h):
                hmap["sex_age"] = i
            elif "斤量" in h:
                hmap["weight_carried"] = i
            elif "騎手" in h:
                hmap["jockey"] = i
            elif h.strip() == "タイム":
                hmap["time"] = i
            elif "着差" in h:
                hmap["margin"] = i
            elif "通過" in h:
                hmap["passing"] = i
            elif "上" in h and ("り" in h or "がり" in h):
                hmap["last3f"] = i
            elif "単勝" in h:
                hmap["odds"] = i
            elif "人気" in h:
                hmap["popularity"] = i
            elif "馬体重" in h:
                hmap["horse_weight"] = i
            elif "調教師" in h or "厩舎" in h:
                hmap["trainer"] = i
            elif "賞金" in h:
                hmap["prize"] = i
        return hmap

    def _parse_row(self, cell_texts, cells, header_map, race_meta) -> dict:
        row = {col: "" for col in CSV_COLUMNS}
        row.update(race_meta)

        def get(key):
            idx = header_map.get(key, -1)
            return cell_texts[idx] if 0 <= idx < len(cell_texts) else ""

        row["finish_position"] = safe_int(get("finish"))
        row["frame_number"] = safe_int(get("frame"))
        row["horse_number"] = safe_int(get("horse_num"))
        row["horse_name"] = get("horse_name").replace("\n", "").strip()
        row["weight_carried"] = safe_float(get("weight_carried"))
        row["jockey_name"] = get("jockey").replace("\n", "").strip()
        row["trainer_name"] = get("trainer").replace("\n", "").strip()
        row["finish_time_sec"] = time_to_seconds(get("time"))
        row["margin"] = get("margin")
        row["passing_order"] = get("passing")
        row["last_3f"] = safe_float(get("last3f"))
        row["win_odds"] = safe_float(get("odds"))
        row["popularity"] = safe_int(get("popularity"))
        row["prize_money"] = safe_float(get("prize"))

        sex, age = parse_sex_age(get("sex_age"))
        row["sex"] = sex
        row["age"] = age

        hw, hwc = parse_horse_weight(get("horse_weight"))
        row["horse_weight"] = hw
        row["horse_weight_change"] = hwc

        return row

    async def _extract_ids(self, cells, header_map, horse_data):
        try:
            name_idx = header_map.get("horse_name", -1)
            if 0 <= name_idx < len(cells):
                horse_links = await cells[name_idx].locator("a[href*='/horse/']").all()
                if horse_links:
                    href = await horse_links[0].get_attribute("href")
                    m = re.search(r'/horse/(\d{10})', href or "")
                    if m:
                        horse_data["horse_id"] = m.group(1)
            jockey_idx = header_map.get("jockey", -1)
            if 0 <= jockey_idx < len(cells):
                jockey_links = await cells[jockey_idx].locator("a[href*='/jockey/']").all()
                if jockey_links:
                    href = await jockey_links[0].get_attribute("href")
                    m = re.search(r'/jockey/(?:result/recent/)?(\d{5})', href or "")
                    if m:
                        horse_data["jockey_id"] = m.group(1)
        except Exception:
            pass


# ============================================================
# CSV 書き出し
# ============================================================

def save_results_csv(results: list, output_path: Path, append: bool = False):
    mode = "a" if append and output_path.exists() else "w"
    write_header = mode == "w" or not output_path.exists()

    with open(output_path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        for row in results:
            writer.writerow(row)


# ============================================================
# 一括収集
# ============================================================

async def collect_races(start_date: str, end_date: str, append: bool = False):
    print(f"\n{'=' * 60}")
    print(f"  過去レース結果収集")
    print(f"  期間: {start_date} ~ {end_date}")
    print(f"{'=' * 60}\n")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RAW_DIR / "results.csv"

    dates = generate_date_range(start_date, end_date)
    print(f"  対象日数: {len(dates)}日（土日のみ）")

    collector = RaceResultScraper(headless=True, debug=True)
    all_results = []
    total_races = 0

    try:
        await collector.start()

        for date_idx, date_str in enumerate(dates):
            print(f"\n[{date_idx+1}/{len(dates)}] {date_str}")

            if date_idx > 0 and date_idx % 20 == 0:
                await collector.restart()

            race_ids = await collector.get_race_ids_for_date(date_str)
            for race_id in race_ids:
                results = await collector.scrape_race_result(race_id)
                if results:
                    all_results.extend(results)
                    total_races += 1
                await asyncio.sleep(1.5)

            if all_results:
                is_append = append or date_idx > 0
                save_results_csv(all_results, output_path, append=is_append)
                print(f"  -> {len(all_results)}行保存済み (累計{total_races}レース)")
                all_results = []
            await asyncio.sleep(1.0)

    except KeyboardInterrupt:
        print("\n[!] 中断")
        if all_results:
            save_results_csv(all_results, output_path, append=True)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        await collector.close()

    print(f"\n[OK] 完了: {total_races}レース -> {output_path}")
