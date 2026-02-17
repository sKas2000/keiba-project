#!/usr/bin/env python3
"""
過去レース結果収集ツール v1.0
==============================
netkeiba.comから過去レース結果をスクレイピングし、CSV形式で保存する。

使い方:
  # 日付範囲指定で収集
  python race_collector.py --start 2024-01-01 --end 2024-12-31

  # 特定の日付のみ
  python race_collector.py --date 2024-06-01

  # 既存CSVに追記
  python race_collector.py --start 2024-01-01 --end 2024-03-31 --append
"""

VERSION = "1.0"

import asyncio
import csv
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from argparse import ArgumentParser

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

try:
    from playwright.async_api import async_playwright
except ImportError:
    print("Playwright が未インストールです:")
    print("  pip install playwright && playwright install chromium")
    sys.exit(1)


# ============================================================
# 定数
# ============================================================

COURSE_CODES = {
    "01": "札幌", "02": "函館", "03": "福島", "04": "新潟",
    "05": "東京", "06": "中山", "07": "中京", "08": "京都",
    "09": "阪神", "10": "小倉",
}

CSV_COLUMNS = [
    "race_id", "race_date", "course_id", "course_name",
    "race_number", "race_name", "race_class", "surface", "distance",
    "track_condition", "weather", "num_entries",
    "finish_position", "frame_number", "horse_number",
    "horse_name", "horse_id", "sex", "age",
    "jockey_name", "jockey_id", "trainer_name",
    "weight_carried", "horse_weight", "horse_weight_change",
    "finish_time_sec", "margin", "passing_order", "last_3f",
    "win_odds", "popularity", "prize_money",
]

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "ml" / "raw"


# ============================================================
# ユーティリティ
# ============================================================

def safe_int(text: str) -> int:
    try:
        return int(re.sub(r'[^\d\-]', '', str(text).strip()))
    except (ValueError, AttributeError):
        return 0


def safe_float(text: str) -> float:
    try:
        return float(str(text).strip().replace(",", ""))
    except (ValueError, AttributeError):
        return 0.0


def time_to_seconds(time_str: str) -> float:
    """タイム文字列を秒に変換 (例: '1:23.4' → 83.4)"""
    time_str = str(time_str).strip()
    if not time_str:
        return 0.0
    m = re.match(r'(\d+):(\d+)\.(\d+)', time_str)
    if m:
        return int(m.group(1)) * 60 + int(m.group(2)) + int(m.group(3)) / 10
    m2 = re.match(r'(\d+)\.(\d+)', time_str)
    if m2:
        return int(m2.group(1)) + int(m2.group(2)) / 10
    return 0.0


def parse_horse_weight(text: str) -> tuple:
    """馬体重文字列をパース (例: '480(+4)' → (480, 4))"""
    text = str(text).strip()
    m = re.match(r'(\d+)\(([+\-]?\d+)\)', text)
    if m:
        return int(m.group(1)), int(m.group(2))
    m2 = re.match(r'(\d+)', text)
    if m2:
        return int(m2.group(1)), 0
    return 0, 0


def parse_sex_age(text: str) -> tuple:
    """性齢文字列をパース (例: '牡3' → ('牡', 3))"""
    text = str(text).strip()
    m = re.match(r'([牡牝セ騸])(\d+)', text)
    if m:
        return m.group(1), int(m.group(2))
    return "", 0


def generate_date_range(start_date: str, end_date: str) -> list:
    """日付範囲を生成（土日のみ）"""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    dates = []
    current = start
    while current <= end:
        if current.weekday() in (5, 6):  # 土日
            dates.append(current.strftime("%Y%m%d"))
        current += timedelta(days=1)
    return dates


# ============================================================
# スクレイパー本体
# ============================================================

class RaceCollector:
    def __init__(self, headless=True, debug=True):
        self.headless = headless
        self.debug = debug
        self.pw = None
        self.browser = None
        self.ctx = None
        self.page = None
        self.max_retries = 3
        self.retry_delay = 2.0

    def log(self, msg):
        if self.debug:
            print(f"  [DEBUG] {msg}")

    async def start(self):
        self.pw = await async_playwright().start()
        self.browser = await self.pw.chromium.launch(headless=self.headless)
        self.ctx = await self.browser.new_context(
            viewport={"width": 1280, "height": 900},
            locale="ja-JP",
        )
        self.page = await self.ctx.new_page()
        self.page.set_default_timeout(15000)

    async def close(self):
        if self.ctx:
            await self.ctx.close()
        if self.browser:
            await self.browser.close()
        if self.pw:
            await self.pw.stop()

    async def restart(self):
        self.log("ブラウザ再起動中...")
        await self.close()
        await asyncio.sleep(2)
        await self.start()

    # ----------------------------------------------------------
    # 日付からレースID一覧を取得
    # ----------------------------------------------------------

    async def get_race_ids_for_date(self, date_str: str) -> list:
        """
        指定日付のレースID一覧を取得
        Args:
            date_str: 日付 (YYYYMMDD)
        Returns:
            レースIDのリスト
        """
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
                        if m:
                            race_id = m.group(1)
                            if race_id not in race_ids:
                                race_ids.append(race_id)

                self.log(f"  {date_str}: {len(race_ids)}レース発見")
                return race_ids

            except Exception as e:
                self.log(f"  [ERROR] レースID取得エラー (試行{attempt+1}): {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))

        return race_ids

    # ----------------------------------------------------------
    # レース結果ページのパース
    # ----------------------------------------------------------

    async def scrape_race_result(self, race_id: str) -> list:
        """
        レース結果ページをスクレイピング
        Args:
            race_id: レースID (12桁)
        Returns:
            各馬のデータ辞書のリスト
        """
        url = f"https://db.netkeiba.com/race/{race_id}/"
        results = []

        for attempt in range(self.max_retries):
            try:
                await self.page.goto(url, wait_until="domcontentloaded", timeout=30000)
                await asyncio.sleep(1.5)

                # レースメタ情報を取得
                race_meta = await self._parse_race_meta(race_id)

                # 結果テーブルをパース
                table = self.page.locator("table.race_table_01, table.nk_tb_common").first
                if await table.count() == 0:
                    # フォールバック: 着順を含むテーブルを探す
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
                        self.log(f"  [!] 結果テーブルなし: {race_id}")
                        return results

                # ヘッダー解析
                ths = await table.locator("th").all()
                headers = []
                for th in ths:
                    text = await th.text_content()
                    headers.append(text.strip() if text else "")

                header_map = self._build_header_map(headers)

                # データ行をパース
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

                        # 馬ID・騎手IDをリンクから抽出
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
        """レースページからメタ情報を取得"""
        meta = {
            "race_id": race_id,
            "race_date": "",
            "course_id": race_id[4:6],
            "course_name": COURSE_CODES.get(race_id[4:6], ""),
            "race_number": int(race_id[10:12]) if race_id[10:12].isdigit() else 0,
            "race_name": "",
            "race_class": "",
            "surface": "",
            "distance": 0,
            "track_condition": "",
            "weather": "",
            "num_entries": 0,
        }

        try:
            # レース名
            title_el = self.page.locator("h1, .racedata dt, dl.racedata h1").first
            if await title_el.count() > 0:
                title_text = await title_el.text_content()
                meta["race_name"] = title_text.strip() if title_text else ""

            # レース条件テキスト（距離・馬場・天候など）
            info_el = self.page.locator(".racedata, dl.racedata, .data_intro, span.smalltxt").first
            if await info_el.count() > 0:
                info_text = await info_el.text_content()
                if info_text:
                    info_text = info_text.strip()

                    # 距離・馬場 (例: "芝1600m" or "ダ1200m")
                    dist_match = re.search(r'([芝ダ障])(\d{3,4})m', info_text)
                    if dist_match:
                        meta["surface"] = dist_match.group(1)
                        meta["distance"] = int(dist_match.group(2))

                    # 天候
                    weather_match = re.search(r'天候\s*[:：]?\s*([^\s/]+)', info_text)
                    if weather_match:
                        meta["weather"] = weather_match.group(1)

                    # 馬場状態
                    cond_match = re.search(r'[芝ダート]*\s*[:：]?\s*(良|稍重|重|不良)', info_text)
                    if cond_match:
                        meta["track_condition"] = cond_match.group(1)

                    # 日付
                    date_match = re.search(r'(\d{4})年(\d{1,2})月(\d{1,2})日', info_text)
                    if date_match:
                        meta["race_date"] = f"{date_match.group(1)}-{int(date_match.group(2)):02d}-{int(date_match.group(3)):02d}"

                    # クラス
                    for cls in ["G1", "G2", "G3", "GI", "GII", "GIII",
                                "オープン", "OP", "3勝", "2勝", "1勝",
                                "未勝利", "新馬", "リステッド"]:
                        if cls in info_text:
                            meta["race_class"] = cls.replace("GI", "G1").replace("GII", "G2").replace("GIII", "G3")
                            break

            # 日付がまだ取得できていない場合、race_idから推定
            if not meta["race_date"]:
                year = race_id[:4]
                # ページ内の日付を別の場所から探す
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
        """ヘッダー文字列からカラムインデックスのマッピングを作成"""
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
            elif "性齢" in h or "性" in h and "齢" in h:
                hmap["sex_age"] = i
            elif "斤量" in h:
                hmap["weight_carried"] = i
            elif "騎手" in h:
                hmap["jockey"] = i
            elif "タイム" == h.strip():
                hmap["time"] = i
            elif "着差" in h:
                hmap["margin"] = i
            elif "通過" in h:
                hmap["passing"] = i
            elif "上" in h and ("り" in h or "がり" in h or "り" in h):
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

    def _parse_row(self, cell_texts: list, cells, header_map: dict, race_meta: dict) -> dict:
        """1行分のデータをパース"""
        row = {col: "" for col in CSV_COLUMNS}
        row.update(race_meta)

        def get(key):
            idx = header_map.get(key, -1)
            if 0 <= idx < len(cell_texts):
                return cell_texts[idx]
            return ""

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

        # 性齢
        sex_age_text = get("sex_age")
        sex, age = parse_sex_age(sex_age_text)
        row["sex"] = sex
        row["age"] = age

        # 馬体重
        hw_text = get("horse_weight")
        hw, hwc = parse_horse_weight(hw_text)
        row["horse_weight"] = hw
        row["horse_weight_change"] = hwc

        return row

    async def _extract_ids(self, cells, header_map: dict, horse_data: dict):
        """セル内リンクから馬ID・騎手IDを抽出"""
        try:
            # 馬ID
            name_idx = header_map.get("horse_name", -1)
            if 0 <= name_idx < len(cells):
                horse_links = await cells[name_idx].locator("a[href*='/horse/']").all()
                if horse_links:
                    href = await horse_links[0].get_attribute("href")
                    m = re.search(r'/horse/(\d{10})', href or "")
                    if m:
                        horse_data["horse_id"] = m.group(1)

            # 騎手ID
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
# CSV書き出し
# ============================================================

def save_results_csv(results: list, output_path: Path, append: bool = False):
    """結果をCSVに保存"""
    mode = "a" if append and output_path.exists() else "w"
    write_header = mode == "w" or not output_path.exists()

    with open(output_path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        for row in results:
            writer.writerow(row)


# ============================================================
# メイン処理
# ============================================================

async def collect_races(start_date: str, end_date: str, append: bool = False):
    """指定期間のレース結果を収集"""
    print()
    print("=" * 60)
    print(f"  過去レース結果収集ツール v{VERSION}")
    print("=" * 60)
    print(f"  期間: {start_date} ~ {end_date}")
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "results.csv"

    dates = generate_date_range(start_date, end_date)
    print(f"  対象日数: {len(dates)}日（土日のみ）")

    collector = RaceCollector(headless=True, debug=True)
    all_results = []
    total_races = 0

    try:
        await collector.start()

        for date_idx, date_str in enumerate(dates):
            print(f"\n[{date_idx+1}/{len(dates)}] {date_str}")

            # 20日ごとにブラウザ再起動
            if date_idx > 0 and date_idx % 20 == 0:
                print("  [メモリリフレッシュ]")
                await collector.restart()

            race_ids = await collector.get_race_ids_for_date(date_str)

            for race_id in race_ids:
                results = await collector.scrape_race_result(race_id)
                if results:
                    all_results.extend(results)
                    total_races += 1

                await asyncio.sleep(1.5)  # 負荷軽減

            # 日毎にCSVに書き出し（中断対策）
            if all_results:
                is_append = append or date_idx > 0
                save_results_csv(all_results, output_path, append=is_append)
                print(f"  → {len(all_results)}行保存済み (累計{total_races}レース)")
                all_results = []

            await asyncio.sleep(1.0)

    except KeyboardInterrupt:
        print("\n[!] 中断されました")
        if all_results:
            save_results_csv(all_results, output_path, append=True)
            print(f"  → 中断前データ{len(all_results)}行保存")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        await collector.close()

    print(f"\n{'=' * 60}")
    print(f"  [OK] 完了: {total_races}レース")
    print(f"  出力: {output_path}")
    print(f"{'=' * 60}")


def main():
    parser = ArgumentParser(description="netkeiba過去レース結果収集ツール")
    parser.add_argument("--start", help="開始日 (YYYY-MM-DD)")
    parser.add_argument("--end", help="終了日 (YYYY-MM-DD)")
    parser.add_argument("--date", help="特定日のみ (YYYY-MM-DD)")
    parser.add_argument("--append", action="store_true", help="既存CSVに追記")
    args = parser.parse_args()

    if args.date:
        start_date = end_date = args.date
    elif args.start and args.end:
        start_date, end_date = args.start, args.end
    else:
        print("使い方:")
        print("  python race_collector.py --start 2024-01-01 --end 2024-12-31")
        print("  python race_collector.py --date 2024-06-01")
        sys.exit(1)

    asyncio.run(collect_races(start_date, end_date, append=args.append))


if __name__ == "__main__":
    main()
