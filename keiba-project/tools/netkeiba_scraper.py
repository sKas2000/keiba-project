#!/usr/bin/env python3
"""
netkeiba.com スクレイパー v0.1
=================================
JRA scraperで取得したinput.jsonに、過去走データ・騎手成績を追加する

必要なデータ:
- 過去走の着順、着差、上がり3F、タイム
- 騎手の年間勝率、複勝率
- 同コース・同距離の成績（優先度中、将来実装）

使い方:
  python netkeiba_scraper.py <input.jsonのパス>
"""

VERSION = "0.1"

import asyncio
import json
import re
import sys
from pathlib import Path
from datetime import datetime

try:
    from playwright.async_api import async_playwright
except ImportError:
    print("Playwright が未インストールです:")
    print("  pip install playwright")
    print("  playwright install chromium")
    sys.exit(1)


# ============================================================
# ユーティリティ
# ============================================================

def safe_int(text: str) -> int:
    """文字列を整数に変換。失敗時は0"""
    try:
        return int(re.sub(r'[^\d]', '', text))
    except (ValueError, AttributeError):
        return 0


def safe_float(text: str) -> float:
    """文字列を浮動小数点に変換。失敗時は0.0"""
    try:
        return float(text.strip().replace(",", ""))
    except (ValueError, AttributeError):
        return 0.0


# ============================================================
# netkeibaスクレイパー本体
# ============================================================

class NetkeibaScraper:
    def __init__(self, headless=False, debug=True):
        self.headless = headless
        self.debug = debug
        self.pw = None
        self.browser = None
        self.ctx = None
        self.page = None

        # キャッシュ（同じ騎手を複数回検索しないため）
        self.jockey_cache = {}

    def log(self, msg):
        if self.debug:
            print(f"  [DEBUG] {msg}")

    async def start(self):
        """ブラウザ起動"""
        self.pw = await async_playwright().start()
        self.browser = await self.pw.chromium.launch(headless=self.headless)
        self.ctx = await self.browser.new_context(
            viewport={"width": 1280, "height": 900},
            locale="ja-JP",
        )
        self.page = await self.ctx.new_page()
        self.page.set_default_timeout(15000)

    async def close(self):
        """ブラウザ終了"""
        if self.ctx:
            await self.ctx.close()
        if self.browser:
            await self.browser.close()
        if self.pw:
            await self.pw.stop()

    # ----------------------------------------------------------
    # 馬名検索 → 馬詳細ページ取得
    # ----------------------------------------------------------

    async def search_horse(self, horse_name: str) -> str:
        """
        netkeibaで馬名検索し、馬詳細ページのURLを返す
        Returns: 馬詳細ページURL or ""
        """
        self.log(f"馬名検索: {horse_name}")

        try:
            # netkeiba検索ページ
            search_url = f"https://db.netkeiba.com/?pid=horse_search_detail&word={horse_name}"
            await self.page.goto(search_url, wait_until="domcontentloaded")
            await asyncio.sleep(1)

            # 検索結果から最初の馬リンクを取得
            # netkeibaの馬リンクは /horse/ を含む
            links = await self.page.locator("a[href*='/horse/']").all()

            if not links:
                self.log(f"  ⚠️ 馬が見つかりません: {horse_name}")
                return ""

            # 最初の馬詳細リンクを取得
            first_link = links[0]
            href = await first_link.get_attribute("href")

            if href:
                # 相対URLを絶対URLに変換
                if href.startswith("/"):
                    href = f"https://db.netkeiba.com{href}"
                self.log(f"  ✅ 馬ページ: {href}")
                return href

        except Exception as e:
            self.log(f"  ❌ 馬検索エラー ({horse_name}): {e}")

        return ""

    # ----------------------------------------------------------
    # 過去走データのパース
    # ----------------------------------------------------------

    async def scrape_horse_past_races(self, horse_url: str) -> list:
        """
        馬詳細ページから過去走データを取得
        Returns: [{"date": "", "venue": "", "race": "", "distance": "", "surface": "",
                   "finish": 1, "margin": "0.2", "time": "1:23.4", "last3f": "34.5"}, ...]
        """
        self.log(f"過去走データ取得中: {horse_url}")

        past_races = []

        try:
            await self.page.goto(horse_url, wait_until="domcontentloaded")
            await asyncio.sleep(1.5)

            # netkeibaの競走成績テーブルを取得
            # db_h_race_results クラスを持つテーブル、または"着順"を含むテーブル
            tables = await self.page.locator("table").all()

            result_table = None
            for table in tables:
                # テーブルのヘッダーをチェック
                ths = await table.locator("th").all()
                headers = []
                for th in ths:
                    text = await th.text_content()
                    headers.append(text.strip() if text else "")

                # "着順" を含むテーブルが競走成績
                if any("着順" in h for h in headers):
                    result_table = table
                    self.log(f"  ★ 競走成績テーブル発見 (列数: {len(headers)})")

                    # ヘッダー位置を記録
                    header_map = {}
                    for i, h in enumerate(headers):
                        if "日付" in h:
                            header_map["date"] = i
                        elif "開催" in h or "競馬場" in h:
                            header_map["venue"] = i
                        elif "レース名" in h:
                            header_map["race"] = i
                        elif "距離" in h:
                            header_map["distance"] = i
                        elif "馬場" in h:
                            header_map["surface"] = i
                        elif "着順" in h:
                            header_map["finish"] = i
                        elif "タイム" in h:
                            header_map["time"] = i
                        elif "着差" in h:
                            header_map["margin"] = i
                        elif "上がり" in h or "上り" in h:
                            header_map["last3f"] = i
                        elif "通過" in h:
                            header_map["position"] = i

                    self.log(f"  ヘッダーマップ: {header_map}")
                    break

            if not result_table:
                self.log(f"  ⚠️ 競走成績テーブルが見つかりません")
                return past_races

            # データ行を取得（最新4走）
            rows = await result_table.locator("tbody tr").all()
            if not rows:
                rows = await result_table.locator("tr").all()
                # ヘッダー行をスキップ
                rows = [r for r in rows if await r.locator("th").count() == 0]

            for row_idx, row in enumerate(rows[:4]):
                cells = await row.locator("td").all()

                if len(cells) < 5:  # 最低限のデータがない
                    continue

                # セルのテキストを取得
                cell_texts = []
                for cell in cells:
                    text = await cell.text_content()
                    cell_texts.append(text.strip() if text else "")

                race_data = {
                    "date": "",
                    "venue": "",
                    "race": "",
                    "distance": "",
                    "surface": "",
                    "finish": 0,
                    "margin": "",
                    "time": "",
                    "last3f": "",
                    "position": "",
                }

                # ヘッダーマップに基づいてデータ抽出
                if "date" in header_map and header_map["date"] < len(cell_texts):
                    race_data["date"] = cell_texts[header_map["date"]]

                if "venue" in header_map and header_map["venue"] < len(cell_texts):
                    venue_text = cell_texts[header_map["venue"]]
                    # "3回東京1日" → "東京" を抽出
                    venue_match = re.search(r'\d+回(.+?)\d+日', venue_text)
                    race_data["venue"] = venue_match.group(1) if venue_match else venue_text

                if "race" in header_map and header_map["race"] < len(cell_texts):
                    race_data["race"] = cell_texts[header_map["race"]]

                if "distance" in header_map and header_map["distance"] < len(cell_texts):
                    dist_text = cell_texts[header_map["distance"]]
                    # "ダ1200" or "芝1600" → 馬場とdistanceを分離
                    dist_match = re.match(r'([芝ダ障])(\d+)', dist_text)
                    if dist_match:
                        race_data["surface"] = dist_match.group(1)
                        race_data["distance"] = dist_match.group(2)
                    else:
                        race_data["distance"] = re.sub(r'[^\d]', '', dist_text)

                if "surface" in header_map and header_map["surface"] < len(cell_texts):
                    if not race_data["surface"]:  # distanceから取得していない場合
                        race_data["surface"] = cell_texts[header_map["surface"]]

                if "finish" in header_map and header_map["finish"] < len(cell_texts):
                    finish_text = cell_texts[header_map["finish"]]
                    race_data["finish"] = safe_int(finish_text)

                if "time" in header_map and header_map["time"] < len(cell_texts):
                    race_data["time"] = cell_texts[header_map["time"]]

                if "margin" in header_map and header_map["margin"] < len(cell_texts):
                    race_data["margin"] = cell_texts[header_map["margin"]]

                if "last3f" in header_map and header_map["last3f"] < len(cell_texts):
                    race_data["last3f"] = cell_texts[header_map["last3f"]]

                if "position" in header_map and header_map["position"] < len(cell_texts):
                    race_data["position"] = cell_texts[header_map["position"]]

                # 着順が取得できた場合のみ追加
                if race_data["finish"] > 0 or race_data["date"]:
                    past_races.append(race_data)
                    self.log(f"    [{row_idx+1}] {race_data['date']} {race_data['venue']} {race_data['finish']}着")

            self.log(f"  → 過去走 {len(past_races)}件取得")

        except Exception as e:
            self.log(f"  ❌ 過去走取得エラー: {e}")
            import traceback
            traceback.print_exc()

        return past_races

    # ----------------------------------------------------------
    # 騎手検索 → 年間成績取得
    # ----------------------------------------------------------

    async def search_jockey(self, jockey_name: str) -> dict:
        """
        騎手名で検索し、年間成績を取得
        Returns: {"win_rate": 0.15, "place_rate": 0.35, "wins": 50, "races": 300}
        """
        # キャッシュチェック
        if jockey_name in self.jockey_cache:
            self.log(f"騎手キャッシュ: {jockey_name}")
            return self.jockey_cache[jockey_name]

        self.log(f"騎手検索: {jockey_name}")

        result = {"win_rate": 0.0, "place_rate": 0.0, "wins": 0, "races": 0}

        try:
            # netkeiba騎手検索
            search_url = f"https://db.netkeiba.com/?pid=jockey_search_detail&word={jockey_name}"
            await self.page.goto(search_url, wait_until="domcontentloaded")
            await asyncio.sleep(1)

            # 騎手詳細ページへのリンクを取得
            links = await self.page.locator("a[href*='/jockey/']").all()

            if not links:
                self.log(f"  ⚠️ 騎手が見つかりません: {jockey_name}")
                self.jockey_cache[jockey_name] = result
                return result

            # 最初の騎手リンクを取得
            first_link = links[0]
            href = await first_link.get_attribute("href")

            if href:
                if href.startswith("/"):
                    href = f"https://db.netkeiba.com{href}"

                self.log(f"  → 騎手ページ: {href}")

                # 騎手ページから成績を取得
                await self.page.goto(href, wait_until="domcontentloaded")
                await asyncio.sleep(1.5)

                # 年間成績を探す
                # 方法1: テーブルから勝率・複勝率を直接取得
                tables = await self.page.locator("table").all()

                for table in tables:
                    text = await table.text_content()

                    # 本年成績や通算成績のセクションを探す
                    if "本年" in text or "2026年" in text or "成績" in text:
                        # 勝率の抽出（例: "勝率 0.150" or "15.0%"）
                        win_match = re.search(r'勝率[^\d]*([\d.]+)', text)
                        if win_match:
                            win_val = safe_float(win_match.group(1))
                            # パーセント表記（15.0）か小数表記（0.15）か判定
                            result["win_rate"] = win_val / 100 if win_val > 1 else win_val

                        # 複勝率の抽出
                        place_match = re.search(r'複勝率[^\d]*([\d.]+)', text)
                        if place_match:
                            place_val = safe_float(place_match.group(1))
                            result["place_rate"] = place_val / 100 if place_val > 1 else place_val

                        # 勝利数と出走数
                        wins_match = re.search(r'(\d+)\s*勝', text)
                        if wins_match:
                            result["wins"] = safe_int(wins_match.group(1))

                        races_match = re.search(r'(\d+)\s*戦', text)
                        if races_match:
                            result["races"] = safe_int(races_match.group(1))

                        if result["win_rate"] > 0:
                            break

                # 方法2: 勝率が取得できなかった場合、ページ全体から探す
                if result["win_rate"] == 0:
                    page_text = await self.page.text_content("body")

                    # 全体から勝率を検索
                    win_match = re.search(r'勝率[^\d]*([\d.]+)', page_text)
                    if win_match:
                        win_val = safe_float(win_match.group(1))
                        result["win_rate"] = win_val / 100 if win_val > 1 else win_val

                    place_match = re.search(r'複勝率[^\d]*([\d.]+)', page_text)
                    if place_match:
                        place_val = safe_float(place_match.group(1))
                        result["place_rate"] = place_val / 100 if place_val > 1 else place_val

                if result["win_rate"] > 0 or result["place_rate"] > 0:
                    self.log(f"  ✅ 勝率{result['win_rate']:.3f} 複勝率{result['place_rate']:.3f}")
                else:
                    self.log(f"  ⚠️ 成績データが見つかりません")

        except Exception as e:
            self.log(f"  ❌ 騎手検索エラー ({jockey_name}): {e}")
            import traceback
            traceback.print_exc()

        # キャッシュに保存
        self.jockey_cache[jockey_name] = result
        return result


# ============================================================
# メイン処理
# ============================================================

async def enrich_race_data(input_path: str):
    """
    input.jsonを読み込み、netkeibaから追加データを取得してenriched_input.jsonに保存
    """
    print()
    print("=" * 60)
    print(f"  🐴 netkeiba スクレイパー v{VERSION}")
    print("=" * 60)
    print()

    # input.json読み込み
    input_file = Path(input_path)
    if not input_file.exists():
        print(f"❌ ファイルが見つかりません: {input_path}")
        return

    print(f"📂 入力: {input_file}")

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    horses = data.get("horses", [])
    print(f"🐴 対象馬: {len(horses)}頭")

    scraper = NetkeibaScraper(headless=False, debug=True)

    try:
        await scraper.start()

        # 各馬のデータを拡張
        for i, horse in enumerate(horses):
            horse_name = horse.get("name", "")
            print(f"\n[{i+1}/{len(horses)}] {horse_name}")

            if not horse_name:
                print("  ⚠️ 馬名が空です。スキップします。")
                horse["past_races"] = []
                horse["jockey_stats"] = {"win_rate": 0.0, "place_rate": 0.0, "wins": 0, "races": 0}
                continue

            # 馬詳細ページ取得
            horse_url = await scraper.search_horse(horse_name)

            if horse_url:
                # 過去走データ取得
                past_races = await scraper.scrape_horse_past_races(horse_url)
                horse["past_races"] = past_races
            else:
                horse["past_races"] = []
                print(f"  ⚠️ 過去走データを取得できませんでした")

            # 騎手成績取得
            jockey_name = horse.get("jockey", "")
            if jockey_name:
                print(f"  騎手: {jockey_name}")
                jockey_stats = await scraper.search_jockey(jockey_name)
                horse["jockey_stats"] = jockey_stats
            else:
                print(f"  ⚠️ 騎手情報がありません（input.jsonに'jockey'フィールドが必要）")
                horse["jockey_stats"] = {"win_rate": 0.0, "place_rate": 0.0, "wins": 0, "races": 0}

            await asyncio.sleep(0.5)  # 負荷軽減

        # 保存
        output_file = input_file.parent / input_file.name.replace("_input.json", "_enriched_input.json")

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"\n{'=' * 60}")
        print(f"  ✅ 保存完了: {output_file}")
        print(f"{'=' * 60}")

    except KeyboardInterrupt:
        print("\n⚠️ 中断")
    except Exception as e:
        print(f"\n❌ エラー: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await scraper.close()


async def main():
    if len(sys.argv) < 2:
        print("使い方: python netkeiba_scraper.py <input.jsonのパス>")
        print("例: python netkeiba_scraper.py ../data/races/20260214_東京_4R_input.json")
        sys.exit(1)

    input_path = sys.argv[1]
    await enrich_race_data(input_path)


if __name__ == "__main__":
    asyncio.run(main())
