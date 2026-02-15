#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JRA公式サイト スクレイパー v1.4.1
===================================
v1.2: 単勝・複勝・馬連・ワイド取得成功
v1.3: レース情報自動取得（距離・グレード一部失敗）
v1.4: レース情報を DIV.race_title から正確に取得
      3連複（二重三角行列）パース追加
      馬単（正方行列）パース追加
v1.4.1: 馬単削除（対象券種: 単勝・複勝・馬連・ワイド・3連複のみ）

使い方:
  python jra_scraper.py
"""

VERSION = "1.4.1"

import asyncio
import json
import re
import sys
from datetime import datetime
from pathlib import Path

# Windows環境での文字化け対策
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

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

def safe_float(text: str) -> float:
    try:
        return float(text.strip().replace(",", ""))
    except (ValueError, AttributeError):
        return 0.0


def parse_odds_range(text: str) -> float:
    """'2.4-6.1' → 中央値 4.25"""
    text = text.strip()
    m = re.match(r"([\d.]+)\s*[-～]\s*([\d.]+)", text)
    if m:
        return round((float(m.group(1)) + float(m.group(2))) / 2, 1)
    return safe_float(text)


def parse_odds_range_low(text: str) -> float:
    """'2.4-6.1' → 下限 2.4"""
    text = text.strip()
    m = re.match(r"([\d.]+)\s*[-～]\s*([\d.]+)", text)
    if m:
        return float(m.group(1))
    return safe_float(text)


def extract_venue(meeting_text: str) -> str:
    """'1回東京5日' → '東京'"""
    m = re.search(r"\d+回(.+?)\d+日", meeting_text)
    return m.group(1) if m else ""


# ============================================================
# スクレイパー本体
# ============================================================

class JRAScraper:
    def __init__(self, headless=False, debug=True):
        self.headless = headless
        self.debug = debug
        self.pw = None
        self.browser = None
        self.ctx = None
        self.page = None

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

    # ----------------------------------------------------------
    # ナビゲーション
    # ----------------------------------------------------------

    async def goto_odds_top(self):
        await self.page.goto("https://www.jra.go.jp/", wait_until="domcontentloaded")
        await asyncio.sleep(1)
        await self.page.get_by_role("link", name="オッズ").first.click()
        await self.page.wait_for_load_state("domcontentloaded")
        await asyncio.sleep(1)

        meetings = []
        links = await self.page.locator("a").all()
        for link in links:
            text = await link.text_content()
            if text and "回" in text and "日" in text:
                meetings.append({"text": text.strip(), "element": link})
        return meetings

    async def select_meeting(self, el):
        await el.click()
        await self.page.wait_for_load_state("domcontentloaded")
        await asyncio.sleep(1)

    async def select_race(self, num):
        await self.page.get_by_role("link", name=f"{num}レース").first.click()
        await self.page.wait_for_load_state("domcontentloaded")
        await asyncio.sleep(1)

    async def click_tab(self, name):
        await self.page.get_by_role("link", name=name).first.click()
        await self.page.wait_for_load_state("domcontentloaded")
        await asyncio.sleep(1)

    # ----------------------------------------------------------
    # レース情報の自動取得（v1.4: DIV.race_title から正確に取得）
    # ----------------------------------------------------------

    async def scrape_race_info(self, meeting_text: str, race_num: int) -> dict:
        info = {
            "venue": extract_venue(meeting_text),
            "race_number": race_num,
            "name": "",
            "grade": "",
            "surface": "",
            "distance": 0,
            "direction": "",
        }

        self.log("レース情報を自動取得中...")

        # 1) SPAN.race_name → レース名
        try:
            el = self.page.locator("span.race_name").first
            text = await el.text_content()
            if text:
                info["name"] = text.strip()
                self.log(f"  レース名: {info['name']}")
        except Exception:
            pass

        # 2) DIV.race_title → 距離・馬場・回り・グレード
        try:
            el = self.page.locator("div.race_title").first
            title = await el.text_content()
            if title:
                title = title.strip()
                self.log(f"  race_title: {title}")

                # 距離: "1,400メートル" or "2000メートル"
                m = re.search(r"([\d,]+)\s*メートル", title)
                if m:
                    info["distance"] = int(m.group(1).replace(",", ""))
                    self.log(f"  距離: {info['distance']}m")

                # 馬場・回り: "（ダート・左）" or "（芝・右 外）"
                m = re.search(r"（(芝|ダート)・(右|左)\s*(外|内)?）", title)
                if m:
                    info["surface"] = m.group(1)
                    info["direction"] = m.group(2) + (m.group(3) or "")
                    self.log(f"  馬場: {info['surface']} {info['direction']}")

                # グレード（race_title 内に限定して検索）
                grade_patterns = [
                    (r"GⅠ|（GⅠ）", "G1"),
                    (r"GⅡ|（GⅡ）", "G2"),
                    (r"GⅢ|（GⅢ）", "G3"),
                    (r"リステッド", "L"),
                    (r"オープン", "OP"),
                    (r"3勝クラス", "3勝"),
                    (r"2勝クラス", "2勝"),
                    (r"1勝クラス", "1勝"),
                    (r"未勝利", "未勝利"),
                    (r"新馬", "新馬"),
                ]
                for pattern, label in grade_patterns:
                    if re.search(pattern, title):
                        info["grade"] = label
                        self.log(f"  グレード: {label}")
                        break
        except Exception as e:
            self.log(f"  race_title取得エラー: {e}")

        # 3) レース名フォールバック（h2から）
        if not info["name"]:
            try:
                h2s = await self.page.locator("h2").all()
                for h2 in h2s:
                    text = await h2.text_content()
                    if text and any(kw in text for kw in
                                    ["特別", "ステークス", "賞", "カップ",
                                     "未勝利", "新馬", "1勝", "2勝", "3勝"]):
                        info["name"] = text.strip()
                        self.log(f"  レース名(h2): {info['name']}")
                        break
            except Exception:
                pass

        # 未検出サマリー
        missing = []
        if not info["name"]:
            missing.append("レース名")
        if not info["surface"]:
            missing.append("馬場")
        if not info["distance"]:
            missing.append("距離")
        if not info["grade"]:
            missing.append("グレード")
        if missing:
            self.log(f"  [!] 未検出: {', '.join(missing)}")
        else:
            self.log("  [OK] 全項目取得成功")

        return info

    # ----------------------------------------------------------
    # 単勝・複勝
    # ----------------------------------------------------------

    async def parse_win_place(self):
        horses = []
        await self.click_tab("単勝・複勝")

        tables = await self.page.locator("table").all()
        self.log(f"単勝・複勝テーブル数: {len(tables)}")

        for ti, table in enumerate(tables):
            ths = await table.locator("th").all()
            headers = []
            for th in ths:
                t = await th.text_content()
                headers.append(t.strip() if t else "")

            if not any("馬番" in h for h in headers):
                continue

            self.log(f"  * 馬番テーブル発見 (テーブル{ti})")

            rows = await table.locator("tbody tr").all()
            if not rows:
                rows = await table.locator("tr").all()

            for row in rows:
                cells = await row.locator("td").all()
                if len(cells) < 4:
                    continue

                ct = []
                for cell in cells:
                    t = await cell.text_content()
                    ct.append(t.strip() if t else "")

                # 馬番セルを探す
                num_idx = -1
                for i, t in enumerate(ct):
                    if re.match(r"^\d{1,2}$", t) and 1 <= int(t) <= 18:
                        num_idx = i
                        break

                if num_idx < 0:
                    continue

                horse_num = int(ct[num_idx])
                idx = num_idx + 1

                def get():
                    nonlocal idx
                    val = ct[idx] if idx < len(ct) else ""
                    idx += 1
                    return val

                name = get()
                odds_win = safe_float(get())
                odds_place_str = get()
                odds_place = parse_odds_range_low(odds_place_str)
                sex_age = get()
                weight = get()
                load = safe_float(get())
                jockey = get()
                jockey = re.sub(r"[▲△☆◇]", "", jockey).strip()

                horses.append({
                    "num": horse_num,
                    "name": name,
                    "odds_win": odds_win,
                    "odds_place": odds_place,
                    "sex_age": sex_age,
                    "weight": weight,
                    "load_weight": load,
                    "jockey": jockey,
                })

            if horses:
                break

        return horses

    # ----------------------------------------------------------
    # 三角行列パーサー（馬連・ワイド共通）
    # ----------------------------------------------------------

    async def _get_numeric_tables(self):
        """ヘッダーが全て数字のテーブルを順番に返す"""
        tables = await self.page.locator("table").all()
        result = []
        for table in tables:
            ths = await table.locator("th").all()
            headers = []
            for th in ths:
                t = await th.text_content()
                headers.append(t.strip() if t else "")
            if not headers:
                continue
            if all(re.match(r"^\d{1,2}$", h) for h in headers):
                # 行データ取得
                rows_data = []
                rows = await table.locator("tr").all()
                for row in rows:
                    cells = await row.locator("td").all()
                    if not cells:
                        continue
                    ct = []
                    for cell in cells:
                        t = await cell.text_content()
                        ct.append(t.strip() if t else "")
                    rows_data.append(ct)
                result.append({
                    "headers": [int(h) for h in headers],
                    "rows": rows_data,
                })
        return result

    async def parse_triangle_odds(self, tab_name, is_range=False):
        """馬連・ワイドの三角行列パース"""
        results = []
        await self.click_tab(tab_name)

        num_tables = await self._get_numeric_tables()
        self.log(f"{tab_name}: 数値テーブル {len(num_tables)}個")

        for tbl in num_tables:
            headers = tbl["headers"]
            base = headers[0] - 1
            if base < 1:
                continue

            for pi, row in enumerate(tbl["rows"]):
                if pi >= len(headers):
                    break
                partner = headers[pi]
                if not row or not row[0]:
                    continue
                odds = parse_odds_range(row[0]) if is_range else safe_float(row[0])
                if odds > 0:
                    results.append({
                        "combo": sorted([base, partner]),
                        "odds": odds,
                    })

        return results

    # ----------------------------------------------------------
    # 3連複パーサー（v1.4: 二重三角行列）
    # ----------------------------------------------------------

    async def parse_trio(self):
        """
        3連複の構造:
          N頭の場合 C(N-1,2) 個のテーブル。
          テーブル順序: (i,j) を i=1..N-1, j=i+1..N-1 で列挙。
          各テーブルのヘッダー = [j+1, j+2, ..., N] = 3頭目の馬番。
          各行のオッズ = 3連複 {i, j, k} の配当。
        """
        results = []
        try:
            await self.click_tab("3連複")

            num_tables = await self._get_numeric_tables()
            n_tables = len(num_tables)
            self.log(f"3連複: 数値テーブル {n_tables}個")

            if n_tables == 0:
                return results

            # 頭数Nを推定: 最初のテーブルのヘッダー数 = N-2
            N = len(num_tables[0]["headers"]) + 2
            expected = (N - 1) * (N - 2) // 2
            self.log(f"3連複: 推定頭数 N={N}, 期待テーブル数={expected}, 実際={n_tables}")

            # ペア(i,j)の順序を事前計算
            pairs = []
            for i in range(1, N):
                for j in range(i + 1, N):
                    pairs.append((i, j))

            # テーブルとペアを対応付け
            for t_idx, (i, j) in enumerate(pairs):
                if t_idx >= n_tables:
                    break

                tbl = num_tables[t_idx]
                headers = tbl["headers"]

                for r_idx, row in enumerate(tbl["rows"]):
                    if r_idx >= len(headers):
                        break
                    k = headers[r_idx]
                    if not row or not row[0]:
                        continue
                    odds = safe_float(row[0])
                    if odds > 0:
                        results.append({
                            "combo": sorted([i, j, k]),
                            "odds": odds,
                        })

            self.log(f"3連複: {len(results)}組取得")

        except Exception as e:
            self.log(f"3連複エラー: {e}")

        return results


# ============================================================
# input.json 生成
# ============================================================

def build_json(scraped, race_info):
    horses = []
    for h in scraped.get("horses", []):
        horses.append({
            "num": h["num"],
            "name": h["name"],
            "score": 0,
            "score_breakdown": {
                "ability": 0, "jockey": 0, "fitness": 0,
                "form": 0, "other": 0,
            },
            "odds_win": h["odds_win"],
            "odds_place": h["odds_place"],
            "jockey": h.get("jockey", ""),
            "sex_age": h.get("sex_age", ""),
            "weight": h.get("weight", ""),
            "load_weight": h.get("load_weight", 0),
            "note": "",
        })

    grade = race_info.get("grade", "")
    defaults = {
        "G1": (8, 10000), "G2": (8, 5000), "G3": (10, 3000),
        "L": (10, 3000), "OP": (10, 3000),
        "3勝": (10, 1500), "2勝": (10, 1500),
        "1勝": (12, 1500), "未勝利": (12, 1500), "新馬": (14, 1500),
    }
    temp, budget = defaults.get(grade, (10, 1500))

    return {
        "race": {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "venue": race_info.get("venue", ""),
            "race_number": race_info.get("race_number", 0),
            "name": race_info.get("name", ""),
            "grade": grade,
            "surface": race_info.get("surface", ""),
            "distance": race_info.get("distance", 0),
            "direction": race_info.get("direction", ""),
            "entries": len(horses),
            "weather": "",
            "track_condition": "",
        },
        "parameters": {
            "temperature": temp,
            "budget": budget,
            "top_n": 6,
        },
        "horses": horses,
        "combo_odds": {
            "quinella": scraped.get("quinella", []),
            "wide": scraped.get("wide", []),
            "trio": scraped.get("trio", []),
        },
    }


# ============================================================
# メイン
# ============================================================

async def main():
    print()
    print("=" * 60)
    print(f"  JRA スクレイパー v{VERSION}")
    print("=" * 60)
    print()

    scraper = JRAScraper(headless=False, debug=True)

    try:
        await scraper.start()

        # --- 開催選択 ---
        print("JRA公式サイトに接続中...")
        meetings = await scraper.goto_odds_top()

        if not meetings:
            print("[ERROR] 開催情報が見つかりません")
            return

        print(f"\n開催一覧 ({len(meetings)}件):")
        for i, m in enumerate(meetings):
            print(f"  [{i+1}] {m['text']}")

        while True:
            c = input(f"\n開催を選択 (1-{len(meetings)}): ").strip()
            if c.isdigit() and 1 <= int(c) <= len(meetings):
                mi = int(c) - 1
                break
            print("  無効")

        meeting_text = meetings[mi]["text"]
        await scraper.select_meeting(meetings[mi]["element"])

        # --- レース選択 ---
        while True:
            c = input("レース番号 (1-12): ").strip()
            if c.isdigit() and 1 <= int(c) <= 12:
                rn = int(c)
                break
            print("  無効")

        print(f"\n{rn}R を選択中...")
        await scraper.select_race(rn)

        # === レース情報 ===
        print("\nレース情報を自動取得中...")
        race_info = await scraper.scrape_race_info(meeting_text, rn)

        venue = race_info["venue"]
        name = race_info["name"]
        surface = race_info["surface"]
        distance = race_info["distance"]
        direction = race_info["direction"]
        grade = race_info["grade"]

        print(f"\n自動取得結果:")
        print(f"  競馬場:   {venue or '（未検出）'}")
        print(f"  レース名: {name or '（未検出）'}")
        print(f"  馬場:     {surface or '（未検出）'} {direction}")
        print(f"  距離:     {distance or '（未検出）'}{'m' if distance else ''}")
        print(f"  グレード: {grade or '（未検出）'}")

        # 未検出項目のみ手動入力
        if not venue:
            venue = input("\n  → 競馬場を入力 (例: 東京): ").strip()
            race_info["venue"] = venue
        if not name:
            name = input(f"  → レース名 (例: {rn}R): ").strip() or f"{rn}R"
            race_info["name"] = name
        if not surface:
            surface = input("  → 馬場 (芝/ダート): ").strip()
            race_info["surface"] = surface
        if not distance:
            d = input("  → 距離 (例: 1400): ").strip()
            race_info["distance"] = int(d) if d.isdigit() else 0
        if not direction:
            direction = input("  → 回り (右/左): ").strip()
            race_info["direction"] = direction
        if not grade:
            grade = input("  → グレード (G1/G2/G3/OP/3勝/2勝/1勝/未勝利/新馬): ").strip()
            race_info["grade"] = grade

        confirm = input("\n[OK] 取得開始 (Enter / n でキャンセル): ").strip()
        if confirm.lower() == "n":
            print("キャンセル")
            return

        # === オッズ取得 ===
        print("\n単勝・複勝を取得中...")
        horses = await scraper.parse_win_place()
        print(f"  → {len(horses)}頭")
        for h in horses:
            print(f"    {h['num']:2d}番 {h['name']}  "
                  f"単勝{h['odds_win']:.1f}  複勝{h['odds_place']:.1f}  "
                  f"{h['jockey']}")

        print("\n馬連を取得中...")
        quinella = await scraper.parse_triangle_odds("馬連", is_range=False)
        print(f"  → {len(quinella)}組")

        print("\nワイドを取得中...")
        wide = await scraper.parse_triangle_odds("ワイド", is_range=True)
        print(f"  → {len(wide)}組")

        print("\n3連複を取得中...")
        trio = await scraper.parse_trio()
        print(f"  → {len(trio)}組")

        scraped = {
            "horses": horses,
            "quinella": quinella,
            "wide": wide,
            "trio": trio,
        }

        # === JSON保存 ===
        out = build_json(scraped, race_info)

        date_str = datetime.now().strftime("%Y%m%d")
        safe_name = re.sub(r'[\\/:*?"<>|\s]', '_', name or "不明")
        fname = f"{date_str}_{venue}{rn}R_{safe_name}_input.json"

        script_dir = Path(__file__).resolve().parent
        out_dir = script_dir.parent / "data" / "races"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / fname

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

        # === サマリー ===
        n_horses = len(horses)
        # 期待組数
        expected_q = n_horses * (n_horses - 1) // 2
        expected_t = (n_horses * (n_horses - 1) * (n_horses - 2)) // 6

        print(f"\n{'=' * 60}")
        print(f"  [OK] 保存完了: {out_path}")
        print(f"{'=' * 60}")
        print(f"\nサマリー:")
        print(f"  {venue}{rn}R {name}")
        print(f"  {surface}{direction} {race_info.get('distance', 0)}m  {grade}")
        print(f"  出走馬: {n_horses}頭")
        print(f"  馬連:   {len(quinella)}/{expected_q}組")
        print(f"  ワイド: {len(wide)}/{expected_q}組")
        print(f"  3連複:  {len(trio)}/{expected_t}組")
        print(f"\n次のステップ: netkeiba_scraper.pyで過去走データ追加")

    except KeyboardInterrupt:
        print("\n[!] 中断")
    except Exception as e:
        print(f"\n[ERROR] エラー: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await scraper.close()


if __name__ == "__main__":
    asyncio.run(main())
