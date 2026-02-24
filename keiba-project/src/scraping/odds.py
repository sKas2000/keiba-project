"""
JRA オッズ取得
JRA公式サイトからオッズ・馬情報をスクレイピング
"""
import asyncio
import logging
import re
from datetime import datetime

from src.scraping.base import BaseScraper
from src.scraping.parsers import safe_float, parse_odds_range, parse_odds_range_low, extract_venue
from config.settings import GRADE_DEFAULTS

logger = logging.getLogger("keiba.scraping.odds")


class OddsScraper(BaseScraper):
    """JRA公式サイトのオッズスクレイパー"""

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
    # レース情報の自動取得
    # ----------------------------------------------------------

    async def scrape_race_info(self, meeting_text: str, race_num: int) -> dict:
        info = {
            "venue": extract_venue(meeting_text),
            "race_number": race_num,
            "name": "", "grade": "", "surface": "",
            "distance": 0, "direction": "", "post_time": "",
        }
        self.log("レース情報を自動取得中...")

        # SPAN.race_name
        try:
            el = self.page.locator("span.race_name").first
            text = await el.text_content()
            if text:
                info["name"] = text.strip()
                self.log(f"  レース名: {info['name']}")
        except Exception as e:
            logger.debug("レース名取得スキップ: %s", e)

        # DIV.race_title
        try:
            el = self.page.locator("div.race_title").first
            title = await el.text_content()
            if title:
                title = title.strip()
                self.log(f"  race_title: {title}")

                m = re.search(r"([\d,]+)\s*メートル", title)
                if m:
                    info["distance"] = int(m.group(1).replace(",", ""))

                m = re.search(r"（(芝|ダート)・(右|左)\s*(外|内)?）", title)
                if m:
                    info["surface"] = m.group(1)
                    info["direction"] = m.group(2) + (m.group(3) or "")

                grade_patterns = [
                    (r"GⅠ|（GⅠ）", "G1"), (r"GⅡ|（GⅡ）", "G2"),
                    (r"GⅢ|（GⅢ）", "G3"), (r"リステッド", "L"),
                    (r"オープン", "OP"), (r"3勝クラス", "3勝"),
                    (r"2勝クラス", "2勝"), (r"1勝クラス", "1勝"),
                    (r"未勝利", "未勝利"), (r"新馬", "新馬"),
                ]
                for pattern, label in grade_patterns:
                    if re.search(pattern, title):
                        info["grade"] = label
                        break
        except Exception as e:
            self.log(f"  race_title取得エラー: {e}")

        # 発走時刻（複数アプローチで検出）
        try:
            post_time = ""

            # Strategy 1: 特定CSSセレクタ（JRA odds page）
            for selector in [".race_time", ".post_time", ".hassou",
                             ".race_header_time", ".race_data time"]:
                try:
                    el = self.page.locator(selector).first
                    if await el.count() > 0:
                        text = await el.text_content()
                        if text:
                            tm = re.search(r"(\d{1,2}):(\d{2})", text)
                            if not tm:
                                tm = re.search(r"(\d{1,2})時(\d{2})分", text)
                            if tm:
                                post_time = f"{int(tm.group(1))}:{tm.group(2)}"
                                break
                except Exception as e:
                    logger.debug("発走時刻Strategy1 要素スキップ: %s", e)
                    continue

            # Strategy 2: race_title周辺のテキストから検出
            if not post_time:
                for selector in ["div.race_title", ".race_header",
                                 ".race_detail", "h1", "h2"]:
                    try:
                        el = self.page.locator(selector).first
                        if await el.count() > 0:
                            text = await el.text_content()
                            if text:
                                # 日本語形式: 10時00分
                                tm = re.search(
                                    r"(\d{1,2})時(\d{2})分", text)
                                if tm:
                                    post_time = f"{int(tm.group(1))}:{tm.group(2)}"
                                    break
                                # HH:MM形式
                                tm = re.search(r"(\d{1,2}:\d{2})", text)
                                if tm:
                                    h, m = tm.group(1).split(":")
                                    if 7 <= int(h) <= 17:
                                        post_time = tm.group(1)
                                        break
                    except Exception as e:
                        logger.debug("発走時刻Strategy2 要素スキップ: %s", e)
                        continue

            # Strategy 3: body全体から発走時刻パターン検索
            if not post_time:
                body_text = await self.page.locator("body").first.text_content()
                if body_text:
                    # 「発走 HH:MM」「HH:MM発走」
                    tm = re.search(r"発走[時刻]*\s*(\d{1,2}:\d{2})", body_text)
                    if not tm:
                        tm = re.search(r"(\d{1,2}:\d{2})\s*発走", body_text)
                    if tm:
                        post_time = tm.group(1)
                    # 日本語形式: 「発走 10時00分」
                    if not post_time:
                        tm = re.search(
                            r"発走[時刻]*\s*(\d{1,2})時(\d{2})分", body_text)
                        if not tm:
                            tm = re.search(
                                r"(\d{1,2})時(\d{2})分\s*発走", body_text)
                        if tm:
                            post_time = f"{int(tm.group(1))}:{tm.group(2)}"

            if post_time:
                info["post_time"] = post_time
                self.log(f"  発走時刻: {info['post_time']}")
        except Exception as e:
            logger.debug("発走時刻検出エラー: %s", e)

        # レース名フォールバック
        if not info["name"]:
            try:
                h2s = await self.page.locator("h2").all()
                for h2 in h2s:
                    text = await h2.text_content()
                    if text and any(kw in text for kw in
                                    ["特別", "ステークス", "賞", "カップ",
                                     "未勝利", "新馬", "1勝", "2勝", "3勝"]):
                        info["name"] = text.strip()
                        break
            except Exception as e:
                logger.debug("レース名フォールバックスキップ: %s", e)

        missing = [k for k in ["name", "surface", "distance", "grade"]
                   if not info.get(k)]
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
        for ti, table in enumerate(tables):
            ths = await table.locator("th").all()
            headers = []
            for th in ths:
                t = await th.text_content()
                headers.append(t.strip() if t else "")

            if not any("馬番" in h for h in headers):
                continue

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
                    "num": horse_num, "name": name,
                    "odds_win": odds_win, "odds_place": odds_place,
                    "sex_age": sex_age, "weight": weight,
                    "load_weight": load, "jockey": jockey,
                })

            if horses:
                break
        return horses

    # ----------------------------------------------------------
    # 三角行列パーサー（馬連・ワイド共通）
    # ----------------------------------------------------------

    async def _get_numeric_tables(self):
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
        results = []
        await self.click_tab(tab_name)
        num_tables = await self._get_numeric_tables()

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
    # 3連複
    # ----------------------------------------------------------

    async def parse_trio(self):
        results = []
        try:
            await self.click_tab("3連複")
            num_tables = await self._get_numeric_tables()
            n_tables = len(num_tables)
            if n_tables == 0:
                return results

            N = len(num_tables[0]["headers"]) + 2
            pairs = []
            for i in range(1, N):
                for j in range(i + 1, N):
                    pairs.append((i, j))

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

def build_input_json(scraped: dict, race_info: dict) -> dict:
    # --- データ検証ゲート ---
    # 必須フィールドの存在チェック（surface/distance が無いと予測不能）
    critical_missing = []
    if not race_info.get("surface"):
        critical_missing.append("surface")
    if not race_info.get("distance"):
        critical_missing.append("distance")
    if critical_missing:
        raise ValueError(
            f"レース必須情報が不足しています: {', '.join(critical_missing)}。"
            f" スクレイピング結果を確認してください。"
            f" (venue={race_info.get('venue')}, race_number={race_info.get('race_number')})"
        )

    # 非必須だが重要なフィールドの警告
    warn_missing = []
    if not race_info.get("name"):
        warn_missing.append("name")
    if not race_info.get("grade"):
        warn_missing.append("grade")
    if not race_info.get("direction"):
        warn_missing.append("direction")
    if warn_missing:
        logger.warning(
            "レース情報に不足あり (venue=%s, R%s): %s",
            race_info.get("venue", "?"), race_info.get("race_number", "?"),
            ", ".join(warn_missing),
        )

    # 出走馬データの検証
    raw_horses = scraped.get("horses", [])
    if not raw_horses:
        raise ValueError("出走馬データが空です。オッズページの解析に失敗した可能性があります。")

    horses = []
    for h in raw_horses:
        horses.append({
            "num": h["num"], "name": h["name"],
            "score": 0,
            "score_breakdown": {
                "ability": 0, "jockey": 0, "fitness": 0,
                "form": 0, "other": 0,
            },
            "odds_win": h["odds_win"], "odds_place": h["odds_place"],
            "jockey": h.get("jockey", ""), "sex_age": h.get("sex_age", ""),
            "weight": h.get("weight", ""), "load_weight": h.get("load_weight", 0),
            "note": "",
        })

    grade = race_info.get("grade", "")
    temp, budget = GRADE_DEFAULTS.get(grade, (10, 1500))

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
            "weather": "", "track_condition": "",
        },
        "parameters": {"temperature": temp, "budget": budget, "top_n": 6},
        "horses": horses,
        "combo_odds": {
            "quinella": scraped.get("quinella", []),
            "wide": scraped.get("wide", []),
            "trio": scraped.get("trio", []),
        },
    }
