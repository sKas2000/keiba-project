#!/usr/bin/env python3
"""
netkeiba.com スクレイパー v0.3
=================================
JRA scraperで取得したinput.jsonに、過去走データ・騎手成績を追加する

必要なデータ:
- 過去走の着順、着差、上がり3F、タイム
- 騎手の年間勝率、複勝率
- 同コース・同距離の成績（優先度中、将来実装）

v0.3の改善点（騎手検索の精度向上）:
- 過去走データから騎手IDを抽出
- 騎手検索を回避し、IDから直接成績ページにアクセス
- 騎手名の正規化による照合精度向上
- 全騎手が同じIDになる問題を解決

v0.2の改善点:
- headless=True（パフォーマンス向上）
- アクセス間隔を2秒に延長（負荷軽減）
- リトライメカニズムの実装（最大3回、指数バックオフ）
- ブラウザ再起動ロジック（5頭ごとにメモリリフレッシュ）
- 文字コード自動判別（UTF-8/EUC-JP/Shift-JISで順次試行）

使い方:
  python netkeiba_scraper.py <input.jsonのパス>
"""

VERSION = "0.5"

import asyncio
import json
import re
import sys
from pathlib import Path
from datetime import datetime

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


def encode_for_netkeiba(text: str) -> str:
    """
    netkeibaの検索用にテキストをURLエンコード
    netkeibaはEUC-JPを使用しているため、UTF-8ではなくEUC-JPでエンコード
    """
    from urllib.parse import quote
    try:
        # EUC-JPでエンコードしてからURLエンコード
        return quote(text.encode('euc-jp'), safe='')
    except UnicodeEncodeError:
        # EUC-JPで表現できない文字がある場合はUTF-8にフォールバック
        return quote(text)


def normalize_jockey_name(name: str) -> str:
    """
    騎手名を正規化（比較用）
    - 全角→半角変換（英数字・記号）
    - 空白・ドット・中黒を除去
    - 大文字→小文字
    """
    import unicodedata
    # 全角→半角
    normalized = unicodedata.normalize('NFKC', name)
    # 空白・ドット・中黒を除去
    normalized = normalized.replace(" ", "").replace("　", "").replace(".", "").replace("・", "").replace("．", "")
    # 小文字化
    normalized = normalized.lower()
    return normalized


# ============================================================
# netkeibaスクレイパー本体
# ============================================================

class NetkeibaScraper:
    def __init__(self, headless=True, debug=True):
        self.headless = headless
        self.debug = debug
        self.pw = None
        self.browser = None
        self.ctx = None
        self.page = None

        # キャッシュ（同じ騎手を複数回検索しないため）
        self.jockey_cache = {}

        # リトライ設定
        self.max_retries = 3
        self.retry_delay = 2.0

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

    async def restart(self):
        """ブラウザ再起動（メモリリフレッシュ）"""
        self.log("ブラウザ再起動中...")
        await self.close()
        await asyncio.sleep(2)
        await self.start()
        self.log("ブラウザ再起動完了")

    # ----------------------------------------------------------
    # 馬名検索 → 馬詳細ページ取得
    # ----------------------------------------------------------

    async def search_horse(self, horse_name: str) -> str:
        """
        netkeibaで馬名検索し、馬詳細ページのURLを返す
        Returns: 馬詳細ページURL or ""
        """
        self.log(f"馬名検索: {horse_name}")

        for attempt in range(self.max_retries):
            try:
                # 検索ページに移動
                await self.page.goto("https://db.netkeiba.com/?pid=horse_search_detail", wait_until="domcontentloaded", timeout=30000)
                await asyncio.sleep(2)

                # フォームに入力（inputフィールドを探す）
                # name="horse_name" または placeholder などで特定
                input_found = False

                # 方法1: name属性で検索
                try:
                    input_field = self.page.locator('input[name="horse_name"]')
                    if await input_field.count() > 0:
                        await input_field.fill(horse_name)
                        input_found = True
                        self.log(f"  フォーム入力 (name=horse_name): {horse_name}")
                except:
                    pass

                # 方法2: 最初のtext inputフィールドに入力
                if not input_found:
                    try:
                        text_inputs = await self.page.locator('input[type="text"]').all()
                        if text_inputs:
                            await text_inputs[0].fill(horse_name)
                            input_found = True
                            self.log(f"  フォーム入力 (first text input): {horse_name}")
                    except:
                        pass

                if not input_found:
                    self.log(f"  [!] 検索フォームが見つかりません")
                    return ""

                # 検索ボタンをクリック
                await asyncio.sleep(1)

                # submitボタンを探してクリック
                submit_clicked = False
                try:
                    submit_button = self.page.locator('input[type="submit"], button[type="submit"]')
                    if await submit_button.count() > 0:
                        await submit_button.first.click()
                        submit_clicked = True
                        self.log(f"  検索ボタンクリック")
                except:
                    pass

                # Enterキーでsubmit
                if not submit_clicked:
                    try:
                        await self.page.keyboard.press("Enter")
                        self.log(f"  Enterキー送信")
                    except:
                        pass

                # 検索結果ページの読み込み待ち
                await asyncio.sleep(3)

                # 検索結果から馬ページのリンクを取得
                links = await self.page.locator("table a[href*='/horse/']").all()
                if not links:
                    links = await self.page.locator("a[href*='/horse/']").all()

                if self.debug:
                    self.log(f"  検索結果リンク数: {len(links)}")

                # 有効な馬ページのURLを探す
                for link in links:
                    href = await link.get_attribute("href")
                    if href and "/horse/" in href:
                        # 不要なページを除外
                        if any(x in href for x in ["search_detail", "top.html", "sire/", "bms_", "leading"]):
                            continue
                        # 馬IDパターンをチェック（/horse/数字10桁/）
                        if re.search(r'/horse/\d{10}', href):
                            if href.startswith("/"):
                                href = f"https://db.netkeiba.com{href}"
                            self.log(f"  [OK] 馬ページ発見: {href}")
                            return href

                self.log(f"  [!] 有効な馬ページが見つかりません: {horse_name}")
                return ""

            except Exception as e:
                self.log(f"  [ERROR] 馬検索エラー (試行{attempt+1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    self.log(f"  → {delay}秒後にリトライします...")
                    await asyncio.sleep(delay)
                else:
                    self.log(f"  → リトライ上限に達しました。スキップします。")

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
                        elif "騎手" in h:
                            header_map["jockey"] = i
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
                self.log(f"  [!] 競走成績テーブルが見つかりません")
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
                    "jockey_name": "",
                    "jockey_id": "",
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

                # 騎手情報を抽出（名前とID）
                if "jockey" in header_map and header_map["jockey"] < len(cells):
                    jockey_cell = cells[header_map["jockey"]]
                    # 騎手名（テキスト）
                    jockey_text = await jockey_cell.text_content()
                    race_data["jockey_name"] = jockey_text.strip() if jockey_text else ""

                    # 騎手ID（リンクから抽出）
                    jockey_links = await jockey_cell.locator("a[href*='/jockey/']").all()
                    if jockey_links:
                        href = await jockey_links[0].get_attribute("href")
                        # /jockey/result/recent/05585/ or /jockey/05585 から ID抽出
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

                # 着順が取得できた場合のみ追加
                if race_data["finish"] > 0 or race_data["date"]:
                    past_races.append(race_data)
                    self.log(f"    [{row_idx+1}] {race_data['date']} {race_data['venue']} {race_data['finish']}着")

            self.log(f"  → 過去走 {len(past_races)}件取得")

        except Exception as e:
            self.log(f"  [ERROR] 過去走取得エラー: {e}")
            import traceback
            traceback.print_exc()

        return past_races

    # ----------------------------------------------------------
    # 過去走データから騎手IDを抽出
    # ----------------------------------------------------------

    def extract_jockey_id_from_past_races(self, jockey_name: str, past_races: list) -> str:
        """
        過去走データから騎手名に一致する騎手IDを抽出
        Args:
            jockey_name: 検索する騎手名（例: "C.ルメール", "北村 友一"）
            past_races: scrape_horse_past_racesで取得した過去走データ
        Returns:
            騎手ID（5桁の数字）or ""
        """
        if not past_races:
            return ""

        normalized_search = normalize_jockey_name(jockey_name)
        self.log(f"過去走から騎手ID検索: {jockey_name} (正規化: {normalized_search})")

        # デバッグ: 過去走の騎手情報を表示
        for idx, race in enumerate(past_races):
            race_jockey_name = race.get("jockey_name", "")
            race_jockey_id = race.get("jockey_id", "")
            self.log(f"  過去走[{idx+1}] 騎手: '{race_jockey_name}' ID: '{race_jockey_id}'")

        for race in past_races:
            race_jockey_name = race.get("jockey_name", "")
            race_jockey_id = race.get("jockey_id", "")

            if race_jockey_name and race_jockey_id:
                normalized_race = normalize_jockey_name(race_jockey_name)

                if normalized_race == normalized_search:
                    self.log(f"  [OK] 過去走で騎手発見: {race_jockey_name} (ID: {race_jockey_id})")
                    return race_jockey_id

        self.log(f"  [!] 過去走に該当騎手なし")
        return ""

    # ----------------------------------------------------------
    # 騎手成績パース（共通処理）
    # ----------------------------------------------------------

    async def _parse_jockey_stats_from_page(self) -> dict:
        """
        現在表示中の騎手ページからテーブルセルを直接解析して成績を取得。
        regex方式ではなく、ヘッダーのカラムインデックスを特定して
        対応するセルを読む方式。

        テーブル構造（netkeiba騎手ページ）:
          ヘッダー: 年度, 順位, 1着, 2着, 3着, 4着〜, 騎乗回数, ..., 勝率, 連対率, 複勝率, 代表馬
          累計行:   累計, ,     2134, ...,                              22.4％, 38.4％, 50.1％, ...
          年度行:   2026, 1,    20, ...,                                30.3％, 50.0％, 65.2％, ...

        Returns:
            {"win_rate": 0.303, "place_rate": 0.652, "wins": 20, "races": 66}
        """
        result = {"win_rate": 0.0, "place_rate": 0.0, "wins": 0, "races": 0}
        found_2026 = False

        tables = await self.page.locator("table").all()

        for table in tables:
            # ヘッダー(th)を取得
            ths = await table.locator("th").all()
            if not ths:
                continue

            header_texts = []
            for th in ths:
                t = await th.text_content()
                header_texts.append(t.strip() if t else "")

            # 「勝率」と「騎乗回数」を含むテーブルを探す（JRA平地成績）
            win_rate_idx = -1
            place_rate_idx = -1
            wins_idx = -1  # 1着
            races_idx = -1  # 騎乗回数

            for i, ht in enumerate(header_texts):
                if ht == "勝率":
                    win_rate_idx = i
                elif ht == "複勝率":
                    place_rate_idx = i
                elif ht == "1着":
                    wins_idx = i
                elif "騎乗回数" in ht or ht == "回数":
                    races_idx = i

            if win_rate_idx == -1:
                continue

            # 騎乗回数があるテーブル = JRA平地成績（最初のテーブル）
            if races_idx == -1:
                continue

            # データ行を取得
            rows = await table.locator("tr").all()

            for row in rows:
                cells = await row.locator("td").all()
                if not cells or len(cells) <= win_rate_idx:
                    continue

                # 年度セルを確認（「2026」の行を優先）
                first_cell_text = await cells[0].text_content()
                first_cell_text = first_cell_text.strip() if first_cell_text else ""

                # 2026年の行を探す
                if first_cell_text == "2026":
                    found_2026 = True

                    # 勝率（例: "7.0％" → 0.070、"0.0％" → 0.0）
                    win_text = await cells[win_rate_idx].text_content()
                    win_text = win_text.strip().replace("％", "").replace("%", "") if win_text else "0"
                    result["win_rate"] = safe_float(win_text) / 100.0

                    # 複勝率
                    if place_rate_idx >= 0 and len(cells) > place_rate_idx:
                        place_text = await cells[place_rate_idx].text_content()
                        place_text = place_text.strip().replace("％", "").replace("%", "") if place_text else "0"
                        result["place_rate"] = safe_float(place_text) / 100.0

                    # 1着数
                    if wins_idx >= 0 and len(cells) > wins_idx:
                        wins_text = await cells[wins_idx].text_content()
                        result["wins"] = safe_int(wins_text.strip() if wins_text else "0")

                    # 騎乗回数
                    if races_idx >= 0 and len(cells) > races_idx:
                        races_text = await cells[races_idx].text_content()
                        races_text = races_text.strip().replace(",", "") if races_text else "0"
                        result["races"] = safe_int(races_text)

                    break

            # 2026年の行が見つかったら終了（0勝でもデータあり）
            if found_2026:
                break

        return result

    # ----------------------------------------------------------
    # 騎手ID → 年間成績取得（直接アクセス）
    # ----------------------------------------------------------

    async def get_jockey_stats_by_id(self, jockey_id: str, jockey_name: str = "") -> dict:
        """
        騎手IDから直接成績ページにアクセスして年間成績を取得
        Args:
            jockey_id: 騎手ID（5桁の数字）
            jockey_name: 騎手名（キャッシュ用、省略可）
        Returns:
            {"win_rate": 0.15, "place_rate": 0.35, "wins": 50, "races": 300}
        """
        # キャッシュチェック
        if jockey_name and jockey_name in self.jockey_cache:
            self.log(f"騎手キャッシュ: {jockey_name}")
            return self.jockey_cache[jockey_name]

        self.log(f"騎手成績取得（ID: {jockey_id}）")

        result = {"win_rate": 0.0, "place_rate": 0.0, "wins": 0, "races": 0}

        try:
            jockey_url = f"https://db.netkeiba.com/jockey/{jockey_id}"
            await self.page.goto(jockey_url, wait_until="domcontentloaded", timeout=30000)
            await asyncio.sleep(1.5)

            # テーブルセル方式で成績を取得
            result = await self._parse_jockey_stats_from_page()

            if result["races"] > 0:
                self.log(f"  [OK] 勝率{result['win_rate']:.3f} 複勝率{result['place_rate']:.3f} ({result['wins']}勝/{result['races']}騎乗)")
            else:
                self.log(f"  [!] 成績データが見つかりません")

            # キャッシュに保存
            if jockey_name:
                self.jockey_cache[jockey_name] = result

        except Exception as e:
            self.log(f"  [ERROR] 騎手成績取得エラー: {e}")

        return result

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

        for attempt in range(self.max_retries):
            try:
                # 検索ページに移動
                await self.page.goto("https://db.netkeiba.com/?pid=jockey_search_detail", wait_until="domcontentloaded", timeout=30000)
                await asyncio.sleep(2)

                # 騎手検索専用フォーム（2番目のフォーム）を使用
                forms = await self.page.locator('form').all()
                jockey_form = None

                # pid=jockey_list を持つフォームを探す
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
                    self.log(f"  [!] 騎手検索フォームが見つかりません")
                    self.jockey_cache[jockey_name] = result
                    return result

                # フォーム内のword inputに入力
                # JRAデータの騎手名にはスペースが含まれる（例: "武 豊"）が、
                # netkeibaの検索はスペースなしの方が精度が高い
                # また、外国人騎手はJRAが半角（C.ルメール）、netkeibaが全角（Ｃ．ルメール）
                # のため、イニシャル部分を全角に変換する
                search_name = jockey_name.replace(" ", "").replace("　", "")
                # 外国人騎手の半角イニシャルを全角に変換（例: "C." → "Ｃ．"）
                search_name = re.sub(
                    r'^([A-Za-z])\.',
                    lambda m: chr(ord(m.group(1).upper()) - ord('A') + ord('Ａ')) + '．',
                    search_name
                )
                word_input = jockey_form.locator('input[name="word"]')
                await word_input.fill(search_name)
                self.log(f"  フォーム入力 (騎手検索): {search_name}")

                # submitボタンをクリック
                await asyncio.sleep(1)
                submit_button = jockey_form.locator('input[type="submit"]')
                await submit_button.click()
                await asyncio.sleep(3)

                # フォーム送信後のURLをチェック
                # 検索結果が1件の場合、直接騎手ページにリダイレクトされる
                # 例: https://db.netkeiba.com/jockey/00666/
                current_url = self.page.url
                jockey_redirect = re.search(r'/jockey/(\d{5})/?$', current_url)

                if jockey_redirect:
                    # 直接騎手ページに遷移した → そのまま成績を取得
                    jockey_id = jockey_redirect.group(1)
                    self.log(f"  [OK] 騎手ページに直接遷移 (ID: {jockey_id}): {current_url}")
                    result = await self._parse_jockey_stats_from_page()
                else:
                    # 検索結果一覧ページ → リンクから騎手を探す
                    links = await self.page.locator("a[href*='/jockey/']").all()
                    self.log(f"  検索結果リンク数: {len(links)}個")

                    jockey_page_url = None
                    normalized_search_name = normalize_jockey_name(jockey_name)

                    for link in links:
                        href = await link.get_attribute("href")
                        if href and "/jockey/" in href:
                            if any(x in href for x in ["search_detail", "top.html", "leading"]):
                                continue
                            # 末尾スラッシュあり/なし両方に対応
                            if re.search(r'/jockey/\d{5}/?$', href):
                                link_text = await link.text_content()
                                if link_text:
                                    normalized_link_text = normalize_jockey_name(link_text.strip())
                                    if normalized_link_text == normalized_search_name:
                                        if href.startswith("/"):
                                            href = f"https://db.netkeiba.com{href}"
                                        self.log(f"  [OK] 騎手ページ発見（名前一致）: {href}")
                                        jockey_page_url = href
                                        break

                    if not jockey_page_url:
                        self.log(f"  [!] 騎手が見つかりません: {jockey_name}")
                        self.jockey_cache[jockey_name] = result
                        return result

                    # 騎手ページから成績を取得（テーブルセル方式）
                    await self.page.goto(jockey_page_url, wait_until="domcontentloaded")
                    await asyncio.sleep(1.5)
                    result = await self._parse_jockey_stats_from_page()

                if result["races"] > 0:
                    self.log(f"  [OK] 勝率{result['win_rate']:.3f} 複勝率{result['place_rate']:.3f} ({result['wins']}勝/{result['races']}騎乗)")
                else:
                    self.log(f"  [!] 成績データが見つかりません")

                # 成功したのでキャッシュに保存して返す
                self.jockey_cache[jockey_name] = result
                return result

            except Exception as e:
                self.log(f"  [ERROR] 騎手検索エラー (試行{attempt+1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    self.log(f"  → {delay}秒後にリトライします...")
                    await asyncio.sleep(delay)
                else:
                    self.log(f"  → リトライ上限に達しました。")

        # 全リトライ失敗後、キャッシュに保存
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
    print(f"  [netkeiba] スクレイパー v{VERSION}")
    print("=" * 60)
    print()

    # input.json読み込み
    input_file = Path(input_path)
    if not input_file.exists():
        print(f"[ERROR] ファイルが見つかりません: {input_path}")
        return

    print(f"[入力] {input_file}")

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    horses = data.get("horses", [])
    print(f"[対象馬] {len(horses)}頭")

    scraper = NetkeibaScraper(headless=True, debug=True)

    try:
        await scraper.start()

        # 各馬のデータを拡張
        for i, horse in enumerate(horses):
            horse_name = horse.get("name", "")
            print(f"\n[{i+1}/{len(horses)}] {horse_name}")

            # 5頭ごとにブラウザ再起動（メモリリフレッシュ）
            if i > 0 and i % 5 == 0:
                print("\n[メモリリフレッシュ] ブラウザを再起動します...")
                await scraper.restart()
                print("[OK] 再起動完了\n")

            if not horse_name:
                print("  [!] 馬名が空です。スキップします。")
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
                print(f"  [!] 過去走データを取得できませんでした")

            # 騎手成績取得
            jockey_name = horse.get("jockey", "")
            if jockey_name:
                print(f"  騎手: {jockey_name}")

                # 方法1: 過去走データから騎手IDを抽出
                past_races = horse.get("past_races", [])
                jockey_id = scraper.extract_jockey_id_from_past_races(jockey_name, past_races)

                if jockey_id:
                    # 過去走でIDが見つかった → 直接アクセス
                    jockey_stats = await scraper.get_jockey_stats_by_id(jockey_id, jockey_name)
                else:
                    # 過去走にない → 検索（リーディングリスト方式）
                    print(f"  → 過去走に騎手なし、検索を試行します")
                    jockey_stats = await scraper.search_jockey(jockey_name)

                horse["jockey_stats"] = jockey_stats
            else:
                print(f"  [!] 騎手情報がありません（input.jsonに'jockey'フィールドが必要）")
                horse["jockey_stats"] = {"win_rate": 0.0, "place_rate": 0.0, "wins": 0, "races": 0}

            await asyncio.sleep(2.0)  # 負荷軽減（0.5秒→2秒に延長）

        # 保存
        output_file = input_file.parent / input_file.name.replace("_input.json", "_enriched_input.json")

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"\n{'=' * 60}")
        print(f"  [OK] 保存完了: {output_file}")
        print(f"{'=' * 60}")

    except KeyboardInterrupt:
        print("\n[!] 中断")
    except Exception as e:
        print(f"\n[ERROR] エラー: {e}")
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
