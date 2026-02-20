"""
レース結果取得（HTTP版）
race.netkeiba.com から過去レース結果を httpx + BeautifulSoup でスクレイピングし CSV で保存
"""
import asyncio
import csv
import re
from datetime import datetime
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

from src.scraping.parsers import (
    safe_int, safe_float, time_to_seconds, parse_horse_weight, parse_sex_age,
)
from config.settings import COURSE_CODES, CSV_COLUMNS, RAW_DIR

# 同時リクエスト数（高すぎるとブロックされる）
CONCURRENCY = 3
# リクエスト間隔（秒）— サーバー負荷軽減
REQUEST_DELAY = 1.0

# race.netkeiba.com ベースURL
_BASE = "https://race.netkeiba.com"

# ブラウザ風 User-Agent
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
}


def _months_in_range(start_date: str, end_date: str) -> list:
    """開始〜終了の年月リストを生成"""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    months = []
    current = start.replace(day=1)
    while current <= end:
        months.append((current.year, current.month))
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)
    return months


# ============================================================
# HTML パーサー（race.netkeiba.com 形式）
# ============================================================

def _build_header_map(headers: list) -> dict:
    hmap = {}
    for i, h in enumerate(headers):
        if "着順" in h:
            hmap["finish"] = i
        elif "枠" in h and "番" not in h:
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
        elif "コーナー" in h or "通過" in h:
            hmap["passing"] = i
        elif "後3F" in h or "3F" in h:
            hmap["last3f"] = i
        elif "上" in h and ("り" in h or "がり" in h):
            hmap["last3f"] = i
        elif "単勝" in h and "オッズ" in h:
            hmap["odds"] = i
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


def _parse_race_meta(soup: BeautifulSoup, race_id: str) -> dict:
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
        # race.netkeiba.com 形式
        race_name_el = soup.select_one(".RaceName")
        if race_name_el:
            meta["race_name"] = race_name_el.get_text(strip=True)

        # 芝/ダート・距離・天候・馬場状態
        info_el = soup.select_one(".RaceData01")
        if info_el:
            info_text = info_el.get_text()
        else:
            # db.netkeiba.com フォールバック
            info_el = (soup.select_one(".racedata")
                       or soup.select_one(".data_intro"))
            info_text = info_el.get_text() if info_el else ""

        if info_text:
            dist_match = re.search(r'([芝ダ障])[^0-9]*?(\d{3,4})m', info_text)
            if dist_match:
                meta["surface"] = dist_match.group(1)
                meta["distance"] = int(dist_match.group(2))
            weather_match = re.search(r'天候\s*[:：]?\s*([^\s/]+)', info_text)
            if weather_match:
                meta["weather"] = weather_match.group(1)
            cond_match = re.search(r'馬場\s*[:：]?\s*(良|稍重|重|不良)', info_text)
            if not cond_match:
                cond_match = re.search(r'[芝ダート]*\s*[:：]?\s*(良|稍重|重|不良)', info_text)
            if cond_match:
                meta["track_condition"] = cond_match.group(1)

        # 日付・クラス
        # titleタグ、RaceData02、その他から探す
        date_sources = []
        title_el = soup.find("title")
        if title_el:
            date_sources.append(title_el.get_text())
        info_el2 = soup.select_one(".RaceData02")
        if info_el2:
            date_sources.append(info_el2.get_text())
        date_sources.append(info_text)

        info_text2 = " ".join(date_sources)

        if info_text2:
            date_match = re.search(r'(\d{4})年(\d{1,2})月(\d{1,2})日', info_text2)
            if date_match:
                meta["race_date"] = (
                    f"{date_match.group(1)}-{int(date_match.group(2)):02d}"
                    f"-{int(date_match.group(3)):02d}"
                )
            for cls in ["G1", "G2", "G3", "GI", "GII", "GIII",
                        "オープン", "OP", "3勝", "2勝", "1勝",
                        "未勝利", "新馬", "リステッド"]:
                if cls in info_text2 or cls in (meta.get("race_name") or ""):
                    meta["race_class"] = (cls.replace("GI", "G1")
                                          .replace("GII", "G2")
                                          .replace("GIII", "G3"))
                    break

        # 日付フォールバック（race_idから推定）
        if not meta["race_date"]:
            for el in soup.select("p.smalltxt, .race_otherdata, dd, .RaceData02"):
                text = el.get_text()
                dm = re.search(r'(\d{4})年(\d{1,2})月(\d{1,2})日', text)
                if dm:
                    meta["race_date"] = (
                        f"{dm.group(1)}-{int(dm.group(2)):02d}"
                        f"-{int(dm.group(3)):02d}"
                    )
                    break
    except Exception:
        pass
    return meta


def parse_race_html(html: bytes, race_id: str) -> list:
    """レース結果HTMLをパースして行データのリストを返す"""
    soup = BeautifulSoup(html, "lxml")
    results = []

    # race.netkeiba.com → RaceTable01, db.netkeiba.com → race_table_01
    table = (soup.find("table", class_="RaceTable01")
             or soup.find("table", class_="race_table_01")
             or soup.find("table", class_="nk_tb_common"))
    if not table:
        for t in soup.find_all("table"):
            if t.find("th", string=re.compile("着順")):
                table = t
                break
    if not table:
        return results

    race_meta = _parse_race_meta(soup, race_id)

    all_tr = table.find_all("tr")
    if not all_tr:
        return results

    ths = all_tr[0].find_all("th")
    headers = [th.get_text(strip=True) for th in ths]
    header_map = _build_header_map(headers)

    def get(cells_text, key):
        idx = header_map.get(key, -1)
        return cells_text[idx] if 0 <= idx < len(cells_text) else ""

    for tr in all_tr[1:]:
        cells = tr.find_all("td")
        if len(cells) < 5:
            continue

        cell_texts = [c.get_text(strip=True) for c in cells]

        row = {col: "" for col in CSV_COLUMNS}
        row.update(race_meta)

        row["finish_position"] = safe_int(get(cell_texts, "finish"))
        row["frame_number"] = safe_int(get(cell_texts, "frame"))
        row["horse_number"] = safe_int(get(cell_texts, "horse_num"))
        row["horse_name"] = get(cell_texts, "horse_name").replace("\n", "").strip()
        row["weight_carried"] = safe_float(get(cell_texts, "weight_carried"))
        row["jockey_name"] = get(cell_texts, "jockey").replace("\n", "").strip()
        row["trainer_name"] = get(cell_texts, "trainer").replace("\n", "").strip()
        row["finish_time_sec"] = time_to_seconds(get(cell_texts, "time"))
        row["margin"] = get(cell_texts, "margin")
        row["passing_order"] = get(cell_texts, "passing")
        row["last_3f"] = safe_float(get(cell_texts, "last3f"))
        row["win_odds"] = safe_float(get(cell_texts, "odds"))
        row["popularity"] = safe_int(get(cell_texts, "popularity"))
        row["prize_money"] = safe_float(get(cell_texts, "prize"))

        sex, age = parse_sex_age(get(cell_texts, "sex_age"))
        row["sex"] = sex
        row["age"] = age

        hw, hwc = parse_horse_weight(get(cell_texts, "horse_weight"))
        row["horse_weight"] = hw
        row["horse_weight_change"] = hwc

        if row["finish_position"] <= 0:
            continue

        # horse_id / jockey_id をリンクから抽出
        name_idx = header_map.get("horse_name", -1)
        if 0 <= name_idx < len(cells):
            link = cells[name_idx].find("a", href=re.compile(r"/horse/\d+"))
            if link:
                m = re.search(r"/horse/(\d{10})", link["href"])
                if m:
                    row["horse_id"] = m.group(1)
        jockey_idx = header_map.get("jockey", -1)
        if 0 <= jockey_idx < len(cells):
            link = cells[jockey_idx].find("a", href=re.compile(r"/jockey/"))
            if link:
                m = re.search(r"/jockey/(?:result/recent/)?(\d{5})", link["href"])
                if m:
                    row["jockey_id"] = m.group(1)

        results.append(row)

    num_entries = len(results)
    for r in results:
        r["num_entries"] = num_entries

    return results


# ============================================================
# 払い戻しテーブルパーサー
# ============================================================

RETURN_COLUMNS = [
    "race_id", "bet_type", "combination", "payout", "popularity",
]

_BET_TYPE_MAP = {
    "単勝": "win",
    "複勝": "place",
    "枠連": "bracket_quinella",
    "馬連": "quinella",
    "ワイド": "wide",
    "馬単": "exacta",
    "三連複": "trio",
    "三連単": "trifecta",
    "3連複": "trio",
    "3連単": "trifecta",
}


def parse_return_html(html: bytes, race_id: str) -> list:
    """レースページHTMLから払い戻しデータをパースする"""
    soup = BeautifulSoup(html, "lxml")
    rows = []

    # race.netkeiba.com 形式: Payout_Detail_Table
    payout_tables = soup.find_all("table", class_="Payout_Detail_Table")
    if payout_tables:
        return _parse_return_race_netkeiba(soup, race_id)

    # db.netkeiba.com 形式: pay_table_01（フォールバック）
    for table in soup.find_all("table", class_="pay_table_01"):
        for tr in table.find_all("tr"):
            th = tr.find("th")
            if not th:
                continue
            bet_label = th.get_text(strip=True)
            bet_type = _BET_TYPE_MAP.get(bet_label)
            if not bet_type:
                continue

            tds = tr.find_all("td")
            if len(tds) < 3:
                continue

            combos = [s.strip() for s in tds[0].decode_contents().split("<br") if s.strip()]
            combos = [re.sub(r'[/>\s]+', '', c).strip() for c in combos]
            payouts = [s.strip() for s in tds[1].decode_contents().split("<br") if s.strip()]
            payouts = [re.sub(r'[/>\s]+', '', p).strip() for p in payouts]
            pops = [s.strip() for s in tds[2].decode_contents().split("<br") if s.strip()]
            pops = [re.sub(r'[/>\s]+', '', p).strip() for p in pops]

            for combo, payout, pop in zip(combos, payouts, pops):
                if not combo or not payout:
                    continue
                rows.append({
                    "race_id": race_id,
                    "bet_type": bet_type,
                    "combination": combo,
                    "payout": safe_int(payout.replace(",", "")),
                    "popularity": safe_int(pop),
                })

    return rows


def _parse_return_race_netkeiba(soup: BeautifulSoup, race_id: str) -> list:
    """race.netkeiba.com 形式の払い戻しテーブルをパース"""
    rows = []

    for table in soup.find_all("table", class_="Payout_Detail_Table"):
        for tr in table.find_all("tr"):
            th = tr.find("th")
            if not th:
                continue
            bet_label = th.get_text(strip=True)
            bet_type = _BET_TYPE_MAP.get(bet_label)
            if not bet_type:
                continue

            tds = tr.find_all("td")
            if len(tds) < 3:
                continue

            # 組合せ: div>span（単複）または ul>li>span（連系）
            combo_td = tds[0]
            combos = []

            uls = combo_td.find_all("ul")
            divs = combo_td.find_all("div")

            if uls:
                # 連系券種: 各ulが1組合せ
                for ul in uls:
                    nums = [li.get_text(strip=True) for li in ul.find_all("li")
                            if li.get_text(strip=True)]
                    if nums:
                        combos.append("-".join(nums))
            elif divs:
                # 単複: 各divが1組合せ（1頭）
                for div in divs:
                    num = div.get_text(strip=True)
                    if num:
                        combos.append(num)

            if not combos:
                # フォールバック: テキストから抽出
                text = combo_td.get_text(strip=True)
                if text:
                    combos.append(text)

            # 払い戻し金額: "950円" or "290円160円410円"
            payout_text = tds[1].get_text(strip=True)
            payouts = re.findall(r'([\d,]+)円', payout_text)

            # 人気: "5人気" or "5人気2人気6人気"
            pop_text = tds[2].get_text(strip=True)
            pops = re.findall(r'(\d+)人気', pop_text)

            for i, combo in enumerate(combos):
                payout = payouts[i] if i < len(payouts) else ""
                pop = pops[i] if i < len(pops) else ""
                if not combo or not payout:
                    continue
                rows.append({
                    "race_id": race_id,
                    "bet_type": bet_type,
                    "combination": combo,
                    "payout": safe_int(payout.replace(",", "")),
                    "popularity": safe_int(pop),
                })

    return rows


def save_returns_csv(results: list, output_path: Path, append: bool = False):
    mode = "a" if append and output_path.exists() else "w"
    write_header = mode == "w" or not output_path.exists()

    with open(output_path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RETURN_COLUMNS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        for row in results:
            writer.writerow(row)


# ============================================================
# CSV 書き出し（レース結果）
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
# HTTP 一括収集
# ============================================================

async def _fetch_calendar_dates(client: httpx.AsyncClient,
                                start_date: str, end_date: str) -> list:
    """netkeibaカレンダーから開催日を取得"""
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    months = _months_in_range(start_date, end_date)
    all_dates = []

    for year, month in months:
        url = f"{_BASE}/top/calendar.html?year={year}&month={month}"
        try:
            r = await client.get(url)
            soup = BeautifulSoup(r.content, "lxml")
            for a in soup.find_all("a", href=re.compile(r"kaisai_date=\d{8}")):
                m = re.search(r"kaisai_date=(\d{8})", a["href"])
                if m:
                    ds = m.group(1)
                    dt = datetime.strptime(ds, "%Y%m%d")
                    if start_dt <= dt <= end_dt and ds not in all_dates:
                        all_dates.append(ds)
        except Exception as e:
            print(f"  [ERROR] カレンダー取得エラー ({year}/{month}): {e}")

    all_dates.sort()
    return all_dates


# db.netkeiba.com ブロック検出フラグ（一度400が返ったらフォールバックのみ使う）
_db_blocked = False


async def _fetch_race_ids(client: httpx.AsyncClient, date_str: str) -> list:
    """日付のレースID一覧を取得（通常 → フォールバック）"""
    global _db_blocked

    if not _db_blocked:
        # 通常: db.netkeiba.com
        url = f"https://db.netkeiba.com/race/list/{date_str}/"
        try:
            r = await client.get(url)
            if r.status_code == 200 and len(r.content) > 0:
                soup = BeautifulSoup(r.content, "lxml")
                race_ids = []
                for a in soup.find_all("a", href=re.compile(r"/race/\d{12}")):
                    m = re.search(r"/race/(\d{12})", a["href"])
                    if m and m.group(1) not in race_ids:
                        race_ids.append(m.group(1))
                if race_ids:
                    return race_ids
            else:
                _db_blocked = True
                print("  [INFO] db.netkeiba.com blocked -> race.netkeiba.com にフォールバック")
        except Exception:
            _db_blocked = True

    # フォールバック: race.netkeiba.com
    url = f"{_BASE}/top/race_list_sub.html?kaisai_date={date_str}"
    r = await client.get(url)
    if r.status_code != 200:
        return []
    soup = BeautifulSoup(r.content, "lxml")
    race_ids = []
    for a in soup.find_all("a", href=re.compile(r"race_id=\d{12}")):
        m = re.search(r"race_id=(\d{12})", a["href"])
        if m and m.group(1) not in race_ids:
            race_ids.append(m.group(1))
    return race_ids


async def _fetch_and_parse_race(client: httpx.AsyncClient, race_id: str,
                                semaphore: asyncio.Semaphore) -> tuple:
    """1レースを取得・パース。(results, returns) を返す（通常 → フォールバック）"""
    global _db_blocked

    async with semaphore:
        if not _db_blocked:
            # 通常: db.netkeiba.com
            url = f"https://db.netkeiba.com/race/{race_id}/"
            r = await client.get(url)
            if r.status_code != 200 or len(r.content) == 0:
                _db_blocked = True
                # フォールバック
                url = f"{_BASE}/race/result.html?race_id={race_id}&rf=race_list"
                r = await client.get(url)
        else:
            url = f"{_BASE}/race/result.html?race_id={race_id}&rf=race_list"
            r = await client.get(url)

        await asyncio.sleep(REQUEST_DELAY)
        results = parse_race_html(r.content, race_id)
        returns = parse_return_html(r.content, race_id)
        if results:
            print(f"  [OK] {race_id}: {len(results)}頭, 払戻{len(returns)}件")
        return results, returns


def _get_collected_dates(results_path: Path) -> set:
    """既存CSVから収集済み日付を取得"""
    if not results_path.exists():
        return set()
    collected = set()
    with open(results_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            d = row.get("race_date", "")
            if d:
                collected.add(d.replace("-", ""))
    return collected


async def collect_races(start_date: str, end_date: str,
                        output_path: str = None, headless: bool = True,
                        append: bool = False):
    print(f"\n{'=' * 60}")
    print(f"  過去レース結果収集 (HTTP版, 並列数: {CONCURRENCY})")
    print(f"  期間: {start_date} ~ {end_date}")
    print(f"{'=' * 60}\n")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    results_path = Path(output_path) if output_path else RAW_DIR / "results.csv"
    returns_path = results_path.parent / "returns.csv"
    total_races = 0

    # 収集済み日付をスキップ（resume対応）
    collected_dates = set()
    if append and results_path.exists():
        collected_dates = _get_collected_dates(results_path)
        if collected_dates:
            print(f"  既存データ: {len(collected_dates)}日分を検出（スキップします）")

    async with httpx.AsyncClient(
        timeout=20, follow_redirects=True, headers=_HEADERS,
    ) as client:
        try:
            print("  カレンダーから開催日を取得中...")
            dates = await _fetch_calendar_dates(client, start_date, end_date)
            print(f"  対象日数: {len(dates)}日（開催日のみ）")

            # 収集済み日付を除外
            if collected_dates:
                dates = [d for d in dates if d not in collected_dates]
                print(f"  未収集: {len(dates)}日")

            if not dates:
                print("  全日付が収集済みです")
                return

            semaphore = asyncio.Semaphore(CONCURRENCY)

            for date_idx, date_str in enumerate(dates):
                print(f"\n[{date_idx+1}/{len(dates)}] {date_str}")

                race_ids = await _fetch_race_ids(client, date_str)
                print(f"  {len(race_ids)}レース発見")

                if not race_ids:
                    print("  [WARN] レースが見つかりません（スキップ）")
                    continue

                tasks = [
                    _fetch_and_parse_race(client, rid, semaphore)
                    for rid in race_ids
                ]
                raw_list = await asyncio.gather(*tasks, return_exceptions=True)

                all_results = []
                all_returns = []
                for item in raw_list:
                    if isinstance(item, Exception):
                        print(f"  [ERROR] {item}")
                    elif item:
                        results, returns = item
                        if results:
                            all_results.extend(results)
                            total_races += 1
                        if returns:
                            all_returns.extend(returns)

                # 常にappendモード（resume対応）
                if all_results:
                    save_results_csv(all_results, results_path, append=True)
                if all_returns:
                    save_returns_csv(all_returns, returns_path, append=True)

                print(f"  -> 結果{len(all_results)}行, 払戻{len(all_returns)}行 (累計{total_races}レース)")

        except KeyboardInterrupt:
            print("\n[!] 中断（--append で再開可能）")
        except Exception as e:
            print(f"\n[ERROR] {e}")
            import traceback
            traceback.print_exc()

    print(f"\n[OK] 完了: {total_races}レース")
    print(f"  結果: {results_path}")
    print(f"  払戻: {returns_path}")
