"""
HTML解析ユーティリティ
全スクレイパー共通のパース関数を集約
"""
import re
import unicodedata
from urllib.parse import quote


def safe_float(text) -> float:
    try:
        return float(str(text).strip().replace(",", ""))
    except (ValueError, AttributeError, TypeError):
        return 0.0


def safe_int(text) -> int:
    try:
        return int(re.sub(r'[^\d\-]', '', str(text).strip()))
    except (ValueError, AttributeError):
        return 0


def parse_odds_range(text: str) -> float:
    """'2.4-6.1' -> 中央値 4.25"""
    text = str(text).strip()
    m = re.match(r"([\d.]+)\s*[-～]\s*([\d.]+)", text)
    if m:
        return round((float(m.group(1)) + float(m.group(2))) / 2, 1)
    return safe_float(text)


def parse_odds_range_low(text: str) -> float:
    """'2.4-6.1' -> 下限 2.4"""
    text = str(text).strip()
    m = re.match(r"([\d.]+)\s*[-～]\s*([\d.]+)", text)
    if m:
        return float(m.group(1))
    return safe_float(text)


def extract_venue(meeting_text: str) -> str:
    """'1回東京5日' -> '東京'"""
    m = re.search(r"\d+回(.+?)\d+日", meeting_text)
    return m.group(1) if m else ""


def normalize_jockey_name(name: str) -> str:
    """騎手名を正規化（比較用）"""
    normalized = unicodedata.normalize('NFKC', name)
    normalized = normalized.replace(" ", "").replace("\u3000", "").replace(".", "").replace("\u30fb", "").replace("\uff0e", "")
    return normalized.lower()


def encode_for_netkeiba(text: str) -> str:
    """netkeibaの検索用にEUC-JPでURLエンコード"""
    try:
        return quote(text.encode('euc-jp'), safe='')
    except UnicodeEncodeError:
        return quote(text)


def time_to_seconds(time_str: str) -> float:
    """タイム文字列を秒に変換 (例: '1:23.4' -> 83.4)"""
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
    """馬体重文字列をパース (例: '480(+4)' -> (480, 4))"""
    text = str(text).strip()
    m = re.match(r'(\d+)\(([+\-]?\d+)\)', text)
    if m:
        return int(m.group(1)), int(m.group(2))
    m2 = re.match(r'(\d+)', text)
    if m2:
        return int(m2.group(1)), 0
    return 0, 0


def parse_sex_age(text: str) -> tuple:
    """性齢文字列をパース (例: '牡3' -> ('牡', 3))"""
    text = str(text).strip()
    m = re.match(r'([牡牝セ騸])(\d+)', text)
    if m:
        return m.group(1), int(m.group(2))
    return "", 0


def build_header_map(headers: list, mapping: dict) -> dict:
    """
    ヘッダー文字列リストからカラムインデックスのマッピングを作成

    Args:
        headers: テーブルヘッダーのテキストリスト
        mapping: {検索キーワード: マップキー} の辞書
    Returns:
        {マップキー: カラムインデックス}
    """
    hmap = {}
    for i, h in enumerate(headers):
        for keyword, key in mapping.items():
            if keyword in h and key not in hmap:
                hmap[key] = i
    return hmap
