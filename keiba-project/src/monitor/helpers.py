"""
モニターヘルパー関数（スタンドアロン）
"""
import re
from datetime import datetime

from config.settings import COURSE_NAME_TO_ID


def build_race_id(meeting_text: str, venue: str, race_number: int) -> str:
    """meeting_text + venue + race_number からnetkeiba race_idを構築

    Args:
        meeting_text: "1回東京8日" 形式のテキスト
        venue: 会場名 ("東京", "中山" 等)
        race_number: レース番号 (1-12)

    Returns:
        12桁のrace_id (e.g. "202605010811") or 空文字
    """
    if not meeting_text:
        return ""

    m = re.match(r"(\d+)回(.+?)(\d+)日", meeting_text)
    if not m:
        return ""

    kai = int(m.group(1))
    day = int(m.group(3))

    course_id = COURSE_NAME_TO_ID.get(venue, 0)
    if not course_id:
        return ""

    year = datetime.now().year
    return f"{year:04d}{course_id:02d}{kai:02d}{day:02d}{race_number:02d}"


def construct_race_id(race: dict, meeting_info: list) -> str:
    """meeting情報からnetkeiba race_idを構築（モニター用ラッパー）
    形式: YYYYCCKKDDNN (12桁)
    """
    m_text = race.get("meeting_text", "")
    if not m_text:
        m_idx = race.get("meeting_idx")
        for idx, text, venue in meeting_info:
            if idx == m_idx:
                m_text = text
                break

    venue_name = race["race_info"].get("venue", "")
    race_num = race["race_info"].get("race_number", 0)
    return build_race_id(m_text, venue_name, race_num)


def collect_recommendations(ev_results: dict) -> list:
    """推奨買い目を収集 → [(bet_type, bet_dict), ...]"""
    recs = []
    if ev_results.get("low_confidence"):
        return recs
    for bt, key in [("馬連", "quinella"), ("ワイド", "wide")]:
        for bet in ev_results.get(key, []):
            if bet["ev"] >= 1.0:
                recs.append((bt, bet))
    return recs


def extract_venue(meeting_text: str) -> str:
    """開催テキストから会場名を抽出"""
    from src.scraping.parsers import extract_venue as _extract
    return _extract(meeting_text)
