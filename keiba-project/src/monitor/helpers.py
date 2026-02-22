"""
モニターヘルパー関数（スタンドアロン）
"""
import re
from datetime import datetime

from config.settings import COURSE_NAME_TO_ID


def construct_race_id(race: dict, meeting_info: list) -> str:
    """meeting情報からnetkeiba race_idを構築
    形式: YYYYCCKK DDNN (12桁)
    """
    m_text = race.get("meeting_text", "")
    if not m_text:
        m_idx = race.get("meeting_idx")
        for idx, text, venue in meeting_info:
            if idx == m_idx:
                m_text = text
                break

    if not m_text:
        return ""

    # "1回東京8日" → kai=1, day=8
    m = re.match(r"(\d+)回(.+?)(\d+)日", m_text)
    if not m:
        return ""

    kai = int(m.group(1))
    day = int(m.group(3))

    venue_name = race["race_info"].get("venue", "")
    course_id = COURSE_NAME_TO_ID.get(venue_name, 0)
    if not course_id:
        return ""

    race_num = race["race_info"].get("race_number", 0)
    year = datetime.now().year

    return f"{year:04d}{course_id:02d}{kai:02d}{day:02d}{race_num:02d}"


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
