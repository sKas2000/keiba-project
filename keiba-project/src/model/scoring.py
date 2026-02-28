"""
ルールベーススコアリング
100点満点（実力50+騎手20+適性15+調子10+他5）
"""
import re
from datetime import datetime

from src.scraping.parsers import safe_int


def calculate_days_between(date1_str: str, date2_str: str) -> int:
    try:
        d1 = datetime.strptime(str(date1_str).replace("/", "-"), "%Y-%m-%d")
        d2 = datetime.strptime(str(date2_str).replace("/", "-"), "%Y-%m-%d")
        return abs((d2 - d1).days)
    except Exception:
        return 0


def is_small_margin(margin_str: str, threshold: float) -> bool:
    if not margin_str:
        return False
    margin_str = str(margin_str).strip()
    sec_match = re.match(r'([\d.]+)', margin_str)
    if sec_match and '馬身' not in margin_str:
        try:
            return float(sec_match.group(1)) <= threshold
        except ValueError:
            return False
    if '馬身' in margin_str or 'バ身' in margin_str:
        if any(k in margin_str for k in ('ハナ', 'アタマ', 'クビ')):
            return True
        if '1/2' in margin_str or '半' in margin_str:
            return True
        if '1' in margin_str and '1/2' not in margin_str:
            return threshold >= 0.2
        if '2' in margin_str:
            return threshold >= 0.4
        if '3' in margin_str:
            return threshold >= 0.6
    return False


def _top3_rate_score(past_races):
    if not past_races:
        return 0.0, "過去走なし"
    recent = past_races[:4]
    top3 = sum(1 for r in recent if r.get("finish", 99) <= 3)
    rate = top3 / len(recent)
    return round(rate * 20, 1), f"3着内率{top3}/{len(recent)}"


def _margin_score(past_races):
    if not past_races:
        return 0.0, "過去走なし"
    scores = []
    for race in past_races[:4]:
        finish = race.get("finish", 99)
        margin = race.get("margin", "")
        if finish == 1:
            scores.append(15)
        elif finish == 2:
            scores.append(12 if is_small_margin(margin, 0.5) else 10)
        elif finish == 3:
            scores.append(9 if is_small_margin(margin, 1.0) else 7)
        elif finish <= 5:
            scores.append(6)
        else:
            scores.append(3)
    avg = sum(scores) / len(scores) if scores else 0
    return round(avg, 1), f"着差評価{avg:.1f}点"


def _course_score(past_races, venue, surface, distance):
    if not past_races or not venue or not distance:
        return 0.0, "同条件なし"
    matching = []
    for r in past_races:
        v = r.get("venue", "") == venue
        rs = r.get("surface", "")
        s = surface and rs and rs[0] == surface[0]
        rd = safe_int(r.get("distance", "0"))
        d = abs(rd - distance) <= 100 if rd > 0 else False
        if v and s and d:
            matching.append(r)
    if not matching:
        return 0.0, "同条件なし"
    points = []
    for r in matching:
        f = r.get("finish", 99)
        points.append(5 if f == 1 else 3 if f == 2 else 2 if f == 3 else 0)
    avg = sum(points) / len(points) if points else 0
    score = min(avg * len(matching) / 2, 15)
    return round(score, 1), f"同条件{len(matching)}戦平均{avg:.1f}点"


def calculate_ability_score(horse, race_info):
    past = horse.get("past_races", [])
    t3, t3d = _top3_rate_score(past)
    ms, msd = _margin_score(past)
    cs, csd = _course_score(past, race_info.get("venue", ""), race_info.get("surface", ""), race_info.get("distance", 0))
    total = min(round(t3 + ms + cs, 1), 50)
    return total, {"top3_rate": t3, "margin_quality": ms, "course_record": cs, "detail": f"{t3d}, {msd}, {csd}"}


def calculate_jockey_score(horse):
    wr = horse.get("jockey_stats", {}).get("win_rate", 0.0)
    score = min(wr * 100, 20)
    return round(score, 1), f"{horse.get('jockey', '不明')}勝率{wr:.3f}"


def calculate_fitness_score(horse, race_info):
    past = horse.get("past_races", [])
    venue = race_info.get("venue", "")
    distance = race_info.get("distance", 0)

    vs = 0.0
    vd = "同場なし"
    if past and venue:
        same = [r for r in past if r.get("venue", "") == venue]
        if same:
            avg = sum(r.get("finish", 99) for r in same) / len(same)
            vs = round(max(0, min(8, 8 * (10 - avg) / 9)), 1)
            vd = f"同場{len(same)}戦平均{avg:.1f}着"

    ds = 0.0
    dd = "同距離なし"
    if past and distance:
        same = [r for r in past if abs(safe_int(r.get("distance", "0")) - distance) <= 200 and safe_int(r.get("distance", "0")) > 0]
        if same:
            avg = sum(r.get("finish", 99) for r in same) / len(same)
            ds = round(max(0, min(7, 7 * (10 - avg) / 9)), 1)
            dd = f"同距離{len(same)}戦平均{avg:.1f}着"

    total = min(round(vs + ds, 1), 15)
    return total, {"venue": vs, "distance": ds, "detail": f"{vd}, {dd}"}


def calculate_form_score(horse, race_date):
    past = horse.get("past_races", [])
    if not past:
        return 2.0, {"rotation": 2.0, "weight_change": 0.0, "training": 0.0, "detail": "初出走2点"}
    last_date = past[0].get("date", "")
    if not last_date or not race_date:
        return 2.0, {"rotation": 2.0, "weight_change": 0.0, "training": 0.0, "detail": "前走日不明2点"}
    days = calculate_days_between(last_date, race_date)
    if 14 <= days <= 28:
        s, d = 4.0, f"前走{days//7}週前4点"
    elif days < 14 or 29 <= days <= 56:
        s, d = 3.0, f"前走{days//7}週前3点"
    else:
        s, d = 2.0, f"前走{days//7}週前2点"
    return s, {"rotation": s, "weight_change": 0.0, "training": 0.0, "detail": d}


def calculate_other_score(horse, all_horses):
    load = horse.get("load_weight", 0)
    if not load or not all_horses:
        return 2.0, {"load_weight": 2, "note": 0, "detail": "斤量データなし2点"}
    all_loads = [h.get("load_weight", 0) for h in all_horses if h.get("load_weight", 0) > 0]
    avg_load = sum(all_loads) / len(all_loads) if all_loads else 54
    diff = avg_load - load
    if diff >= 2:
        s, d = 3.0, f"斤量{load}kg(平均-{abs(diff):.1f}kg)3点"
    elif diff <= -2:
        s, d = 1.0, f"斤量{load}kg(平均+{abs(diff):.1f}kg)1点"
    else:
        s, d = 2.0, f"斤量{load}kg(平均±{abs(diff):.1f}kg)2点"
    return s, {"load_weight": s, "note": 0, "detail": d}


def score_rule_based(data: dict) -> dict:
    """全馬のルールベーススコアを計算"""
    race_info = data.get("race", {})
    horses = data.get("horses", [])
    race_date = race_info.get("date", "")

    for horse in horses:
        ab_s, ab_b = calculate_ability_score(horse, race_info)
        jk_s, jk_d = calculate_jockey_score(horse)
        ft_s, ft_b = calculate_fitness_score(horse, race_info)
        fm_s, fm_b = calculate_form_score(horse, race_date)
        ot_s, ot_b = calculate_other_score(horse, horses)

        total = round(ab_s + jk_s + ft_s + fm_s + ot_s, 1)
        horse["score"] = total
        horse["score_breakdown"] = {
            "ability": round(ab_s, 1), "jockey": round(jk_s, 1),
            "fitness": round(ft_s, 1), "form": round(fm_s, 1),
            "other": round(ot_s, 1),
        }
        horse["note"] = " ".join([
            "[AUTO]",
            f"ability={ab_s:.1f}({ab_b['detail']})",
            f"jockey={jk_s:.1f}({jk_d})",
            f"fitness={ft_s:.1f}({ft_b['detail']})",
            f"form={fm_s:.1f}({fm_b['detail']})",
            f"other={ot_s:.1f}({ot_b['detail']})",
        ])

    return data
