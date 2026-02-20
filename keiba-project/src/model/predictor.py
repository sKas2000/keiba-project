"""
予測モジュール
ルールベーススコアリング + ML予測 + 期待値計算
"""
import json
import math
import re
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np

from config.settings import FEATURE_COLUMNS, GRADE_DEFAULTS, MODEL_DIR
from src.scraping.parsers import safe_int, safe_float


# ============================================================
# ルールベーススコアリング (scoring_engine.py 移行)
# ============================================================

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

    # 会場適性
    vs = 0.0
    vd = "同場なし"
    if past and venue:
        same = [r for r in past if r.get("venue", "") == venue]
        if same:
            avg = sum(r.get("finish", 99) for r in same) / len(same)
            vs = round(max(0, min(8, 8 * (10 - avg) / 9)), 1)
            vd = f"同場{len(same)}戦平均{avg:.1f}着"

    # 距離適性
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


# ============================================================
# ML 予測 (predictor.py 移行)
# ============================================================

def score_ml(data: dict, model_dir: Path = None) -> dict | None:
    """ML モデルで予測してスコア付与"""
    import lightgbm as lgb
    from src.data.feature import extract_features_from_enriched

    model_dir = model_dir or MODEL_DIR
    binary_model_path = model_dir / "binary_model.txt"
    if not binary_model_path.exists():
        print(f"[WARN] モデル未学習: {binary_model_path}")
        return None

    binary_model = lgb.Booster(model_file=str(binary_model_path))

    meta_path = model_dir / "binary_meta.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    feature_names = meta.get("feature_names", FEATURE_COLUMNS)

    df = extract_features_from_enriched(data)
    available = [c for c in feature_names if c in df.columns]
    for col in [c for c in feature_names if c not in df.columns]:
        df[col] = 0.0
    X = df[feature_names].values.astype(np.float32)

    raw_probs = binary_model.predict(X)

    # Platt Scaling キャリブレーション（利用可能な場合）
    from src.model.trainer import load_calibrator, calibrate_probs
    calibrator = load_calibrator(model_dir)
    if calibrator is not None:
        probs = calibrate_probs(raw_probs, calibrator)
    else:
        probs = raw_probs

    rank_model_path = model_dir / "ranking_model.txt"
    rank_scores = None
    if rank_model_path.exists():
        rank_model = lgb.Booster(model_file=str(rank_model_path))
        rank_scores = rank_model.predict(X)

    horses = data.get("horses", [])
    for i, horse in enumerate(horses):
        if i < len(probs):
            prob = float(probs[i])
            horse["score"] = round(prob * 100, 1)
            horse["ml_top3_prob"] = round(prob, 4)
            horse["ml_calibrated"] = calibrator is not None
            if rank_scores is not None and i < len(rank_scores):
                horse["ml_rank_score"] = round(float(rank_scores[i]), 4)
            horse["score_breakdown"] = {
                "ml_binary": round(prob * 100, 1),
                "ability": 0, "jockey": 0, "fitness": 0, "form": 0, "other": 0,
            }
            cal_label = "calibrated" if calibrator is not None else "raw"
            horse["note"] = f"[ML:{cal_label}] top3_prob={prob:.3f}"

    return data


# ============================================================
# EV 計算 (ev_calculator.py 移行)
# ============================================================

def softmax(scores: list, temperature: float) -> list:
    max_score = max(scores)
    exps = [math.exp((s - max_score) / temperature) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]


def calc_place_probs(win_probs: list) -> list:
    n = len(win_probs)
    place_probs = [0.0] * n
    for i in range(n):
        p = win_probs[i]
        for j in range(n):
            if j == i:
                continue
            if win_probs[j] < 1.0:
                p += win_probs[j] * (win_probs[i] / (1 - win_probs[j]))
        for j in range(n):
            if j == i:
                continue
            for k in range(n):
                if k == i or k == j:
                    continue
                d1 = 1 - win_probs[j]
                d2 = 1 - win_probs[j] - win_probs[k]
                if d1 > 0 and d2 > 0:
                    p += win_probs[j] * (win_probs[k] / d1) * (win_probs[i] / d2)
        place_probs[i] = min(p, 1.0)
    return place_probs


def calc_quinella_prob(win_probs, i, j):
    p_ij = win_probs[i] * (win_probs[j] / (1 - win_probs[i])) if win_probs[i] < 1.0 else 0
    p_ji = win_probs[j] * (win_probs[i] / (1 - win_probs[j])) if win_probs[j] < 1.0 else 0
    return p_ij + p_ji


def calc_wide_prob(win_probs, i, j):
    n = len(win_probs)
    prob = 0.0
    if win_probs[i] < 1.0:
        p = win_probs[j] / (1 - win_probs[i])
        for k in range(n):
            if k == i or k == j:
                continue
            d1 = 1 - win_probs[i]
            d2 = 1 - win_probs[i] - win_probs[k]
            if d1 > 0 and d2 > 0:
                p += (win_probs[k] / d1) * (win_probs[j] / d2)
        prob += win_probs[i] * p
    if win_probs[j] < 1.0:
        p = win_probs[i] / (1 - win_probs[j])
        for k in range(n):
            if k == i or k == j:
                continue
            d1 = 1 - win_probs[j]
            d2 = 1 - win_probs[j] - win_probs[k]
            if d1 > 0 and d2 > 0:
                p += (win_probs[k] / d1) * (win_probs[i] / d2)
        prob += win_probs[j] * p
    for k in range(n):
        if k == i or k == j:
            continue
        d = 1 - win_probs[k]
        if d > 0:
            di = d - win_probs[i]
            dj = d - win_probs[j]
            if di > 0 and dj > 0:
                t = (win_probs[i] / d) * (win_probs[j] / di) + (win_probs[j] / d) * (win_probs[i] / dj)
                prob += win_probs[k] * t
    return min(prob, 1.0)


def calc_trio_prob(win_probs, i, j, k):
    perms = [(i, j, k), (i, k, j), (j, i, k), (j, k, i), (k, i, j), (k, j, i)]
    prob = 0.0
    for a, b, c in perms:
        d1 = 1 - win_probs[a]
        d2 = 1 - win_probs[a] - win_probs[b]
        if d1 > 0 and d2 > 0:
            prob += win_probs[a] * (win_probs[b] / d1) * (win_probs[c] / d2)
    return prob


def get_ev_rank(ev_ratio: float) -> str:
    if ev_ratio >= 1.5:
        return "S"
    elif ev_ratio >= 1.2:
        return "A"
    elif ev_ratio >= 1.0:
        return "B"
    return "C"


def calculate_ev(data: dict) -> dict:
    """base_scored.json から期待値を計算"""
    horses = data.get("horses", [])
    combo_odds = data.get("combo_odds", {})
    params = data.get("parameters", {})
    temperature = params.get("temperature", 10)
    budget = params.get("budget", 1500)
    top_n = params.get("top_n", 6)

    # MLモード: キャリブレーション済み確率をそのまま正規化（温度不要）
    # 未キャリブレーション: logit変換 + 温度 softmax
    # ルールベース: スコア（0-100）をそのままsoftmax
    is_ml = any(h.get("ml_top3_prob") is not None for h in horses)
    is_calibrated = any(h.get("ml_calibrated") for h in horses)
    if is_ml and is_calibrated:
        # キャリブレーション済み: 正規化のみ
        probs = [max(h.get("ml_top3_prob", 0.001), 0.001) for h in horses]
        total = sum(probs)
        win_probs = [p / total for p in probs]
    elif is_ml:
        probs = [max(min(h.get("ml_top3_prob", 0.5), 0.999), 0.001) for h in horses]
        logits = [math.log(p / (1 - p)) for p in probs]
        win_probs = softmax(logits, temperature)
    else:
        scores = [h.get("score", 0) for h in horses]
        win_probs = softmax(scores, temperature)
    place_probs = calc_place_probs(win_probs)

    for idx, horse in enumerate(horses):
        horse["index"] = idx
        horse["win_prob"] = win_probs[idx]
        horse["place_prob"] = place_probs[idx]

    ranked = sorted(horses, key=lambda h: h.get("score", 0), reverse=True)
    top_indices = [h["index"] for h in ranked[:top_n]]

    # 単勝・複勝
    win_bets, place_bets = [], []
    for horse in ranked:
        ow = horse.get("odds_win", 0)
        op = horse.get("odds_place", 0)
        wp = horse["win_prob"]
        pp = horse["place_prob"]
        ew = ow * wp
        ep = op * pp
        win_bets.append({"num": horse["num"], "name": horse["name"], "prob": wp, "odds": ow, "ev": ew, "ev_ratio": ew, "rank": get_ev_rank(ew)})
        place_bets.append({"num": horse["num"], "name": horse["name"], "prob": pp, "odds": op, "ev": ep, "ev_ratio": ep, "rank": get_ev_rank(ep)})

    # 馬連
    q_map = {tuple(sorted(i["combo"])): i["odds"] for i in combo_odds.get("quinella", [])}
    quinella_bets = []
    for a, b in combinations(top_indices, 2):
        i, j = min(a, b), max(a, b)
        combo = (horses[i]["num"], horses[j]["num"])
        if combo not in q_map:
            continue
        prob = calc_quinella_prob(win_probs, i, j)
        ev = q_map[combo] * prob
        quinella_bets.append({"combo": f"{combo[0]}-{combo[1]}", "names": f"{horses[i]['name']}-{horses[j]['name']}", "prob": prob, "odds": q_map[combo], "ev": ev, "ev_ratio": ev, "rank": get_ev_rank(ev)})
    quinella_bets.sort(key=lambda x: x["ev"], reverse=True)

    # ワイド
    w_map = {}
    for item in combo_odds.get("wide", []):
        combo = tuple(sorted(item["combo"]))
        w_map[combo] = sum(item["odds"]) / 2 if isinstance(item["odds"], list) else item["odds"]
    wide_bets = []
    for a, b in combinations(top_indices, 2):
        i, j = min(a, b), max(a, b)
        combo = (horses[i]["num"], horses[j]["num"])
        if combo not in w_map:
            continue
        prob = calc_wide_prob(win_probs, i, j)
        ev = w_map[combo] * prob
        wide_bets.append({"combo": f"{combo[0]}-{combo[1]}", "names": f"{horses[i]['name']}-{horses[j]['name']}", "prob": prob, "odds": w_map[combo], "ev": ev, "ev_ratio": ev, "rank": get_ev_rank(ev)})
    wide_bets.sort(key=lambda x: x["ev"], reverse=True)

    # 3連複
    t_map = {tuple(sorted(i["combo"])): i["odds"] for i in combo_odds.get("trio", [])}
    trio_indices = top_indices[:min(top_n, 7)]
    trio_bets = []
    for a, b, c in combinations(trio_indices, 3):
        i, j, k = sorted([a, b, c])
        combo = (horses[i]["num"], horses[j]["num"], horses[k]["num"])
        if combo not in t_map:
            continue
        prob = calc_trio_prob(win_probs, i, j, k)
        ev = t_map[combo] * prob
        trio_bets.append({"combo": f"{combo[0]}-{combo[1]}-{combo[2]}", "names": f"{horses[i]['name']}-{horses[j]['name']}-{horses[k]['name']}", "prob": prob, "odds": t_map[combo], "ev": ev, "ev_ratio": ev, "rank": get_ev_rank(ev)})
    trio_bets.sort(key=lambda x: x["ev"], reverse=True)

    confidence = sum(h["win_prob"] for h in ranked[:3])
    return {
        "win": win_bets, "place": place_bets,
        "quinella": quinella_bets, "wide": wide_bets, "trio": trio_bets,
        "confidence": confidence, "temperature": temperature, "budget": budget,
    }


def print_ev_results(results: dict, race_info: dict):
    """期待値結果を表示"""
    print(f"\nレース: {race_info.get('venue', '')} {race_info.get('race_number', 0)}R {race_info.get('name', '')}")
    print(f"温度: {results['temperature']}  確信度: {results['confidence']:.1%}\n")

    for label, key in [("単勝", "win"), ("複勝", "place"), ("馬連", "quinella"), ("ワイド", "wide"), ("3連複", "trio")]:
        print(f"[{label}] トップ5")
        for i, bet in enumerate(results[key][:5], 1):
            combo = bet.get("combo", f"{bet.get('num', '?')}番")
            print(f"  [{i}] {combo:12} {bet['odds']:7.1f}倍 x {bet['prob']:5.1%} = EV {bet['ev']:.2f} ({bet['rank']}級)")
        print()

    # 推奨買い目
    print("=" * 50)
    print("  推奨買い目（EV 1.0以上）")
    print("=" * 50)
    recs = []
    for bt, bets in [("単勝", results["win"]), ("複勝", results["place"]),
                      ("馬連", results["quinella"]), ("ワイド", results["wide"]),
                      ("3連複", results["trio"])]:
        for bet in bets:
            if bet["ev"] >= 1.0:
                recs.append((bt, bet))
    if not recs:
        print("\n  [見送り推奨] EV 1.0以上の買い目がありません")
    else:
        recs.sort(key=lambda x: x[1]["ev"], reverse=True)
        for i, (bt, bet) in enumerate(recs[:10], 1):
            combo = bet.get("combo", f"{bet.get('num', '?')}番")
            print(f"  [{i}] {bt:6} {combo:12} EV {bet['ev']:.2f} ({bet['rank']}級)")
    print()
