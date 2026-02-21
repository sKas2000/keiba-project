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

from config.settings import (
    FEATURE_COLUMNS, GRADE_DEFAULTS, MODEL_DIR,
    EXPANDING_BEST_PARAMS, CLASS_MAP,
)
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
    """ML モデルで予測してスコア付与

    binary_model: 3着以内確率（ソート・スコア用）
    win_model: 勝率直接推定（EV計算のwin_prob用）
    Isotonic校正があればPlatt Scalingより優先
    """
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

    # キャリブレーション: Isotonic優先 > Platt Scaling > raw
    from src.model.trainer import (
        load_calibrator, calibrate_probs,
        load_isotonic_calibrator, calibrate_isotonic,
    )
    iso_binary = load_isotonic_calibrator("binary_isotonic", model_dir)
    platt_cal = load_calibrator(model_dir)
    if iso_binary is not None:
        probs = calibrate_isotonic(raw_probs, iso_binary)
        cal_method = "isotonic"
    elif platt_cal is not None:
        probs = calibrate_probs(raw_probs, platt_cal)
        cal_method = "platt"
    else:
        probs = raw_probs
        cal_method = "raw"

    # 勝率直接推定モデル（Expanding Window検証で使用）
    win_model_path = model_dir / "win_model.txt"
    win_probs_direct = None
    if win_model_path.exists():
        win_model = lgb.Booster(model_file=str(win_model_path))
        raw_win = win_model.predict(X)
        iso_win = load_isotonic_calibrator("win_isotonic", model_dir)
        if iso_win is not None:
            win_probs_direct = calibrate_isotonic(raw_win, iso_win)
        else:
            win_probs_direct = raw_win

    horses = data.get("horses", [])
    for i, horse in enumerate(horses):
        if i < len(probs):
            prob = float(probs[i])
            horse["score"] = round(prob * 100, 1)
            horse["ml_top3_prob"] = round(prob, 4)
            horse["ml_calibrated"] = cal_method != "raw"
            if win_probs_direct is not None and i < len(win_probs_direct):
                horse["ml_win_prob"] = round(float(win_probs_direct[i]), 6)
            horse["score_breakdown"] = {
                "ml_binary": round(prob * 100, 1),
                "ability": 0, "jockey": 0, "fitness": 0, "form": 0, "other": 0,
            }
            horse["note"] = f"[ML:{cal_method}] top3={prob:.3f}"
            if win_probs_direct is not None and i < len(win_probs_direct):
                horse["note"] += f" win={win_probs_direct[i]:.4f}"

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


def _grade_to_class_code(grade: str) -> int:
    """レースグレード文字列をクラスコードに変換"""
    if not grade:
        return 0
    return CLASS_MAP.get(grade, 0)


def calculate_ev(data: dict, strategy: dict = None) -> dict:
    """base_scored.json から期待値を計算

    Args:
        data: scored JSON data
        strategy: 戦略パラメータ (None=EXPANDING_BEST_PARAMS使用)
            - confidence_min: 確信度フィルタ（Top1-Top2のwin_prob差）
            - quinella_top_n: 馬連用top_n
            - wide_top_n: ワイド用top_n
            - skip_classes: スキップするクラスコードリスト
            - top_n: 上位N頭（デフォルト3）
    """
    horses = data.get("horses", [])
    combo_odds = data.get("combo_odds", {})
    params = data.get("parameters", {})
    temperature = params.get("temperature", 10)
    budget = params.get("budget", 1500)

    # 戦略パラメータ（ML時はEXPANDING_BEST_PARAMSをデフォルト使用）
    is_ml = any(h.get("ml_top3_prob") is not None for h in horses)
    if strategy is None and is_ml:
        strategy = EXPANDING_BEST_PARAMS.copy()
    strategy = strategy or {}

    top_n = strategy.get("top_n", params.get("top_n", 6))
    confidence_min = strategy.get("confidence_min", 0)
    quinella_top_n = strategy.get("quinella_top_n", 0) or top_n
    wide_top_n = strategy.get("wide_top_n", 0) or top_n
    skip_classes = strategy.get("skip_classes", [])

    # 警告フラグ
    warnings = []

    # クラスチェック
    race_grade = data.get("race", {}).get("grade", "")
    race_class_code = _grade_to_class_code(race_grade)
    if skip_classes and race_class_code in skip_classes:
        class_names = {4: "2勝", 6: "OP"}
        skipped = [class_names.get(c, str(c)) for c in skip_classes]
        warnings.append(f"このクラス({race_grade})はバックテストで不利（{'/'.join(skipped)}クラス除外推奨）")

    # MLモード: win_prob_directがあれば使用（Expanding Window方式）
    # なければキャリブレーション済みtop3_probを正規化
    is_calibrated = any(h.get("ml_calibrated") for h in horses)
    has_win_model = any(h.get("ml_win_prob") is not None for h in horses)

    if is_ml and has_win_model:
        # Expanding Window方式: win_modelのIsotonic校正済み確率を正規化
        wprobs = [max(h.get("ml_win_prob", 0.001), 0.001) for h in horses]
        total = sum(wprobs)
        win_probs = [p / total for p in wprobs]
    elif is_ml and is_calibrated:
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

    # 確信度チェック
    confidence_gap = win_probs[ranked[0]["index"]] - win_probs[ranked[1]["index"]] if len(ranked) >= 2 else 0
    low_confidence = False
    if confidence_min > 0 and confidence_gap < confidence_min:
        low_confidence = True
        warnings.append(
            f"確信度不足 (Top1-Top2差={confidence_gap:.3f} < {confidence_min})"
            " → ベット見送り推奨"
        )

    # 券種別top_n
    effective_top = max(top_n, quinella_top_n, wide_top_n)
    top_indices = [h["index"] for h in ranked[:effective_top]]
    win_top = [h["index"] for h in ranked[:top_n]]

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

    # 馬連（quinella_top_n で制御）
    q_map = {tuple(sorted(i["combo"])): i["odds"] for i in combo_odds.get("quinella", [])}
    q_indices = [h["index"] for h in ranked[:quinella_top_n]]
    quinella_bets = []
    for a, b in combinations(q_indices, 2):
        i, j = min(a, b), max(a, b)
        combo = (horses[i]["num"], horses[j]["num"])
        if combo not in q_map:
            continue
        prob = calc_quinella_prob(win_probs, i, j)
        ev = q_map[combo] * prob
        quinella_bets.append({"combo": f"{combo[0]}-{combo[1]}", "names": f"{horses[i]['name']}-{horses[j]['name']}", "prob": prob, "odds": q_map[combo], "ev": ev, "ev_ratio": ev, "rank": get_ev_rank(ev)})
    quinella_bets.sort(key=lambda x: x["ev"], reverse=True)

    # ワイド（wide_top_n で制御）
    w_map = {}
    for item in combo_odds.get("wide", []):
        combo = tuple(sorted(item["combo"]))
        w_map[combo] = sum(item["odds"]) / 2 if isinstance(item["odds"], list) else item["odds"]
    w_indices = [h["index"] for h in ranked[:wide_top_n]]
    wide_bets = []
    for a, b in combinations(w_indices, 2):
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
    trio_indices = [h["index"] for h in ranked[:min(top_n, 7)]]
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
        "confidence": confidence, "confidence_gap": confidence_gap,
        "low_confidence": low_confidence,
        "temperature": temperature, "budget": budget,
        "strategy": {
            "top_n": top_n, "quinella_top_n": quinella_top_n,
            "wide_top_n": wide_top_n, "confidence_min": confidence_min,
            "skip_classes": skip_classes,
            "has_win_model": has_win_model,
        },
        "warnings": warnings,
    }


def print_ev_results(results: dict, race_info: dict):
    """期待値結果を表示"""
    print(f"\n{'=' * 60}")
    print(f"  {race_info.get('venue', '')} {race_info.get('race_number', 0)}R {race_info.get('name', '')}")
    print(f"{'=' * 60}")

    strategy = results.get("strategy", {})
    has_win_model = strategy.get("has_win_model", False)
    model_label = "ML(win_model+isotonic)" if has_win_model else f"温度={results['temperature']}"
    print(f"  モデル: {model_label}")
    print(f"  確信度: {results['confidence']:.1%} (Top1-Top2差: {results.get('confidence_gap', 0):.3f})")

    if strategy.get("quinella_top_n"):
        print(f"  馬連: Top{strategy['quinella_top_n']}  ワイド: Top{strategy.get('wide_top_n', 3)}")

    # 警告表示
    warnings = results.get("warnings", [])
    for w in warnings:
        print(f"  [!] {w}")
    print()

    # 馬連・ワイドは「推奨」、3連複は「参考」
    # 3-way split検証: 馬連 Val110%/Test111%, ワイド Test107%
    # 3連複は Val111%だがTest93% → 参考扱い
    bet_sections = [
        ("馬連", "quinella", "推奨"),
        ("ワイド", "wide", "推奨"),
        ("3連複", "trio", "参考"),
        ("単勝", "win", "参考"),
        ("複勝", "place", "参考"),
    ]

    for label, key, tier in bet_sections:
        bets = results.get(key, [])
        tag = f"[{tier}]" if strategy else ""
        print(f"[{label}] {tag} トップ5")
        for i, bet in enumerate(bets[:5], 1):
            combo = bet.get("combo", f"{bet.get('num', '?')}番")
            ev_mark = " *" if bet['ev'] >= 1.0 else ""
            print(f"  [{i}] {combo:12} {bet['odds']:7.1f}倍 x {bet['prob']:5.1%} = EV {bet['ev']:.2f} ({bet['rank']}級){ev_mark}")
        print()

    # 推奨買い目まとめ
    print("=" * 60)
    if results.get("low_confidence"):
        print("  [見送り推奨] 確信度不足 — 全券種ベット見送り")
        print("=" * 60)
        print()
        return

    # 推奨券種（馬連・ワイド）のEV>=1.0
    print("  推奨買い目（馬連・ワイド: EV 1.0以上）")
    print("-" * 60)
    recs = []
    for bt, bets, tier in [("馬連", results.get("quinella", []), "推奨"),
                            ("ワイド", results.get("wide", []), "推奨")]:
        for bet in bets:
            if bet["ev"] >= 1.0:
                recs.append((bt, bet, tier))

    if not recs:
        print("  該当なし")
    else:
        recs.sort(key=lambda x: x[1]["ev"], reverse=True)
        for i, (bt, bet, tier) in enumerate(recs[:10], 1):
            combo = bet.get("combo", f"{bet.get('num', '?')}番")
            print(f"  [{i}] {bt:6} {combo:12} EV {bet['ev']:.2f} ({bet['rank']}級)")

    # 参考券種（3連複）
    trio_recs = [bet for bet in results.get("trio", []) if bet["ev"] >= 1.0]
    if trio_recs:
        print()
        print("  参考買い目（3連複: Testで不利、少額推奨）")
        print("-" * 60)
        for i, bet in enumerate(trio_recs[:5], 1):
            combo = bet.get("combo", f"{bet.get('num', '?')}番")
            print(f"  [{i}] 3連複 {combo:12} EV {bet['ev']:.2f} ({bet['rank']}級)")

    print("=" * 60)
    print()
