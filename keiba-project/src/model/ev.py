"""
期待値(EV)計算 + 確率モデル + 結果表示
"""
import math
from itertools import combinations

from config.settings import EXPANDING_BEST_PARAMS, CLASS_MAP


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
    if not grade:
        return 0
    return CLASS_MAP.get(grade, 0)


def calculate_ev(data: dict, strategy: dict = None) -> dict:
    """base_scored.json から期待値を計算"""
    horses = data.get("horses", [])
    combo_odds = data.get("combo_odds", {})
    params = data.get("parameters", {})
    temperature = params.get("temperature", 10)
    budget = params.get("budget", 1500)

    is_ml = any(h.get("ml_top3_prob") is not None for h in horses)
    if strategy is None and is_ml:
        strategy = EXPANDING_BEST_PARAMS.copy()
    strategy = strategy or {}

    top_n = strategy.get("top_n", params.get("top_n", 6))
    confidence_min = strategy.get("confidence_min", 0)
    quinella_top_n = strategy.get("quinella_top_n", 0) or top_n
    wide_top_n = strategy.get("wide_top_n", 0) or top_n
    skip_classes = strategy.get("skip_classes", [])

    warnings = []

    race_grade = data.get("race", {}).get("grade", "")
    race_class_code = _grade_to_class_code(race_grade)
    if skip_classes and race_class_code in skip_classes:
        class_names = {4: "2勝", 6: "OP"}
        skipped = [class_names.get(c, str(c)) for c in skip_classes]
        warnings.append(f"このクラス({race_grade})はバックテストで不利（{'/'.join(skipped)}クラス除外推奨）")

    is_calibrated = any(h.get("ml_calibrated") for h in horses)
    has_win_model = any(h.get("ml_win_prob") is not None for h in horses)

    if is_ml and has_win_model:
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

    confidence_gap = win_probs[ranked[0]["index"]] - win_probs[ranked[1]["index"]] if len(ranked) >= 2 else 0
    low_confidence = False
    if confidence_min > 0 and confidence_gap < confidence_min:
        low_confidence = True
        warnings.append(
            f"確信度不足 (Top1-Top2差={confidence_gap:.3f} < {confidence_min})"
            " → ベット見送り推奨"
        )

    effective_top = max(top_n, quinella_top_n, wide_top_n)

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

    # ワイド
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

    warnings = results.get("warnings", [])
    for w in warnings:
        print(f"  [!] {w}")
    print()

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

    print("=" * 60)
    if results.get("low_confidence"):
        print("  [見送り推奨] 確信度不足 — 全券種ベット見送り")
        print("=" * 60)
        print()
        return

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
