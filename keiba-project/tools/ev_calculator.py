#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
期待値計算ツール v0.1
===================
base_scored.jsonから期待値を計算し、買い目リストを生成

使い方:
  python ev_calculator.py ../data/races/YYYYMMDD_会場_レース名_base_scored.json
"""

VERSION = "0.1"

import sys
import json
import math
from pathlib import Path
from itertools import combinations
from datetime import datetime

# Windows環境での文字化け対策
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


# ============================================================
# 確率計算（ソフトマックス変換）
# ============================================================

def softmax(scores: list, temperature: float) -> list:
    """
    スコアをソフトマックス変換で確率に変換
    Args:
        scores: 評価点のリスト
        temperature: 温度パラメータ（高いほど均一化）
    Returns:
        確率のリスト（合計1.0）
    """
    max_score = max(scores)
    exps = [math.exp((s - max_score) / temperature) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]


def calc_place_probs(win_probs: list) -> list:
    """
    1着確率から複勝（3着内）確率を計算
    Args:
        win_probs: 1着確率のリスト
    Returns:
        複勝確率のリスト
    """
    n = len(win_probs)
    place_probs = [0.0] * n

    for i in range(n):
        # 自分が1着
        p_top3 = win_probs[i]

        # 自分が2着（他馬が1着）
        for j in range(n):
            if j == i:
                continue
            if win_probs[j] < 1.0:
                p_top3 += win_probs[j] * (win_probs[i] / (1 - win_probs[j]))

        # 自分が3着（他2頭が1-2着）
        for j in range(n):
            if j == i:
                continue
            for k in range(n):
                if k == i or k == j:
                    continue
                denom1 = 1 - win_probs[j]
                denom2 = 1 - win_probs[j] - win_probs[k]
                if denom1 > 0 and denom2 > 0:
                    p_k_2nd = win_probs[k] / denom1
                    p_i_3rd = win_probs[i] / denom2
                    p_top3 += win_probs[j] * p_k_2nd * p_i_3rd

        place_probs[i] = min(p_top3, 1.0)

    return place_probs


def calc_quinella_prob(win_probs: list, i: int, j: int) -> float:
    """
    馬連の的中確率を計算
    Args:
        win_probs: 1着確率のリスト
        i, j: 対象馬のインデックス
    Returns:
        的中確率
    """
    # i-j（iが1着、jが2着）
    prob_ij = win_probs[i] * (win_probs[j] / (1 - win_probs[i])) if win_probs[i] < 1.0 else 0
    # j-i（jが1着、iが2着）
    prob_ji = win_probs[j] * (win_probs[i] / (1 - win_probs[j])) if win_probs[j] < 1.0 else 0
    return prob_ij + prob_ji


def calc_wide_prob(win_probs: list, i: int, j: int) -> float:
    """
    ワイドの的中確率を計算
    Args:
        win_probs: 1着確率のリスト
        i, j: 対象馬のインデックス
    Returns:
        的中確率
    """
    n = len(win_probs)
    prob = 0.0

    # iが1着のときjが2-3着
    if win_probs[i] < 1.0:
        p_j_top2_given_i = win_probs[j] / (1 - win_probs[i])
        for k in range(n):
            if k == i or k == j:
                continue
            denom1 = 1 - win_probs[i]
            denom2 = 1 - win_probs[i] - win_probs[k]
            if denom1 > 0 and denom2 > 0:
                p_j_top2_given_i += (win_probs[k] / denom1) * (win_probs[j] / denom2)
        prob += win_probs[i] * p_j_top2_given_i

    # jが1着のときiが2-3着
    if win_probs[j] < 1.0:
        p_i_top2_given_j = win_probs[i] / (1 - win_probs[j])
        for k in range(n):
            if k == i or k == j:
                continue
            denom1 = 1 - win_probs[j]
            denom2 = 1 - win_probs[j] - win_probs[k]
            if denom1 > 0 and denom2 > 0:
                p_i_top2_given_j += (win_probs[k] / denom1) * (win_probs[i] / denom2)
        prob += win_probs[j] * p_i_top2_given_j

    # kが1着のときi,jが2-3着
    for k in range(n):
        if k == i or k == j:
            continue
        denom = 1 - win_probs[k]
        if denom > 0:
            denom_i = denom - win_probs[i]
            denom_j = denom - win_probs[j]
            if denom_i > 0 and denom_j > 0:
                term = (win_probs[i] / denom) * (win_probs[j] / denom_i) + \
                       (win_probs[j] / denom) * (win_probs[i] / denom_j)
                prob += win_probs[k] * term

    return min(prob, 1.0)


def calc_trio_prob(win_probs: list, i: int, j: int, k: int) -> float:
    """
    3連複の的中確率を計算
    Args:
        win_probs: 1着確率のリスト
        i, j, k: 対象馬のインデックス
    Returns:
        的中確率
    """
    perms = [(i, j, k), (i, k, j), (j, i, k), (j, k, i), (k, i, j), (k, j, i)]
    prob = 0.0

    for a, b, c in perms:
        denom1 = 1 - win_probs[a]
        denom2 = 1 - win_probs[a] - win_probs[b]
        if denom1 > 0 and denom2 > 0:
            prob += win_probs[a] * (win_probs[b] / denom1) * (win_probs[c] / denom2)

    return prob


# ============================================================
# 期待値計算
# ============================================================

def calculate_ev_from_scored(data: dict) -> dict:
    """
    base_scored.jsonから期待値を計算
    Args:
        data: input.json形式のデータ（スコア付き）
    Returns:
        期待値計算結果
    """
    horses = data.get("horses", [])
    combo_odds = data.get("combo_odds", {})
    parameters = data.get("parameters", {})

    temperature = parameters.get("temperature", 10)
    budget = parameters.get("budget", 1500)
    top_n = parameters.get("top_n", 6)

    # スコアから確率を計算
    scores = [h.get("score", 0) for h in horses]
    win_probs = softmax(scores, temperature)
    place_probs = calc_place_probs(win_probs)

    # 馬にインデックス・確率を付与
    for idx, horse in enumerate(horses):
        horse["index"] = idx
        horse["win_prob"] = win_probs[idx]
        horse["place_prob"] = place_probs[idx]

    # スコアでソート（上位top_n頭を対象）
    ranked = sorted(horses, key=lambda h: h.get("score", 0), reverse=True)
    top_horses = ranked[:top_n]
    top_indices = [h["index"] for h in top_horses]

    # 単勝・複勝EV計算
    win_bets = []
    place_bets = []
    for horse in ranked:
        odds_win = horse.get("odds_win", 0)
        odds_place = horse.get("odds_place", 0)
        win_prob = horse["win_prob"]
        place_prob = horse["place_prob"]

        ev_win = odds_win * win_prob
        ev_place = odds_place * place_prob

        win_bets.append({
            "num": horse["num"],
            "name": horse["name"],
            "prob": win_prob,
            "odds": odds_win,
            "ev": ev_win,
            "ev_ratio": ev_win,
            "rank": get_ev_rank(ev_win),
        })

        place_bets.append({
            "num": horse["num"],
            "name": horse["name"],
            "prob": place_prob,
            "odds": odds_place,
            "ev": ev_place,
            "ev_ratio": ev_place,
            "rank": get_ev_rank(ev_place),
        })

    # 馬連EV計算（実オッズ使用）
    quinella_odds_map = {}
    for item in combo_odds.get("quinella", []):
        combo = tuple(sorted(item["combo"]))
        quinella_odds_map[combo] = item["odds"]

    quinella_bets = []
    for a, b in combinations(top_indices, 2):
        i, j = min(a, b), max(a, b)
        combo = (horses[i]["num"], horses[j]["num"])

        if combo not in quinella_odds_map:
            continue

        prob = calc_quinella_prob(win_probs, i, j)
        odds = quinella_odds_map[combo]
        ev = odds * prob

        quinella_bets.append({
            "combo": f"{horses[i]['num']}-{horses[j]['num']}",
            "names": f"{horses[i]['name']}-{horses[j]['name']}",
            "prob": prob,
            "odds": odds,
            "ev": ev,
            "ev_ratio": ev,
            "rank": get_ev_rank(ev),
        })

    quinella_bets.sort(key=lambda x: x["ev"], reverse=True)

    # ワイドEV計算（実オッズ使用）
    wide_odds_map = {}
    for item in combo_odds.get("wide", []):
        combo = tuple(sorted(item["combo"]))
        # ワイドはオッズ範囲があるので中央値を使用
        if isinstance(item["odds"], list):
            odds_val = sum(item["odds"]) / 2
        else:
            odds_val = item["odds"]
        wide_odds_map[combo] = odds_val

    wide_bets = []
    for a, b in combinations(top_indices, 2):
        i, j = min(a, b), max(a, b)
        combo = (horses[i]["num"], horses[j]["num"])

        if combo not in wide_odds_map:
            continue

        prob = calc_wide_prob(win_probs, i, j)
        odds = wide_odds_map[combo]
        ev = odds * prob

        wide_bets.append({
            "combo": f"{horses[i]['num']}-{horses[j]['num']}",
            "names": f"{horses[i]['name']}-{horses[j]['name']}",
            "prob": prob,
            "odds": odds,
            "ev": ev,
            "ev_ratio": ev,
            "rank": get_ev_rank(ev),
        })

    wide_bets.sort(key=lambda x: x["ev"], reverse=True)

    # 3連複EV計算（実オッズ使用）
    trio_odds_map = {}
    for item in combo_odds.get("trio", []):
        combo = tuple(sorted(item["combo"]))
        trio_odds_map[combo] = item["odds"]

    # 3連複は組み合わせが多いのでtop_n内で計算
    trio_top = min(top_n, 7)  # 最大7頭まで
    trio_indices = top_indices[:trio_top]

    trio_bets = []
    for a, b, c in combinations(trio_indices, 3):
        i, j, k = sorted([a, b, c])
        combo = (horses[i]["num"], horses[j]["num"], horses[k]["num"])

        if combo not in trio_odds_map:
            continue

        prob = calc_trio_prob(win_probs, i, j, k)
        odds = trio_odds_map[combo]
        ev = odds * prob

        trio_bets.append({
            "combo": f"{horses[i]['num']}-{horses[j]['num']}-{horses[k]['num']}",
            "names": f"{horses[i]['name']}-{horses[j]['name']}-{horses[k]['name']}",
            "prob": prob,
            "odds": odds,
            "ev": ev,
            "ev_ratio": ev,
            "rank": get_ev_rank(ev),
        })

    trio_bets.sort(key=lambda x: x["ev"], reverse=True)

    # 確信度計算（上位3頭の確率合計）
    confidence = sum(h["win_prob"] for h in ranked[:3])

    return {
        "win": win_bets,
        "place": place_bets,
        "quinella": quinella_bets,
        "wide": wide_bets,
        "trio": trio_bets,
        "confidence": confidence,
        "temperature": temperature,
        "budget": budget,
    }


def get_ev_rank(ev_ratio: float) -> str:
    """
    EV率からランクを返す
    Args:
        ev_ratio: EV率（= EV / 掛金）
    Returns:
        S/A/B/Cランク
    """
    if ev_ratio >= 1.5:
        return "S"
    elif ev_ratio >= 1.2:
        return "A"
    elif ev_ratio >= 1.0:
        return "B"
    else:
        return "C"


# ============================================================
# 出力
# ============================================================

def print_ev_results(results: dict, race_info: dict):
    """
    期待値計算結果を整形して表示
    Args:
        results: calculate_ev_from_scored()の戻り値
        race_info: レース情報
    """
    print("=" * 60)
    print(f"  期待値計算 v{VERSION}")
    print("=" * 60)
    print()
    print(f"レース: {race_info.get('venue', '')} {race_info.get('race_number', 0)}R {race_info.get('name', '')}")
    print(f"グレード: {race_info.get('grade', '')}")
    print(f"温度パラメータ: {results['temperature']}")
    print(f"予算: ¥{results['budget']:,}")
    print(f"確信度: {results['confidence']:.1%}")
    print()

    # 単勝EV
    print("[単勝] トップ5")
    for i, bet in enumerate(results["win"][:5], 1):
        print(f"  [{i}] {bet['num']:2}番 {bet['name']:12} "
              f"{bet['odds']:5.1f}倍 × {bet['prob']:5.1%} = EV {bet['ev']:.2f} ({bet['rank']}級)")
    print()

    # 複勝EV
    print("[複勝] トップ5")
    for i, bet in enumerate(results["place"][:5], 1):
        print(f"  [{i}] {bet['num']:2}番 {bet['name']:12} "
              f"{bet['odds']:5.1f}倍 × {bet['prob']:5.1%} = EV {bet['ev']:.2f} ({bet['rank']}級)")
    print()

    # 馬連EV
    print("[馬連] トップ5")
    for i, bet in enumerate(results["quinella"][:5], 1):
        print(f"  [{i}] {bet['combo']:8} {bet['names']:30} "
              f"{bet['odds']:6.1f}倍 × {bet['prob']:5.1%} = EV {bet['ev']:.2f} ({bet['rank']}級)")
    print()

    # ワイドEV
    print("[ワイド] トップ5")
    for i, bet in enumerate(results["wide"][:5], 1):
        print(f"  [{i}] {bet['combo']:8} {bet['names']:30} "
              f"{bet['odds']:6.1f}倍 × {bet['prob']:5.1%} = EV {bet['ev']:.2f} ({bet['rank']}級)")
    print()

    # 3連複EV
    print("[3連複] トップ5")
    for i, bet in enumerate(results["trio"][:5], 1):
        print(f"  [{i}] {bet['combo']:12} "
              f"{bet['odds']:7.1f}倍 × {bet['prob']:5.1%} = EV {bet['ev']:.2f} ({bet['rank']}級)")
    print()

    # 推奨買い目（EV 1.0以上）
    print("=" * 60)
    print("  推奨買い目（EV 1.0以上）")
    print("=" * 60)

    recommendations = []
    for bet_type, bets in [("単勝", results["win"]), ("複勝", results["place"]),
                           ("馬連", results["quinella"]), ("ワイド", results["wide"]),
                           ("3連複", results["trio"])]:
        for bet in bets:
            if bet["ev"] >= 1.0:
                recommendations.append({
                    "type": bet_type,
                    "bet": bet,
                })

    if not recommendations:
        print("\n  [見送り推奨] EV 1.0以上の買い目がありません")
    else:
        recommendations.sort(key=lambda x: x["bet"]["ev"], reverse=True)
        for i, rec in enumerate(recommendations[:10], 1):
            bet = rec["bet"]
            if "combo" in bet:
                combo = bet["combo"]
            else:
                combo = f"{bet['num']}番"
            print(f"  [{i}] {rec['type']:6} {combo:12} EV {bet['ev']:.2f} ({bet['rank']}級)")

    print()


def save_ev_output(results: dict, data: dict, output_path: str):
    """
    期待値計算結果をJSONで保存
    Args:
        results: calculate_ev_from_scored()の戻り値
        data: 元のinput.jsonデータ
        output_path: 出力ファイルパス
    """
    output = {
        "race": data.get("race", {}),
        "parameters": data.get("parameters", {}),
        "horses": data.get("horses", []),
        "combo_odds": data.get("combo_odds", {}),
        "ev_results": results,
        "timestamp": datetime.now().isoformat(),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"[OK] 保存完了: {output_path}")


# ============================================================
# メイン
# ============================================================

def main():
    if len(sys.argv) < 2:
        print("使い方: python ev_calculator.py <base_scored.json>")
        sys.exit(1)

    input_path = sys.argv[1]

    try:
        # データ読み込み
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 期待値計算
        results = calculate_ev_from_scored(data)

        # 結果表示
        print_ev_results(results, data.get("race", {}))

        # JSON保存
        input_file = Path(input_path)
        output_path = input_file.parent / input_file.name.replace("_base_scored.json", "_ev_results.json")
        if output_path == input_file:
            output_path = input_file.parent / (input_file.stem + "_ev_results.json")

        save_ev_output(results, data, str(output_path))

    except FileNotFoundError:
        print(f"[ERROR] ファイルが見つかりません: {input_path}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"[ERROR] JSONパースエラー: {input_path}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] エラー: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
