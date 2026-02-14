#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scoring Engine v0.1
===================
enriched_input.jsonから基礎点を自動算出する

入力: enriched_input.json（過去走データ・騎手成績付き）
出力: base_scored.json（基礎点付き）

使い方:
  python scoring_engine.py <enriched_input.jsonのパス>
"""

VERSION = "0.1"

import json
import re
import sys
from datetime import datetime
from pathlib import Path

# Windows環境での文字化け対策
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


# ============================================================
# ユーティリティ関数
# ============================================================

def safe_int(text: str) -> int:
    """文字列を整数に変換。失敗時は0"""
    try:
        return int(re.sub(r'[^\d]', '', str(text)))
    except (ValueError, AttributeError):
        return 0


def safe_float(text: str) -> float:
    """文字列を浮動小数点に変換。失敗時は0.0"""
    try:
        return float(str(text).strip().replace(",", ""))
    except (ValueError, AttributeError):
        return 0.0


def calculate_days_between(date1_str: str, date2_str: str) -> int:
    """
    2つの日付文字列の日数差を計算

    想定フォーマット: "2026/01/15" or "2026-01-15"
    """
    try:
        # 区切り文字を正規化
        date1_str = str(date1_str).replace("/", "-")
        date2_str = str(date2_str).replace("/", "-")

        d1 = datetime.strptime(date1_str, "%Y-%m-%d")
        d2 = datetime.strptime(date2_str, "%Y-%m-%d")

        return abs((d2 - d1).days)
    except Exception:
        return 0


def is_small_margin(margin_str: str, threshold: float) -> bool:
    """
    着差が閾値以内かを判定

    Args:
        margin_str: 着差文字列（例: "0.3", "1馬身", "ハナ"）
        threshold: 閾値（秒）

    Returns:
        着差が閾値以内ならTrue
    """
    if not margin_str:
        return False

    margin_str = str(margin_str).strip()

    # 秒数の場合
    sec_match = re.match(r'([\d.]+)', margin_str)
    if sec_match and '馬身' not in margin_str:
        try:
            return float(sec_match.group(1)) <= threshold
        except ValueError:
            return False

    # 馬身の場合（1馬身≈0.2秒と換算）
    if '馬身' in margin_str or 'バ身' in margin_str:
        if 'ハナ' in margin_str or 'アタマ' in margin_str or 'クビ' in margin_str:
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


# ============================================================
# 実力(ability)スコア: 0-50点
# ============================================================

def calculate_top3_rate_score(past_races: list) -> tuple[float, str]:
    """
    過去4走の3着内率から0-20点を算出

    Returns:
        (score, detail): スコアと計算詳細
    """
    if not past_races:
        return 0.0, "過去走なし"

    recent_races = past_races[:4]
    top3_count = sum(1 for r in recent_races if r.get("finish", 99) <= 3)
    rate = top3_count / len(recent_races)
    score = rate * 20

    detail = f"3着内率{top3_count}/{len(recent_races)}"
    return round(score, 1), detail


def calculate_margin_score(past_races: list) -> tuple[float, str]:
    """
    過去4走の着差から0-15点を算出

    着差評価基準:
    - 1着: 15点
    - 2着(着差小): 12点、(着差大): 10点
    - 3着(着差小): 9点、(着差大): 7点
    - 4-5着: 6点
    - 6着以降: 3点
    """
    if not past_races:
        return 0.0, "過去走なし"

    recent_races = past_races[:4]
    scores = []

    for race in recent_races:
        finish = race.get("finish", 99)
        margin = race.get("margin", "")

        if finish == 1:
            scores.append(15)
        elif finish == 2:
            # 着差が小さければ12点、それ以外10点
            if is_small_margin(margin, threshold=0.5):
                scores.append(12)
            else:
                scores.append(10)
        elif finish == 3:
            if is_small_margin(margin, threshold=1.0):
                scores.append(9)
            else:
                scores.append(7)
        elif finish <= 5:
            scores.append(6)
        else:
            scores.append(3)

    avg_score = sum(scores) / len(scores) if scores else 0
    detail = f"着差評価{avg_score:.1f}点"
    return round(avg_score, 1), detail


def calculate_course_score(past_races: list, current_venue: str, current_surface: str, current_distance: int) -> tuple[float, str]:
    """
    同コース・同馬場・同距離での成績から0-15点を算出

    完全一致: venue + surface + distance±100m
    点数配分: 1着=5点、2着=3点、3着=2点の平均（上限15点）
    """
    if not past_races or not current_venue or not current_distance:
        return 0.0, "同条件なし"

    matching_races = []
    for race in past_races:
        venue_match = race.get("venue", "") == current_venue

        # 馬場は先頭文字で比較（"ダ" vs "ダート"）
        race_surface = race.get("surface", "")
        surface_match = False
        if current_surface and race_surface:
            surface_match = race_surface[0] == current_surface[0]

        # 距離は±100m以内
        race_dist = safe_int(race.get("distance", "0"))
        distance_match = abs(race_dist - current_distance) <= 100 if race_dist > 0 else False

        if venue_match and surface_match and distance_match:
            matching_races.append(race)

    if not matching_races:
        return 0.0, "同条件なし"

    # 成績から点数算出
    points = []
    for race in matching_races:
        finish = race.get("finish", 99)
        if finish == 1:
            points.append(5)
        elif finish == 2:
            points.append(3)
        elif finish == 3:
            points.append(2)
        else:
            points.append(0)

    avg_points = sum(points) / len(points) if points else 0
    score = min(avg_points * len(matching_races) / 2, 15)  # N回走って平均点を元にスケール

    detail = f"同条件{len(matching_races)}戦平均{avg_points:.1f}点"
    return round(score, 1), detail


def calculate_ability_score(horse: dict, race_info: dict) -> tuple[float, dict]:
    """
    実力スコア(0-50点)を算出

    Returns:
        (total_score, breakdown_dict)
    """
    past_races = horse.get("past_races", [])

    # 3つのサブスコア算出
    top3_score, top3_detail = calculate_top3_rate_score(past_races)
    margin_score, margin_detail = calculate_margin_score(past_races)
    course_score, course_detail = calculate_course_score(
        past_races,
        race_info.get("venue", ""),
        race_info.get("surface", ""),
        race_info.get("distance", 0)
    )

    total = top3_score + margin_score + course_score

    breakdown = {
        "top3_rate": top3_score,
        "margin_quality": margin_score,
        "course_record": course_score,
        "detail": f"{top3_detail}, {margin_detail}, {course_detail}"
    }

    return min(round(total, 1), 50), breakdown


# ============================================================
# 騎手(jockey)スコア: 0-20点
# ============================================================

def calculate_jockey_score(horse: dict) -> tuple[float, str]:
    """
    騎手の年間勝率×100（上限20点）

    例: 勝率0.185 → 18.5点
    """
    jockey_stats = horse.get("jockey_stats", {})
    win_rate = jockey_stats.get("win_rate", 0.0)

    score = min(win_rate * 100, 20)

    jockey_name = horse.get("jockey", "不明")
    detail = f"{jockey_name}勝率{win_rate:.3f}"

    return round(score, 1), detail


# ============================================================
# 適性(fitness)スコア: 0-15点
# ============================================================

def calculate_venue_fitness(past_races: list, current_venue: str) -> tuple[float, str]:
    """
    同競馬場での成績から0-8点を算出
    """
    if not past_races or not current_venue:
        return 0.0, "同場なし"

    same_venue_races = [r for r in past_races if r.get("venue", "") == current_venue]

    if not same_venue_races:
        return 0.0, "同場なし"

    # 平均着順から算出
    finishes = [r.get("finish", 99) for r in same_venue_races]
    avg_finish = sum(finishes) / len(finishes)

    # 1着平均→8点、5着平均→4点、10着以降→0点 のスケール
    score = max(0, min(8, 8 * (10 - avg_finish) / 9))

    detail = f"同場{len(same_venue_races)}戦平均{avg_finish:.1f}着"
    return round(score, 1), detail


def calculate_distance_fitness(past_races: list, current_distance: int) -> tuple[float, str]:
    """
    同距離帯（±200m）での成績から0-7点を算出
    """
    if not past_races or not current_distance:
        return 0.0, "同距離なし"

    same_distance_races = []
    for race in past_races:
        race_dist = safe_int(race.get("distance", "0"))
        if abs(race_dist - current_distance) <= 200 and race_dist > 0:
            same_distance_races.append(race)

    if not same_distance_races:
        return 0.0, "同距離なし"

    finishes = [r.get("finish", 99) for r in same_distance_races]
    avg_finish = sum(finishes) / len(finishes)

    score = max(0, min(7, 7 * (10 - avg_finish) / 9))

    detail = f"同距離{len(same_distance_races)}戦平均{avg_finish:.1f}着"
    return round(score, 1), detail


def calculate_fitness_score(horse: dict, race_info: dict) -> tuple[float, dict]:
    """適性スコア(0-15点)を算出"""
    past_races = horse.get("past_races", [])

    venue_score, venue_detail = calculate_venue_fitness(
        past_races, race_info.get("venue", "")
    )
    distance_score, distance_detail = calculate_distance_fitness(
        past_races, race_info.get("distance", 0)
    )

    total = venue_score + distance_score

    breakdown = {
        "venue": venue_score,
        "distance": distance_score,
        "detail": f"{venue_detail}, {distance_detail}"
    }

    return min(round(total, 1), 15), breakdown


# ============================================================
# 調子(form)スコア: 0-10点
# ============================================================

def calculate_rotation_score(horse: dict, race_date: str) -> tuple[float, str]:
    """
    前走からの間隔でスコア算出

    - 2-4週: 4点（ベスト）
    - 1週 or 5-8週: 3点
    - 9週以上 or 初出走: 2点
    """
    past_races = horse.get("past_races", [])

    if not past_races:
        return 2.0, "初出走2点"

    last_race_date = past_races[0].get("date", "")

    if not last_race_date or not race_date:
        return 2.0, "前走日不明2点"

    days = calculate_days_between(last_race_date, race_date)
    weeks = days / 7

    if 14 <= days <= 28:  # 2-4週
        score = 4.0
        detail = f"前走{int(weeks)}週前4点"
    elif days < 14 or (29 <= days <= 56):  # 1週 or 5-8週
        score = 3.0
        detail = f"前走{int(weeks)}週前3点"
    else:  # 9週以上
        score = 2.0
        detail = f"前走{int(weeks)}週前2点"

    return score, detail


def calculate_form_score(horse: dict, race_date: str) -> tuple[float, dict]:
    """
    調子スコア(0-10点)を算出

    第1段階では:
    - ローテーション: 4点満点
    - 馬体重変動: 0点（データなし）
    - 調教タイム: 0点（データなし）
    """
    rotation_score, rotation_detail = calculate_rotation_score(horse, race_date)

    # 将来実装用のプレースホルダ
    weight_score = 0.0
    training_score = 0.0

    total = rotation_score + weight_score + training_score

    breakdown = {
        "rotation": rotation_score,
        "weight_change": weight_score,
        "training": training_score,
        "detail": rotation_detail
    }

    return min(round(total, 1), 10), breakdown


# ============================================================
# その他(other)スコア: 0-5点
# ============================================================

def calculate_other_score(horse: dict, all_horses: list) -> tuple[float, dict]:
    """
    その他スコア(0-5点)を算出

    第1段階では:
    - 斤量: 3点満点（相対評価）
    - 特記: 0点（Claude補正フェーズで追加）
    """
    load_weight = horse.get("load_weight", 0)

    if not load_weight or not all_horses:
        return 2.0, {"load_weight": 2, "note": 0, "detail": "斤量データなし2点"}

    # 平均斤量を算出
    all_loads = [h.get("load_weight", 0) for h in all_horses if h.get("load_weight", 0) > 0]
    avg_load = sum(all_loads) / len(all_loads) if all_loads else 54

    diff = avg_load - load_weight  # プラスなら軽い

    if diff >= 2:
        score = 3.0
        detail = f"斤量{load_weight}kg(平均-{abs(diff):.1f}kg)3点"
    elif diff <= -2:
        score = 1.0
        detail = f"斤量{load_weight}kg(平均+{abs(diff):.1f}kg)1点"
    else:
        score = 2.0
        detail = f"斤量{load_weight}kg(平均±{abs(diff):.1f}kg)2点"

    breakdown = {
        "load_weight": score,
        "note": 0,
        "detail": detail
    }

    return score, breakdown


# ============================================================
# スコア集約
# ============================================================

def score_single_horse(horse: dict, race_info: dict, all_horses: list) -> dict:
    """
    1頭の全カテゴリスコアを計算

    Returns:
        更新されたhorse辞書（score, score_breakdown, noteが追加される）
    """
    race_date = race_info.get("date", "")

    # 各カテゴリスコア算出
    ability_score, ability_breakdown = calculate_ability_score(horse, race_info)
    jockey_score, jockey_detail = calculate_jockey_score(horse)
    fitness_score, fitness_breakdown = calculate_fitness_score(horse, race_info)
    form_score, form_breakdown = calculate_form_score(horse, race_date)
    other_score, other_breakdown = calculate_other_score(horse, all_horses)

    # 合計スコア
    total_score = ability_score + jockey_score + fitness_score + form_score + other_score

    # score_breakdown構築
    horse["score"] = round(total_score, 1)
    horse["score_breakdown"] = {
        "ability": round(ability_score, 1),
        "jockey": round(jockey_score, 1),
        "fitness": round(fitness_score, 1),
        "form": round(form_score, 1),
        "other": round(other_score, 1)
    }

    # note構築（計算根拠を記録）
    note_parts = [
        "[AUTO]",
        f"ability={ability_score:.1f}({ability_breakdown['detail']})",
        f"jockey={jockey_score:.1f}({jockey_detail})",
        f"fitness={fitness_score:.1f}({fitness_breakdown['detail']})",
        f"form={form_score:.1f}({form_breakdown['detail']})",
        f"other={other_score:.1f}({other_breakdown['detail']})"
    ]
    horse["note"] = " ".join(note_parts)

    return horse


def score_all_horses(data: dict) -> dict:
    """全馬のスコアを計算"""
    race_info = data.get("race", {})
    horses = data.get("horses", [])

    for horse in horses:
        score_single_horse(horse, race_info, horses)

    return data


# ============================================================
# 検証機能
# ============================================================

def validate_input_data(data: dict) -> tuple[bool, list[str]]:
    """
    入力データの妥当性チェック

    Returns:
        (is_valid, error_messages)
    """
    errors = []

    # 必須フィールドチェック
    if "race" not in data:
        errors.append("[ERROR] 'race'フィールドが存在しません")

    if "horses" not in data or not isinstance(data["horses"], list):
        errors.append("[ERROR] 'horses'フィールドが存在しないか、配列ではありません")

    # レース情報チェック
    race = data.get("race", {})
    required_race_fields = ["date", "venue", "distance", "surface"]
    for field in required_race_fields:
        if not race.get(field):
            errors.append(f"[!] race.{field}が空です")

    # 馬情報チェック
    horses = data.get("horses", [])
    if not horses:
        errors.append("[ERROR] 出走馬データがありません")

    for i, horse in enumerate(horses):
        if "name" not in horse or not horse["name"]:
            errors.append(f"[!] {i+1}頭目: 馬名が空です")

    is_valid = len([e for e in errors if e.startswith("[ERROR]")]) == 0
    return is_valid, errors


def validate_scores(data: dict) -> list[str]:
    """
    スコアの妥当性を検証

    Returns:
        警告メッセージのリスト
    """
    warnings = []
    horses = data.get("horses", [])

    for horse in horses:
        num = horse.get("num", "?")
        name = horse.get("name", "不明")
        score = horse.get("score", 0)
        breakdown = horse.get("score_breakdown", {})

        # 合計値チェック
        if score > 100:
            warnings.append(f"[!] {num}番{name}: スコア{score}点が100点を超過")

        # カテゴリ上限チェック
        limits = {
            "ability": 50,
            "jockey": 20,
            "fitness": 15,
            "form": 10,
            "other": 5
        }

        for category, limit in limits.items():
            value = breakdown.get(category, 0)
            if value > limit:
                warnings.append(f"[!] {num}番{name}: {category}={value}点が上限{limit}点を超過")

        # 合計一致チェック
        calc_total = sum(breakdown.values())
        if abs(calc_total - score) > 0.5:
            warnings.append(f"[!] {num}番{name}: 合計不一致 score={score} vs breakdown合計={calc_total}")

    return warnings


# ============================================================
# メイン処理
# ============================================================

def load_enriched_input(file_path: str) -> dict:
    """enriched_input.jsonを読み込み"""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def determine_race_parameters(race_info: dict) -> dict:
    """
    レースグレードに応じたデフォルトパラメータを決定
    """
    grade = race_info.get("grade", "")

    defaults = {
        "G1": (8, 10000),
        "G2": (8, 5000),
        "G3": (10, 3000),
        "L": (10, 3000),
        "OP": (10, 3000),
        "3勝": (10, 1500),
        "2勝": (10, 1500),
        "1勝": (12, 1500),
        "未勝利": (12, 1500),
        "新馬": (14, 1500),
    }

    temperature, budget = defaults.get(grade, (10, 1500))

    return {
        "temperature": temperature,
        "budget": budget,
        "top_n": 6
    }


def save_output(data: dict, output_path: str):
    """base_scored.jsonを保存"""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    """メイン処理"""
    print()
    print("=" * 60)
    print(f"  Scoring Engine v{VERSION}")
    print("=" * 60)
    print()

    # コマンドライン引数チェック
    if len(sys.argv) < 2:
        print("使い方: python scoring_engine.py <enriched_input.jsonのパス>")
        print("例: python scoring_engine.py ../data/races/20260214_京都_4R_enriched_input.json")
        sys.exit(1)

    input_path = Path(sys.argv[1])

    if not input_path.exists():
        print(f"[ERROR] ファイルが見つかりません: {input_path}")
        sys.exit(1)

    print(f"入力: {input_path}")

    # データ読み込み
    data = load_enriched_input(input_path)

    # データ検証
    is_valid, messages = validate_input_data(data)
    if messages:
        print("\n[データ検証]")
        for msg in messages:
            print(f"  {msg}")

    if not is_valid:
        print("\n[ERROR] 入力データが不正です。処理を中止します。")
        sys.exit(1)

    # パラメータ決定
    race_info = data.get("race", {})
    if "parameters" not in data:
        data["parameters"] = determine_race_parameters(race_info)
        print(f"\n[パラメータ自動設定]")
        print(f"  温度: {data['parameters']['temperature']}")
        print(f"  予算: ¥{data['parameters']['budget']:,}")

    # スコア計算
    print(f"\n[スコア計算中]")
    print(f"  対象: {len(data.get('horses', []))}頭")

    data = score_all_horses(data)

    # スコア検証
    warnings = validate_scores(data)
    if warnings:
        print("\n[スコア検証]")
        for warn in warnings:
            print(f"  {warn}")

    # スコアサマリー表示
    print(f"\n[スコア一覧]")
    horses = data.get("horses", [])
    horses_sorted = sorted(horses, key=lambda h: h.get("score", 0), reverse=True)

    for i, horse in enumerate(horses_sorted[:5]):
        num = horse.get("num", "?")
        name = horse.get("name", "不明")
        score = horse.get("score", 0)
        breakdown = horse.get("score_breakdown", {})

        print(f"  [{i+1}] {num:2d}番 {name:12s} {score:5.1f}点 "
              f"(実{breakdown.get('ability', 0):.1f} "
              f"騎{breakdown.get('jockey', 0):.1f} "
              f"適{breakdown.get('fitness', 0):.1f} "
              f"調{breakdown.get('form', 0):.1f} "
              f"他{breakdown.get('other', 0):.1f})")

    # 保存
    output_path = input_path.parent / input_path.name.replace("_enriched_input.json", "_base_scored.json")
    if not output_path.name.endswith("_base_scored.json"):
        # フォールバック（ファイル名が想定と異なる場合）
        output_path = input_path.parent / "base_scored.json"

    save_output(data, output_path)

    print(f"\n{'=' * 60}")
    print(f"  [OK] 保存完了: {output_path}")
    print(f"{'=' * 60}")
    print(f"\n次のステップ: Claude APIによる補正フェーズ")


if __name__ == "__main__":
    main()
