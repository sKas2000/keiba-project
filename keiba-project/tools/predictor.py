#!/usr/bin/env python3
"""
予測パイプライン v1.0
=====================
学習済みLightGBMモデルを使って新しいレースの3着以内確率を予測し、
ev_calculator.py に接続する。

使い方:
  # enriched_input.json に対して予測
  python predictor.py <enriched_input.json>

  # モデル指定
  python predictor.py <enriched_input.json> --model-dir data/ml/models
"""

VERSION = "1.0"

import json
import re
import sys
from pathlib import Path
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import lightgbm as lgb

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "ml"
MODEL_DIR = DATA_DIR / "models"

# feature_engineer.py と同じ特徴量リスト
FEATURE_COLUMNS = [
    "frame_number", "horse_number", "sex_code", "age",
    "weight_carried", "horse_weight", "horse_weight_change",
    "num_entries",
    "surface_code", "distance", "track_condition_code",
    "race_class_code", "distance_cat", "course_id_code",
    "prev_finish_1", "prev_finish_2", "prev_finish_3",
    "avg_finish_last5", "best_finish_last5",
    "win_rate_last5", "place_rate_last5",
    "avg_last3f_last5", "days_since_last_race",
    "total_races", "career_win_rate", "career_place_rate",
    "surface_win_rate", "surface_place_rate",
    "distance_cat_win_rate",
    "jockey_win_rate_365d", "jockey_place_rate_365d",
    "jockey_ride_count_365d",
]


# ============================================================
# エンコーディング定数（feature_engineer.py と同一）
# ============================================================

SEX_MAP = {"牡": 0, "牝": 1, "セ": 2, "騸": 2}
SURFACE_MAP = {"芝": 0, "ダ": 1, "障": 2}
COND_MAP = {"良": 0, "稍重": 1, "重": 2, "不良": 3}
CLASS_MAP = {
    "新馬": 1, "未勝利": 2, "1勝": 3, "2勝": 4, "3勝": 5,
    "OP": 6, "オープン": 6, "リステッド": 7, "L": 7,
    "G3": 8, "G2": 9, "G1": 10,
}
COURSE_MAP = {
    "札幌": 1, "函館": 2, "福島": 3, "新潟": 4, "東京": 5,
    "中山": 6, "中京": 7, "京都": 8, "阪神": 9, "小倉": 10,
}


# ============================================================
# enriched_input.json → 特徴量ベクトル変換
# ============================================================

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


def extract_features_from_enriched(data: dict) -> pd.DataFrame:
    """
    enriched_input.json からML特徴量を抽出
    Args:
        data: enriched_input.json のデータ
    Returns:
        特徴量DataFrame (1行=1馬)
    """
    race = data.get("race", {})
    horses = data.get("horses", [])
    num_entries = len(horses)

    # レース条件のエンコード
    surface = race.get("surface", "")
    surface_code = SURFACE_MAP.get(surface, 0)
    distance = race.get("distance", 0)
    track_condition_code = COND_MAP.get(race.get("track_condition", ""), 0)
    race_class_code = CLASS_MAP.get(race.get("grade", ""), 3)
    venue = race.get("venue", "")
    course_id_code = COURSE_MAP.get(venue, 0)

    # 距離カテゴリ
    if distance <= 1400:
        distance_cat = 0
    elif distance <= 1800:
        distance_cat = 1
    elif distance <= 2200:
        distance_cat = 2
    else:
        distance_cat = 3

    rows = []
    for horse in horses:
        row = {}

        # 基本特徴量
        row["horse_name"] = horse.get("name", "")
        row["horse_number"] = horse.get("num", 0)
        row["frame_number"] = horse.get("frame_number", 0)
        row["num_entries"] = num_entries

        # 性齢
        sex_age = horse.get("sex_age", "")
        m = re.match(r'([牡牝セ騸])(\d+)', sex_age)
        if m:
            row["sex_code"] = SEX_MAP.get(m.group(1), 0)
            row["age"] = int(m.group(2))
        else:
            row["sex_code"] = 0
            row["age"] = 0

        # 斤量・体重
        row["weight_carried"] = safe_float(horse.get("load_weight", 0))
        row["horse_weight"] = 0
        row["horse_weight_change"] = 0

        # レース条件
        row["surface_code"] = surface_code
        row["distance"] = distance
        row["track_condition_code"] = track_condition_code
        row["race_class_code"] = race_class_code
        row["distance_cat"] = distance_cat
        row["course_id_code"] = course_id_code

        # 過去成績特徴量（past_racesから計算）
        past_races = horse.get("past_races", [])
        _compute_past_features(row, past_races, race, surface, distance_cat)

        # 騎手統計
        jockey_stats = horse.get("jockey_stats", {})
        row["jockey_win_rate_365d"] = jockey_stats.get("win_rate", 0.0)
        row["jockey_place_rate_365d"] = jockey_stats.get("place_rate", 0.0)
        row["jockey_ride_count_365d"] = jockey_stats.get("races", 0)

        # オッズ（EVを計算に使う、予測には使わない）
        row["odds_win"] = horse.get("odds_win", 0)
        row["odds_place"] = horse.get("odds_place", 0)
        row["num"] = horse.get("num", 0)
        row["name"] = horse.get("name", "")

        rows.append(row)

    return pd.DataFrame(rows)


def _compute_past_features(row: dict, past_races: list, race: dict,
                           surface: str, distance_cat: int):
    """過去走データから特徴量を計算"""
    if not past_races:
        row["prev_finish_1"] = 0
        row["prev_finish_2"] = 0
        row["prev_finish_3"] = 0
        row["avg_finish_last5"] = 0
        row["best_finish_last5"] = 0
        row["win_rate_last5"] = 0
        row["place_rate_last5"] = 0
        row["avg_last3f_last5"] = 0
        row["days_since_last_race"] = 365
        row["total_races"] = 0
        row["career_win_rate"] = 0
        row["career_place_rate"] = 0
        row["surface_win_rate"] = 0
        row["surface_place_rate"] = 0
        row["distance_cat_win_rate"] = 0
        return

    # 着順リスト
    finishes = [safe_int(r.get("finish", 0)) for r in past_races if safe_int(r.get("finish", 0)) > 0]
    recent5 = finishes[:5]

    row["prev_finish_1"] = recent5[0] if len(recent5) >= 1 else 0
    row["prev_finish_2"] = recent5[1] if len(recent5) >= 2 else 0
    row["prev_finish_3"] = recent5[2] if len(recent5) >= 3 else 0

    if recent5:
        arr = np.array(recent5)
        row["avg_finish_last5"] = round(arr.mean(), 2)
        row["best_finish_last5"] = int(arr.min())
        row["win_rate_last5"] = round((arr == 1).mean(), 3)
        row["place_rate_last5"] = round((arr <= 3).mean(), 3)
    else:
        row["avg_finish_last5"] = 0
        row["best_finish_last5"] = 0
        row["win_rate_last5"] = 0
        row["place_rate_last5"] = 0

    # 上り3F
    last3f_vals = [safe_float(r.get("last3f", 0)) for r in past_races[:5]]
    valid_3f = [v for v in last3f_vals if v > 0]
    row["avg_last3f_last5"] = round(np.mean(valid_3f), 2) if valid_3f else 0

    # 前走からの日数
    race_date_str = race.get("date", "")
    last_race_date = past_races[0].get("date", "") if past_races else ""
    if race_date_str and last_race_date:
        try:
            from datetime import datetime
            rd = race_date_str.replace("/", "-")
            ld = last_race_date.replace("/", "-")
            d1 = datetime.strptime(rd, "%Y-%m-%d")
            d2 = datetime.strptime(ld, "%Y-%m-%d")
            row["days_since_last_race"] = abs((d1 - d2).days)
        except Exception:
            row["days_since_last_race"] = 30
    else:
        row["days_since_last_race"] = 30

    # 通算成績
    all_finishes = np.array(finishes)
    row["total_races"] = len(all_finishes)
    row["career_win_rate"] = round((all_finishes == 1).mean(), 3) if len(all_finishes) > 0 else 0
    row["career_place_rate"] = round((all_finishes <= 3).mean(), 3) if len(all_finishes) > 0 else 0

    # 同馬場成績
    surface_races = [r for r in past_races if r.get("surface", "") == surface]
    if surface_races:
        sf = np.array([safe_int(r.get("finish", 0)) for r in surface_races if safe_int(r.get("finish", 0)) > 0])
        row["surface_win_rate"] = round((sf == 1).mean(), 3) if len(sf) > 0 else 0
        row["surface_place_rate"] = round((sf <= 3).mean(), 3) if len(sf) > 0 else 0
    else:
        row["surface_win_rate"] = 0
        row["surface_place_rate"] = 0

    # 距離帯成績
    row["distance_cat_win_rate"] = 0
    for r in past_races:
        d = safe_int(r.get("distance", 0))
        if d > 0:
            if d <= 1400:
                cat = 0
            elif d <= 1800:
                cat = 1
            elif d <= 2200:
                cat = 2
            else:
                cat = 3
            if cat == distance_cat:
                dist_finishes = [safe_int(r2.get("finish", 0))
                                 for r2 in past_races
                                 if safe_int(r2.get("distance", 0)) > 0
                                 and _dist_cat(safe_int(r2.get("distance", 0))) == distance_cat
                                 and safe_int(r2.get("finish", 0)) > 0]
                if dist_finishes:
                    row["distance_cat_win_rate"] = round(np.mean(np.array(dist_finishes) == 1), 3)
                break


def _dist_cat(distance: int) -> int:
    if distance <= 1400:
        return 0
    elif distance <= 1800:
        return 1
    elif distance <= 2200:
        return 2
    return 3


# ============================================================
# 予測実行
# ============================================================

def predict_race(data: dict, model_dir: Path = MODEL_DIR) -> dict:
    """
    enriched_input.json に対して予測を実行し、スコアを付与
    Args:
        data: enriched_input.json のデータ
        model_dir: モデルディレクトリ
    Returns:
        予測結果付きデータ（scored形式）
    """
    # モデル読み込み
    binary_model_path = model_dir / "binary_model.txt"
    if not binary_model_path.exists():
        print(f"[WARN] モデル未学習: {binary_model_path}")
        print("  → ルールベーススコアリングにフォールバック")
        return None

    binary_model = lgb.Booster(model_file=str(binary_model_path))

    # メタ情報読み込み
    meta_path = model_dir / "binary_meta.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    feature_names = meta.get("feature_names", FEATURE_COLUMNS)

    # 特徴量抽出
    df = extract_features_from_enriched(data)

    # 特徴量行列
    available = [c for c in feature_names if c in df.columns]
    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        for col in missing:
            df[col] = 0.0
        available = feature_names

    X = df[available].values.astype(np.float32)

    # 予測
    probs = binary_model.predict(X)

    # ランキングモデルがあれば追加
    rank_model_path = model_dir / "ranking_model.txt"
    rank_scores = None
    if rank_model_path.exists():
        rank_model = lgb.Booster(model_file=str(rank_model_path))
        rank_scores = rank_model.predict(X)

    # 結果をdata に反映
    horses = data.get("horses", [])
    for i, horse in enumerate(horses):
        if i < len(probs):
            prob = float(probs[i])
            # 確率をスコアに変換 (0-100点)
            horse["score"] = round(prob * 100, 1)
            horse["ml_top3_prob"] = round(prob, 4)

            if rank_scores is not None and i < len(rank_scores):
                horse["ml_rank_score"] = round(float(rank_scores[i]), 4)

            horse["score_breakdown"] = {
                "ml_binary": round(prob * 100, 1),
                "ability": 0, "jockey": 0, "fitness": 0,
                "form": 0, "other": 0,
            }
            horse["note"] = f"[ML] top3_prob={prob:.3f}"

    return data


# ============================================================
# メイン処理
# ============================================================

def main():
    parser = ArgumentParser(description="ML予測パイプライン")
    parser.add_argument("input", help="enriched_input.json パス")
    parser.add_argument("--model-dir", default=str(MODEL_DIR), help="モデルディレクトリ")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] ファイルが見つかりません: {input_path}")
        sys.exit(1)

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"\n{'=' * 60}")
    print(f"  ML予測パイプライン v{VERSION}")
    print(f"{'=' * 60}")
    print(f"  入力: {input_path}")
    print(f"  モデル: {args.model_dir}")
    print(f"  馬数: {len(data.get('horses', []))}")

    result = predict_race(data, Path(args.model_dir))

    if result is None:
        print("\n  [!] MLモデル未学習。先にmodel_trainer.pyで学習してください。")
        sys.exit(1)

    # 結果表示
    horses = sorted(result.get("horses", []), key=lambda h: h.get("score", 0), reverse=True)
    print(f"\n  [予測結果]")
    for i, h in enumerate(horses[:5]):
        print(f"    [{i+1}] {h.get('num', '?'):2d}番 {h.get('name', ''):12s} "
              f"3着内確率 {h.get('ml_top3_prob', 0):.1%}  "
              f"スコア {h.get('score', 0):.1f}点")

    # 保存
    output_path = str(input_path).replace("_enriched_input.json", "_ml_scored.json")
    if output_path == str(input_path):
        output_path = str(input_path).replace(".json", "_ml_scored.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n  [OK] 保存: {output_path}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
