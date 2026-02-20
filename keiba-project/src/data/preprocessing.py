"""
データ前処理・バリデーション
CSV 読み込み + カテゴリカルエンコーディング + 入力検証
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import (
    SEX_MAP, SURFACE_MAP, TRACK_CONDITION_MAP, CLASS_MAP, SCORE_LIMITS,
)


# ============================================================
# パース補助
# ============================================================

_MARGIN_MAP = {"ハナ": 0.05, "クビ": 0.1, "アタマ": 0.15, "大": 10.0, "同着": 0.0}
_MARGIN_RE = re.compile(r'^(\d+)?\.?(\d+/\d+)?$')


def parse_margin(margin_str) -> float:
    """着差文字列を馬身数(float)に変換"""
    if pd.isna(margin_str) or str(margin_str).strip() == "":
        return 0.0
    s = str(margin_str).strip()
    if s in _MARGIN_MAP:
        return _MARGIN_MAP[s]
    m = _MARGIN_RE.match(s)
    if m:
        whole = int(m.group(1)) if m.group(1) else 0
        frac = 0.0
        if m.group(2):
            num, denom = m.group(2).split("/")
            frac = int(num) / int(denom)
        return whole + frac
    try:
        return float(s)
    except ValueError:
        return 0.0


def parse_first_corner(passing_str) -> int:
    """通過順文字列から最初のコーナー通過位置を返す"""
    if pd.isna(passing_str) or str(passing_str).strip() == "":
        return 0
    first = str(passing_str).strip().split("-")[0]
    try:
        return int(first)
    except ValueError:
        return 0


# ============================================================
# CSV データ読み込み・前処理
# ============================================================

def load_results(path: str | Path) -> pd.DataFrame:
    """results.csv を読み込み基本的な前処理を行う"""
    df = pd.read_csv(path, dtype={"race_id": str, "horse_id": str, "jockey_id": str, "course_id": str})

    df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce")
    df = df.dropna(subset=["race_date"])

    int_cols = ["finish_position", "frame_number", "horse_number", "age",
                "horse_weight", "horse_weight_change", "popularity", "num_entries"]
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    float_cols = ["weight_carried", "finish_time_sec", "last_3f", "win_odds", "prize_money"]
    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # 着差・通過順をパース
    if "margin" in df.columns:
        df["margin_float"] = df["margin"].apply(parse_margin)
    if "passing_order" in df.columns:
        df["first_corner_pos"] = df["passing_order"].apply(parse_first_corner)

    df = df[df["finish_position"] > 0].copy()
    df = df.sort_values(["race_date", "race_id", "finish_position"]).reset_index(drop=True)

    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """カテゴリカル変数を数値に変換"""
    df["sex_code"] = df["sex"].map(SEX_MAP).fillna(0).astype(int)
    df["surface_code"] = df["surface"].map(SURFACE_MAP).fillna(0).astype(int)
    df["track_condition_code"] = df["track_condition"].map(TRACK_CONDITION_MAP).fillna(0).astype(int)
    df["race_class_code"] = df["race_class"].map(CLASS_MAP).fillna(3).astype(int)

    df["distance_cat"] = pd.cut(
        df["distance"],
        bins=[0, 1400, 1800, 2200, 9999],
        labels=[0, 1, 2, 3],
    ).astype(int)

    df["course_id_code"] = pd.to_numeric(df["course_id"], errors="coerce").fillna(0).astype(int)

    return df


# ============================================================
# 入力データ検証
# ============================================================

def validate_input_data(data: dict) -> tuple[bool, list[str]]:
    """enriched_input.json の妥当性チェック"""
    errors = []

    if "race" not in data:
        errors.append("[ERROR] 'race'フィールドが存在しません")
    if "horses" not in data or not isinstance(data["horses"], list):
        errors.append("[ERROR] 'horses'フィールドが存在しないか、配列ではありません")

    race = data.get("race", {})
    for field in ["date", "venue", "distance", "surface"]:
        if not race.get(field):
            errors.append(f"[!] race.{field}が空です")

    horses = data.get("horses", [])
    if not horses:
        errors.append("[ERROR] 出走馬データがありません")
    for i, horse in enumerate(horses):
        if not horse.get("name"):
            errors.append(f"[!] {i+1}頭目: 馬名が空です")

    is_valid = len([e for e in errors if e.startswith("[ERROR]")]) == 0
    return is_valid, errors


def validate_scores(data: dict) -> list[str]:
    """スコアの妥当性を検証"""
    warnings = []
    for horse in data.get("horses", []):
        num = horse.get("num", "?")
        name = horse.get("name", "不明")
        score = horse.get("score", 0)
        breakdown = horse.get("score_breakdown", {})

        if score > 100:
            warnings.append(f"[!] {num}番{name}: スコア{score}点が100点を超過")

        for category, limit in SCORE_LIMITS.items():
            value = breakdown.get(category, 0)
            if value > limit:
                warnings.append(f"[!] {num}番{name}: {category}={value}点が上限{limit}点を超過")

        calc_total = sum(breakdown.get(k, 0) for k in SCORE_LIMITS)
        if abs(calc_total - score) > 0.5:
            warnings.append(f"[!] {num}番{name}: 合計不一致 score={score} vs breakdown合計={calc_total}")

    return warnings
