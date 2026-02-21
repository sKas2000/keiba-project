"""
特徴量作成
CSV ベースの ML 特徴量 + enriched_input.json からの特徴量抽出を統合
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import (
    FEATURE_COLUMNS, META_COLUMNS,
    SEX_MAP, SURFACE_MAP, TRACK_CONDITION_MAP, CLASS_MAP, COURSE_NAME_TO_ID,
    RAW_DIR, PROCESSED_DIR,
)
from src.data.preprocessing import load_results, encode_categoricals
from src.scraping.parsers import safe_float, safe_int


# ============================================================
# CSV ベースの過去成績集約特徴量
# ============================================================

def compute_horse_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """各馬の過去成績から集約特徴量を計算（リーケージ防止）"""
    # 決定論的ソート: race_date + race_id + finish_position で一意に決まる
    df = df.sort_values(["race_date", "race_id", "finish_position"]).reset_index(drop=True)

    feature_cols = [
        "prev_finish_1", "prev_finish_2", "prev_finish_3",
        "avg_finish_last5", "best_finish_last5",
        "win_rate_last5", "place_rate_last5",
        "avg_last3f_last5", "days_since_last_race",
        "total_races", "career_win_rate", "career_place_rate",
        "surface_win_rate", "surface_place_rate",
        "distance_cat_win_rate",
        # v2: 新特徴量
        "prev_margin_1", "prev_last3f_1",
        "distance_change",
        "running_style", "avg_early_position_last5",
        "track_cond_place_rate",
        # v3: クラス変動特徴量
        "class_change",
        # v4: 斤量変化
        "weight_carried_change",
    ]
    for col in feature_cols:
        df[col] = 0.0

    has_margin = "margin_float" in df.columns
    has_corner = "first_corner_pos" in df.columns

    # グルーピングキー: horse_id優先、NaN行はhorse_nameでフォールバック
    if "horse_id" in df.columns and df["horse_id"].notna().any():
        # NaN horse_id 行には horse_name を代入して統一
        df["_group_key"] = df["horse_id"].fillna(df["horse_name"])
    else:
        df["_group_key"] = df["horse_name"]
    grouped = df.groupby("_group_key", sort=False)

    for _, group in grouped:
        group = group.sort_values("race_date")
        indices = group.index.tolist()

        for pos, idx in enumerate(indices):
            past = group.iloc[:pos]
            if len(past) == 0:
                continue

            recent = past.tail(5)
            finishes = recent["finish_position"].values

            if len(finishes) >= 1:
                df.at[idx, "prev_finish_1"] = finishes[-1]
            if len(finishes) >= 2:
                df.at[idx, "prev_finish_2"] = finishes[-2]
            if len(finishes) >= 3:
                df.at[idx, "prev_finish_3"] = finishes[-3]

            df.at[idx, "avg_finish_last5"] = round(finishes.mean(), 2)
            df.at[idx, "best_finish_last5"] = int(finishes.min())
            df.at[idx, "win_rate_last5"] = round((finishes == 1).mean(), 3)
            df.at[idx, "place_rate_last5"] = round((finishes <= 3).mean(), 3)

            last3f_vals = recent["last_3f"].values
            valid_3f = last3f_vals[last3f_vals > 0]
            if len(valid_3f) > 0:
                df.at[idx, "avg_last3f_last5"] = round(valid_3f.mean(), 2)

            last_race_date = past.iloc[-1]["race_date"]
            current_race_date = df.at[idx, "race_date"]
            if pd.notna(last_race_date) and pd.notna(current_race_date):
                df.at[idx, "days_since_last_race"] = (current_race_date - last_race_date).days

            all_finishes = past["finish_position"].values
            df.at[idx, "total_races"] = len(all_finishes)
            df.at[idx, "career_win_rate"] = round((all_finishes == 1).mean(), 3)
            df.at[idx, "career_place_rate"] = round((all_finishes <= 3).mean(), 3)

            current_surface = df.at[idx, "surface"]
            surface_past = past[past["surface"] == current_surface]
            if len(surface_past) > 0:
                sf = surface_past["finish_position"].values
                df.at[idx, "surface_win_rate"] = round((sf == 1).mean(), 3)
                df.at[idx, "surface_place_rate"] = round((sf <= 3).mean(), 3)

            current_dist_cat = df.at[idx, "distance_cat"]
            dist_past = past[past["distance_cat"] == current_dist_cat]
            if len(dist_past) > 0:
                df_vals = dist_past["finish_position"].values
                df.at[idx, "distance_cat_win_rate"] = round((df_vals == 1).mean(), 3)

            # --- v2: 新特徴量 ---

            # 前走着差（馬身数）
            if has_margin:
                df.at[idx, "prev_margin_1"] = past.iloc[-1].get("margin_float", 0)

            # 前走上がり3F（個別値）
            prev_3f = past.iloc[-1]["last_3f"]
            if prev_3f > 0:
                df.at[idx, "prev_last3f_1"] = prev_3f

            # 距離変更（今走 - 前走）
            prev_dist = past.iloc[-1].get("distance", 0)
            curr_dist = df.at[idx, "distance"]
            if prev_dist > 0 and curr_dist > 0:
                df.at[idx, "distance_change"] = curr_dist - prev_dist

            # 脚質・序盤位置
            if has_corner:
                corner_vals = recent["first_corner_pos"].values
                valid_pos = corner_vals[corner_vals > 0]
                if len(valid_pos) > 0:
                    avg_pos = valid_pos.mean()
                    df.at[idx, "avg_early_position_last5"] = round(avg_pos, 2)
                    num_e = recent["num_entries"].mean() if "num_entries" in recent.columns else 14
                    ratio = avg_pos / max(num_e, 1)
                    if ratio <= 0.15:
                        style = 0  # 逃げ
                    elif ratio <= 0.35:
                        style = 1  # 先行
                    elif ratio <= 0.65:
                        style = 2  # 差し
                    else:
                        style = 3  # 追込
                    df.at[idx, "running_style"] = style

            # 同馬場状態（良/稍重/重/不良）での複勝率
            if "track_condition" in df.columns:
                current_tc = df.at[idx, "track_condition"]
                tc_past = past[past["track_condition"] == current_tc]
                if len(tc_past) > 0:
                    tc_f = tc_past["finish_position"].values
                    df.at[idx, "track_cond_place_rate"] = round((tc_f <= 3).mean(), 3)

            # v3: クラス変動（升級=正, 降級=負）
            if "race_class_code" in df.columns:
                current_class = df.at[idx, "race_class_code"]
                prev_class = past.iloc[-1]["race_class_code"] if "race_class_code" in past.columns else current_class
                df.at[idx, "class_change"] = current_class - prev_class

            # v4: 斤量変化（今走 - 前走）
            if "weight_carried" in df.columns:
                curr_wc = df.at[idx, "weight_carried"]
                prev_wc = past.iloc[-1]["weight_carried"] if "weight_carried" in past.columns else curr_wc
                if curr_wc > 0 and prev_wc > 0:
                    df.at[idx, "weight_carried_change"] = curr_wc - prev_wc

    # 一時カラム削除
    df = df.drop(columns=["_group_key"])
    return df


# ============================================================
# 騎手統計特徴量
# ============================================================

def compute_jockey_features(df: pd.DataFrame) -> pd.DataFrame:
    """騎手の過去365日成績を計算"""
    df["jockey_win_rate_365d"] = 0.0
    df["jockey_place_rate_365d"] = 0.0
    df["jockey_ride_count_365d"] = 0

    # 騎手キー: jockey_id 優先、NaN行は jockey_name でフォールバック
    if "jockey_id" in df.columns and df["jockey_id"].notna().any():
        df["_jockey_key"] = df["jockey_id"].fillna(df["jockey_name"])
    else:
        df["_jockey_key"] = df["jockey_name"]
    jockey_col = "_jockey_key"
    dates = df["race_date"].unique()

    for date in sorted(dates):
        date_mask = df["race_date"] == date
        past_mask = (df["race_date"] < date) & (df["race_date"] >= date - pd.Timedelta(days=365))
        past_data = df[past_mask]
        if len(past_data) == 0:
            continue

        jockey_stats = past_data.groupby(jockey_col)["finish_position"].agg(
            jockey_win_rate_365d=lambda x: (x == 1).mean(),
            jockey_place_rate_365d=lambda x: (x <= 3).mean(),
            jockey_ride_count_365d="count",
        ).reset_index()

        for _, row in jockey_stats.iterrows():
            mask = date_mask & (df[jockey_col] == row[jockey_col])
            df.loc[mask, "jockey_win_rate_365d"] = round(row["jockey_win_rate_365d"], 3)
            df.loc[mask, "jockey_place_rate_365d"] = round(row["jockey_place_rate_365d"], 3)
            df.loc[mask, "jockey_ride_count_365d"] = int(row["jockey_ride_count_365d"])

    df = df.drop(columns=["_jockey_key"])
    return df


# ============================================================
# 調教師統計特徴量
# ============================================================

def compute_trainer_features(df: pd.DataFrame) -> pd.DataFrame:
    """調教師の過去365日成績を計算"""
    df["trainer_win_rate_365d"] = 0.0
    df["trainer_place_rate_365d"] = 0.0

    if "trainer_name" not in df.columns or df["trainer_name"].isna().all():
        return df

    dates = df["race_date"].unique()

    for date in sorted(dates):
        date_mask = df["race_date"] == date
        past_mask = (df["race_date"] < date) & (df["race_date"] >= date - pd.Timedelta(days=365))
        past_data = df[past_mask]
        if len(past_data) == 0:
            continue

        stats = past_data.groupby("trainer_name")["finish_position"].agg(
            trainer_win_rate_365d=lambda x: (x == 1).mean(),
            trainer_place_rate_365d=lambda x: (x <= 3).mean(),
        ).reset_index()

        for _, row in stats.iterrows():
            mask = date_mask & (df["trainer_name"] == row["trainer_name"])
            df.loc[mask, "trainer_win_rate_365d"] = round(row["trainer_win_rate_365d"], 3)
            df.loc[mask, "trainer_place_rate_365d"] = round(row["trainer_place_rate_365d"], 3)

    return df


# ============================================================
# 目的変数・特徴量選択
# ============================================================

def create_target(df: pd.DataFrame) -> pd.DataFrame:
    df["top3"] = (df["finish_position"] <= 3).astype(int)
    return df


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    available = [c for c in FEATURE_COLUMNS + META_COLUMNS if c in df.columns]
    return df[available].copy()


# ============================================================
# CSV パイプライン実行
# ============================================================

def run_feature_pipeline(input_path: str | Path = None, output_path: str | Path = None) -> pd.DataFrame:
    input_path = input_path or RAW_DIR / "results.csv"
    output_path = output_path or PROCESSED_DIR / "features.csv"

    print(f"\n[1/8] データ読み込み: {input_path}")
    df = load_results(input_path)
    print(f"  -> {len(df)}行, {df['race_id'].nunique()}レース")

    print("[2/8] カテゴリカルエンコーディング")
    df = encode_categoricals(df)

    print("[3/8] 基本特徴量追加")
    df["race_month"] = df["race_date"].dt.month
    if "win_odds" in df.columns:
        df["log_odds"] = np.log1p(df["win_odds"].clip(lower=0).fillna(0))

    print("[4/8] 過去成績集約特徴量を計算中...")
    df = compute_horse_history_features(df)

    print("[5/8] 騎手統計特徴量を計算中...")
    df = compute_jockey_features(df)

    print("[6/8] 調教師統計特徴量を計算中...")
    df = compute_trainer_features(df)

    print("[7/8] 目的変数作成")
    df = create_target(df)

    print("[8/8] 特徴量選択・保存")
    df_features = select_features(df)

    # 決定論的ソート（再現性保証）
    sort_keys = ["race_date", "race_id", "finish_position"]
    sort_keys = [k for k in sort_keys if k in df_features.columns]
    df_features = df_features.sort_values(sort_keys).reset_index(drop=True)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df_features.to_csv(output_path, index=False)

    available = [c for c in FEATURE_COLUMNS if c in df_features.columns]
    print(f"\n  特徴量数: {len(available)}")
    print(f"  出力: {output_path}")
    print(f"  行数: {len(df_features)}")
    print(f"  3着内率: {df_features['top3'].mean():.1%}")

    return df_features


# ============================================================
# enriched_input.json -> 特徴量ベクトル変換
# ============================================================

def _dist_cat(distance: int) -> int:
    if distance <= 1400:
        return 0
    elif distance <= 1800:
        return 1
    elif distance <= 2200:
        return 2
    return 3


def _compute_past_features(row: dict, past_races: list, race: dict,
                           surface: str, distance_cat: int):
    """過去走データから特徴量を計算"""
    if not past_races:
        for k in ["prev_finish_1", "prev_finish_2", "prev_finish_3",
                   "avg_finish_last5", "best_finish_last5",
                   "win_rate_last5", "place_rate_last5",
                   "avg_last3f_last5", "total_races",
                   "career_win_rate", "career_place_rate",
                   "surface_win_rate", "surface_place_rate",
                   "distance_cat_win_rate",
                   "prev_margin_1", "prev_last3f_1",
                   "distance_change", "running_style",
                   "avg_early_position_last5", "track_cond_place_rate",
                   "class_change", "weight_carried_change"]:
            row[k] = 0
        row["days_since_last_race"] = 365
        return

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
        row["avg_finish_last5"] = row["best_finish_last5"] = 0
        row["win_rate_last5"] = row["place_rate_last5"] = 0

    last3f_vals = [safe_float(r.get("last3f", 0)) for r in past_races[:5]]
    valid_3f = [v for v in last3f_vals if v > 0]
    row["avg_last3f_last5"] = round(np.mean(valid_3f), 2) if valid_3f else 0

    # 前走上がり3F個別値
    row["prev_last3f_1"] = safe_float(past_races[0].get("last3f", 0)) if past_races else 0

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

    all_finishes = np.array(finishes)
    row["total_races"] = len(all_finishes)
    row["career_win_rate"] = round((all_finishes == 1).mean(), 3) if len(all_finishes) > 0 else 0
    row["career_place_rate"] = round((all_finishes <= 3).mean(), 3) if len(all_finishes) > 0 else 0

    surface_races = [r for r in past_races if r.get("surface", "") == surface]
    if surface_races:
        sf = np.array([safe_int(r.get("finish", 0)) for r in surface_races if safe_int(r.get("finish", 0)) > 0])
        row["surface_win_rate"] = round((sf == 1).mean(), 3) if len(sf) > 0 else 0
        row["surface_place_rate"] = round((sf <= 3).mean(), 3) if len(sf) > 0 else 0
    else:
        row["surface_win_rate"] = row["surface_place_rate"] = 0

    row["distance_cat_win_rate"] = 0
    for r in past_races:
        d = safe_int(r.get("distance", 0))
        if d > 0 and _dist_cat(d) == distance_cat:
            dist_finishes = [safe_int(r2.get("finish", 0))
                             for r2 in past_races
                             if safe_int(r2.get("distance", 0)) > 0
                             and _dist_cat(safe_int(r2.get("distance", 0))) == distance_cat
                             and safe_int(r2.get("finish", 0)) > 0]
            if dist_finishes:
                row["distance_cat_win_rate"] = round(np.mean(np.array(dist_finishes) == 1), 3)
            break

    # v2: 距離変更
    if past_races:
        prev_dist = safe_int(past_races[0].get("distance", 0))
        curr_dist = race.get("distance", 0)
        if prev_dist > 0 and curr_dist > 0:
            row["distance_change"] = curr_dist - prev_dist
        else:
            row["distance_change"] = 0
    else:
        row["distance_change"] = 0

    # v2: 前走着差
    if past_races:
        margin_str = past_races[0].get("margin", "")
        from src.data.preprocessing import parse_margin
        row["prev_margin_1"] = parse_margin(margin_str)
    else:
        row["prev_margin_1"] = 0

    # v2: 脚質（通過順位データから算出）
    early_positions = []
    for r in past_races[:5]:
        pos_str = r.get("position", "")
        if pos_str:
            first = pos_str.split("-")[0]
            try:
                early_positions.append(int(first))
            except ValueError:
                pass

    if early_positions:
        avg_pos = np.mean(early_positions)
        row["avg_early_position_last5"] = round(avg_pos, 2)
        # 脚質分類（比率ベース: CSV側と統一）
        num_e = row.get("num_entries", 14)
        ratio = avg_pos / max(num_e, 1)
        if ratio <= 0.15:
            row["running_style"] = 0  # 逃げ
        elif ratio <= 0.35:
            row["running_style"] = 1  # 先行
        elif ratio <= 0.65:
            row["running_style"] = 2  # 差し
        else:
            row["running_style"] = 3  # 追込
    else:
        row["running_style"] = 0
        row["avg_early_position_last5"] = 0

    # v2: 馬場状態適性（過去走から算出）
    track_cond = race.get("track_condition", "")
    if track_cond and past_races:
        tc_races = [r for r in past_races
                    if r.get("track_condition", "") == track_cond
                    and safe_int(r.get("finish", 0)) > 0]
        if tc_races:
            tc_finishes = np.array([safe_int(r.get("finish", 0)) for r in tc_races])
            row["track_cond_place_rate"] = round((tc_finishes <= 3).mean(), 3)
        else:
            row["track_cond_place_rate"] = 0
    else:
        row["track_cond_place_rate"] = 0

    # v3: クラス変動（升級=正, 降級=負）
    current_class = CLASS_MAP.get(race.get("grade", ""), 3)
    if past_races:
        prev_class_str = past_races[0].get("race_class", past_races[0].get("grade", ""))
        prev_class = CLASS_MAP.get(prev_class_str, 3)
        row["class_change"] = current_class - prev_class
    else:
        row["class_change"] = 0

    # v4: 斤量変化（今走 - 前走）
    curr_wc = safe_float(row.get("weight_carried", 0))
    if past_races and curr_wc > 0:
        prev_wc = safe_float(past_races[0].get("weight_carried", past_races[0].get("load_weight", 0)))
        if prev_wc > 0:
            row["weight_carried_change"] = curr_wc - prev_wc
        else:
            row["weight_carried_change"] = 0
    else:
        row["weight_carried_change"] = 0


def _load_trainer_lookup(race_date_str: str) -> dict:
    """results.csvから調教師の直近365日成績をルックアップ辞書で返す"""
    results_path = RAW_DIR / "results.csv"
    if not results_path.exists():
        return {}
    try:
        from datetime import datetime, timedelta
        race_date = datetime.strptime(race_date_str.replace("/", "-"), "%Y-%m-%d")
        df = pd.read_csv(results_path, dtype={"race_id": str},
                         usecols=["race_date", "trainer_name", "finish_position"])
        df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce")
        df = df.dropna(subset=["race_date", "trainer_name"])
        df["finish_position"] = pd.to_numeric(df["finish_position"], errors="coerce")
        df = df[df["finish_position"] > 0]
        cutoff = race_date - timedelta(days=365)
        recent = df[(df["race_date"] >= cutoff) & (df["race_date"] < race_date)]
        if len(recent) == 0:
            return {}
        stats = recent.groupby("trainer_name")["finish_position"].agg(
            win_rate=lambda x: round((x == 1).mean(), 3),
            place_rate=lambda x: round((x <= 3).mean(), 3),
        )
        return stats.to_dict(orient="index")
    except Exception:
        return {}


def extract_features_from_enriched(data: dict) -> pd.DataFrame:
    """enriched_input.json -> ML 特徴量 DataFrame"""
    race = data.get("race", {})
    horses = data.get("horses", [])
    num_entries = len(horses)

    surface = race.get("surface", "")
    surface_code = SURFACE_MAP.get(surface, 0)
    distance = race.get("distance", 0)
    track_condition_code = TRACK_CONDITION_MAP.get(race.get("track_condition", ""), 0)
    race_class_code = CLASS_MAP.get(race.get("grade", ""), 3)
    course_id_code = COURSE_NAME_TO_ID.get(race.get("venue", ""), 0)
    distance_cat = _dist_cat(distance)

    # レース月
    race_date_str = race.get("date", "")
    race_month = 0
    if race_date_str:
        try:
            from datetime import datetime
            dt = datetime.strptime(race_date_str.replace("/", "-"), "%Y-%m-%d")
            race_month = dt.month
        except Exception:
            pass

    # 調教師成績ルックアップ（results.csvから直近365日分を事前計算）
    trainer_lookup = _load_trainer_lookup(race_date_str) if race_date_str else {}

    rows = []
    for horse in horses:
        row = {}
        row["horse_name"] = horse.get("name", "")
        row["horse_number"] = horse.get("num", 0)
        row["frame_number"] = horse.get("frame_number", 0)
        row["num_entries"] = num_entries

        sex_age = horse.get("sex_age", "")
        m = re.match(r'([牡牝セ騸])(\d+)', sex_age)
        row["sex_code"] = SEX_MAP.get(m.group(1), 0) if m else 0
        row["age"] = int(m.group(2)) if m else 0

        row["weight_carried"] = safe_float(horse.get("load_weight", 0))

        # weight は "502(+2)" or "502(-8)" 形式の場合がある
        weight_raw = str(horse.get("weight", "0"))
        wm = re.match(r'(\d+)\s*\(([+-]?\d+)\)', weight_raw)
        if wm:
            row["horse_weight"] = int(wm.group(1))
            row["horse_weight_change"] = int(wm.group(2))
        else:
            row["horse_weight"] = safe_int(weight_raw)
            row["horse_weight_change"] = safe_int(horse.get("weight_change", 0))
        row["surface_code"] = surface_code
        row["distance"] = distance
        row["track_condition_code"] = track_condition_code
        row["race_class_code"] = race_class_code
        row["distance_cat"] = distance_cat
        row["course_id_code"] = course_id_code
        row["race_month"] = race_month

        _compute_past_features(row, horse.get("past_races", []), race, surface, distance_cat)

        jockey_stats = horse.get("jockey_stats", {})
        row["jockey_win_rate_365d"] = jockey_stats.get("win_rate", 0.0)
        row["jockey_place_rate_365d"] = jockey_stats.get("place_rate", 0.0)
        row["jockey_ride_count_365d"] = jockey_stats.get("races", 0)

        # 調教師成績（enriched_input.jsonのtrainer_name → results.csvルックアップ）
        trainer_name = horse.get("trainer_name", "")
        if trainer_name and trainer_name in trainer_lookup:
            row["trainer_win_rate_365d"] = trainer_lookup[trainer_name]["win_rate"]
            row["trainer_place_rate_365d"] = trainer_lookup[trainer_name]["place_rate"]
        else:
            row["trainer_win_rate_365d"] = 0.0
            row["trainer_place_rate_365d"] = 0.0

        row["odds_win"] = horse.get("odds_win", 0)
        row["odds_place"] = horse.get("odds_place", 0)
        row["win_odds"] = horse.get("odds_win", 0)
        row["log_odds"] = float(np.log1p(max(horse.get("odds_win", 0), 0)))
        row["popularity"] = horse.get("popularity", 0)
        row["num"] = horse.get("num", 0)
        row["name"] = horse.get("name", "")

        rows.append(row)

    return pd.DataFrame(rows)
