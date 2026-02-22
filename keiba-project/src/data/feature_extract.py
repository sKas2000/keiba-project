"""
enriched_input.json からの特徴量抽出（ライブ予測用）
CSV ベースの特徴量は feature.py を参照
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import (
    FEATURE_COLUMNS,
    SEX_MAP, SURFACE_MAP, TRACK_CONDITION_MAP, CLASS_MAP, COURSE_NAME_TO_ID,
    RAW_DIR,
)
from src.data.feature import _dist_cat, Z_SCORE_BASE_COLS
from src.scraping.parsers import safe_float, safe_int


# ============================================================
# 過去走データから特徴量計算
# ============================================================

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
                   "class_change", "weight_carried_change",
                   "prev_interval_2"]:
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

    # surface比較: "ダート"と"ダ"の不一致を吸収（先頭1文字で比較）
    surface_key = surface[:1] if surface else ""
    surface_races = [r for r in past_races if r.get("surface", "")[:1] == surface_key]
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

    # v5: 前々走間隔（prev_interval_2）
    row["prev_interval_2"] = 0
    if len(past_races) >= 2:
        d0 = past_races[0].get("date", "")
        d1 = past_races[1].get("date", "")
        if d0 and d1:
            try:
                from datetime import datetime
                dt0 = datetime.strptime(d0.replace("/", "-"), "%Y-%m-%d")
                dt1 = datetime.strptime(d1.replace("/", "-"), "%Y-%m-%d")
                row["prev_interval_2"] = abs((dt0 - dt1).days)
            except Exception:
                pass


# ============================================================
# results.csv ルックアップ関数群
# ============================================================

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

        df["_short"] = df["trainer_name"].str.replace(
            r"^\[.+?\]", "", regex=True
        ).str.replace(r"^(美浦|栗東|地方)", "", regex=True)

        bracket_mask = df["trainer_name"].str.startswith("[")
        full_to_short = {}
        if bracket_mask.any():
            for full_name in df.loc[bracket_mask, "_short"].unique():
                short_names = df.loc[~bracket_mask, "_short"].unique()
                for short in short_names:
                    if len(short) >= 2 and full_name.startswith(short) and full_name != short:
                        full_to_short[full_name] = short
                        break

        df["_key"] = df["_short"].map(lambda x: full_to_short.get(x, x))

        cutoff = race_date - timedelta(days=365)
        recent = df[(df["race_date"] >= cutoff) & (df["race_date"] < race_date)]
        if len(recent) == 0:
            return {}
        stats = recent.groupby("_key")["finish_position"].agg(
            win_rate=lambda x: round((x == 1).mean(), 3),
            place_rate=lambda x: round((x <= 3).mean(), 3),
        )
        result = stats.to_dict(orient="index")

        for full_name, short in full_to_short.items():
            if short in result and full_name not in result:
                result[full_name] = result[short]

        return result
    except Exception:
        return {}


def _load_jockey_horse_course_lookup(
    race_date_str: str, course_id_code: int, surface_code: int, distance_cat: int
) -> tuple:
    """results.csvから騎手×馬の過去騎乗・コース適性をルックアップ辞書で返す"""
    results_path = RAW_DIR / "results.csv"
    if not results_path.exists():
        return {}, {}
    try:
        from datetime import datetime
        race_date = datetime.strptime(race_date_str.replace("/", "-"), "%Y-%m-%d")
        cols = ["race_date", "horse_id", "jockey_name", "finish_position",
                "course_id", "surface", "distance"]
        df = pd.read_csv(results_path, dtype={"race_id": str, "horse_id": str},
                         usecols=[c for c in cols if c in pd.read_csv(results_path, nrows=0).columns])
        df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce")
        df = df.dropna(subset=["race_date", "horse_id"])
        df["finish_position"] = pd.to_numeric(df["finish_position"], errors="coerce")
        df = df[(df["finish_position"] > 0) & (df["race_date"] < race_date)]

        jh_lookup = {}
        if "jockey_name" in df.columns:
            for (hid, jname), grp in df.groupby(["horse_id", "jockey_name"]):
                fps = grp["finish_position"].values
                jh_lookup[(hid, jname)] = {
                    "rides": len(fps),
                    "win_rate": round((fps == 1).mean(), 3),
                }

        ca_lookup = {}
        if "surface" in df.columns and "distance" in df.columns:
            df["_sc"] = df["surface"].map(SURFACE_MAP).fillna(-1).astype(int)
            df["_dc"] = df["distance"].apply(
                lambda d: _dist_cat(int(d)) if pd.notna(d) and int(d) > 0 else -1)
            if "course_id" in df.columns:
                df["_cid"] = pd.to_numeric(df["course_id"], errors="coerce").fillna(0).astype(int)
            else:
                df["_cid"] = 0
            matched = df[(df["_cid"] == course_id_code) &
                         (df["_sc"] == surface_code) &
                         (df["_dc"] == distance_cat)]
            for hid, grp in matched.groupby("horse_id"):
                fps = grp["finish_position"].values
                if len(fps) >= 2:
                    ca_lookup[hid] = {
                        "win_rate": round((fps == 1).mean(), 3),
                        "place_rate": round((fps <= 3).mean(), 3),
                    }

        return jh_lookup, ca_lookup
    except Exception:
        return {}, {}


def _load_jockey_lookup(race_date_str: str) -> dict:
    """results.csvから騎手の直近365日成績をルックアップ辞書で返す"""
    results_path = RAW_DIR / "results.csv"
    if not results_path.exists():
        return {}
    try:
        from datetime import datetime, timedelta
        race_date = datetime.strptime(race_date_str.replace("/", "-"), "%Y-%m-%d")
        df = pd.read_csv(results_path, dtype={"race_id": str},
                         usecols=["race_date", "jockey_name", "finish_position"])
        df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce")
        df = df.dropna(subset=["race_date", "jockey_name"])
        df["finish_position"] = pd.to_numeric(df["finish_position"], errors="coerce")
        df = df[df["finish_position"] > 0]

        df["jockey_name"] = df["jockey_name"].str.replace(
            r"^[▲△☆◇]", "", regex=True)

        cutoff = race_date - timedelta(days=365)
        recent = df[(df["race_date"] >= cutoff) & (df["race_date"] < race_date)]
        if len(recent) == 0:
            return {}
        stats = recent.groupby("jockey_name").agg(
            win_rate=("finish_position", lambda x: round((x == 1).mean(), 3)),
            place_rate=("finish_position", lambda x: round((x <= 3).mean(), 3)),
            count=("finish_position", "count"),
        )
        return stats.to_dict(orient="index")
    except Exception:
        return {}


def _load_post_position_bias_lookup(
    race_date_str: str, course_id_code: int, distance_cat: int
) -> dict:
    """results.csvからコース×距離カテゴリ×枠番の複勝率バイアスをルックアップ"""
    results_path = RAW_DIR / "results.csv"
    if not results_path.exists():
        return {}
    try:
        from datetime import datetime
        race_date = datetime.strptime(race_date_str.replace("/", "-"), "%Y-%m-%d")
        cols = ["race_date", "course_id", "distance", "frame_number",
                "finish_position"]
        df = pd.read_csv(results_path, dtype={"race_id": str},
                         usecols=cols)
        df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce")
        df = df.dropna(subset=["race_date"])
        df["finish_position"] = pd.to_numeric(df["finish_position"], errors="coerce")
        df = df[(df["finish_position"] > 0) & (df["race_date"] < race_date)]
        df["frame_number"] = pd.to_numeric(df["frame_number"], errors="coerce")
        df = df[df["frame_number"] > 0]

        df["_cid"] = pd.to_numeric(df["course_id"], errors="coerce").fillna(0).astype(int)
        df["_dc"] = df["distance"].apply(
            lambda d: _dist_cat(int(d)) if pd.notna(d) and int(d) > 0 else -1)
        matched = df[(df["_cid"] == course_id_code) & (df["_dc"] == distance_cat)]

        lookup = {}
        for fn, grp in matched.groupby("frame_number"):
            if len(grp) >= 20:
                rate = (grp["finish_position"] <= 3).mean()
                lookup[int(fn)] = round(rate - 0.22, 4)
        return lookup
    except Exception:
        return {}


# ============================================================
# メイン: enriched_input.json -> 特徴量ベクトル
# ============================================================

def extract_features_from_enriched(data: dict) -> pd.DataFrame:
    """enriched_input.json -> ML 特徴量 DataFrame"""
    race = data.get("race", {})
    horses = data.get("horses", [])
    num_entries = len(horses)

    surface = race.get("surface", "")
    surface_code = SURFACE_MAP.get(surface, SURFACE_MAP.get(surface[:1], 0) if surface else 0)
    distance = race.get("distance", 0)
    track_condition_code = TRACK_CONDITION_MAP.get(race.get("track_condition", ""), 0)
    race_class_code = CLASS_MAP.get(race.get("grade", ""), 3)
    course_id_code = COURSE_NAME_TO_ID.get(race.get("venue", ""), 0)
    distance_cat = _dist_cat(distance)

    race_date_str = race.get("date", "")
    race_month = 0
    if race_date_str:
        try:
            from datetime import datetime
            dt = datetime.strptime(race_date_str.replace("/", "-"), "%Y-%m-%d")
            race_month = dt.month
        except Exception:
            pass

    trainer_lookup = _load_trainer_lookup(race_date_str) if race_date_str else {}
    jockey_lookup = _load_jockey_lookup(race_date_str) if race_date_str else {}

    jh_lookup, ca_lookup = _load_jockey_horse_course_lookup(
        race_date_str, course_id_code, surface_code, distance_cat)

    def _estimate_frame(horse_num, n):
        if n <= 8:
            return horse_num
        doubles = n - 8
        singles = 8 - doubles
        if horse_num <= singles:
            return horse_num
        return singles + (horse_num - singles - 1) // 2 + 1

    rows = []
    for horse in horses:
        row = {}
        row["horse_name"] = horse.get("name", "")
        row["horse_number"] = horse.get("num", 0)
        row["frame_number"] = horse.get("frame_number", 0) or _estimate_frame(horse.get("num", 0), num_entries)
        row["num_entries"] = num_entries

        sex_age = horse.get("sex_age", "")
        m = re.match(r'([牡牝セ騸])(\d+)', sex_age)
        row["sex_code"] = SEX_MAP.get(m.group(1), 0) if m else 0
        row["age"] = int(m.group(2)) if m else 0

        row["weight_carried"] = safe_float(horse.get("load_weight", 0))

        weight_raw = str(horse.get("weight", "0"))
        wm = re.match(r'(\d+)\s*\(([+-]?\d+)\)', weight_raw)
        if wm:
            row["horse_weight"] = int(wm.group(1))
            row["horse_weight_change"] = int(wm.group(2))
        else:
            row["horse_weight"] = safe_int(weight_raw)
            row["horse_weight_change"] = safe_int(horse.get("weight_change", 0))

        if row["horse_weight"] == 0:
            past_races = horse.get("past_races", [])
            for pr in past_races:
                pw = pr.get("horse_weight", 0)
                if pw and int(pw) > 200:
                    row["horse_weight"] = int(pw)
                    break
            if row["horse_weight"] == 0:
                sex = m.group(1) if m else ""
                row["horse_weight"] = 440 if sex == "牝" else 480
        row["surface_code"] = surface_code
        row["distance"] = distance
        row["track_condition_code"] = track_condition_code
        row["race_class_code"] = race_class_code
        row["distance_cat"] = distance_cat
        row["course_id_code"] = course_id_code
        row["race_month"] = race_month

        _compute_past_features(row, horse.get("past_races", []), race, surface, distance_cat)

        jockey_stats = horse.get("jockey_stats", {})
        jockey_name_raw = horse.get("jockey", "")
        jockey_name_clean = jockey_name_raw.replace(" ", "").replace("\u3000", "")
        jl = None
        if jockey_name_clean:
            if jockey_name_clean in jockey_lookup:
                jl = jockey_lookup[jockey_name_clean]
            else:
                for plen in [3, 2]:
                    if len(jockey_name_clean) >= plen:
                        prefix = jockey_name_clean[:plen]
                        if prefix in jockey_lookup:
                            jl = jockey_lookup[prefix]
                            break
        if jl:
            row["jockey_win_rate_365d"] = jl["win_rate"]
            row["jockey_place_rate_365d"] = jl["place_rate"]
            row["jockey_ride_count_365d"] = jl["count"]
        else:
            row["jockey_win_rate_365d"] = jockey_stats.get("win_rate", 0.0)
            row["jockey_place_rate_365d"] = jockey_stats.get("place_rate", 0.0)
            row["jockey_ride_count_365d"] = jockey_stats.get("races", 0)

        trainer_name = horse.get("trainer_name", "")
        tl = None
        if trainer_name:
            if trainer_name in trainer_lookup:
                tl = trainer_lookup[trainer_name]
            else:
                for plen in [3, 2]:
                    if len(trainer_name) >= plen:
                        prefix = trainer_name[:plen]
                        if prefix in trainer_lookup:
                            tl = trainer_lookup[prefix]
                            break
        if tl:
            row["trainer_win_rate_365d"] = tl["win_rate"]
            row["trainer_place_rate_365d"] = tl["place_rate"]
        else:
            row["trainer_win_rate_365d"] = 0.0
            row["trainer_place_rate_365d"] = 0.0

        horse_id = horse.get("horse_id", "")
        jockey_name = horse.get("jockey_name", jockey_stats.get("name", ""))
        jh_key = (horse_id, jockey_name)
        if jh_key in jh_lookup:
            jh = jh_lookup[jh_key]
            row["same_jockey_rides"] = jh["rides"]
            row["same_jockey_win_rate"] = jh["win_rate"]
        else:
            row["same_jockey_rides"] = 0
            row["same_jockey_win_rate"] = 0.0

        if horse_id and horse_id in ca_lookup:
            ca = ca_lookup[horse_id]
            row["course_dist_win_rate"] = ca["win_rate"]
            row["course_dist_place_rate"] = ca["place_rate"]
        else:
            row["course_dist_win_rate"] = 0.0
            row["course_dist_place_rate"] = 0.0

        row["odds_win"] = horse.get("odds_win", 0)
        row["odds_place"] = horse.get("odds_place", 0)
        row["win_odds"] = horse.get("odds_win", 0)
        row["log_odds"] = float(np.log1p(max(horse.get("odds_win", 0), 0)))
        row["popularity"] = horse.get("popularity", 0)
        row["num"] = horse.get("num", 0)
        row["name"] = horse.get("name", "")

        rows.append(row)

    df = pd.DataFrame(rows)

    # v6: 展開予測特徴量（レース内脚質構成）
    if "running_style" in df.columns:
        styles = df["running_style"].values
        n_front = int(np.sum(styles <= 1))
        n_mid = int(np.sum(styles == 2))
        n_back = int(np.sum(styles == 3))
        df["race_n_front"] = n_front
        df["race_n_mid"] = n_mid
        df["race_n_back"] = n_back
        df["pace_advantage"] = 0.0
        for idx in df.index:
            style = df.at[idx, "running_style"]
            if n_front >= 4:
                df.at[idx, "pace_advantage"] = 1.0 if style >= 2 else -1.0
            elif n_front <= 2:
                df.at[idx, "pace_advantage"] = 1.0 if style <= 1 else -1.0
    else:
        df["race_n_front"] = 0
        df["race_n_mid"] = 0
        df["race_n_back"] = 0
        df["pace_advantage"] = 0.0

    # v7: コース×距離×枠順バイアス
    df["post_position_bias"] = 0.0
    if "frame_number" in df.columns:
        ppb_lookup = _load_post_position_bias_lookup(
            race_date_str, course_id_code, distance_cat)
        if ppb_lookup:
            for idx in df.index:
                fn = int(df.at[idx, "frame_number"])
                if fn in ppb_lookup:
                    df.at[idx, "post_position_bias"] = ppb_lookup[fn]

    # v8: レース内Z-score
    for col in Z_SCORE_BASE_COLS:
        zcol = f"z_{col}"
        if col in df.columns and len(df) >= 3:
            vals = df[col].values.astype(float)
            mean, std = vals.mean(), vals.std()
            df[zcol] = ((vals - mean) / std) if std > 1e-8 else 0.0
        else:
            df[zcol] = 0.0

    return df
