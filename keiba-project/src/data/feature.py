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
        # v5: ローテーションパターン
        "prev_interval_2", "is_second_start",
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

            # v5: ローテーションパターン（前走間隔 + 前々走間隔）
            if len(past) >= 2:
                d0 = past.iloc[-1]["race_date"]
                d1 = past.iloc[-2]["race_date"]
                current_date = df.at[idx, "race_date"]
                if pd.notna(d0) and pd.notna(d1) and pd.notna(current_date):
                    interval_1 = (current_date - d0).days
                    interval_2 = (d0 - d1).days
                    df.at[idx, "prev_interval_2"] = interval_2
                    # 叩き良化判定: 前々走間隔>60日 and 前走間隔<45日 → 叩き2走目
                    df.at[idx, "is_second_start"] = 1.0 if interval_2 > 60 and interval_1 < 45 else 0.0

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

def compute_race_pace_features(df: pd.DataFrame) -> pd.DataFrame:
    """レース内の脚質構成から展開予測特徴量を計算
    running_style: 0=逃げ, 1=先行, 2=差し, 3=追込
    """
    pace_cols = ["race_n_front", "race_n_mid", "race_n_back", "pace_advantage"]
    for col in pace_cols:
        df[col] = 0.0

    for race_id, group in df.groupby("race_id"):
        styles = group["running_style"].values
        n_front = np.sum(styles <= 1)   # 逃げ+先行
        n_mid = np.sum(styles == 2)     # 差し
        n_back = np.sum(styles == 3)    # 追込

        indices = group.index
        df.loc[indices, "race_n_front"] = n_front
        df.loc[indices, "race_n_mid"] = n_mid
        df.loc[indices, "race_n_back"] = n_back

        # 自分の脚質に対するペース有利不利
        # 逃げ・先行馬が多い=ハイペース=差し有利
        # 逃げ・先行馬が少ない=スローペース=先行有利
        for idx in indices:
            style = df.at[idx, "running_style"]
            if n_front >= 4:
                # ハイペース: 差し・追込に有利
                df.at[idx, "pace_advantage"] = 1.0 if style >= 2 else -1.0
            elif n_front <= 2:
                # スローペース: 逃げ・先行に有利
                df.at[idx, "pace_advantage"] = 1.0 if style <= 1 else -1.0
            else:
                df.at[idx, "pace_advantage"] = 0.0

    return df


def compute_post_position_bias(df: pd.DataFrame) -> pd.DataFrame:
    """コース×距離カテゴリ×枠番の複勝率バイアスを計算（リーケージ防止）"""
    df["post_position_bias"] = 0.0
    if "course_id_code" not in df.columns or "frame_number" not in df.columns:
        return df

    df = df.sort_values(["race_date", "race_id", "finish_position"]).reset_index(drop=True)
    dates = sorted(df["race_date"].unique())

    # 累積統計を効率的に計算するためにdaily batch
    # キー: (course_id_code, distance_cat, frame_number) → {place_count, total_count}
    cumulative = {}

    for date in dates:
        date_mask = df["race_date"] == date
        date_indices = df[date_mask].index

        # 現在の累積値を適用（この日のレースに対して過去データを使う）
        for idx in date_indices:
            key = (
                int(df.at[idx, "course_id_code"]),
                int(df.at[idx, "distance_cat"]),
                int(df.at[idx, "frame_number"]),
            )
            stats = cumulative.get(key)
            if stats and stats["total"] >= 20:
                # 複勝率の偏差（全体平均≒22%からの乖離）
                rate = stats["place"] / stats["total"]
                df.at[idx, "post_position_bias"] = round(rate - 0.22, 4)

        # この日のデータを累積に追加
        for idx in date_indices:
            key = (
                int(df.at[idx, "course_id_code"]),
                int(df.at[idx, "distance_cat"]),
                int(df.at[idx, "frame_number"]),
            )
            if key not in cumulative:
                cumulative[key] = {"place": 0, "total": 0}
            cumulative[key]["total"] += 1
            if df.at[idx, "finish_position"] <= 3:
                cumulative[key]["place"] += 1

    return df


# ============================================================
# Phase 5-1: レース内相対特徴量（Z-score）
# ============================================================

Z_SCORE_BASE_COLS = [
    "surface_place_rate", "jockey_place_rate_365d",
    "avg_finish_last5", "career_place_rate", "trainer_place_rate_365d",
]


def compute_zscore_features(df: pd.DataFrame) -> pd.DataFrame:
    """レース内Z-score特徴量を計算（自馬 - レース平均）/ レース標準偏差"""
    for col in Z_SCORE_BASE_COLS:
        zcol = f"z_{col}"
        df[zcol] = 0.0

    for race_id, group in df.groupby("race_id"):
        if len(group) < 3:
            continue
        indices = group.index
        for col in Z_SCORE_BASE_COLS:
            if col not in group.columns:
                continue
            vals = group[col].values.astype(float)
            mean = vals.mean()
            std = vals.std()
            if std > 1e-8:
                df.loc[indices, f"z_{col}"] = (vals - mean) / std
            else:
                df.loc[indices, f"z_{col}"] = 0.0

    return df


# ============================================================
# Phase 5-2: 騎手×馬の騎乗経験特徴量
# ============================================================

def compute_jockey_horse_features(df: pd.DataFrame) -> pd.DataFrame:
    """同じ騎手×同じ馬の過去騎乗回数・勝率を計算"""
    df["same_jockey_rides"] = 0
    df["same_jockey_win_rate"] = 0.0

    # グルーピングキー
    if "horse_id" not in df.columns or df["horse_id"].isna().all():
        return df
    if "jockey_name" not in df.columns or df["jockey_name"].isna().all():
        return df

    df = df.sort_values(["race_date", "race_id", "finish_position"]).reset_index(drop=True)

    # 高速化: (horse_id, jockey_name) ペアごとに過去走を追跡
    pair_history = {}  # (horse_id, jockey_name) -> [finish_positions...]

    for idx in range(len(df)):
        hid = df.at[idx, "horse_id"]
        jname = df.at[idx, "jockey_name"]
        if pd.isna(hid) or pd.isna(jname):
            continue

        key = (hid, jname)
        past = pair_history.get(key, [])
        if past:
            df.at[idx, "same_jockey_rides"] = len(past)
            df.at[idx, "same_jockey_win_rate"] = round(sum(1 for p in past if p == 1) / len(past), 3)

        # 現在のレースを履歴に追加
        fp = df.at[idx, "finish_position"]
        if key not in pair_history:
            pair_history[key] = []
        pair_history[key].append(fp)

    return df


# ============================================================
# Phase 5-3: コース適性特徴量
# ============================================================

def compute_course_aptitude_features(df: pd.DataFrame) -> pd.DataFrame:
    """course_id × surface × distance_cat での過去成績を計算"""
    df["course_dist_win_rate"] = 0.0
    df["course_dist_place_rate"] = 0.0

    if "horse_id" not in df.columns or df["horse_id"].isna().all():
        return df

    df = df.sort_values(["race_date", "race_id", "finish_position"]).reset_index(drop=True)

    # (horse_id, course_id_code, surface_code, distance_cat) -> [finish_positions...]
    course_history = {}

    for idx in range(len(df)):
        hid = df.at[idx, "horse_id"]
        if pd.isna(hid):
            continue

        cid = int(df.at[idx, "course_id_code"]) if "course_id_code" in df.columns else 0
        sc = int(df.at[idx, "surface_code"]) if "surface_code" in df.columns else 0
        dc = int(df.at[idx, "distance_cat"]) if "distance_cat" in df.columns else 0
        key = (hid, cid, sc, dc)

        past = course_history.get(key, [])
        if len(past) >= 2:  # 最低2走以上でノイズ抑制
            arr = np.array(past)
            df.at[idx, "course_dist_win_rate"] = round((arr == 1).mean(), 3)
            df.at[idx, "course_dist_place_rate"] = round((arr <= 3).mean(), 3)

        fp = df.at[idx, "finish_position"]
        if key not in course_history:
            course_history[key] = []
        course_history[key].append(fp)

    return df


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

    print("[7/13] 展開予測特徴量（レース内脚質構成）を計算中...")
    df = compute_race_pace_features(df)

    print("[8/13] コース×距離×枠順バイアスを計算中...")
    df = compute_post_position_bias(df)

    print("[9/13] 騎手×馬の騎乗経験特徴量を計算中...")
    df = compute_jockey_horse_features(df)

    print("[10/13] コース適性特徴量を計算中...")
    df = compute_course_aptitude_features(df)

    print("[11/13] レース内Z-score特徴量を計算中...")
    df = compute_zscore_features(df)

    print("[12/13] 目的変数作成")
    df = create_target(df)

    print("[13/13] 特徴量選択・保存")
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
# enriched_input.json -> 特徴量ベクトル変換（feature_extract.py に移行）
# 後方互換 re-export
# ============================================================

# 後方互換エイリアス: settings.distance_category が正規の定義
from config.settings import distance_category as _dist_cat  # noqa: E402, F401


# 後方互換: feature_extract.py から re-export
from src.data.feature_extract import extract_features_from_enriched  # noqa: E402, F401
