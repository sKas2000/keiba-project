#!/usr/bin/env python3
"""
特徴量エンジニアリング v1.0
============================
results.csv から ML用特徴量を生成する。
データリーケージを防ぎ、時系列順を尊重して集約特徴量を計算する。

使い方:
  python feature_engineer.py [--input data/ml/raw/results.csv] [--output data/ml/processed/features.csv]
"""

VERSION = "1.0"

import sys
import re
from pathlib import Path
from argparse import ArgumentParser

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "ml"
DEFAULT_INPUT = DATA_DIR / "raw" / "results.csv"
DEFAULT_OUTPUT = DATA_DIR / "processed" / "features.csv"


# ============================================================
# データ読み込み・前処理
# ============================================================

def load_results(path: str | Path) -> pd.DataFrame:
    """results.csv を読み込み基本的な前処理を行う"""
    df = pd.read_csv(path, dtype={"race_id": str, "horse_id": str, "jockey_id": str, "course_id": str})

    # 日付をdatetime型に変換
    df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce")
    df = df.dropna(subset=["race_date"])

    # 型変換
    int_cols = ["finish_position", "frame_number", "horse_number", "age",
                "horse_weight", "horse_weight_change", "popularity", "num_entries"]
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    float_cols = ["weight_carried", "finish_time_sec", "last_3f", "win_odds", "prize_money"]
    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # 不正データ除外（除外・中止馬）
    df = df[df["finish_position"] > 0].copy()

    # 日付順ソート
    df = df.sort_values(["race_date", "race_id", "finish_position"]).reset_index(drop=True)

    return df


# ============================================================
# カテゴリカルエンコーディング
# ============================================================

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """カテゴリカル変数を数値に変換"""

    # 性別
    sex_map = {"牡": 0, "牝": 1, "セ": 2, "騸": 2}
    df["sex_code"] = df["sex"].map(sex_map).fillna(0).astype(int)

    # 馬場
    surface_map = {"芝": 0, "ダ": 1, "障": 2}
    df["surface_code"] = df["surface"].map(surface_map).fillna(0).astype(int)

    # 馬場状態
    cond_map = {"良": 0, "稍重": 1, "重": 2, "不良": 3}
    df["track_condition_code"] = df["track_condition"].map(cond_map).fillna(0).astype(int)

    # クラス
    class_map = {
        "新馬": 1, "未勝利": 2, "1勝": 3, "2勝": 4, "3勝": 5,
        "OP": 6, "オープン": 6, "リステッド": 7, "L": 7,
        "G3": 8, "G2": 9, "G1": 10,
    }
    df["race_class_code"] = df["race_class"].map(class_map).fillna(3).astype(int)

    # 距離カテゴリ
    df["distance_cat"] = pd.cut(
        df["distance"],
        bins=[0, 1400, 1800, 2200, 9999],
        labels=[0, 1, 2, 3],  # 短距離, マイル, 中距離, 長距離
    ).astype(int)

    # コースID
    df["course_id_code"] = pd.to_numeric(df["course_id"], errors="coerce").fillna(0).astype(int)

    return df


# ============================================================
# 過去成績集約特徴量
# ============================================================

def compute_horse_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    各馬の過去成績から集約特徴量を計算する。
    データリーケージ防止: 各レース時点で利用可能な過去データのみ使用。
    """
    df = df.sort_values(["race_date", "race_id"]).reset_index(drop=True)

    # 結果格納用
    feature_cols = [
        "prev_finish_1", "prev_finish_2", "prev_finish_3",
        "avg_finish_last5", "best_finish_last5",
        "win_rate_last5", "place_rate_last5",
        "avg_last3f_last5",
        "days_since_last_race",
        "total_races",
        "career_win_rate", "career_place_rate",
        "surface_win_rate", "surface_place_rate",
        "distance_cat_win_rate",
    ]
    for col in feature_cols:
        df[col] = 0.0

    # 馬ごとにグループ化して計算
    grouped = df.groupby("horse_id" if "horse_id" in df.columns and df["horse_id"].notna().any() else "horse_name")

    for _, group in grouped:
        group = group.sort_values("race_date")
        indices = group.index.tolist()

        for pos, idx in enumerate(indices):
            past = group.iloc[:pos]  # 現在レースより前のデータのみ

            if len(past) == 0:
                continue

            recent = past.tail(5)
            finishes = recent["finish_position"].values

            # 直近着順
            if len(finishes) >= 1:
                df.at[idx, "prev_finish_1"] = finishes[-1]
            if len(finishes) >= 2:
                df.at[idx, "prev_finish_2"] = finishes[-2]
            if len(finishes) >= 3:
                df.at[idx, "prev_finish_3"] = finishes[-3]

            # 平均・最高着順
            df.at[idx, "avg_finish_last5"] = round(finishes.mean(), 2)
            df.at[idx, "best_finish_last5"] = int(finishes.min())

            # 勝率・複勝率
            df.at[idx, "win_rate_last5"] = round((finishes == 1).mean(), 3)
            df.at[idx, "place_rate_last5"] = round((finishes <= 3).mean(), 3)

            # 上り3F平均
            last3f_vals = recent["last_3f"].values
            valid_3f = last3f_vals[last3f_vals > 0]
            if len(valid_3f) > 0:
                df.at[idx, "avg_last3f_last5"] = round(valid_3f.mean(), 2)

            # 前走からの日数
            last_race_date = past.iloc[-1]["race_date"]
            current_race_date = df.at[idx, "race_date"]
            if pd.notna(last_race_date) and pd.notna(current_race_date):
                df.at[idx, "days_since_last_race"] = (current_race_date - last_race_date).days

            # 通算成績
            all_finishes = past["finish_position"].values
            df.at[idx, "total_races"] = len(all_finishes)
            df.at[idx, "career_win_rate"] = round((all_finishes == 1).mean(), 3)
            df.at[idx, "career_place_rate"] = round((all_finishes <= 3).mean(), 3)

            # 同馬場成績
            current_surface = df.at[idx, "surface"]
            surface_past = past[past["surface"] == current_surface]
            if len(surface_past) > 0:
                sf = surface_past["finish_position"].values
                df.at[idx, "surface_win_rate"] = round((sf == 1).mean(), 3)
                df.at[idx, "surface_place_rate"] = round((sf <= 3).mean(), 3)

            # 同距離帯成績
            current_dist_cat = df.at[idx, "distance_cat"]
            dist_past = past[past["distance_cat"] == current_dist_cat]
            if len(dist_past) > 0:
                df_vals = dist_past["finish_position"].values
                df.at[idx, "distance_cat_win_rate"] = round((df_vals == 1).mean(), 3)

    return df


# ============================================================
# 騎手統計特徴量
# ============================================================

def compute_jockey_features(df: pd.DataFrame) -> pd.DataFrame:
    """騎手の過去365日成績を計算"""
    df["jockey_win_rate_365d"] = 0.0
    df["jockey_place_rate_365d"] = 0.0
    df["jockey_ride_count_365d"] = 0

    jockey_col = "jockey_id" if "jockey_id" in df.columns and df["jockey_id"].notna().any() else "jockey_name"
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

    return df


# ============================================================
# 目的変数
# ============================================================

def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """目的変数を作成"""
    df["top3"] = (df["finish_position"] <= 3).astype(int)
    return df


# ============================================================
# 特徴量選択
# ============================================================

FEATURE_COLUMNS = [
    # 基本
    "frame_number", "horse_number", "sex_code", "age",
    "weight_carried", "horse_weight", "horse_weight_change",
    "num_entries",
    # レース条件
    "surface_code", "distance", "track_condition_code",
    "race_class_code", "distance_cat", "course_id_code",
    # 過去成績
    "prev_finish_1", "prev_finish_2", "prev_finish_3",
    "avg_finish_last5", "best_finish_last5",
    "win_rate_last5", "place_rate_last5",
    "avg_last3f_last5", "days_since_last_race",
    "total_races", "career_win_rate", "career_place_rate",
    "surface_win_rate", "surface_place_rate",
    "distance_cat_win_rate",
    # 騎手
    "jockey_win_rate_365d", "jockey_place_rate_365d",
    "jockey_ride_count_365d",
]

META_COLUMNS = [
    "race_id", "race_date", "horse_name", "horse_id",
    "jockey_name", "finish_position", "top3",
    "win_odds", "popularity",
]


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """ML用の特徴量+メタデータのみに絞る"""
    available = [c for c in FEATURE_COLUMNS + META_COLUMNS if c in df.columns]
    return df[available].copy()


# ============================================================
# パイプライン実行
# ============================================================

def run_pipeline(input_path: str | Path, output_path: str | Path) -> pd.DataFrame:
    """特徴量エンジニアリングパイプラインを実行"""
    print(f"\n{'=' * 60}")
    print(f"  特徴量エンジニアリング v{VERSION}")
    print(f"{'=' * 60}")

    # 1. データ読み込み
    print(f"\n[1/6] データ読み込み: {input_path}")
    df = load_results(input_path)
    print(f"  → {len(df)}行, {df['race_id'].nunique()}レース")

    # 2. カテゴリカルエンコーディング
    print("[2/6] カテゴリカルエンコーディング")
    df = encode_categoricals(df)

    # 3. 過去成績特徴量
    print("[3/6] 過去成績集約特徴量を計算中...")
    df = compute_horse_history_features(df)

    # 4. 騎手統計特徴量
    print("[4/6] 騎手統計特徴量を計算中...")
    df = compute_jockey_features(df)

    # 5. 目的変数作成
    print("[5/6] 目的変数作成")
    df = create_target(df)

    # 6. 特徴量選択・保存
    print("[6/6] 特徴量選択・保存")
    df_features = select_features(df)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df_features.to_csv(output_path, index=False)

    print(f"\n  特徴量数: {len(FEATURE_COLUMNS)}")
    print(f"  出力: {output_path}")
    print(f"  行数: {len(df_features)}")
    print(f"  3着内率: {df_features['top3'].mean():.1%}")
    print(f"{'=' * 60}\n")

    return df_features


def main():
    parser = ArgumentParser(description="特徴量エンジニアリング")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="入力CSV")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="出力CSV")
    args = parser.parse_args()

    run_pipeline(args.input, args.output)


if __name__ == "__main__":
    main()
