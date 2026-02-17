#!/usr/bin/env python3
"""
MLパイプライン統合テスト v1.0
==============================
合成データを生成し、特徴量エンジニアリング → モデル学習 → 予測 → バックテスト
のパイプライン全体を検証する。

使い方:
  python test_ml_pipeline.py
"""

import sys
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

# パス設定
TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS_DIR))


# ============================================================
# 合成データ生成
# ============================================================

def generate_synthetic_results(n_races: int = 200, seed: int = 42) -> pd.DataFrame:
    """
    テスト用の合成レース結果データを生成する。
    実際のnetkeiba結果CSVと同じスキーマ。
    """
    rng = np.random.RandomState(seed)

    rows = []
    courses = [("05", "東京"), ("06", "中山"), ("08", "京都"), ("09", "阪神")]
    surfaces = ["芝", "ダ"]
    conditions = ["良", "稍重", "重"]
    classes = ["未勝利", "1勝", "2勝", "3勝", "OP", "G3", "G2", "G1"]
    distances = [1200, 1400, 1600, 1800, 2000, 2200, 2400]

    # 50頭の馬プール
    horse_pool = [f"テストホース{i:02d}" for i in range(50)]
    horse_ids = [f"20200000{i:02d}" for i in range(50)]
    horse_abilities = rng.normal(50, 15, 50)  # 各馬の実力値

    # 20人の騎手プール
    jockey_pool = [f"ジョッキー{i:02d}" for i in range(20)]
    jockey_ids = [f"0{i:04d}" for i in range(20)]
    jockey_skill = rng.normal(0, 5, 20)

    base_date = pd.Timestamp("2024-01-06")

    for race_idx in range(n_races):
        race_date = base_date + pd.Timedelta(weeks=race_idx // 12)
        course_id, course_name = courses[rng.randint(len(courses))]
        race_number = (race_idx % 12) + 1
        race_id = f"{race_date.year}{course_id}{race_idx // 12 + 1:02d}01{race_number:02d}"

        surface = rng.choice(surfaces)
        distance = rng.choice(distances)
        condition = rng.choice(conditions)
        race_class = classes[min(race_idx // 25, len(classes) - 1)]
        n_entries = rng.randint(8, 17)

        # 出走馬を選択
        horse_indices = rng.choice(50, size=n_entries, replace=False)

        # 各馬のパフォーマンスを計算（実力 + 騎手 + ランダム）
        jockey_indices = rng.choice(20, size=n_entries, replace=True)
        performances = []
        for h_idx, j_idx in zip(horse_indices, jockey_indices):
            perf = horse_abilities[h_idx] + jockey_skill[j_idx] + rng.normal(0, 10)
            performances.append(perf)

        # 着順はパフォーマンス降順
        finish_order = np.argsort(-np.array(performances)) + 1

        for pos, (h_idx, j_idx) in enumerate(zip(horse_indices, jockey_indices)):
            finish = int(finish_order[pos])

            # タイムは距離に応じた基準 + ランダム
            base_time = distance / 16.5  # 約16.5m/秒
            finish_time = base_time + (finish - 1) * 0.2 + rng.normal(0, 0.5)

            # オッズ（実力の逆数ベース）
            ability_rank = np.argsort(-horse_abilities[horse_indices])
            odds_base = (np.where(ability_rank == pos)[0][0] + 1) * 2.5
            odds = max(1.1, odds_base + rng.normal(0, 3))

            rows.append({
                "race_id": race_id,
                "race_date": race_date.strftime("%Y-%m-%d"),
                "course_id": course_id,
                "course_name": course_name,
                "race_number": race_number,
                "race_name": f"テストレース{race_idx+1}",
                "race_class": race_class,
                "surface": surface,
                "distance": distance,
                "track_condition": condition,
                "weather": "晴",
                "num_entries": n_entries,
                "finish_position": finish,
                "frame_number": (pos // 2) + 1,
                "horse_number": pos + 1,
                "horse_name": horse_pool[h_idx],
                "horse_id": horse_ids[h_idx],
                "sex": rng.choice(["牡", "牝"]),
                "age": rng.randint(2, 8),
                "jockey_name": jockey_pool[j_idx],
                "jockey_id": jockey_ids[j_idx],
                "trainer_name": f"トレーナー{j_idx % 10:02d}",
                "weight_carried": rng.choice([54.0, 55.0, 56.0, 57.0]),
                "horse_weight": rng.randint(440, 520),
                "horse_weight_change": rng.randint(-10, 11),
                "finish_time_sec": round(finish_time, 1),
                "margin": f"{(finish-1)*0.2:.1f}" if finish > 1 else "",
                "passing_order": f"{min(finish+rng.randint(-2,3), n_entries)}-{finish}",
                "last_3f": round(33.0 + rng.normal(0, 1.5) + (finish - 1) * 0.3, 1),
                "win_odds": round(odds, 1),
                "popularity": int(np.where(ability_rank == pos)[0][0] + 1),
                "prize_money": max(0, (n_entries - finish + 1) * 100) if finish <= 5 else 0,
            })

    return pd.DataFrame(rows)


# ============================================================
# テスト関数
# ============================================================

def test_data_generation():
    """合成データ生成テスト"""
    print("[TEST] 合成データ生成...")
    df = generate_synthetic_results(n_races=100)

    assert len(df) > 0, "データが空"
    assert "race_id" in df.columns, "race_id カラムなし"
    assert "finish_position" in df.columns, "finish_position カラムなし"
    assert df["finish_position"].min() >= 1, "着順が1未満"
    assert df["race_id"].nunique() == 100, f"レース数が100でない: {df['race_id'].nunique()}"

    print(f"  → OK: {len(df)}行, {df['race_id'].nunique()}レース, {df['horse_name'].nunique()}頭")
    return df


def test_feature_engineering(df: pd.DataFrame, tmp_dir: Path):
    """特徴量エンジニアリングテスト"""
    print("[TEST] 特徴量エンジニアリング...")

    from feature_engineer import (
        load_results, encode_categoricals, compute_horse_history_features,
        compute_jockey_features, create_target, select_features, FEATURE_COLUMNS,
    )

    # CSVに一時保存して読み込み
    raw_path = tmp_dir / "results.csv"
    df.to_csv(raw_path, index=False)
    df_loaded = load_results(raw_path)
    assert len(df_loaded) > 0, "読み込みデータが空"

    # エンコーディング
    df_enc = encode_categoricals(df_loaded)
    assert "sex_code" in df_enc.columns, "sex_code なし"
    assert "surface_code" in df_enc.columns, "surface_code なし"

    # 過去成績特徴量
    df_hist = compute_horse_history_features(df_enc)
    assert "avg_finish_last5" in df_hist.columns, "avg_finish_last5 なし"
    assert "prev_finish_1" in df_hist.columns, "prev_finish_1 なし"

    # 値の妥当性: 最初のレースの馬は過去データがない
    first_race_id = df_hist["race_id"].iloc[0]
    first_race = df_hist[df_hist["race_id"] == first_race_id]
    # 全馬の過去特徴量が0であること（まだ過去走がないため）
    assert (first_race["avg_finish_last5"] == 0).all(), "初戦なのに過去特徴量が非ゼロ"

    # 騎手統計
    df_jockey = compute_jockey_features(df_hist)
    assert "jockey_win_rate_365d" in df_jockey.columns, "jockey_win_rate_365d なし"

    # 目的変数
    df_target = create_target(df_jockey)
    assert "top3" in df_target.columns, "top3 なし"
    assert set(df_target["top3"].unique()).issubset({0, 1}), "top3 が 0/1 でない"

    # 特徴量選択
    df_features = select_features(df_target)
    available = [c for c in FEATURE_COLUMNS if c in df_features.columns]
    assert len(available) >= 20, f"利用可能特徴量が少なすぎ: {len(available)}"

    features_path = tmp_dir / "features.csv"
    df_features.to_csv(features_path, index=False)

    print(f"  → OK: {len(df_features)}行, 特徴量{len(available)}個, 3着内率={df_features['top3'].mean():.1%}")
    return features_path


def test_model_training(features_path: Path, tmp_dir: Path):
    """モデル学習テスト"""
    print("[TEST] モデル学習...")

    from model_trainer import (
        time_based_split, train_binary_model, train_ranking_model,
        save_model, get_feature_importance, FEATURE_COLUMNS,
    )

    df = pd.read_csv(features_path, dtype={"race_id": str, "horse_id": str})
    df["race_date"] = pd.to_datetime(df["race_date"])

    # 時系列分割（データの後半1/4を検証に）
    dates = sorted(df["race_date"].unique())
    split_idx = int(len(dates) * 0.75)
    val_start = str(dates[split_idx].date())

    train_df, val_df = time_based_split(df, val_start)
    assert len(train_df) > 0, "学習データが空"
    assert len(val_df) > 0, "検証データが空"
    print(f"  学習: {len(train_df)}行, 検証: {len(val_df)}行")

    # 二値分類モデル
    binary_model, binary_metrics = train_binary_model(train_df, val_df)
    assert binary_metrics["auc"] > 0.5, f"AUC が 0.5 以下: {binary_metrics['auc']}"
    print(f"  二値分類: AUC={binary_metrics['auc']:.4f}, "
          f"Top3的中={binary_metrics['top3_hit_rate']:.1%}")

    # ランキングモデル
    rank_model, rank_metrics = train_ranking_model(train_df, val_df)
    assert rank_metrics["top3_hit_rate"] > 0, "ランキングTop3的中率が0"
    print(f"  ランキング: Top3的中={rank_metrics['top3_hit_rate']:.1%}")

    # 特徴量重要度
    available = [c for c in FEATURE_COLUMNS if c in df.columns]
    fi = get_feature_importance(binary_model, available)
    assert len(fi) > 0, "特徴量重要度が空"
    print(f"  最重要特徴量: {fi.iloc[0]['feature']} ({fi.iloc[0]['importance_pct']:.1f}%)")

    # モデル保存
    model_dir = tmp_dir / "models"
    save_model(binary_model, binary_metrics, available, "binary", model_dir)
    save_model(rank_model, rank_metrics, available, "ranking", model_dir)
    assert (model_dir / "binary_model.txt").exists(), "二値モデル保存失敗"
    assert (model_dir / "ranking_model.txt").exists(), "ランキングモデル保存失敗"

    print(f"  → OK: モデル保存完了")
    return model_dir, val_start


def test_predictor(tmp_dir: Path, model_dir: Path):
    """予測パイプラインテスト"""
    print("[TEST] 予測パイプライン...")

    from predictor import extract_features_from_enriched, predict_race

    # テスト用enriched_input.json を作成
    test_data = {
        "race": {
            "date": "2025-03-01",
            "venue": "東京",
            "race_number": 5,
            "name": "テスト500万下",
            "grade": "1勝",
            "surface": "芝",
            "distance": 1600,
            "direction": "左",
            "entries": 14,
            "weather": "晴",
            "track_condition": "良",
        },
        "horses": [
            {
                "num": i + 1,
                "name": f"テスト馬{i+1}",
                "sex_age": f"牡{3 + i % 4}",
                "load_weight": 55.0 + (i % 3),
                "jockey": f"騎手{i % 5}",
                "odds_win": 5.0 + i * 2,
                "odds_place": 2.0 + i * 0.5,
                "frame_number": (i // 2) + 1,
                "past_races": [
                    {
                        "date": "2025/02/15",
                        "venue": "東京",
                        "surface": "芝",
                        "distance": "1600",
                        "finish": 2 + (i % 5),
                        "last3f": f"{33.5 + i * 0.3:.1f}",
                        "margin": "0.3",
                        "jockey_name": f"騎手{i % 5}",
                        "jockey_id": f"0{i % 5:04d}",
                    },
                    {
                        "date": "2025/01/20",
                        "venue": "中山",
                        "surface": "芝",
                        "distance": "1800",
                        "finish": 3 + (i % 6),
                        "last3f": f"{34.0 + i * 0.2:.1f}",
                        "margin": "0.5",
                        "jockey_name": f"騎手{i % 5}",
                        "jockey_id": f"0{i % 5:04d}",
                    },
                ],
                "jockey_stats": {
                    "win_rate": 0.15 - i * 0.01,
                    "place_rate": 0.35 - i * 0.02,
                    "wins": 20 - i,
                    "races": 100,
                },
                "score": 0,
                "score_breakdown": {},
                "note": "",
            }
            for i in range(10)
        ],
    }

    # 特徴量抽出テスト
    df = extract_features_from_enriched(test_data)
    assert len(df) == 10, f"馬数が10でない: {len(df)}"
    assert "prev_finish_1" in df.columns, "prev_finish_1 なし"
    assert "jockey_win_rate_365d" in df.columns, "jockey_win_rate_365d なし"
    print(f"  特徴量抽出: {len(df)}馬, {len(df.columns)}カラム")

    # 予測テスト（モデルがある場合）
    result = predict_race(test_data, model_dir)
    if result is not None:
        horses = result.get("horses", [])
        assert len(horses) == 10, "予測馬数が10でない"
        assert all("ml_top3_prob" in h for h in horses), "ml_top3_prob なし"

        probs = [h["ml_top3_prob"] for h in horses]
        assert all(0 <= p <= 1 for p in probs), "確率が0-1の範囲外"
        print(f"  予測: top確率={max(probs):.3f}, min確率={min(probs):.3f}")
    else:
        print(f"  予測: モデル未学習のためスキップ")

    print(f"  → OK")


def test_backtest(features_path: Path, model_dir: Path, val_start: str):
    """バックテストテスト"""
    print("[TEST] バックテスト...")

    from backtest import run_backtest, print_backtest_report

    results = run_backtest(
        str(features_path),
        str(model_dir),
        val_start,
        bet_threshold=0.30,
        top_n=3,
    )

    assert results, "バックテスト結果が空"
    assert results["races"] > 0, "レース数が0"
    assert "prediction_accuracy" in results, "prediction_accuracy なし"

    pa = results["prediction_accuracy"]
    total = pa["total"]
    if total > 0:
        top1_rate = pa["top1_hit"] / total
        print(f"  1着的中率: {top1_rate:.1%}")
        print(f"  レース数: {results['races']}")

    print_backtest_report(results)

    print(f"  → OK")


# ============================================================
# メイン
# ============================================================

def main():
    print()
    print("=" * 60)
    print("  MLパイプライン統合テスト v1.0")
    print("=" * 60)
    print()

    passed = 0
    failed = 0
    errors = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # 1. 合成データ生成
        try:
            df = test_data_generation()
            passed += 1
        except Exception as e:
            failed += 1
            errors.append(f"合成データ生成: {e}")
            import traceback; traceback.print_exc()
            return

        # 2. 特徴量エンジニアリング
        try:
            features_path = test_feature_engineering(df, tmp_path)
            passed += 1
        except Exception as e:
            failed += 1
            errors.append(f"特徴量エンジニアリング: {e}")
            import traceback; traceback.print_exc()
            return

        # 3. モデル学習
        try:
            model_dir, val_start = test_model_training(features_path, tmp_path)
            passed += 1
        except Exception as e:
            failed += 1
            errors.append(f"モデル学習: {e}")
            import traceback; traceback.print_exc()
            return

        # 4. 予測
        try:
            test_predictor(tmp_path, model_dir)
            passed += 1
        except Exception as e:
            failed += 1
            errors.append(f"予測パイプライン: {e}")
            import traceback; traceback.print_exc()

        # 5. バックテスト
        try:
            test_backtest(features_path, model_dir, val_start)
            passed += 1
        except Exception as e:
            failed += 1
            errors.append(f"バックテスト: {e}")
            import traceback; traceback.print_exc()

    print()
    print("=" * 60)
    print(f"  テスト結果: {passed} passed, {failed} failed")
    if errors:
        print(f"\n  エラー:")
        for err in errors:
            print(f"    - {err}")
    print("=" * 60)
    print()

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
