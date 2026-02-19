"""
統合テスト: 合成データでパイプライン全体を検証
tools/test_ml_pipeline.py の移植版
"""
import pytest
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import FEATURE_COLUMNS
from src.data.preprocessing import load_results, encode_categoricals
from src.data.feature import (
    compute_horse_history_features, compute_jockey_features,
    create_target, select_features, extract_features_from_enriched,
)
from src.model.trainer import (
    time_based_split, train_binary_model, train_ranking_model,
    save_model, get_feature_importance,
)
from src.model.evaluator import run_backtest
from src.model.predictor import score_rule_based, calculate_ev


# ============================================================
# 合成データ生成
# ============================================================

def generate_synthetic_results(n_races: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    rows = []
    courses = [("05", "東京"), ("06", "中山"), ("08", "京都"), ("09", "阪神")]
    surfaces = ["芝", "ダ"]
    conditions = ["良", "稍重", "重"]
    classes = ["未勝利", "1勝", "2勝", "3勝", "OP", "G3", "G2", "G1"]
    distances = [1200, 1400, 1600, 1800, 2000, 2200, 2400]

    horse_pool = [f"テストホース{i:02d}" for i in range(50)]
    horse_ids = [f"20200000{i:02d}" for i in range(50)]
    horse_abilities = rng.normal(50, 15, 50)
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
        horse_indices = rng.choice(50, size=n_entries, replace=False)
        jockey_indices = rng.choice(20, size=n_entries, replace=True)

        performances = []
        for h_idx, j_idx in zip(horse_indices, jockey_indices):
            perf = horse_abilities[h_idx] + jockey_skill[j_idx] + rng.normal(0, 10)
            performances.append(perf)

        finish_order = np.argsort(-np.array(performances)) + 1

        for pos, (h_idx, j_idx) in enumerate(zip(horse_indices, jockey_indices)):
            finish = int(finish_order[pos])
            base_time = distance / 16.5
            finish_time = base_time + (finish - 1) * 0.2 + rng.normal(0, 0.5)
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
# テスト
# ============================================================

@pytest.fixture(scope="module")
def synthetic_data():
    return generate_synthetic_results(n_races=100)


@pytest.fixture(scope="module")
def tmp_dir():
    # LightGBM can't write to paths with non-ASCII chars on Windows
    import shutil
    local_tmp = Path(__file__).resolve().parent.parent / ".test_tmp"
    local_tmp.mkdir(exist_ok=True)
    yield local_tmp
    shutil.rmtree(local_tmp, ignore_errors=True)


class TestSyntheticData:
    def test_generation(self, synthetic_data):
        assert len(synthetic_data) > 0
        assert synthetic_data["race_id"].nunique() == 100
        assert synthetic_data["finish_position"].min() >= 1


class TestFeatureEngineering:
    @pytest.fixture(scope="class")
    def features_csv(self, synthetic_data, tmp_dir):
        raw_path = tmp_dir / "results.csv"
        synthetic_data.to_csv(raw_path, index=False)

        df = load_results(raw_path)
        df = encode_categoricals(df)
        df = compute_horse_history_features(df)
        df = compute_jockey_features(df)
        df = create_target(df)
        df_feat = select_features(df)

        out = tmp_dir / "features.csv"
        df_feat.to_csv(out, index=False)
        return out, df_feat

    def test_has_features(self, features_csv):
        _, df = features_csv
        available = [c for c in FEATURE_COLUMNS if c in df.columns]
        assert len(available) >= 20

    def test_top3_binary(self, features_csv):
        _, df = features_csv
        assert set(df["top3"].unique()).issubset({0, 1})

    def test_no_leakage(self, features_csv):
        """初戦の馬は過去特徴量がゼロ"""
        _, df = features_csv
        first_race_id = df.sort_values("race_date")["race_id"].iloc[0] if "race_date" in df.columns else df["race_id"].iloc[0]
        first_race = df[df["race_id"] == first_race_id]
        if "avg_finish_last5" in first_race.columns:
            assert (first_race["avg_finish_last5"] == 0).all()


class TestModelTraining:
    @pytest.fixture(scope="class")
    def trained_models(self, tmp_dir):
        data = generate_synthetic_results(n_races=100)
        raw_path = tmp_dir / "results_train.csv"
        data.to_csv(raw_path, index=False)

        df = load_results(raw_path)
        df = encode_categoricals(df)
        df = compute_horse_history_features(df)
        df = compute_jockey_features(df)
        df = create_target(df)
        df_feat = select_features(df)

        feat_path = tmp_dir / "features_train.csv"
        df_feat.to_csv(feat_path, index=False)

        df2 = pd.read_csv(feat_path, dtype={"race_id": str, "horse_id": str})
        df2["race_date"] = pd.to_datetime(df2["race_date"])
        dates = sorted(df2["race_date"].unique())
        val_start = str(dates[int(len(dates) * 0.75)].date())
        train_df, val_df = time_based_split(df2, val_start)

        model_dir = tmp_dir / "models"
        available = [c for c in FEATURE_COLUMNS if c in df2.columns]

        binary_model, binary_metrics = train_binary_model(train_df, val_df)
        save_model(binary_model, binary_metrics, available, "binary", model_dir)

        rank_model, rank_metrics = train_ranking_model(train_df, val_df)
        save_model(rank_model, rank_metrics, available, "ranking", model_dir)

        return model_dir, feat_path, val_start, binary_metrics, rank_metrics

    def test_binary_auc_above_chance(self, trained_models):
        _, _, _, binary_metrics, _ = trained_models
        assert binary_metrics["auc"] > 0.5

    def test_ranking_hits(self, trained_models):
        _, _, _, _, rank_metrics = trained_models
        assert rank_metrics["top3_hit_rate"] > 0

    def test_model_files_exist(self, trained_models):
        model_dir, _, _, _, _ = trained_models
        assert (model_dir / "binary_model.txt").exists()
        assert (model_dir / "ranking_model.txt").exists()


class TestRuleBasedScoring:
    def test_scores_computed(self):
        data = {
            "race": {
                "date": "2025-03-01",
                "venue": "東京",
                "surface": "芝",
                "distance": 1600,
                "grade": "1勝",
            },
            "horses": [
                {
                    "num": 1, "name": "テスト馬1",
                    "jockey": "騎手A",
                    "load_weight": 55.0,
                    "past_races": [
                        {"date": "2025/02/15", "venue": "東京", "surface": "芝",
                         "distance": "1600", "finish": 1, "margin": ""},
                    ],
                    "jockey_stats": {"win_rate": 0.15, "place_rate": 0.35, "races": 100},
                },
                {
                    "num": 2, "name": "テスト馬2",
                    "jockey": "騎手B",
                    "load_weight": 56.0,
                    "past_races": [],
                    "jockey_stats": {},
                },
            ],
        }
        result = score_rule_based(data)
        for horse in result["horses"]:
            assert "score" in horse
            assert "score_breakdown" in horse
            assert horse["score"] >= 0

    def test_winner_scores_higher(self):
        data = {
            "race": {"date": "2025-03-01", "venue": "東京", "surface": "芝", "distance": 1600, "grade": "1勝"},
            "horses": [
                {
                    "num": 1, "name": "強い馬",
                    "jockey": "騎手A", "load_weight": 55.0,
                    "past_races": [
                        {"date": "2025/02/15", "venue": "東京", "surface": "芝", "distance": "1600", "finish": 1, "margin": ""},
                        {"date": "2025/01/20", "venue": "東京", "surface": "芝", "distance": "1600", "finish": 1, "margin": ""},
                    ],
                    "jockey_stats": {"win_rate": 0.20, "place_rate": 0.45, "races": 200},
                },
                {
                    "num": 2, "name": "弱い馬",
                    "jockey": "騎手B", "load_weight": 55.0,
                    "past_races": [
                        {"date": "2025/02/15", "venue": "中山", "surface": "ダ", "distance": "1800", "finish": 10, "margin": "5.0"},
                        {"date": "2025/01/20", "venue": "中山", "surface": "ダ", "distance": "1800", "finish": 12, "margin": "8.0"},
                    ],
                    "jockey_stats": {"win_rate": 0.02, "place_rate": 0.10, "races": 50},
                },
            ],
        }
        result = score_rule_based(data)
        assert result["horses"][0]["score"] > result["horses"][1]["score"]


class TestEVCalculation:
    def test_ev_computed(self):
        data = {
            "race": {"venue": "東京", "race_number": 5, "name": "テスト"},
            "parameters": {"temperature": 10, "budget": 1500, "top_n": 4},
            "combo_odds": {"quinella": [], "wide": [], "trio": []},
            "horses": [
                {"num": 1, "name": "馬A", "score": 80, "odds_win": 3.0, "odds_place": 1.5},
                {"num": 2, "name": "馬B", "score": 60, "odds_win": 5.0, "odds_place": 2.0},
                {"num": 3, "name": "馬C", "score": 40, "odds_win": 10.0, "odds_place": 3.5},
                {"num": 4, "name": "馬D", "score": 30, "odds_win": 20.0, "odds_place": 5.0},
            ],
        }
        result = calculate_ev(data)
        assert "win" in result
        assert "place" in result
        assert len(result["win"]) == 4
        assert all(bet["prob"] > 0 for bet in result["win"])
