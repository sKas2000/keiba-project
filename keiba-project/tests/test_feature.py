"""src/data/feature.py のユニットテスト"""
import pytest
import numpy as np
import pandas as pd

from config.settings import FEATURE_COLUMNS
from src.data.feature import (
    extract_features_from_enriched,
    create_target,
    _dist_cat,
)


class TestDistCat:
    def test_sprint(self):
        assert _dist_cat(1200) == 0

    def test_mile(self):
        assert _dist_cat(1600) == 1

    def test_middle(self):
        assert _dist_cat(2000) == 2

    def test_long(self):
        assert _dist_cat(2400) == 3

    def test_boundary_1400(self):
        assert _dist_cat(1400) == 0

    def test_boundary_1800(self):
        assert _dist_cat(1800) == 1

    def test_boundary_2200(self):
        assert _dist_cat(2200) == 2


class TestCreateTarget:
    def test_top3(self):
        df = pd.DataFrame({"finish_position": [1, 2, 3, 4, 5]})
        df = create_target(df)
        assert list(df["top3"]) == [1, 1, 1, 0, 0]


class TestExtractFeaturesFromEnriched:
    @pytest.fixture
    def enriched_data(self):
        return {
            "race": {
                "date": "2025-03-01",
                "venue": "東京",
                "race_number": 5,
                "name": "テストレース",
                "grade": "1勝",
                "surface": "芝",
                "distance": 1600,
                "track_condition": "良",
            },
            "horses": [
                {
                    "num": 1,
                    "name": "テスト馬1",
                    "sex_age": "牡3",
                    "load_weight": 55.0,
                    "frame_number": 1,
                    "odds_win": 5.0,
                    "odds_place": 2.0,
                    "past_races": [
                        {
                            "date": "2025/02/15",
                            "venue": "東京",
                            "surface": "芝",
                            "distance": "1600",
                            "finish": 2,
                            "last3f": "33.5",
                        },
                        {
                            "date": "2025/01/20",
                            "venue": "中山",
                            "surface": "芝",
                            "distance": "1800",
                            "finish": 3,
                            "last3f": "34.0",
                        },
                    ],
                    "jockey_stats": {
                        "win_rate": 0.15,
                        "place_rate": 0.35,
                        "races": 100,
                    },
                },
                {
                    "num": 2,
                    "name": "テスト馬2",
                    "sex_age": "牝4",
                    "load_weight": 54.0,
                    "frame_number": 1,
                    "odds_win": 10.0,
                    "odds_place": 4.0,
                    "past_races": [],
                    "jockey_stats": {},
                },
            ],
        }

    def test_returns_dataframe(self, enriched_data):
        df = extract_features_from_enriched(enriched_data)
        assert isinstance(df, pd.DataFrame)

    def test_correct_row_count(self, enriched_data):
        df = extract_features_from_enriched(enriched_data)
        assert len(df) == 2

    def test_has_feature_columns(self, enriched_data):
        df = extract_features_from_enriched(enriched_data)
        for col in ["sex_code", "age", "distance", "surface_code",
                     "prev_finish_1", "jockey_win_rate_365d"]:
            assert col in df.columns, f"{col} が見つかりません"

    def test_sex_encoding(self, enriched_data):
        df = extract_features_from_enriched(enriched_data)
        assert df.iloc[0]["sex_code"] == 0  # 牡
        assert df.iloc[1]["sex_code"] == 1  # 牝

    def test_past_features_computed(self, enriched_data):
        df = extract_features_from_enriched(enriched_data)
        # 過去走あり
        assert df.iloc[0]["prev_finish_1"] == 2
        assert df.iloc[0]["prev_finish_2"] == 3
        # 過去走なし
        assert df.iloc[1]["prev_finish_1"] == 0

    def test_jockey_stats(self, enriched_data):
        df = extract_features_from_enriched(enriched_data)
        assert df.iloc[0]["jockey_win_rate_365d"] == 0.15
        assert df.iloc[1]["jockey_win_rate_365d"] == 0.0

    def test_no_past_races(self, enriched_data):
        df = extract_features_from_enriched(enriched_data)
        horse2 = df.iloc[1]
        assert horse2["avg_finish_last5"] == 0
        assert horse2["days_since_last_race"] == 365
        assert horse2["total_races"] == 0
