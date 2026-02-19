"""src/data/preprocessing.py のユニットテスト"""
import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path

from src.data.preprocessing import load_results, encode_categoricals, validate_input_data


@pytest.fixture
def sample_csv(tmp_path):
    """テスト用の results.csv を生成"""
    rows = []
    for i in range(20):
        rows.append({
            "race_id": f"2024050101{(i//10)+1:02d}",
            "race_date": f"2024-01-{6+i//10:02d}",
            "course_id": "05",
            "course_name": "東京",
            "race_number": (i % 10) + 1,
            "race_name": "テストレース",
            "race_class": "未勝利",
            "surface": "芝",
            "distance": 1600,
            "track_condition": "良",
            "weather": "晴",
            "num_entries": 10,
            "finish_position": (i % 10) + 1,
            "frame_number": (i % 10) // 2 + 1,
            "horse_number": (i % 10) + 1,
            "horse_name": f"テスト馬{i%10}",
            "horse_id": f"2020{i%10:06d}",
            "sex": "牡" if i % 3 != 1 else "牝",
            "age": 3,
            "jockey_name": f"騎手{i%3}",
            "jockey_id": f"0{i%3:04d}",
            "trainer_name": f"調教師{i%2}",
            "weight_carried": 55.0,
            "horse_weight": 480,
            "horse_weight_change": 0,
            "finish_time_sec": 96.0 + i * 0.2,
            "margin": "",
            "passing_order": "",
            "last_3f": 34.5,
            "win_odds": 5.0 + i,
            "popularity": (i % 10) + 1,
            "prize_money": 0,
        })

    df = pd.DataFrame(rows)
    csv_path = tmp_path / "results.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


class TestLoadResults:
    def test_loads_data(self, sample_csv):
        df = load_results(sample_csv)
        assert len(df) > 0

    def test_date_parsed(self, sample_csv):
        df = load_results(sample_csv)
        assert pd.api.types.is_datetime64_any_dtype(df["race_date"])

    def test_finish_position_int(self, sample_csv):
        df = load_results(sample_csv)
        assert df["finish_position"].dtype in [np.int64, np.int32, int]

    def test_excludes_zero_finish(self, sample_csv):
        df = load_results(sample_csv)
        assert (df["finish_position"] > 0).all()


class TestEncodeCategoricals:
    def test_sex_code(self, sample_csv):
        df = load_results(sample_csv)
        df = encode_categoricals(df)
        assert "sex_code" in df.columns
        assert set(df["sex_code"].unique()).issubset({0, 1, 2})

    def test_surface_code(self, sample_csv):
        df = load_results(sample_csv)
        df = encode_categoricals(df)
        assert "surface_code" in df.columns

    def test_distance_cat(self, sample_csv):
        df = load_results(sample_csv)
        df = encode_categoricals(df)
        assert "distance_cat" in df.columns


class TestValidateInputData:
    def test_valid_data(self):
        data = {
            "race": {"venue": "東京", "distance": 1600, "surface": "芝", "date": "2025-03-01"},
            "horses": [{"num": 1, "name": "テスト馬"}],
        }
        is_valid, errors = validate_input_data(data)
        assert is_valid
        assert len([e for e in errors if e.startswith("[ERROR]")]) == 0

    def test_missing_race(self):
        data = {"horses": [{"num": 1}]}
        is_valid, errors = validate_input_data(data)
        assert not is_valid

    def test_empty_horses(self):
        data = {"race": {"venue": "東京"}, "horses": []}
        is_valid, errors = validate_input_data(data)
        assert not is_valid
