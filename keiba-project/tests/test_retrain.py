"""自動再学習（retrain）のユニットテスト"""
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# LightGBMはWindows上で日本語パスに書き込めないため、
# プロジェクトルート配下にテスト用一時ディレクトリを作成
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_TMP = PROJECT_ROOT / "tests" / "_tmp_retrain"


@pytest.fixture
def work_dir():
    """プロジェクト内の一時ディレクトリ（日本語パス回避）"""
    TEST_TMP.mkdir(parents=True, exist_ok=True)
    yield TEST_TMP
    if TEST_TMP.exists():
        shutil.rmtree(TEST_TMP, ignore_errors=True)


@pytest.fixture
def synthetic_features(work_dir):
    """テスト用の合成特徴量CSV"""
    np.random.seed(42)

    race_dates = pd.date_range("2023-01-01", periods=300, freq="3D")
    rows = []
    for i, date in enumerate(race_dates):
        n_horses = np.random.randint(8, 15)
        race_id = f"2023{i:06d}"
        for j in range(n_horses):
            fp = j + 1
            rows.append({
                "race_id": race_id,
                "race_date": date,
                "horse_name": f"Horse_{i}_{j}",
                "horse_id": f"H{i:04d}{j:02d}",
                "horse_number": j + 1,
                "frame_number": (j % 8) + 1,
                "finish_position": fp,
                "top3": 1 if fp <= 3 else 0,
                "win_odds": round(np.random.uniform(1.5, 50.0), 1),
                "popularity": j + 1,
                "sex_code": np.random.randint(0, 3),
                "age": np.random.randint(2, 8),
                "weight_carried": round(np.random.uniform(51, 58), 1),
                "horse_weight": np.random.randint(420, 560),
                "horse_weight_change": np.random.randint(-10, 10),
                "num_entries": n_horses,
                "surface_code": np.random.choice([0, 1]),
                "distance": np.random.choice([1200, 1600, 2000, 2400]),
                "race_class_code": np.random.choice([2, 3, 4, 5, 6]),
                "course_id_code": np.random.choice([5, 6, 9]),
                "prev_finish_1": np.random.randint(1, 15),
                "prev_finish_2": np.random.randint(1, 15),
                "prev_finish_3": np.random.randint(1, 15),
                "avg_finish_last5": round(np.random.uniform(3, 10), 1),
                "best_finish_last5": np.random.randint(1, 5),
                "win_rate_last5": round(np.random.uniform(0, 0.4), 2),
                "place_rate_last5": round(np.random.uniform(0, 0.8), 2),
                "avg_last3f_last5": round(np.random.uniform(33, 38), 1),
                "days_since_last_race": np.random.randint(14, 120),
                "total_races": np.random.randint(1, 30),
                "career_win_rate": round(np.random.uniform(0, 0.3), 2),
                "career_place_rate": round(np.random.uniform(0, 0.6), 2),
                "surface_win_rate": round(np.random.uniform(0, 0.3), 2),
                "surface_place_rate": round(np.random.uniform(0, 0.6), 2),
                "distance_cat_win_rate": round(np.random.uniform(0, 0.3), 2),
                "jockey_win_rate_365d": round(np.random.uniform(0, 0.25), 3),
                "jockey_place_rate_365d": round(np.random.uniform(0, 0.5), 3),
                "jockey_ride_count_365d": np.random.randint(10, 300),
                "prev_margin_1": round(np.random.uniform(-2, 5), 1),
                "prev_last3f_1": round(np.random.uniform(33, 38), 1),
                "distance_change": np.random.randint(-400, 400),
                "running_style": np.random.randint(0, 4),
                "avg_early_position_last5": round(np.random.uniform(1, 15), 1),
                "track_cond_place_rate": round(np.random.uniform(0, 0.6), 2),
                "trainer_win_rate_365d": round(np.random.uniform(0, 0.2), 3),
                "trainer_place_rate_365d": round(np.random.uniform(0, 0.4), 3),
                "race_month": np.random.randint(1, 13),
                "class_change": np.random.choice([-1, 0, 1]),
                "weight_carried_change": round(np.random.uniform(-2, 2), 1),
                "prev_interval_2": np.random.randint(14, 120),
                "race_n_front": np.random.randint(1, 5),
                "race_n_mid": np.random.randint(2, 8),
                "race_n_back": np.random.randint(1, 5),
                "pace_advantage": round(np.random.uniform(-1, 1), 2),
                "post_position_bias": round(np.random.uniform(-0.1, 0.1), 3),
                "z_surface_place_rate": round(np.random.normal(0, 1), 2),
                "z_jockey_place_rate_365d": round(np.random.normal(0, 1), 2),
                "z_avg_finish_last5": round(np.random.normal(0, 1), 2),
                "z_career_place_rate": round(np.random.normal(0, 1), 2),
                "z_trainer_place_rate_365d": round(np.random.normal(0, 1), 2),
                "same_jockey_rides": np.random.randint(0, 10),
                "same_jockey_win_rate": round(np.random.uniform(0, 0.5), 2),
                "course_dist_win_rate": round(np.random.uniform(0, 0.3), 2),
                "course_dist_place_rate": round(np.random.uniform(0, 0.6), 2),
            })

    df = pd.DataFrame(rows)
    csv_path = work_dir / "features.csv"
    df.to_csv(csv_path, index=False)
    return csv_path, work_dir


class TestRetrain:
    """retrain()のテスト"""

    def test_retrain_creates_versioned_dir(self, synthetic_features, monkeypatch):
        """バージョンディレクトリが作成される"""
        csv_path, work_dir = synthetic_features
        model_dir = work_dir / "models"
        model_dir.mkdir(exist_ok=True)
        monkeypatch.setattr("src.model.trainer.MODEL_DIR", model_dir)

        from src.model.trainer import retrain
        result = retrain(input_path=str(csv_path), calibration_pct=0.15)

        assert result is not None
        assert result.exists()
        assert result.name.startswith("v")
        assert (result / "binary_model.txt").exists()
        assert (result / "win_model.txt").exists()
        assert (result / "binary_isotonic.pkl").exists()
        assert (result / "win_isotonic.pkl").exists()
        assert (result / "version_meta.json").exists()

    def test_retrain_creates_active_version(self, synthetic_features, monkeypatch):
        """active_version.txtが作成される"""
        csv_path, work_dir = synthetic_features
        model_dir = work_dir / "models"
        model_dir.mkdir(exist_ok=True)
        monkeypatch.setattr("src.model.trainer.MODEL_DIR", model_dir)

        from src.model.trainer import retrain
        result = retrain(input_path=str(csv_path), calibration_pct=0.15)

        active_path = model_dir / "active_version.txt"
        assert active_path.exists()
        version = active_path.read_text(encoding="utf-8").strip()
        assert version == result.name

    def test_retrain_version_meta_correct(self, synthetic_features, monkeypatch):
        """バージョンメタデータが正しい"""
        csv_path, work_dir = synthetic_features
        model_dir = work_dir / "models"
        model_dir.mkdir(exist_ok=True)
        monkeypatch.setattr("src.model.trainer.MODEL_DIR", model_dir)

        from src.model.trainer import retrain
        result = retrain(input_path=str(csv_path), calibration_pct=0.15)

        meta_path = result / "version_meta.json"
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        assert meta["method"] == "expanding_window_retrain"
        assert meta["calibration_pct"] == 0.15
        assert meta["binary_auc"] > 0
        assert meta["win_auc"] > 0
        assert meta["feature_count"] > 0
        assert meta["train_rows"] > 0
        assert meta["cal_rows"] > 0

    def test_get_active_model_dir_with_version(self, synthetic_features, monkeypatch):
        """get_active_model_dir()がバージョンを返す"""
        csv_path, work_dir = synthetic_features
        model_dir = work_dir / "models"
        model_dir.mkdir(exist_ok=True)
        monkeypatch.setattr("src.model.trainer.MODEL_DIR", model_dir)

        from src.model.trainer import retrain, get_active_model_dir
        result = retrain(input_path=str(csv_path), calibration_pct=0.15)

        active = get_active_model_dir(model_dir)
        assert active == result

    def test_get_active_model_dir_fallback(self, work_dir):
        """active_version.txtがない場合はMODEL_DIRを返す"""
        from src.model.trainer import get_active_model_dir
        model_dir = work_dir / "models_fallback"
        model_dir.mkdir(exist_ok=True)
        assert get_active_model_dir(model_dir) == model_dir

    def test_retrain_cleanup_old_versions(self, synthetic_features, monkeypatch):
        """古いバージョンが自動削除される"""
        csv_path, work_dir = synthetic_features
        model_dir = work_dir / "models"
        model_dir.mkdir(exist_ok=True)
        monkeypatch.setattr("src.model.trainer.MODEL_DIR", model_dir)

        # 古いバージョンを手動作成
        for ver in ["v20240101", "v20240201", "v20240301"]:
            d = model_dir / ver
            d.mkdir(exist_ok=True)
            (d / "binary_model.txt").write_text("dummy")

        from src.model.trainer import retrain
        retrain(input_path=str(csv_path), calibration_pct=0.15, keep_versions=2)

        # keep_versions=2 なので最新2つだけ残る
        from src.model.trainer import _list_model_versions
        versions = _list_model_versions(model_dir)
        assert len(versions) <= 2
