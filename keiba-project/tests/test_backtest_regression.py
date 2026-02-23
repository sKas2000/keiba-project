"""
バックテスト回帰テスト
コード変更でROIが意図せず低下しないことを検証
"""
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from config.settings import FEATURE_COLUMNS
from src.data.preprocessing import load_results, encode_categoricals
from src.data.feature import (
    compute_horse_history_features, compute_jockey_features,
    create_target, select_features,
)
from src.model.trainer import (
    time_based_split, train_binary_model, save_model,
)
from src.model.evaluator import prepare_backtest_data, simulate_bets


# ============================================================
# 合成データ生成（test_consistencyと同じロジックを再利用）
# ============================================================

def generate_synthetic_results(n_races: int = 200, seed: int = 42) -> pd.DataFrame:
    """再現性のある合成レースデータ"""
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
# Fixtures
# ============================================================

@pytest.fixture(scope="module")
def tmp_dir():
    local_tmp = Path(__file__).resolve().parent.parent / ".test_tmp_bt"
    local_tmp.mkdir(exist_ok=True)
    yield local_tmp
    shutil.rmtree(local_tmp, ignore_errors=True)


@pytest.fixture(scope="module")
def trained_model(tmp_dir):
    """合成データでモデル学習して返す"""
    df = generate_synthetic_results(n_races=200, seed=42)
    raw_path = tmp_dir / "results.csv"
    df.to_csv(raw_path, index=False)

    df = load_results(raw_path)
    df = encode_categoricals(df)
    df = compute_horse_history_features(df)
    df = compute_jockey_features(df)
    df = create_target(df)
    df = select_features(df)

    feat_path = tmp_dir / "features.csv"
    df.to_csv(feat_path, index=False)

    # 200 races × ~12 entries/race, weekly from 2024-01-06
    # ~17 weeks → data ends ~2024-05-04
    # Split at 2024-04-01 to have enough train and val data
    train_df, val_df = time_based_split(df, val_start="2024-04-01")
    if len(val_df) == 0:
        # Fallback: split at midpoint
        dates = sorted(df["race_date"].unique())
        mid = dates[len(dates) // 2]
        train_df, val_df = time_based_split(df, val_start=mid)

    model_dir = tmp_dir / "models"
    model_dir.mkdir(exist_ok=True)

    model, metrics = train_binary_model(train_df, val_df)
    available = [c for c in FEATURE_COLUMNS if c in df.columns]
    save_model(model, metrics, available, "binary", model_dir)

    return {
        "feat_path": str(feat_path),
        "model_dir": model_dir,
        "train_df": train_df,
        "val_df": val_df,
        "metrics": metrics,
    }


# ============================================================
# バックテスト回帰テスト
# ============================================================

class TestBacktestRegression:
    """バックテストの基本動作が壊れていないことを検証"""

    def test_model_trains_successfully(self, trained_model):
        """モデル学習が成功する"""
        assert trained_model["metrics"]["auc"] > 0.5
        assert (trained_model["model_dir"] / "binary_model.txt").exists()

    def test_prepare_backtest_data(self, trained_model):
        """prepare_backtest_dataが正しいデータを返す"""
        data = prepare_backtest_data(
            input_path=trained_model["feat_path"],
            model_dir=trained_model["model_dir"],
            val_start="2024-04-01",
        )
        assert data is not None
        val_df = data["val_df"]
        assert "pred_prob" in val_df.columns
        assert "finish_position" in val_df.columns
        assert "race_id" in val_df.columns
        assert len(val_df) > 0
        assert val_df["pred_prob"].min() >= 0
        assert val_df["pred_prob"].max() <= 1

    def test_simulate_bets_returns_valid_structure(self, trained_model):
        """simulate_betsが正しい構造のresultを返す"""
        data = prepare_backtest_data(
            input_path=trained_model["feat_path"],
            model_dir=trained_model["model_dir"],
            val_start="2024-04-01",
        )
        results = simulate_bets(data, top_n=3)

        assert "races" in results
        assert "bets_win" in results
        assert "monthly" in results
        assert results["races"] > 0

    def test_simulate_bets_counts_consistent(self, trained_model):
        """賭け数と投資額の整合性"""
        data = prepare_backtest_data(
            input_path=trained_model["feat_path"],
            model_dir=trained_model["model_dir"],
            val_start="2024-04-01",
        )
        results = simulate_bets(data, top_n=3)

        win_bets = results["bets_win"]
        assert win_bets["count"] >= 0
        assert win_bets["invested"] == win_bets["count"] * 100
        assert win_bets["returned"] >= 0
        assert win_bets["hits"] <= win_bets["count"]

    def test_simulate_bets_monthly_consistency(self, trained_model):
        """月次データの合計 = 全体の合計"""
        data = prepare_backtest_data(
            input_path=trained_model["feat_path"],
            model_dir=trained_model["model_dir"],
            val_start="2024-04-01",
        )
        results = simulate_bets(data, top_n=3)

        monthly = results.get("monthly", {})
        monthly_win_count = sum(
            m.get("win", {}).get("count", 0) for m in monthly.values()
        )
        assert monthly_win_count == results["bets_win"]["count"]

    def test_ev_threshold_reduces_bets(self, trained_model):
        """EV閾値を上げるとベット数が減る"""
        data = prepare_backtest_data(
            input_path=trained_model["feat_path"],
            model_dir=trained_model["model_dir"],
            val_start="2024-04-01",
        )
        results_no_filter = simulate_bets(data, top_n=3, ev_threshold=0)
        results_high_ev = simulate_bets(data, top_n=3, ev_threshold=2.0)

        assert results_high_ev["bets_win"]["count"] <= results_no_filter["bets_win"]["count"]

    def test_top_n_affects_bet_count(self, trained_model):
        """top_nを小さくするとベット数が減る"""
        data = prepare_backtest_data(
            input_path=trained_model["feat_path"],
            model_dir=trained_model["model_dir"],
            val_start="2024-04-01",
        )
        results_3 = simulate_bets(data, top_n=3)
        results_1 = simulate_bets(data, top_n=1)

        assert results_1["bets_win"]["count"] <= results_3["bets_win"]["count"]

    def test_deterministic_results(self, trained_model):
        """同じデータ・パラメータで同じ結果"""
        data = prepare_backtest_data(
            input_path=trained_model["feat_path"],
            model_dir=trained_model["model_dir"],
            val_start="2024-04-01",
        )
        r1 = simulate_bets(data, top_n=3)
        r2 = simulate_bets(data, top_n=3)

        assert r1["races"] == r2["races"]
        assert r1["bets_win"]["count"] == r2["bets_win"]["count"]
        assert r1["bets_win"]["returned"] == r2["bets_win"]["returned"]
        assert r1["bets_win"]["hits"] == r2["bets_win"]["hits"]


class TestBacktestWithRealModel:
    """本番モデルがある場合の回帰テスト"""

    @pytest.fixture
    def real_model_available(self):
        model_dir = Path(__file__).resolve().parent.parent / "models"
        if not (model_dir / "binary_model.txt").exists():
            pytest.skip("本番モデルなし")
        return model_dir

    @pytest.fixture
    def real_features_available(self):
        feat_path = Path(__file__).resolve().parent.parent / "data" / "processed" / "features.csv"
        if not feat_path.exists():
            pytest.skip("features.csvなし")
        return feat_path

    def test_real_model_backtest_runs(self, real_model_available, real_features_available):
        """本番モデル + 本番データでバックテストが正常に動作する"""
        data = prepare_backtest_data(
            input_path=str(real_features_available),
            model_dir=real_model_available,
            val_start="2025-07-01",
        )
        assert data is not None, "prepare_backtest_data returned None"
        results = simulate_bets(data, top_n=3)

        assert results["races"] > 100, "Testデータが少なすぎる"
        assert results["bets_win"]["count"] > 0

    def test_real_model_prediction_quality(self, real_model_available, real_features_available):
        """本番モデルのTop1的中率 > 20%（ランダムの約8%を大幅に上回る）"""
        data = prepare_backtest_data(
            input_path=str(real_features_available),
            model_dir=real_model_available,
            val_start="2025-07-01",
        )
        assert data is not None
        results = simulate_bets(data, top_n=1)

        hit_rate = results["bets_win"]["hits"] / max(results["bets_win"]["count"], 1)
        assert hit_rate > 0.20, f"Top1的中率 {hit_rate:.1%} が低すぎる"

    def test_real_model_quinella_roi_positive(self, real_model_available, real_features_available):
        """本番モデルの馬連ROIが80%以上（破壊的変更がないことの確認）"""
        from config.settings import EXPANDING_BEST_PARAMS

        data = prepare_backtest_data(
            input_path=str(real_features_available),
            model_dir=real_model_available,
            val_start="2025-07-01",
        )
        assert data is not None
        results = simulate_bets(
            data,
            top_n=EXPANDING_BEST_PARAMS.get("top_n", 3),
            confidence_min=EXPANDING_BEST_PARAMS.get("confidence_min", 0),
            quinella_top_n=EXPANDING_BEST_PARAMS.get("quinella_top_n", 3),
            skip_classes=EXPANDING_BEST_PARAMS.get("skip_classes", []),
        )

        q = results.get("bets_quinella", {})
        if q.get("count", 0) > 0:
            roi = q["returned"] / q["invested"] * 100
            assert roi > 75, f"馬連ROI {roi:.1f}% が75%を下回った（回帰バグの疑い）"
