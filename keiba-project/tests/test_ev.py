"""
EV計算（ev.py）のユニットテスト
確率計算の数学的正当性を検証
"""
import math
from itertools import combinations, permutations

import pytest

from src.model.ev import (
    softmax, calc_place_probs, calc_quinella_prob,
    calc_wide_prob, calc_trio_prob, calculate_ev, get_ev_rank,
)


# ==========================================================
# softmax テスト
# ==========================================================

class TestSoftmax:
    def test_sum_to_one(self):
        """確率の合計が1になる"""
        scores = [10, 20, 30, 40]
        probs = softmax(scores, temperature=10)
        assert abs(sum(probs) - 1.0) < 1e-10

    def test_order_preserved(self):
        """スコア順が確率順に反映される"""
        scores = [10, 20, 30, 40]
        probs = softmax(scores, temperature=10)
        for i in range(len(probs) - 1):
            assert probs[i] < probs[i + 1]

    def test_equal_scores_uniform(self):
        """全スコア同一 → 均等確率"""
        scores = [50, 50, 50, 50]
        probs = softmax(scores, temperature=10)
        for p in probs:
            assert abs(p - 0.25) < 1e-10

    def test_high_temperature_uniform(self):
        """高温度 → 均等に近づく"""
        scores = [10, 20, 30, 40]
        probs = softmax(scores, temperature=10000)
        for p in probs:
            assert abs(p - 0.25) < 0.01

    def test_low_temperature_peaked(self):
        """低温度 → 最高スコアに集中"""
        scores = [10, 20, 30, 40]
        probs = softmax(scores, temperature=0.01)
        assert probs[-1] > 0.99

    def test_single_element(self):
        """1要素 → 確率1.0"""
        probs = softmax([50], temperature=10)
        assert abs(probs[0] - 1.0) < 1e-10

    def test_two_elements(self):
        """2要素の対称性"""
        probs = softmax([10, 20], temperature=10)
        assert abs(sum(probs) - 1.0) < 1e-10
        assert probs[1] > probs[0]

    def test_negative_scores(self):
        """負のスコアでも正常動作"""
        probs = softmax([-10, -5, 0, 5], temperature=5)
        assert abs(sum(probs) - 1.0) < 1e-10
        for p in probs:
            assert p > 0

    def test_large_scores_no_overflow(self):
        """大きなスコアでオーバーフローしない"""
        probs = softmax([1000, 2000, 3000], temperature=10)
        assert abs(sum(probs) - 1.0) < 1e-10
        assert all(0 <= p <= 1 for p in probs)


# ==========================================================
# 複勝確率 テスト
# ==========================================================

class TestPlaceProbs:
    def test_valid_range(self):
        """全確率が0〜1の範囲内"""
        win_probs = softmax([10, 20, 30, 40, 50], temperature=10)
        place_probs = calc_place_probs(win_probs)
        for p in place_probs:
            assert 0 <= p <= 1.0

    def test_top_horse_highest_place_prob(self):
        """勝率最高馬が複勝確率も最高"""
        win_probs = softmax([10, 20, 30, 40, 50], temperature=10)
        place_probs = calc_place_probs(win_probs)
        best_win = max(range(len(win_probs)), key=lambda i: win_probs[i])
        best_place = max(range(len(place_probs)), key=lambda i: place_probs[i])
        assert best_win == best_place

    def test_equal_probs_equal_place(self):
        """均等勝率 → 均等複勝"""
        n = 5
        win_probs = [1.0 / n] * n
        place_probs = calc_place_probs(win_probs)
        for i in range(n - 1):
            assert abs(place_probs[i] - place_probs[i + 1]) < 1e-6

    def test_place_gte_win(self):
        """複勝確率 >= 勝率（3着以内は1着を含む）"""
        win_probs = softmax([10, 20, 30, 40, 50], temperature=10)
        place_probs = calc_place_probs(win_probs)
        for w, p in zip(win_probs, place_probs):
            assert p >= w - 1e-10

    def test_three_horses(self):
        """3頭の場合、全馬の複勝確率が1.0"""
        win_probs = softmax([10, 20, 30], temperature=10)
        place_probs = calc_place_probs(win_probs)
        for p in place_probs:
            assert abs(p - 1.0) < 0.05  # 近似計算のため少し許容


# ==========================================================
# 馬連確率 テスト
# ==========================================================

class TestQuinellaProb:
    def test_symmetric(self):
        """i-j と j-i は同じ確率"""
        win_probs = softmax([10, 20, 30, 40, 50], temperature=10)
        p1 = calc_quinella_prob(win_probs, 0, 1)
        p2 = calc_quinella_prob(win_probs, 1, 0)
        assert abs(p1 - p2) < 1e-10

    def test_valid_range(self):
        """全組合せが0〜1の範囲"""
        win_probs = softmax([10, 20, 30, 40, 50], temperature=10)
        for i in range(5):
            for j in range(5):
                if i == j:
                    continue
                p = calc_quinella_prob(win_probs, i, j)
                assert 0 <= p <= 1.0, f"quinella({i},{j})={p}"

    def test_top_pair_highest(self):
        """上位2頭ペアが最高確率"""
        win_probs = softmax([10, 20, 30, 40, 50], temperature=10)
        top_prob = calc_quinella_prob(win_probs, 3, 4)
        low_prob = calc_quinella_prob(win_probs, 0, 1)
        assert top_prob > low_prob

    def test_sum_approximately_one(self):
        """全馬連の確率合計 ≈ 1.0"""
        win_probs = softmax([10, 20, 30, 40, 50], temperature=10)
        total = sum(
            calc_quinella_prob(win_probs, i, j)
            for i, j in combinations(range(5), 2)
        )
        assert abs(total - 1.0) < 0.02

    def test_equal_probs(self):
        """均等勝率 → 全組合せ均等"""
        n = 4
        win_probs = [1.0 / n] * n
        probs = [
            calc_quinella_prob(win_probs, i, j)
            for i, j in combinations(range(n), 2)
        ]
        expected = 2.0 / (n * (n - 1))
        for p in probs:
            assert abs(p - expected) < 0.01


# ==========================================================
# ワイド確率 テスト
# ==========================================================

class TestWideProb:
    def test_symmetric(self):
        """i-j と j-i は同じ確率"""
        win_probs = softmax([10, 20, 30, 40, 50], temperature=10)
        p1 = calc_wide_prob(win_probs, 0, 1)
        p2 = calc_wide_prob(win_probs, 1, 0)
        assert abs(p1 - p2) < 1e-10

    def test_valid_range(self):
        """全組合せが0〜1の範囲"""
        win_probs = softmax([10, 20, 30, 40, 50], temperature=10)
        for i in range(5):
            for j in range(5):
                if i == j:
                    continue
                p = calc_wide_prob(win_probs, i, j)
                assert 0 <= p <= 1.0, f"wide({i},{j})={p}"

    def test_wide_gte_quinella(self):
        """ワイド確率 >= 馬連確率（3着以内 ⊇ 1-2着）"""
        win_probs = softmax([10, 20, 30, 40, 50], temperature=10)
        for i, j in combinations(range(5), 2):
            q = calc_quinella_prob(win_probs, i, j)
            w = calc_wide_prob(win_probs, i, j)
            assert w >= q - 1e-6, (
                f"wide({i},{j})={w:.6f} < quinella({i},{j})={q:.6f}"
            )

    def test_sum_approximately_three(self):
        """全ワイドの確率合計 ≈ 3.0（3着以内から2頭選ぶ = C(3,2)=3通り）"""
        win_probs = softmax([10, 20, 30, 40, 50], temperature=10)
        total = sum(
            calc_wide_prob(win_probs, i, j)
            for i, j in combinations(range(5), 2)
        )
        assert abs(total - 3.0) < 0.15


# ==========================================================
# 3連複確率 テスト
# ==========================================================

class TestTrioProb:
    def test_valid_range(self):
        """確率が0〜1の範囲"""
        win_probs = softmax([10, 20, 30, 40, 50], temperature=10)
        p = calc_trio_prob(win_probs, 0, 1, 2)
        assert 0 <= p <= 1.0

    def test_order_independent(self):
        """3連複は順不同"""
        win_probs = softmax([10, 20, 30, 40, 50], temperature=10)
        probs = set()
        for perm in permutations([2, 3, 4]):
            p = calc_trio_prob(win_probs, *perm)
            probs.add(round(p, 12))
        assert len(probs) == 1, f"Trio should be symmetric: {probs}"

    def test_sum_approximately_one(self):
        """全3連複の確率合計 ≈ 1.0"""
        win_probs = softmax([10, 20, 30, 40, 50], temperature=10)
        total = sum(
            calc_trio_prob(win_probs, i, j, k)
            for i, j, k in combinations(range(5), 3)
        )
        assert abs(total - 1.0) < 0.02

    def test_top_trio_highest(self):
        """上位3頭の3連複が最高確率"""
        win_probs = softmax([10, 20, 30, 40, 50], temperature=10)
        top = calc_trio_prob(win_probs, 2, 3, 4)
        low = calc_trio_prob(win_probs, 0, 1, 2)
        assert top > low


# ==========================================================
# EVランク テスト
# ==========================================================

class TestEVRank:
    @pytest.mark.parametrize("ev,expected", [
        (2.0, "S"), (1.5, "S"),
        (1.3, "A"), (1.2, "A"),
        (1.1, "B"), (1.0, "B"),
        (0.9, "C"), (0.5, "C"), (0.0, "C"),
    ])
    def test_rank_thresholds(self, ev, expected):
        assert get_ev_rank(ev) == expected


# ==========================================================
# calculate_ev 統合テスト
# ==========================================================

class TestCalculateEV:
    @pytest.fixture
    def sample_data(self):
        return {
            "race": {
                "venue": "東京", "race_number": 1,
                "name": "テスト", "grade": "未勝利",
            },
            "parameters": {"temperature": 10, "budget": 1500, "top_n": 3},
            "horses": [
                {"num": 1, "name": "A", "score": 50,
                 "odds_win": 3.0, "odds_place": 1.5},
                {"num": 2, "name": "B", "score": 40,
                 "odds_win": 5.0, "odds_place": 2.0},
                {"num": 3, "name": "C", "score": 30,
                 "odds_win": 10.0, "odds_place": 3.0},
                {"num": 4, "name": "D", "score": 20,
                 "odds_win": 20.0, "odds_place": 5.0},
                {"num": 5, "name": "E", "score": 10,
                 "odds_win": 50.0, "odds_place": 10.0},
            ],
            "combo_odds": {
                "quinella": [
                    {"combo": [1, 2], "odds": 5.0},
                    {"combo": [1, 3], "odds": 10.0},
                    {"combo": [2, 3], "odds": 15.0},
                ],
                "wide": [
                    {"combo": [1, 2], "odds": [2.0, 3.0]},
                    {"combo": [1, 3], "odds": [3.0, 5.0]},
                    {"combo": [2, 3], "odds": [4.0, 7.0]},
                ],
                "trio": [
                    {"combo": [1, 2, 3], "odds": 20.0},
                ],
            },
        }

    def test_win_probs_sum_to_one(self, sample_data):
        result = calculate_ev(sample_data)
        total = sum(b["prob"] for b in result["win"])
        assert abs(total - 1.0) < 1e-6

    def test_all_bet_types_present(self, sample_data):
        result = calculate_ev(sample_data)
        for key in ["win", "place", "quinella", "wide", "trio"]:
            assert key in result

    def test_ev_non_negative(self, sample_data):
        result = calculate_ev(sample_data)
        for bet in result["win"]:
            assert bet["ev"] >= 0
        for bet in result["quinella"]:
            assert bet["ev"] >= 0

    def test_confidence_valid(self, sample_data):
        result = calculate_ev(sample_data)
        assert 0 < result["confidence"] <= 1.0
        assert result["confidence_gap"] >= 0

    def test_no_combo_odds(self):
        """組合せオッズなしでも動作する"""
        data = {
            "race": {"venue": "東京", "race_number": 1,
                     "name": "テスト", "grade": "未勝利"},
            "parameters": {"temperature": 10},
            "horses": [
                {"num": 1, "name": "A", "score": 50,
                 "odds_win": 3.0, "odds_place": 1.5},
                {"num": 2, "name": "B", "score": 40,
                 "odds_win": 5.0, "odds_place": 2.0},
            ],
            "combo_odds": {},
        }
        result = calculate_ev(data)
        assert len(result["win"]) == 2
        assert result["quinella"] == []

    def test_ml_mode_with_win_probs(self):
        """ML(win_model)モードの確率計算"""
        data = {
            "race": {"venue": "東京", "race_number": 1,
                     "name": "テスト", "grade": "未勝利"},
            "parameters": {"temperature": 1.0},
            "horses": [
                {"num": 1, "name": "A", "score": 50, "ml_top3_prob": 0.5,
                 "ml_win_prob": 0.3, "odds_win": 3.0, "odds_place": 1.5},
                {"num": 2, "name": "B", "score": 40, "ml_top3_prob": 0.3,
                 "ml_win_prob": 0.2, "odds_win": 5.0, "odds_place": 2.0},
                {"num": 3, "name": "C", "score": 30, "ml_top3_prob": 0.2,
                 "ml_win_prob": 0.1, "odds_win": 10.0, "odds_place": 3.0},
            ],
            "combo_odds": {},
        }
        result = calculate_ev(data)
        # win_probはml_win_probの正規化
        total_wp = sum(b["prob"] for b in result["win"])
        assert abs(total_wp - 1.0) < 1e-6

    def test_confidence_min_triggers_low_confidence(self):
        """confidence_min戦略で確信度不足を検知"""
        data = {
            "race": {"venue": "東京", "race_number": 1,
                     "name": "テスト", "grade": "未勝利"},
            "parameters": {"temperature": 10},
            "horses": [
                {"num": i, "name": f"H{i}", "score": 50,
                 "odds_win": 5.0, "odds_place": 2.0}
                for i in range(1, 6)
            ],
            "combo_odds": {},
        }
        strategy = {"confidence_min": 0.5}
        result = calculate_ev(data, strategy=strategy)
        assert result["low_confidence"] is True

    def test_two_horse_race(self):
        """2頭立てでもエラーにならない"""
        data = {
            "race": {"venue": "東京", "race_number": 1,
                     "name": "テスト", "grade": "未勝利"},
            "parameters": {"temperature": 10},
            "horses": [
                {"num": 1, "name": "A", "score": 60,
                 "odds_win": 1.5, "odds_place": 1.1},
                {"num": 2, "name": "B", "score": 40,
                 "odds_win": 3.0, "odds_place": 1.5},
            ],
            "combo_odds": {
                "quinella": [{"combo": [1, 2], "odds": 2.0}],
                "wide": [],
                "trio": [],
            },
        }
        result = calculate_ev(data)
        assert len(result["win"]) == 2
        assert len(result["quinella"]) == 1

    def test_extreme_odds(self):
        """極端なオッズでもエラーにならない"""
        data = {
            "race": {"venue": "東京", "race_number": 1,
                     "name": "テスト", "grade": "未勝利"},
            "parameters": {"temperature": 10},
            "horses": [
                {"num": 1, "name": "A", "score": 90,
                 "odds_win": 1.1, "odds_place": 1.0},
                {"num": 2, "name": "B", "score": 10,
                 "odds_win": 500.0, "odds_place": 100.0},
                {"num": 3, "name": "C", "score": 5,
                 "odds_win": 999.9, "odds_place": 200.0},
            ],
            "combo_odds": {},
        }
        result = calculate_ev(data)
        assert all(0 <= b["prob"] <= 1 for b in result["win"])
        assert abs(sum(b["prob"] for b in result["win"]) - 1.0) < 1e-6
