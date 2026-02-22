"""
モニターパッケージのユニットテスト
ネットワーク不要な関数・ロジックのみテスト
"""
import pytest

from src.monitor.constants import DEFAULT_POST_TIMES, RESULT_CHECK_DELAY
from src.monitor.helpers import (
    construct_race_id, collect_recommendations, extract_venue,
)
from src.monitor.results import _calculate_bet_results


# ============================================================
# Constants
# ============================================================

class TestConstants:
    def test_default_post_times_coverage(self):
        """1R〜12Rの時刻が全て定義されている"""
        for i in range(1, 13):
            assert i in DEFAULT_POST_TIMES
            h, m = map(int, DEFAULT_POST_TIMES[i].split(":"))
            assert 9 <= h <= 16
            assert 0 <= m <= 59

    def test_post_times_ascending(self):
        """発走時刻が昇順"""
        times = [DEFAULT_POST_TIMES[i] for i in range(1, 13)]
        assert times == sorted(times)

    def test_result_check_delay(self):
        assert RESULT_CHECK_DELAY > 0


# ============================================================
# helpers.construct_race_id
# ============================================================

class TestConstructRaceId:
    def test_basic(self):
        """正常ケース: meeting_textからrace_id構築"""
        race = {
            "meeting_text": "1回東京8日",
            "race_info": {"venue": "東京", "race_number": 11},
        }
        rid = construct_race_id(race, [])
        # 2026 + 東京=05 + kai=01 + day=08 + race=11
        assert rid.endswith("05010811")
        assert len(rid) == 12

    def test_from_meeting_info(self):
        """meeting_textがない場合meeting_infoから取得"""
        race = {
            "meeting_text": "",
            "meeting_idx": 0,
            "race_info": {"venue": "中山", "race_number": 1},
        }
        meeting_info = [(0, "2回中山3日", "中山")]
        rid = construct_race_id(race, meeting_info)
        assert "06020301" in rid

    def test_empty_meeting_text(self):
        """meeting_textもmeeting_infoもない"""
        race = {
            "meeting_text": "",
            "meeting_idx": 99,
            "race_info": {"venue": "東京", "race_number": 1},
        }
        assert construct_race_id(race, []) == ""

    def test_invalid_format(self):
        """パース不可能なmeeting_text"""
        race = {
            "meeting_text": "invalid",
            "race_info": {"venue": "東京", "race_number": 1},
        }
        assert construct_race_id(race, []) == ""

    def test_unknown_venue(self):
        """不明な会場"""
        race = {
            "meeting_text": "1回テスト1日",
            "race_info": {"venue": "テスト", "race_number": 1},
        }
        assert construct_race_id(race, []) == ""


# ============================================================
# helpers.collect_recommendations
# ============================================================

class TestCollectRecommendations:
    def test_empty(self):
        assert collect_recommendations({}) == []

    def test_low_confidence(self):
        """low_confidenceフラグがあると空リスト"""
        ev = {"low_confidence": True, "quinella": [{"ev": 2.0}]}
        assert collect_recommendations(ev) == []

    def test_filters_by_ev(self):
        """EV >= 1.0 のみ収集"""
        ev = {
            "quinella": [
                {"ev": 1.5, "combo": "1-2"},
                {"ev": 0.8, "combo": "3-4"},
            ],
            "wide": [
                {"ev": 1.1, "combo": "5-6"},
            ],
        }
        recs = collect_recommendations(ev)
        assert len(recs) == 2
        assert recs[0] == ("馬連", {"ev": 1.5, "combo": "1-2"})
        assert recs[1] == ("ワイド", {"ev": 1.1, "combo": "5-6"})

    def test_no_qualifying_bets(self):
        """全てEV < 1.0"""
        ev = {
            "quinella": [{"ev": 0.5, "combo": "1-2"}],
            "wide": [{"ev": 0.9, "combo": "3-4"}],
        }
        assert collect_recommendations(ev) == []


# ============================================================
# helpers.extract_venue
# ============================================================

class TestExtractVenue:
    def test_standard(self):
        assert extract_venue("1回東京8日") == "東京"

    def test_nakayama(self):
        assert extract_venue("2回中山3日") == "中山"


# ============================================================
# results._calculate_bet_results
# ============================================================

class TestCalculateBetResults:
    def _make_top3(self, nums):
        return [{"horse_number": n, "horse_name": f"馬{n}"} for n in nums]

    def test_low_confidence_skipped(self):
        """low_confidenceはスキップ"""
        result = _calculate_bet_results(
            {"low_confidence": True}, self._make_top3([1, 2, 3]), []
        )
        assert result["skipped"] is True
        assert result["total_invested"] == 0

    def test_no_bets(self):
        """推奨買い目なし"""
        ev = {"quinella": [], "wide": []}
        result = _calculate_bet_results(ev, self._make_top3([1, 2, 3]), [])
        assert result["total_invested"] == 0
        assert len(result["bets"]) == 0

    def test_ev_below_threshold(self):
        """EV < 1.0 は除外"""
        ev = {
            "quinella": [{"ev": 0.9, "combo": "1-2", "odds": 5.0}],
            "wide": [],
        }
        result = _calculate_bet_results(ev, self._make_top3([1, 2, 3]), [])
        assert result["total_invested"] == 0

    def test_quinella_hit(self):
        """馬連的中"""
        ev = {
            "quinella": [{"ev": 1.5, "combo": "1-2", "odds": 10.0}],
            "wide": [],
        }
        payoffs = [
            {"bet_type": "quinella", "combination": "1-2", "payout": 1000},
        ]
        result = _calculate_bet_results(
            ev, self._make_top3([1, 2, 3]), payoffs
        )
        assert result["total_invested"] == 100
        assert result["total_returned"] == 1000
        assert result["bets"][0]["won"] is True

    def test_quinella_miss(self):
        """馬連外れ"""
        ev = {
            "quinella": [{"ev": 1.5, "combo": "1-3", "odds": 8.0}],
            "wide": [],
        }
        payoffs = [
            {"bet_type": "quinella", "combination": "1-2", "payout": 1000},
        ]
        result = _calculate_bet_results(
            ev, self._make_top3([1, 2, 3]), payoffs
        )
        assert result["total_invested"] == 100
        assert result["total_returned"] == 0
        assert result["bets"][0]["won"] is False

    def test_wide_hit(self):
        """ワイド的中"""
        ev = {
            "quinella": [],
            "wide": [{"ev": 1.2, "combo": "2-3", "odds": 4.0}],
        }
        payoffs = [
            {"bet_type": "wide", "combination": "2-3", "payout": 400},
        ]
        result = _calculate_bet_results(
            ev, self._make_top3([1, 2, 3]), payoffs
        )
        assert result["total_invested"] == 100
        assert result["total_returned"] == 400
        assert result["bets"][0]["type"] == "ワイド"

    def test_multiple_bets(self):
        """複数買い目"""
        ev = {
            "quinella": [
                {"ev": 1.5, "combo": "1-2", "odds": 10.0},
                {"ev": 1.3, "combo": "1-3", "odds": 15.0},
            ],
            "wide": [
                {"ev": 1.1, "combo": "1-2", "odds": 3.0},
            ],
        }
        payoffs = [
            {"bet_type": "quinella", "combination": "1-2", "payout": 1000},
            {"bet_type": "wide", "combination": "1-2", "payout": 300},
        ]
        result = _calculate_bet_results(
            ev, self._make_top3([1, 2, 3]), payoffs
        )
        assert result["total_invested"] == 300  # 3 bets × 100
        assert result["total_returned"] == 1300  # 1000 + 0 + 300
        assert len(result["bets"]) == 3

    def test_combo_normalization(self):
        """組合せの順序が違っても正しくマッチ"""
        ev = {
            "quinella": [{"ev": 1.5, "combo": "3-1", "odds": 10.0}],
            "wide": [],
        }
        payoffs = [
            {"bet_type": "quinella", "combination": "1-3", "payout": 800},
        ]
        result = _calculate_bet_results(
            ev, self._make_top3([1, 3, 5]), payoffs
        )
        assert result["bets"][0]["won"] is True
        assert result["total_returned"] == 800

    def test_top3_in_result(self):
        """結果にtop3が含まれる"""
        result = _calculate_bet_results(
            {}, self._make_top3([5, 3, 8]), []
        )
        assert result["skipped"] is True
        assert len(result["top3"]) == 3
        assert result["top3"][0]["num"] == 5
