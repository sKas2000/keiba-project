"""
スクレイパーHTMLスナップショットテスト
HTMLフィクスチャ（模擬HTML）に対してパーサーを実行し、
サイト側のHTML構造変更を検知する回帰テスト
"""
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


# ============================================================
# レース結果パーサー (race.py: parse_race_html)
# ============================================================

class TestParseRaceResult:
    """race.netkeiba.com 形式のレース結果HTMLパーサー"""

    @pytest.fixture
    def html(self):
        return (FIXTURES / "race_result.html").read_bytes()

    @pytest.fixture
    def rows(self, html):
        from src.scraping.race import parse_race_html
        return parse_race_html(html, "202506010101")

    def test_row_count(self, rows):
        """4頭分のデータが取得できる"""
        assert len(rows) == 4

    def test_finish_positions(self, rows):
        """着順が正しくパースされる"""
        finishes = [r["finish_position"] for r in rows]
        assert finishes == [1, 2, 3, 4]

    def test_horse_names(self, rows):
        """馬名が正しく取得される"""
        names = [r["horse_name"] for r in rows]
        assert names == ["テストホース1", "テストホース2", "テストホース3", "テストホース4"]

    def test_horse_numbers(self, rows):
        """馬番が正しくパースされる"""
        nums = [r["horse_number"] for r in rows]
        assert nums == [5, 2, 8, 12]

    def test_frame_numbers(self, rows):
        """枠番が正しくパースされる"""
        frames = [r["frame_number"] for r in rows]
        assert frames == [3, 1, 5, 7]

    def test_sex_age(self, rows):
        """性齢が正しくパースされる"""
        sexes = [r["sex"] for r in rows]
        ages = [r["age"] for r in rows]
        assert sexes == ["牡", "牝", "セ", "牡"]
        assert ages == [3, 3, 3, 3]

    def test_weight_carried(self, rows):
        """斤量が正しくパースされる"""
        wts = [r["weight_carried"] for r in rows]
        assert wts == [56.0, 54.0, 56.0, 56.0]

    def test_jockey_names(self, rows):
        """騎手名が正しく取得される"""
        jockeys = [r["jockey_name"] for r in rows]
        assert jockeys == ["ルメール", "武豊", "デムーロ", "横山武史"]

    def test_finish_time(self, rows):
        """タイムが秒数に変換される"""
        assert rows[0]["finish_time_sec"] == pytest.approx(96.2)
        assert rows[1]["finish_time_sec"] == pytest.approx(96.5)

    def test_last_3f(self, rows):
        """上がり3Fが正しくパースされる"""
        l3f = [r["last_3f"] for r in rows]
        assert l3f == [34.5, 35.1, 34.8, 34.9]

    def test_win_odds(self, rows):
        """単勝オッズが正しくパースされる"""
        odds = [r["win_odds"] for r in rows]
        assert odds == [3.2, 5.8, 8.5, 12.3]

    def test_popularity(self, rows):
        """人気順が正しくパースされる"""
        pops = [r["popularity"] for r in rows]
        assert pops == [1, 3, 4, 5]

    def test_horse_weight(self, rows):
        """馬体重が正しくパースされる"""
        hw = [r["horse_weight"] for r in rows]
        hwc = [r["horse_weight_change"] for r in rows]
        assert hw == [480, 460, 500, 470]
        assert hwc == [4, -6, 0, 2]

    def test_horse_id(self, rows):
        """馬IDがリンクから抽出される"""
        ids = [r.get("horse_id", "") for r in rows]
        assert ids == ["2022104567", "2022109876", "2022103333", "2022107777"]

    def test_jockey_id(self, rows):
        """騎手IDがリンクから抽出される"""
        ids = [r.get("jockey_id", "") for r in rows]
        assert ids == ["01234", "05678", "09012", "03456"]

    def test_race_meta_surface(self, rows):
        """馬場情報がメタデータから取得される"""
        assert rows[0]["surface"] == "芝"
        assert rows[0]["distance"] == 1600

    def test_race_meta_date(self, rows):
        """日付がメタデータから取得される"""
        assert rows[0]["race_date"] == "2025-01-05"

    def test_race_meta_course(self, rows):
        """コース名がrace_idから取得される"""
        assert rows[0]["course_id"] == "06"
        assert rows[0]["course_name"] == "中山"

    def test_num_entries(self, rows):
        """出走頭数が自動設定される"""
        for r in rows:
            assert r["num_entries"] == 4

    def test_trainer(self, rows):
        """調教師名が正しく取得される"""
        trainers = [r["trainer_name"] for r in rows]
        assert trainers == ["田中太郎", "佐藤花子", "鈴木一郎", "高橋次郎"]


# ============================================================
# 払い戻しパーサー (race.py: parse_return_html)
# ============================================================

class TestParseReturnData:
    """払い戻しテーブルのパーサー"""

    @pytest.fixture
    def html(self):
        return (FIXTURES / "race_result.html").read_bytes()

    @pytest.fixture
    def returns(self, html):
        from src.scraping.race import parse_return_html
        return parse_return_html(html, "202506010101")

    def test_return_count(self, returns):
        """全券種の払い戻しデータが取得される"""
        assert len(returns) >= 7  # 単勝+複勝3+馬連+ワイド3+馬単+3連複+3連単

    def test_win_payout(self, returns):
        """単勝の払い戻し"""
        win = [r for r in returns if r["bet_type"] == "win"]
        assert len(win) == 1
        assert win[0]["combination"] == "5"
        assert win[0]["payout"] == 320

    def test_place_payouts(self, returns):
        """複勝の払い戻し（3頭分）"""
        place = [r for r in returns if r["bet_type"] == "place"]
        assert len(place) == 3
        payouts = sorted([p["payout"] for p in place])
        assert payouts == [150, 210, 290]

    def test_quinella_payout(self, returns):
        """馬連の払い戻し"""
        q = [r for r in returns if r["bet_type"] == "quinella"]
        assert len(q) == 1
        assert q[0]["payout"] == 950

    def test_wide_payouts(self, returns):
        """ワイドの払い戻し（3組分）"""
        w = [r for r in returns if r["bet_type"] == "wide"]
        assert len(w) == 3
        payouts = sorted([p["payout"] for p in w])
        assert payouts == [380, 520, 860]

    def test_exacta_payout(self, returns):
        """馬単の払い戻し"""
        ex = [r for r in returns if r["bet_type"] == "exacta"]
        assert len(ex) == 1
        assert ex[0]["payout"] == 1230

    def test_trio_payout(self, returns):
        """3連複の払い戻し"""
        trio = [r for r in returns if r["bet_type"] == "trio"]
        assert len(trio) == 1
        assert trio[0]["payout"] == 2450

    def test_trifecta_payout(self, returns):
        """3連単の払い戻し"""
        tri = [r for r in returns if r["bet_type"] == "trifecta"]
        assert len(tri) == 1
        assert tri[0]["payout"] == 8960


# ============================================================
# 出馬表パーサー (race.py: parse_shutuba_html)
# ============================================================

class TestParseShutubaTable:
    """出馬表HTMLのパーサー"""

    @pytest.fixture
    def html(self):
        return (FIXTURES / "race_shutuba.html").read_bytes()

    @pytest.fixture
    def result(self, html):
        from src.scraping.race import parse_shutuba_html
        return parse_shutuba_html(html, "202506010101")

    def test_horse_count(self, result):
        """3頭分のデータが取得できる"""
        assert len(result["horses"]) == 3

    def test_horse_numbers(self, result):
        """馬番が正しくパースされる"""
        nums = [h["horse_number"] for h in result["horses"]]
        assert nums == [1, 3, 7]

    def test_horse_names(self, result):
        """馬名が正しく取得される"""
        names = [h["horse_name"] for h in result["horses"]]
        assert names == ["シャトルA", "シャトルB", "シャトルC"]

    def test_jockey_names(self, result):
        """騎手名が正しく取得される"""
        jockeys = [h["jockey_name"] for h in result["horses"]]
        assert jockeys == ["ルメール", "川田将雅", "横山武史"]

    def test_trainer_names(self, result):
        """調教師名が正しく取得される"""
        trainers = [h["trainer_name"] for h in result["horses"]]
        assert trainers == ["国枝栄", "矢作芳人", "木村哲也"]

    def test_sex_age(self, result):
        """性齢がパースされる"""
        sa = [h["sex_age"] for h in result["horses"]]
        assert sa == ["牡3", "牝3", "牡3"]

    def test_weight_carried(self, result):
        """斤量がパースされる"""
        wts = [h["weight_carried"] for h in result["horses"]]
        assert wts == [56.0, 54.0, 56.0]

    def test_horse_weight(self, result):
        """馬体重がパースされる"""
        hw = [h["horse_weight"] for h in result["horses"]]
        hwc = [h["horse_weight_change"] for h in result["horses"]]
        assert hw == [488, 456, 502]
        assert hwc == [2, -4, 0]

    def test_horse_id(self, result):
        """馬IDがリンクから抽出される"""
        ids = [h.get("horse_id", "") for h in result["horses"]]
        assert ids == ["2022101111", "2022102222", "2022103333"]

    def test_race_meta_surface(self, result):
        """馬場情報がメタデータから取得される"""
        meta = result["race_meta"]
        assert meta["surface"] == "ダ"
        assert meta["distance"] == 1200

    def test_race_meta_condition(self, result):
        """馬場状態が取得される"""
        meta = result["race_meta"]
        assert meta["track_condition"] == "稍重"

    def test_race_meta_date(self, result):
        """日付がメタデータから取得される"""
        meta = result["race_meta"]
        assert meta["race_date"] == "2025-01-05"
