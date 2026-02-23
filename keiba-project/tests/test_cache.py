"""src/scraping/cache.py のユニットテスト"""
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from src.scraping.cache import HorseCache


@pytest.fixture
def cache(tmp_path):
    """一時DBを使うキャッシュインスタンス"""
    db_path = tmp_path / "test_cache.db"
    c = HorseCache(db_path=db_path)
    yield c
    c.close()


class TestHorseData:
    """馬データのキャッシュテスト"""

    def test_set_and_get(self, cache):
        """保存したデータが取得できる"""
        cache.set_horse("2022104567", "テスト馬",
                        [{"finish": 1, "date": "2025-01-01"}], "田中太郎")
        result = cache.get_horse("2022104567")
        assert result is not None
        assert result["past_races"] == [{"finish": 1, "date": "2025-01-01"}]
        assert result["trainer_name"] == "田中太郎"

    def test_get_nonexistent(self, cache):
        """存在しないキーはNone"""
        assert cache.get_horse("9999999999") is None

    def test_get_empty_id(self, cache):
        """空IDはNone"""
        assert cache.get_horse("") is None

    def test_update_existing(self, cache):
        """同じIDで上書き保存"""
        cache.set_horse("2022104567", "馬A", [{"finish": 3}], "調教師A")
        cache.set_horse("2022104567", "馬A", [{"finish": 1}], "調教師B")
        result = cache.get_horse("2022104567")
        assert result["past_races"] == [{"finish": 1}]
        assert result["trainer_name"] == "調教師B"

    def test_ttl_expired(self, cache):
        """TTL超過でNone返却"""
        cache.set_horse("2022104567", "テスト馬", [], "")
        # updated_atを8日前に強制変更
        conn = cache._connect()
        old_date = (datetime.now() - timedelta(days=8)).isoformat()
        conn.execute("UPDATE horse_data SET updated_at = ? WHERE horse_id = ?",
                     (old_date, "2022104567"))
        conn.commit()
        assert cache.get_horse("2022104567", ttl_days=7) is None

    def test_ttl_not_expired(self, cache):
        """TTL内ならデータ返却"""
        cache.set_horse("2022104567", "テスト馬", [{"finish": 2}], "")
        result = cache.get_horse("2022104567", ttl_days=7)
        assert result is not None


class TestJockeyStats:
    """騎手成績のキャッシュテスト"""

    def test_set_and_get(self, cache):
        """保存した成績が取得できる"""
        stats = {"win_rate": 0.15, "place_rate": 0.35, "wins": 20, "races": 133}
        cache.set_jockey("ルメール", stats)
        result = cache.get_jockey("ルメール")
        assert result is not None
        assert result["win_rate"] == 0.15
        assert result["races"] == 133

    def test_get_nonexistent(self, cache):
        """存在しない騎手はNone"""
        assert cache.get_jockey("存在しない騎手") is None

    def test_get_empty_name(self, cache):
        """空名はNone"""
        assert cache.get_jockey("") is None

    def test_ttl_expired(self, cache):
        """TTL超過でNone返却"""
        cache.set_jockey("テスト騎手", {"win_rate": 0.1})
        conn = cache._connect()
        old_date = (datetime.now() - timedelta(days=2)).isoformat()
        conn.execute("UPDATE jockey_stats SET updated_at = ? WHERE jockey_name = ?",
                     (old_date, "テスト騎手"))
        conn.commit()
        assert cache.get_jockey("テスト騎手", ttl_days=1) is None


class TestCacheUtilities:
    """ユーティリティ機能のテスト"""

    def test_stats(self, cache):
        """統計情報の取得"""
        cache.set_horse("001", "馬1", [], "")
        cache.set_horse("002", "馬2", [], "")
        cache.set_jockey("騎手1", {})
        s = cache.stats()
        assert s["horses"] == 2
        assert s["jockeys"] == 1

    def test_clear(self, cache):
        """全消去"""
        cache.set_horse("001", "馬1", [], "")
        cache.set_jockey("騎手1", {})
        cache.clear()
        s = cache.stats()
        assert s["horses"] == 0
        assert s["jockeys"] == 0

    def test_cleanup_expired(self, cache):
        """期限切れエントリの削除"""
        cache.set_horse("001", "新しい馬", [], "")
        cache.set_horse("002", "古い馬", [], "")
        # 002を30日前に変更
        conn = cache._connect()
        old_date = (datetime.now() - timedelta(days=30)).isoformat()
        conn.execute("UPDATE horse_data SET updated_at = ? WHERE horse_id = ?",
                     (old_date, "002"))
        conn.commit()

        cache.cleanup_expired(horse_ttl=7)
        assert cache.get_horse("001") is not None
        assert cache.get_horse("002", ttl_days=999) is None  # 削除されている

    def test_japanese_content(self, cache):
        """日本語コンテンツの保存・取得"""
        cache.set_horse("2022104567", "ディープインパクト",
                        [{"race": "有馬記念", "venue": "中山"}], "池江泰寿")
        result = cache.get_horse("2022104567")
        assert result["past_races"][0]["race"] == "有馬記念"
        assert result["trainer_name"] == "池江泰寿"
