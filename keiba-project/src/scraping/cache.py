"""
馬情報キャッシュDB（SQLite）
スクレイピング結果をキャッシュして同一馬・騎手の再取得を回避
"""
import json
import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

from config.settings import DATA_DIR

logger = logging.getLogger("keiba.scraping.cache")

# デフォルトキャッシュDB パス
DEFAULT_DB_PATH = DATA_DIR / "cache" / "horse_cache.db"

# キャッシュ有効期間（日）
HORSE_TTL_DAYS = 7       # 過去走データ（週1更新で十分）
JOCKEY_TTL_DAYS = 1      # 騎手成績（毎日変動しうる）
TRAINER_TTL_DAYS = 7     # 調教師名（ほぼ変わらない）


class HorseCache:
    """SQLiteベースの馬情報キャッシュ"""

    def __init__(self, db_path: Path = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = None

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._create_tables()
        return self._conn

    def _create_tables(self):
        conn = self._conn
        conn.execute("""
            CREATE TABLE IF NOT EXISTS horse_data (
                horse_id TEXT PRIMARY KEY,
                horse_name TEXT,
                past_races TEXT,
                trainer_name TEXT,
                updated_at TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS jockey_stats (
                jockey_name TEXT PRIMARY KEY,
                stats TEXT,
                updated_at TEXT
            )
        """)
        conn.commit()

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    # ----------------------------------------------------------
    # 馬データ
    # ----------------------------------------------------------

    def get_horse(self, horse_id: str, ttl_days: int = HORSE_TTL_DAYS) -> dict | None:
        """キャッシュから馬データを取得（TTL超過ならNone）"""
        if not horse_id:
            return None
        conn = self._connect()
        row = conn.execute(
            "SELECT past_races, trainer_name, updated_at FROM horse_data WHERE horse_id = ?",
            (horse_id,),
        ).fetchone()
        if row is None:
            return None
        past_races_json, trainer_name, updated_at = row
        if self._is_expired(updated_at, ttl_days):
            return None
        return {
            "past_races": json.loads(past_races_json) if past_races_json else [],
            "trainer_name": trainer_name or "",
        }

    def set_horse(self, horse_id: str, horse_name: str,
                  past_races: list, trainer_name: str = ""):
        """馬データをキャッシュに保存"""
        if not horse_id:
            return
        conn = self._connect()
        conn.execute(
            """INSERT OR REPLACE INTO horse_data
               (horse_id, horse_name, past_races, trainer_name, updated_at)
               VALUES (?, ?, ?, ?, ?)""",
            (horse_id, horse_name, json.dumps(past_races, ensure_ascii=False),
             trainer_name, datetime.now().isoformat()),
        )
        conn.commit()

    # ----------------------------------------------------------
    # 騎手成績
    # ----------------------------------------------------------

    def get_jockey(self, jockey_name: str, ttl_days: int = JOCKEY_TTL_DAYS) -> dict | None:
        """キャッシュから騎手成績を取得（TTL超過ならNone）"""
        if not jockey_name:
            return None
        conn = self._connect()
        row = conn.execute(
            "SELECT stats, updated_at FROM jockey_stats WHERE jockey_name = ?",
            (jockey_name,),
        ).fetchone()
        if row is None:
            return None
        stats_json, updated_at = row
        if self._is_expired(updated_at, ttl_days):
            return None
        return json.loads(stats_json) if stats_json else None

    def set_jockey(self, jockey_name: str, stats: dict):
        """騎手成績をキャッシュに保存"""
        if not jockey_name:
            return
        conn = self._connect()
        conn.execute(
            """INSERT OR REPLACE INTO jockey_stats
               (jockey_name, stats, updated_at)
               VALUES (?, ?, ?)""",
            (jockey_name, json.dumps(stats, ensure_ascii=False),
             datetime.now().isoformat()),
        )
        conn.commit()

    # ----------------------------------------------------------
    # ユーティリティ
    # ----------------------------------------------------------

    def _is_expired(self, updated_at: str, ttl_days: int) -> bool:
        """TTL超過チェック"""
        try:
            updated = datetime.fromisoformat(updated_at)
            return datetime.now() - updated > timedelta(days=ttl_days)
        except (ValueError, TypeError):
            return True

    def stats(self) -> dict:
        """キャッシュ統計"""
        conn = self._connect()
        horse_count = conn.execute("SELECT COUNT(*) FROM horse_data").fetchone()[0]
        jockey_count = conn.execute("SELECT COUNT(*) FROM jockey_stats").fetchone()[0]
        return {"horses": horse_count, "jockeys": jockey_count}

    def clear(self):
        """キャッシュ全消去"""
        conn = self._connect()
        conn.execute("DELETE FROM horse_data")
        conn.execute("DELETE FROM jockey_stats")
        conn.commit()

    def cleanup_expired(self, horse_ttl: int = HORSE_TTL_DAYS,
                        jockey_ttl: int = JOCKEY_TTL_DAYS):
        """期限切れエントリを削除"""
        conn = self._connect()
        horse_cutoff = (datetime.now() - timedelta(days=horse_ttl)).isoformat()
        jockey_cutoff = (datetime.now() - timedelta(days=jockey_ttl)).isoformat()
        conn.execute("DELETE FROM horse_data WHERE updated_at < ?", (horse_cutoff,))
        conn.execute("DELETE FROM jockey_stats WHERE updated_at < ?", (jockey_cutoff,))
        conn.commit()
