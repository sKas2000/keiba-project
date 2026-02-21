"""
keiba-ai 設定ファイル
全定数・パス・マップを一元管理
"""
import sys
from pathlib import Path

# ============================================================
# パス
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"
RACES_DIR = DATA_DIR / "races"
MODEL_DIR = PROJECT_ROOT / "models"
LOG_DIR = PROJECT_ROOT / "logs"

# ============================================================
# スクレイピング
# ============================================================
JRA_BASE_URL = "https://www.jra.go.jp/"
NETKEIBA_BASE_URL = "https://db.netkeiba.com"
PLAYWRIGHT_VIEWPORT = {"width": 1280, "height": 900}
PLAYWRIGHT_LOCALE = "ja-JP"
PLAYWRIGHT_TIMEOUT = 15000
SCRAPE_DELAY = 2.0
MAX_RETRIES = 3
RETRY_DELAY = 2.0
BROWSER_RESTART_INTERVAL = 5

# ============================================================
# エンコーディングマップ
# ============================================================
SEX_MAP = {"牡": 0, "牝": 1, "セ": 2, "騸": 2}
SURFACE_MAP = {"芝": 0, "ダ": 1, "障": 2}
TRACK_CONDITION_MAP = {"良": 0, "稍重": 1, "重": 2, "不良": 3}

CLASS_MAP = {
    "新馬": 1, "未勝利": 2, "1勝": 3, "2勝": 4, "3勝": 5,
    "OP": 6, "オープン": 6, "リステッド": 7, "L": 7,
    "G3": 8, "G2": 9, "G1": 10,
}

COURSE_CODES = {
    "01": "札幌", "02": "函館", "03": "福島", "04": "新潟",
    "05": "東京", "06": "中山", "07": "中京", "08": "京都",
    "09": "阪神", "10": "小倉",
}

COURSE_NAME_TO_ID = {v: int(k) for k, v in COURSE_CODES.items()}

# ============================================================
# スコアリング
# ============================================================
GRADE_DEFAULTS = {
    "G1": (8, 10000), "G2": (8, 5000), "G3": (10, 3000),
    "L": (10, 3000), "OP": (10, 3000),
    "3勝": (10, 1500), "2勝": (10, 1500),
    "1勝": (12, 1500), "未勝利": (12, 1500), "新馬": (14, 1500),
}

SCORE_LIMITS = {
    "ability": 50, "jockey": 20, "fitness": 15,
    "form": 10, "other": 5,
}

EV_RANK_THRESHOLDS = {"S": 1.5, "A": 1.2, "B": 1.0}

# バックテスト最適パラメータ（Platt Scaling + class_change）
# Train: ~2024-12, Val: 2025-01~06（キャリブレーター学習）, Test: 2025-07~
BACKTEST_BEST_PARAMS = {
    "ev_threshold": 2.0,       # Platt Scaling使用時の最適閾値
    "top_n": 3,
    "test_win_roi": 103.8,     # OOS Test set ← 黒字
    "test_place_roi": 107.5,   # OOS Test set ← 黒字
    "calibrated": True,        # Platt Scalingキャリブレーション済み
}

# ML用デフォルト温度（キャリブレーション未使用時のフォールバック）
ML_TEMPERATURE_DEFAULT = 1.0

# ============================================================
# ML特徴量カラム（Single Source of Truth）
# ============================================================
FEATURE_COLUMNS = [
    "frame_number", "horse_number", "sex_code", "age",
    "weight_carried", "horse_weight", "horse_weight_change",
    "num_entries",
    "surface_code", "distance", "track_condition_code",
    "race_class_code", "distance_cat", "course_id_code",
    # 過去成績
    "prev_finish_1", "prev_finish_2", "prev_finish_3",
    "avg_finish_last5", "best_finish_last5",
    "win_rate_last5", "place_rate_last5",
    "avg_last3f_last5", "days_since_last_race",
    "total_races", "career_win_rate", "career_place_rate",
    "surface_win_rate", "surface_place_rate",
    "distance_cat_win_rate",
    # 騎手
    "jockey_win_rate_365d", "jockey_place_rate_365d",
    "jockey_ride_count_365d",
    # v2: 新特徴量
    "prev_margin_1", "prev_last3f_1",
    "distance_change",
    "running_style", "avg_early_position_last5",
    "track_cond_place_rate",
    "trainer_win_rate_365d", "trainer_place_rate_365d",
    "race_month",
    # v3: クラス変動
    "class_change",
    # v4: 斤量変化
    "weight_carried_change",
    # v5: ローテーションパターン
    "prev_interval_2", "is_second_start",
    # v6: 展開予測（レース内脚質構成）
    "race_n_front", "race_n_mid", "race_n_back", "pace_advantage",
    # v7: コース×距離×枠順バイアス
    "post_position_bias",
]

CATEGORICAL_FEATURES = [
    "sex_code", "surface_code", "track_condition_code",
    "race_class_code", "distance_cat", "course_id_code",
    "running_style", "race_month",
]

META_COLUMNS = [
    "race_id", "race_date", "horse_name", "horse_id",
    "jockey_name", "finish_position", "top3",
    "win_odds", "popularity",
]

CSV_COLUMNS = [
    "race_id", "race_date", "course_id", "course_name",
    "race_number", "race_name", "race_class", "surface", "distance",
    "track_condition", "weather", "num_entries",
    "finish_position", "frame_number", "horse_number",
    "horse_name", "horse_id", "sex", "age",
    "jockey_name", "jockey_id", "trainer_name",
    "weight_carried", "horse_weight", "horse_weight_change",
    "finish_time_sec", "margin", "passing_order", "last_3f",
    "win_odds", "popularity", "prize_money",
]


# ============================================================
# Windows エンコーディング修正（起動時に1回だけ呼ぶ）
# ============================================================
def setup_encoding():
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
