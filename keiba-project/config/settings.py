"""
keiba-ai 設定ファイル
全定数・パス・マップを一元管理
"""
import os
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


def distance_category(distance: int) -> int:
    """距離をカテゴリに変換 (0=短距離~1400, 1=マイル~1800, 2=中距離~2200, 3=長距離)"""
    if distance <= 1400:
        return 0
    elif distance <= 1800:
        return 1
    elif distance <= 2200:
        return 2
    return 3


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

# バックテスト最適パラメータ（固定分割: Phase6-7）
# Train: ~2024-12, Val: 2025-01~06, Test: 2025-07~
BACKTEST_BEST_PARAMS = {
    "kelly_fraction": 0.25,
    "confidence_min": 0.05,
    "skip_classes": [5, 6, 7],   # 3勝+OP/L除外
    "quinella_top_n": 2,          # 馬連: top2のみ
    "wide_top_n": 4,              # ワイド: top4まで
    "axis_flow": True,            # 軸流し
    "top_n": 3,
    "calibrated": True,
}

# Expanding Window最適パラメータ（Phase14: 体系的再精査実験で再検証済み）
# B7特徴量（+9列）+ Isotonic校正 + ランキングなし + 6ヶ月ウィンドウ
# 再精査実験: バグ修正後のコードで全パラメータを体系的に再検証
# エビデンス: docs/reeval_evidence.md
EXPANDING_BEST_PARAMS = {
    "kelly_fraction": 0,          # フラットベット（Kelly非使用、0.10/0.25で単勝-4pt悪化）
    "confidence_min": 0.04,       # 確信度フィルタ（0.04が最適、馬連+5.8pt）
    "quinella_top_n": 2,          # 馬連: top2のみ
    "wide_top_n": 2,              # ワイド: top2のみ
    "trio_top_n": 3,              # 3連複: top3（1点買い、trio=4→3でROI+11.5pt）
    "skip_classes": [4, 6],       # 2勝+OP除外（全券種改善）
    "top_n": 3,
    "use_calibration": True,      # Isotonic校正（必須）
    "use_ranking": False,         # ランキング不使用
    "calibration_pct": 0.10,      # 最適校正割合（10%）
    "window_months": 6,           # テスト期間（6ヶ月、再精査: 3→6で安定性向上）
    "ev_threshold_win": 1.0,      # 単勝EVフィルタ（expanding window検証: +2.5pt）
    # 2026-02-28 買い方改善実験:
    #   ev_threshold_win=1.0: 単勝86.8%(+2.5pt) 他券種不変
    #   PL確率フィルタ(quinella/wide/trio_prob_min): 2ヶ月では効果あり、長期では過学習
    #   ev_threshold=1.5(統合): 単勝+1.1pt だが複勝-4.7pt、分離がベスト
    #   skip=[4,5,6]: Trio+1.7pt だがQ-1.9pt（トレードオフ、現行[4,6]維持）
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
    "surface_code", "distance",
    "race_class_code", "course_id_code",
    # 過去成績
    "prev_finish_1", "prev_finish_2", "prev_finish_3",
    "avg_finish_last5", "best_finish_last5",
    "place_rate_last5",
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
    "prev_interval_2",
    # v6: 展開予測（レース内脚質構成）
    "race_n_front", "race_n_mid", "race_n_back",
    # v7: コース×距離×枠順バイアス
    "post_position_bias",
    # プルーニング済み（importance < 0.15%でノイズ源）:
    # track_condition_code(0.03%), is_second_start(0.06%), distance_cat(0.11%)
    # pace_advantage(0.05%), win_rate_last5(0.12%), same_jockey_win_rate(0.13%), course_dist_win_rate(0.13%)
    # v8: レース内Z-score（Phase14再精査で復活: B7_full Val 125.9% / Test 111.4%）
    # バグ修正後のフラットベットでは安定してROI改善
    "z_surface_place_rate", "z_jockey_place_rate_365d",
    "z_avg_finish_last5", "z_career_place_rate", "z_trainer_place_rate_365d",
    # v9: 騎手×馬の騎乗経験（Phase14再精査で復活: B7全体の一部として貢献）
    "same_jockey_rides",
    # v10: コース適性（Phase14再精査で復活: B7全体の一部として貢献）
    "course_dist_place_rate",
    # v11: 時間減衰特徴量 — 実験の結果、ROI悪化（市場追従化）のため不採用
    # "decay_finish_30d", "decay_finish_60d", "decay_finish_90d", "momentum",
    # v12: スーパープレミアム追加特徴量（ペース、馬体重）
    "avg_pace_front_last5", "pace_balance",
    # avg_pace_back_last5 は avg_last3f_last5(重要度1.57%)と重複するため除外
    "avg_weight_last3", "weight_stability",
    # v13: 穴馬発見特徴量（人気度・賞金） — 実験の結果、ROI悪化（市場追従化）のため不採用
    # "prev_popularity_1", "avg_popularity_last5",
    # "prev_prize_money_1", "avg_prize_money_last5",
]

CATEGORICAL_FEATURES = [
    "sex_code", "surface_code",
    "race_class_code", "course_id_code",
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

TRAINING_CSV_COLUMNS = [
    "race_id", "race_date", "horse_number", "horse_name", "horse_id",
    "training_date", "training_course", "track_condition",
    "rider", "training_time", "lap_times",
    "overall_time", "final_3f", "final_1f",
    "sparring_info", "training_load",
    "evaluation", "grade", "review",
]

PEDIGREE_CSV_COLUMNS = [
    "horse_id", "horse_name", "sire", "bms", "breeder",
]


# ============================================================
# Windows エンコーディング修正（起動時に1回だけ呼ぶ）
# ============================================================
_encoding_setup_done = False

def setup_encoding():
    global _encoding_setup_done
    if _encoding_setup_done:
        return
    _encoding_setup_done = True
    if sys.platform == 'win32':
        import io
        if hasattr(sys.stdout, 'buffer'):
            sys.stdout = io.TextIOWrapper(
                sys.stdout.buffer, encoding='utf-8', line_buffering=True)
        if hasattr(sys.stderr, 'buffer'):
            sys.stderr = io.TextIOWrapper(
                sys.stderr.buffer, encoding='utf-8', line_buffering=True)


_logging_setup_done = False

def setup_logging():
    """Python標準loggingを設定（コンソール + ファイル出力）"""
    global _logging_setup_done
    if _logging_setup_done:
        return
    _logging_setup_done = True

    import logging

    LOG_DIR.mkdir(exist_ok=True)
    log_file = LOG_DIR / "keiba.log"

    fmt = logging.Formatter(
        "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ファイルハンドラ（UTF-8, 追記, 5MB ローテーション）
    from logging.handlers import RotatingFileHandler
    fh = RotatingFileHandler(
        str(log_file), maxBytes=5 * 1024 * 1024, backupCount=3,
        encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    # コンソールハンドラ
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    root = logging.getLogger("keiba")
    root.setLevel(logging.DEBUG)
    root.addHandler(fh)
    root.addHandler(ch)


# ============================================================
# 環境変数ヘルパー（.envファイル対応）
# ============================================================

def load_env_var(key: str, default: str = "") -> str:
    """環境変数 or .envファイルから値を取得"""
    val = os.environ.get(key, "")
    if val:
        return val

    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            if k.strip() == key:
                return v.strip().strip('"').strip("'")
    return default
