"""
予測モジュール
ML予測 + 後方互換re-export（scoring / ev）
"""
import json
from pathlib import Path

import numpy as np

from config.settings import FEATURE_COLUMNS, MODEL_DIR

# 後方互換: scoring.py / ev.py から re-export
from src.model.scoring import (  # noqa: F401
    score_rule_based,
    calculate_ability_score, calculate_jockey_score,
    calculate_fitness_score, calculate_form_score,
    calculate_other_score,
)
from src.model.ev import (  # noqa: F401
    softmax, calc_place_probs, calc_quinella_prob,
    calc_wide_prob, calc_trio_prob, get_ev_rank,
    calculate_ev, print_ev_results,
)


# ============================================================
# ML 予測
# ============================================================

def score_ml(data: dict, model_dir: Path = None) -> dict | None:
    """ML モデルで予測してスコア付与

    binary_model: 3着以内確率（ソート・スコア用）
    win_model: 勝率直接推定（EV計算のwin_prob用）
    Isotonic校正があればPlatt Scalingより優先
    """
    import lightgbm as lgb
    from src.data.feature import extract_features_from_enriched

    model_dir = model_dir or MODEL_DIR
    binary_model_path = model_dir / "binary_model.txt"
    if not binary_model_path.exists():
        print(f"[WARN] モデル未学習: {binary_model_path}")
        return None

    binary_model = lgb.Booster(model_file=str(binary_model_path))

    meta_path = model_dir / "binary_meta.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    feature_names = meta.get("feature_names", FEATURE_COLUMNS)

    df = extract_features_from_enriched(data)
    available = [c for c in feature_names if c in df.columns]
    for col in [c for c in feature_names if c not in df.columns]:
        df[col] = 0.0
    X = df[feature_names].values.astype(np.float32)

    raw_probs = binary_model.predict(X)

    # キャリブレーション: Isotonic優先 > Platt Scaling > raw
    from src.model.trainer import (
        load_calibrator, calibrate_probs,
        load_isotonic_calibrator, calibrate_isotonic,
    )
    iso_binary = load_isotonic_calibrator("binary_isotonic", model_dir)
    platt_cal = load_calibrator(model_dir)
    if iso_binary is not None:
        probs = calibrate_isotonic(raw_probs, iso_binary)
        cal_method = "isotonic"
    elif platt_cal is not None:
        probs = calibrate_probs(raw_probs, platt_cal)
        cal_method = "platt"
    else:
        probs = raw_probs
        cal_method = "raw"

    # 勝率直接推定モデル（Expanding Window検証で使用）
    win_model_path = model_dir / "win_model.txt"
    win_probs_direct = None
    if win_model_path.exists():
        win_model = lgb.Booster(model_file=str(win_model_path))
        raw_win = win_model.predict(X)
        iso_win = load_isotonic_calibrator("win_isotonic", model_dir)
        if iso_win is not None:
            win_probs_direct = calibrate_isotonic(raw_win, iso_win)
        else:
            win_probs_direct = raw_win

    horses = data.get("horses", [])
    for i, horse in enumerate(horses):
        if i < len(probs):
            prob = float(probs[i])
            horse["score"] = round(prob * 100, 1)
            horse["ml_top3_prob"] = round(prob, 4)
            horse["ml_calibrated"] = cal_method != "raw"
            if win_probs_direct is not None and i < len(win_probs_direct):
                horse["ml_win_prob"] = round(float(win_probs_direct[i]), 6)
            horse["score_breakdown"] = {
                "ml_binary": round(prob * 100, 1),
                "ability": 0, "jockey": 0, "fitness": 0, "form": 0, "other": 0,
            }
            horse["note"] = f"[ML:{cal_method}] top3={prob:.3f}"
            if win_probs_direct is not None and i < len(win_probs_direct):
                horse["note"] += f" win={win_probs_direct[i]:.4f}"

    return data
