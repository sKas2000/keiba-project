"""Phase 13: 3-way split準拠のExpanding Windowパラメータ最適化
モデル学習は1回、bet_kwargsを変えてVal/Test別に評価
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from sklearn.isotonic import IsotonicRegression
from dateutil.relativedelta import relativedelta

from config.settings import FEATURE_COLUMNS, PROCESSED_DIR, RAW_DIR, CATEGORICAL_FEATURES
from src.model.evaluator import (
    simulate_bets, _load_returns, _empty_bet_stats, BET_TYPES
)
from src.model.trainer import DEFAULT_BINARY_PARAMS

input_path = str(PROCESSED_DIR / "features.csv")
returns_path = RAW_DIR / "returns.csv"

df = pd.read_csv(input_path, dtype={"race_id": str, "horse_id": str})
df["race_date"] = pd.to_datetime(df["race_date"])
returns = _load_returns(returns_path)

# Filter
df = df[df["race_class_code"] != 1]
df = df[df["surface_code"] != 2]
df = df[df["num_entries"] >= 6]
print(f"Data: {len(df)} rows")

available = [c for c in FEATURE_COLUMNS if c in df.columns]
cat_indices = [i for i, f in enumerate(available) if f in CATEGORICAL_FEATURES]
quiet_cb = [lgb.log_evaluation(0), lgb.early_stopping(50)]

# Step 1: Train models per window and cache prepared dicts
train_end = pd.Timestamp("2024-01-01")
max_date = df["race_date"].max() + pd.Timedelta(days=1)
window_months = 3
calibration_pct = 0.10

cached_prepared = []
window_idx = 0

print("Training models per window...")
while train_end < max_date:
    test_start = train_end
    test_end = train_end + relativedelta(months=window_months)
    if test_end > max_date:
        test_end = max_date

    full_train = df[df["race_date"] < train_end].copy()
    test_df = df[(df["race_date"] >= test_start) & (df["race_date"] < test_end)].copy()

    if len(full_train) < 1000 or len(test_df) < 100:
        train_end = test_end
        continue

    # Train/Cal split
    train_dates = np.sort(full_train["race_date"].unique())
    cal_idx = int(len(train_dates) * (1 - calibration_pct))
    cal_split = train_dates[cal_idx]
    train_inner = full_train[full_train["race_date"] < cal_split].copy()
    cal_set = full_train[full_train["race_date"] >= cal_split].copy()

    if len(train_inner) < 500 or len(cal_set) < 200:
        train_inner = full_train
        cal_set = None

    X_inner = train_inner[available].values.astype(np.float32)
    y_inner_top3 = train_inner["top3"].values
    y_inner_win = (train_inner["finish_position"] == 1).astype(int).values

    # Binary model
    bp = DEFAULT_BINARY_PARAMS.copy()
    if cal_set is not None:
        X_cal = cal_set[available].values.astype(np.float32)
        y_cal_top3 = cal_set["top3"].values
        d_tr = lgb.Dataset(X_inner, label=y_inner_top3, feature_name=available,
                           categorical_feature=cat_indices if cat_indices else "auto")
        d_cal = lgb.Dataset(X_cal, label=y_cal_top3, feature_name=available, reference=d_tr)
        binary_model = lgb.train(bp, d_tr, num_boost_round=1000,
                                 valid_sets=[d_cal], valid_names=["val"], callbacks=quiet_cb)
    else:
        X_test_tmp = test_df[available].values.astype(np.float32)
        d_tr = lgb.Dataset(X_inner, label=y_inner_top3, feature_name=available,
                           categorical_feature=cat_indices if cat_indices else "auto")
        d_te = lgb.Dataset(X_test_tmp, label=test_df["top3"].values,
                           feature_name=available, reference=d_tr)
        binary_model = lgb.train(bp, d_tr, num_boost_round=1000,
                                 valid_sets=[d_te], valid_names=["val"], callbacks=quiet_cb)

    # Win model
    wp = DEFAULT_BINARY_PARAMS.copy()
    wp["is_unbalance"] = True
    if cal_set is not None:
        y_cal_win = (cal_set["finish_position"] == 1).astype(int).values
        d_tr_w = lgb.Dataset(X_inner, label=y_inner_win, feature_name=available,
                             categorical_feature=cat_indices if cat_indices else "auto")
        d_cal_w = lgb.Dataset(X_cal, label=y_cal_win, feature_name=available, reference=d_tr_w)
        win_model = lgb.train(wp, d_tr_w, num_boost_round=1000,
                              valid_sets=[d_cal_w], valid_names=["val"], callbacks=quiet_cb)
    else:
        y_test_win = (test_df["finish_position"] == 1).astype(int).values
        d_tr_w = lgb.Dataset(X_inner, label=y_inner_win, feature_name=available,
                             categorical_feature=cat_indices if cat_indices else "auto")
        d_te_w = lgb.Dataset(X_test_tmp, label=y_test_win,
                             feature_name=available, reference=d_tr_w)
        win_model = lgb.train(wp, d_tr_w, num_boost_round=1000,
                              valid_sets=[d_te_w], valid_names=["val"], callbacks=quiet_cb)

    # Prediction + Isotonic
    X_test = test_df[available].values.astype(np.float32)
    raw_binary = binary_model.predict(X_test)
    raw_win = win_model.predict(X_test)

    if cal_set is not None:
        cal_binary_raw = binary_model.predict(X_cal)
        cal_win_raw = win_model.predict(X_cal)
        iso_binary = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds="clip")
        iso_binary.fit(cal_binary_raw, y_cal_top3)
        test_df["pred_prob"] = iso_binary.predict(raw_binary)
        iso_win = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds="clip")
        iso_win.fit(cal_win_raw, y_cal_win)
        test_df["win_prob_direct"] = iso_win.predict(raw_win)
    else:
        test_df["pred_prob"] = raw_binary
        test_df["win_prob_direct"] = raw_win

    test_df = test_df.sort_values(["race_id", "pred_prob"], ascending=[True, False])

    prepared = {
        "val_df": test_df.copy(),
        "full_df": df,
        "returns": returns,
        "has_win_model": True,
        "has_ranking": False,
        "calibrator": "isotonic",
        "val_start": str(test_start.date()),
        "val_end": str(test_end.date()),
    }

    cached_prepared.append((window_idx, test_start, test_end, prepared))
    print(f"  W{window_idx:02d} {test_start.date()}~{test_end.date()} cached ({len(test_df)} rows)")

    window_idx += 1
    train_end = test_end

print(f"\nCached {len(cached_prepared)} windows")


# Step 2: Define parameter grid
param_grid = [
    ("baseline",                {"top_n": 3}),
    ("conf=0.03",               {"top_n": 3, "confidence_min": 0.03}),
    ("conf=0.04",               {"top_n": 3, "confidence_min": 0.04}),
    ("conf=0.05",               {"top_n": 3, "confidence_min": 0.05}),
    ("skip[4,6]",               {"top_n": 3, "skip_classes": [4, 6]}),
    ("skip[5,6,7]",             {"top_n": 3, "skip_classes": [5, 6, 7]}),
    ("Q2",                      {"top_n": 3, "quinella_top_n": 2}),
    ("Q2+W2",                   {"top_n": 3, "quinella_top_n": 2, "wide_top_n": 2}),
    ("conf=0.03+skip[4,6]",     {"top_n": 3, "confidence_min": 0.03, "skip_classes": [4, 6]}),
    ("conf=0.04+skip[4,6]",     {"top_n": 3, "confidence_min": 0.04, "skip_classes": [4, 6]}),
    ("conf=0.05+skip[4,6]",     {"top_n": 3, "confidence_min": 0.05, "skip_classes": [4, 6]}),
    ("conf=0.04+skip[4,6]+Q2",  {"top_n": 3, "confidence_min": 0.04, "skip_classes": [4, 6],
                                 "quinella_top_n": 2}),
    ("conf=0.04+skip[4,6]+Q2+W2", {"top_n": 3, "confidence_min": 0.04, "skip_classes": [4, 6],
                                    "quinella_top_n": 2, "wide_top_n": 2}),
    ("conf=0.05+skip[4,6]+Q2+W2", {"top_n": 3, "confidence_min": 0.05, "skip_classes": [4, 6],
                                    "quinella_top_n": 2, "wide_top_n": 2}),
    ("conf=0.03+skip[4,6]+Q2+W2", {"top_n": 3, "confidence_min": 0.03, "skip_classes": [4, 6],
                                    "quinella_top_n": 2, "wide_top_n": 2}),
    ("conf=0.04+skip[4,6]+Q2+W3", {"top_n": 3, "confidence_min": 0.04, "skip_classes": [4, 6],
                                    "quinella_top_n": 2, "wide_top_n": 3}),
]

# Step 3: Evaluate each parameter set per period
val_indices = [4, 5]
test_indices = [6, 7, 8]
train_indices = [0, 1, 2, 3]
all_indices = list(range(len(cached_prepared)))


def eval_period(cached, indices, bet_kwargs):
    period_bets = {bt: _empty_bet_stats() for bt in BET_TYPES}
    total_races = 0
    for widx, ts, te, prep in cached:
        if widx not in indices:
            continue
        res = simulate_bets(prep, **bet_kwargs)
        total_races += res.get("races", 0)
        for bt in BET_TYPES:
            b = res.get(f"bets_{bt}", {})
            for k in ["count", "invested", "returned", "hits"]:
                period_bets[bt][k] += b.get(k, 0)
    rois = {}
    for bt in BET_TYPES:
        b = period_bets[bt]
        rois[bt] = round((b["returned"] / b["invested"] * 100) if b["invested"] > 0 else 0, 1)
    return rois, total_races


def fmt_rois(rois):
    parts = []
    for bt in ["win", "place", "quinella", "wide", "trio"]:
        r = rois[bt]
        m = "*" if r >= 100 else " "
        parts.append(f"{r:>5.1f}%{m}")
    return " ".join(parts)


print()
print("=" * 130)
print("  EXPANDING WINDOW: 3-WAY SPLIT PROPER EVALUATION")
print("=" * 130)
header = f"  {'Strategy':38s}"
header += f" | {'--- Val (W04-W05) ---':^42s}"
header += f" | {'--- Test (W06-W08) ---':^42s}"
header += f" | {'--- ALL ---':^42s} |"
print(header)

sub = f"  {'':38s}"
for _ in range(3):
    sub += f" | {'Win':>6s} {'Place':>6s} {'Q':>7s} {'Wide':>6s} {'Trio':>7s}"
sub += " |"
print(sub)
print(f"  {'-' * 128}")

for label, kwargs in param_grid:
    val_rois, _ = eval_period(cached_prepared, val_indices, kwargs)
    test_rois, _ = eval_period(cached_prepared, test_indices, kwargs)
    all_rois, _ = eval_period(cached_prepared, all_indices, kwargs)

    line = f"  {label:38s} | {fmt_rois(val_rois)} | {fmt_rois(test_rois)} | {fmt_rois(all_rois)} |"
    print(line)

print(f"  {'-' * 128}")
print("  (* = ROI >= 100%)")
print(f"  Val:  W04(2025-01~04) + W05(2025-04~07)")
print(f"  Test: W06(2025-07~10) + W07(2025-10~2026-01) + W08(2026-01~02)")
print(f"  ALL:  W00~W08 (all windows)")
