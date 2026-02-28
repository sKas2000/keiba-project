"""
バックテスト最適化・戦略探索・条件別分析・ウォークフォワード検証
"""
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

from config.settings import FEATURE_COLUMNS, MODEL_DIR, PROCESSED_DIR, RAW_DIR
from src.model.evaluator import (
    BET_TYPES, BET_LABELS,
    _empty_bet_stats, _load_returns,
    prepare_backtest_data, simulate_bets,
)
from src.model.reporting import print_ev_comparison


def compare_ev_thresholds(input_path: str = None, model_dir: Path = None,
                          returns_path: str = None,
                          val_start: str = "2025-01-01",
                          val_end: str = None,
                          thresholds: list = None,
                          temperature: float = 1.0,
                          **filter_kwargs) -> list:
    """複数のEV閾値でバックテストを実行し比較"""
    thresholds = thresholds or [0.0, 0.8, 1.0, 1.2, 1.5, 2.0]

    prepared = prepare_backtest_data(
        input_path=input_path, model_dir=model_dir,
        returns_path=returns_path, val_start=val_start, val_end=val_end,
    )
    if prepared is None:
        return []

    comparison = []
    for t in thresholds:
        res = simulate_bets(
            prepared, ev_threshold=t, top_n=3, temperature=temperature,
            **filter_kwargs,
        )
        if not res:
            continue

        entry = {"ev_threshold": t}
        for bt in BET_TYPES:
            b = res[f"bets_{bt}"]
            roi = (b["returned"] / b["invested"] * 100) if b["invested"] > 0 else 0
            hit_rate = round(b["hits"] / b["count"] * 100, 1) if b["count"] > 0 else 0
            entry[f"{bt}_bets"] = b["count"]
            entry[f"{bt}_hits"] = b["hits"]
            entry[f"{bt}_hit_rate"] = hit_rate
            entry[f"{bt}_roi"] = round(roi, 1)
        comparison.append(entry)

    return comparison


def optimize_temperature(input_path: str = None, model_dir: Path = None,
                         returns_path: str = None,
                         val_start: str = "2025-01-01",
                         val_end: str = None,
                         ev_threshold: float = 1.0,
                         **filter_kwargs) -> dict:
    """ソフトマックス温度パラメータの最適化"""
    temperatures = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0]
    best = {"temperature": 1.0, "win_roi": 0, "place_roi": 0}

    prepared = prepare_backtest_data(
        input_path=input_path, model_dir=model_dir,
        returns_path=returns_path, val_start=val_start, val_end=val_end,
    )
    if prepared is None:
        return best

    period = f"{val_start}〜{val_end}" if val_end else f"{val_start}〜"
    print(f"\n  [温度パラメータ最適化] EV閾値={ev_threshold} 期間={period}")
    print(f"    {'温度':>6s} {'単勝回数':>8s} {'単勝的中率':>10s} {'単勝回収率':>10s} "
          f"{'複勝回数':>8s} {'複勝的中率':>10s} {'複勝回収率':>10s}")
    print(f"    {'─' * 70}")

    for temp in temperatures:
        res = simulate_bets(
            prepared, ev_threshold=ev_threshold, top_n=3, temperature=temp,
            **filter_kwargs,
        )
        if not res:
            continue
        bw = res["bets_win"]
        bp = res["bets_place"]
        roi_win = (bw["returned"] / bw["invested"] * 100) if bw["invested"] > 0 else 0
        roi_place = (bp["returned"] / bp["invested"] * 100) if bp["invested"] > 0 else 0
        hit_w = bw["hits"] / bw["count"] * 100 if bw["count"] > 0 else 0
        hit_p = bp["hits"] / bp["count"] * 100 if bp["count"] > 0 else 0

        mark = ""
        if roi_win > best["win_roi"]:
            best = {"temperature": temp, "win_roi": roi_win, "place_roi": roi_place,
                    "win_bets": bw["count"], "place_bets": bp["count"]}
            mark = " ★"

        print(f"    {temp:>6.1f} {bw['count']:>8d} {hit_w:>9.1f}% {roi_win:>9.1f}%{mark}"
              f" {bp['count']:>8d} {hit_p:>9.1f}% {roi_place:>9.1f}%")

    print(f"\n  最適温度: {best['temperature']} (単勝回収率: {best['win_roi']:.1f}%)")
    return best


def explore_strategies(input_path: str = None, model_dir: Path = None,
                       returns_path: str = None,
                       val_start: str = "2025-01-01",
                       val_end: str = None,
                       **filter_kwargs) -> list:
    """戦略探索: 確信度・オッズ帯・軸流しの組合せを網羅的にテスト"""
    strategies = [
        ("ベースライン（現状）", {}),
        ("確信度>=0.02", {"confidence_min": 0.02}),
        ("確信度>=0.03", {"confidence_min": 0.03}),
        ("確信度>=0.05", {"confidence_min": 0.05}),
        ("オッズ4-30倍", {"odds_min": 4.0, "odds_max": 30.0}),
        ("オッズ4-50倍", {"odds_min": 4.0, "odds_max": 50.0}),
        ("オッズ3-20倍", {"odds_min": 3.0, "odds_max": 20.0}),
        ("軸流し", {"axis_flow": True}),
        ("確信度>=0.03 + オッズ4-30倍",
         {"confidence_min": 0.03, "odds_min": 4.0, "odds_max": 30.0}),
        ("確信度>=0.03 + オッズ4-50倍",
         {"confidence_min": 0.03, "odds_min": 4.0, "odds_max": 50.0}),
        ("確信度>=0.03 + 軸流し",
         {"confidence_min": 0.03, "axis_flow": True}),
        ("確信度>=0.03 + オッズ4-30倍 + 軸流し",
         {"confidence_min": 0.03, "odds_min": 4.0, "odds_max": 30.0, "axis_flow": True}),
        ("EV>=1.0 + 確信度>=0.03",
         {"ev_threshold": 1.0, "confidence_min": 0.03}),
        ("EV>=1.0 + オッズ4-30倍",
         {"ev_threshold": 1.0, "odds_min": 4.0, "odds_max": 30.0}),
        ("EV>=1.0 + 確信度>=0.03 + オッズ4-30倍",
         {"ev_threshold": 1.0, "confidence_min": 0.03, "odds_min": 4.0, "odds_max": 30.0}),
        ("Kelly 1/4", {"kelly_fraction": 0.25}),
        ("Kelly 1/4 + 確信度>=0.03",
         {"kelly_fraction": 0.25, "confidence_min": 0.03}),
        ("Kelly 1/4 + 確信度>=0.05",
         {"kelly_fraction": 0.25, "confidence_min": 0.05}),
        ("Kelly 1/4 + conf>=0.03 + 3勝除外",
         {"kelly_fraction": 0.25, "confidence_min": 0.03, "skip_classes": [5]}),
        ("Kelly 1/4 + conf>=0.03 + 3勝OP除外",
         {"kelly_fraction": 0.25, "confidence_min": 0.03, "skip_classes": [5, 6, 7]}),
        ("Kelly 1/4 + conf>=0.05 + 3勝OP除外",
         {"kelly_fraction": 0.25, "confidence_min": 0.05, "skip_classes": [5, 6, 7]}),
        ("conf>=0.05 + skip3OP + Q2/W4",
         {"kelly_fraction": 0.25, "confidence_min": 0.05, "skip_classes": [5, 6, 7],
          "quinella_top_n": 2, "wide_top_n": 4}),
    ]

    period = f"{val_start}〜{val_end}" if val_end else f"{val_start}〜"
    print(f"\n  [戦略探索] 期間={period}")
    print(f"  {'─' * 105}")
    print(f"  {'戦略':36s} {'単勝':>6s} {'複勝':>6s} {'馬連':>6s} {'ﾜｲﾄﾞ':>6s}"
          f" {'馬単':>6s} {'3連複':>6s} {'3連単':>6s} {'Races':>6s} {'Skip':>5s}")
    print(f"  {'─' * 105}")

    prepared = prepare_backtest_data(
        input_path=input_path, model_dir=model_dir,
        returns_path=returns_path, val_start=val_start, val_end=val_end,
    )
    if prepared is None:
        return []

    all_results = []
    for label, kwargs in strategies:
        merged = {**filter_kwargs, **kwargs}
        res = simulate_bets(prepared, top_n=3, **merged)
        if not res:
            continue

        entry = {"label": label, "kwargs": kwargs, "races": res["races"]}
        entry["skipped"] = res.get("races_skipped", 0)
        row = f"  {label:36s}"
        for bt in BET_TYPES:
            b = res[f"bets_{bt}"]
            roi = (b["returned"] / b["invested"] * 100) if b["invested"] > 0 else 0
            entry[f"{bt}_roi"] = round(roi, 1)
            entry[f"{bt}_bets"] = b["count"]
            mark = "*" if roi >= 100 else ""
            row += f" {roi:>5.1f}%{mark}" if b["count"] > 0 else f" {'---':>6s}"
        row += f" {res['races']:>6d}"
        row += f" {entry['skipped']:>5d}"
        print(row)
        all_results.append(entry)

    print(f"  {'─' * 105}")
    print(f"  (* = 回収率100%超)")
    return all_results


def analyze_by_condition(input_path: str = None, model_dir: Path = None,
                         returns_path: str = None,
                         val_start: str = "2025-01-01",
                         val_end: str = None,
                         kelly_fraction: float = 0.25,
                         confidence_min: float = 0.03,
                         **filter_kwargs) -> dict:
    """クラス別・条件別にバックテストを実行し、エッジのある条件を特定"""
    conditions = {
        "クラス別": {
            "column": "race_class_code",
            "values": {
                "未勝利(2)": 2, "1勝(3)": 3, "2勝(4)": 4, "3勝(5)": 5,
                "OP/L(6-7)": [6, 7], "重賞(8-10)": [8, 9, 10],
            },
        },
        "馬場別": {
            "column": "surface_code",
            "values": {"芝(0)": 0, "ダート(1)": 1},
        },
        "距離別": {
            "column": "distance_cat",
            "values": {"短距離(0)": 0, "マイル(1)": 1, "中距離(2)": 2, "長距離(3)": 3},
        },
    }

    prepared = prepare_backtest_data(
        input_path=input_path, model_dir=model_dir,
        returns_path=returns_path, val_start=val_start, val_end=val_end,
    )
    if prepared is None:
        return {}

    val_df = prepared["val_df"]

    period = f"{val_start}〜{val_end}" if val_end else f"{val_start}〜"
    print(f"\n  [条件別分析] 期間={period}")

    all_results = {}
    for category_name, config in conditions.items():
        col = config["column"]
        if col not in val_df.columns:
            continue

        print(f"\n  ■ {category_name}")
        print(f"  {'条件':18s} {'レース':>6s} {'単勝ROI':>8s} {'複勝ROI':>8s} "
              f"{'馬連ROI':>8s} {'ﾜｲﾄﾞROI':>8s} {'3連複ROI':>9s}")
        print(f"  {'─' * 70}")

        category_results = {}
        for label, values in config["values"].items():
            if isinstance(values, list):
                target_races = set(val_df[val_df[col].isin(values)]["race_id"].unique())
            else:
                target_races = set(val_df[val_df[col] == values]["race_id"].unique())

            if not target_races:
                continue

            res = simulate_bets(
                prepared, top_n=3,
                kelly_fraction=kelly_fraction,
                confidence_min=confidence_min,
                race_ids=target_races,
                **filter_kwargs,
            )

            if not res or res.get("races", 0) == 0:
                continue

            rois = {}
            for bt in ["win", "place", "quinella", "wide", "trio"]:
                b = res.get(f"bets_{bt}", {})
                rois[bt] = (b["returned"] / b["invested"] * 100) if b.get("invested", 0) > 0 else 0

            marks = {bt: "*" if rois[bt] >= 100 else "" for bt in rois}
            print(f"  {label:18s} {res['races']:>6d} "
                  f"{rois['win']:>6.1f}%{marks['win']} "
                  f"{rois['place']:>6.1f}%{marks['place']} "
                  f"{rois['quinella']:>6.1f}%{marks['quinella']} "
                  f"{rois['wide']:>6.1f}%{marks['wide']} "
                  f"{rois['trio']:>7.1f}%{marks['trio']}")

            category_results[label] = {
                "races": res["races"],
                **{f"{bt}_roi": round(rois.get(bt, 0), 1) for bt in rois},
            }

        all_results[category_name] = category_results

    print(f"\n  {'─' * 70}")
    print(f"  (* = 回収率100%超)")
    return all_results


def expanding_window_backtest(
        input_path: str = None,
        returns_path: str = None,
        window_months: int = 3,
        initial_train_end: str = "2024-01-01",
        final_test_end: str = None,
        calibration_pct: float = 0.10,
        use_calibration: bool = True,
        use_ranking: bool = True,
        **bet_kwargs,
) -> dict:
    """Expanding Window Backtest（ウォーキングフォワード検証）"""
    from src.model.trainer import (
        DEFAULT_BINARY_PARAMS, DEFAULT_RANK_PARAMS,
        _get_categorical_indices, _make_rank_labels,
    )
    from sklearn.isotonic import IsotonicRegression
    from dateutil.relativedelta import relativedelta
    from config.settings import CATEGORICAL_FEATURES

    input_path = input_path or str(PROCESSED_DIR / "features.csv")
    returns_path = Path(returns_path) if returns_path else RAW_DIR / "returns.csv"

    df = pd.read_csv(input_path, dtype={"race_id": str, "horse_id": str})
    df["race_date"] = pd.to_datetime(df["race_date"])
    returns = _load_returns(returns_path)

    pre_filter = len(df)
    if "race_class_code" in df.columns:
        df = df[df["race_class_code"] != 1]
    if "surface_code" in df.columns:
        df = df[df["surface_code"] != 2]
    if "num_entries" in df.columns:
        df = df[df["num_entries"] >= 6]
    print(f"  [フィルタ] {pre_filter - len(df)}行除外 -> {len(df)}行")

    available = [c for c in FEATURE_COLUMNS if c in df.columns]
    cat_indices = [i for i, f in enumerate(available) if f in CATEGORICAL_FEATURES]

    train_end = pd.Timestamp(initial_train_end)
    if final_test_end:
        max_date = pd.Timestamp(final_test_end)
    else:
        max_date = df["race_date"].max() + pd.Timedelta(days=1)

    all_window_results = []
    total_bets = {bt: _empty_bet_stats() for bt in BET_TYPES}
    window_idx = 0

    flags = []
    if use_calibration:
        flags.append(f"cal={calibration_pct:.0%}")
    if use_ranking:
        flags.append("rank")
    flag_str = ", ".join(flags) if flags else "basic"
    print(f"\n  [Expanding Window Backtest] window={window_months}M, {flag_str}")
    print(f"  {'='*115}")
    print(f"  {'Win':8s} {'Train':>12s} {'Test':>18s} {'Races':>6s}"
          f" {'Win':>7s} {'Place':>7s} {'Q':>7s} {'Wide':>7s} {'Exacta':>7s} {'Trio':>7s} {'Trifct':>7s}")
    print(f"  {'-'*115}")

    quiet_cb = [lgb.log_evaluation(0), lgb.early_stopping(50)]

    while train_end < max_date:
        test_start = train_end
        test_end = train_end + relativedelta(months=window_months)
        if test_end > max_date:
            test_end = max_date

        full_train = df[df["race_date"] < train_end].copy()
        test_df = df[(df["race_date"] >= test_start) &
                     (df["race_date"] < test_end)].copy()

        if len(full_train) < 1000 or len(test_df) < 100:
            train_end = test_end
            continue

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
                                     valid_sets=[d_cal], valid_names=["val"],
                                     callbacks=quiet_cb)
        else:
            X_test_tmp = test_df[available].values.astype(np.float32)
            d_tr = lgb.Dataset(X_inner, label=y_inner_top3, feature_name=available,
                               categorical_feature=cat_indices if cat_indices else "auto")
            d_te = lgb.Dataset(X_test_tmp, label=test_df["top3"].values,
                               feature_name=available, reference=d_tr)
            binary_model = lgb.train(bp, d_tr, num_boost_round=1000,
                                     valid_sets=[d_te], valid_names=["val"],
                                     callbacks=quiet_cb)

        # Win model
        wp = DEFAULT_BINARY_PARAMS.copy()
        wp["is_unbalance"] = True
        if cal_set is not None:
            y_cal_win = (cal_set["finish_position"] == 1).astype(int).values
            d_tr_w = lgb.Dataset(X_inner, label=y_inner_win, feature_name=available,
                                 categorical_feature=cat_indices if cat_indices else "auto")
            d_cal_w = lgb.Dataset(X_cal, label=y_cal_win, feature_name=available, reference=d_tr_w)
            win_model = lgb.train(wp, d_tr_w, num_boost_round=1000,
                                  valid_sets=[d_cal_w], valid_names=["val"],
                                  callbacks=quiet_cb)
        else:
            y_test_win = (test_df["finish_position"] == 1).astype(int).values
            d_tr_w = lgb.Dataset(X_inner, label=y_inner_win, feature_name=available,
                                 categorical_feature=cat_indices if cat_indices else "auto")
            d_te_w = lgb.Dataset(X_test_tmp, label=y_test_win,
                                 feature_name=available, reference=d_tr_w)
            win_model = lgb.train(wp, d_tr_w, num_boost_round=1000,
                                  valid_sets=[d_te_w], valid_names=["val"],
                                  callbacks=quiet_cb)

        # Ranking model
        has_ranking = False
        if use_ranking:
            rp = DEFAULT_RANK_PARAMS.copy()
            train_rank = train_inner.sort_values("race_id").reset_index(drop=True)
            X_rank = train_rank[available].values.astype(np.float32)
            y_rank, grp_sizes = [], []
            for _, grp in train_rank.groupby("race_id", sort=True):
                labels = _make_rank_labels(grp["finish_position"].values, len(grp))
                y_rank.extend(labels)
                grp_sizes.append(len(grp))

        if use_ranking and len(grp_sizes) > 10:
            d_tr_r = lgb.Dataset(X_rank, label=np.array(y_rank, dtype=np.float32),
                                 group=grp_sizes, feature_name=available,
                                 categorical_feature=cat_indices if cat_indices else "auto")
            if cal_set is not None:
                cal_rank = cal_set.sort_values("race_id").reset_index(drop=True)
            else:
                cal_rank = test_df.sort_values("race_id").reset_index(drop=True)
            X_cal_r = cal_rank[available].values.astype(np.float32)
            y_cal_r, cal_grp = [], []
            for _, grp in cal_rank.groupby("race_id", sort=True):
                labels = _make_rank_labels(grp["finish_position"].values, len(grp))
                y_cal_r.extend(labels)
                cal_grp.append(len(grp))
            d_cal_r = lgb.Dataset(X_cal_r, label=np.array(y_cal_r, dtype=np.float32),
                                  group=cal_grp, feature_name=available, reference=d_tr_r)
            try:
                ranking_model = lgb.train(rp, d_tr_r, num_boost_round=1000,
                                          valid_sets=[d_cal_r], valid_names=["val"],
                                          callbacks=quiet_cb)
                has_ranking = True
            except Exception:
                has_ranking = False

        # Prediction + optional Isotonic Calibration
        X_test = test_df[available].values.astype(np.float32)
        raw_binary = binary_model.predict(X_test)
        raw_win = win_model.predict(X_test)

        if use_calibration and cal_set is not None:
            cal_binary_raw = binary_model.predict(X_cal)
            cal_win_raw = win_model.predict(X_cal)

            iso_binary = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds='clip')
            iso_binary.fit(cal_binary_raw, y_cal_top3)
            test_df["pred_prob"] = iso_binary.predict(raw_binary)

            iso_win = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds='clip')
            iso_win.fit(cal_win_raw, y_cal_win)
            test_df["win_prob_direct"] = iso_win.predict(raw_win)
        else:
            test_df["pred_prob"] = raw_binary
            test_df["win_prob_direct"] = raw_win

        if has_ranking:
            test_df["rank_score"] = ranking_model.predict(X_test)

        test_df = test_df.sort_values(
            ["race_id", "pred_prob"], ascending=[True, False]
        )

        prepared = {
            "val_df": test_df,
            "full_df": df,
            "returns": returns,
            "has_win_model": True,
            "has_ranking": has_ranking,
            "calibrator": "isotonic",
            "val_start": str(test_start.date()),
            "val_end": str(test_end.date()),
        }

        res = simulate_bets(prepared, **bet_kwargs)
        n_races = res.get("races", 0)
        if n_races == 0:
            train_end = test_end
            continue

        rois = {}
        for bt in BET_TYPES:
            b = res.get(f"bets_{bt}", {})
            rois[bt] = (b["returned"] / b["invested"] * 100) if b.get("invested", 0) > 0 else 0
            for k in ["count", "invested", "returned", "hits"]:
                total_bets[bt][k] = total_bets[bt].get(k, 0) + b.get(k, 0)

        mark_w = "*" if rois["win"] >= 100 else ""
        train_label = f"~{train_end.strftime('%Y-%m')}"
        test_label = f"{test_start.strftime('%Y-%m')}~{test_end.strftime('%Y-%m')}"
        print(f"  W{window_idx:02d}     {train_label:>12s} {test_label:>18s} {n_races:>6d}"
              f" {rois['win']:>6.1f}%{mark_w}"
              f" {rois['place']:>6.1f}%"
              f" {rois['quinella']:>6.1f}%"
              f" {rois['wide']:>6.1f}%"
              f" {rois['exacta']:>6.1f}%"
              f" {rois['trio']:>6.1f}%"
              f" {rois['trifecta']:>6.1f}%")

        window_bets = {}
        for bt in BET_TYPES:
            b = res.get(f"bets_{bt}", {})
            window_bets[bt] = {k: b.get(k, 0) for k in ["count", "invested", "returned", "hits"]}

        all_window_results.append({
            "window": window_idx,
            "train_end": str(train_end.date()),
            "test_start": str(test_start.date()),
            "test_end": str(test_end.date()),
            "races": n_races,
            "rois": {bt: round(rois.get(bt, 0), 1) for bt in BET_TYPES},
            "bets": window_bets,
        })

        window_idx += 1
        train_end = test_end

    # 総合結果
    print(f"  {'-'*115}")
    total_rois = {}
    for bt in BET_TYPES:
        b = total_bets[bt]
        total_rois[bt] = (b["returned"] / b["invested"] * 100) if b.get("invested", 0) > 0 else 0
    total_races = sum(w["races"] for w in all_window_results)
    mark_w = "*" if total_rois["win"] >= 100 else ""
    print(f"  {'TOTAL':8s} {'':>12s} {'':>18s} {total_races:>6d}"
          f" {total_rois['win']:>6.1f}%{mark_w}"
          f" {total_rois['place']:>6.1f}%"
          f" {total_rois['quinella']:>6.1f}%"
          f" {total_rois['wide']:>6.1f}%"
          f" {total_rois['exacta']:>6.1f}%"
          f" {total_rois['trio']:>6.1f}%"
          f" {total_rois['trifecta']:>6.1f}%")
    print(f"  {'='*115}")
    print(f"  (* = ROI >= 100%)")

    return {
        "windows": all_window_results,
        "total_rois": {bt: round(total_rois.get(bt, 0), 1) for bt in BET_TYPES},
        "total_races": total_races,
    }


def aggregate_windows_by_period(result: dict,
                                val_windows: list = None,
                                test_windows: list = None) -> dict:
    """Expanding Window結果を期間別に集計"""
    windows = result["windows"]
    val_windows = val_windows or []
    test_windows = test_windows or []
    train_windows = [w["window"] for w in windows
                     if w["window"] not in val_windows and w["window"] not in test_windows]

    period_results = {}
    for period_name, indices in [("train", train_windows), ("val", val_windows), ("test", test_windows)]:
        period_bets = {bt: _empty_bet_stats() for bt in BET_TYPES}
        period_races = 0
        for w in windows:
            if w["window"] not in indices:
                continue
            period_races += w["races"]
            for bt in BET_TYPES:
                wb = w.get("bets", {}).get(bt, {})
                for k in ["count", "invested", "returned", "hits"]:
                    period_bets[bt][k] += wb.get(k, 0)

        rois = {}
        for bt in BET_TYPES:
            b = period_bets[bt]
            rois[bt] = round((b["returned"] / b["invested"] * 100) if b["invested"] > 0 else 0, 1)

        period_results[f"{period_name}_rois"] = rois
        period_results[f"{period_name}_races"] = period_races
        period_results[f"{period_name}_bets"] = period_bets

    return period_results
