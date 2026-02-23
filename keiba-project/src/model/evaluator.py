"""
バックテスト・評価モジュール
学習済みモデルの予測結果で過去レースの回収率をシミュレーション
全7券種対応: 単勝・複勝・馬連・ワイド・馬単・3連複・3連単
EVフィルタリング + 実払い戻しデータ対応
Phase 3: 勝率直接推定モデル + Isotonic Regression + Kelly基準
"""
import json
import pickle
from itertools import combinations, permutations
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

from config.settings import FEATURE_COLUMNS, MODEL_DIR, PROCESSED_DIR, RAW_DIR

# 複勝オッズ推定回帰係数（returns.csv 30,378件から算出）
# 旧: 線形モデル（R²=0.804, MAE=0.789）
# 新: 多変量モデル（R²=0.807, MAE=0.758） win_odds + num_entries + log(1+win_odds)
PLACE_ODDS_COEF_WIN = 0.13470
PLACE_ODDS_COEF_ENTRIES = 0.06456
PLACE_ODDS_COEF_LOG = 0.24967
PLACE_ODDS_INTERCEPT = -0.1849
# フォールバック用（num_entriesが不明な場合）
PLACE_ODDS_SLOPE = 0.1414
PLACE_ODDS_INTERCEPT_SIMPLE = 1.1475

# 全券種キー（内部用）
BET_TYPES = ["win", "place", "quinella", "wide", "exacta", "trio", "trifecta"]
BET_LABELS = {
    "win": "単勝", "place": "複勝", "quinella": "馬連", "wide": "ワイド",
    "exacta": "馬単", "trio": "3連複", "trifecta": "3連単",
}

# Kelly基準のデフォルト銀行残高（ベットサイズ計算用）
KELLY_BANKROLL = 100000


def _estimate_place_odds(win_odds: float, num_entries: int = 0) -> float:
    """複勝オッズ推定（多変量モデル）"""
    if num_entries > 0:
        log_win = np.log1p(max(win_odds, 0))
        est = (PLACE_ODDS_COEF_WIN * win_odds
               + PLACE_ODDS_COEF_ENTRIES * num_entries
               + PLACE_ODDS_COEF_LOG * log_win
               + PLACE_ODDS_INTERCEPT)
    else:
        est = PLACE_ODDS_SLOPE * win_odds + PLACE_ODDS_INTERCEPT_SIMPLE
    return max(est, 1.0)


def _kelly_fraction_calc(prob: float, odds: float) -> float:
    """Kelly基準のベット割合を計算
    f* = (b*p - q) / b  where b = odds-1, p = prob, q = 1-p
    """
    if odds <= 1 or prob <= 0:
        return 0.0
    b = odds - 1.0
    f = (b * prob - (1 - prob)) / b
    return max(0.0, f)


# ============================================================
# 払い戻しデータ読み込み
# ============================================================

def _load_returns(returns_path: Path) -> dict:
    """returns.csv を {race_id -> {bet_type -> [{combination, payout}]}} に変換
    馬単・3連単の区切り文字を → から - に統一
    """
    if not returns_path.exists():
        return {}
    df = pd.read_csv(returns_path, dtype={"race_id": str})
    lookup = {}
    for _, row in df.iterrows():
        rid = row["race_id"]
        if rid not in lookup:
            lookup[rid] = {}
        bt = row["bet_type"]
        if bt not in lookup[rid]:
            lookup[rid][bt] = []
        comb = str(row["combination"]).replace("\u2192", "-")
        lookup[rid][bt].append({
            "combination": comb,
            "payout": float(row["payout"]),
        })
    return lookup


def _get_payout(returns: dict, race_id: str, bet_type: str, combination: str) -> float:
    """指定券種・組合せの実払い戻しを取得（100円あたり）"""
    race_returns = returns.get(race_id, {})
    entries = race_returns.get(bet_type, [])
    for entry in entries:
        if entry["combination"] == combination:
            return entry["payout"]
    return 0.0


def _make_combination_key(numbers: list, ordered: bool = False) -> str:
    """馬番リストから組合せキーを生成
    ordered=False: ソート済みダッシュ区切り (例: "3-5-8")  馬連・ワイド・3連複用
    ordered=True: 順序保持ダッシュ区切り (例: "3-5-8")  馬単・3連単用
    """
    strs = [str(int(n)) for n in numbers]
    if ordered:
        return "-".join(strs)
    else:
        return "-".join(sorted(strs, key=int))


# ============================================================
# Plackett-Luce 確率推定
# ============================================================

def _plackett_luce_prob(win_probs: np.ndarray, indices: list, top_k: int) -> float:
    """Plackett-Luce モデルで指定馬が上位 top_k に入る確率を推定
    indices: 対象馬のインデックスリスト（win_probs配列内）
    top_k: 上位何着までに入るか（2=1-2着, 3=1-2-3着）
    順序なし（組合せ）の確率を返す
    """
    n = len(win_probs)
    target_set = set(indices)

    # 対象馬が top_k 枠に入る全順列を考慮
    total_prob = 0.0
    for perm in permutations(indices):
        # この順列での確率を計算
        prob = 1.0
        remaining = np.sum(win_probs)
        for pos in range(top_k):
            if pos < len(perm):
                idx = perm[pos]
            else:
                break
            prob *= win_probs[idx] / remaining
            remaining -= win_probs[idx]
        # 残りの top_k - len(indices) 枠は対象外の馬が埋める
        # ここでは対象馬だけで top_k 枠を埋める場合のみ
        if len(perm) == top_k:
            total_prob += prob

    return total_prob


def _pl_ordered_prob(win_probs: np.ndarray, indices: list) -> float:
    """Plackett-Luce で指定順序での確率（馬単・3連単用）
    indices[0]が1着, indices[1]が2着, ... の確率
    """
    prob = 1.0
    remaining = np.sum(win_probs)
    for idx in indices:
        prob *= win_probs[idx] / remaining
        remaining -= win_probs[idx]
    return prob


def _pl_unordered_top_k(win_probs: np.ndarray, indices: list, top_k: int) -> float:
    """Plackett-Luce で指定馬群が上位 top_k を占める確率（順序不問）
    馬連(top_k=2), ワイド(top_k=3のうち2頭), 3連複(top_k=3) 用
    """
    # indices の全順列を列挙し、各順列の確率を合算
    total = 0.0
    for perm in permutations(indices):
        total += _pl_ordered_prob(win_probs, list(perm))
    return total


def _pl_wide_prob(win_probs: np.ndarray, idx_a: int, idx_b: int, n_horses: int) -> float:
    """ワイド用: 2頭が共に3着以内に入る確率（Plackett-Luce）
    3着の残り1枠を全ての他馬で場合分けして合算
    """
    other_indices = [i for i in range(n_horses) if i != idx_a and i != idx_b]
    total = 0.0
    for other in other_indices:
        trio = [idx_a, idx_b, other]
        total += _pl_unordered_top_k(win_probs, trio, 3)
    return total


# ============================================================
# バックテスト
# ============================================================

def _empty_bet_stats():
    return {"count": 0, "invested": 0, "returned": 0, "hits": 0}


def _predict_with_model(val_df: pd.DataFrame, model_dir: Path) -> tuple:
    """指定ディレクトリのモデルで予測を付与。(val_df, has_win_model, has_ranking, calibrator) を返す"""
    from src.model.trainer import (
        load_calibrator, calibrate_probs,
        load_isotonic_calibrator, calibrate_isotonic,
    )

    binary_model_path = model_dir / "binary_model.txt"
    if not binary_model_path.exists():
        return None, False, False, None

    model = lgb.Booster(model_file=str(binary_model_path))

    available = [c for c in FEATURE_COLUMNS if c in val_df.columns]
    X_val = val_df[available].values.astype(np.float32)
    raw_probs = model.predict(X_val)

    # Platt Scaling キャリブレーション（利用可能な場合）
    calibrator = load_calibrator(model_dir)
    if calibrator is not None:
        val_df["pred_prob"] = calibrate_probs(raw_probs, calibrator)
    else:
        val_df["pred_prob"] = raw_probs

    # 勝率直接推定モデル（利用可能な場合）
    win_model_path = model_dir / "win_model.txt"
    has_win_model = False
    if win_model_path.exists():
        win_model = lgb.Booster(model_file=str(win_model_path))
        win_raw = win_model.predict(X_val)
        win_iso = load_isotonic_calibrator("win_isotonic", model_dir)
        if win_iso is not None:
            val_df["win_prob_direct"] = calibrate_isotonic(win_raw, win_iso)
        else:
            val_df["win_prob_direct"] = win_raw
        has_win_model = True

    # ランキングモデル（アンサンブル用）
    ranking_model_path = model_dir / "ranking_model.txt"
    has_ranking = False
    if ranking_model_path.exists():
        ranking_model = lgb.Booster(model_file=str(ranking_model_path))
        val_df["rank_score"] = ranking_model.predict(X_val)
        has_ranking = True

    return val_df, has_win_model, has_ranking, calibrator


def prepare_backtest_data(input_path: str = None, model_dir: Path = None,
                          returns_path: str = None,
                          val_start: str = "2025-01-01",
                          val_end: str = None,
                          exclude_newcomer: bool = True,
                          exclude_hurdle: bool = True,
                          min_entries: int = 6,
                          surface_split: bool = False,
                          ) -> dict | None:
    """バックテストのデータ準備（CSV読込・モデル予測・キャリブレーション）

    Args:
        surface_split: True の場合、芝・ダート別モデルで予測

    Returns:
        dict with keys: val_df, returns, has_win_model, has_ranking, calibrator
        or None if error
    """
    input_path = input_path or str(PROCESSED_DIR / "features.csv")
    model_dir = model_dir or MODEL_DIR
    returns_path = Path(returns_path) if returns_path else RAW_DIR / "returns.csv"

    df = pd.read_csv(input_path, dtype={"race_id": str, "horse_id": str})
    df["race_date"] = pd.to_datetime(df["race_date"])

    # レースフィルタリング
    pre_filter = len(df)
    if exclude_newcomer and "race_class_code" in df.columns:
        df = df[df["race_class_code"] != 1]
    if exclude_hurdle and "surface_code" in df.columns:
        df = df[df["surface_code"] != 2]
    if min_entries > 0 and "num_entries" in df.columns:
        df = df[df["num_entries"] >= min_entries]
    filtered = pre_filter - len(df)
    if filtered > 0:
        print(f"  [フィルタ] {filtered}行除外（新馬/障害/少頭数）→ {len(df)}行")

    val_df = df[df["race_date"] >= pd.Timestamp(val_start)].copy()
    if val_end:
        val_df = val_df[val_df["race_date"] < pd.Timestamp(val_end)].copy()
    if len(val_df) == 0:
        print("[ERROR] 検証期間のデータがありません")
        return None

    if surface_split and "surface_code" in val_df.columns:
        # 芝・ダート別モデルで予測し結合
        turf_dir = model_dir / "turf"
        dirt_dir = model_dir / "dirt"

        if not (turf_dir / "binary_model.txt").exists():
            print(f"[ERROR] 芝モデルが見つかりません: {turf_dir}")
            return None
        if not (dirt_dir / "binary_model.txt").exists():
            print(f"[ERROR] ダートモデルが見つかりません: {dirt_dir}")
            return None

        turf_df = val_df[val_df["surface_code"] == 0].copy()
        dirt_df = val_df[val_df["surface_code"] == 1].copy()

        print(f"  [芝・ダート分離予測] 芝={len(turf_df)}行, ダート={len(dirt_df)}行")

        parts = []
        has_win_model = True
        has_ranking = True
        calibrator = None

        if len(turf_df) > 0:
            turf_result, t_win, t_rank, t_cal = _predict_with_model(turf_df, turf_dir)
            if turf_result is None:
                print("[ERROR] 芝モデルの予測に失敗")
                return None
            parts.append(turf_result)
            has_win_model &= t_win
            has_ranking &= t_rank

        if len(dirt_df) > 0:
            dirt_result, d_win, d_rank, d_cal = _predict_with_model(dirt_df, dirt_dir)
            if dirt_result is None:
                print("[ERROR] ダートモデルの予測に失敗")
                return None
            parts.append(dirt_result)
            has_win_model &= d_win
            has_ranking &= d_rank

        val_df = pd.concat(parts, ignore_index=False).sort_values(["race_date", "race_id", "horse_number"])
    else:
        # 統合モデルで予測（従来動作）
        binary_model_path = model_dir / "binary_model.txt"
        if not binary_model_path.exists():
            print(f"[ERROR] モデルが見つかりません: {binary_model_path}")
            return None

        val_df, has_win_model, has_ranking, calibrator = _predict_with_model(val_df, model_dir)
        if val_df is None:
            print(f"[ERROR] モデルの予測に失敗")
            return None

    # 払い戻しデータ読み込み
    returns = _load_returns(returns_path)

    return {
        "val_df": val_df,
        "full_df": df,
        "returns": returns,
        "has_win_model": has_win_model,
        "has_ranking": has_ranking,
        "calibrator": calibrator,
        "val_start": val_start,
        "val_end": val_end,
    }


def simulate_bets(prepared: dict,
                  ev_threshold: float = 0.0,
                  bet_threshold: float = 0.0,
                  top_n: int = 3,
                  temperature: float = 1.0,
                  confidence_min: float = 0.0,
                  odds_min: float = 0.0,
                  odds_max: float = 0.0,
                  axis_flow: bool = False,
                  kelly_fraction: float = 0.0,
                  race_ids: set = None,
                  skip_classes: list = None,
                  quinella_top_n: int = 0,
                  wide_top_n: int = 0,
                  ) -> dict:
    """賭けシミュレーション（prepare済みデータに対して実行）

    Args:
        prepared: prepare_backtest_data() の戻り値
        race_ids: 対象レースIDのセット（Noneで全レース）
        skip_classes: スキップするクラスコードのリスト（例: [5] で3勝クラスを除外）
        quinella_top_n: 馬連用top_n（0でtop_nを使用）
        wide_top_n: ワイド用top_n（0でtop_nを使用）
        他のパラメータはrun_backtestと同一
    """
    val_df = prepared["val_df"]
    returns = prepared["returns"]
    has_win_model = prepared["has_win_model"]
    has_ranking = prepared["has_ranking"]
    calibrator = prepared["calibrator"]
    val_start = prepared["val_start"]

    if race_ids is not None:
        val_df = val_df[val_df["race_id"].isin(race_ids)]

    results = {
        "val_start": val_start,
        "ev_threshold": ev_threshold,
        "races": 0,
        "prediction_accuracy": {"top1_hit": 0, "top3_hit": 0, "total": 0},
        "race_details": [],
        "monthly": {},
    }
    for bt in BET_TYPES:
        results[f"bets_{bt}"] = _empty_bet_stats()

    for race_id, race_group in val_df.groupby("race_id"):
        if len(race_group) < 4:
            continue

        # Phase 5-4: クラス別スキップ
        if skip_classes and "race_class_code" in race_group.columns:
            race_class = race_group.iloc[0]["race_class_code"]
            if int(race_class) in skip_classes:
                results["races_class_skipped"] = results.get("races_class_skipped", 0) + 1
                continue

        # Phase 4-4: アンサンブルソート（ranking + binary）
        if has_ranking and "rank_score" in race_group.columns:
            # ランキングスコアをsoftmaxで正規化 → binaryと平均して安定化
            rs = race_group["rank_score"].values
            exp_rs = np.exp(rs - rs.max())
            rank_probs = exp_rs / exp_rs.sum()
            bp = np.clip(race_group["pred_prob"].values, 1e-6, 1 - 1e-6)
            bp_norm = bp / bp.sum()
            # 加重平均: binary 0.6 + ranking 0.4（binaryの方が確率として信頼性が高い）
            ensemble_sort = 0.6 * bp_norm + 0.4 * rank_probs
            race_group = race_group.copy()
            race_group["_ensemble_sort"] = ensemble_sort
            race_group = race_group.sort_values("_ensemble_sort", ascending=False)
        else:
            race_group = race_group.sort_values("pred_prob", ascending=False)

        results["races"] += 1

        # 月別集計用
        month_key = str(race_group.iloc[0]["race_date"])[:7]
        if month_key not in results["monthly"]:
            results["monthly"][month_key] = {bt: _empty_bet_stats() for bt in BET_TYPES}

        # レース内の勝率を計算
        # Note: win_probsはキャリブレーション済みモデルのみ使用（アンサンブルはソートのみ）
        pred_probs = np.clip(race_group["pred_prob"].values, 1e-6, 1 - 1e-6)
        if has_win_model and "win_prob_direct" in race_group.columns:
            # Phase 3: 勝率直接推定モデルから win_prob を算出（温度不要）
            wprobs = np.clip(race_group["win_prob_direct"].values, 1e-6, 1.0)
            win_probs = wprobs / wprobs.sum()
        elif calibrator is not None:
            win_probs = pred_probs / pred_probs.sum()
        else:
            logits = np.log(pred_probs / (1 - pred_probs))
            scores = logits / temperature
            exp_s = np.exp(scores - scores.max())
            win_probs = exp_s / exp_s.sum()
        race_group = race_group.copy()
        race_group["win_prob"] = win_probs

        # 予測精度
        top1_pred = race_group.iloc[0]
        results["prediction_accuracy"]["total"] += 1
        if top1_pred.get("finish_position", 99) == 1:
            results["prediction_accuracy"]["top1_hit"] += 1

        top3_pred = race_group.head(3)
        top3_actual = (top3_pred["finish_position"] <= 3).sum()
        results["prediction_accuracy"]["top3_hit"] += top3_actual

        # 確信度フィルタ: Top1-Top2 の win_prob 差
        if confidence_min > 0 and len(race_group) >= 2:
            gap = win_probs[0] - win_probs[1]
            if gap < confidence_min:
                # レース詳細のみ記録して次へ（賭けはスキップ）
                detail = {
                    "race_id": race_id,
                    "date": str(race_group.iloc[0].get("race_date", "")),
                    "predictions": [],
                    "skipped": True,
                }
                for _, horse in race_group.head(5).iterrows():
                    detail["predictions"].append({
                        "horse": horse.get("horse_name", ""),
                        "pred_prob": round(float(horse["pred_prob"]), 3),
                        "win_prob": round(float(horse["win_prob"]), 3),
                        "actual_finish": int(horse.get("finish_position", 0)),
                        "odds": float(horse.get("win_odds", 0)),
                        "ev_win": round(float(horse["win_prob"] * horse.get("win_odds", 0)), 2),
                    })
                results["race_details"].append(detail)
                results["races_skipped"] = results.get("races_skipped", 0) + 1
                continue

        # 馬番と着順の取得（券種別top_nの最大値を使用）
        effective_top = max(top_n, quinella_top_n, wide_top_n)
        top_horses = race_group.head(effective_top)
        horse_numbers = top_horses["horse_number"].values if "horse_number" in top_horses.columns else []
        finish_positions = top_horses["finish_position"].values

        # --- 単勝シミュレーション ---
        for i, (_, horse) in enumerate(top_horses.head(top_n).iterrows()):
            odds = horse.get("win_odds", 0)
            if odds <= 0:
                continue
            # オッズ帯フィルタ
            if odds_min > 0 and odds < odds_min:
                continue
            if odds_max > 0 and odds > odds_max:
                continue
            ev_win = horse["win_prob"] * odds
            if ev_threshold > 0 and ev_win < ev_threshold:
                continue
            if horse["pred_prob"] < bet_threshold:
                continue

            # Kelly基準ベットサイジング
            bet_amount = 100
            if kelly_fraction > 0:
                kf = _kelly_fraction_calc(horse["win_prob"], odds) * kelly_fraction
                if kf <= 0:
                    continue  # エッジなし → スキップ
                bet_amount = max(100, round(kf * KELLY_BANKROLL / 100) * 100)

            hit = horse.get("finish_position", 99) == 1
            payout = odds * bet_amount if hit else 0
            _record_bet(results, month_key, "win", bet_amount, hit, payout)

        # --- 複勝シミュレーション ---
        for i, (_, horse) in enumerate(top_horses.head(top_n).iterrows()):
            pred_place = horse["pred_prob"]
            odds = horse.get("win_odds", 0)
            if odds <= 0:
                continue
            # オッズ帯フィルタ
            if odds_min > 0 and odds < odds_min:
                continue
            if odds_max > 0 and odds > odds_max:
                continue
            n_entries = len(race_group)
            est_place_odds = _estimate_place_odds(odds, n_entries)
            ev_place = pred_place * est_place_odds
            if ev_threshold > 0 and ev_place < ev_threshold:
                continue
            if pred_place < bet_threshold:
                continue

            # Kelly基準ベットサイジング
            bet_amount = 100
            if kelly_fraction > 0:
                kf = _kelly_fraction_calc(pred_place, est_place_odds) * kelly_fraction
                if kf <= 0:
                    continue  # エッジなし → スキップ
                bet_amount = max(100, round(kf * KELLY_BANKROLL / 100) * 100)

            hit = horse.get("finish_position", 99) <= 3
            payout = 0
            if hit:
                actual_payout = _get_payout(
                    returns, race_id, "place",
                    str(int(horse.get("horse_number", 0))))
                payout = (actual_payout * bet_amount / 100) if actual_payout > 0 else est_place_odds * bet_amount

            _record_bet(results, month_key, "place", bet_amount, hit, payout)

        # --- 馬連シミュレーション（top_n から2頭組合せ、順不同） ---
        q_top = quinella_top_n if quinella_top_n > 0 else top_n
        if len(horse_numbers) >= 2:
            for idx_pair in combinations(range(min(q_top, len(top_horses))), 2):
                i, j = idx_pair
                h_i = top_horses.iloc[i]
                h_j = top_horses.iloc[j]
                nums = [h_i.get("horse_number", 0), h_j.get("horse_number", 0)]
                if 0 in nums:
                    continue
                combo_key = _make_combination_key(nums, ordered=False)
                actual_1st = race_group[race_group["finish_position"] == 1]
                actual_2nd = race_group[race_group["finish_position"] == 2]
                if len(actual_1st) == 0 or len(actual_2nd) == 0:
                    continue
                actual_combo = _make_combination_key(
                    [actual_1st.iloc[0].get("horse_number", 0),
                     actual_2nd.iloc[0].get("horse_number", 0)], ordered=False)
                hit = combo_key == actual_combo
                payout = 0
                if hit:
                    payout = _get_payout(returns, race_id, "quinella", combo_key)
                _record_bet(results, month_key, "quinella", 100, hit, payout)

        # --- ワイド シミュレーション（top_n から2頭組合せ、3着以内） ---
        w_top = wide_top_n if wide_top_n > 0 else top_n
        if len(horse_numbers) >= 2:
            # 実際の3着以内馬番セット
            actual_top3 = set(
                race_group[race_group["finish_position"] <= 3]["horse_number"].astype(int).tolist()
            ) if "horse_number" in race_group.columns else set()

            for idx_pair in combinations(range(min(w_top, len(top_horses))), 2):
                i, j = idx_pair
                h_i = top_horses.iloc[i]
                h_j = top_horses.iloc[j]
                nums = [h_i.get("horse_number", 0), h_j.get("horse_number", 0)]
                if 0 in nums:
                    continue
                combo_key = _make_combination_key(nums, ordered=False)
                hit = int(nums[0]) in actual_top3 and int(nums[1]) in actual_top3
                payout = 0
                if hit:
                    payout = _get_payout(returns, race_id, "wide", combo_key)
                _record_bet(results, month_key, "wide", 100, hit, payout)

        # --- 馬単シミュレーション ---
        if len(horse_numbers) >= 2:
            actual_1st = race_group[race_group["finish_position"] == 1]
            actual_2nd = race_group[race_group["finish_position"] == 2]
            if len(actual_1st) > 0 and len(actual_2nd) > 0:
                actual_combo = _make_combination_key(
                    [actual_1st.iloc[0].get("horse_number", 0),
                     actual_2nd.iloc[0].get("horse_number", 0)], ordered=True)

                if axis_flow:
                    # 軸流し: Top1=1着固定 → Top2~TopNへ流し（N-1点）
                    axis_num = top_horses.iloc[0].get("horse_number", 0)
                    if axis_num != 0:
                        for j in range(1, min(top_n, len(top_horses))):
                            target_num = top_horses.iloc[j].get("horse_number", 0)
                            if target_num == 0:
                                continue
                            combo_key = _make_combination_key([axis_num, target_num], ordered=True)
                            hit = combo_key == actual_combo
                            payout = _get_payout(returns, race_id, "exacta", combo_key) if hit else 0
                            _record_bet(results, month_key, "exacta", 100, hit, payout)
                else:
                    # ボックス: 全順列（N*(N-1)点）
                    for idx_pair in permutations(range(min(top_n, len(top_horses))), 2):
                        i, j = idx_pair
                        h_i = top_horses.iloc[i]
                        h_j = top_horses.iloc[j]
                        nums = [h_i.get("horse_number", 0), h_j.get("horse_number", 0)]
                        if 0 in nums:
                            continue
                        combo_key = _make_combination_key(nums, ordered=True)
                        hit = combo_key == actual_combo
                        payout = _get_payout(returns, race_id, "exacta", combo_key) if hit else 0
                        _record_bet(results, month_key, "exacta", 100, hit, payout)

        # --- 3連複シミュレーション（top_n=3 の場合1点、順不同） ---
        if len(horse_numbers) >= 3:
            for idx_trio in combinations(range(min(top_n, len(top_horses))), 3):
                i, j, k = idx_trio
                nums = [
                    top_horses.iloc[i].get("horse_number", 0),
                    top_horses.iloc[j].get("horse_number", 0),
                    top_horses.iloc[k].get("horse_number", 0),
                ]
                if 0 in nums:
                    continue
                combo_key = _make_combination_key(nums, ordered=False)
                actual_top3_horses = race_group[race_group["finish_position"] <= 3]
                if len(actual_top3_horses) < 3:
                    continue
                actual_combo = _make_combination_key(
                    actual_top3_horses["horse_number"].astype(int).tolist()[:3],
                    ordered=False)
                hit = combo_key == actual_combo
                payout = 0
                if hit:
                    payout = _get_payout(returns, race_id, "trio", combo_key)
                _record_bet(results, month_key, "trio", 100, hit, payout)

        # --- 3連単シミュレーション ---
        if len(horse_numbers) >= 3:
            actual_1st = race_group[race_group["finish_position"] == 1]
            actual_2nd = race_group[race_group["finish_position"] == 2]
            actual_3rd = race_group[race_group["finish_position"] == 3]
            if len(actual_1st) > 0 and len(actual_2nd) > 0 and len(actual_3rd) > 0:
                actual_combo = _make_combination_key(
                    [actual_1st.iloc[0].get("horse_number", 0),
                     actual_2nd.iloc[0].get("horse_number", 0),
                     actual_3rd.iloc[0].get("horse_number", 0)],
                    ordered=True)

                if axis_flow:
                    # 軸流し: Top1=1着固定 → Top2~TopNの2着3着順列（(N-1)*(N-2)点）
                    axis_num = top_horses.iloc[0].get("horse_number", 0)
                    if axis_num != 0:
                        others = []
                        for j in range(1, min(top_n, len(top_horses))):
                            n = top_horses.iloc[j].get("horse_number", 0)
                            if n != 0:
                                others.append(n)
                        for perm in permutations(others, 2):
                            combo_key = _make_combination_key([axis_num, perm[0], perm[1]], ordered=True)
                            hit = combo_key == actual_combo
                            payout = _get_payout(returns, race_id, "trifecta", combo_key) if hit else 0
                            _record_bet(results, month_key, "trifecta", 100, hit, payout)
                else:
                    # ボックス: 全順列
                    for idx_trio in combinations(range(min(top_n, len(top_horses))), 3):
                        for perm in permutations(idx_trio):
                            i, j, k = perm
                            nums = [
                                top_horses.iloc[i].get("horse_number", 0),
                                top_horses.iloc[j].get("horse_number", 0),
                                top_horses.iloc[k].get("horse_number", 0),
                            ]
                            if 0 in nums:
                                continue
                            combo_key = _make_combination_key(nums, ordered=True)
                            hit = combo_key == actual_combo
                            payout = _get_payout(returns, race_id, "trifecta", combo_key) if hit else 0
                            _record_bet(results, month_key, "trifecta", 100, hit, payout)

        # レース詳細（上位5頭）
        detail = {
            "race_id": race_id,
            "date": str(race_group.iloc[0].get("race_date", "")),
            "predictions": [],
        }
        for _, horse in race_group.head(5).iterrows():
            detail["predictions"].append({
                "horse": horse.get("horse_name", ""),
                "pred_prob": round(float(horse["pred_prob"]), 3),
                "win_prob": round(float(horse["win_prob"]), 3),
                "actual_finish": int(horse.get("finish_position", 0)),
                "odds": float(horse.get("win_odds", 0)),
                "ev_win": round(float(horse["win_prob"] * horse.get("win_odds", 0)), 2),
            })
        results["race_details"].append(detail)

    return results


def run_backtest(input_path: str = None, model_dir: Path = None,
                 returns_path: str = None,
                 val_start: str = "2025-01-01",
                 val_end: str = None,
                 ev_threshold: float = 0.0,
                 bet_threshold: float = 0.0,
                 top_n: int = 3,
                 temperature: float = 1.0,
                 exclude_newcomer: bool = True,
                 exclude_hurdle: bool = True,
                 min_entries: int = 6,
                 confidence_min: float = 0.0,
                 odds_min: float = 0.0,
                 odds_max: float = 0.0,
                 axis_flow: bool = False,
                 kelly_fraction: float = 0.0,
                 skip_classes: list = None,
                 quinella_top_n: int = 0,
                 wide_top_n: int = 0,
                 _prepared: dict = None,
                 surface_split: bool = False,
                 ) -> dict:
    """全券種対応EVフィルタ付きバックテスト（後方互換ラッパー）

    _prepared を渡すとデータ準備をスキップしてシミュレーションのみ実行。
    """
    if _prepared is None:
        _prepared = prepare_backtest_data(
            input_path=input_path, model_dir=model_dir,
            returns_path=returns_path, val_start=val_start, val_end=val_end,
            exclude_newcomer=exclude_newcomer, exclude_hurdle=exclude_hurdle,
            min_entries=min_entries, surface_split=surface_split,
        )
        if _prepared is None:
            return {}

    return simulate_bets(
        _prepared,
        ev_threshold=ev_threshold, bet_threshold=bet_threshold,
        top_n=top_n, temperature=temperature,
        confidence_min=confidence_min, odds_min=odds_min,
        odds_max=odds_max, axis_flow=axis_flow,
        kelly_fraction=kelly_fraction,
        skip_classes=skip_classes,
        quinella_top_n=quinella_top_n,
        wide_top_n=wide_top_n,
    )


def _record_bet(results: dict, month_key: str, bet_type: str,
                invested: int, hit: bool, payout: float):
    """共通の賭け記録ヘルパー"""
    key = f"bets_{bet_type}"
    results[key]["count"] += 1
    results[key]["invested"] += invested
    results["monthly"][month_key][bet_type]["count"] += 1
    results["monthly"][month_key][bet_type]["invested"] += invested
    if hit and payout > 0:
        results[key]["returned"] += payout
        results[key]["hits"] += 1
        results["monthly"][month_key][bet_type]["returned"] += payout
        results["monthly"][month_key][bet_type]["hits"] += 1


