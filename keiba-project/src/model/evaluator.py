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

# 複勝オッズ推定回帰係数（returns.csv 30,378件から算出、R²=0.80、MAE=0.79）
# place_odds = PLACE_ODDS_SLOPE * win_odds + PLACE_ODDS_INTERCEPT
PLACE_ODDS_SLOPE = 0.1414
PLACE_ODDS_INTERCEPT = 1.1475

# 全券種キー（内部用）
BET_TYPES = ["win", "place", "quinella", "wide", "exacta", "trio", "trifecta"]
BET_LABELS = {
    "win": "単勝", "place": "複勝", "quinella": "馬連", "wide": "ワイド",
    "exacta": "馬単", "trio": "3連複", "trifecta": "3連単",
}

# Kelly基準のデフォルト銀行残高（ベットサイズ計算用）
KELLY_BANKROLL = 100000


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
                 # Phase 1 フィルタ
                 confidence_min: float = 0.0,
                 odds_min: float = 0.0,
                 odds_max: float = 0.0,
                 axis_flow: bool = False,
                 # Phase 3: Kelly基準
                 kelly_fraction: float = 0.0,
                 ) -> dict:
    """
    全券種対応EVフィルタ付きバックテスト

    券種: 単勝・複勝・馬連・ワイド・馬単・3連複・3連単
    確率推定: Plackett-Luce モデル

    Args:
        ev_threshold: EV がこの値以上で購入（0=フィルタなし、単勝・複勝のみ適用）
        bet_threshold: 予測確率がこの値以上で購入対象
        returns_path: returns.csv パス（実払い戻し使用）
        val_end: 検証終了日（指定しない場合はデータ末尾まで）
        temperature: ソフトマックス温度パラメータ（logitスケール、低いほど鋭い分布）
        confidence_min: Top1-Top2のwin_prob差がこの値以上のレースのみ購入（0=無効）
        odds_min: 単勝・複勝の最低オッズ（0=無効）
        odds_max: 単勝・複勝の最高オッズ（0=無効）
        axis_flow: True=馬単・3連単をTop1軸流しに変更（点数削減）
        kelly_fraction: Kelly基準の割合（0=均一賭け、0.25=1/4 Kelly推奨）
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
        return {}

    binary_model_path = model_dir / "binary_model.txt"
    if not binary_model_path.exists():
        print(f"[ERROR] モデルが見つかりません: {binary_model_path}")
        return {}

    model = lgb.Booster(model_file=str(binary_model_path))

    available = [c for c in FEATURE_COLUMNS if c in val_df.columns]
    X_val = val_df[available].values.astype(np.float32)
    raw_probs = model.predict(X_val)

    # Platt Scaling キャリブレーション（利用可能な場合）
    from src.model.trainer import load_calibrator, calibrate_probs
    calibrator = load_calibrator(model_dir)
    if calibrator is not None:
        val_df["pred_prob"] = calibrate_probs(raw_probs, calibrator)
    else:
        val_df["pred_prob"] = raw_probs

    # Phase 3: 勝率直接推定モデル（利用可能な場合）
    win_model_path = model_dir / "win_model.txt"
    has_win_model = False
    if win_model_path.exists():
        from src.model.trainer import load_isotonic_calibrator, calibrate_isotonic
        win_model = lgb.Booster(model_file=str(win_model_path))
        win_raw = win_model.predict(X_val)
        win_iso = load_isotonic_calibrator("win_isotonic", model_dir)
        if win_iso is not None:
            val_df["win_prob_direct"] = calibrate_isotonic(win_raw, win_iso)
        else:
            val_df["win_prob_direct"] = win_raw
        has_win_model = True

    # 払い戻しデータ読み込み
    returns = _load_returns(returns_path)

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

        race_group = race_group.sort_values("pred_prob", ascending=False)
        results["races"] += 1

        # 月別集計用
        month_key = str(race_group.iloc[0]["race_date"])[:7]
        if month_key not in results["monthly"]:
            results["monthly"][month_key] = {bt: _empty_bet_stats() for bt in BET_TYPES}

        # レース内の勝率を計算
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

        # 馬番と着順の取得
        top_horses = race_group.head(top_n)
        horse_numbers = top_horses["horse_number"].values if "horse_number" in top_horses.columns else []
        finish_positions = top_horses["finish_position"].values

        # --- 単勝シミュレーション ---
        for i, (_, horse) in enumerate(top_horses.iterrows()):
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
        for i, (_, horse) in enumerate(top_horses.iterrows()):
            pred_place = horse["pred_prob"]
            odds = horse.get("win_odds", 0)
            if odds <= 0:
                continue
            # オッズ帯フィルタ
            if odds_min > 0 and odds < odds_min:
                continue
            if odds_max > 0 and odds > odds_max:
                continue
            est_place_odds = max(PLACE_ODDS_SLOPE * odds + PLACE_ODDS_INTERCEPT, 1.0)
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
        if len(horse_numbers) >= 2:
            for idx_pair in combinations(range(min(top_n, len(top_horses))), 2):
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
        if len(horse_numbers) >= 2:
            # 実際の3着以内馬番セット
            actual_top3 = set(
                race_group[race_group["finish_position"] <= 3]["horse_number"].astype(int).tolist()
            ) if "horse_number" in race_group.columns else set()

            for idx_pair in combinations(range(min(top_n, len(top_horses))), 2):
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


# ============================================================
# EV閾値比較
# ============================================================

def compare_ev_thresholds(input_path: str = None, model_dir: Path = None,
                          returns_path: str = None,
                          val_start: str = "2025-01-01",
                          val_end: str = None,
                          thresholds: list = None,
                          temperature: float = 1.0,
                          **filter_kwargs) -> list:
    """複数のEV閾値でバックテストを実行し比較（EVフィルタは単勝・複勝のみ）"""
    thresholds = thresholds or [0.0, 0.8, 1.0, 1.2, 1.5, 2.0]
    comparison = []

    for t in thresholds:
        res = run_backtest(
            input_path=input_path, model_dir=model_dir,
            returns_path=returns_path, val_start=val_start,
            val_end=val_end,
            ev_threshold=t, top_n=3, temperature=temperature,
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
    """ソフトマックス温度パラメータの最適化（グリッドサーチ、logitスケール）"""
    temperatures = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0]
    best = {"temperature": 1.0, "win_roi": 0, "place_roi": 0}

    period = f"{val_start}\u301c{val_end}" if val_end else f"{val_start}\u301c"
    print(f"\n  [温度パラメータ最適化] EV閾値={ev_threshold} 期間={period}")
    print(f"    {'温度':>6s} {'単勝回数':>8s} {'単勝的中率':>10s} {'単勝回収率':>10s} "
          f"{'複勝回数':>8s} {'複勝的中率':>10s} {'複勝回収率':>10s}")
    print(f"    {'─' * 70}")

    for temp in temperatures:
        res = run_backtest(
            input_path=input_path, model_dir=model_dir,
            returns_path=returns_path, val_start=val_start,
            val_end=val_end,
            ev_threshold=ev_threshold, top_n=3, temperature=temp,
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


# ============================================================
# レポート表示
# ============================================================

def print_backtest_report(results: dict):
    """バックテスト結果を表示（全券種対応）"""
    if not results:
        return

    races = results["races"]
    ev_t = results.get("ev_threshold", 0)
    print(f"\n  対象レース数: {races}")
    if ev_t > 0:
        print(f"  EVフィルタ: >= {ev_t}")

    pa = results["prediction_accuracy"]
    total = pa["total"]
    if total > 0:
        print(f"\n  [予測精度]")
        print(f"    1着的中率:   {pa['top1_hit']}/{total} = {pa['top1_hit']/total:.1%}")
        print(f"    3着内的中数: {pa['top3_hit']}/{total*3} = {pa['top3_hit']/(total*3):.1%}")

    # 全券種のシミュレーション結果
    for bt in BET_TYPES:
        b = results.get(f"bets_{bt}", {})
        label = BET_LABELS.get(bt, bt)
        if b.get("count", 0) > 0:
            roi = b["returned"] / b["invested"] * 100 if b["invested"] > 0 else 0
            mark = " ★" if roi >= 100 else ""
            print(f"\n  [{label}シミュレーション]")
            print(f"    購入回数: {b['count']}")
            print(f"    的中回数: {b['hits']} ({b['hits']/b['count']:.1%})")
            print(f"    投資額:   ¥{b['invested']:,}")
            print(f"    回収額:   ¥{b['returned']:,.0f}")
            print(f"    回収率:   {roi:.1f}%{mark}")
        else:
            print(f"\n  [{label}] 購入対象なし")

    # 月別
    monthly = results.get("monthly", {})
    if monthly:
        print(f"\n  [月別回収率]")
        header = f"    {'月':8s}"
        for bt in BET_TYPES:
            header += f" {BET_LABELS[bt]:>6s}"
        print(header)
        for month in sorted(monthly.keys()):
            m = monthly[month]
            row = f"    {month:8s}"
            for bt in BET_TYPES:
                b = m.get(bt, {})
                if b.get("count", 0) > 0 and b.get("invested", 0) > 0:
                    roi = b["returned"] / b["invested"] * 100
                    row += f" {roi:>5.0f}%"
                else:
                    row += f" {'---':>6s}"
            print(row)

    details = results.get("race_details", [])
    if details:
        print(f"\n  [サンプルレース（最新5件）]")
        for detail in details[-5:]:
            print(f"\n    {detail['race_id']} ({detail['date'][:10]})")
            for pred in detail["predictions"][:3]:
                hit = "◎" if pred["actual_finish"] <= 3 else "×"
                print(f"      {hit} {pred['horse']:12s} "
                      f"top3={pred['pred_prob']:.1%} "
                      f"win={pred['win_prob']:.1%} "
                      f"EV={pred['ev_win']:.2f} "
                      f"実際{pred['actual_finish']}着 "
                      f"オッズ{pred['odds']:.1f}")


def print_ev_comparison(comparison: list):
    """EV閾値比較を表示（全券種対応）"""
    if not comparison:
        return
    print(f"\n  [EV閾値比較 — 全券種]")

    # 単勝・複勝（EVフィルタ対象）
    print(f"\n  ■ 単勝・複勝（EVフィルタ対象）")
    print(f"    {'EV閾値':>7s} {'単勝回数':>8s} {'単勝的中':>8s} {'単勝回収率':>10s} "
          f"{'複勝回数':>8s} {'複勝的中':>8s} {'複勝回収率':>10s}")
    print(f"    {'─' * 65}")
    for c in comparison:
        t_str = "なし" if c["ev_threshold"] == 0 else f">={c['ev_threshold']}"
        w_roi_str = f"{c['win_roi']:.1f}%"
        p_roi_str = f"{c['place_roi']:.1f}%"
        mark_w = " ★" if c["win_roi"] >= 100 else ""
        mark_p = " ★" if c["place_roi"] >= 100 else ""
        print(f"    {t_str:>7s} {c['win_bets']:>8d} {c['win_hit_rate']:>7.1f}% "
              f"{w_roi_str:>10s}{mark_w}  "
              f"{c['place_bets']:>7d} {c['place_hit_rate']:>7.1f}% "
              f"{p_roi_str:>10s}{mark_p}")

    # 組合せ券種（EVフィルタなし = threshold=0 の結果のみ表示）
    base = None
    for c in comparison:
        if c["ev_threshold"] == 0:
            base = c
            break
    if base is None and comparison:
        base = comparison[0]
    if base:
        multi_types = ["quinella", "wide", "exacta", "trio", "trifecta"]
        print(f"\n  ■ 組合せ券種（Top3全組合せ）")
        print(f"    {'券種':8s} {'購入点数':>8s} {'的中率':>8s} {'回収率':>10s}")
        print(f"    {'─' * 40}")
        for bt in multi_types:
            label = BET_LABELS[bt]
            bets = base.get(f"{bt}_bets", 0)
            hits = base.get(f"{bt}_hits", 0)
            roi = base.get(f"{bt}_roi", 0)
            hit_rate = base.get(f"{bt}_hit_rate", 0)
            if bets > 0:
                mark = " ★" if roi >= 100 else ""
                print(f"    {label:8s} {bets:>8d} {hit_rate:>7.1f}% {roi:>9.1f}%{mark}")
            else:
                print(f"    {label:8s} {'---':>8s} {'---':>8s} {'---':>10s}")


def explore_strategies(input_path: str = None, model_dir: Path = None,
                       returns_path: str = None,
                       val_start: str = "2025-01-01",
                       val_end: str = None,
                       **filter_kwargs) -> list:
    """Phase 1 戦略探索: 確信度・オッズ帯・軸流しの組合せを網羅的にテスト"""
    strategies = [
        # (label, kwargs)
        ("ベースライン（現状）",
         {}),
        ("確信度>=0.02",
         {"confidence_min": 0.02}),
        ("確信度>=0.03",
         {"confidence_min": 0.03}),
        ("確信度>=0.05",
         {"confidence_min": 0.05}),
        ("オッズ4-30倍",
         {"odds_min": 4.0, "odds_max": 30.0}),
        ("オッズ4-50倍",
         {"odds_min": 4.0, "odds_max": 50.0}),
        ("オッズ3-20倍",
         {"odds_min": 3.0, "odds_max": 20.0}),
        ("軸流し",
         {"axis_flow": True}),
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
        # Phase 3: Kelly基準
        ("Kelly 1/4",
         {"kelly_fraction": 0.25}),
        ("Kelly 1/4 + 確信度>=0.03",
         {"kelly_fraction": 0.25, "confidence_min": 0.03}),
        ("Kelly 1/4 + 確信度>=0.05",
         {"kelly_fraction": 0.25, "confidence_min": 0.05}),
    ]

    period = f"{val_start}\u301c{val_end}" if val_end else f"{val_start}\u301c"
    print(f"\n  [戦略探索] 期間={period}")
    print(f"  {'─' * 105}")
    print(f"  {'戦略':36s} {'単勝':>6s} {'複勝':>6s} {'馬連':>6s} {'ﾜｲﾄﾞ':>6s}"
          f" {'馬単':>6s} {'3連複':>6s} {'3連単':>6s} {'Races':>6s} {'Skip':>5s}")
    print(f"  {'─' * 105}")

    all_results = []
    for label, kwargs in strategies:
        merged = {**filter_kwargs, **kwargs}
        res = run_backtest(
            input_path=input_path, model_dir=model_dir,
            returns_path=returns_path, val_start=val_start,
            val_end=val_end, top_n=3, **merged,
        )
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


def save_backtest_report(results: dict, output_path: Path = None):
    """バックテスト結果をJSONで保存"""
    output_path = output_path or (PROCESSED_DIR / "backtest_report.json")
    save_results = results.copy()
    save_results["race_details"] = results.get("race_details", [])[-20:]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(save_results, f, ensure_ascii=False, indent=2, default=str)
    print(f"  [OK] 保存: {output_path}")
