"""
パラメータ包括スイープ: 各パラメータの値を系統的に変化させてROI影響を可視化
Usage: python scripts/sweep_params.py [--val-start 2026-01-01]
"""
import sys
import json
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.family"] = "MS Gothic"

from config.settings import PROCESSED_DIR
from src.model.evaluator import (
    BET_TYPES, BET_LABELS, prepare_backtest_data, simulate_bets,
)


# ベースラインパラメータ（EXPANDING_BEST_PARAMS準拠）
BASELINE = {
    "confidence_min": 0.04,
    "quinella_top_n": 2,
    "wide_top_n": 2,
    "trio_top_n": 3,
    "skip_classes": [4, 6],
    "top_n": 3,
    "ev_threshold": 0.0,
    "quinella_prob_min": 0.0,
    "wide_prob_min": 0.0,
    "trio_prob_min": 0.0,
}

# スイープ対象パラメータと値
SWEEPS = {
    "confidence_min": [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10],
    "ev_threshold": [0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5],
    "quinella_prob_min": [0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20],
    "wide_prob_min": [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
    "trio_prob_min": [0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08],
    "skip_classes": [[], [4], [6], [4, 6], [4, 5, 6], [5, 6, 7]],
    "quinella_top_n": [2, 3, 4],
    "wide_top_n": [2, 3, 4],
    "trio_top_n": [3, 4, 5],
}

# グラフに表示する券種
PLOT_BET_TYPES = ["win", "place", "quinella", "wide", "trio"]
PLOT_COLORS = {
    "win": "#e74c3c",
    "place": "#3498db",
    "quinella": "#2ecc71",
    "wide": "#f39c12",
    "trio": "#9b59b6",
}


def run_sweep(prepared: dict, param_name: str, values: list) -> list:
    """1パラメータをスイープし、各値でのROI等を記録"""
    results = []
    for val in values:
        kwargs = BASELINE.copy()
        kwargs[param_name] = val

        res = simulate_bets(prepared, **kwargs)
        if not res:
            continue

        entry = {"param": param_name, "value": str(val), "races": res["races"]}
        for bt in BET_TYPES:
            b = res[f"bets_{bt}"]
            roi = (b["returned"] / b["invested"] * 100) if b["invested"] > 0 else 0
            hit_rate = (b["hits"] / b["count"] * 100) if b["count"] > 0 else 0
            entry[f"{bt}_roi"] = round(roi, 1)
            entry[f"{bt}_bets"] = b["count"]
            entry[f"{bt}_hits"] = b["hits"]
            entry[f"{bt}_hit_rate"] = round(hit_rate, 1)
            entry[f"{bt}_invested"] = b["invested"]
            entry[f"{bt}_returned"] = round(b["returned"], 0)
        results.append(entry)

    return results


def plot_sweep(results: list, param_name: str, output_dir: Path):
    """スイープ結果をグラフ化"""
    if not results:
        return

    values = [r["value"] for r in results]
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # ROI plot
    ax1 = axes[0]
    for bt in PLOT_BET_TYPES:
        rois = [r.get(f"{bt}_roi", 0) for r in results]
        label = BET_LABELS.get(bt, bt)
        ax1.plot(values, rois, marker="o", color=PLOT_COLORS[bt], label=label, linewidth=2)
    ax1.axhline(y=100, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax1.set_ylabel("ROI (%)")
    ax1.set_title(f"Parameter Sweep: {param_name}")
    ax1.legend(loc="best", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Bet count plot
    ax2 = axes[1]
    for bt in PLOT_BET_TYPES:
        bets = [r.get(f"{bt}_bets", 0) for r in results]
        label = BET_LABELS.get(bt, bt)
        ax2.plot(values, bets, marker="s", color=PLOT_COLORS[bt], label=label,
                 linewidth=1.5, linestyle="--")
    ax2.set_ylabel("Bet Count")
    ax2.set_xlabel(param_name)
    ax2.legend(loc="best", fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = output_dir / f"sweep_{param_name}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [GRAPH] {out_path}")


def print_sweep_table(results: list, param_name: str):
    """スイープ結果をテーブル表示"""
    if not results:
        return
    print(f"\n  {'='*90}")
    print(f"  Sweep: {param_name}")
    print(f"  {'-'*90}")
    header = f"  {'Value':>12s} {'Races':>6s}"
    for bt in PLOT_BET_TYPES:
        header += f" {BET_LABELS[bt]:>6s}"
    header += f" {'Q_bets':>7s} {'W_bets':>7s} {'T_bets':>7s}"
    print(header)
    print(f"  {'-'*90}")

    for r in results:
        row = f"  {r['value']:>12s} {r['races']:>6d}"
        for bt in PLOT_BET_TYPES:
            roi = r.get(f"{bt}_roi", 0)
            mark = "*" if roi >= 100 else " "
            row += f" {roi:>5.1f}%{mark}" if r.get(f"{bt}_bets", 0) > 0 else f" {'---':>6s} "
        row += f" {r.get('quinella_bets', 0):>7d}"
        row += f" {r.get('wide_bets', 0):>7d}"
        row += f" {r.get('trio_bets', 0):>7d}"
        print(row)
    print(f"  {'='*90}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="パラメータ包括スイープ")
    parser.add_argument("--val-start", default="2026-01-01", help="検証開始日")
    parser.add_argument("--val-end", default=None, help="検証終了日")
    parser.add_argument("--params", nargs="*", default=None,
                        help="スイープするパラメータ名（省略=全パラメータ）")
    parser.add_argument("--no-plot", action="store_true", help="グラフ生成をスキップ")
    args = parser.parse_args()

    print(f"\n  [パラメータスイープ] val_start={args.val_start}")
    print(f"  ベースライン: {json.dumps(BASELINE, ensure_ascii=False)}")

    prepared = prepare_backtest_data(val_start=args.val_start, val_end=args.val_end)
    if prepared is None:
        print("[ERROR] データ準備失敗")
        return

    output_dir = PROCESSED_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results = []

    sweep_targets = args.params or list(SWEEPS.keys())
    for param_name in sweep_targets:
        if param_name not in SWEEPS:
            print(f"  [SKIP] 不明なパラメータ: {param_name}")
            continue

        values = SWEEPS[param_name]
        print(f"\n  Sweeping {param_name}: {values}")
        results = run_sweep(prepared, param_name, values)
        print_sweep_table(results, param_name)
        all_results.extend(results)

        if not args.no_plot:
            plot_sweep(results, param_name, output_dir)

    # CSV出力
    if all_results:
        csv_path = output_dir / "sweep_results.csv"
        df = pd.DataFrame(all_results)
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"\n  [CSV] {csv_path} ({len(all_results)} rows)")


if __name__ == "__main__":
    main()
