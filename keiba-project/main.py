#!/usr/bin/env python3
"""
keiba-ai CLI エントリポイント
"""
import asyncio
import sys
from argparse import ArgumentParser

from config.settings import setup_encoding


def cmd_run(args):
    """ルールベースパイプライン実行"""
    from src.pipeline import run_rule_pipeline
    asyncio.run(run_rule_pipeline(
        meeting_index=args.meeting_index,
        race=args.race,
        non_interactive=args.non_interactive,
        headless=not args.no_headless,
    ))


def cmd_ml(args):
    """MLパイプライン実行"""
    from src.pipeline import run_ml_pipeline
    asyncio.run(run_ml_pipeline(
        input_file=args.input,
        meeting_index=args.meeting_index,
        race=args.race,
        non_interactive=args.non_interactive,
        headless=not args.no_headless,
    ))


def cmd_score(args):
    """既存JSONにスコアリング"""
    from src.pipeline import run_score
    run_score(args.input, mode=args.mode)


def cmd_feature(args):
    """特徴量エンジニアリング"""
    from src.pipeline import run_feature_pipeline
    run_feature_pipeline(input_path=args.input, output_path=args.output)


def cmd_train(args):
    """モデル学習"""
    from src.pipeline import run_train_pipeline
    run_train_pipeline(
        input_path=args.input,
        val_start=args.val_start,
        tune=args.tune,
    )


def cmd_backtest(args):
    """バックテスト"""
    from src.pipeline import run_backtest_pipeline
    run_backtest_pipeline(
        input_path=args.input,
        model_dir=args.model_dir,
        val_start=args.val_start,
        val_end=getattr(args, "val_end", None),
        threshold=args.threshold,
        top_n=args.top_n,
        save=args.save,
        ev_threshold=args.ev_threshold,
        compare_ev=args.compare_ev,
        optimize_temp=getattr(args, "optimize_temp", False),
        temperature=getattr(args, "temperature", 1.0),
        explore=getattr(args, "explore", False),
        confidence_min=getattr(args, "confidence_min", 0.0),
        odds_min=getattr(args, "odds_min", 0.0),
        odds_max=getattr(args, "odds_max", 0.0),
        axis_flow=getattr(args, "axis_flow", False),
        kelly_fraction=getattr(args, "kelly", 0.0),
        analyze_cond=getattr(args, "analyze_cond", False),
    )


def cmd_collect(args):
    """過去レース収集"""
    from src.pipeline import run_collect_pipeline
    asyncio.run(run_collect_pipeline(
        start_date=args.start,
        end_date=args.end,
        output_path=args.output,
        headless=not args.no_headless,
        append=args.append,
    ))


def cmd_monitor(args):
    """オッズ監視サーバー"""
    from src.pipeline import run_monitor_pipeline
    asyncio.run(run_monitor_pipeline(
        before=args.before,
        webhook=args.webhook,
        headless=not args.no_headless,
        venue=args.venue,
    ))


def main():
    setup_encoding()

    parser = ArgumentParser(description="keiba-ai 競馬分析CLI")
    sub = parser.add_subparsers(dest="command", help="実行コマンド")

    # run: ルールベースパイプライン
    p_run = sub.add_parser("run", help="ルールベースパイプライン（スクレイピング→スコア→EV）")
    p_run.add_argument("--meeting-index", type=int, help="開催インデックス（1始まり）")
    p_run.add_argument("--race", type=int, help="レース番号（1-12）")
    p_run.add_argument("--non-interactive", action="store_true", help="非対話モード")
    p_run.add_argument("--no-headless", action="store_true", help="ブラウザを表示")
    p_run.set_defaults(func=cmd_run)

    # ml: MLパイプライン
    p_ml = sub.add_parser("ml", help="MLパイプライン（スクレイピング or JSON→ML予測→EV）")
    p_ml.add_argument("--input", help="enriched_input.json（指定しない場合はスクレイピングから開始）")
    p_ml.add_argument("--meeting-index", type=int, help="開催インデックス（1始まり）")
    p_ml.add_argument("--race", type=int, help="レース番号（1-12）")
    p_ml.add_argument("--non-interactive", action="store_true", help="非対話モード")
    p_ml.add_argument("--no-headless", action="store_true", help="ブラウザを表示")
    p_ml.set_defaults(func=cmd_ml)

    # score: 既存JSONにスコアリング
    p_score = sub.add_parser("score", help="既存JSONにスコアリング実行")
    p_score.add_argument("input", help="enriched_input.json パス")
    p_score.add_argument("--mode", choices=["rule", "ml"], default="rule", help="スコアリング方式")
    p_score.set_defaults(func=cmd_score)

    # feature: 特徴量エンジニアリング
    p_feat = sub.add_parser("feature", help="results.csvから特徴量作成")
    p_feat.add_argument("--input", help="入力CSV")
    p_feat.add_argument("--output", help="出力CSV")
    p_feat.set_defaults(func=cmd_feature)

    # train: モデル学習
    p_train = sub.add_parser("train", help="LightGBMモデル学習")
    p_train.add_argument("--input", help="特徴量CSV")
    p_train.add_argument("--val-start", default="2025-01-01", help="検証開始日")
    p_train.add_argument("--tune", action="store_true", help="Optuna最適化")
    p_train.set_defaults(func=cmd_train)

    # backtest: バックテスト
    p_bt = sub.add_parser("backtest", help="バックテスト実行")
    p_bt.add_argument("--input", help="特徴量CSV")
    p_bt.add_argument("--model-dir", help="モデルディレクトリ")
    p_bt.add_argument("--val-start", default="2025-01-01", help="検証開始日")
    p_bt.add_argument("--val-end", default=None, help="検証終了日（3-way split用）")
    p_bt.add_argument("--threshold", type=float, default=0.0, help="予測確率の購入閾値")
    p_bt.add_argument("--top-n", type=int, default=3, help="上位N頭を対象")
    p_bt.add_argument("--save", action="store_true", help="結果をJSON保存")
    p_bt.add_argument("--ev-threshold", type=float, default=0.0, help="EV閾値（0=フィルタなし）")
    p_bt.add_argument("--compare-ev", action="store_true", help="複数EV閾値で比較")
    p_bt.add_argument("--optimize-temp", action="store_true", help="温度パラメータ最適化")
    p_bt.add_argument("--temperature", type=float, default=1.0, help="ソフトマックス温度（logitスケール）")
    p_bt.add_argument("--explore", action="store_true", help="戦略探索モード（Phase1フィルタ網羅テスト）")
    p_bt.add_argument("--confidence-min", type=float, default=0.0, help="確信度フィルタ（Top1-Top2のwin_prob差）")
    p_bt.add_argument("--odds-min", type=float, default=0.0, help="最低オッズ（単勝・複勝）")
    p_bt.add_argument("--odds-max", type=float, default=0.0, help="最高オッズ（単勝・複勝）")
    p_bt.add_argument("--axis-flow", action="store_true", help="馬単・3連単をTop1軸流しに変更")
    p_bt.add_argument("--kelly", type=float, default=0.0, help="Kelly基準の割合（0=均一賭け、0.25=1/4 Kelly推奨）")
    p_bt.add_argument("--analyze-cond", action="store_true", help="条件別分析（クラス・馬場・距離別ROI）")
    p_bt.set_defaults(func=cmd_backtest)

    # collect: レース結果収集
    p_col = sub.add_parser("collect", help="過去レース結果収集")
    p_col.add_argument("--start", required=True, help="開始日 (YYYY-MM-DD)")
    p_col.add_argument("--end", required=True, help="終了日 (YYYY-MM-DD)")
    p_col.add_argument("--output", help="出力CSVパス")
    p_col.add_argument("--no-headless", action="store_true", help="ブラウザを表示")
    p_col.add_argument("--append", action="store_true", help="既存CSVに追記（収集済み日付をスキップ）")
    p_col.set_defaults(func=cmd_collect)

    # monitor: オッズ監視サーバー
    p_mon = sub.add_parser("monitor", help="オッズ監視サーバー（発走時刻連動Discord通知）")
    p_mon.add_argument("--before", type=int, default=5, help="発走何分前にバッチ実行（デフォルト5）")
    p_mon.add_argument("--webhook", help="Discord Webhook URL（.envでも設定可）")
    p_mon.add_argument("--venue", help="会場フィルタ（例: 東京 or 中山,東京）")
    p_mon.add_argument("--no-headless", action="store_true", help="ブラウザを表示")
    p_mon.set_defaults(func=cmd_monitor)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
