"""
パイプラインオーケストレーション
ルールベース / ML / ハイブリッドの各パイプラインを統合実行
"""
import json
import logging
from pathlib import Path

from config.settings import (
    RACES_DIR, MODEL_DIR, GRADE_DEFAULTS, ML_TEMPERATURE_DEFAULT,
    EXPANDING_BEST_PARAMS, setup_encoding,
)
from src.data.storage import read_json, write_json, generate_output_filename

logger = logging.getLogger("keiba.pipeline")


# ============================================================
# ルールベースパイプライン
# input.json → enriched → scored → ev_results
# ============================================================

async def run_rule_pipeline(
    meeting_index: int = None,
    race: int = None,
    non_interactive: bool = False,
    headless: bool = True,
):
    """ルールベース全自動パイプライン"""
    from src.scraping.odds import OddsScraper, build_input_json
    from src.scraping.horse import HorseScraper, enrich_race_data
    from src.model.scoring import score_rule_based
    from src.model.ev import calculate_ev, print_ev_results

    logger.info("ルールベースパイプライン開始")

    # Step 1: JRA オッズ取得
    print("\n[Step 1/4] JRA オッズ取得")
    print("-" * 40)
    odds_scraper = OddsScraper(headless=headless)
    await odds_scraper.start()
    try:
        data = await build_input_json(
            odds_scraper,
            meeting_index=meeting_index,
            race=race,
            non_interactive=non_interactive,
        )
    finally:
        await odds_scraper.close()

    input_path = generate_output_filename(data, "input")
    write_json(data, input_path)
    print(f"  出力: {input_path}")

    # Step 2: netkeiba 過去走追加
    print("\n[Step 2/4] netkeiba 過去走・騎手成績追加")
    print("-" * 40)
    horse_scraper = HorseScraper(headless=headless)
    await horse_scraper.start()
    try:
        data = await enrich_race_data(horse_scraper, data)
    finally:
        await horse_scraper.close()

    enriched_path = generate_output_filename(data, "enriched_input")
    write_json(data, enriched_path)
    print(f"  出力: {enriched_path}")

    # Step 3: ルールベーススコアリング
    print("\n[Step 3/4] 基礎点自動算出")
    print("-" * 40)
    grade = data.get("race", {}).get("grade", "")
    temp, budget = GRADE_DEFAULTS.get(grade, (10, 1500))
    data["parameters"] = {"temperature": temp, "budget": budget, "top_n": 6}
    data = score_rule_based(data)

    scored_path = generate_output_filename(data, "base_scored")
    write_json(data, scored_path)
    print(f"  出力: {scored_path}")

    # Step 4: 期待値計算
    print("\n[Step 4/4] 期待値計算")
    print("-" * 40)
    ev_results = calculate_ev(data)
    data["ev_results"] = ev_results

    ev_path = generate_output_filename(data, "ev_results")
    write_json(data, ev_path)
    print(f"  出力: {ev_path}")

    print_ev_results(ev_results, data.get("race", {}))
    logger.info("ルールベースパイプライン完了: %s", ev_path)
    return data


# ============================================================
# ML パイプライン
# enriched_input.json → ML予測 → ev_results
# ============================================================

async def run_ml_pipeline(
    input_file: str = None,
    meeting_index: int = None,
    race: int = None,
    non_interactive: bool = False,
    headless: bool = True,
    model_dir: Path = None,
):
    """MLモデルベースパイプライン"""
    from src.model.predictor import score_ml
    from src.model.ev import calculate_ev, print_ev_results

    logger.info("MLパイプライン開始 (input=%s)", input_file or "scraping")
    model_dir = model_dir or MODEL_DIR

    if input_file:
        data = read_json(input_file)
    else:
        # スクレイピングから開始
        from src.scraping.odds import OddsScraper, build_input_json
        from src.scraping.horse import HorseScraper, enrich_race_data

        print("\n[Step 1/3] JRA オッズ取得")
        print("-" * 40)
        odds_scraper = OddsScraper(headless=headless)
        await odds_scraper.start()
        try:
            data = await build_input_json(
                odds_scraper,
                meeting_index=meeting_index,
                race=race,
                non_interactive=non_interactive,
            )
        finally:
            await odds_scraper.close()

        input_path = generate_output_filename(data, "input")
        write_json(data, input_path)

        print("\n[Step 2/3] netkeiba 過去走追加")
        print("-" * 40)
        horse_scraper = HorseScraper(headless=headless)
        await horse_scraper.start()
        try:
            data = await enrich_race_data(horse_scraper, data)
        finally:
            await horse_scraper.close()

        enriched_path = generate_output_filename(data, "enriched_input")
        write_json(data, enriched_path)

    # ML 予測
    step = "Step 3/3" if not input_file else "Step 1/2"
    print(f"\n[{step}] ML 予測")
    print("-" * 40)
    result = score_ml(data, model_dir=model_dir)
    if result is None:
        print("[WARN] MLモデル未学習。ルールベースにフォールバック")
        from src.model.scoring import score_rule_based
        result = score_rule_based(data)

    grade = data.get("race", {}).get("grade", "")
    _, budget = GRADE_DEFAULTS.get(grade, (10, 1500))
    result["parameters"] = {"temperature": ML_TEMPERATURE_DEFAULT, "budget": budget, "top_n": 3}

    scored_path = generate_output_filename(result, "ml_scored")
    write_json(result, scored_path)
    print(f"  出力: {scored_path}")

    # EV 計算（EXPANDING_BEST_PARAMS 戦略を適用）
    step = "Step 3/3" if input_file else "Step 3/3"
    print(f"\n[{step}] 期待値計算（Expanding Window戦略）")
    print("-" * 40)
    ev_results = calculate_ev(result, strategy=EXPANDING_BEST_PARAMS)
    result["ev_results"] = ev_results

    ev_path = generate_output_filename(result, "ev_results")
    write_json(result, ev_path)
    print(f"  出力: {ev_path}")

    print_ev_results(ev_results, result.get("race", {}))
    logger.info("MLパイプライン完了: %s", ev_path)
    return result


# ============================================================
# ML 学習パイプライン
# ============================================================

def run_train_pipeline(input_path: str = None, val_start: str = "2025-01-01",
                       tune: bool = False):
    """ML学習パイプライン"""
    from src.model.trainer import train_all
    train_all(input_path=input_path, val_start=val_start, tune=tune)


# ============================================================
# 特徴量エンジニアリングパイプライン
# ============================================================

def run_feature_pipeline(input_path: str = None, output_path: str = None):
    """特徴量エンジニアリングパイプライン"""
    from src.data.feature import run_feature_pipeline as _run
    return _run(input_path=input_path, output_path=output_path)


# ============================================================
# バックテストパイプライン
# ============================================================

def run_backtest_pipeline(input_path: str = None, model_dir: str = None,
                          val_start: str = "2025-01-01",
                          val_end: str = None,
                          threshold: float = 0.0, top_n: int = 3,
                          save: bool = False,
                          ev_threshold: float = 0.0,
                          compare_ev: bool = False,
                          optimize_temp: bool = False,
                          temperature: float = 1.0,
                          explore: bool = False,
                          confidence_min: float = 0.0,
                          odds_min: float = 0.0,
                          odds_max: float = 0.0,
                          axis_flow: bool = False,
                          kelly_fraction: float = 0.0,
                          analyze_cond: bool = False):
    """バックテストパイプライン"""
    from src.model.evaluator import run_backtest
    from src.model.optimization import (
        compare_ev_thresholds, optimize_temperature,
        explore_strategies, analyze_by_condition,
    )
    from src.model.reporting import (
        print_backtest_report, save_backtest_report,
        print_ev_comparison,
    )

    model_path = Path(model_dir) if model_dir else None

    if analyze_cond:
        return analyze_by_condition(
            input_path=input_path,
            model_dir=model_path,
            val_start=val_start,
            val_end=val_end,
            kelly_fraction=kelly_fraction,
            confidence_min=confidence_min,
        )

    if explore:
        return explore_strategies(
            input_path=input_path,
            model_dir=model_path,
            val_start=val_start,
            val_end=val_end,
        )

    if optimize_temp:
        best = optimize_temperature(
            input_path=input_path,
            model_dir=model_path,
            val_start=val_start,
            val_end=val_end,
            ev_threshold=ev_threshold,
        )
        return best

    if compare_ev:
        comparison = compare_ev_thresholds(
            input_path=input_path,
            model_dir=model_path,
            val_start=val_start,
            val_end=val_end,
            temperature=temperature,
        )
        print_ev_comparison(comparison)
        return comparison

    results = run_backtest(
        input_path=input_path,
        model_dir=model_path,
        val_start=val_start,
        val_end=val_end,
        bet_threshold=threshold,
        top_n=top_n,
        ev_threshold=ev_threshold,
        temperature=temperature,
        confidence_min=confidence_min,
        odds_min=odds_min,
        odds_max=odds_max,
        axis_flow=axis_flow,
        kelly_fraction=kelly_fraction,
    )
    print_backtest_report(results)

    if save and results:
        save_backtest_report(results)

    return results


# ============================================================
# レース結果収集パイプライン
# ============================================================

async def run_collect_pipeline(start_date: str, end_date: str,
                               output_path: str = None,
                               headless: bool = True,
                               append: bool = False):
    """過去レース結果収集"""
    logger.info("レース収集開始: %s ~ %s", start_date, end_date)
    from src.scraping.race import collect_races
    await collect_races(
        start_date=start_date, end_date=end_date,
        output_path=output_path, headless=headless,
        append=append,
    )


# ============================================================
# オッズ監視パイプライン
# ============================================================

async def run_monitor_pipeline(before: int = 5, webhook: str = None,
                               headless: bool = True, venue: str = None):
    """オッズ監視サーバー（発走時刻連動Discord通知）"""
    from src.monitor import RaceMonitor
    monitor = RaceMonitor(
        before=before,
        token=webhook,
        headless=headless,
        venue_filter=venue,
    )
    await monitor.run()


# ============================================================
# スコアリング単体実行（既存JSONに対して）
# ============================================================

def run_score(input_file: str, mode: str = "rule"):
    """既存の enriched_input.json にスコアリングだけ実行"""
    from src.model.scoring import score_rule_based
    from src.model.predictor import score_ml
    from src.model.ev import calculate_ev, print_ev_results

    data = read_json(input_file)

    if mode == "ml":
        result = score_ml(data)
        if result is None:
            print("[WARN] MLモデル未学習。ルールベースにフォールバック")
            result = score_rule_based(data)
    else:
        result = score_rule_based(data)

    grade = data.get("race", {}).get("grade", "")
    temp, budget = GRADE_DEFAULTS.get(grade, (10, 1500))
    if mode == "ml":
        temp = ML_TEMPERATURE_DEFAULT
    result["parameters"] = {"temperature": temp, "budget": budget, "top_n": 3}

    suffix = "ml_scored" if mode == "ml" else "base_scored"
    scored_path = generate_output_filename(result, suffix)
    write_json(result, scored_path)
    print(f"  出力: {scored_path}")

    # ML時はEXPANDING_BEST_PARAMS戦略を適用
    strategy = EXPANDING_BEST_PARAMS if mode == "ml" else None
    ev_results = calculate_ev(result, strategy=strategy)
    result["ev_results"] = ev_results

    ev_path = generate_output_filename(result, "ev_results")
    write_json(result, ev_path)
    print(f"  出力: {ev_path}")

    print_ev_results(ev_results, result.get("race", {}))
    return result
