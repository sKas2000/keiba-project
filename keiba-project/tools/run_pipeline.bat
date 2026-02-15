@echo off
REM -*- coding: utf-8 -*-
REM
REM 全自動パイプライン実行スクリプト v1.0 (Windows版)
REM ========================================================
REM jra_scraper → netkeiba_scraper → scoring_engine → ev_calculator
REM
REM 使い方:
REM   cd tools
REM   run_pipeline.bat

setlocal enabledelayedexpansion

echo ============================================================
echo   競馬分析 全自動パイプライン v1.0
echo ============================================================
echo.

REM カレントディレクトリ確認
if not exist "jra_scraper.py" (
    echo [ERROR] tools\ ディレクトリで実行してください
    exit /b 1
)

REM Step 1: JRAオッズ取得
echo [Step 1/4] JRAオッズ取得
echo ------------------------------------------------------------
python jra_scraper.py --non-interactive --headless --yes
if errorlevel 1 (
    echo [ERROR] jra_scraper.py でエラーが発生しました
    exit /b 1
)
echo.

REM 最新のinput.jsonを取得
for /f "delims=" %%f in ('dir /b /o-d ..\data\races\*_input.json 2^>nul') do (
    set "LATEST_INPUT=..\data\races\%%f"
    goto :found_input
)
echo [ERROR] input.json が見つかりません
exit /b 1

:found_input
echo 対象ファイル: %LATEST_INPUT%
echo.

REM Step 2: netkeiba過去走追加
echo [Step 2/4] netkeiba過去走・騎手成績追加
echo ------------------------------------------------------------
python netkeiba_scraper.py "%LATEST_INPUT%"
if errorlevel 1 (
    echo [ERROR] netkeiba_scraper.py でエラーが発生しました
    exit /b 1
)
echo.

REM enriched_input.jsonを取得
set "ENRICHED=%LATEST_INPUT:_input.json=_enriched_input.json%"
if not exist "%ENRICHED%" (
    echo [ERROR] enriched_input.json が見つかりません: %ENRICHED%
    exit /b 1
)

echo 出力: %ENRICHED%
echo.

REM Step 3: 基礎点算出
echo [Step 3/4] 基礎点自動算出
echo ------------------------------------------------------------
python scoring_engine.py "%ENRICHED%"
if errorlevel 1 (
    echo [ERROR] scoring_engine.py でエラーが発生しました
    exit /b 1
)
echo.

REM base_scored.jsonを取得
set "SCORED=%ENRICHED:_enriched_input.json=_base_scored.json%"
if not exist "%SCORED%" (
    echo [ERROR] base_scored.json が見つかりません: %SCORED%
    exit /b 1
)

echo 出力: %SCORED%
echo.

REM Step 4: 期待値計算
echo [Step 4/4] 期待値計算・買い目リスト生成
echo ------------------------------------------------------------
python ev_calculator.py "%SCORED%"
if errorlevel 1 (
    echo [ERROR] ev_calculator.py でエラーが発生しました
    exit /b 1
)
echo.

REM EV結果ファイル
set "EV_RESULTS=%SCORED:_base_scored.json=_ev_results.json%"

echo ============================================================
echo   パイプライン完了
echo ============================================================
echo.
echo 生成ファイル:
echo   1. %LATEST_INPUT%
echo   2. %ENRICHED%
echo   3. %SCORED%
echo   4. %EV_RESULTS%
echo.
echo 次のステップ: ev_results.json を確認して買い目を決定

endlocal
