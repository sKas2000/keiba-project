#!/bin/bash
# -*- coding: utf-8 -*-
#
# 全自動パイプライン実行スクリプト v1.0
# ========================================
# jra_scraper → netkeiba_scraper → scoring_engine → ev_calculator
#
# 使い方:
#   cd tools
#   bash run_pipeline.sh

set -e  # エラーで即座に終了

echo "============================================================"
echo "  競馬分析 全自動パイプライン v1.0"
echo "============================================================"
echo ""

# カレントディレクトリ確認
if [ ! -f "jra_scraper.py" ]; then
    echo "[ERROR] tools/ ディレクトリで実行してください"
    exit 1
fi

# Step 1: JRAオッズ取得
echo "[Step 1/4] JRAオッズ取得"
echo "------------------------------------------------------------"
python jra_scraper.py
if [ $? -ne 0 ]; then
    echo "[ERROR] jra_scraper.py でエラーが発生しました"
    exit 1
fi
echo ""

# 最新のinput.jsonを取得
LATEST_INPUT=$(ls -t ../data/races/*_input.json 2>/dev/null | head -1)
if [ -z "$LATEST_INPUT" ]; then
    echo "[ERROR] input.json が見つかりません"
    exit 1
fi

echo "対象ファイル: $LATEST_INPUT"
echo ""

# Step 2: netkeiba過去走追加
echo "[Step 2/4] netkeiba過去走・騎手成績追加"
echo "------------------------------------------------------------"
python netkeiba_scraper.py "$LATEST_INPUT"
if [ $? -ne 0 ]; then
    echo "[ERROR] netkeiba_scraper.py でエラーが発生しました"
    exit 1
fi
echo ""

# enriched_input.jsonを取得
ENRICHED="${LATEST_INPUT/_input.json/_enriched_input.json}"
if [ ! -f "$ENRICHED" ]; then
    echo "[ERROR] enriched_input.json が見つかりません: $ENRICHED"
    exit 1
fi

echo "出力: $ENRICHED"
echo ""

# Step 3: 基礎点算出
echo "[Step 3/4] 基礎点自動算出"
echo "------------------------------------------------------------"
python scoring_engine.py "$ENRICHED"
if [ $? -ne 0 ]; then
    echo "[ERROR] scoring_engine.py でエラーが発生しました"
    exit 1
fi
echo ""

# base_scored.jsonを取得
SCORED="${ENRICHED/_enriched_input.json/_base_scored.json}"
if [ ! -f "$SCORED" ]; then
    echo "[ERROR] base_scored.json が見つかりません: $SCORED"
    exit 1
fi

echo "出力: $SCORED"
echo ""

# Step 4: 期待値計算
echo "[Step 4/4] 期待値計算・買い目リスト生成"
echo "------------------------------------------------------------"
python ev_calculator.py "$SCORED"
if [ $? -ne 0 ]; then
    echo "[ERROR] ev_calculator.py でエラーが発生しました"
    exit 1
fi
echo ""

# EV結果ファイルを取得
EV_RESULTS="${SCORED/_base_scored.json/_ev_results.json}"

echo "============================================================"
echo "  パイプライン完了"
echo "============================================================"
echo ""
echo "生成ファイル:"
echo "  1. $LATEST_INPUT"
echo "  2. $ENRICHED"
echo "  3. $SCORED"
echo "  4. $EV_RESULTS"
echo ""
echo "次のステップ: ev_results.json を確認して買い目を決定"
