#!/bin/bash
# JRAスクレイパー セットアップスクリプト
# 使い方: bash setup.sh

echo "🏇 JRAスクレイパー セットアップ"
echo "================================"

# Python確認
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 が見つかりません"
    exit 1
fi

echo "✅ Python3: $(python3 --version)"

# pip install
echo ""
echo "📦 Playwright をインストール中..."
pip install playwright --break-system-packages 2>/dev/null || pip install playwright

# Chromium インストール
echo ""
echo "🌐 Chromium ブラウザをインストール中..."
playwright install chromium

echo ""
echo "✅ セットアップ完了！"
echo ""
echo "使い方:"
echo "  python3 jra_scraper.py          # インタラクティブモード"
echo "  python3 jra_scraper.py -m '1回京都6日' -r 8  # クイックモード"
