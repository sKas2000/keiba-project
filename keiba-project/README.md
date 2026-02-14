# 競馬分析プロジェクト

日本競馬のレースを体系的に分析し、期待値に基づいた馬券購入判断を行うプロジェクト。

## セットアップ

```bash
pip install playwright
playwright install chromium
```

## 使い方

```bash
# オッズ取得
python tools/jra_scraper.py

# → input.json が data/races/ に生成される
# → Claudeに評価を依頼
# → 期待値計算（パイプライン開発中）
```

## ドキュメント
- `docs/project_instructions_v3.md` - プロジェクト指示書
- `docs/discussion_20260214_architecture.md` - アーキテクチャ議論ログ
- `CLAUDE.md` - Claude Code用コンテキスト
