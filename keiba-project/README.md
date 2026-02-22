# 競馬分析プロジェクト

日本競馬のレースを体系的に分析し、期待値に基づいた馬券購入判断を行うプロジェクト。

## セットアップ

```bash
pip install -r requirements.txt
playwright install chromium
```

## 使い方

```bash
# ルールベースパイプライン（スクレイピング→スコア→EV）
python main.py run [--meeting-index N] [--race N]

# MLパイプライン
python main.py ml [--input enriched_input.json]

# オッズ監視（発走時刻連動・Discord通知）
python main.py monitor [--before 5] [--webhook URL]

# 既存JSONにスコアリング
python main.py score <enriched_input.json> [--mode rule|ml]

# 特徴量エンジニアリング
python main.py feature [--input results.csv] [--output features.csv]

# モデル学習
python main.py train [--tune] [--val-start 2025-01-01]

# バックテスト
python main.py backtest [--save] [--explore] [--optimize-temp]

# 過去レース収集
python main.py collect --start 2024-01-01 --end 2024-12-31

# テスト
python -m pytest tests/ -v
```

## ドキュメント
- `docs/project_instructions_v3.md` — プロジェクト指示書
- `docs/ml_design.md` — ML設計書
- `docs/pipeline_usage_guide.md` — パイプライン使い方
- `docs/data_architecture.drawio` — データフロー図
- `CLAUDE.md` — Claude Code用コンテキスト
