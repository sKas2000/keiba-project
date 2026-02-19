# 競馬分析プロジェクト (keiba-ai)

## プロジェクト概要
日本競馬のレースを体系的に分析し、期待値に基づいた馬券購入判断を行う。
詳細な指示書: `docs/project_instructions_v3.md`

## ディレクトリ構成
```
keiba-project/
├── config/
│   ├── settings.py          # 全定数・パス・マップを一元管理
│   └── logging.conf         # ログ設定
├── src/
│   ├── scraping/
│   │   ├── base.py          # 共通Playwright管理（リトライ、レート制限）
│   │   ├── calendar.py      # 開催日・レース選択UI
│   │   ├── race.py          # 過去レース結果取得
│   │   ├── horse.py         # 馬の過去成績・騎手成績取得
│   │   ├── odds.py          # JRAオッズ取得
│   │   └── parsers.py       # HTML解析ユーティリティ
│   ├── data/
│   │   ├── storage.py       # JSON/CSV読み書き
│   │   ├── preprocessing.py # データ前処理・バリデーション
│   │   └── feature.py       # 特徴量作成（CSV/JSON両対応）
│   ├── model/
│   │   ├── trainer.py       # LightGBM学習
│   │   ├── predictor.py     # ルールベース＋ML予測＋EV計算
│   │   └── evaluator.py     # バックテスト・評価
│   └── pipeline.py          # パイプラインオーケストレーション
├── tests/                   # pytest テスト
├── tools/                   # 旧スクリプト（移行元、参照用）
├── data/
│   ├── raw/                 # ML生データ（results.csv）
│   ├── processed/           # 前処理済みデータ
│   ├── features/            # 特徴量データ
│   └── races/               # レース分析JSON
├── models/                  # 学習済みモデル
├── logs/                    # 実行ログ
├── docs/                    # 設計ドキュメント
├── main.py                  # CLIエントリポイント
├── pyproject.toml
└── requirements.txt
```

## CLI 使い方
```bash
# ルールベースパイプライン（スクレイピング→スコア→EV）
python main.py run [--meeting-index N] [--race N] [--non-interactive]

# MLパイプライン
python main.py ml [--input enriched_input.json]

# 既存JSONにスコアリング
python main.py score <enriched_input.json> [--mode rule|ml]

# 特徴量エンジニアリング
python main.py feature [--input results.csv] [--output features.csv]

# モデル学習
python main.py train [--tune] [--val-start 2025-01-01]

# バックテスト
python main.py backtest [--save]

# 過去レース収集
python main.py collect --start 2024-01-01 --end 2024-12-31

# テスト実行
python -m pytest tests/ -v
```

## アーキテクチャ

### 定数管理
- 全定数は `config/settings.py` に集約（Single Source of Truth）
- `FEATURE_COLUMNS`（33特徴量）、エンコーディングマップ等

### 予測方式（2系統）
1. **ルールベース**: 100点満点（実力50+騎手20+適性15+調子10+他5）
2. **ML**: LightGBM二値分類（3着以内） + LambdaRankランキング

### パイプライン（4ステップ）
1. `src/scraping/odds.py` → input.json（JRAオッズ取得）
2. `src/scraping/horse.py` → enriched_input.json（過去走・騎手追加）
3. `src/model/predictor.py` → base_scored.json（スコアリング）
4. `src/model/predictor.py` → ev_results.json（EV計算）

## 主要ルール
1. 評価点は100点満点（実力50+騎手20+適性15+調子10+他5）
2. 対象券種: 単勝・複勝・馬連・ワイド・3連複（馬単・3連単は対象外）
3. 期待値C級（EV率1.0未満）は推奨しない
4. 1レースの結果で評価基準を大幅変更しない（10レース単位）
5. 本命は必ず1頭

## 技術スタック
- Python 3 + Playwright（スクレイパー）
- LightGBM + scikit-learn（ML予測）
- pandas + numpy（データ処理）
- pytest（テスト）
- Git/GitHub（バージョン管理）
