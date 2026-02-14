# 競馬分析プロジェクト

## プロジェクト概要
日本競馬のレースを体系的に分析し、期待値に基づいた馬券購入判断を行う。
詳細な指示書: `docs/project_instructions_v3.md`

## 現在の状況（2026-02-14時点）
- jra_scraper.py v1.4 完成（単複・馬連・ワイド・3連複の全オッズ取得）
- ev_calculator.jsx v0.1 は保留（全自動Pythonパイプラインを優先）
- 評価点の定量化（第1段階）に着手予定
- 方針詳細: `docs/discussion_20260214_architecture.md`

## ディレクトリ構成
```
keiba-project/
├── CLAUDE.md              ← このファイル（Claude Codeが自動読込）
├── docs/                  # 設計ドキュメント
├── tools/                 # スクレイパー・計算ツール
├── data/races/            # レース分析データ（JSON）
└── data/templates/        # データテンプレート
```

## 主要ルール
1. 評価点は100点満点（実力50+騎手20+適性15+調子10+他5）
2. 対象券種: 単勝・複勝・馬連・ワイド・3連複（馬単・3連単は対象外）
3. 期待値C級（EV率1.0未満）は推奨しない
4. 1レースの結果で評価基準を大幅変更しない（10レース単位）
5. 本命は必ず1頭

## 優先課題
1. 評価点の定量化（基礎点自動算出 + Claude補正に分離）
2. パイプライン設計（scraper → netkeiba → scoring → API → EV計算）
3. scraper v1.4.1（馬単削除）

## 技術スタック
- Python 3 + Playwright（スクレイパー）
- Anthropic API（評価点付与、将来）
- Git/GitHub（データ管理）
