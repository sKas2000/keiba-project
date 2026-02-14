# 競馬分析プロジェクト

## プロジェクト概要
日本競馬のレースを体系的に分析し、期待値に基づいた馬券購入判断を行う。
詳細な指示書: `docs/project_instructions_v3.md`

## 現在の状況（2026-02-14時点）
- ✅ jra_scraper.py v1.4.1 完成（対象券種: 単勝・複勝・馬連・ワイド・3連複のみ）
- ✅ netkeiba_scraper.py v0.1 完成（過去走データ・騎手成績の自動取得）
- ✅ scoring_engine.py v0.1 完成（基礎点自動算出、動作テスト済み）
- ✅ ev_calculator.py v0.1 完成（期待値計算・買い目リスト生成、動作テスト済み）
- ✅ 一括実行スクリプト完成（run_pipeline.bat / run_pipeline.sh）
- ✅ ドキュメント完備（使用ガイド・統合テストガイド・チェックリスト）
- ⏸️ ev_calculator.jsx v0.1 は保留（Pythonパイプライン完成により不要）
- 📋 次の課題: 実データでの統合テスト（次回開催日）
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
1. ✅ 評価点の定量化（基礎点自動算出 + Claude補正に分離）第1段階完了
2. ✅ 全自動Pythonパイプライン実装完了
   - ✅ jra_scraper.py v1.4.1 → input.json
   - ✅ netkeiba_scraper.py → enriched_input.json
   - ✅ scoring_engine.py → base_scored.json
   - ✅ ev_calculator.py → ev_results.json（買い目リスト）
3. 📋 実データでの統合テスト（jra_scraper → netkeiba → scoring → ev_calculator）
4. 📋 Claude API連携（base_scored.json → Claude補正 → final_scored.json）

## 技術スタック
- Python 3 + Playwright（スクレイパー）
- Anthropic API（評価点付与、将来）
- Git/GitHub（データ管理）
