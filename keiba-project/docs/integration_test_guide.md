# 統合テストガイド

## 概要

jra_scraper → netkeiba_scraper → scoring_engine の完全パイプラインを実際のレースデータでテストする手順。

## 前提条件

- Playwright インストール済み
- JRAの開催日（土日または祝日）

## テスト手順

### 1. JRAからレースデータ取得

```bash
cd tools
python jra_scraper.py
```

**対話的な操作:**
1. 開催を選択（例: 1回東京5日）
2. レース番号を入力（例: 11）
3. レース情報の確認
4. Enterで取得開始

**出力:**
```
data/races/YYYYMMDD_会場_レース名_input.json
```

### 2. netkeibaから過去走・騎手データ追加

```bash
python netkeiba_scraper.py ../data/races/YYYYMMDD_会場_レース名_input.json
```

**所要時間:** 約1〜2分（16頭の場合）

**出力:**
```
data/races/YYYYMMDD_会場_レース名_enriched_input.json
```

### 3. 基礎点自動算出

```bash
python scoring_engine.py ../data/races/YYYYMMDD_会場_レース名_enriched_input.json
```

**出力:**
```
data/races/YYYYMMDD_会場_レース名_base_scored.json
```

---

## 期待される結果

### scoring_engineの出力例

```
============================================================
  Scoring Engine v0.1
============================================================

入力: ../data/races/20260215_東京_11R_enriched_input.json

[スコア計算中]
  対象: 16頭

[スコア一覧]
  [1]  5番 レースホース1      78.0点 (実38.0 騎20.0 適15.0 調4.0 他2.0)
  [2]  8番 レースホース2      72.0点 (実35.0 騎15.0 適12.0 調3.0 他2.0)
  [3] 10番 レースホース3      65.5点 (実28.0 騎18.5 適12.0 調2.0 他3.0)
  ...

============================================================
  [OK] 保存完了: ../data/races/20260215_東京_11R_base_scored.json
============================================================
```

### base_scored.jsonの内容確認

```bash
cat ../data/races/YYYYMMDD_会場_レース名_base_scored.json | head -50
```

各馬に以下が追加されていることを確認：
- `score`: 基礎点合計（0-100点）
- `score_breakdown`: カテゴリ別内訳
- `note`: 計算根拠（例: `[AUTO] ability=35.0(3着内率3/4, 着差評価12.0, 同条件3戦平均1.7点)...`）

---

## 検証ポイント

### 1. データ取得の成功率

- **jra_scraper**: 全オッズ（単複・馬連・ワイド・3連複）が期待組数通り取得できているか
- **netkeiba_scraper**: 全馬の過去走データが取得できているか（過去4走以内）
- **netkeiba_scraper**: 全騎手の成績が取得できているか

### 2. スコアの妥当性

- 実績豊富な馬が高評価になっているか
- 超一流騎手が高評価（20点満点近く）になっているか
- コース実績がある馬が適性スコアで高評価か

### 3. 再現性

同じenriched_input.jsonから常に同じbase_scored.jsonが生成されるか確認：

```bash
# 1回目
python scoring_engine.py ../data/races/YYYYMMDD_会場_レース名_enriched_input.json
cp ../data/races/YYYYMMDD_会場_レース名_base_scored.json /tmp/result1.json

# 2回目
python scoring_engine.py ../data/races/YYYYMMDD_会場_レース名_enriched_input.json
cp ../data/races/YYYYMMDD_会場_レース名_base_scored.json /tmp/result2.json

# 比較（差分がないはず）
diff /tmp/result1.json /tmp/result2.json
```

### 4. エラーハンドリング

以下のケースで適切に動作するか：
- 過去走データが少ない馬（新馬など）
- 騎手データが取得できない場合
- netkeibaでヒットしない馬名

---

## トラブルシューティング

### Q: netkeiba_scraperで「馬が見つかりません」

**A:** netkeibaでの正式名称を確認。全角・半角、スペースの有無に注意。

### Q: 過去走データが0件

**A:** netkeibaのHTML構造変更の可能性。以下で確認：
```bash
# デバッグモードでブラウザを開いて手動確認
# netkeiba_scraper.py の headless=False で実行
```

### Q: スコアが全馬0点

**A:** enriched_input.jsonにpast_racesやjockey_statsが含まれているか確認：
```bash
cat ../data/races/YYYYMMDD_会場_レース名_enriched_input.json | grep "past_races"
cat ../data/races/YYYYMMDD_会場_レース名_enriched_input.json | grep "jockey_stats"
```

---

## 次のステップ

統合テスト成功後：

1. **Claude API連携**（base_scored.json → Claude補正 → final_scored.json）
2. **ev_calculator.py実装**（final_scored.json → 期待値計算 → 買い目リスト）
3. **実レース検証**（予想vs結果の分析）

---

## テストケース例

### ケース1: G1レース（東京優駿など）

期待値:
- 温度パラメータ: 8
- 実力スコアの差が大きい（実績馬が高評価）
- 超一流騎手が20点満点近い

### ケース2: 未勝利戦

期待値:
- 温度パラメータ: 12
- スコアのばらつきが大きい（データ不足の馬が多い）
- 新馬戦好走馬の適性スコアが低い（同条件なし）

### ケース3: ハンデ戦

期待値:
- 斤量差によるotherスコアの差（±1〜3点）
- 実力上位馬が高斤量で総合スコアが下がる

---

## ログ保存

テスト実行のログを保存：

```bash
# 全パイプライン実行ログ
cd tools
python jra_scraper.py 2>&1 | tee ../logs/jra_scraper_YYYYMMDD.log
python netkeiba_scraper.py ../data/races/YYYYMMDD_会場_レース名_input.json 2>&1 | tee ../logs/netkeiba_scraper_YYYYMMDD.log
python scoring_engine.py ../data/races/YYYYMMDD_会場_レース名_enriched_input.json 2>&1 | tee ../logs/scoring_engine_YYYYMMDD.log
```

---

## バージョン情報

- jra_scraper.py: v1.4
- netkeiba_scraper.py: v0.1
- scoring_engine.py: v0.1

作成日: 2026-02-14
