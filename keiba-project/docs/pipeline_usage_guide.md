# 全自動パイプライン使用ガイド

## 概要

jra_scraper → netkeiba_scraper → scoring_engine → ev_calculator の完全自動化パイプライン。

## パイプライン構成

```
┌─────────────────┐
│ jra_scraper.py  │ JRAから全オッズ・馬情報を取得
└────────┬────────┘
         │ input.json
         ↓
┌─────────────────┐
│netkeiba_scraper │ 過去走・騎手成績を追加
└────────┬────────┘
         │ enriched_input.json
         ↓
┌─────────────────┐
│ scoring_engine  │ 基礎点を自動算出
└────────┬────────┘
         │ base_scored.json
         ↓
┌─────────────────┐
│ ev_calculator   │ 期待値計算・買い目リスト生成
└────────┬────────┘
         │ ev_results.json
         ↓
     買い目決定
```

---

## ステップバイステップ実行

### 事前準備

```bash
# Playwright インストール（初回のみ）
playwright install chromium
```

### Step 1: JRAからオッズ取得

```bash
cd tools
python jra_scraper.py
```

**対話的な操作:**
1. 開催を選択（例: 1回東京5日）
2. レース番号を入力（例: 11）
3. レース情報を確認
4. Enterで取得開始

**出力ファイル:**
```
data/races/YYYYMMDD_会場_レース名_input.json
```

**所要時間:** 約30秒〜1分

---

### Step 2: netkeibaから過去走・騎手データ追加

```bash
python netkeiba_scraper.py ../data/races/YYYYMMDD_会場_レース名_input.json
```

**出力ファイル:**
```
data/races/YYYYMMDD_会場_レース名_enriched_input.json
```

**所要時間:** 約1〜2分（16頭の場合）

---

### Step 3: 基礎点自動算出

```bash
python scoring_engine.py ../data/races/YYYYMMDD_会場_レース名_enriched_input.json
```

**出力ファイル:**
```
data/races/YYYYMMDD_会場_レース名_base_scored.json
```

**出力例:**
```
============================================================
  Scoring Engine v0.1
============================================================

[スコア一覧]
  [1]  5番 レースホース1      78.0点 (実38.0 騎20.0 適15.0 調4.0 他2.0)
  [2]  8番 レースホース2      72.0点 (実35.0 騎15.0 適12.0 調3.0 他2.0)
  ...
```

**所要時間:** 数秒

---

### Step 4: 期待値計算・買い目リスト生成

```bash
python ev_calculator.py ../data/races/YYYYMMDD_会場_レース名_base_scored.json
```

**出力ファイル:**
```
data/races/YYYYMMDD_会場_レース名_ev_results.json
```

**出力例:**
```
============================================================
  期待値計算 v0.1
============================================================

レース: 東京 11R フェブラリーステークス
グレード: G1
温度パラメータ: 8
予算: ¥10,000
確信度: 65.3%

[単勝] トップ5
  [1]  3番 レースホース1      3.2倍 × 28.5% = EV 0.91 (C級)
  [2]  7番 レースホース2      4.5倍 × 22.1% = EV 0.99 (C級)
  [3]  1番 レースホース3      5.8倍 × 14.7% = EV 0.85 (C級)
  ...

[馬連] トップ5
  [1] 3-7      レースホース1-レースホース2       8.5倍 × 15.2% = EV 1.29 (A級)
  [2] 3-1      レースホース1-レースホース3      12.3倍 × 10.5% = EV 1.29 (A級)
  ...

============================================================
  推奨買い目（EV 1.0以上）
============================================================
  [1] 馬連     3-7          EV 1.29 (A級)
  [2] 馬連     3-1          EV 1.29 (A級)
  [3] ワイド    3-7          EV 1.15 (B級)
  [4] 複勝     7番           EV 1.08 (B級)
  ...
```

**所要時間:** 数秒

---

## 一括実行スクリプト（将来実装予定）

```bash
# run_pipeline.sh（未実装）
#!/bin/bash
echo "=== 全自動パイプライン ==="
echo ""
echo "[Step 1] JRAオッズ取得"
python jra_scraper.py

# 最新のinput.jsonを取得
LATEST_INPUT=$(ls -t ../data/races/*_input.json | head -1)
echo "対象: $LATEST_INPUT"

echo "[Step 2] netkeiba過去走追加"
python netkeiba_scraper.py "$LATEST_INPUT"

# enriched_input.jsonを取得
ENRICHED="${LATEST_INPUT/_input.json/_enriched_input.json}"

echo "[Step 3] 基礎点算出"
python scoring_engine.py "$ENRICHED"

# base_scored.jsonを取得
SCORED="${ENRICHED/_enriched_input.json/_base_scored.json}"

echo "[Step 4] 期待値計算"
python ev_calculator.py "$SCORED"

echo ""
echo "=== パイプライン完了 ==="
```

---

## トラブルシューティング

### Q: jra_scraper.pyで「開催が見つかりません」

**A:** JRAの開催日（土日または祝日）を確認。平日は開催がないことがあります。

### Q: netkeiba_scraper.pyで「馬が見つかりません」

**A:** netkeibaでの正式名称を確認。全角・半角、スペースの有無に注意。

### Q: scoring_engine.pyでスコアが全馬0点

**A:** enriched_input.jsonに`past_races`や`jockey_stats`が含まれているか確認：
```bash
cat ../data/races/YYYYMMDD_会場_レース名_enriched_input.json | grep "past_races"
```

### Q: ev_calculator.pyで「combo_odds」エラー

**A:** input.jsonにcombo_oddsが含まれているか確認。jra_scraper.py v1.4.1以降で取得可能。

---

## データフロー詳細

### input.json（jra_scraper出力）

```json
{
  "race": {
    "date": "2026-02-15",
    "venue": "東京",
    "race_number": 11,
    "name": "フェブラリーS",
    "grade": "G1",
    "surface": "ダート",
    "distance": 1600
  },
  "horses": [
    {
      "num": 1,
      "name": "馬名",
      "odds_win": 3.2,
      "odds_place": 1.5,
      "jockey": "騎手名",
      "load_weight": 57.0
    }
  ],
  "combo_odds": {
    "quinella": [{"combo": [1, 2], "odds": 8.5}],
    "wide": [{"combo": [1, 2], "odds": [2.1, 3.5]}],
    "trio": [{"combo": [1, 2, 3], "odds": 45.6}]
  }
}
```

### enriched_input.json（netkeiba_scraper出力）

input.jsonに以下を追加：

```json
{
  "horses": [
    {
      "num": 1,
      "name": "馬名",
      "past_races": [
        {
          "date": "2026-01-28",
          "venue": "東京",
          "distance": "1600",
          "surface": "ダ",
          "finish": 2,
          "margin": "0.2",
          "last3f": "34.5"
        }
      ],
      "jockey_stats": {
        "win_rate": 0.185,
        "place_rate": 0.412,
        "wins": 95,
        "races": 513
      }
    }
  ]
}
```

### base_scored.json（scoring_engine出力）

enriched_input.jsonに以下を追加：

```json
{
  "horses": [
    {
      "num": 1,
      "name": "馬名",
      "score": 75.5,
      "score_breakdown": {
        "ability": 38.0,
        "jockey": 18.5,
        "fitness": 12.0,
        "form": 4.0,
        "other": 3.0
      },
      "note": "[AUTO] ability=38.0(3着内率3/4, 着差評価12.0, 同条件3戦平均1.7点) jockey=18.5(勝率0.185) fitness=12.0(同場3戦平均2.3着, 同距離4戦平均2.0着) form=4.0(前走2週前4点) other=3.0(斤量57.0kg(平均-1.5kg)3点)"
    }
  ]
}
```

### ev_results.json（ev_calculator出力）

base_scored.jsonに以下を追加：

```json
{
  "ev_results": {
    "win": [
      {
        "num": 1,
        "name": "馬名",
        "prob": 0.285,
        "odds": 3.2,
        "ev": 0.91,
        "ev_ratio": 0.91,
        "rank": "C"
      }
    ],
    "quinella": [
      {
        "combo": "1-2",
        "names": "馬名A-馬名B",
        "prob": 0.152,
        "odds": 8.5,
        "ev": 1.29,
        "ev_ratio": 1.29,
        "rank": "A"
      }
    ],
    "confidence": 0.653,
    "temperature": 8,
    "budget": 10000
  }
}
```

---

## 次のステップ

### Phase 1: パイプライン検証（現在）

1. 実際のレースで全パイプラインを実行
2. 買い目リストの妥当性を確認
3. 10レース分のデータを蓄積

### Phase 2: Claude API連携（オプション）

```
base_scored.json
  ↓
Claude API（主観補正）
  ↓
final_scored.json
  ↓
ev_calculator.py
```

- 基礎点に対してClaudeが±補正を加える
- 補正理由をnote欄に記録
- 第1段階（基礎点）と第2段階（補正）の分離により透明性を維持

### Phase 3: 検証・改善（10レース後）

- 的中率・回収率の分析
- 温度パラメータの最適化
- スコア配点の調整（レースカテゴリ別）

---

## バージョン情報

- jra_scraper.py: v1.4.1
- netkeiba_scraper.py: v0.1
- scoring_engine.py: v0.1
- ev_calculator.py: v0.1

作成日: 2026-02-14
更新日: 2026-02-14
