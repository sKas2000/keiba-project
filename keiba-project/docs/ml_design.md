# 教師あり学習パイプライン設計書

## 1. 概要

ルールベーススコアリングから教師あり学習（LightGBM）に移行する。
netkeiba.comから過去レース結果を収集し、特徴量を生成、モデルを学習して
3着以内予測の二値分類および着順ランキング予測を行う。

## 2. アーキテクチャ

```
[netkeiba.com] → race_collector.py → data/ml/raw/results.csv
                                                ↓
                                    feature_engineer.py → data/ml/processed/features.csv
                                                ↓
                                    model_trainer.py → data/ml/models/model.txt
                                                ↓
[JRA当日オッズ] → predictor.py → 予測確率 → ev_calculator.py → 買い目
                                                ↓
                                    backtest.py → 回収率レポート
```

## 3. データスキーマ

### results.csv（1行 = 1頭 × 1レース）

| カラム | 型 | 説明 |
|--------|------|------|
| race_id | str | レースID (YYYYPPNNDDRR) |
| race_date | str | 日付 (YYYY-MM-DD) |
| course_id | str | 競馬場コード (01-10) |
| course_name | str | 競馬場名 |
| race_number | int | レース番号 |
| race_name | str | レース名 |
| race_class | str | クラス |
| surface | str | 芝/ダ |
| distance | int | 距離(m) |
| track_condition | str | 馬場状態 |
| weather | str | 天候 |
| num_entries | int | 出走頭数 |
| finish_position | int | 着順 **[目的変数]** |
| frame_number | int | 枠番 |
| horse_number | int | 馬番 |
| horse_name | str | 馬名 |
| horse_id | str | 馬ID |
| sex | str | 性別 |
| age | int | 年齢 |
| jockey_name | str | 騎手名 |
| jockey_id | str | 騎手ID |
| trainer_name | str | 調教師名 |
| weight_carried | float | 斤量 |
| horse_weight | int | 馬体重 |
| horse_weight_change | int | 体重増減 |
| finish_time_sec | float | タイム(秒) |
| margin | str | 着差 |
| passing_order | str | 通過順 |
| last_3f | float | 上り3F |
| win_odds | float | 単勝オッズ |
| popularity | int | 人気 |
| prize_money | float | 賞金(万円) |

### features.csv（1行 = 1頭 × 1レース、特徴量付き）

事前に利用可能な特徴量のみ（データリーケージなし）。

#### 基本特徴量
- frame_number, horse_number, sex, age, weight_carried
- horse_weight, horse_weight_change
- surface, distance, track_condition, weather, course_id, race_class

#### 過去成績集約特徴量（直近5走）
- avg_finish_last5, best_finish_last5
- win_rate_last5, place_rate_last5
- avg_last3f_last5
- days_since_last_race
- prev_finish_1, prev_finish_2, prev_finish_3

#### 条件別成績
- surface_win_rate, surface_place_rate
- distance_cat_win_rate（短距離/マイル/中距離/長距離）

#### 騎手統計
- jockey_win_rate_365d, jockey_place_rate_365d

#### 目的変数
- finish_position（着順）
- top3（3着以内=1, 4着以下=0）

## 4. モデル

### 二値分類（LightGBM Binary）
- 目的変数: top3 (0/1)
- 評価指標: AUC, 的中率
- 出力: 3着以内確率

### ランキング（LightGBM LambdaRank）
- 目的変数: finish_position（低い=良い）
- グループ: race_id
- 評価指標: NDCG

## 5. データリーケージ防止

### 除外する特徴量
- finish_time_sec, margin, passing_order, last_3f（事後データ）
- win_odds, popularity（市場情報リーケージ）

### 時系列分割
- 学習: ~2024年末
- 検証: 2025年
- 未来のレースデータから特徴量を計算しない
