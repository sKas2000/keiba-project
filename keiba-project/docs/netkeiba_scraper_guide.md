# netkeiba_scraper.py ガイド

## 概要
jra_scraper.pyで取得したinput.jsonに、netkeibaから以下のデータを追加します：
- 各馬の過去走データ（最新4走）
- 騎手の年間成績（勝率・複勝率）

## 使い方

### 1. 前提条件
- jra_scraper.pyでinput.jsonを作成済み
- Playwrightがインストール済み

### 2. 実行
```bash
cd tools
python netkeiba_scraper.py ../data/races/YYYYMMDD_会場_レース名/input.json
```

### 3. 出力
元のディレクトリに `*_enriched_input.json` が生成されます。

**追加されるデータ構造:**
```json
{
  "horses": [
    {
      "num": 1,
      "name": "馬名",
      "jockey": "騎手名",
      "past_races": [
        {
          "date": "2026/01/15",
          "venue": "京都",
          "race": "3歳未勝利",
          "distance": "1400",
          "surface": "ダ",
          "finish": 2,
          "margin": "0.3",
          "time": "1:25.4",
          "last3f": "37.2",
          "position": "2-2"
        }
      ],
      "jockey_stats": {
        "win_rate": 0.185,
        "place_rate": 0.412,
        "wins": 45,
        "races": 243
      }
    }
  ]
}
```

## 取得データの詳細

### 過去走データ (past_races)
- `date`: レース日
- `venue`: 競馬場
- `race`: レース名
- `distance`: 距離（メートル）
- `surface`: 馬場（芝/ダ/障）
- `finish`: 着順
- `margin`: 着差
- `time`: タイム
- `last3f`: 上がり3F
- `position`: 通過順位

### 騎手成績 (jockey_stats)
- `win_rate`: 勝率（0.0〜1.0）
- `place_rate`: 複勝率（0.0〜1.0）
- `wins`: 勝利数
- `races`: 出走数

## 注意事項

### 1. 実行時間
- 1頭あたり約3〜5秒（検索+データ取得）
- 16頭のレースで約1〜2分

### 2. ネットワークエラー
- netkeiba側の負荷制限により失敗する可能性あり
- 失敗した馬は `past_races: []` となる
- 再実行時は成功する場合が多い

### 3. データの精度
- netkeibaのHTML構造変更により取得失敗する可能性あり
- 取得データは必ず目視確認を推奨

### 4. 騎手名のキャッシュ
- 同じ騎手を複数回検索しないようキャッシュを使用
- 実行中のメモリに保持（再起動で消去）

## トラブルシューティング

### Q: "馬が見つかりません" が表示される
A: 馬名の表記が完全一致しない可能性があります。netkeibaでの正式名称を確認してください。

### Q: 過去走データが取得できない
A: netkeibaのHTML構造が変更された可能性があります。`scrape_horse_past_races` 関数のセレクタを調整してください。

### Q: 騎手の成績が0.0になる
A: 騎手ページの構造が想定と異なる可能性があります。`search_jockey` 関数の正規表現を調整してください。

## 今後の改善予定

- [ ] レースIDから直接取得（手動入力なし）
- [ ] 同コース・同距離の成績集計
- [ ] 調教タイムの取得
- [ ] 馬体重推移のグラフ化
- [ ] リトライ機能（エラー時の自動再試行）

## バージョン履歴

### v0.1 (2026-02-14)
- 初版リリース
- 馬名検索による過去走データ取得
- 騎手の年間成績取得
