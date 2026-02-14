#!/usr/bin/env python3
"""
馬ページから騎手リンクを抽出

過去走データのテーブルに騎手リンクがあるか確認
"""

import asyncio
import sys
import re

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from playwright.async_api import async_playwright


async def extract_jockey_from_horse_page(horse_url: str, jockey_name: str):
    """馬ページから特定の騎手のリンクを抽出"""
    print(f"馬ページ: {horse_url}")
    print(f"探す騎手: {jockey_name}")
    print("=" * 60)

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=False)
        ctx = await browser.new_context(
            viewport={"width": 1280, "height": 900},
            locale="ja-JP",
        )
        page = await ctx.new_page()

        try:
            await page.goto(horse_url, wait_until="domcontentloaded", timeout=30000)
            await asyncio.sleep(2)

            # 競走成績テーブルを探す
            tables = await page.locator("table").all()
            print(f"\nページ内のテーブル数: {len(tables)}")

            for table_idx, table in enumerate(tables):
                # テーブルのヘッダーをチェック
                ths = await table.locator("th").all()
                headers = []
                for th in ths:
                    text = await th.text_content()
                    headers.append(text.strip() if text else "")

                # "騎手" を含むテーブルが競走成績
                if any("騎手" in h for h in headers):
                    print(f"\n[テーブル{table_idx+1}] 競走成績テーブル発見")
                    print(f"  ヘッダー: {headers}")

                    # 騎手列の位置を特定
                    jockey_col_idx = None
                    for i, h in enumerate(headers):
                        if "騎手" in h:
                            jockey_col_idx = i
                            print(f"  騎手列: {i}")
                            break

                    if jockey_col_idx is None:
                        continue

                    # データ行を取得
                    rows = await table.locator("tbody tr").all()
                    if not rows:
                        rows = await table.locator("tr").all()
                        # ヘッダー行をスキップ
                        rows = [r for r in rows if await r.locator("th").count() == 0]

                    print(f"  データ行数: {len(rows)}")

                    # 各行から騎手リンクを抽出
                    print("\n  過去走の騎手リンク:")
                    for row_idx, row in enumerate(rows[:5]):
                        cells = await row.locator("td").all()

                        if jockey_col_idx < len(cells):
                            jockey_cell = cells[jockey_col_idx]

                            # セル内のテキスト
                            cell_text = await jockey_cell.text_content()
                            cell_text = cell_text.strip() if cell_text else ""

                            # セル内の騎手リンク
                            jockey_links = await jockey_cell.locator("a[href*='/jockey/']").all()

                            if jockey_links:
                                for link in jockey_links:
                                    href = await link.get_attribute("href")
                                    link_text = await link.text_content()
                                    print(f"    [{row_idx+1}] {link_text.strip() if link_text else '(なし)'} -> {href}")

                                    # 騎手IDを抽出
                                    match = re.search(r'/jockey/(\d{5})', href)
                                    if match:
                                        jockey_id = match.group(1)
                                        print(f"          騎手ID: {jockey_id}")
                            else:
                                print(f"    [{row_idx+1}] リンクなし: {cell_text}")

                    break

            # 30秒待つ
            print("\n" + "=" * 60)
            print("ブラウザを30秒開いたままにします...")
            await asyncio.sleep(30)

        finally:
            await browser.close()


async def main():
    # テスト用: ドリームコアの馬ページ
    horse_url = "https://db.netkeiba.com/horse/2023107146/"
    jockey_name = "C.ルメール"

    await extract_jockey_from_horse_page(horse_url, jockey_name)


if __name__ == "__main__":
    asyncio.run(main())
