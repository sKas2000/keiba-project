#!/usr/bin/env python3
"""
騎手情報抽出のテスト

過去走データから騎手名・IDが正しく抽出されているか確認
"""

import asyncio
import json
import sys
import re

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from playwright.async_api import async_playwright


async def test_jockey_extraction():
    """馬ページから騎手情報を抽出"""
    horse_url = "https://db.netkeiba.com/horse/2023107146/"  # ドリームコア

    print(f"テスト対象: {horse_url}")
    print("=" * 60)

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=False)
        ctx = await browser.new_context(locale="ja-JP")
        page = await ctx.new_page()

        try:
            await page.goto(horse_url, wait_until="domcontentloaded", timeout=30000)
            await asyncio.sleep(2)

            # 競走成績テーブルを探す
            tables = await page.locator("table").all()

            for table in tables:
                ths = await table.locator("th").all()
                headers = []
                for th in ths:
                    text = await th.text_content()
                    headers.append(text.strip() if text else "")

                if any("着順" in h for h in headers):
                    print(f"\n競走成績テーブル発見")
                    print(f"ヘッダー: {headers}")

                    # 騎手列を探す
                    jockey_col = None
                    for i, h in enumerate(headers):
                        if "騎手" in h:
                            jockey_col = i
                            print(f"騎手列: {i}")
                            break

                    if jockey_col is None:
                        print("騎手列が見つかりません")
                        continue

                    # データ行を取得
                    rows = await table.locator("tbody tr").all()
                    if not rows:
                        rows = await table.locator("tr").all()
                        rows = [r for r in rows if await r.locator("th").count() == 0]

                    print(f"\n過去走データ（上位5件）:")
                    for idx, row in enumerate(rows[:5]):
                        cells = await row.locator("td").all()

                        if jockey_col < len(cells):
                            jockey_cell = cells[jockey_col]

                            # 騎手名（テキスト）
                            jockey_text = await jockey_cell.text_content()
                            jockey_name = jockey_text.strip() if jockey_text else ""

                            # 騎手リンク
                            jockey_links = await jockey_cell.locator("a[href*='/jockey/']").all()
                            jockey_id = ""
                            jockey_href = ""

                            if jockey_links:
                                href = await jockey_links[0].get_attribute("href")
                                jockey_href = href
                                # IDを抽出
                                match = re.search(r'/jockey/(?:result/recent/)?(\d{5})', href)
                                if match:
                                    jockey_id = match.group(1)

                            print(f"  [{idx+1}] 騎手名: '{jockey_name}'")
                            print(f"       リンク: {jockey_href}")
                            print(f"       ID: {jockey_id}")

                    break

            # 30秒待つ
            print("\n" + "=" * 60)
            print("ブラウザを30秒開いたままにします...")
            await asyncio.sleep(30)

        finally:
            await browser.close()


if __name__ == "__main__":
    asyncio.run(test_jockey_extraction())
