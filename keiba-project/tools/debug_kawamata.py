"""川又賢治の騎手ページ調査"""
import asyncio
import sys
import io

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from playwright.async_api import async_playwright

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        url = "https://db.netkeiba.com/jockey/01166/"
        print(f"URL: {url}")
        await page.goto(url, wait_until="domcontentloaded", timeout=30000)
        await asyncio.sleep(2)

        tables = await page.locator("table").all()
        print(f"\nテーブル数: {len(tables)}")

        for ti, table in enumerate(tables):
            ths = await table.locator("th").all()
            if not ths:
                continue

            header_texts = []
            for th in ths:
                t = await th.text_content()
                header_texts.append(t.strip() if t else "")

            if "勝率" not in header_texts:
                continue

            print(f"\n--- テーブル {ti} (勝率を含む) ---")
            print(f"ヘッダー: {header_texts}")

            # 騎乗回数のカラムを探す
            win_rate_idx = -1
            races_idx = -1
            for i, ht in enumerate(header_texts):
                if ht == "勝率":
                    win_rate_idx = i
                elif "騎乗回数" in ht or ht == "回数":
                    races_idx = i
            print(f"勝率カラム: {win_rate_idx}, 騎乗回数カラム: {races_idx}")

            rows = await table.locator("tr").all()
            print(f"行数: {len(rows)}")
            for ri, row in enumerate(rows):
                cells = await row.locator("td").all()
                if not cells:
                    continue
                cell_texts = []
                for cell in cells:
                    t = await cell.text_content()
                    cell_texts.append(t.strip() if t else "")
                print(f"  行{ri}: {cell_texts[:8]}...")  # 最初の8セルだけ

        await browser.close()

asyncio.run(main())
