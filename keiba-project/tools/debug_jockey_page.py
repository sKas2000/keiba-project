"""騎手ページのHTML構造を確認するデバッグスクリプト"""
import asyncio
import sys
import io

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from playwright.async_api import async_playwright

async def main():
    # テスト対象: 高杉吏麒 (01213), C.ルメール (05339)
    jockey_ids = [("01213", "高杉吏麒"), ("05339", "C.ルメール")]

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        for jid, name in jockey_ids:
            url = f"https://db.netkeiba.com/jockey/{jid}"
            print(f"\n{'='*60}")
            print(f"  {name} ({jid}) - {url}")
            print(f"{'='*60}")

            await page.goto(url, wait_until="domcontentloaded", timeout=30000)
            await asyncio.sleep(2)

            # 全テーブルを確認
            tables = await page.locator("table").all()
            print(f"\nテーブル数: {len(tables)}")

            for ti, table in enumerate(tables):
                text = await table.text_content()
                # 勝率を含むテーブルだけ詳細表示
                if "勝率" not in text:
                    continue

                print(f"\n--- テーブル {ti} (勝率を含む) ---")

                # ヘッダー行(th)を確認
                ths = await table.locator("th").all()
                th_texts = []
                for th in ths:
                    t = await th.text_content()
                    th_texts.append(t.strip() if t else "")
                print(f"ヘッダー(th): {th_texts}")

                # 全行を確認
                rows = await table.locator("tr").all()
                print(f"行数: {len(rows)}")
                for ri, row in enumerate(rows[:5]):  # 最初の5行だけ
                    cells = await row.locator("td").all()
                    cell_texts = []
                    for cell in cells:
                        t = await cell.text_content()
                        cell_texts.append(t.strip() if t else "")
                    if cell_texts:
                        print(f"  行{ri}: {cell_texts}")

                # text_content()の最初の200文字
                print(f"  text_content(): {text[:300]}")

        await browser.close()

asyncio.run(main())
