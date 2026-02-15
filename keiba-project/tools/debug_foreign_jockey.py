"""外国人騎手の検索テスト"""
import asyncio
import sys
import io
import re

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from playwright.async_api import async_playwright

async def main():
    # テスト: 色々な名前パターンで検索
    test_patterns = [
        "T.ハマーハンセン",      # JRAの表記
        "ハマーハンセン",         # イニシャルなし
        "Ｔ．ハマーハンセン",    # 全角
        "ハマー",                # 部分一致
    ]

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        for name in test_patterns:
            print(f"\n{'='*50}")
            print(f"  検索: {name}")
            print(f"{'='*50}")

            await page.goto("https://db.netkeiba.com/?pid=jockey_search_detail",
                          wait_until="domcontentloaded", timeout=30000)
            await asyncio.sleep(2)

            forms = await page.locator('form').all()
            jockey_form = None
            for form in forms:
                hidden_inputs = await form.locator('input[type="hidden"]').all()
                for h in hidden_inputs:
                    n = await h.get_attribute('name')
                    v = await h.get_attribute('value')
                    if n == 'pid' and v == 'jockey_list':
                        jockey_form = form
                        break
                if jockey_form:
                    break

            if jockey_form:
                word_input = jockey_form.locator('input[name="word"]')
                await word_input.fill(name)
                submit = jockey_form.locator('input[type="submit"]')
                await submit.click()
                await asyncio.sleep(3)

                current_url = page.url
                print(f"  URL: {current_url}")

                # リダイレクトチェック
                jockey_match = re.search(r'/jockey/(\d{5})/?$', current_url)
                if jockey_match:
                    print(f"  → 直接遷移！ ID: {jockey_match.group(1)}")

                links = await page.locator("a[href*='/jockey/']").all()
                print(f"  リンク数: {len(links)}")
                for idx, link in enumerate(links[:10]):
                    href = await link.get_attribute("href")
                    text = await link.text_content()
                    text = text.strip() if text else ''
                    if re.search(r'/jockey/\d{5}', href):
                        print(f"    [{idx+1}] {href} | {text}")

        await browser.close()

asyncio.run(main())
