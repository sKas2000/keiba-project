"""騎手検索のデバッグスクリプト - フォーム送信 vs URL直接アクセス"""
import asyncio
import sys
import io
import re
import urllib.parse

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from playwright.async_api import async_playwright

async def main():
    test_names = ["武豊", "池添謙一", "西村淳也"]

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        for name in test_names:
            print(f"\n{'='*60}")
            print(f"  検索テスト: {name}")
            print(f"{'='*60}")

            # 方法1: フォーム送信
            print("\n--- 方法1: フォーム送信 ---")
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

                print(f"  遷移先URL: {page.url}")
                links = await page.locator("a[href*='/jockey/']").all()
                print(f"  騎手リンク数: {len(links)}")
                for idx, link in enumerate(links[:10]):
                    href = await link.get_attribute("href")
                    text = await link.text_content()
                    text = text.strip() if text else ''
                    match = "✓" if re.search(r'/jockey/\d{5}$', href) else " "
                    print(f"    [{match}] {href} | {text}")

            # 方法2: URL直接アクセス（word= パラメータ）
            print(f"\n--- 方法2: URL直接 (?pid=jockey_list&word=) ---")
            encoded = urllib.parse.quote(name)
            url2 = f"https://db.netkeiba.com/?pid=jockey_list&word={encoded}"
            await page.goto(url2, wait_until="domcontentloaded", timeout=30000)
            await asyncio.sleep(2)
            print(f"  実際のURL: {page.url}")
            links2 = await page.locator("a[href*='/jockey/']").all()
            print(f"  騎手リンク数: {len(links2)}")
            for idx, link in enumerate(links2[:10]):
                href = await link.get_attribute("href")
                text = await link.text_content()
                text = text.strip() if text else ''
                match = "✓" if re.search(r'/jockey/\d{5}$', href) else " "
                print(f"    [{match}] {href} | {text}")

            # 方法3: netkeiba検索ページ（サイト内検索）
            print(f"\n--- 方法3: サイト内検索 (?pid=search_list) ---")
            url3 = f"https://db.netkeiba.com/?pid=search_list&word={encoded}&search_type=jockey"
            await page.goto(url3, wait_until="domcontentloaded", timeout=30000)
            await asyncio.sleep(2)
            print(f"  実際のURL: {page.url}")
            links3 = await page.locator("a[href*='/jockey/']").all()
            print(f"  騎手リンク数: {len(links3)}")
            for idx, link in enumerate(links3[:10]):
                href = await link.get_attribute("href")
                text = await link.text_content()
                text = text.strip() if text else ''
                match = "✓" if re.search(r'/jockey/\d{5}$', href) else " "
                print(f"    [{match}] {href} | {text}")

            # 方法4: 騎手IDを直接推測してアクセス
            # netkeibaの騎手検索API (suggest)
            print(f"\n--- 方法4: suggest API ---")
            url4 = f"https://race.netkeiba.com/api/api_get_suggest.html?word={encoded}&type=jockey"
            try:
                resp = await page.goto(url4, wait_until="domcontentloaded", timeout=10000)
                body = await page.text_content("body")
                print(f"  レスポンス(先頭300文字): {body[:300] if body else '(空)'}")
            except Exception as e:
                print(f"  エラー: {e}")

        await browser.close()

asyncio.run(main())
