#!/usr/bin/env python3
"""
騎手検索のURL・フォームパラメータ調査

検索結果がソートではなくフィルタされる方法を探す
"""

import asyncio
import sys

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from playwright.async_api import async_playwright


async def test_jockey_search_methods(jockey_name: str):
    """様々な検索方法をテスト"""
    print(f"騎手検索テスト: {jockey_name}")
    print("=" * 60)

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=False)
        ctx = await browser.new_context(
            viewport={"width": 1280, "height": 900},
            locale="ja-JP",
        )
        page = await ctx.new_page()

        try:
            # ===============================================
            # 方法1: 現在の方法（フォーム入力）
            # ===============================================
            print("\n[方法1] フォーム入力（現状）")
            print("-" * 60)

            await page.goto("https://db.netkeiba.com/?pid=jockey_search_detail", wait_until="domcontentloaded", timeout=30000)
            await asyncio.sleep(2)

            # フォームを探す
            forms = await page.locator('form').all()
            jockey_form = None
            for form in forms:
                hidden_inputs = await form.locator('input[type="hidden"]').all()
                for h in hidden_inputs:
                    name = await h.get_attribute('name')
                    value = await h.get_attribute('value')
                    if name == 'pid' and value == 'jockey_list':
                        jockey_form = form
                        break
                if jockey_form:
                    break

            if jockey_form:
                # フォーム内のすべてのhidden inputを確認
                all_hidden = await jockey_form.locator('input[type="hidden"]').all()
                print("  フォームのhidden inputs:")
                for h in all_hidden:
                    name = await h.get_attribute('name')
                    value = await h.get_attribute('value')
                    print(f"    {name}={value}")

                # フォーム入力
                word_input = jockey_form.locator('input[name="word"]')
                await word_input.fill(jockey_name)

                # submitの前にフォームのaction URLを確認
                action = await jockey_form.get_attribute('action')
                method = await jockey_form.get_attribute('method')
                print(f"  フォームaction: {action}")
                print(f"  フォームmethod: {method}")

                # submit
                submit_button = jockey_form.locator('input[type="submit"]')
                await submit_button.click()
                await asyncio.sleep(3)

                # 結果のURLを確認
                result_url = page.url
                print(f"  結果URL: {result_url}")

                # リンク数
                links = await page.locator("a[href*='/jockey/']").all()
                print(f"  騎手リンク数: {len(links)}")

                # 最初の5件を表示
                print("  上位5件:")
                import re
                for idx, link in enumerate(links[:10]):
                    href = await link.get_attribute("href")
                    if re.search(r'/jockey/\d{5}$', href):
                        text = await link.text_content()
                        print(f"    [{idx+1}] {text.strip() if text else '(なし)'} -> {href}")

            # ===============================================
            # 方法2: URL直接指定（sortパラメータなし）
            # ===============================================
            print("\n[方法2] URL直接指定（sortなし）")
            print("-" * 60)

            from urllib.parse import quote
            encoded_name = quote(jockey_name.encode('euc-jp'), safe='')
            test_url = f"https://db.netkeiba.com/?pid=jockey_list&word={encoded_name}&list=100"
            print(f"  テストURL: {test_url}")

            await page.goto(test_url, wait_until="domcontentloaded", timeout=30000)
            await asyncio.sleep(3)

            result_url = page.url
            print(f"  結果URL: {result_url}")

            links = await page.locator("a[href*='/jockey/']").all()
            print(f"  騎手リンク数: {len(links)}")

            print("  上位10件:")
            import re
            for idx, link in enumerate(links[:15]):
                href = await link.get_attribute("href")
                if re.search(r'/jockey/\d{5}$', href):
                    text = await link.text_content()
                    print(f"    [{idx+1}] {text.strip() if text else '(なし)'} -> {href}")

            # ===============================================
            # 方法3: URL直接指定（sort=nameで名前順）
            # ===============================================
            print("\n[方法3] URL直接指定（sort=name）")
            print("-" * 60)

            test_url = f"https://db.netkeiba.com/?pid=jockey_list&word={encoded_name}&sort=name&list=100"
            print(f"  テストURL: {test_url}")

            await page.goto(test_url, wait_until="domcontentloaded", timeout=30000)
            await asyncio.sleep(3)

            result_url = page.url
            print(f"  結果URL: {result_url}")

            links = await page.locator("a[href*='/jockey/']").all()
            print(f"  騎手リンク数: {len(links)}")

            print("  上位10件:")
            for idx, link in enumerate(links[:15]):
                href = await link.get_attribute("href")
                if re.search(r'/jockey/\d{5}$', href):
                    text = await link.text_content()
                    print(f"    [{idx+1}] {text.strip() if text else '(なし)'} -> {href}")

            # 30秒待つ
            print("\n" + "=" * 60)
            print("ブラウザを30秒開いたままにします...")
            await asyncio.sleep(30)

        finally:
            await browser.close()


async def main():
    if len(sys.argv) < 2:
        jockey_name = "北村 友一"
    else:
        jockey_name = sys.argv[1]

    await test_jockey_search_methods(jockey_name)


if __name__ == "__main__":
    asyncio.run(main())
