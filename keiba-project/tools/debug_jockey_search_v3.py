#!/usr/bin/env python3
"""
騎手詳細検索のデバッグ

jockey_search_detail を使った詳細検索を試す
"""

import asyncio
import sys
import re

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from playwright.async_api import async_playwright


async def test_detailed_search(jockey_name: str):
    """騎手詳細検索をテスト"""
    print(f"騎手詳細検索: {jockey_name}")
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
            # 詳細検索フォームを使用
            # ===============================================
            print("\n[詳細検索] pid=jockey_search_detail")
            print("-" * 60)

            await page.goto("https://db.netkeiba.com/?pid=jockey_search_detail", wait_until="domcontentloaded", timeout=30000)
            await asyncio.sleep(2)

            # ページのすべてのフォームを表示
            forms = await page.locator('form').all()
            print(f"フォーム数: {len(forms)}")

            # 1番目のフォーム（詳細検索フォーム）を使用
            if forms:
                detail_form = forms[0]

                print("\nフォームの全input要素:")
                all_inputs = await detail_form.locator('input, select').all()
                for inp in all_inputs:
                    tag = await inp.evaluate('el => el.tagName')
                    inp_type = await inp.get_attribute('type')
                    name = await inp.get_attribute('name')
                    value = await inp.get_attribute('value')
                    print(f"  {tag} type={inp_type} name={name} value={value}")

                # フリーワード入力欄を探す
                print(f"\n騎手名を入力: {jockey_name}")

                # 方法1: name="free_word" を試す
                free_word_inputs = await detail_form.locator('input[name="free_word"]').all()
                if free_word_inputs:
                    await free_word_inputs[0].fill(jockey_name)
                    print("  → free_word フィールドに入力")
                else:
                    # 方法2: 最初のtext inputに入力
                    text_inputs = await detail_form.locator('input[type="text"]').all()
                    if text_inputs:
                        await text_inputs[0].fill(jockey_name)
                        print("  → 最初のtext inputに入力")

                # submitボタンをクリック
                await asyncio.sleep(1)
                submit_buttons = await detail_form.locator('input[type="submit"], button[type="submit"]').all()
                if submit_buttons:
                    await submit_buttons[0].click()
                    print("  → 検索実行")
                    await asyncio.sleep(3)

                    # 検索結果
                    result_url = page.url
                    print(f"\n結果URL: {result_url}")

                    # タイトル
                    title = await page.title()
                    print(f"タイトル: {title}")

                    # 騎手リンク
                    links = await page.locator("a[href*='/jockey/']").all()
                    print(f"騎手リンク数: {len(links)}")

                    print("\n上位15件:")
                    for idx, link in enumerate(links[:20]):
                        href = await link.get_attribute("href")
                        if re.search(r'/jockey/\d{5}', href):
                            text = await link.text_content()
                            print(f"  [{idx+1}] {text.strip() if text else '(なし)'} -> {href}")

            # 30秒待つ
            print("\n" + "=" * 60)
            print("ブラウザを30秒開いたままにします（手動確認用）...")
            await asyncio.sleep(30)

        finally:
            await browser.close()


async def main():
    if len(sys.argv) < 2:
        jockey_name = "北村 友一"
    else:
        jockey_name = sys.argv[1]

    await test_detailed_search(jockey_name)


if __name__ == "__main__":
    asyncio.run(main())
