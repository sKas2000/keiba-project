#!/usr/bin/env python3
"""
騎手検索のデバッグスクリプト

特定の騎手名で検索し、検索結果ページのHTMLとリンクを詳しく調査する
"""

import asyncio
import sys

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from playwright.async_api import async_playwright


async def debug_jockey_search(jockey_name: str):
    """騎手検索をデバッグ"""
    print(f"騎手検索デバッグ: {jockey_name}")
    print("=" * 60)

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=False)  # headless=False で確認
        ctx = await browser.new_context(
            viewport={"width": 1280, "height": 900},
            locale="ja-JP",
        )
        page = await ctx.new_page()

        try:
            # 検索ページに移動
            print("\n[1] 検索ページに移動")
            await page.goto("https://db.netkeiba.com/?pid=jockey_search_detail", wait_until="domcontentloaded", timeout=30000)
            await asyncio.sleep(2)

            # フォーム数を確認
            forms = await page.locator('form').all()
            print(f"\n[2] ページ内のフォーム数: {len(forms)}")

            # 各フォームの詳細を確認
            for idx, form in enumerate(forms):
                print(f"\n--- フォーム {idx+1} ---")
                hidden_inputs = await form.locator('input[type="hidden"]').all()
                for h in hidden_inputs:
                    name = await h.get_attribute('name')
                    value = await h.get_attribute('value')
                    print(f"  hidden: {name}={value}")

            # pid=jockey_list のフォームを探す
            jockey_form = None
            for form in forms:
                hidden_inputs = await form.locator('input[type="hidden"]').all()
                for h in hidden_inputs:
                    name = await h.get_attribute('name')
                    value = await h.get_attribute('value')
                    if name == 'pid' and value == 'jockey_list':
                        jockey_form = form
                        print(f"\n[3] 騎手検索フォーム発見")
                        break
                if jockey_form:
                    break

            if not jockey_form:
                print("\n[ERROR] 騎手検索フォームが見つかりません")
                return

            # フォームに入力
            print(f"\n[4] フォームに騎手名を入力: {jockey_name}")
            word_input = jockey_form.locator('input[name="word"]')
            await word_input.fill(jockey_name)

            # 入力値を確認
            input_value = await word_input.input_value()
            print(f"  → 入力値確認: '{input_value}'")

            # submit
            print("\n[5] 検索実行")
            submit_button = jockey_form.locator('input[type="submit"]')
            await submit_button.click()
            await asyncio.sleep(3)

            # 現在のURL
            current_url = page.url
            print(f"\n[6] 検索後のURL: {current_url}")

            # ページタイトル
            title = await page.title()
            print(f"  ページタイトル: {title}")

            # 検索結果のリンクを取得
            print("\n[7] 検索結果のリンク取得")
            links = await page.locator("a[href*='/jockey/']").all()
            print(f"  見つかったリンク数: {len(links)}")

            print("\n  最初の10件:")
            for idx, link in enumerate(links[:10]):
                href = await link.get_attribute("href")
                text = await link.text_content()
                print(f"    [{idx+1}] {href}")
                print(f"         テキスト: {text.strip() if text else '(なし)'}")

            # パターンマッチング確認
            print("\n[8] URLパターンマッチング")
            import re
            for idx, link in enumerate(links[:10]):
                href = await link.get_attribute("href")
                if re.search(r'/jockey/\d{5}$', href):
                    print(f"  ✓ マッチ: {href}")
                else:
                    print(f"  ✗ 不一致: {href}")

            # 30秒待って確認（手動確認用）
            print("\n[9] ブラウザを30秒間開いたままにします（手動確認用）")
            await asyncio.sleep(30)

        finally:
            await browser.close()


async def main():
    if len(sys.argv) < 2:
        # デフォルトテスト
        test_jockeys = ["C.ルメール", "武 豊", "池添 謙一"]
        for jockey in test_jockeys:
            await debug_jockey_search(jockey)
            print("\n" + "=" * 60 + "\n")
    else:
        jockey_name = sys.argv[1]
        await debug_jockey_search(jockey_name)


if __name__ == "__main__":
    asyncio.run(main())
