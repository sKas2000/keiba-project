"""
スーパープレミアム会員で利用可能なデータ探索
"""
import asyncio
import sys
import io
from pathlib import Path

# UTF-8出力設定（Windows文字化け対策）
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

sys.path.insert(0, str(Path(__file__).parent))

from playwright.async_api import async_playwright
from config.settings import load_env_var


async def login_netkeiba(page):
    """netkeibaログイン"""
    email = load_env_var("NETKEIBA_EMAIL")
    password = load_env_var("NETKEIBA_PASSWORD")

    if not email or not password:
        print("[ERROR] .envにNETKEIBA_EMAIL/PASSWORDが設定されていません")
        return False

    print("\n=== netkeibaログイン ===")
    await page.goto("https://regist.netkeiba.com/account/?pid=login", wait_until="domcontentloaded")
    await asyncio.sleep(2)

    # メール入力
    await page.locator('input[name="login_id"]').fill(email)
    # パスワード入力
    pw_input = page.locator('input[name="pswd"]')
    if await pw_input.count() == 0:
        pw_input = page.locator('input[type="password"]')
    await pw_input.first.fill(password)

    await asyncio.sleep(0.5)

    # ログインボタン
    login_btn = page.locator('input[type="image"], input[type="submit"], button[type="submit"]')
    await login_btn.first.click()
    await page.wait_for_load_state("domcontentloaded")
    await asyncio.sleep(3)

    if "pid=login" in page.url:
        print("[ERROR] ログイン失敗")
        return False

    print("[OK] ログイン成功")
    return True


async def explore_training_page(page, race_id):
    """調教ページ探索"""
    print(f"\n{'='*60}")
    print(f"=== 調教ページ探索: {race_id} ===")
    print(f"{'='*60}")

    url = f"https://race.netkeiba.com/race/oikiri.html?race_id={race_id}"
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=30000)
        await asyncio.sleep(2)

        # ページタイトル確認
        title = await page.title()
        print(f"ページタイトル: {title}")

        # 調教テーブル探索
        tables = await page.locator("table").all()
        print(f"\n見つかったテーブル数: {len(tables)}")

        for idx, table in enumerate(tables):
            # ヘッダー取得
            ths = await table.locator("th").all()
            if len(ths) > 0:
                headers = []
                for th in ths:
                    text = await th.text_content()
                    headers.append(text.strip() if text else "")

                print(f"\n--- テーブル {idx+1} ---")
                print(f"列数: {len(headers)}")
                print(f"ヘッダー: {headers}")

                # 最初の数行のデータを取得
                rows = await table.locator("tbody tr").all()
                if not rows:
                    rows = await table.locator("tr").all()
                    rows = [r for r in rows if await r.locator("th").count() == 0]

                print(f"データ行数: {len(rows)}")

                for i, row in enumerate(rows[:3]):  # 最初の3行
                    cells = await row.locator("td").all()
                    cell_texts = []
                    for cell in cells:
                        text = await cell.text_content()
                        cell_texts.append(text.strip() if text else "")

                    if cell_texts:
                        print(f"  行{i+1}: {cell_texts}")

        return True

    except Exception as e:
        print(f"[ERROR] 調教ページ探索失敗: {e}")
        return False


async def explore_newspaper_page(page, race_id):
    """競馬新聞ページ探索"""
    print(f"\n{'='*60}")
    print(f"=== 競馬新聞ページ探索: {race_id} ===")
    print(f"{'='*60}")

    url = f"https://race.netkeiba.com/race/newspaper.html?race_id={race_id}"
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=30000)
        await asyncio.sleep(2)

        title = await page.title()
        print(f"ページタイトル: {title}")

        # テーブル探索
        tables = await page.locator("table").all()
        print(f"\n見つかったテーブル数: {len(tables)}")

        for idx, table in enumerate(tables[:5]):  # 最初の5テーブル
            ths = await table.locator("th").all()
            if len(ths) > 0:
                headers = []
                for th in ths:
                    text = await th.text_content()
                    headers.append(text.strip() if text else "")

                print(f"\n--- テーブル {idx+1} ---")
                print(f"ヘッダー: {headers}")

                # データサンプル
                rows = await table.locator("tbody tr").all()
                if not rows:
                    rows = await table.locator("tr").all()
                    rows = [r for r in rows if await r.locator("th").count() == 0]

                print(f"データ行数: {len(rows)}")

                for i, row in enumerate(rows[:2]):
                    cells = await row.locator("td").all()
                    cell_texts = []
                    for cell in cells:
                        text = await cell.text_content()
                        cell_texts.append(text.strip()[:50] if text else "")  # 長すぎる場合は切り詰め

                    if cell_texts:
                        print(f"  行{i+1}: {cell_texts}")

        # 印（◎○▲△等）を探す
        marks = await page.locator("span:has-text('◎'), span:has-text('○'), span:has-text('▲')").all()
        print(f"\n見つかった印: {len(marks)}個")

        return True

    except Exception as e:
        print(f"[ERROR] 競馬新聞ページ探索失敗: {e}")
        return False


async def explore_horse_past_races(page, horse_id):
    """馬の過去走テーブル（29列構造）を探索"""
    print(f"\n{'='*60}")
    print(f"=== 馬の過去走テーブル探索: {horse_id} ===")
    print(f"{'='*60}")

    url = f"https://db.netkeiba.com/horse/{horse_id}/"
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=30000)
        await asyncio.sleep(2)

        title = await page.title()
        print(f"ページタイトル: {title}")

        # 過去走テーブルを探す
        tables = await page.locator("table").all()

        for idx, table in enumerate(tables):
            ths = await table.locator("th").all()
            headers = []
            for th in ths:
                text = await th.text_content()
                headers.append(text.strip() if text else "")

            # 「着順」列があるテーブルを探す
            if any("着順" in h for h in headers):
                print(f"\n--- 過去走テーブル発見（テーブル{idx+1}） ---")
                print(f"列数: {len(headers)}")
                print(f"\n全ヘッダー:")
                for i, h in enumerate(headers):
                    print(f"  [{i:2d}] {h}")

                # データサンプル（最初の1行）
                rows = await table.locator("tbody tr").all()
                if not rows:
                    rows = await table.locator("tr").all()
                    rows = [r for r in rows if await r.locator("th").count() == 0]

                if rows:
                    print(f"\nデータサンプル（1行目）:")
                    cells = await rows[0].locator("td").all()
                    for i, cell in enumerate(cells):
                        text = await cell.text_content()
                        text = text.strip() if text else ""
                        if i < len(headers):
                            print(f"  [{i:2d}] {headers[i]:12s} = {text[:40]}")

                break

        return True

    except Exception as e:
        print(f"[ERROR] 馬ページ探索失敗: {e}")
        return False


async def main():
    """メイン処理"""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()

        try:
            # ログイン
            if not await login_netkeiba(page):
                return

            # 複数のrace_idを試す（最近のレース）
            race_ids = [
                "202605010101",  # 2026年5月1回東京1日1R
                "202605020701",  # 2026年5月2回中山7日1R
                "202505010101",  # 2025年5月1回東京1日1R
                "202602020711",  # 2026年2月2回東京7日11R（最近のGI/重賞）
            ]

            # 1. 調教ページ探索
            for race_id in race_ids:
                if await explore_training_page(page, race_id):
                    break  # 成功したら次へ
                await asyncio.sleep(1)

            # 2. 競馬新聞ページ探索
            for race_id in race_ids:
                if await explore_newspaper_page(page, race_id):
                    break
                await asyncio.sleep(1)

            # 3. 馬の過去走テーブル探索（実在する馬ID）
            horse_ids = [
                "2022104038",  # ドウデュース
                "2019105039",  # イクイノックス
                "2020104895",  # ソールオリエンス
            ]

            for horse_id in horse_ids:
                if await explore_horse_past_races(page, horse_id):
                    break
                await asyncio.sleep(1)

            print("\n" + "="*60)
            print("=== 探索完了 ===")
            print("="*60)

        except Exception as e:
            print(f"\n[ERROR] エラー発生: {e}")
            import traceback
            traceback.print_exc()

        finally:
            await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
