#!/usr/bin/env python3
"""
JRA デバッグツール v2
======================
レース情報（距離・グレード・レース名の位置）と
3連複テーブル構造をダンプする。

使い方:
  python jra_debug2.py

出力: jra_debug2_output.txt
"""

import asyncio
import re
import sys

try:
    from playwright.async_api import async_playwright
except ImportError:
    print("pip install playwright && playwright install chromium")
    sys.exit(1)

OUTPUT_FILE = "jra_debug2_output.txt"
lines = []

def out(msg=""):
    print(msg)
    lines.append(msg)

def save():
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n💾 保存完了: {OUTPUT_FILE}")


async def main():
    out("=" * 60)
    out("JRA デバッグツール v2")
    out("=" * 60)

    pw = await async_playwright().start()
    browser = await pw.chromium.launch(headless=False)
    ctx = await browser.new_context(viewport={"width": 1280, "height": 900}, locale="ja-JP")
    page = await ctx.new_page()
    page.set_default_timeout(15000)

    try:
        # === JRAトップ → オッズ ===
        out("\n[1] JRAトップ → オッズページ")
        await page.goto("https://www.jra.go.jp/", wait_until="domcontentloaded")
        await asyncio.sleep(1)
        await page.get_by_role("link", name="オッズ").first.click()
        await page.wait_for_load_state("domcontentloaded")
        await asyncio.sleep(1)

        # === 開催一覧 ===
        meetings = []
        links = await page.locator("a").all()
        for link in links:
            text = await link.text_content()
            if text and "回" in text and "日" in text:
                meetings.append({"text": text.strip(), "element": link})

        out(f"\n[2] 開催一覧 ({len(meetings)}件):")
        for i, m in enumerate(meetings):
            out(f"  [{i+1}] {m['text']}")

        if not meetings:
            out("❌ 開催なし")
            save()
            return

        # 開催選択
        while True:
            c = input(f"\n開催を選択 (1-{len(meetings)}): ").strip()
            if c.isdigit() and 1 <= int(c) <= len(meetings):
                mi = int(c) - 1
                break

        await meetings[mi]["element"].click()
        await page.wait_for_load_state("domcontentloaded")
        await asyncio.sleep(1)

        # レース選択
        while True:
            c = input("レース番号 (1-12): ").strip()
            if c.isdigit() and 1 <= int(c) <= 12:
                rn = int(c)
                break

        out(f"\n[3] {meetings[mi]['text']} {rn}R を選択")
        await page.get_by_role("link", name=f"{rn}レース").first.click()
        await page.wait_for_load_state("domcontentloaded")
        await asyncio.sleep(1)

        # ==========================================
        # Part A: レース情報の探索
        # ==========================================
        out("\n" + "=" * 60)
        out("Part A: レース情報の探索")
        out("=" * 60)

        # A-1: ページ全体のテキストからキーワード周辺を抜き出す
        out("\n[A-1] ページ全体テキストからキーワード検索")
        body = await page.locator("body").text_content()
        if body:
            # 距離関連: "芝" "ダート" "ダ" + 数字 + "m"
            for pattern, label in [
                (r".{0,30}(芝|ダート|ダ)\s*.{0,20}\d{3,4}\s*[mM].{0,20}", "馬場+距離"),
                (r".{0,20}\d{3,4}\s*[mM].{0,30}", "距離(m)"),
                (r".{0,20}(GⅠ|GⅡ|GⅢ|G1|G2|G3|リステッド|オープン|[123]勝|未勝利|新馬).{0,30}", "グレード"),
                (r".{0,10}(右|左)\s*(外|内)?.{0,20}", "回り"),
            ]:
                matches = re.findall(pattern, body)
                if matches:
                    out(f"  [{label}] ヒット {len(matches) if isinstance(matches[0], str) else len(matches)}件:")
                    seen = set()
                    for m in matches[:10]:
                        s = m if isinstance(m, str) else " ".join(m)
                        s = re.sub(r"\s+", " ", s).strip()
                        if s not in seen and len(s) > 2:
                            seen.add(s)
                            out(f"    「{s}」")
                else:
                    out(f"  [{label}] ヒットなし")

        # A-2: 見出し要素 (h1-h4)
        out("\n[A-2] 見出し要素 (h1-h4)")
        for tag in ["h1", "h2", "h3", "h4"]:
            els = await page.locator(tag).all()
            for el in els:
                text = await el.text_content()
                if text and text.strip():
                    text = re.sub(r"\s+", " ", text.strip())
                    if len(text) < 200:
                        out(f"  <{tag}> {text}")

        # A-3: class/id に race, header, title, info を含む要素
        out("\n[A-3] class/idに race/header/title/info を含む要素")
        for selector in [
            "[class*='race']", "[class*='Race']",
            "[class*='header']", "[class*='Header']",
            "[class*='title']", "[class*='Title']",
            "[class*='info']", "[class*='Info']",
            "[id*='race']", "[id*='Race']",
            "caption",
        ]:
            try:
                els = await page.locator(selector).all()
                for el in els[:5]:
                    tag_name = await el.evaluate("e => e.tagName")
                    cls = await el.evaluate("e => e.className || ''")
                    eid = await el.evaluate("e => e.id || ''")
                    text = await el.text_content()
                    if text:
                        text = re.sub(r"\s+", " ", text.strip())[:120]
                    if text and len(text) > 1:
                        ident = f"{tag_name}"
                        if cls:
                            ident += f".{cls}"
                        if eid:
                            ident += f"#{eid}"
                        out(f"  [{selector}] <{ident}> {text}")
            except Exception:
                pass

        # A-4: 全リンクの中からレース情報っぽいもの
        out("\n[A-4] ページ内リンクテキスト（レース関連）")
        all_links = await page.locator("a").all()
        race_keywords = ["芝", "ダート", "ダ", "特別", "ステークス", "賞", "カップ",
                         "未勝利", "新馬", "1勝", "2勝", "3勝", "オープン",
                         "リステッド", "ハンデ", "定量", "別定", "馬齢"]
        found = 0
        for link in all_links:
            text = await link.text_content()
            if text:
                text = text.strip()
                if any(kw in text for kw in race_keywords) and len(text) < 80:
                    out(f"  リンク: 「{text}」")
                    found += 1
                    if found >= 20:
                        break

        # A-5: 単勝・複勝タブ画面のテーブル上部テキスト
        out("\n[A-5] 単勝・複勝タブのテーブル周辺テキスト")
        await page.get_by_role("link", name="単勝・複勝").first.click()
        await page.wait_for_load_state("domcontentloaded")
        await asyncio.sleep(1)

        # テーブルの前後にある非テーブル要素
        for selector in ["table", "div", "p", "span", "caption", "th"]:
            try:
                els = await page.locator(selector).all()
                for el in els[:30]:
                    text = await el.text_content()
                    if not text:
                        continue
                    text = text.strip()
                    # 距離情報を含む要素を探す
                    if re.search(r"(芝|ダート|ダ)\s*.{0,5}\d{3,4}", text) and len(text) < 100:
                        tag_name = await el.evaluate("e => e.tagName")
                        cls = await el.evaluate("e => e.className || ''")
                        out(f"  <{tag_name} class='{cls}'> {text}")
                    # グレード情報
                    if re.search(r"(GⅠ|GⅡ|GⅢ|[123]勝クラス|オープン|リステッド|未勝利)", text) and len(text) < 100:
                        tag_name = await el.evaluate("e => e.tagName")
                        cls = await el.evaluate("e => e.className || ''")
                        out(f"  <{tag_name} class='{cls}'> {text}")
            except Exception:
                pass

        # ==========================================
        # Part B: 3連複テーブル構造
        # ==========================================
        out("\n" + "=" * 60)
        out("Part B: 3連複テーブル構造")
        out("=" * 60)

        await page.get_by_role("link", name="3連複").first.click()
        await page.wait_for_load_state("domcontentloaded")
        await asyncio.sleep(1)

        out(f"\n[B-1] URL: {page.url}")

        # ページ全体の構造概要
        body_text = await page.locator("body").text_content()
        if body_text:
            # 「1 - 2 - 3」パターンがあるか
            trio_matches = re.findall(r"\d{1,2}\s*[-ー－]\s*\d{1,2}\s*[-ー－]\s*\d{1,2}", body_text)
            out(f"\n[B-2] '馬番-馬番-馬番' パターン: {len(trio_matches)}件")
            if trio_matches:
                for m in trio_matches[:10]:
                    out(f"  {m}")

        # テーブル構造ダンプ
        tables = await page.locator("table").all()
        out(f"\n[B-3] テーブル数: {len(tables)}")

        for ti, table in enumerate(tables):
            # ヘッダー
            ths = await table.locator("th").all()
            headers = []
            for th in ths:
                t = await th.text_content()
                headers.append(t.strip() if t else "")

            rows = await table.locator("tr").all()
            out(f"\n  テーブル{ti} ({len(rows)}行):")
            if headers:
                out(f"    ヘッダー: {headers[:10]}{'...' if len(headers) > 10 else ''}")

            # 最初の5行をダンプ
            dumped = 0
            for row in rows:
                cells = await row.locator("td").all()
                if not cells:
                    continue
                ct = []
                for cell in cells:
                    t = await cell.text_content()
                    ct.append(t.strip() if t else "")

                out(f"    行{dumped}: {ct[:8]}{'...' if len(ct) > 8 else ''}")
                dumped += 1
                if dumped >= 5:
                    out(f"    ... 残り{len(rows) - 5 - len(headers)}行")
                    break

        # B-4: 3連複ページのセレクトボックスやフォーム要素
        out("\n[B-4] フォーム要素・セレクトボックス")
        for selector in ["select", "input[type='text']", "input[type='number']",
                         "button", "[class*='popular']", "[class*='ninki']",
                         "[class*='combo']", "[class*='result']"]:
            try:
                els = await page.locator(selector).all()
                if els:
                    out(f"  {selector}: {len(els)}個")
                    for el in els[:3]:
                        tag = await el.evaluate("e => e.tagName")
                        cls = await el.evaluate("e => e.className || ''")
                        eid = await el.evaluate("e => e.id || ''")
                        text = await el.text_content()
                        text = (text or "").strip()[:60]
                        out(f"    <{tag} class='{cls}' id='{eid}'> {text}")
            except Exception:
                pass

        # B-5: 3連複ページのリンク（人気順/流し/ボックスなどのサブタブ）
        out("\n[B-5] 3連複ページ内のリンク・タブ")
        trio_links = await page.locator("a").all()
        trio_keywords = ["人気", "流し", "ボックス", "軸", "フォーメーション",
                         "全組合せ", "通り", "番号"]
        for link in trio_links:
            text = await link.text_content()
            if text:
                text = text.strip()
                if any(kw in text for kw in trio_keywords) and len(text) < 50:
                    out(f"  リンク: 「{text}」")

        # ==========================================
        # Part C: 馬単テーブル構造（参考）
        # ==========================================
        out("\n" + "=" * 60)
        out("Part C: 馬単テーブル構造（参考）")
        out("=" * 60)

        try:
            await page.get_by_role("link", name="馬単").first.click()
            await page.wait_for_load_state("domcontentloaded")
            await asyncio.sleep(1)

            tables = await page.locator("table").all()
            out(f"\n[C-1] テーブル数: {len(tables)}")

            for ti, table in enumerate(tables[:3]):
                ths = await table.locator("th").all()
                headers = []
                for th in ths:
                    t = await th.text_content()
                    headers.append(t.strip() if t else "")

                rows = await table.locator("tr").all()
                out(f"\n  テーブル{ti} ({len(rows)}行):")
                if headers:
                    out(f"    ヘッダー: {headers[:10]}")

                dumped = 0
                for row in rows:
                    cells = await row.locator("td").all()
                    if not cells:
                        continue
                    ct = []
                    for cell in cells:
                        t = await cell.text_content()
                        ct.append(t.strip() if t else "")
                    out(f"    行{dumped}: {ct[:8]}")
                    dumped += 1
                    if dumped >= 3:
                        break
        except Exception as e:
            out(f"  馬単エラー: {e}")

        out("\n" + "=" * 60)
        out("診断完了")
        out("=" * 60)

    except Exception as e:
        out(f"\n❌ エラー: {e}")
        import traceback
        out(traceback.format_exc())
    finally:
        save()
        await ctx.close()
        await browser.close()
        await pw.stop()


if __name__ == "__main__":
    asyncio.run(main())
