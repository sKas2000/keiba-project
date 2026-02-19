"""
共通スクレイピングロジック
Playwright ブラウザ管理、リトライ、レート制限
"""
import asyncio

from config.settings import (
    PLAYWRIGHT_VIEWPORT, PLAYWRIGHT_LOCALE, PLAYWRIGHT_TIMEOUT,
    MAX_RETRIES, RETRY_DELAY,
)

try:
    from playwright.async_api import async_playwright
except ImportError:
    async_playwright = None


class BaseScraper:
    """全スクレイパーの基底クラス"""

    def __init__(self, headless=True, debug=True):
        self.headless = headless
        self.debug = debug
        self.pw = None
        self.browser = None
        self.ctx = None
        self.page = None
        self.max_retries = MAX_RETRIES
        self.retry_delay = RETRY_DELAY

    def log(self, msg):
        if self.debug:
            print(f"  [DEBUG] {msg}")

    async def start(self):
        if async_playwright is None:
            raise ImportError(
                "Playwright が未インストールです:\n"
                "  pip install playwright\n"
                "  playwright install chromium"
            )
        self.pw = await async_playwright().start()
        self.browser = await self.pw.chromium.launch(headless=self.headless)
        self.ctx = await self.browser.new_context(
            viewport=PLAYWRIGHT_VIEWPORT,
            locale=PLAYWRIGHT_LOCALE,
        )
        self.page = await self.ctx.new_page()
        self.page.set_default_timeout(PLAYWRIGHT_TIMEOUT)

    async def close(self):
        if self.ctx:
            await self.ctx.close()
        if self.browser:
            await self.browser.close()
        if self.pw:
            await self.pw.stop()

    async def restart(self):
        """ブラウザ再起動（メモリリフレッシュ）"""
        self.log("ブラウザ再起動中...")
        await self.close()
        await asyncio.sleep(2)
        await self.start()
        self.log("ブラウザ再起動完了")

    async def retry(self, coro_func, *args, max_retries=None, delay=None):
        """リトライ付きでコルーチンを実行"""
        max_retries = max_retries or self.max_retries
        delay = delay or self.retry_delay

        for attempt in range(max_retries):
            try:
                return await coro_func(*args)
            except Exception as e:
                self.log(f"  [ERROR] (試行{attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    wait = delay * (2 ** attempt)
                    self.log(f"  -> {wait}秒後にリトライ...")
                    await asyncio.sleep(wait)
                else:
                    self.log("  -> リトライ上限に達しました。")
                    raise

    async def rate_limit(self, delay=None):
        """負荷軽減のための待機"""
        await asyncio.sleep(delay or self.retry_delay)
