"""
共通スクレイピングロジック
Playwright ブラウザ管理、リトライ、レート制限
"""
import asyncio
import logging

from config.settings import (
    PLAYWRIGHT_VIEWPORT, PLAYWRIGHT_LOCALE, PLAYWRIGHT_TIMEOUT,
    MAX_RETRIES, RETRY_DELAY, load_env_var,
)

logger = logging.getLogger("keiba.scraping.base")

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
        self._netkeiba_logged_in = False

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
        # 再起動後にnetkeiba再ログイン
        if self._netkeiba_logged_in:
            await self.login_netkeiba()
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

    # ----------------------------------------------------------
    # netkeiba ログイン
    # ----------------------------------------------------------

    async def login_netkeiba(self) -> bool:
        """netkeibaにログイン（.envに認証情報がある場合のみ）

        Returns:
            True: ログイン成功, False: 認証情報なし or 失敗
        """
        email = load_env_var("NETKEIBA_EMAIL")
        password = load_env_var("NETKEIBA_PASSWORD")
        if not email or not password:
            return False

        try:
            self.log("netkeibaログイン中...")
            await self.page.goto(
                "https://regist.netkeiba.com/account/?pid=login",
                wait_until="domcontentloaded", timeout=30000,
            )
            await asyncio.sleep(3)

            # メールアドレス入力
            email_input = self.page.locator('input[name="login_id"]')
            if await email_input.count() > 0:
                await email_input.fill(email)
            else:
                self.log("  [!] メール入力フィールドが見つかりません")
                return False

            # パスワード入力
            pw_input = self.page.locator('input[name="pswd"]')
            if await pw_input.count() == 0:
                pw_input = self.page.locator('input[type="password"]')
            if await pw_input.count() > 0:
                await pw_input.first.fill(password)
            else:
                self.log("  [!] パスワード入力フィールドが見つかりません")
                return False

            # ログインボタンクリック（type="image" のログインボタン）
            await asyncio.sleep(0.5)
            login_btn = self.page.locator(
                'input[type="image"], input[type="submit"], '
                'button[type="submit"]'
            )
            if await login_btn.count() > 0:
                await login_btn.first.click()
            else:
                await self.page.keyboard.press("Enter")

            # ページ遷移を待機
            await self.page.wait_for_load_state("domcontentloaded")
            await asyncio.sleep(3)

            # ログイン成功確認
            current_url = self.page.url
            # ログインページにまだいる（pid=login）なら失敗
            if "pid=login" in current_url:
                self.log("  [!] netkeibaログイン失敗（認証情報を確認してください）")
                logger.warning("netkeibaログイン失敗")
                return False

            self._netkeiba_logged_in = True
            self.log("  [OK] netkeibaログイン成功")
            logger.info("netkeibaログイン成功 (email=%s)", email[:3] + "***")
            return True

        except Exception as e:
            self.log(f"  [ERROR] netkeibaログインエラー: {e}")
            logger.error("netkeibaログインエラー: %s", e)
            return False
