"""
RaceMonitor: オッズ監視サーバーのメインクラス
各フェーズのロジックはMixinで分離
"""
import logging
from datetime import datetime

from config.settings import RACES_DIR
from src.notify import send_notify, get_webhook_url

from .scanner import ScannerMixin
from .enricher import EnricherMixin
from .executor import ExecutorMixin
from .results import ResultsMixin

_logger = logging.getLogger("keiba.monitor")


class RaceMonitor(ScannerMixin, EnricherMixin, ExecutorMixin, ResultsMixin):
    """レース発走時刻に合わせたスケジュール実行モニター"""

    def __init__(self, before: int = 5, token: str = None,
                 headless: bool = True, venue_filter: str = None):
        self.before = before
        self.webhook_url = token or get_webhook_url()
        self.headless = headless
        self.venue_filter = (
            set(venue_filter.split(",")) if venue_filter else None
        )

        self.monitor_dir = (
            RACES_DIR / "monitor" / datetime.now().strftime("%Y%m%d")
        )
        self.races = {}          # race_key -> {data, enriched, race_info, ...}
        self.meeting_info = []   # [(index, text, venue), ...]
        self.schedule = []       # [(trigger_dt, post_dt, race_key), ...] sorted

    def log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] {msg}")
        _logger.info(msg)

    async def run(self):
        """メインループ"""
        if not self.webhook_url:
            print("=" * 60)
            print("[ERROR] Discord Webhook URLが未設定")
            print()
            print("設定方法:")
            print("  1. Discordサーバーのチャンネル設定→連携サービス→ウェブフック作成")
            print("  2. Webhook URLをコピー")
            print("  3. 以下のいずれかで設定:")
            print("     a) .envファイルに DISCORD_WEBHOOK_URL=https://... を記載")
            print("     b) コマンド引数: python main.py monitor --webhook URL")
            print("=" * 60)
            return

        self.monitor_dir.mkdir(parents=True, exist_ok=True)
        self.log("オッズ監視サーバー開始")
        self.log(f"  発走 {self.before}分前 にバッチ実行")
        from .constants import RESULT_CHECK_DELAY
        self.log(f"  発走 {RESULT_CHECK_DELAY}分後 に結果確認")
        self.log(f"  会場フィルタ: {self.venue_filter or '全会場'}")

        # Discord疎通テスト
        if send_notify(
            "[監視開始] オッズ監視サーバーが起動しました", self.webhook_url
        ):
            self.log("Discord通知テスト: OK")
        else:
            self.log("[WARN] Discord通知テスト失敗 — Webhook URLを確認してください")
            return

        try:
            # Phase 1
            self.log("=" * 50)
            self.log("Phase 1: レース一覧・発走時刻取得")
            self.log("=" * 50)
            await self._scan_race_schedule()

            if not self.races:
                self.log("[END] 本日のレースがありません")
                send_notify(
                    "\n本日のJRAレースはありません", self.webhook_url
                )
                return

            self._build_schedule()

            if not self.schedule:
                self.log("[END] 実行予定のレースがありません（全て発走済み）")
                send_notify("\n全レース発走済みです", self.webhook_url)
                return

            # Phase 2
            self.log("=" * 50)
            self.log("Phase 2: 馬データ補完（netkeiba）")
            self.log("  ※ 1レースあたり約3分かかります")
            self.log("=" * 50)
            await self._enrich_all()

            schedule_text = self._format_schedule()
            send_notify(schedule_text, self.webhook_url)

            # Phase 3
            self.log("=" * 50)
            self.log("Phase 3: スケジュール実行開始")
            self.log("=" * 50)
            await self._run_schedule()

            # 日次サマリ
            self._send_daily_summary()

            self.log("[完了] 全レースの処理が終了しました")
            send_notify(
                "\n[完了] 本日の全レース処理が終了しました", self.webhook_url
            )

        except KeyboardInterrupt:
            self.log("\n[停止] Ctrl+Cで停止しました")
            send_notify(
                "\n[停止] オッズ監視サーバーを停止しました", self.webhook_url
            )
