"""
Discord Webhook 通知モジュール
レース予測結果をDiscordに送信
"""
import json
import os
import urllib.request


def get_webhook_url() -> str:
    """Discord Webhook URLを取得（環境変数 or .envファイル）"""
    url = os.environ.get("DISCORD_WEBHOOK_URL", "")
    if url:
        return url

    from config.settings import PROJECT_ROOT
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("DISCORD_WEBHOOK_URL="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def send_notify(message: str, webhook_url: str = None) -> bool:
    """Discordにメッセージ送信

    Args:
        message: 送信メッセージ（最大2000文字）
        webhook_url: Discord Webhook URL（省略時は環境変数から取得）

    Returns:
        送信成功ならTrue
    """
    webhook_url = webhook_url or get_webhook_url()
    if not webhook_url:
        print("[ERROR] DISCORD_WEBHOOK_URL が未設定です")
        print("  1. Discordサーバーでチャンネル設定→連携サービス→ウェブフック作成")
        print("  2. Webhook URLをコピー")
        print("  3. .env ファイルに DISCORD_WEBHOOK_URL=https://... を記載")
        return False

    # 2000文字制限
    if len(message) > 2000:
        message = message[:1997] + "..."

    payload = json.dumps({"content": message}).encode("utf-8")
    req = urllib.request.Request(
        webhook_url, data=payload,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "keiba-ai/1.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status in (200, 204)
    except Exception as e:
        print(f"[ERROR] Discord通知送信失敗: {e}")
        return False


def format_race_notification(race_info: dict, ev_results: dict) -> str:
    """レース予測結果を通知用にフォーマット"""
    venue = race_info.get("venue", "")
    race_num = race_info.get("race_number", 0)
    name = race_info.get("name", "")
    grade = race_info.get("grade", "")

    header = f"**{venue}{race_num}R {name}**"
    if grade:
        header += f" [{grade}]"
    lines = [header]

    if ev_results.get("low_confidence"):
        lines.append("> [見送り] 確信度不足")
        return "\n".join(lines)

    for w in ev_results.get("warnings", []):
        lines.append(f"> ⚠ {w}")

    # 推奨買い目（馬連・ワイド EV>=1.0）
    recs = []
    for bt, key in [("馬連", "quinella"), ("ワイド", "wide")]:
        for bet in ev_results.get(key, []):
            if bet["ev"] >= 1.0:
                recs.append((bt, bet))

    if not recs:
        lines.append("推奨買い目なし")
        return "\n".join(lines)

    recs.sort(key=lambda x: x[1]["ev"], reverse=True)
    lines.append("```")
    lines.append("券種   組合せ       オッズ   EV    級")
    lines.append("-" * 42)
    for bt, bet in recs[:8]:
        combo = bet.get("combo", "?")
        lines.append(f"{bt:6} {combo:12} {bet['odds']:6.1f}倍 {bet['ev']:.2f} ({bet['rank']})")
    lines.append("```")

    # 参考: 3連複
    trio_recs = [b for b in ev_results.get("trio", []) if b["ev"] >= 1.0]
    if trio_recs:
        lines.append("参考(3連複):")
        for bet in trio_recs[:3]:
            combo = bet.get("combo", "?")
            lines.append(f"> {combo} {bet['odds']:.1f}倍 EV{bet['ev']:.2f}")

    return "\n".join(lines)


def format_odds_change_notification(
    race_info: dict, prev_bets: list, curr_bets: list
) -> str:
    """オッズ変動通知フォーマット

    Returns:
        変動通知テキスト（変動がない場合は空文字列）
    """
    venue = race_info.get("venue", "")
    race_num = race_info.get("race_number", 0)
    name = race_info.get("name", "")

    prev_keys = {(bt, b.get("combo", "")): b for bt, b in prev_bets}
    curr_keys = {(bt, b.get("combo", "")): b for bt, b in curr_bets}

    new_bets = []
    dropped = []
    changed = []

    for key, bet in curr_keys.items():
        if key not in prev_keys:
            new_bets.append((key[0], bet))
        else:
            prev_ev = prev_keys[key]["ev"]
            curr_ev = bet["ev"]
            if abs(curr_ev - prev_ev) >= 0.1:
                changed.append((key[0], bet, prev_ev))

    for key, bet in prev_keys.items():
        if key not in curr_keys:
            dropped.append((key[0], bet))

    if not new_bets and not dropped and not changed:
        return ""

    lines = [f"**[変動] {venue}{race_num}R {name}**"]

    if new_bets:
        lines.append("新規:")
        for bt, bet in new_bets:
            lines.append(f"> {bt} {bet.get('combo','')} EV{bet['ev']:.2f}")

    if dropped:
        lines.append("消滅:")
        for bt, bet in dropped:
            lines.append(f"> {bt} {bet.get('combo','')} (旧EV{bet['ev']:.2f})")

    if changed:
        lines.append("変動:")
        for bt, bet, prev_ev in changed:
            arrow = "↑" if bet["ev"] > prev_ev else "↓"
            lines.append(f"> {bt} {bet.get('combo','')} EV{prev_ev:.2f} → {bet['ev']:.2f} {arrow}")

    return "\n".join(lines)
