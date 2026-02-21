"""
LINE Notify 通知モジュール
レース予測結果をLINEに送信
"""
import os
import urllib.request
import urllib.parse


def get_token() -> str:
    """LINE Notify トークンを取得（環境変数 or .envファイル）"""
    token = os.environ.get("LINE_NOTIFY_TOKEN", "")
    if token:
        return token

    # .env ファイルから読み込み
    from config.settings import PROJECT_ROOT
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("LINE_NOTIFY_TOKEN="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def send_line_notify(message: str, token: str = None) -> bool:
    """LINE Notifyでメッセージ送信

    Args:
        message: 送信メッセージ（最大1000文字）
        token: LINE Notifyトークン（省略時は環境変数から取得）

    Returns:
        送信成功ならTrue
    """
    token = token or get_token()
    if not token:
        print("[ERROR] LINE_NOTIFY_TOKEN が未設定です")
        print("  1. https://notify-bot.line.me/ でトークン取得")
        print("  2. 環境変数 LINE_NOTIFY_TOKEN に設定")
        print("     または .env ファイルに LINE_NOTIFY_TOKEN=xxx を記載")
        return False

    # 1000文字制限
    if len(message) > 1000:
        message = message[:997] + "..."

    url = "https://notify-api.line.me/api/notify"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/x-www-form-urlencoded",
    }
    data = urllib.parse.urlencode({"message": message}).encode("utf-8")

    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except Exception as e:
        print(f"[ERROR] LINE通知送信失敗: {e}")
        return False


def format_race_notification(race_info: dict, ev_results: dict) -> str:
    """レース予測結果をLINE通知用にフォーマット

    Args:
        race_info: race dict (venue, race_number, name, grade, etc.)
        ev_results: calculate_ev() の返り値

    Returns:
        LINE通知用テキスト
    """
    venue = race_info.get("venue", "")
    race_num = race_info.get("race_number", 0)
    name = race_info.get("name", "")
    grade = race_info.get("grade", "")

    lines = [f"\n{venue}{race_num}R {name}"]
    if grade:
        lines[0] += f" [{grade}]"

    # 確信度チェック
    if ev_results.get("low_confidence"):
        lines.append("[見送り] 確信度不足")
        return "\n".join(lines)

    # 警告
    for w in ev_results.get("warnings", []):
        lines.append(f"[!]{w}")

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
    lines.append("---推奨---")
    for bt, bet in recs[:8]:
        combo = bet.get("combo", "?")
        lines.append(f"{bt} {combo} {bet['odds']:.1f}倍 EV{bet['ev']:.2f}({bet['rank']})")

    # 参考: 3連複
    trio_recs = [b for b in ev_results.get("trio", []) if b["ev"] >= 1.0]
    if trio_recs:
        lines.append("---参考(3連複)---")
        for bet in trio_recs[:3]:
            combo = bet.get("combo", "?")
            lines.append(f"{combo} {bet['odds']:.1f}倍 EV{bet['ev']:.2f}")

    return "\n".join(lines)


def format_odds_change_notification(
    race_info: dict, prev_bets: list, curr_bets: list
) -> str:
    """オッズ変動通知フォーマット

    Args:
        race_info: レース情報
        prev_bets: 前回の推奨買い目 [(bet_type, bet_dict), ...]
        curr_bets: 今回の推奨買い目

    Returns:
        変動通知テキスト（変動がない場合は空文字列）
    """
    venue = race_info.get("venue", "")
    race_num = race_info.get("race_number", 0)
    name = race_info.get("name", "")

    # 前回と今回の買い目をキーで比較
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

    lines = [f"\n[変動]{venue}{race_num}R {name}"]

    if new_bets:
        lines.append("▼新規")
        for bt, bet in new_bets:
            lines.append(f" {bt} {bet.get('combo','')} EV{bet['ev']:.2f}")

    if dropped:
        lines.append("▼消滅")
        for bt, bet in dropped:
            lines.append(f" {bt} {bet.get('combo','')} (旧EV{bet['ev']:.2f})")

    if changed:
        lines.append("▼変動")
        for bt, bet, prev_ev in changed:
            arrow = "↑" if bet["ev"] > prev_ev else "↓"
            lines.append(f" {bt} {bet.get('combo','')} EV{prev_ev:.2f}→{bet['ev']:.2f}{arrow}")

    return "\n".join(lines)
