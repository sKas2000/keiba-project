"""
開催日・レース選択UI
JRA の開催選択・レース選択の対話ロジック
"""


def select_meeting_interactive(meetings: list, args) -> int:
    """開催を選択して index を返す"""
    if args.meeting_index is not None:
        if 1 <= args.meeting_index <= len(meetings):
            return args.meeting_index - 1
        raise ValueError(f"meeting-index {args.meeting_index} は無効です")

    if getattr(args, 'non_interactive', False):
        return 0

    while True:
        c = input(f"\n開催を選択 (1-{len(meetings)}): ").strip()
        if c.isdigit() and 1 <= int(c) <= len(meetings):
            return int(c) - 1
        print("  無効")


def select_race_interactive(args) -> int:
    """レース番号を選択"""
    if args.race is not None:
        if 1 <= args.race <= 12:
            return args.race
        raise ValueError(f"race {args.race} は無効です")

    if getattr(args, 'non_interactive', False):
        return 1

    while True:
        c = input("レース番号 (1-12): ").strip()
        if c.isdigit() and 1 <= int(c) <= 12:
            return int(c)
        print("  無効")


def prompt_missing_race_info(race_info: dict, non_interactive: bool = False) -> dict:
    """未検出のレース情報を手動入力で補完"""
    if non_interactive:
        return race_info

    fields = [
        ("venue", "競馬場を入力 (例: 東京)"),
        ("name", "レース名"),
        ("surface", "馬場 (芝/ダート)"),
        ("direction", "回り (右/左)"),
        ("grade", "グレード (G1/G2/G3/OP/3勝/2勝/1勝/未勝利/新馬)"),
    ]

    for key, prompt in fields:
        if not race_info.get(key):
            val = input(f"  -> {prompt}: ").strip()
            if val:
                race_info[key] = val

    if not race_info.get("distance"):
        d = input("  -> 距離 (例: 1400): ").strip()
        if d.isdigit():
            race_info["distance"] = int(d)

    return race_info
