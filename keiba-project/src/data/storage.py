"""
データの読み書き
JSON / CSV の統一 I/O + ファイル名生成
"""
import csv
import json
import re
from datetime import datetime
from pathlib import Path

from config.settings import DATA_DIR, RAW_DIR, PROCESSED_DIR, FEATURES_DIR, RACES_DIR, MODEL_DIR, LOG_DIR


def read_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def read_csv(path: str | Path, dtypes: dict = None) -> "pd.DataFrame":
    import pandas as pd
    return pd.read_csv(path, dtype=dtypes)


def write_csv(df: "pd.DataFrame", path: str | Path, append: bool = False) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append and Path(path).exists() else "w"
    header = mode == "w" or not Path(path).exists()
    df.to_csv(path, mode=mode, header=header, index=False)


def generate_output_filename(data: dict, stage: str) -> str:
    """
    YYYYMMDD_会場RR_レース名_<stage>.json
    stage: input | enriched_input | base_scored | ev_results | ml_scored
    data: race_info dict or full data dict with "race" sub-key
    """
    # full data dict の場合は race サブキーを参照
    race_info = data.get("race", data) if "race" in data else data
    date_str = datetime.now().strftime("%Y%m%d")
    venue = race_info.get("venue", "")
    rn = race_info.get("race_number", 0)
    name = race_info.get("name", "")
    safe_name = re.sub(r'[\\/:*?"<>|\s]', '_', name or "不明")
    return f"{date_str}_{venue}{rn}R_{safe_name}_{stage}.json"


def ensure_data_dirs() -> None:
    """必要なデータディレクトリを作成"""
    for d in [DATA_DIR, RAW_DIR, PROCESSED_DIR, FEATURES_DIR, RACES_DIR, MODEL_DIR, LOG_DIR]:
        d.mkdir(parents=True, exist_ok=True)
