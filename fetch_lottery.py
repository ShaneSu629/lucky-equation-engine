# fetch_lottery.py
import os
import time
import requests
import pandas as pd
from datetime import datetime

BASE_URL = "http://api.huiniao.top/interface/home/lotteryHistory"

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# 数据库管理模块（替代 CSV 直写）
from db_manager import (
    init_db, upsert_lottery_rows, read_lottery_data as _db_read_lottery,
    get_known_codes, write_lottery_data,
)


def fetch_page(name: str, page_no: int, page_size: int = 30, retry=5) -> list[dict]:
    params = {
        "type": name,
        "page": page_no,
        "limit": page_size
    }
    
    for attempt in range(retry):
        try:
            r = requests.get(BASE_URL, params=params, timeout=15)
            r.raise_for_status()
            js = r.json()
            
            if js.get("code") == 1:
                data = js.get("data", {})
                inner_data = data.get("data", {})
                rows = inner_data.get("list", [])
                return rows or []
            elif js.get("code") == 401:
                wait_time = 3 + attempt * 2
                time.sleep(wait_time)
            else:
                return []
        except Exception as e:
            if attempt < retry - 1:
                time.sleep(2)
    
    return []


def parse_ssq(rows: list[dict]) -> pd.DataFrame:
    out = []
    for r in rows:
        try:
            out.append({
                "code": r["code"],
                "date": r["day"],
                "r1": int(r["one"]),
                "r2": int(r["two"]),
                "r3": int(r["three"]),
                "r4": int(r["four"]),
                "r5": int(r["five"]),
                "r6": int(r["six"]),
                "blue": int(r["seven"]),
            })
        except (KeyError, ValueError):
            continue
    return pd.DataFrame(out)


def parse_kl8(rows: list[dict]) -> pd.DataFrame:
    out = []
    keys = [
        "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
        "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty"
    ]
    for r in rows:
        try:
            row = {
                "code": r["code"],
                "date": r["day"]
            }
            for i, k in enumerate(keys, 1):
                row[f"n{i:02d}"] = int(r[k])
            out.append(row)
        except (KeyError, ValueError):
            continue
    return pd.DataFrame(out)


def parse_fcsd(rows: list[dict]) -> pd.DataFrame:
    out = []
    for r in rows:
        try:
            out.append({
                "code": r["code"],
                "date": r["day"],
                "n1": int(r["one"]),
                "n2": int(r["two"]),
                "n3": int(r["three"]),
            })
        except (KeyError, ValueError):
            continue
    return pd.DataFrame(out)


def parse_dlt(rows: list[dict]) -> pd.DataFrame:
    """大乐透：5个前区(1-35) + 2个后区(1-12)"""
    out = []
    for r in rows:
        try:
            out.append({
                "code": r["code"],
                "date": r["day"],
                "f1": int(r["one"]),
                "f2": int(r["two"]),
                "f3": int(r["three"]),
                "f4": int(r["four"]),
                "f5": int(r["five"]),
                "b1": int(r["six"]),
                "b2": int(r["seven"]),
            })
        except (KeyError, ValueError):
            continue
    return pd.DataFrame(out)


def parse_qxc(rows: list[dict]) -> pd.DataFrame:
    """七星彩：7个位置各0-9"""
    out = []
    for r in rows:
        try:
            out.append({
                "code": r["code"],
                "date": r["day"],
                "n1": int(r["one"]),
                "n2": int(r["two"]),
                "n3": int(r["three"]),
                "n4": int(r["four"]),
                "n5": int(r["five"]),
                "n6": int(r["six"]),
                "n7": int(r["seven"]),
            })
        except (KeyError, ValueError):
            continue
    return pd.DataFrame(out)


def parse_pl3(rows: list[dict]) -> pd.DataFrame:
    """排列三：3个位置各0-9"""
    out = []
    for r in rows:
        try:
            out.append({
                "code": r["code"],
                "date": r["day"],
                "n1": int(r["one"]),
                "n2": int(r["two"]),
                "n3": int(r["three"]),
            })
        except (KeyError, ValueError):
            continue
    return pd.DataFrame(out)


def update(name: str, force_full: bool = False) -> pd.DataFrame:
    """从 API 同步数据到数据库。"""
    init_db()
    old = _db_read_lottery(name)
    known = get_known_codes(name)

    is_first_sync = len(known) == 0

    print(f"[{name}] 模式: {'全量同步' if (is_first_sync or force_full) else '增量同步'}")
    print(f"[{name}] 本地已有期数: {len(known)}")

    api_name_map = {"kl8": "klb", "pl3": "pls"}
    api_name = api_name_map.get(name, name)
    print(f"[{name}] API类型: {api_name}")

    all_new, page = [], 1
    stop_fetching = False

    while not stop_fetching:
        rows = fetch_page(api_name, page)

        if not rows:
            break

        new_rows_in_page = []
        for r in rows:
            if not is_first_sync and not force_full:
                date_str = r.get("day", "")
                if not date_str or date_str < "2025-01-01":
                    stop_fetching = True
                    break

            if r["code"] not in known:
                new_rows_in_page.append(r)
            elif not is_first_sync and not force_full:
                stop_fetching = True
                break

        all_new.extend(new_rows_in_page)

        if len(rows) < 30:
            break

        page += 1
        time.sleep(0.8)

    print(f"[{name}] 总计抓取 {len(all_new)} 条新数据")

    if all_new:
        if name == "ssq":
            df_new = parse_ssq(all_new)
        elif name == "kl8":
            df_new = parse_kl8(all_new)
        elif name == "fcsd":
            df_new = parse_fcsd(all_new)
        elif name == "dlt":
            df_new = parse_dlt(all_new)
        elif name == "qxc":
            df_new = parse_qxc(all_new)
        elif name == "pl3":
            df_new = parse_pl3(all_new)
        else:
            df_new = pd.DataFrame()

        # 增量写入数据库
        upsert_lottery_rows(name, df_new)

        # 返回合并后的完整数据
        df = _db_read_lottery(name)

        if len(df) > 0:
            print(f"[{name}] ✅ 数据范围: {df.iloc[0]['code']} ~ {df.iloc[-1]['code']} (最新→最旧)")

        if is_first_sync or force_full:
            print(f"[{name}] 🎉 全量同步完成，共 {len(df)} 期")
        else:
            print(f"[{name}] 📊 增量更新：新增 {len(df_new)} 期，累计 {len(df)} 期")
        return df

    if is_first_sync or force_full:
        print(f"[{name}] ✅ 已是最新，共 {len(old)} 期")
    else:
        print(f"[{name}] ✅ 已是最新")
    return old