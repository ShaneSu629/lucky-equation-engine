# db_manager.py
"""
SQLite 数据库管理模块
=====================
统一管理所有彩票数据存储，替代 CSV 文件方案。
解决并发读写 Permission denied 问题，提供事务安全。

表结构：
- lottery_history: 6个彩种的开奖历史（通用宽表，动态列）
- predictions: 预测记录
"""

import os
import json
import sqlite3
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict
from contextlib import contextmanager

DB_DIR = Path(__file__).parent / "data"
DB_PATH = DB_DIR / "lottery.db"

# 日志配置
logger = logging.getLogger("db_manager")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s.%(funcName)s: %(message)s",
        datefmt="%H:%M:%S"
    ))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

# 各彩种的列定义（不含 code/date，这两个是公共列）
LOTTERY_COLUMNS = {
    "ssq":  ["r1", "r2", "r3", "r4", "r5", "r6", "blue"],
    "dlt":  ["f1", "f2", "f3", "f4", "f5", "b1", "b2"],
    "kl8":  [f"n{i:02d}" for i in range(1, 21)],
    "fcsd": ["n1", "n2", "n3"],
    "qxc":  ["n1", "n2", "n3", "n4", "n5", "n6", "n7"],
    "pl3":  ["n1", "n2", "n3"],
}

ALL_LOTTERY_TYPES = list(LOTTERY_COLUMNS.keys())


def _ensure_db():
    """确保数据库目录和文件存在。"""
    DB_DIR.mkdir(parents=True, exist_ok=True)


@contextmanager
def get_connection():
    """获取数据库连接的上下文管理器，自动关闭。"""
    _ensure_db()
    conn = sqlite3.connect(str(DB_PATH), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")  # WAL模式，读写不互斥
    conn.execute("PRAGMA synchronous=NORMAL")
    try:
        yield conn
    finally:
        conn.close()


def init_db():
    """初始化数据库，创建所有表。"""
    _ensure_db()
    logger.info("初始化数据库表结构...")
    with get_connection() as conn:
        # 1. 开奖历史表 —— 每个彩种一张表，表名 = lottery_{type}
        for lt, cols in LOTTERY_COLUMNS.items():
            col_defs = ["code TEXT PRIMARY KEY", "date TEXT"]
            for c in cols:
                col_defs.append(f'"{c}" INTEGER')
            sql = f'CREATE TABLE IF NOT EXISTS "lottery_{lt}" ({", ".join(col_defs)})'
            conn.execute(sql)
            # 索引：按期号降序查询最频繁
            conn.execute(f'CREATE INDEX IF NOT EXISTS "idx_{lt}_code" ON "lottery_{lt}" (code)')
            conn.execute(f'CREATE INDEX IF NOT EXISTS "idx_{lt}_date" ON "lottery_{lt}" (date DESC)')

        # 2. 预测记录表
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lottery_type TEXT NOT NULL,
                code TEXT NOT NULL,
                predictions TEXT NOT NULL DEFAULT '[]',
                play_type TEXT NOT NULL DEFAULT '',
                predict_time TEXT NOT NULL,
                compared TEXT NOT NULL DEFAULT 'false',
                compare_result TEXT NOT NULL DEFAULT '',
                UNIQUE(lottery_type, code)
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_pred_lot ON predictions (lottery_type, code)")

        conn.commit()
    logger.info("数据库表结构初始化完成")


def migrate_from_csv():
    """从现有 CSV 文件迁移数据到 SQLite（幂等，已有数据跳过）。"""
    logger.info("开始 CSV → SQLite 数据迁移...")
    init_db()
    csv_dir = DB_DIR
    migrated = []

    with get_connection() as conn:
        # 1. 迁移各彩种历史数据
        for lt, cols in LOTTERY_COLUMNS.items():
            csv_path = csv_dir / f"{lt}.csv"
            if not csv_path.exists():
                logger.debug(f"{lt}: CSV不存在，跳过")
                continue

            try:
                encodings = ['utf-8', 'utf-8-sig', 'gbk']
                df = None
                for enc in encodings:
                    try:
                        df = pd.read_csv(str(csv_path), encoding=enc, dtype={"code": str})
                        break
                    except (UnicodeDecodeError, UnicodeError):
                        continue
                if df is None or df.empty:
                    continue

                table = f"lottery_{lt}"
                # 检查已有记录数
                existing = conn.execute(f"SELECT COUNT(*) FROM \"{table}\"").fetchone()[0]
                if existing > 0:
                    logger.info(f"{lt}: 已有{existing}期数据，跳过迁移")
                    continue  # 已迁移过

                all_cols = ["code", "date"] + cols
                for _, row in df.iterrows():
                    values = []
                    for c in all_cols:
                        v = row.get(c, None)
                        if pd.isna(v):
                            values.append(None)
                        elif c == "code":
                            values.append(str(int(float(v))) if str(v).replace('.', '').isdigit() else str(v))
                        elif c == "date":
                            values.append(str(v))
                        else:
                            try:
                                values.append(int(v))
                            except (ValueError, TypeError):
                                values.append(None)

                    placeholders = ", ".join(["?"] * len(all_cols))
                    col_names = ", ".join(f'"{c}"' for c in all_cols)
                    sql = f'INSERT OR IGNORE INTO "{table}" ({col_names}) VALUES ({placeholders})'
                    try:
                        conn.execute(sql, values)
                    except Exception:
                        pass

                conn.commit()
                migrated.append(f"{lt}: {len(df)}期")
                logger.info(f"{lt}: 迁移{len(df)}期历史数据完成")
            except Exception as e:
                migrated.append(f"{lt}: 迁移失败({e})")
                logger.error(f"{lt}: 迁移失败 - {e}")

        # 2. 迁移预测记录
        pred_csv = csv_dir / "predictions.csv"
        if pred_csv.exists():
            try:
                for enc in ['utf-8', 'utf-8-sig', 'gbk']:
                    try:
                        df = pd.read_csv(str(pred_csv), encoding=enc,
                                         dtype={"code": str, "compared": str, "compare_result": str})
                        break
                    except (UnicodeDecodeError, UnicodeError):
                        continue
                else:
                    df = pd.DataFrame()

                if not df.empty:
                    existing = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
                    if existing == 0:
                        for _, row in df.iterrows():
                            try:
                                conn.execute("""
                                    INSERT OR IGNORE INTO predictions
                                    (lottery_type, code, predictions, play_type, predict_time, compared, compare_result)
                                    VALUES (?, ?, ?, ?, ?, ?, ?)
                                """, (
                                    str(row.get('lottery_type', '')),
                                    str(row.get('code', '')),
                                    str(row.get('predictions', '[]')),
                                    str(row.get('play_type', '')) if not pd.isna(row.get('play_type')) else '',
                                    str(row.get('predict_time', '')) if not pd.isna(row.get('predict_time')) else '',
                                    str(row.get('compared', 'false')) if not pd.isna(row.get('compared')) else 'false',
                                    str(row.get('compare_result', '')) if not pd.isna(row.get('compare_result')) else '',
                                ))
                            except Exception:
                                pass
                        conn.commit()
                        migrated.append(f"predictions: {len(df)}条")
                        logger.info(f"预测记录迁移{len(df)}条完成")
                    else:
                        logger.info(f"预测记录已有{existing}条，跳过迁移")
            except Exception as e:
                migrated.append(f"predictions: 迁移失败({e})")
                logger.error(f"预测记录迁移失败 - {e}")

    logger.info(f"迁移完成: {migrated}")
    return migrated


# ============================================================
# 开奖历史数据 —— 读/写
# ============================================================

def read_lottery_data(name: str) -> pd.DataFrame:
    """从数据库读取彩种历史开奖数据，返回 DataFrame（兼容原 CSV 接口）。"""
    init_db()
    if name not in LOTTERY_COLUMNS:
        logger.warning(f"未知彩种 '{name}'，返回空DataFrame")
        return pd.DataFrame()

    with get_connection() as conn:
        try:
            df = pd.read_sql_query(f'SELECT * FROM "lottery_{name}" ORDER BY date DESC', conn)
            logger.info(f"读取 {name} 历史数据: {len(df)}期")
            return df
        except Exception as e:
            logger.error(f"读取 {name} 历史数据失败: {e}")
            return pd.DataFrame()


def write_lottery_data(name: str, df: pd.DataFrame):
    """将 DataFrame 写入彩种历史表（全量替换，用于数据同步）。"""
    init_db()
    if name not in LOTTERY_COLUMNS:
        logger.warning(f"未知彩种 '{name}'，跳过写入")
        return

    logger.info(f"全量写入 {name} 历史数据: {len(df)}期")
    cols = LOTTERY_COLUMNS[name]
    all_cols = ["code", "date"] + cols

    with get_connection() as conn:
        # 清空 + 重写（数据同步场景）
        conn.execute(f'DELETE FROM "lottery_{name}"')

        for _, row in df.iterrows():
            values = []
            for c in all_cols:
                v = row.get(c, None)
                if pd.isna(v):
                    values.append(None)
                elif c == "code":
                    values.append(str(v))
                elif c == "date":
                    values.append(str(v))
                else:
                    try:
                        values.append(int(v))
                    except (ValueError, TypeError):
                        values.append(None)

            placeholders = ", ".join(["?"] * len(all_cols))
            col_names = ", ".join(f'"{c}"' for c in all_cols)
            conn.execute(f'INSERT OR IGNORE INTO "lottery_{name}" ({col_names}) VALUES ({placeholders})', values)

        conn.commit()
    logger.info(f"全量写入 {name} 完成: {len(df)}期")


def upsert_lottery_rows(name: str, new_df: pd.DataFrame):
    """增量插入新行（已有期号跳过），用于 fetch 增量同步。"""
    init_db()
    if name not in LOTTERY_COLUMNS:
        logger.warning(f"未知彩种 '{name}'，跳过增量写入")
        return

    logger.info(f"增量写入 {name}: {len(new_df)}期")
    cols = LOTTERY_COLUMNS[name]
    all_cols = ["code", "date"] + cols

    inserted = 0
    with get_connection() as conn:
        for _, row in new_df.iterrows():
            values = []
            for c in all_cols:
                v = row.get(c, None)
                if pd.isna(v):
                    values.append(None)
                elif c == "code":
                    values.append(str(v))
                elif c == "date":
                    values.append(str(v))
                else:
                    try:
                        values.append(int(v))
                    except (ValueError, TypeError):
                        values.append(None)

            placeholders = ", ".join(["?"] * len(all_cols))
            col_names = ", ".join(f'"{c}"' for c in all_cols)
            cursor = conn.execute(f'INSERT OR IGNORE INTO "lottery_{name}" ({col_names}) VALUES ({placeholders})', values)
            if cursor.rowcount > 0:
                inserted += 1

        conn.commit()
    logger.info(f"增量写入 {name} 完成: 尝试{len(new_df)}期, 实际新增{inserted}期")


def get_latest_code(name: str) -> Optional[str]:
    """获取彩种最新一期开奖期号。"""
    init_db()
    with get_connection() as conn:
        try:
            row = conn.execute(f'SELECT code FROM "lottery_{name}" ORDER BY date DESC LIMIT 1').fetchone()
            code = row["code"] if row else None
            logger.debug(f"获取 {name} 最新期号: {code}")
            return code
        except Exception as e:
            logger.error(f"获取 {name} 最新期号失败: {e}")
            return None


def get_known_codes(name: str) -> set:
    """获取彩种所有已有期号（用于增量同步去重）。"""
    init_db()
    with get_connection() as conn:
        try:
            rows = conn.execute(f'SELECT code FROM "lottery_{name}"').fetchall()
            codes = {r["code"] for r in rows}
            logger.debug(f"获取 {name} 已有期号: {len(codes)}个")
            return codes
        except Exception as e:
            logger.error(f"获取 {name} 已有期号失败: {e}")
            return set()


# ============================================================
# 预测记录 —— CRUD
# ============================================================

def _validate_prediction_item(lottery_type: str, p) -> bool:
    """校验单条预测记录的格式是否合法，防止脏数据入库。"""
    if not isinstance(p, dict):
        return False
    try:
        if lottery_type == "ssq":
            # 必须含 red/blue，且不能用 nums 代替 red（历史旧数据污染）
            if "red" not in p or "blue" not in p:
                return False
            reds = p["red"]
            blue = p["blue"]
            if not isinstance(reds, list) or len(reds) != 6:
                return False
            if len(set(reds)) != 6 or not all(1 <= int(r) <= 33 for r in reds):
                return False
            if not (1 <= int(blue) <= 16):
                return False
            return True
        elif lottery_type == "dlt":
            if "nums" not in p:
                return False
            nums = p["nums"]
            if not isinstance(nums, list) or len(nums) != 7:
                return False
            fronts = nums[:5]
            backs = nums[5:7]
            if len(set(fronts)) != 5 or not all(1 <= int(x) <= 35 for x in fronts):
                return False
            if len(set(backs)) != 2 or not all(1 <= int(x) <= 12 for x in backs):
                return False
            return True
        elif lottery_type == "kl8":
            if "nums" not in p:
                return False
            nums = p["nums"]
            if not isinstance(nums, list) or not (1 <= len(nums) <= 10):
                return False
            if len(set(nums)) != len(nums) or not all(1 <= int(x) <= 80 for x in nums):
                return False
            return True
        elif lottery_type == "qxc":
            if "nums" not in p:
                return False
            nums = p["nums"]
            return isinstance(nums, list) and len(nums) == 7 and all(0 <= int(x) <= 9 for x in nums)
        elif lottery_type in ("pl3", "fcsd"):
            if "nums" not in p:
                return False
            nums = p["nums"]
            return isinstance(nums, list) and len(nums) == 3 and all(0 <= int(x) <= 9 for x in nums)
        else:
            return False
    except (ValueError, TypeError):
        return False


def save_prediction_record(lottery_type: str, code: str, predictions: list, play_type: str = None):
    """保存或追加预测记录。

    同期同彩种：追加 predictions（合并去重）。
    不同期：新增行。
    新增：逐条校验格式，过滤非法记录并打日志。
    """
    init_db()
    import time

    # 格式校验：过滤非法项
    original_len = len(predictions) if isinstance(predictions, list) else 0
    valid_predictions = [p for p in predictions if _validate_prediction_item(lottery_type, p)] if isinstance(predictions, list) else []
    dropped = original_len - len(valid_predictions)
    if dropped:
        logger.warning(f"保存预测: {lottery_type} 第{code}期, 过滤掉{dropped}组格式非法记录")
    if not valid_predictions:
        logger.warning(f"保存预测: {lottery_type} 第{code}期, 无有效记录可保存")
        return

    n_preds = len(valid_predictions)
    logger.info(f"保存预测: {lottery_type} 第{code}期, {n_preds}组, play_type={play_type}")

    # 期号合法性校验
    latest_code = get_latest_code(lottery_type)
    if latest_code and str(latest_code).isdigit() and str(code).isdigit():
        diff = int(code) - int(latest_code)
        if diff > 2 or diff < -200:
            err_msg = (
                f"期号异常：保存的第 {code} 期与最新开奖期号 {latest_code} 相差过大（差值 {diff}），"
                f"正常预测期应落在 [{int(latest_code)-200}, {int(latest_code)+2}] 之间。"
            )
            logger.error(err_msg)
            raise ValueError(err_msg)

    now = time.strftime('%Y-%m-%d %H:%M:%S')
    preds_json = json.dumps(valid_predictions, ensure_ascii=False)

    with get_connection() as conn:
        # 查找同期同彩种
        row = conn.execute(
            "SELECT predictions, play_type FROM predictions WHERE lottery_type=? AND code=?",
            (lottery_type, code)
        ).fetchone()

        if row:
            # 同期：合并去重
            existing_preds = json.loads(row["predictions"]) if row["predictions"] else []
            if not isinstance(existing_preds, list):
                existing_preds = []

            existing_keys = set()
            for p in existing_preds:
                existing_keys.add(json.dumps(p, sort_keys=True, ensure_ascii=False))

            added = 0
            for p in valid_predictions:
                key = json.dumps(p, sort_keys=True, ensure_ascii=False)
                if key not in existing_keys:
                    existing_preds.append(p)
                    existing_keys.add(key)
                    added += 1

            existing_pt = row["play_type"] or ''
            if existing_pt.strip() == '':
                existing_pt = play_type or ''

            conn.execute("""
                UPDATE predictions
                SET predictions=?, play_type=?, predict_time=?, compared='false', compare_result=''
                WHERE lottery_type=? AND code=?
            """, (json.dumps(existing_preds, ensure_ascii=False), existing_pt, now, lottery_type, code))
            logger.info(f"同期追加: {lottery_type} 第{code}期, 新增{added}组(去重后共{len(existing_preds)}组)")
        else:
            # 新增
            conn.execute("""
                INSERT INTO predictions (lottery_type, code, predictions, play_type, predict_time, compared, compare_result)
                VALUES (?, ?, ?, ?, ?, 'false', '')
            """, (lottery_type, code, preds_json, play_type or '', now))
            logger.info(f"新增预测: {lottery_type} 第{code}期, {n_preds}组, play_type={play_type or ''}")

        conn.commit()

    # 限制总记录数
    _trim_predictions()


def _trim_predictions(max_rows: int = 100):
    """限制预测记录总数，超出时删除最旧的。"""
    with get_connection() as conn:
        count = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        if count > max_rows:
            deleted = count - max_rows
            conn.execute("""
                DELETE FROM predictions WHERE id IN (
                    SELECT id FROM predictions ORDER BY predict_time ASC LIMIT ?
                )
            """, (deleted,))
            conn.commit()
            logger.info(f"清理旧预测记录: 删除{deleted}条, 保留{max_rows}条")


def get_prediction_records(lottery_type: str = None) -> list:
    """读取预测记录，返回 list[dict]（兼容原接口）。"""
    init_db()
    with get_connection() as conn:
        try:
            # 按期号降序排列：最新期号放最前面（避免 predict_time 格式不统一导致排序错乱）
            if lottery_type:
                df = pd.read_sql_query(
                    "SELECT * FROM predictions WHERE lottery_type=? ORDER BY code DESC",
                    conn, params=(lottery_type,)
                )
            else:
                df = pd.read_sql_query(
                    "SELECT * FROM predictions ORDER BY code DESC", conn
                )
        except Exception as e:
            logger.error(f"读取预测记录失败: {e}")
            return []

    records = []
    for _, row in df.iterrows():
        records.append(_parse_prediction_row(row))

    logger.info(f"读取预测记录: {lottery_type or '全部'} → {len(records)}条")
    return records


def _parse_prediction_row(row) -> dict:
    """将 predictions 表的一行（pandas Series）解析为 dict。"""
    compare_result = None
    cr_val = row.get('compare_result', '')
    if cr_val and not pd.isna(cr_val):
        try:
            compare_result = json.loads(str(cr_val).strip())
        except Exception:
            compare_result = None

    predictions_str = row.get('predictions', '[]')
    predictions = json.loads(predictions_str) if predictions_str and not pd.isna(predictions_str) else []

    play_type_val = row.get('play_type', '')
    play_type = '' if pd.isna(play_type_val) else str(play_type_val)

    return {
        'lottery_type': row.get('lottery_type', ''),
        'code': str(row.get('code', '')),
        'predictions': predictions,
        'play_type': play_type,
        'predict_time': row.get('predict_time', ''),
        'compared': str(row.get('compared', 'false')).lower() == 'true',
        'compare_result': compare_result
    }


def _build_prediction_where(lottery_type: str,
                            code_start=None, code_end=None,
                            status: str = 'all', win: str = 'all'):
    """构建预测记录筛选 WHERE 子句与参数（数据库级筛选）。

    - code_start/code_end：按数字期号范围（CAST(code AS INTEGER)），空/None 表示不限制
    - status: 'all' | 'compared' | 'pending'
    - win:    'all' | 'won' | 'lost'（仅对已对比记录有效，用 JSON1 读取总奖金判断）
    """
    clauses = ["lottery_type = ?"]
    params: list = [lottery_type]

    if code_start is not None and str(code_start).strip() != "":
        clauses.append("CAST(code AS INTEGER) >= ?")
        params.append(int(str(code_start).strip()))
    if code_end is not None and str(code_end).strip() != "":
        clauses.append("CAST(code AS INTEGER) <= ?")
        params.append(int(str(code_end).strip()))

    if status == 'compared':
        clauses.append("compared = 'true'")
    elif status == 'pending':
        clauses.append("compared = 'false'")

    if win == 'won':
        clauses.append(
            "compared = 'true' AND compare_result <> '' "
            "AND CAST(json_extract(compare_result, '$.prize_result.total_prize') AS REAL) > 0"
        )
    elif win == 'lost':
        clauses.append(
            "compared = 'true' AND compare_result <> '' "
            "AND CAST(json_extract(compare_result, '$.prize_result.total_prize') AS REAL) = 0"
        )

    return " AND ".join(clauses), params


def count_prediction_records(lottery_type: str,
                             code_start=None, code_end=None,
                             status: str = 'all', win: str = 'all') -> int:
    """按筛选条件统计预测记录数量（用于分页总页数）。"""
    init_db()
    where, params = _build_prediction_where(lottery_type, code_start, code_end, status, win)
    with get_connection() as conn:
        try:
            cur = conn.execute(f"SELECT COUNT(*) FROM predictions WHERE {where}", params)
            return int(cur.fetchone()[0])
        except Exception as e:
            logger.error(f"统计预测记录失败: {e}")
            return 0


def get_prediction_records_paged(lottery_type: str, offset: int = 0, limit: int = 20,
                                 code_start=None, code_end=None,
                                 status: str = 'all', win: str = 'all') -> list:
    """按筛选条件分页读取预测记录（数据库级 LIMIT/OFFSET，避免一次捞出全部）。

    返回 list[dict]，结构与 get_prediction_records 一致。
    """
    init_db()
    where, params = _build_prediction_where(lottery_type, code_start, code_end, status, win)
    params = params + [int(limit), int(offset)]
    with get_connection() as conn:
        try:
            df = pd.read_sql_query(
                f"SELECT * FROM predictions WHERE {where} "
                f"ORDER BY code DESC LIMIT ? OFFSET ?",
                conn, params=params
            )
        except Exception as e:
            logger.error(f"分页读取预测记录失败: {e}")
            return []

    records = [_parse_prediction_row(row) for _, row in df.iterrows()]
    logger.info(f"分页读取预测记录: {lottery_type} offset={offset} limit={limit} → {len(records)}条")
    return records


def get_prediction_codes(lottery_type: str) -> list:
    """只读 code 列，返回该彩种全部期号（降序），用于对比期号下拉框（轻量）。"""
    init_db()
    with get_connection() as conn:
        try:
            cur = conn.execute(
                "SELECT code FROM predictions WHERE lottery_type=? ORDER BY code DESC",
                (lottery_type,)
            )
            return [str(r[0]) for r in cur.fetchall()]
        except Exception as e:
            logger.error(f"读取预测期号失败: {e}")
            return []


def get_prediction_for_code(lottery_type: str, code: str) -> Optional[dict]:
    """获取指定彩种+期号的预测记录。"""
    records = get_prediction_records(lottery_type)
    for record in records:
        if record.get('code') == code:
            return record
    logger.debug(f"未找到预测记录: {lottery_type} 第{code}期")
    return None


def update_prediction_compare(lottery_type: str, code: str, compare_result: dict):
    """更新预测记录的开奖对比结果。"""
    init_db()
    logger.info(f"更新对比结果: {lottery_type} 第{code}期")
    with get_connection() as conn:
        cr_json = json.dumps(compare_result, ensure_ascii=False, default=lambda x: int(x) if isinstance(x, (int, float, np.integer, np.floating)) else bool(x) if isinstance(x, (bool, np.bool_)) else x)
        conn.execute("""
            UPDATE predictions SET compared='true', compare_result=? 
            WHERE lottery_type=? AND code=?
        """, (cr_json, lottery_type, code))
        conn.commit()
    logger.info(f"对比结果更新完成: {lottery_type} 第{code}期")


# ============================================================
# 兼容层：为 app.py 中直接 pd.read_csv 的场景提供快捷方法
# ============================================================

def get_lottery_df(name: str, dtype: dict = None) -> pd.DataFrame:
    """读取彩种数据为 DataFrame，可选指定 dtype（兼容原 pd.read_csv(dtype=...) 接口）。"""
    df = read_lottery_data(name)
    if dtype and not df.empty:
        for col, tp in dtype.items():
            if col in df.columns:
                df[col] = df[col].astype(tp)
    return df
