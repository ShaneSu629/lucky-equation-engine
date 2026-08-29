# ai_predict.py
"""
AI 预测与分析模块
==================
基于 OpenAI 兼容接口的大语言模型，提供：
1. AI 智能号码预测（双色球/快乐8/福彩3D）
2. 历史趋势深度分析
3. 对冲策略智能优化
"""

import os
import re
import json
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)
from pathlib import Path
from typing import Dict, List, Optional

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

CONFIG_DIR = Path.home() / ".lottery_ai"
CONFIG_FILE = CONFIG_DIR / "config.json"
DATA_DIR = Path(__file__).parent / "data"

# 数据库管理模块（替代 CSV 直读写）
from db_manager import (
    init_db, read_lottery_data as _db_read_lottery,
    save_prediction_record as _db_save_prediction,
    get_prediction_records as _db_get_prediction_records,
    get_prediction_for_code as _db_get_prediction_for_code,
    update_prediction_compare as _db_update_prediction_compare,
    get_latest_code as _db_get_latest_code,
    get_lottery_df as _db_get_lottery_df,
)

DEFAULT_CONFIG = {
    "api_key": "",
    "base_url": "https://api.siliconflow.cn/v1",
    "model": "deepseek-ai/DeepSeek-R1"
}


def is_cloud_deployed() -> bool:
    """检测是否运行在 Streamlit Cloud（通过 Secrets 配置）"""
    try:
        import streamlit as st
        if hasattr(st, 'secrets') and 'ai_config' in st.secrets:
            return True
    except Exception:
        pass
    return False


def load_config() -> dict:
    # 优先从 Streamlit Secrets 读取（云部署环境）
    try:
        import streamlit as st
        if hasattr(st, 'secrets') and 'ai_config' in st.secrets:
            sec = st.secrets['ai_config']
            cfg = {
                "api_key": str(sec.get("api_key", "")),
                "base_url": str(sec.get("base_url", DEFAULT_CONFIG["base_url"])),
                "model": str(sec.get("model", DEFAULT_CONFIG["model"]))
            }
            # 若 Secrets 中未配置 api_key，则回退到本地配置文件
            if cfg["api_key"].strip():
                return cfg
    except Exception:
        pass

    # 本地配置文件
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                config = json.load(f)
            for key, value in DEFAULT_CONFIG.items():
                if key not in config:
                    config[key] = value
            return config
        except Exception:
            return DEFAULT_CONFIG.copy()
    return DEFAULT_CONFIG.copy()


def save_ai_config(api_key: str, base_url: str, model_name: str):
    CONFIG_DIR.mkdir(exist_ok=True)
    config = {
        "api_key": api_key,
        "base_url": base_url,
        "model": model_name
    }
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def is_ai_configured() -> bool:
    config = load_config()
    return bool(config.get("api_key", "").strip())


def test_ai_connection(api_key: str, base_url: str, model_name: str) -> str:
    if OpenAI is None:
        return "❌ openai 库未安装，请运行: pip install openai"
    
    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "你好，测试连接"}],
            max_tokens=50
        )
        return "✅ AI 连接成功！"
    except Exception as e:
        return f"❌ 连接失败: {str(e)}"


def _read_csv_file(file_path, **kwargs):
    encodings = ['utf-8', 'utf-8-sig', 'gbk', 'gb2312', 'cp1252']
    for enc in encodings:
        try:
            return pd.read_csv(file_path, encoding=enc, **kwargs)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return pd.DataFrame()


def _read_lottery_data(name: str) -> pd.DataFrame:
    """从数据库读取彩种历史数据（兼容原 CSV 接口）。"""
    return _db_read_lottery(name)


def _get_ai_client():
    if not is_ai_configured():
        return None, None
    
    config = load_config()
    api_key = config["api_key"].strip()
    base_url = config.get("base_url", "https://api.siliconflow.cn/v1").strip()
    model = config.get("model", "deepseek-ai/DeepSeek-R1").strip()
    
    if not api_key:
        return None, None
    
    if OpenAI is None:
        return None, None
    
    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        return client, model
    except Exception:
        return None, None


def _call_ai(messages: list, max_tokens: int = 4000, temperature: float = 0.7) -> dict:
    client, model = _get_ai_client()
    
    if client is None:
        return {"error": "AI 未配置或连接失败，请在侧边栏配置 API Key"}
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        choice = response.choices[0]
        msg = choice.message
        content = msg.content

        # DeepSeek-R1 等推理模型：content 可能为 None/空，
        # 推理内容在 reasoning_content 里；另外 finish_reason=stop 也有可能
        # 只输出了推理但未生成最终回答（length 截断）
        if not content or not str(content).strip():
            # 尝试从 reasoning_content 恢复（部分 API 支持）
            reasoning = getattr(msg, 'reasoning_content', None) or ''
            if reasoning and str(reasoning).strip():
                content = str(reasoning)
            else:
                # finish_reason=length 说明推理占满了 token，正式回答被截断
                finish = getattr(choice, 'finish_reason', '')
                if finish == 'length':
                    return {"error": "AI 推理过长导致输出被截断，请减少组数或缩短 prompt 后重试"}
                # 完全空响应
                return {"error": "AI 返回空内容，可能是推理模型思考过长未生成正式回答，请重试"}

        return {"result": str(content)}
    except Exception as e:
        return {"error": f"AI 调用失败: {str(e)}"}


def _try_fix_and_load(candidate: str):
    """尝试解析 JSON，失败时做常见容错修复再解析。"""
    if not candidate:
        return None
    # 1) 直接解析
    try:
        return json.loads(candidate)
    except Exception:
        pass
    # 2) 常见非法 JSON 修复：尾逗号 / NaN / Infinity / Python 布尔与 None
    fixed = candidate
    fixed = re.sub(r",\s*([}\]])", r"\1", fixed)          # 去掉 { 或 [ 前的尾逗号
    fixed = re.sub(r"\bNaN\b", "null", fixed)
    fixed = re.sub(r"\b(-?Infinity)\b", "null", fixed)
    fixed = re.sub(r"\bTrue\b", "true", fixed)
    fixed = re.sub(r"\bFalse\b", "false", fixed)
    fixed = re.sub(r"\bNone\b", "null", fixed)
    # 前导零（彩票场景高频）：JSON 不允许 05/07 这类数字，而模型看到
    # 历史开奖格式（01 12 14 ...）后会模仿输出带前导零的号码，
    # 导致 json.loads 直接抛错。仅处理数组内部，避免误伤字符串。
    fixed = re.sub(r"\[[^\]]*\]",
                   lambda m: re.sub(r"(?<![\w.])0+(\d+)", r"\1", m.group(0)),
                   fixed)
    try:
        return json.loads(fixed)
    except Exception:
        pass
    # 3) 全单引号 -> 双引号（仅在原文本无双引号时尝试，避免误伤）
    if '"' not in candidate:
        try:
            return json.loads(candidate.replace("'", '"'))
        except Exception:
            pass
    return None


def _safe_json_parse(text):
    """容错地从模型文本中提取 JSON 对象或数组。

    依次尝试：直接解析 -> 剥离 <think> 推理块 -> 剥离 ```json 代码围栏
    -> 括号配平截取最外层 { } 或 [ ]（含截断补全）。
    返回 dict/list，解析失败返回 None。
    """
    if not text or not isinstance(text, str):
        return None
    s = text.strip()
    # 0) 剥离推理模型的 <think>...</think> 块（DeepSeek-R1 等）
    s = re.sub(r"<think>[\s\S]*?</think>", "", s, flags=re.IGNORECASE).strip()
    # 1) 直接解析整段
    try:
        return json.loads(s)
    except Exception:
        pass
    # 2) 剥离 ```json ... ``` 或 ``` ... ``` 代码围栏
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", s, re.IGNORECASE)
    if fence:
        inner = fence.group(1).strip()
        parsed = _try_fix_and_load(inner)
        if parsed is not None:
            return parsed
        s = inner
    # 3) 括号配平：从最外层 { 或 [ 起，用完整括号栈截取（支持嵌套 + 截断补全）
    for opener in ("{", "["):
        start = s.find(opener)
        if start == -1:
            continue
        stack = []
        in_str = False
        esc = False
        end = len(s)
        for i in range(start, len(s)):
            c = s[i]
            if esc:
                esc = False
                continue
            if c == "\\":
                esc = True
                continue
            if c == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if c in "{[":
                stack.append(c)
            elif c in "]}":
                if stack and ((c == "}" and stack[-1] == "{") or (c == "]" and stack[-1] == "[")):
                    stack.pop()
                    if not stack:
                        end = i + 1
                        break
        candidate = s[start:end]
        parsed = _try_fix_and_load(candidate)
        if parsed is not None:
            return parsed
        # 截断补全：JSON 被 max_tokens 截断导致未闭合，按栈逆序补闭合括号再解析
        if stack:
            closers = "".join("}" if o == "{" else "]" for o in reversed(stack))
            recovered = _try_fix_and_load(s[start:] + closers)
            if recovered is not None:
                return recovered
    return None


def _call_ai_json(messages, max_tokens=5000, temperature=0.5, retries=2):
    """调用 AI 并解析 JSON；解析失败时自动降温度重试（输出在 temperature>0 时具有随机性）。

    返回 (parsed, error)：成功时 error 为 None，失败时 parsed 为 None 且 error 为可读信息。
    """
    last_err = "AI 返回格式异常，请重试"
    last_raw = ""
    for attempt in range(retries + 1):
        t = temperature if attempt == 0 else max(0.1, temperature - 0.2)
        # 推理模型（DeepSeek-R1）重试时加大 max_tokens，避免推理占满配额
        mt = max_tokens if attempt == 0 else max_tokens + 2000
        result = _call_ai(messages, max_tokens=mt, temperature=t)
        if "error" in result:
            # 空响应/截断等可恢复错误，继续重试而非直接返回
            if attempt < retries:
                last_raw = ""
                logger.info(f"[_call_ai_json] attempt {attempt+1} 失败: {result['error']}，重试中...")
                continue
            return None, result["error"]
        raw = result.get("result", "")
        last_raw = raw
        parsed = _safe_json_parse(raw)
        if parsed is not None:
            return parsed, None
    # 解析失败：记录原始返回，便于排查（日志 + 错误提示带片段）
    logger.warning(f"[_call_ai_json] 解析失败，原始返回前300字: {last_raw[:300]!r}")
    snippet = last_raw.strip()[:200].replace("\n", " ") if last_raw else "(空响应)"
    return None, f"{last_err}。原始返回前200字：{snippet}"


def _calculate_missing_periods(series, population):
    periods = {}
    for num in population:
        last_occurrence = -1
        for idx, val in enumerate(series):
            if val == num:
                periods[num] = idx - last_occurrence - 1 if last_occurrence >= 0 else idx
                last_occurrence = idx
        if last_occurrence == -1:
            periods[num] = len(series)
        else:
            periods[num] = len(series) - last_occurrence - 1
    return periods


def _calculate_consecutive_prob(df, cols):
    consecutive_count = 0
    total_count = 0
    for _, row in df.iterrows():
        nums = sorted([row[col] for col in cols if col in df.columns])
        for i in range(len(nums) - 1):
            total_count += 1
            if nums[i + 1] == nums[i] + 1:
                consecutive_count += 1
    return round(consecutive_count / total_count * 100, 1) if total_count > 0 else 0


def _calculate_sum_distribution(df, cols):
    sums = []
    for _, row in df.iterrows():
        row_sum = sum([row[col] for col in cols if col in df.columns])
        sums.append(row_sum)
    if not sums:
        return {"min": 0, "max": 0, "avg": 0, "mode": 0}
    return {
        "min": min(sums),
        "max": max(sums),
        "avg": round(sum(sums) / len(sums), 1),
        "mode": max(set(sums), key=sums.count)
    }


def _calculate_parity_trend(df, cols):
    trends = []
    for _, row in df.iterrows():
        nums = [row[col] for col in cols if col in df.columns]
        odd_count = sum(1 for n in nums if n % 2 == 1)
        even_count = len(nums) - odd_count
        trends.append(f"{odd_count}:{even_count}")
    recent_trends = trends[-10:]
    return recent_trends


def _calculate_zone_distribution(df, cols, zones):
    zone_counts = {name: 0 for name in zones.keys()}
    for _, row in df.iterrows():
        nums = [row[col] for col in cols if col in df.columns]
        for num in nums:
            for name, (low, high) in zones.items():
                if low <= num <= high:
                    zone_counts[name] += 1
                    break
    total = sum(zone_counts.values())
    return {k: round(v / total * 100, 1) for k, v in zone_counts.items()} if total > 0 else zone_counts


def _calculate_volatility(df, cols):
    recent_data = df.head(20)
    variances = []
    for col in cols:
        if col in recent_data.columns:
            variances.append(recent_data[col].var())
    if not variances:
        return 0
    avg_variance = sum(variances) / len(variances)
    max_variance = max(variances)
    volatility = avg_variance / max_variance if max_variance > 0 else 0
    return min(volatility, 1.0)


# ==== 新增 v2.0 特征工程函数 ====

def _calculate_ac_values(df, cols):
    """计算最近 N 期的 AC 值序列"""
    ac_values = []
    for _, row in df.iterrows():
        nums = sorted([int(row[col]) for col in cols if col in df.columns])
        if len(nums) >= 2:
            n = len(nums)
            diffs = set()
            for i in range(n):
                for j in range(i + 1, n):
                    diffs.add(abs(nums[i] - nums[j]))
            ac = len(diffs) - (n - 1)
            ac_values.append(ac)
    if not ac_values:
        return {"recent": [], "avg": 0, "mode": 0}
    recent = ac_values[:10]
    return {
        "recent": recent,
        "avg": round(sum(ac_values) / len(ac_values), 1),
        "mode": max(set(ac_values), key=ac_values.count)
    }


def _calculate_span_distribution(df, cols):
    """计算跨度分布"""
    spans = []
    for _, row in df.iterrows():
        nums = [int(row[col]) for col in cols if col in df.columns]
        if nums:
            spans.append(max(nums) - min(nums))
    if not spans:
        return {"recent": [], "avg": 0, "min": 0, "max": 0}
    recent = spans[:10]
    return {
        "recent": recent,
        "avg": round(sum(spans) / len(spans), 1),
        "min": min(spans),
        "max": max(spans)
    }


def _calculate_012_distribution(df, cols):
    """计算 012 路分布"""
    dist = {0: 0, 1: 0, 2: 0}
    total = 0
    for _, row in df.iterrows():
        for col in cols:
            if col in df.columns:
                val = int(row[col])
                dist[val % 3] += 1
                total += 1
    if total == 0:
        return {0: 0, 1: 0, 2: 0}
    return {k: round(v / total * 100, 1) for k, v in dist.items()}


def _calculate_prime_composite_ratio(df, cols):
    """计算质合比"""
    primes_set = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31}
    prime_count = 0
    composite_count = 0
    for _, row in df.iterrows():
        for col in cols:
            if col in df.columns:
                val = int(row[col])
                if val in primes_set:
                    prime_count += 1
                else:
                    composite_count += 1
    total = prime_count + composite_count
    if total == 0:
        return {"prime_pct": 0, "composite_pct": 0}
    return {
        "prime_pct": round(prime_count / total * 100, 1),
        "composite_pct": round(composite_count / total * 100, 1)
    }


def _calculate_tail_distribution(df, cols):
    """计算尾数分布，并返回最近10期尾数种类数"""
    tail_counts = {}
    for _, row in df.iterrows():
        for col in cols:
            if col in df.columns:
                tail = int(row[col]) % 10
                tail_counts[tail] = tail_counts.get(tail, 0) + 1
    total = sum(tail_counts.values())
    if total == 0:
        return {"recent": 0, "distribution": {}}
    distribution = {str(k): round(v / total * 100, 1) for k, v in sorted(tail_counts.items())}
    # 最近10期出现的尾数种类数
    recent_tails = set()
    for _, row in df.head(10).iterrows():
        for col in cols:
            if col in df.columns:
                recent_tails.add(int(row[col]) % 10)
    return {"recent": len(recent_tails), "distribution": distribution}


def _calculate_big_small_ratio(df, cols, midpoint=17):
    """计算大小比（双色球: 1-16小 17-33大；3D: 0-4小 5-9大；快乐8: 1-40小 41-80大）"""
    big = 0
    small = 0
    for _, row in df.iterrows():
        for col in cols:
            if col in df.columns:
                val = int(row[col])
                if val >= midpoint:
                    big += 1
                else:
                    small += 1
    total = big + small
    if total == 0:
        return {"big_pct": 0, "small_pct": 0}
    return {
        "big_pct": round(big / total * 100, 1),
        "small_pct": round(small / total * 100, 1)
    }


def _calculate_decay_weighted_freq(df, cols, population, decay=0.95):
    """指数衰减加权频率 TOP N"""
    import numpy as np
    weights = {}
    n = len(df)
    for idx, (_, row) in enumerate(df.iterrows()):
        w = decay ** idx
        for col in cols:
            if col in df.columns:
                val = int(row[col])
                weights[val] = weights.get(val, 0) + w
    for p in population:
        if p not in weights:
            weights[p] = 0
    sorted_items = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    return {str(k): round(v, 2) for k, v in sorted_items[:15]}


def _calculate_dragon_phoenix(df, cols):
    """龙头凤尾分析"""
    if df.empty:
        return {}
    dragons = []
    phoenixes = []
    for _, row in df.iterrows():
        nums = sorted([int(row[col]) for col in cols if col in df.columns])
        if nums:
            dragons.append(nums[0])
            phoenixes.append(nums[-1])
    return {
        "dragon_recent": dragons[:10],
        "dragon_avg": round(sum(dragons) / len(dragons), 1) if dragons else 0,
        "phoenix_recent": phoenixes[:10],
        "phoenix_avg": round(sum(phoenixes) / len(phoenixes), 1) if phoenixes else 0
    }


# 获取增强特征摘要（从 enhanced_predict 模块）
def _get_enhanced_feature_summary(lottery_type: str) -> dict:
    """从增强预测引擎获取特征摘要"""
    try:
        from enhanced_predict import get_feature_summary
        return get_feature_summary(lottery_type)
    except Exception:
        return {}


def ai_predict_ssq(n_groups: int = 5) -> dict:
    df = _read_lottery_data("ssq")
    
    if df.empty:
        return {"error": "暂无双色球历史数据，请先同步数据"}
    
    recent_10 = df.head(10)
    recent_30 = df.head(30)
    recent_50 = df.head(50)
    
    all_reds_30 = pd.concat([recent_30['r1'], recent_30['r2'], recent_30['r3'], 
                            recent_30['r4'], recent_30['r5'], recent_30['r6']])
    all_reds_50 = pd.concat([recent_50['r1'], recent_50['r2'], recent_50['r3'], 
                            recent_50['r4'], recent_50['r5'], recent_50['r6']])
    red_counts_30 = all_reds_30.value_counts()
    red_counts_50 = all_reds_50.value_counts()
    blue_counts_30 = recent_30['blue'].value_counts()
    blue_counts_50 = recent_50['blue'].value_counts()
    
    red_cols = ['r1', 'r2', 'r3', 'r4', 'r5', 'r6']
    red_population = list(range(1, 34))
    blue_population = list(range(1, 17))
    
    missing_periods = _calculate_missing_periods(all_reds_50, red_population)
    sorted_missing = sorted(missing_periods.items(), key=lambda x: x[1], reverse=True)
    coldest_reds = {k: v for k, v in sorted_missing[:8]}
    hottest_reds = {k: v for k, v in sorted_missing[-8:]}
    
    consecutive_prob = _calculate_consecutive_prob(recent_30, red_cols)
    sum_dist = _calculate_sum_distribution(recent_30, red_cols)
    parity_trend = _calculate_parity_trend(recent_30, red_cols)
    zone_dist = _calculate_zone_distribution(recent_30, red_cols, {
        "一区(01-11)": (1, 11),
        "二区(12-22)": (12, 22),
        "三区(23-33)": (23, 33)
    })
    recent_20 = df.head(20)
    volatility = _calculate_volatility(recent_20, red_cols)
    
    # ==== v2.0 新增特征 ====
    ac_data = _calculate_ac_values(recent_30, red_cols)
    span_data = _calculate_span_distribution(recent_30, red_cols)
    d012_data = _calculate_012_distribution(recent_30, red_cols)
    pc_data = _calculate_prime_composite_ratio(recent_30, red_cols)
    tail_data = _calculate_tail_distribution(recent_30, red_cols)
    bs_data = _calculate_big_small_ratio(recent_30, red_cols, midpoint=17)
    decay_data = _calculate_decay_weighted_freq(recent_50, red_cols, red_population, decay=0.95)
    dp_data = _calculate_dragon_phoenix(recent_30, red_cols)
    
    # 增强特征摘要
    enhanced_summary = _get_enhanced_feature_summary("ssq")
    enhanced_str = ""
    if enhanced_summary and "error" not in enhanced_summary:
        fusion_top = enhanced_summary.get("fusion_top10", {})
        bounce_top = enhanced_summary.get("missing_bounce_top5", {})
        markov_top = enhanced_summary.get("markov_top10", {})
        mc_top = enhanced_summary.get("monte_carlo_top10", {})
        if fusion_top:
            enhanced_str += f"""
【AI增强分析 - 贝叶斯融合权重 TOP10】
{fusion_top}

【遗漏回补概率 TOP5】
{bounce_top}

【马尔可夫转移概率 TOP10】
{markov_top}

【蒙特卡洛模拟置信度 TOP10】
{mc_top}
"""
    
    temperature = 0.5 + volatility * 0.4
    
    prompt = f"""你是一个专业的彩票数据分析师。请基于以下双色球历史多维数据，生成 {n_groups} 组推荐号码。

【最近10期开奖数据】
{recent_10[['code', 'date', 'r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'blue']].to_string(index=False)}

【最近30期热号TOP10（红球）】
{red_counts_30.head(10).to_dict()}

【最近50期热号TOP10（红球）】
{red_counts_50.head(10).to_dict()}

【最近30期蓝球频次TOP5】
{blue_counts_30.head(5).to_dict()}

【最近50期蓝球频次TOP5】
{blue_counts_50.head(5).to_dict()}

【衰减加权频率 TOP15（越近权重越高）】
{decay_data}

【遗漏周期分析】
- 最冷红球TOP8（遗漏期数最多）: {coldest_reds}
- 最热红球TOP8（遗漏期数最少）: {hottest_reds}

【AC值（算术复杂度）分析】最近30期
- 最近10期: {ac_data['recent']}
- 平均值: {ac_data['avg']}（理想区间 4-6）

【跨度分析】最近30期
- 最近10期: {span_data['recent']}
- 平均值: {span_data['avg']: .1f}（理想区间 18-30）

【龙头凤尾分析】最近30期
- 龙头（最小值）最近10期: {dp_data.get('dragon_recent', [])}
- 凤尾（最大值）最近10期: {dp_data.get('phoenix_recent', [])}
- 龙头均值: {dp_data.get('dragon_avg', 0)}，凤尾均值: {dp_data.get('phoenix_avg', 0)}

【012路分布】最近30期
- 0路: {d012_data[0]}% | 1路: {d012_data[1]}% | 2路: {d012_data[2]}%
- 合理分布：2-2-2(35%) 或 3-2-1(45%)

【质合比】最近30期
- 质数: {pc_data['prime_pct']}% | 合数: {pc_data['composite_pct']}%

【大小比】最近30期（1-16小, 17-33大）
- 大号: {bs_data['big_pct']}% | 小号: {bs_data['small_pct']}%

【尾数分布】最近30期 - {tail_data}

【连号概率】最近30期连号出现概率: {consecutive_prob}%

【和值分布】最近30期红球和值统计：
- 最小值: {sum_dist['min']}，最大值: {sum_dist['max']}
- 平均值: {sum_dist['avg']}，众数: {sum_dist['mode']}

【奇偶比例趋势】最近10期奇偶比例: {', '.join(parity_trend)}

【区间分布】最近30期各区间占比：
- 一区(01-11): {zone_dist['一区(01-11)']}%
- 二区(12-22): {zone_dist['二区(12-22)']}%
- 三区(23-33): {zone_dist['三区(23-33)']}%

【数据波动指数】当前数据波动: {round(volatility, 2)}（0=稳定，1=波动大）
{enhanced_str}
【分析要求 — 请严格按照以下约束生成】
1. 每组6个红球（1-33，不重复）+ 1个蓝球（1-16）
2. **AC值约束**：确保红球组合 AC 值落在 [3, 7]（理想4-6）
3. **跨度约束**：跨度（最大-最小）应在 {int(span_data['avg'])-5} 到 {int(span_data['avg'])+5} 之间
4. **012路平衡**：每路的红球数不能为 0，避免 5-1-0 极端分布
5. **质合比**：质数 1-4 个，合数 2-5 个
6. **尾数多样性**：6 个红球应有至少 4 种不同的尾数
7. **奇偶比**：建议 3:3 或 4:2（参考趋势）
8. **区间覆盖**：三个区间都不能完全断档（0个）
9. **和值范围**：建议在 {int(sum_dist['avg'])-12} 到 {int(sum_dist['avg'])+12} 之间
10. 避免与最近5期开奖号码重复超过3个红球
11. 每组给出基于数据的简短分析理由

【输出格式】严格按以下JSON格式，不要添加任何其他文字：
{{
  "recommendations": [
    {{
      "group": 1,
      "numbers": {{"red": [红球1, 红球2, 红球3, 红球4, 红球5, 红球6], "blue": 蓝球}},
      "reason": "简短分析"
    }}
  ],
  "analysis": "整体趋势分析（300字以内）"
}}"""

    messages = [
        {"role": "system", "content": "你是专业的彩票数据分析师，擅长基于历史数据预测号码。请严格按照要求的JSON格式输出，确保号码满足所有约束条件，只返回一个用```json代码块包裹的JSON对象，不要输出任何解释性文字。"},
        {"role": "user", "content": prompt}
    ]
    
    parsed, err = _call_ai_json(messages, max_tokens=max(6000, n_groups * 90 + 3000), temperature=temperature)
    if err:
        return {"error": err}
    return parsed


def ai_predict_kl8(n_groups: int = 5, pick_size: int = 10) -> dict:
    """快乐8 AI预测。

    Args:
        n_groups: 生成组数
        pick_size: 选号个数（选一=1 ~ 选十=10），默认10（选十玩法）
    """
    pick_size = max(1, min(10, int(pick_size)))  # 限制 1-10
    _play_names = {1:"选一",2:"选二",3:"选三",4:"选四",5:"选五",
                   6:"选六",7:"选七",8:"选八",9:"选九",10:"选十"}
    play_name = _play_names.get(pick_size, f"选{pick_size}")

    df = _read_lottery_data("kl8")

    if df.empty:
        return {"error": "暂无快乐8历史数据，请先同步数据"}
    
    recent_10 = df.head(10)
    recent_30 = df.head(30)
    recent_50 = df.head(50)
    
    cols = [f"n{i:02d}" for i in range(1, 21)]
    kl8_population = list(range(1, 81))
    all_nums_30 = pd.concat([recent_30[col] for col in cols if col in recent_30.columns])
    all_nums_50 = pd.concat([recent_50[col] for col in cols if col in recent_50.columns])
    num_counts_30 = all_nums_30.value_counts()
    num_counts_50 = all_nums_50.value_counts()
    
    missing_periods = _calculate_missing_periods(all_nums_50, kl8_population)
    sorted_missing = sorted(missing_periods.items(), key=lambda x: x[1], reverse=True)
    coldest_nums = {k: v for k, v in sorted_missing[:10]}
    hottest_nums = {k: v for k, v in sorted_missing[-10:]}
    
    consecutive_prob = _calculate_consecutive_prob(recent_30, cols)
    sum_dist = _calculate_sum_distribution(recent_30, cols)
    parity_trend = _calculate_parity_trend(recent_30, cols)
    zone_dist = _calculate_zone_distribution(recent_30, cols, {
        "一区(01-20)": (1, 20),
        "二区(21-40)": (21, 40),
        "三区(41-60)": (41, 60),
        "四区(61-80)": (61, 80)
    })
    recent_20 = df.head(20)
    volatility = _calculate_volatility(recent_20, cols)
    
    # v2.0 新增
    span_data = _calculate_span_distribution(recent_30, cols)
    d012_data = _calculate_012_distribution(recent_30, cols)
    tail_data = _calculate_tail_distribution(recent_30, cols)
    bs_data = _calculate_big_small_ratio(recent_30, cols, midpoint=41)
    decay_data = _calculate_decay_weighted_freq(recent_50, cols, kl8_population, decay=0.95)
    
    enhanced_summary = _get_enhanced_feature_summary("kl8")
    enhanced_str = ""
    if enhanced_summary and "error" not in enhanced_summary:
        fusion_top = enhanced_summary.get("fusion_top10", {})
        bounce_top = enhanced_summary.get("missing_bounce_top5", {})
        markov_top = enhanced_summary.get("markov_top10", {})
        mc_top = enhanced_summary.get("monte_carlo_top10", {})
        if fusion_top:
            enhanced_str += f"""
【贝叶斯融合权重 TOP10】{fusion_top}
【遗漏回补概率 TOP5】{bounce_top}
【马尔可夫转移 TOP10】{markov_top}
【蒙特卡洛置信度 TOP10】{mc_top}
"""
    
    temperature = 0.5 + volatility * 0.4
    
    # 动态约束：根据 pick_size 调整区间和大小约束
    if pick_size >= 8:
        _zone_hint = "四个区间各有 2-3 个号码，不能有区间完全断档"
        _hot_cold = "热号与冷号的平衡（建议 6:4 或 7:3）"
        _repeat_limit = 5
    elif pick_size >= 5:
        _zone_hint = "尽量覆盖至少3个区间，避免过度集中"
        _hot_cold = "热号与冷号的平衡（建议 6:4 或 5:5）"
        _repeat_limit = 4
    else:
        _zone_hint = "尽量分散在不同区间"
        _hot_cold = "优先选择热号，兼顾冷号回补"
        _repeat_limit = 3

    prompt = f"""你是一个专业的彩票数据分析专家。请基于以下快乐8历史多维数据，生成 {n_groups} 组"{play_name}"推荐号码（每组选 {pick_size} 个号码）。

【最近10期开奖数据（前5期示例）】
{recent_10.head(5)[['code', 'date', 'n01', 'n02', 'n03', 'n04', 'n05']].to_string(index=False)}

【最近30期热号TOP15】
{num_counts_30.head(15).to_dict()}

【最近50期热号TOP15】
{num_counts_50.head(15).to_dict()}

【衰减加权频率 TOP15】
{decay_data}

【遗漏周期分析】
- 最冷号码TOP10（遗漏期数最多）: {coldest_nums}
- 最热号码TOP10（遗漏期数最少）: {hottest_nums}

【跨度分析】最近30期 - 最近10期跨度: {span_data['recent']}，均值: {span_data['avg']:.1f}

【012路分布】最近30期 - 0路: {d012_data[0]}% | 1路: {d012_data[1]}% | 2路: {d012_data[2]}%

【尾数分布】最近30期 - {tail_data}

【大小比】最近30期（1-40小, 41-80大）- 大号: {bs_data['big_pct']}% | 小号: {bs_data['small_pct']}%

【连号概率】最近30期连号出现概率: {consecutive_prob}%

【和值分布】最近30期号码和值统计：
- 最小: {sum_dist['min']}，最大: {sum_dist['max']}
- 平均: {sum_dist['avg']}，众数: {sum_dist['mode']}

【奇偶比例趋势】最近10期: {', '.join(parity_trend)}

【区间分布】最近30期各区间占比：
- 一区(01-20): {zone_dist['一区(01-20)']}%
- 二区(21-40): {zone_dist['二区(21-40)']}%
- 三区(41-60): {zone_dist['三区(41-60)']}%
- 四区(61-80): {zone_dist['四区(61-80)']}%

【数据波动指数】{round(volatility, 2)}
{enhanced_str}
【分析要求】
1. 每组{pick_size}个号码（1-80），不重复
2. **区间约束**：{_zone_hint}
3. **大小平衡**：参考趋势分布
4. **012路约束**：避免极端偏态
5. {_hot_cold}
6. 和值范围在 {int(sum_dist['avg'] * pick_size / 20)-20} 到 {int(sum_dist['avg'] * pick_size / 20)+20} 之间
7. 避免与最近5期重复超过{_repeat_limit}个号码
8. 每组给出基于数据的简短分析

【输出格式】严格按以下JSON格式：
{{
  "recommendations": [
    {{
      "group": 1,
      "numbers": [号码1, ..., 号码{pick_size}],
      "reason": "简短分析"
    }}
  ],
  "analysis": "整体趋势分析（300字以内）"
}}"""

    messages = [
        {"role": "system", "content": "你是专业的彩票数据分析师，擅长基于历史数据预测号码。请严格按照要求的JSON格式输出，只返回一个用```json代码块包裹的JSON对象，不要输出任何解释性文字。"},
        {"role": "user", "content": prompt}
    ]
    
    parsed, err = _call_ai_json(messages, max_tokens=max(6000, n_groups * 90 + 3000), temperature=temperature)
    if err:
        return {"error": err}
    # 附带玩法信息
    if isinstance(parsed, dict):
        parsed["play_name"] = play_name
        parsed["pick_size"] = pick_size
    return parsed


def ai_predict_fcsd(n_groups: int = 5) -> dict:
    df = _read_lottery_data("fcsd")
    
    if df.empty:
        return {"error": "暂无福彩3D历史数据，请先同步数据"}
    
    recent_10 = df.head(10)
    recent_30 = df.head(30)
    recent_50 = df.head(50)
    
    n1_hot_30 = recent_30['n1'].value_counts().head(5).to_dict()
    n2_hot_30 = recent_30['n2'].value_counts().head(5).to_dict()
    n3_hot_30 = recent_30['n3'].value_counts().head(5).to_dict()
    
    n1_hot_50 = recent_50['n1'].value_counts().head(5).to_dict()
    n2_hot_50 = recent_50['n2'].value_counts().head(5).to_dict()
    n3_hot_50 = recent_50['n3'].value_counts().head(5).to_dict()
    
    fcsd_pop = list(range(10))
    n1_missing = _calculate_missing_periods(recent_50['n1'], fcsd_pop)
    n2_missing = _calculate_missing_periods(recent_50['n2'], fcsd_pop)
    n3_missing = _calculate_missing_periods(recent_50['n3'], fcsd_pop)
    
    sorted_n1_missing = sorted(n1_missing.items(), key=lambda x: x[1], reverse=True)
    sorted_n2_missing = sorted(n2_missing.items(), key=lambda x: x[1], reverse=True)
    sorted_n3_missing = sorted(n3_missing.items(), key=lambda x: x[1], reverse=True)
    
    n1_coldest = {k: v for k, v in sorted_n1_missing[:3]}
    n2_coldest = {k: v for k, v in sorted_n2_missing[:3]}
    n3_coldest = {k: v for k, v in sorted_n3_missing[:3]}
    
    n1_hottest = {k: v for k, v in sorted_n1_missing[-3:]}
    n2_hottest = {k: v for k, v in sorted_n2_missing[-3:]}
    n3_hottest = {k: v for k, v in sorted_n3_missing[-3:]}
    
    cols = ['n1', 'n2', 'n3']
    sum_dist = _calculate_sum_distribution(recent_30, cols)
    parity_trend = _calculate_parity_trend(recent_30, cols)
    
    recent_20 = df.head(20)
    volatility = _calculate_volatility(recent_20, cols)
    
    # v2.0 新增
    d012_data = _calculate_012_distribution(recent_30, cols)
    bs_data = _calculate_big_small_ratio(recent_30, cols, midpoint=5)
    span_data = _calculate_span_distribution(recent_30, cols)
    
    enhanced_summary = _get_enhanced_feature_summary("fcsd")
    enhanced_str = ""
    if enhanced_summary and "error" not in enhanced_summary:
        fusion_top = enhanced_summary.get("fusion_top10", {})
        if fusion_top:
            enhanced_str = f"\n【贝叶斯融合权重】{fusion_top}\n"
    
    temperature = 0.5 + volatility * 0.4
    
    prompt = f"""你是一个专业的彩票数据分析专家。请基于以下福彩3D历史多维数据，生成 {n_groups} 组推荐号码。

【最近10期开奖数据】
{recent_10[['code', 'date', 'n1', 'n2', 'n3']].to_string(index=False)}

【百位热号TOP5（最近30期）】{n1_hot_30}
【十位热号TOP5（最近30期）】{n2_hot_30}
【个位热号TOP5（最近30期）】{n3_hot_30}

【百位热号TOP5（最近50期）】{n1_hot_50}
【十位热号TOP5（最近50期）】{n2_hot_50}
【个位热号TOP5（最近50期）】{n3_hot_50}

【遗漏周期分析（最近50期）】
- 百位最冷TOP3: {n1_coldest}，最热TOP3: {n1_hottest}
- 十位最冷TOP3: {n2_coldest}，最热TOP3: {n2_hottest}
- 个位最冷TOP3: {n3_coldest}，最热TOP3: {n3_hottest}

【和值分布】最近30期 - 最小: {sum_dist['min']}，最大: {sum_dist['max']}，平均: {sum_dist['avg']}，众数: {sum_dist['mode']}

【跨度分布】最近30期 - 最近10期跨度: {span_data['recent']}，均值: {span_data['avg']:.1f}

【012路分布】最近30期 - 0路: {d012_data[0]}% | 1路: {d012_data[1]}% | 2路: {d012_data[2]}%

【奇偶比例趋势】最近10期: {', '.join(parity_trend)}

【大小比】最近30期（0-4小, 5-9大）- 大号: {bs_data['big_pct']}% | 小号: {bs_data['small_pct']}%

【数据波动指数】{round(volatility, 2)}
{enhanced_str}
【分析要求】
1. 每组3个数字（0-9），直选
2. **分位独立分析**：每位综合考虑其独立的热/冷/温趋势
3. **和值约束**：和值在 {int(sum_dist['avg'])-6} 到 {int(sum_dist['avg'])+6} 之间
4. **跨度约束**：跨度在 {int(span_data['avg'])-3} 到 {int(span_data['avg'])+3} 之间
5. **大小平衡**：至少有大号也有小号
6. 避免与最近3期完全重复
7. 每组给出基于数据的简短分析

【输出格式】严格按以下JSON格式：
{{
  "recommendations": [
    {{
      "group": 1,
      "numbers": [百位, 十位, 个位],
      "reason": "简短分析"
    }}
  ],
  "analysis": "整体趋势分析（200字以内）"
}}"""

    messages = [
        {"role": "system", "content": "你是专业的彩票数据分析师，擅长基于历史数据预测号码。请严格按照要求的JSON格式输出，只返回一个用```json代码块包裹的JSON对象，不要输出任何解释性文字。"},
        {"role": "user", "content": prompt}
    ]
    
    parsed, err = _call_ai_json(messages, max_tokens=max(6000, n_groups * 90 + 3000), temperature=temperature)
    if err:
        return {"error": err}
    return parsed


def ai_predict_dlt(n_groups: int = 5) -> dict:
    """大乐透AI预测：5个前区(1-35) + 2个后区(1-12)"""
    df = _read_lottery_data("dlt")
    
    if df.empty:
        return {"error": "暂无大乐透历史数据，请先同步数据"}
    
    recent_10 = df.head(10)
    recent_30 = df.head(30)
    recent_50 = df.head(50)
    
    all_fronts_30 = pd.concat([recent_30['f1'], recent_30['f2'], recent_30['f3'],
                               recent_30['f4'], recent_30['f5']])
    all_fronts_50 = pd.concat([recent_50['f1'], recent_50['f2'], recent_50['f3'],
                               recent_50['f4'], recent_50['f5']])
    front_counts_30 = all_fronts_30.value_counts()
    front_counts_50 = all_fronts_50.value_counts()
    
    all_backs_30 = pd.concat([recent_30['b1'], recent_30['b2']])
    all_backs_50 = pd.concat([recent_50['b1'], recent_50['b2']])
    back_counts_30 = all_backs_30.value_counts()
    back_counts_50 = all_backs_50.value_counts()
    
    front_cols = ['f1', 'f2', 'f3', 'f4', 'f5']
    front_population = list(range(1, 36))
    
    missing_periods = _calculate_missing_periods(all_fronts_50, front_population)
    sorted_missing = sorted(missing_periods.items(), key=lambda x: x[1], reverse=True)
    coldest_fronts = {k: v for k, v in sorted_missing[:10]}
    hottest_fronts = {k: v for k, v in sorted_missing[-10:]}
    
    consecutive_prob = _calculate_consecutive_prob(recent_30, front_cols)
    sum_dist = _calculate_sum_distribution(recent_30, front_cols)
    parity_trend = _calculate_parity_trend(recent_30, front_cols)
    zone_dist = _calculate_zone_distribution(recent_30, front_cols, {
        "一区(01-12)": (1, 12),
        "二区(13-24)": (13, 24),
        "三区(25-35)": (25, 35)
    })
    recent_20 = df.head(20)
    volatility = _calculate_volatility(recent_20, front_cols)
    
    # v2.0 新增特征
    ac_data = _calculate_ac_values(recent_30, front_cols)
    span_data = _calculate_span_distribution(recent_30, front_cols)
    d012_data = _calculate_012_distribution(recent_30, front_cols)
    pc_data = _calculate_prime_composite_ratio(recent_30, front_cols)
    tail_data = _calculate_tail_distribution(recent_30, front_cols)
    bs_data = _calculate_big_small_ratio(recent_30, front_cols, midpoint=18)
    decay_freq = _calculate_decay_weighted_freq(recent_30, front_cols, front_population)
    dragon_phoenix = _calculate_dragon_phoenix(recent_30, front_cols)
    
    enhanced_summary = _get_enhanced_feature_summary("dlt")
    enhanced_str = ""
    if enhanced_summary and "error" not in enhanced_summary:
        fusion_top = enhanced_summary.get("fusion_top10", {})
        omit_top = enhanced_summary.get("omission_top5", {})
        mc_top = enhanced_summary.get("monte_carlo_top10", {})
        if fusion_top:
            enhanced_str = f"\n【贝叶斯融合权重】{fusion_top}\n"
        if omit_top:
            enhanced_str += f"【遗漏回补概率】{omit_top}\n"
        if mc_top:
            enhanced_str += f"【蒙特卡洛置信度】{mc_top}\n"
    
    temperature = 0.5 + volatility * 0.4
    
    prompt = f"""你是一个专业的彩票数据分析专家。请基于以下大乐透历史多维数据，生成 {n_groups} 组推荐号码。

【最近10期开奖数据】
{recent_10[['code', 'date', 'f1', 'f2', 'f3', 'f4', 'f5', 'b1', 'b2']].to_string(index=False)}

【最近30期前区热号TOP10】{front_counts_30.head(10).to_dict()}
【最近50期前区热号TOP10】{front_counts_50.head(10).to_dict()}
【最近30期后区频次TOP5】{back_counts_30.head(5).to_dict()}
【最近50期后区频次TOP5】{back_counts_50.head(5).to_dict()}

【遗漏周期分析】
- 最冷前区TOP10: {coldest_fronts}
- 最热前区TOP10: {hottest_fronts}

【连号概率】最近30期: {consecutive_prob}%
【和值分布】最小: {sum_dist['min']}，最大: {sum_dist['max']}，平均: {sum_dist['avg']}，众数: {sum_dist['mode']}
【AC值趋势】最近10期: {ac_data['recent']}，均值: {ac_data['avg']:.1f}（理想4-6）
【跨度分布】最近10期: {span_data['recent']}，均值: {span_data['avg']:.1f}
【012路分布】0路: {d012_data[0]}% | 1路: {d012_data[1]}% | 2路: {d012_data[2]}%
【质合比】质数: {pc_data['prime_pct']}% | 合数: {pc_data['composite_pct']}%
【尾数分布】最近10期尾数种类: {tail_data['recent']}
【大小比】大号(18-35): {bs_data['big_pct']}% | 小号(1-17): {bs_data['small_pct']}%
【奇偶比例趋势】最近10期: {', '.join(parity_trend)}
【区间分布】一区(01-12): {zone_dist['一区(01-12)']}% | 二区(13-24): {zone_dist['二区(13-24)']}% | 三区(25-35): {zone_dist['三区(25-35)']}%
【龙头凤尾】龙头: {dragon_phoenix.get('dragon_recent', [])} | 凤尾: {dragon_phoenix.get('phoenix_recent', [])}
【衰减加权频率TOP10】{dict(sorted(decay_freq.items(), key=lambda x: x[1], reverse=True)[:10])}
【数据波动指数】{round(volatility, 2)}
{enhanced_str}
【分析要求】
1. 每组5个前区(1-35) + 2个后区(1-12)，前区升序排列，后区升序排列
2. **AC值约束**：AC值在3-7之间
3. **跨度约束**：跨度在{int(span_data['avg'])-8}到{int(span_data['avg'])+8}之间
4. **012路均衡**：任一路不超过3个
5. **质数1-3个**，**尾数≥4种**
6. **和值范围**：{int(sum_dist['avg'])-15}到{int(sum_dist['avg'])+15}
7. 后区2个号码不重复
8. 避免与最近5期重复超过3个前区号码
9. 每组给出简短分析理由

【输出格式】严格按以下JSON格式：
{{
  "recommendations": [
    {{
      "group": 1,
      "numbers": {{"front": [前区1-5], "back": [后区1-2]}},
      "reason": "简短分析"
    }}
  ],
  "analysis": "整体趋势分析（300字以内）"
}}"""

    messages = [
        {"role": "system", "content": "你是专业的彩票数据分析师，擅长基于历史数据预测号码。请严格按照要求的JSON格式输出，只返回一个用```json代码块包裹的JSON对象，不要输出任何解释性文字。"},
        {"role": "user", "content": prompt}
    ]
    
    parsed, err = _call_ai_json(messages, max_tokens=max(6000, n_groups * 90 + 3000), temperature=temperature)
    if err:
        return {"error": err}
    return parsed


def ai_predict_qxc(n_groups: int = 5) -> dict:
    """七星彩AI预测：7个位置各0-9"""
    df = _read_lottery_data("qxc")
    
    if df.empty:
        return {"error": "暂无七星彩历史数据，请先同步数据"}
    
    recent_10 = df.head(10)
    recent_30 = df.head(30)
    recent_50 = df.head(50)
    
    cols = ['n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7']
    pop = list(range(10))
    
    # 分位频次统计
    pos_hot_30 = {}
    pos_hot_50 = {}
    pos_coldest = {}
    pos_hottest = {}
    for col in cols:
        pos_hot_30[col] = recent_30[col].value_counts().head(5).to_dict()
        pos_hot_50[col] = recent_50[col].value_counts().head(5).to_dict()
        missing = _calculate_missing_periods(recent_50[col], pop)
        sorted_m = sorted(missing.items(), key=lambda x: x[1], reverse=True)
        pos_coldest[col] = {k: v for k, v in sorted_m[:3]}
        pos_hottest[col] = {k: v for k, v in sorted_m[-3:]}
    
    sum_dist = _calculate_sum_distribution(recent_30, cols)
    parity_trend = _calculate_parity_trend(recent_30, cols)
    recent_20 = df.head(20)
    volatility = _calculate_volatility(recent_20, cols)
    d012_data = _calculate_012_distribution(recent_30, cols)
    bs_data = _calculate_big_small_ratio(recent_30, cols, midpoint=5)
    span_data = _calculate_span_distribution(recent_30, cols)
    
    enhanced_summary = _get_enhanced_feature_summary("qxc")
    enhanced_str = ""
    if enhanced_summary and "error" not in enhanced_summary:
        fusion_top = enhanced_summary.get("fusion_top10", {})
        if fusion_top:
            enhanced_str = f"\n【贝叶斯融合权重】{fusion_top}\n"
    
    temperature = 0.5 + volatility * 0.4
    
    prompt = f"""你是一个专业的彩票数据分析专家。请基于以下七星彩历史多维数据，生成 {n_groups} 组推荐号码。

【最近10期开奖数据】
{recent_10[['code', 'date'] + cols].to_string(index=False)}

【各位热号TOP5（30期）】
{chr(10).join(f'- {col}: {pos_hot_30[col]}' for col in cols)}

【各位遗漏分析（50期）】
{chr(10).join(f'- {col}: 最冷{pos_coldest[col]} 最热{pos_hottest[col]}' for col in cols)}

【和值分布】最小: {sum_dist['min']}，最大: {sum_dist['max']}，平均: {sum_dist['avg']}，众数: {sum_dist['mode']}
【跨度分布】最近10期: {span_data['recent']}，均值: {span_data['avg']:.1f}
【012路分布】0路: {d012_data[0]}% | 1路: {d012_data[1]}% | 2路: {d012_data[2]}%
【奇偶比例趋势】最近10期: {', '.join(parity_trend)}
【大小比】大号(5-9): {bs_data['big_pct']}% | 小号(0-4): {bs_data['small_pct']}%
【数据波动指数】{round(volatility, 2)}
{enhanced_str}
【分析要求】
1. 每组7个数字（0-9），各位独立
2. **分位独立分析**：每位综合考虑其独立的热/冷/温趋势
3. **和值约束**：{int(sum_dist['avg'])-8}到{int(sum_dist['avg'])+8}
4. **大小平衡**：至少2个大号(5-9)和2个小号(0-4)
5. 避免与最近3期完全重复
6. 每组给出简短分析

【输出格式】严格按以下JSON格式：
{{
  "recommendations": [
    {{
      "group": 1,
      "numbers": [n1, n2, n3, n4, n5, n6, n7],
      "reason": "简短分析"
    }}
  ],
  "analysis": "整体趋势分析（200字以内）"
}}"""

    messages = [
        {"role": "system", "content": "你是专业的彩票数据分析师，擅长基于历史数据预测号码。请严格按照要求的JSON格式输出，只返回一个用```json代码块包裹的JSON对象，不要输出任何解释性文字。"},
        {"role": "user", "content": prompt}
    ]
    
    parsed, err = _call_ai_json(messages, max_tokens=max(6000, n_groups * 90 + 3000), temperature=temperature)
    if err:
        return {"error": err}
    return parsed


def ai_predict_pl3(n_groups: int = 5) -> dict:
    """排列三AI预测：3个位置各0-9"""
    df = _read_lottery_data("pl3")
    
    if df.empty:
        return {"error": "暂无排列三历史数据，请先同步数据"}
    
    recent_10 = df.head(10)
    recent_30 = df.head(30)
    recent_50 = df.head(50)
    
    n1_hot_30 = recent_30['n1'].value_counts().head(5).to_dict()
    n2_hot_30 = recent_30['n2'].value_counts().head(5).to_dict()
    n3_hot_30 = recent_30['n3'].value_counts().head(5).to_dict()
    
    n1_hot_50 = recent_50['n1'].value_counts().head(5).to_dict()
    n2_hot_50 = recent_50['n2'].value_counts().head(5).to_dict()
    n3_hot_50 = recent_50['n3'].value_counts().head(5).to_dict()
    
    pl3_pop = list(range(10))
    n1_missing = _calculate_missing_periods(recent_50['n1'], pl3_pop)
    n2_missing = _calculate_missing_periods(recent_50['n2'], pl3_pop)
    n3_missing = _calculate_missing_periods(recent_50['n3'], pl3_pop)
    
    sorted_n1_missing = sorted(n1_missing.items(), key=lambda x: x[1], reverse=True)
    sorted_n2_missing = sorted(n2_missing.items(), key=lambda x: x[1], reverse=True)
    sorted_n3_missing = sorted(n3_missing.items(), key=lambda x: x[1], reverse=True)
    
    n1_coldest = {k: v for k, v in sorted_n1_missing[:3]}
    n2_coldest = {k: v for k, v in sorted_n2_missing[:3]}
    n3_coldest = {k: v for k, v in sorted_n3_missing[:3]}
    
    n1_hottest = {k: v for k, v in sorted_n1_missing[-3:]}
    n2_hottest = {k: v for k, v in sorted_n2_missing[-3:]}
    n3_hottest = {k: v for k, v in sorted_n3_missing[-3:]}
    
    cols = ['n1', 'n2', 'n3']
    sum_dist = _calculate_sum_distribution(recent_30, cols)
    parity_trend = _calculate_parity_trend(recent_30, cols)
    
    recent_20 = df.head(20)
    volatility = _calculate_volatility(recent_20, cols)
    d012_data = _calculate_012_distribution(recent_30, cols)
    bs_data = _calculate_big_small_ratio(recent_30, cols, midpoint=5)
    span_data = _calculate_span_distribution(recent_30, cols)
    
    enhanced_summary = _get_enhanced_feature_summary("pl3")
    enhanced_str = ""
    if enhanced_summary and "error" not in enhanced_summary:
        fusion_top = enhanced_summary.get("fusion_top10", {})
        if fusion_top:
            enhanced_str = f"\n【贝叶斯融合权重】{fusion_top}\n"
    
    temperature = 0.5 + volatility * 0.4
    
    prompt = f"""你是一个专业的彩票数据分析专家。请基于以下排列三历史多维数据，生成 {n_groups} 组推荐号码。

【最近10期开奖数据】
{recent_10[['code', 'date', 'n1', 'n2', 'n3']].to_string(index=False)}

【百位热号TOP5（30期）】{n1_hot_30}
【十位热号TOP5（30期）】{n2_hot_30}
【个位热号TOP5（30期）】{n3_hot_30}

【百位热号TOP5（50期）】{n1_hot_50}
【十位热号TOP5（50期）】{n2_hot_50}
【个位热号TOP5（50期）】{n3_hot_50}

【遗漏周期分析（50期）】
- 百位最冷TOP3: {n1_coldest}，最热TOP3: {n1_hottest}
- 十位最冷TOP3: {n2_coldest}，最热TOP3: {n2_hottest}
- 个位最冷TOP3: {n3_coldest}，最热TOP3: {n3_hottest}

【和值分布】最小: {sum_dist['min']}，最大: {sum_dist['max']}，平均: {sum_dist['avg']}，众数: {sum_dist['mode']}
【跨度分布】最近10期: {span_data['recent']}，均值: {span_data['avg']:.1f}
【012路分布】0路: {d012_data[0]}% | 1路: {d012_data[1]}% | 2路: {d012_data[2]}%
【奇偶比例趋势】最近10期: {', '.join(parity_trend)}
【大小比】大号(5-9): {bs_data['big_pct']}% | 小号(0-4): {bs_data['small_pct']}%
【数据波动指数】{round(volatility, 2)}
{enhanced_str}
【分析要求】
1. 每组3个数字（0-9），直选
2. **分位独立分析**：每位综合考虑其独立的热/冷/温趋势
3. **和值约束**：{int(sum_dist['avg'])-6}到{int(sum_dist['avg'])+6}
4. **跨度约束**：{int(span_data['avg'])-3}到{int(span_data['avg'])+3}
5. **大小平衡**：至少有大号也有小号
6. 避免与最近3期完全重复
7. 每组给出简短分析

【输出格式】严格按以下JSON格式：
{{
  "recommendations": [
    {{
      "group": 1,
      "numbers": [百位, 十位, 个位],
      "reason": "简短分析"
    }}
  ],
  "analysis": "整体趋势分析（200字以内）"
}}"""

    messages = [
        {"role": "system", "content": "你是专业的彩票数据分析师，擅长基于历史数据预测号码。请严格按照要求的JSON格式输出，只返回一个用```json代码块包裹的JSON对象，不要输出任何解释性文字。"},
        {"role": "user", "content": prompt}
    ]
    
    parsed, err = _call_ai_json(messages, max_tokens=max(6000, n_groups * 90 + 3000), temperature=temperature)
    if err:
        return {"error": err}
    return parsed


def ai_analyze_trend(name: str) -> str:
    df = _read_lottery_data(name)
    
    if df.empty:
        return "❌ 暂无数据，请先同步数据"
    
    if name == "ssq":
        recent_20 = df.head(20)
        all_reds = pd.concat([recent_20['r1'], recent_20['r2'], recent_20['r3'], 
                             recent_20['r4'], recent_20['r5'], recent_20['r6']])
        red_counts = all_reds.value_counts()
        blue_counts = recent_20['blue'].value_counts()
        
        latest = recent_20.iloc[0]
        
        prompt = f"""请分析以下双色球历史数据趋势，给出专业解读：

【最近20期统计】
- 最热红球TOP10: {red_counts.head(10).to_dict()}
- 最冷红球BOTTOM5: {red_counts.tail(5).to_dict()}
- 蓝球频次TOP5: {blue_counts.head(5).to_dict()}

【最新一期开奖】
期号: {latest['code']}，日期: {latest['date']}
红球: {latest['r1']} {latest['r2']} {latest['r3']} {latest['r4']} {latest['r5']} {latest['r6']}
蓝球: {latest['blue']}

请从以下角度分析：
1. 近期热号、冷号趋势
2. 奇偶比例变化
3. 区间分布特征
4. 蓝球走势
5. 下期关注方向

用通俗易懂的语言，200-400字。"""

    elif name == "kl8":
        recent_20 = df.head(20)
        cols = [f"n{i:02d}" for i in range(1, 21)]
        all_nums = pd.concat([recent_20[col] for col in cols if col in recent_20.columns])
        num_counts = all_nums.value_counts()
        
        prompt = f"""请分析以下快乐8历史数据趋势：

【最近20期统计】
- 最热号码TOP15: {num_counts.head(15).to_dict()}
- 最冷号码BOTTOM10: {num_counts.tail(10).to_dict()}

请从以下角度分析：
1. 号码冷热分布
2. 区间（1-20, 21-40, 41-60, 61-80）出号特征
3. 连号情况
4. 下期关注方向

用通俗易懂的语言，200-400字。"""

    elif name == "dlt":
        recent_20 = df.head(20)
        all_fronts = pd.concat([recent_20['f1'], recent_20['f2'], recent_20['f3'],
                                recent_20['f4'], recent_20['f5']])
        front_counts = all_fronts.value_counts()
        back_counts = pd.concat([recent_20['b1'], recent_20['b2']]).value_counts()

        latest = recent_20.iloc[0]

        prompt = f"""请分析以下大乐透历史数据趋势，给出专业解读：

【最近20期统计】
- 最热前区TOP10: {front_counts.head(10).to_dict()}
- 最冷前区BOTTOM5: {front_counts.tail(5).to_dict()}
- 后区频次: {back_counts.to_dict()}

【最新一期开奖】
期号: {latest['code']}，日期: {latest['date']}
前区: {latest['f1']} {latest['f2']} {latest['f3']} {latest['f4']} {latest['f5']}
后区: {latest['b1']} {latest['b2']}

请从以下角度分析：
1. 近期热号、冷号趋势
2. 奇偶比例变化
3. 区间分布特征
4. 后区走势
5. 下期关注方向

用通俗易懂的语言，200-400字。"""

    elif name == "qxc":
        recent_20 = df.head(20)
        pos_cols = [f'n{i}' for i in range(1, 8)]
        all_nums = pd.concat([recent_20[c] for c in pos_cols])
        num_counts = all_nums.value_counts()
        latest = recent_20.iloc[0]

        prompt = f"""请分析以下七星彩历史数据趋势：

【最近20期统计】
- 最热号码TOP10: {num_counts.head(10).to_dict()}
- 最冷号码BOTTOM5: {num_counts.tail(5).to_dict()}

【最新一期开奖】
期号: {latest['code']}，日期: {latest['date']}
号码: {' '.join(str(latest[c]) for c in pos_cols)}

请从以下角度分析：
1. 各位冷热号趋势
2. 奇偶、大小分布
3. 下期关注方向

用通俗易懂的语言，200-400字。"""

    else:
        # fcsd / pl3 等三位玩法
        recent_20 = df.head(20)
        n1_counts = recent_20['n1'].value_counts().to_dict()
        n2_counts = recent_20['n2'].value_counts().to_dict()
        n3_counts = recent_20['n3'].value_counts().to_dict()

        lt_label = "福彩3D" if name == "fcsd" else "排列三"

        prompt = f"""请分析以下{lt_label}历史数据趋势：

【最近20期统计】
- 百位频次: {n1_counts}
- 十位频次: {n2_counts}
- 个位频次: {n3_counts}

请从以下角度分析：
1. 各位冷热号趋势
2. 和值走势
3. 跨度变化
4. 下期关注方向

用通俗易懂的语言，200-400字。"""

    messages = [
        {"role": "system", "content": "你是专业的彩票数据分析师，请用通俗易懂的语言分析趋势。"},
        {"role": "user", "content": prompt}
    ]
    
    result = _call_ai(messages, max_tokens=2000)
    
    if "error" in result:
        return f"❌ {result['error']}"
    
    return result.get("result", "暂无分析结果")


def ai_optimize_hedge(ssq_bets: int, hedge_strategy: str) -> dict:
    ssq_cost = ssq_bets * 2
    
    strategy_info = {
        "方案 A": {"name": "快乐8选一", "win_rate": 0.25, "prize": 4.6, "cost": 2},
        "方案 B": {"name": "快乐8选四", "win_rate": 0.2589, "prize": 100, "cost": 2},
        "方案 C": {"name": "福彩3D组选六", "win_rate": 1/167, "prize": 173, "cost": 2}
    }
    
    current_strategy_name = None
    current_strategy_detail = None
    for key, detail in strategy_info.items():
        if key in hedge_strategy:
            current_strategy_name = key
            current_strategy_detail = detail
            break
    
    ssq_df = _read_lottery_data("ssq")
    kl8_df = _read_lottery_data("kl8")
    fcsd_df = _read_lottery_data("fcsd")
    
    ssq_stats = ""
    if not ssq_df.empty:
        recent_10 = ssq_df.head(10)
        recent_30 = ssq_df.head(30)
        recent_50 = ssq_df.head(50)
        
        avg_blue = recent_10['blue'].mean()
        
        all_reds_30 = pd.concat([recent_30['r1'], recent_30['r2'], recent_30['r3'], 
                                recent_30['r4'], recent_30['r5'], recent_30['r6']])
        red_counts_30 = all_reds_30.value_counts()
        red_counts_50 = pd.concat([recent_50['r1'], recent_50['r2'], recent_50['r3'], 
                                   recent_50['r4'], recent_50['r5'], recent_50['r6']]).value_counts()
        
        red_cols = ['r1', 'r2', 'r3', 'r4', 'r5', 'r6']
        missing_periods = _calculate_missing_periods(all_reds_30, list(range(1, 34)))
        sorted_missing = sorted(missing_periods.items(), key=lambda x: x[1], reverse=True)
        coldest_reds = {k: v for k, v in sorted_missing[:5]}
        hottest_reds = {k: v for k, v in sorted_missing[-5:]}
        
        consecutive_prob = _calculate_consecutive_prob(recent_30, red_cols)
        sum_dist = _calculate_sum_distribution(recent_30, red_cols)
        parity_trend = _calculate_parity_trend(recent_30, red_cols)
        zone_dist = _calculate_zone_distribution(recent_30, red_cols, {
            "一区(01-11)": (1, 11),
            "二区(12-22)": (12, 22),
            "三区(23-33)": (23, 33)
        })
        
        ssq_stats = f"""
- 双色球历史数据：{len(ssq_df)} 期
- 最近10期蓝球均值：{avg_blue:.1f}
- 最近30期热号TOP5（红球）：{red_counts_30.head(5).to_dict()}
- 最近50期热号TOP5（红球）：{red_counts_50.head(5).to_dict()}
- 最冷红球TOP5（遗漏期数）：{coldest_reds}
- 最热红球TOP5（遗漏期数）：{hottest_reds}
- 连号概率：{consecutive_prob}%
- 和值统计：最小={sum_dist['min']}, 最大={sum_dist['max']}, 平均={sum_dist['avg']}, 众数={sum_dist['mode']}
- 奇偶比例趋势（最近10期）：{', '.join(parity_trend)}
- 区间分布：一区{zone_dist['一区(01-11)']}%, 二区{zone_dist['二区(12-22)']}%, 三区{zone_dist['三区(23-33)']}%"""
    
    kl8_stats = ""
    if not kl8_df.empty:
        recent_30 = kl8_df.head(30)
        cols = [f"n{i:02d}" for i in range(1, 21)]
        all_nums_30 = pd.concat([recent_30[col] for col in cols if col in recent_30.columns])
        num_counts_30 = all_nums_30.value_counts()
        kl8_stats = f"""
- 快乐8历史数据：{len(kl8_df)} 期
- 最近30期热号TOP10：{num_counts_30.head(10).to_dict()}"""
    
    fcsd_stats = ""
    if not fcsd_df.empty:
        recent_30 = fcsd_df.head(30)
        n1_hot = recent_30['n1'].value_counts().head(5).to_dict()
        n2_hot = recent_30['n2'].value_counts().head(5).to_dict()
        n3_hot = recent_30['n3'].value_counts().head(5).to_dict()
        fcsd_stats = f"""
- 福彩3D历史数据：{len(fcsd_df)} 期
- 百位热号TOP5：{n1_hot}
- 十位热号TOP5：{n2_hot}
- 个位热号TOP5：{n3_hot}"""
    
    current_strategy_desc = ""
    if current_strategy_detail:
        current_strategy_desc = f"""
当前选择的对冲方案：{current_strategy_name}（{current_strategy_detail['name']}）
- 中奖率：{current_strategy_detail['win_rate']*100:.2f}%
- 奖金：{current_strategy_detail['prize']}元
- 每注成本：{current_strategy_detail['cost']}元
"""
    
    prompt = f"""你是一个专业的彩票投资策略顾问，擅长运用概率论和统计学分析对冲策略。

【用户投注计划】
- 双色球主投：{ssq_bets} 注（{ssq_cost} 元）
{current_strategy_desc}

【历史数据概况】{ssq_stats}{kl8_stats}{fcsd_stats}

【请完成以下分析】

1. 📊 当前方案分析（如果用户选择了固定方案）：
   - 分析该方案的优缺点
   - 指出该方案在当前数据趋势下的不合理之处
   - 计算数学期望：E = 中奖率 × 奖金 - 成本

2. 🤖 AI 智能推荐：
   - 不局限于现有的固定方案，可以提出创新的对冲策略
   - 推荐理由必须基于历史数据趋势分析
   - 例如：快乐8选二、选三、选五，福彩3D直选、组选三，或者组合策略

3. 💰 预算分配建议：
   - 双色球和对冲的资金比例
   - 具体的投注注数

【输出格式】严格按照以下JSON格式输出：
{{
  "advice": "分析建议文本（Markdown格式，500字以内）",
  "recommended_hedge": {{
    "type": "对冲类型代码",
    "name": "对冲类型中文名称",
    "bets": 推荐注数
  }}
}}

【对冲类型代码说明】
- kl8_pick1: 快乐8选一
- kl8_pick4: 快乐8选四
- kl8_pick5: 快乐8选五
- fcsd_group3: 福彩3D组选三
- fcsd_group6: 福彩3D组选六
- fcsd_straight: 福彩3D直选

请根据数据分析给出最佳推荐。"""

    messages = [
        {"role": "system", "content": "你是专业的彩票投资策略顾问，帮助用户科学分配投注资金，控制风险。"},
        {"role": "user", "content": prompt}
    ]
    
    result = _call_ai(messages, max_tokens=2000)
    
    if "error" in result:
        return result
    
    advice = ""
    recommended_hedge = None
    
    try:
        parsed = _safe_json_parse(result.get("result", ""))
        if isinstance(parsed, dict):
            advice = parsed.get("advice", "")
            recommended_hedge = parsed.get("recommended_hedge", None)
    except Exception:
        advice = result.get("result", "")
    
    ssq_groups = []
    hedge_groups = []
    hedge_type = ""
    hedge_name = ""
    hedge_bets = 5
    
    if not ssq_df.empty:
        ssq_groups = _generate_ssq_groups(ssq_bets, ssq_df)
    
    if recommended_hedge:
        hedge_type_code = recommended_hedge.get("type", "")
        hedge_name = recommended_hedge.get("name", "")
        hedge_bets = recommended_hedge.get("bets", 5)
        
        if hedge_type_code.startswith("kl8"):
            hedge_type = "kl8"
            if not kl8_df.empty:
                if hedge_type_code == "kl8_pick1":
                    hedge_groups = _generate_kl8_pick1(hedge_bets, kl8_df)
                elif hedge_type_code == "kl8_pick4":
                    hedge_groups = _generate_kl8_pick4(hedge_bets, kl8_df)
                else:
                    hedge_groups = _generate_kl8_pick5(hedge_bets, kl8_df)
        elif hedge_type_code.startswith("fcsd"):
            hedge_type = "fcsd"
            if not fcsd_df.empty:
                if hedge_type_code == "fcsd_group3":
                    hedge_groups = _generate_fcsd_group3(hedge_bets, fcsd_df)
                elif hedge_type_code == "fcsd_straight":
                    hedge_groups = _generate_fcsd_straight(hedge_bets, fcsd_df)
                else:
                    hedge_groups = _generate_fcsd_group6(hedge_bets, fcsd_df)
    else:
        if "快乐8" in advice or (current_strategy_detail and "快乐8" in current_strategy_detail["name"]):
            hedge_type = "kl8"
            hedge_name = "快乐8"
            if not kl8_df.empty:
                if "选四" in advice or (current_strategy_detail and "选四" in current_strategy_detail["name"]):
                    hedge_groups = _generate_kl8_pick4(5, kl8_df)
                    hedge_name = "快乐8选四"
                elif "选一" in advice or (current_strategy_detail and "选一" in current_strategy_detail["name"]):
                    hedge_groups = _generate_kl8_pick1(10, kl8_df)
                    hedge_name = "快乐8选一"
                else:
                    hedge_groups = _generate_kl8_pick5(5, kl8_df)
                    hedge_name = "快乐8选五"
        elif "3D" in advice or "福彩" in advice or (current_strategy_detail and "3D" in current_strategy_detail["name"]):
            hedge_type = "fcsd"
            hedge_name = "福彩3D"
            if not fcsd_df.empty:
                if "组选三" in advice:
                    hedge_groups = _generate_fcsd_group3(5, fcsd_df)
                    hedge_name = "福彩3D组选三"
                elif "直选" in advice:
                    hedge_groups = _generate_fcsd_straight(5, fcsd_df)
                    hedge_name = "福彩3D直选"
                else:
                    hedge_groups = _generate_fcsd_group6(5, fcsd_df)
                    hedge_name = "福彩3D组选六"
    
    return {
        "advice": advice,
        "ssq_groups": ssq_groups,
        "hedge_groups": hedge_groups,
        "hedge_type": hedge_type,
        "hedge_name": hedge_name,
        "hedge_bets": hedge_bets
    }


# ========== 体彩版对冲（组合配比策略） ==========
# 与 ai_optimize_hedge 对称：核心=大乐透，对冲=排列三/七星彩。

def _generate_dlt_groups(n: int, df: pd.DataFrame) -> list:
    """大乐透：基于近30期热号生成前区5+后区2组合（dict 结构便于 UI 渲染）。"""
    if df.empty:
        return []
    recent_30 = df.head(30)
    try:
        all_fronts = pd.concat([recent_30['f1'], recent_30['f2'], recent_30['f3'],
                                recent_30['f4'], recent_30['f5']])
        front_counts = all_fronts.value_counts()
        back_counts = pd.concat([recent_30['b1'], recent_30['b2']]).value_counts()
        hot_fronts = []
        for x in front_counts.head(20).index.tolist():
            try:
                hot_fronts.append(int(x))
            except (ValueError, TypeError):
                continue
        hot_backs = []
        for x in back_counts.head(8).index.tolist():
            try:
                hot_backs.append(int(x))
            except (ValueError, TypeError):
                continue
        import random
        groups = []
        used = set()
        while len(groups) < n and len(hot_fronts) >= 5 and len(hot_backs) >= 2:
            try:
                fronts = sorted(random.sample(hot_fronts, 5))
                backs = sorted(random.sample(hot_backs, 2))
                key = tuple(fronts + backs)
                if key not in used:
                    used.add(key)
                    groups.append({"front": fronts, "back": backs})
            except Exception:
                break
        return groups
    except Exception:
        return []


def _generate_pl3_group6(n: int, df: pd.DataFrame) -> list:
    """排列三组选六：三个互不相同的数字(0-9)，顺序无关（基于历史热号并集）。"""
    if df.empty:
        return []
    recent_30 = df.head(30)
    try:
        import random
        all_nums = pd.concat([recent_30['n1'], recent_30['n2'], recent_30['n3']])
        hot = []
        for x in all_nums.value_counts().head(8).index.tolist():
            try:
                hot.append(int(x))
            except (ValueError, TypeError):
                continue
        if len(hot) < 3:
            hot = list(range(10))
        used = set()
        groups = []
        while len(groups) < n:
            combo = tuple(sorted(random.sample(hot, 3)))
            if combo not in used:
                used.add(combo)
                groups.append(list(combo))
            if len(used) >= min(len(hot), 10) or len(used) >= 120:
                break
        return groups
    except Exception:
        return []


def _generate_qxc_pick7(n: int, df: pd.DataFrame) -> list:
    """七星彩七位：每位0-9独立随机（历史不足时回退纯随机）。"""
    if df.empty:
        return []
    try:
        import random
        return [tuple(random.randint(0, 9) for _ in range(7)) for _ in range(n)]
    except Exception:
        return []


def ai_optimize_hedge_sports(dlt_bets: int, hedge_strategy: str) -> dict:
    """体彩版对冲优化：核心主投大乐透，对冲排列三/七星彩。

    Args:
        dlt_bets: 大乐透主投注数。
        hedge_strategy: 当前选择的方案文案（含"方案 A"/"方案 B"）。
    Returns:
        含 advice / dlt_groups / hedge_groups / hedge_type / hedge_name / hedge_bets 的字典；
        AI 调用失败时含 "error"。
    """
    dlt_cost = dlt_bets * 2

    strategy_info = {
        "方案 A": {"name": "排列三组选六", "win_rate": 1/167, "prize": 173, "cost": 2},
        "方案 B": {"name": "七星彩七位直选", "win_rate": 1e-7, "prize": 0, "cost": 2}
    }

    current_strategy_name = None
    current_strategy_detail = None
    for key, detail in strategy_info.items():
        if key in hedge_strategy:
            current_strategy_name = key
            current_strategy_detail = detail
            break

    dlt_df = _read_lottery_data("dlt")
    pl3_df = _read_lottery_data("pl3")
    qxc_df = _read_lottery_data("qxc")

    dlt_stats = ""
    if not dlt_df.empty:
        recent_10 = dlt_df.head(10)
        recent_30 = dlt_df.head(30)
        recent_50 = dlt_df.head(50)
        front_cols = ['f1', 'f2', 'f3', 'f4', 'f5']
        all_fronts_30 = pd.concat([recent_30[c] for c in front_cols])
        front_counts_30 = all_fronts_30.value_counts()
        front_counts_50 = pd.concat([recent_50[c] for c in front_cols]).value_counts()
        missing_periods = _calculate_missing_periods(all_fronts_30, list(range(1, 36)))
        sorted_missing = sorted(missing_periods.items(), key=lambda x: x[1], reverse=True)
        coldest_fronts = {k: v for k, v in sorted_missing[:5]}
        hottest_fronts = {k: v for k, v in sorted_missing[-5:]}
        consecutive_prob = _calculate_consecutive_prob(recent_30, front_cols)
        sum_dist = _calculate_sum_distribution(recent_30, front_cols)
        parity_trend = _calculate_parity_trend(recent_30, front_cols)
        zone_dist = _calculate_zone_distribution(recent_30, front_cols, {
            "一区(01-12)": (1, 12),
            "二区(13-24)": (13, 24),
            "三区(25-35)": (25, 35)
        })
        dlt_stats = f"""
- 大乐透历史数据：{len(dlt_df)} 期
- 最近10期后区均值：{recent_10['b1'].mean():.1f} / {recent_10['b2'].mean():.1f}
- 最近30期前区热号TOP5：{front_counts_30.head(5).to_dict()}
- 最近50期前区热号TOP5：{front_counts_50.head(5).to_dict()}
- 最冷前区TOP5（遗漏期数）：{coldest_fronts}
- 最热前区TOP5（遗漏期数）：{hottest_fronts}
- 连号概率：{consecutive_prob}%
- 和值统计：最小={sum_dist['min']}, 最大={sum_dist['max']}, 平均={sum_dist['avg']}, 众数={sum_dist['mode']}
- 奇偶比例趋势（最近10期）：{', '.join(parity_trend)}
- 区间分布：一区{zone_dist['一区(01-12)']}%, 二区{zone_dist['二区(13-24)']}%, 三区{zone_dist['三区(25-35)']}%"""

    pl3_stats = ""
    if not pl3_df.empty:
        recent_30 = pl3_df.head(30)
        n1_hot = recent_30['n1'].value_counts().head(5).to_dict()
        n2_hot = recent_30['n2'].value_counts().head(5).to_dict()
        n3_hot = recent_30['n3'].value_counts().head(5).to_dict()
        pl3_stats = f"""
- 排列三历史数据：{len(pl3_df)} 期
- 百位热号TOP5：{n1_hot}
- 十位热号TOP5：{n2_hot}
- 个位热号TOP5：{n3_hot}"""

    qxc_stats = ""
    if not qxc_df.empty:
        recent_30 = qxc_df.head(30)
        qxc_cols = [f"n{i}" for i in range(1, 8)]
        qxc_hot = {}
        for c in qxc_cols:
            if c in recent_30.columns:
                qxc_hot[c] = recent_30[c].value_counts().head(3).to_dict()
        qxc_stats = f"""
- 七星彩历史数据：{len(qxc_df)} 期
- 各位热号TOP3：{qxc_hot}"""

    current_strategy_desc = ""
    if current_strategy_detail:
        prize_desc = f"{current_strategy_detail['prize']}" if current_strategy_detail['prize'] else "浮动(高奖级)"
        current_strategy_desc = f"""
当前选择的对冲方案：{current_strategy_name}（{current_strategy_detail['name']}）
- 中奖率：{current_strategy_detail['win_rate']*100:.4f}%
- 奖金：{prize_desc}元
- 每注成本：{current_strategy_detail['cost']}元
"""

    prompt = f"""你是一个专业的体育彩票投资策略顾问，擅长运用概率论和统计学分析对冲策略。

【用户投注计划】
- 大乐透主投：{dlt_bets} 注（{dlt_cost} 元）
{current_strategy_desc}

【历史数据概况】{dlt_stats}{pl3_stats}{qxc_stats}

【请完成以下分析】

1. 📊 当前方案分析（如果用户选择了固定方案）：
   - 分析该方案的优缺点
   - 指出该方案在当前数据趋势下的不合理之处
   - 计算数学期望：E = 中奖率 × 奖金 - 成本

2. 🤖 AI 智能推荐：
   - 不局限于现有的固定方案，可以提出创新的对冲策略
   - 推荐理由必须基于历史数据趋势分析
   - 例如：排列三直选、排列三组选三、七星彩组选，或者组合策略

3. 💰 预算分配建议：
   - 大乐透和对冲的资金比例
   - 具体的投注注数

【输出格式】严格按照以下JSON格式输出：
{{
  "advice": "分析建议文本（Markdown格式，500字以内）",
  "recommended_hedge": {{
    "type": "对冲类型代码",
    "name": "对冲类型中文名称",
    "bets": 推荐注数
  }}
}}

【对冲类型代码说明】
- pl3_group6: 排列三组选六
- qxc_pick7: 七星彩七位直选

请根据数据分析给出最佳推荐。请确保只输出JSON，不要解释文字。"""

    messages = [
        {"role": "system", "content": "你是专业的体育彩票投资策略顾问，帮助用户科学分配投注资金，控制风险。"},
        {"role": "user", "content": prompt}
    ]

    result = _call_ai(messages, max_tokens=2000)
    if "error" in result:
        return result

    advice = ""
    recommended_hedge = None
    try:
        parsed = _safe_json_parse(result.get("result", ""))
        if isinstance(parsed, dict):
            advice = parsed.get("advice", "")
            recommended_hedge = parsed.get("recommended_hedge", None)
    except Exception:
        advice = result.get("result", "")

    dlt_groups = []
    hedge_groups = []
    hedge_type = ""
    hedge_name = ""
    hedge_bets = 5

    if not dlt_df.empty:
        dlt_groups = _generate_dlt_groups(dlt_bets, dlt_df)

    if recommended_hedge:
        hedge_type_code = recommended_hedge.get("type", "")
        hedge_name = recommended_hedge.get("name", "")
        hedge_bets = recommended_hedge.get("bets", 5)
        if hedge_type_code.startswith("pl3"):
            hedge_type = "pl3"
            if not pl3_df.empty:
                hedge_groups = _generate_pl3_group6(hedge_bets, pl3_df)
        elif hedge_type_code.startswith("qxc"):
            hedge_type = "qxc"
            if not qxc_df.empty:
                hedge_groups = _generate_qxc_pick7(hedge_bets, qxc_df)
    else:
        if "排列三" in advice or (current_strategy_detail and "排列三" in current_strategy_detail["name"]):
            hedge_type = "pl3"
            hedge_name = "排列三组选六"
            if not pl3_df.empty:
                hedge_groups = _generate_pl3_group6(5, pl3_df)
        elif "七星彩" in advice or (current_strategy_detail and "七星彩" in current_strategy_detail["name"]):
            hedge_type = "qxc"
            hedge_name = "七星彩七位直选"
            if not qxc_df.empty:
                hedge_groups = _generate_qxc_pick7(5, qxc_df)

    return {
        "advice": advice,
        "dlt_groups": dlt_groups,
        "hedge_groups": hedge_groups,
        "hedge_type": hedge_type,
        "hedge_name": hedge_name,
        "hedge_bets": hedge_bets
    }


def _generate_ssq_groups(n: int, df: pd.DataFrame) -> list:
    if df.empty:
        return []
    
    recent_30 = df.head(30)
    
    try:
        all_reds = pd.concat([recent_30['r1'], recent_30['r2'], recent_30['r3'],
                              recent_30['r4'], recent_30['r5'], recent_30['r6']])
        red_counts = all_reds.value_counts()
        blue_counts = recent_30['blue'].value_counts()
        
        hot_reds = []
        for x in red_counts.head(15).index.tolist():
            try:
                hot_reds.append(int(x))
            except (ValueError, TypeError):
                continue
        
        hot_blues = []
        for x in blue_counts.head(8).index.tolist():
            try:
                hot_blues.append(int(x))
            except (ValueError, TypeError):
                continue
        
        import random
        groups = []
        used_combinations = set()
        
        while len(groups) < n and len(hot_reds) >= 6 and len(hot_blues) > 0:
            try:
                reds = sorted(random.sample(hot_reds, 6))
                blue = random.choice(hot_blues)
                
                combo_key = tuple(reds + [blue])
                if combo_key not in used_combinations:
                    used_combinations.add(combo_key)
                    groups.append({"red": reds, "blue": blue})
            except Exception:
                break
        
        return groups
    except Exception:
        return []


def _generate_kl8_pick1(n: int, df: pd.DataFrame) -> list:
    if df.empty:
        return []
    
    recent_30 = df.head(30)
    cols = [f"n{i:02d}" for i in range(1, 21)]
    
    try:
        all_nums = pd.concat([recent_30[col] for col in cols if col in recent_30.columns])
        num_counts = all_nums.value_counts()
        
        hot_nums = []
        for x in num_counts.head(20).index.tolist():
            try:
                hot_nums.append(int(x))
            except (ValueError, TypeError):
                continue
        
        import random
        return [[random.choice(hot_nums)] for _ in range(n)]
    except Exception:
        return []


def _generate_kl8_pick4(n: int, df: pd.DataFrame) -> list:
    if df.empty:
        return []
    
    recent_30 = df.head(30)
    cols = [f"n{i:02d}" for i in range(1, 21)]
    
    try:
        all_nums = pd.concat([recent_30[col] for col in cols if col in recent_30.columns])
        num_counts = all_nums.value_counts()
        
        hot_nums = []
        for x in num_counts.head(30).index.tolist():
            try:
                hot_nums.append(int(x))
            except (ValueError, TypeError):
                continue
        
        import random
        groups = []
        used_combinations = set()
        
        while len(groups) < n and len(hot_nums) >= 4:
            try:
                nums = sorted(random.sample(hot_nums, 4))
                combo_key = tuple(nums)
                if combo_key not in used_combinations:
                    used_combinations.add(combo_key)
                    groups.append(nums)
            except Exception:
                break
        
        return groups
    except Exception:
        return []


def _generate_kl8_pick5(n: int, df: pd.DataFrame) -> list:
    if df.empty:
        return []
    
    recent_30 = df.head(30)
    cols = [f"n{i:02d}" for i in range(1, 21)]
    
    try:
        all_nums = pd.concat([recent_30[col] for col in cols if col in recent_30.columns])
        num_counts = all_nums.value_counts()
        
        hot_nums = []
        for x in num_counts.head(35).index.tolist():
            try:
                hot_nums.append(int(x))
            except (ValueError, TypeError):
                continue
        
        import random
        groups = []
        used_combinations = set()
        
        while len(groups) < n and len(hot_nums) >= 5:
            try:
                nums = sorted(random.sample(hot_nums, 5))
                combo_key = tuple(nums)
                if combo_key not in used_combinations:
                    used_combinations.add(combo_key)
                    groups.append(nums)
            except Exception:
                break
        
        return groups
    except Exception:
        return []


def _generate_fcsd_group6(n: int, df: pd.DataFrame) -> list:
    if df.empty:
        return []
    
    recent_30 = df.head(30)
    
    try:
        n1_hot = []
        for x in recent_30['n1'].value_counts().head(5).index.tolist():
            try:
                n1_hot.append(int(x))
            except (ValueError, TypeError):
                continue
        
        n2_hot = []
        for x in recent_30['n2'].value_counts().head(5).index.tolist():
            try:
                n2_hot.append(int(x))
            except (ValueError, TypeError):
                continue
        
        n3_hot = []
        for x in recent_30['n3'].value_counts().head(5).index.tolist():
            try:
                n3_hot.append(int(x))
            except (ValueError, TypeError):
                continue
        
        import random
        groups = []
        used_combinations = set()
        
        while len(groups) < n and n1_hot and n2_hot and n3_hot:
            try:
                nums = sorted([random.choice(n1_hot), random.choice(n2_hot), random.choice(n3_hot)])
                if len(set(nums)) == 3:
                    combo_key = tuple(nums)
                    if combo_key not in used_combinations:
                        used_combinations.add(combo_key)
                        groups.append(nums)
            except Exception:
                break
        
        return groups
    except Exception:
        return []


def _generate_fcsd_group3(n: int, df: pd.DataFrame) -> list:
    if df.empty:
        return []
    
    recent_30 = df.head(30)
    
    try:
        n1_hot = []
        for x in recent_30['n1'].value_counts().head(5).index.tolist():
            try:
                n1_hot.append(int(x))
            except (ValueError, TypeError):
                continue
        
        n2_hot = []
        for x in recent_30['n2'].value_counts().head(5).index.tolist():
            try:
                n2_hot.append(int(x))
            except (ValueError, TypeError):
                continue
        
        n3_hot = []
        for x in recent_30['n3'].value_counts().head(5).index.tolist():
            try:
                n3_hot.append(int(x))
            except (ValueError, TypeError):
                continue
        
        import random
        groups = []
        used_combinations = set()
        
        while len(groups) < n and n1_hot and n2_hot and n3_hot:
            try:
                num1 = random.choice(n1_hot)
                num2 = random.choice(n2_hot)
                num3 = random.choice(n3_hot)
                
                counts = {num1: 0, num2: 0, num3: 0}
                counts[num1] += 1
                counts[num2] += 1
                counts[num3] += 1
                
                if any(v == 2 for v in counts.values()) and all(v <= 2 for v in counts.values()):
                    nums = sorted([num1, num2, num3])
                    combo_key = tuple(nums)
                    if combo_key not in used_combinations:
                        used_combinations.add(combo_key)
                        groups.append(nums)
            except Exception:
                break
        
        return groups
    except Exception:
        return []


def _generate_fcsd_straight(n: int, df: pd.DataFrame) -> list:
    if df.empty:
        return []
    
    recent_30 = df.head(30)
    
    try:
        n1_hot = []
        for x in recent_30['n1'].value_counts().head(5).index.tolist():
            try:
                n1_hot.append(int(x))
            except (ValueError, TypeError):
                continue
        
        n2_hot = []
        for x in recent_30['n2'].value_counts().head(5).index.tolist():
            try:
                n2_hot.append(int(x))
            except (ValueError, TypeError):
                continue
        
        n3_hot = []
        for x in recent_30['n3'].value_counts().head(5).index.tolist():
            try:
                n3_hot.append(int(x))
            except (ValueError, TypeError):
                continue
        
        import random
        groups = []
        used_combinations = set()
        
        while len(groups) < n and n1_hot and n2_hot and n3_hot:
            try:
                nums = [random.choice(n1_hot), random.choice(n2_hot), random.choice(n3_hot)]
                combo_key = tuple(nums)
                if combo_key not in used_combinations:
                    used_combinations.add(combo_key)
                    groups.append(nums)
            except Exception:
                break
        
        return groups
    except Exception:
        return []


def ai_compare_last_draw(name: str) -> dict:
    df = _read_lottery_data(name)
    
    if df.empty:
        return {"error": "暂无历史数据"}
    
    latest = df.iloc[0]
    latest_date = latest.get('date', '')
    latest_code = latest.get('code', '')
    
    if name == "ssq":
        actual_reds = [latest['r1'], latest['r2'], latest['r3'], 
                       latest['r4'], latest['r5'], latest['r6']]
        actual_blue = latest['blue']
        
        ai_result = ai_predict_ssq(5)
        if "error" in ai_result:
            return {"error": ai_result["error"]}
        
        ai_recommendations = ai_result.get("recommendations", [])
        
        best_match = {
            "group": 0,
            "red_matches": 0,
            "blue_match": False,
            "nums": {}
        }
        
        for rec in ai_recommendations:
            nums = rec.get("numbers", {})
            ai_reds = nums.get("red", [])
            ai_blue = nums.get("blue", 0)
            
            red_match_count = len(set(ai_reds) & set(actual_reds))
            blue_match = ai_blue == actual_blue
            
            if red_match_count > best_match["red_matches"] or \
               (red_match_count == best_match["red_matches"] and blue_match):
                best_match = {
                    "group": rec.get("group", 0),
                    "red_matches": red_match_count,
                    "blue_match": blue_match,
                    "nums": nums
                }
        
        return {
            "latest": {
                "code": latest_code,
                "date": latest_date,
                "reds": actual_reds,
                "blue": actual_blue
            },
            "ai_best": best_match,
            "ai_analysis": ai_result.get("analysis", "")
        }
    
    elif name == "kl8":
        cols = [f"n{i:02d}" for i in range(1, 21)]
        actual_nums = [latest[col] for col in cols if col in latest]
        
        ai_result = ai_predict_kl8(5)
        if "error" in ai_result:
            return {"error": ai_result["error"]}
        
        ai_recommendations = ai_result.get("recommendations", [])
        
        best_match = {
            "group": 0,
            "matches": 0,
            "nums": []
        }
        
        for rec in ai_recommendations:
            ai_nums = rec.get("numbers", [])
            match_count = len(set(ai_nums) & set(actual_nums))
            
            if match_count > best_match["matches"]:
                best_match = {
                    "group": rec.get("group", 0),
                    "matches": match_count,
                    "nums": ai_nums
                }
        
        return {
            "latest": {
                "code": latest_code,
                "date": latest_date,
                "nums": actual_nums
            },
            "ai_best": best_match,
            "ai_analysis": ai_result.get("analysis", "")
        }
    
    else:
        actual_nums = [latest['n1'], latest['n2'], latest['n3']]
        
        ai_result = ai_predict_fcsd(5)
        if "error" in ai_result:
            return {"error": ai_result["error"]}
        
        ai_recommendations = ai_result.get("recommendations", [])
        
        best_match = {
            "group": 0,
            "matches": 0,
            "nums": []
        }
        
        for rec in ai_recommendations:
            ai_nums = rec.get("numbers", [])
            match_count = 0
            for i in range(min(len(ai_nums), 3)):
                if ai_nums[i] == actual_nums[i]:
                    match_count += 1
            
            if match_count > best_match["matches"]:
                best_match = {
                    "group": rec.get("group", 0),
                    "matches": match_count,
                    "nums": ai_nums
                }
        
        return {
            "latest": {
                "code": latest_code,
                "date": latest_date,
                "nums": actual_nums
            },
            "ai_best": best_match,
            "ai_analysis": ai_result.get("analysis", "")
        }


def _get_latest_code_from_csv(lottery_type: str) -> Optional[str]:
    """从数据库读取最新开奖期号。"""
    return _db_get_latest_code(lottery_type)


def save_prediction_record(lottery_type: str, code: str, predictions: list, play_type: str = None):
    """保存预测记录（委托数据库模块）。"""
    _db_save_prediction(lottery_type, code, predictions, play_type)


def get_prediction_records(lottery_type: str = None) -> list:
    """读取预测记录（委托数据库模块）。"""
    return _db_get_prediction_records(lottery_type)


def get_prediction_for_code(lottery_type: str, code: str) -> dict:
    """获取指定期号的预测记录（委托数据库模块）。"""
    return _db_get_prediction_for_code(lottery_type, code)


def update_prediction_compare(lottery_type: str, code: str, compare_result: dict):
    """更新预测对比结果（委托数据库模块）。"""
    _db_update_prediction_compare(lottery_type, code, compare_result)


def analyze_saved_predictions(lottery_type: str, target_code: str = None, force_refresh: bool = False) -> dict:
    df = _read_lottery_data(lottery_type)
    if df.empty:
        return {'error': '暂无历史数据'}
    
    records = get_prediction_records(lottery_type)
    if not records:
        return {'error': '暂无预测记录，请先进行AI预测'}
    
    latest = df.iloc[0]
    latest_code = str(latest.get('code', ''))
    latest_date = latest.get('date', '')
    
    available_codes = [str(r['code']) for r in records]
    
    if target_code:
        target_code = str(target_code)
        record = get_prediction_for_code(lottery_type, target_code)
        if not record:
            return {
                'latest': {'code': latest_code, 'date': latest_date},
                'error': f'未找到第 {target_code} 期的预测记录',
                'available_codes': available_codes
            }
    else:
        record = get_prediction_for_code(lottery_type, latest_code)
        if not record:
            next_code = str(int(latest_code) + 1)
            record = get_prediction_for_code(lottery_type, next_code)
            
            if not record:
                return {
                    'latest': {'code': latest_code, 'date': latest_date},
                    'error': '暂无该期的预测记录',
                    'available_codes': available_codes
                }
    
    record_code = str(record['code'])
    
    # 在开奖数据中查找对应期号
    draw = None
    if 'code' in df.columns:
        draw_match = df[df['code'].astype(str) == record_code]
        if not draw_match.empty:
            draw = draw_match.iloc[0]
    
    if draw is None:
        return {
            'latest': {'code': latest_code, 'date': latest_date},
            'error': f'第 {record_code} 期尚未开奖（最新开奖期号：{latest_code}），请等待开奖后再对比',
            'available_codes': available_codes
        }
    
    if not force_refresh and record.get('compared') and record.get('compare_result'):
        cached = record['compare_result']
        # 命中缓存时确保返回的 actual 与当前选中的开奖期一致
        if str(cached.get('latest', {}).get('code', '')) == record_code:
            return cached
    
    draw_date = draw.get('date', '')
    actual_nums = []
    if lottery_type == 'ssq':
        actual_nums = {
            'reds': [draw['r1'], draw['r2'], draw['r3'],
                     draw['r4'], draw['r5'], draw['r6']],
            'blue': draw['blue']
        }
    elif lottery_type == 'kl8':
        cols = [f'n{i:02d}' for i in range(1, 21)]
        actual_nums = {'nums': [draw[col] for col in cols if col in draw]}
    elif lottery_type == 'dlt':
        actual_nums = {
            'fronts': [draw['f1'], draw['f2'], draw['f3'],
                       draw['f4'], draw['f5']],
            'backs': [draw['b1'], draw['b2']]
        }
    elif lottery_type == 'qxc':
        actual_nums = {'nums': [draw[f'n{i}'] for i in range(1, 8)]}
    else:
        # fcsd / pl3 共用 n1,n2,n3
        actual_nums = {'nums': [draw['n1'], draw['n2'], draw['n3']]}
    
    predictions = record.get('predictions', [])
    best_match = None
    
    if lottery_type == 'ssq':
        best_match = {
            'group': 0,
            'red_matches': 0,
            'blue_match': False,
            'nums': {}
        }
        for i, pred in enumerate(predictions, 1):
            reds = pred.get('red', [])
            blue = pred.get('blue', 0)
            red_match_count = len(set(reds) & set(actual_nums['reds']))
            blue_match = blue == actual_nums['blue']
            if red_match_count > best_match['red_matches'] or \
               (red_match_count == best_match['red_matches'] and blue_match):
                best_match = {
                    'group': i,
                    'red_matches': red_match_count,
                    'blue_match': blue_match,
                    'nums': pred
                }
    
    elif lottery_type == 'dlt':
        best_match = {
            'group': 0,
            'front_matches': 0,
            'back_matches': 0,
            'nums': []
        }
        for i, pred in enumerate(predictions, 1):
            nums = pred.get('nums', [])
            # 大乐透保存格式: nums = [前区5个 + 后区2个]
            fronts = nums[:5] if len(nums) >= 5 else []
            backs = nums[5:] if len(nums) > 5 else []
            front_match = len(set(fronts) & set(actual_nums['fronts']))
            back_match = len(set(backs) & set(actual_nums['backs']))
            if (front_match + back_match) > (best_match['front_matches'] + best_match['back_matches']):
                best_match = {
                    'group': i,
                    'front_matches': front_match,
                    'back_matches': back_match,
                    'nums': nums
                }
    
    elif lottery_type == 'kl8':
        best_match = {
            'group': 0,
            'matches': 0,
            'nums': []
        }
        for i, pred in enumerate(predictions, 1):
            nums = pred.get('nums', [])
            match_count = len(set(nums) & set(actual_nums['nums']))
            if match_count > best_match['matches']:
                best_match = {
                    'group': i,
                    'matches': match_count,
                    'nums': nums
                }
    
    elif lottery_type == 'qxc':
        best_match = {
            'group': 0,
            'matches': 0,
            'nums': []
        }
        for i, pred in enumerate(predictions, 1):
            nums = pred.get('nums', [])
            match_count = 0
            for j in range(min(len(nums), 7)):
                if nums[j] == actual_nums['nums'][j]:
                    match_count += 1
            if match_count > best_match['matches']:
                best_match = {
                    'group': i,
                    'matches': match_count,
                    'nums': nums
                }
    
    else:
        # fcsd / pl3
        best_match = {
            'group': 0,
            'matches': 0,
            'nums': []
        }
        for i, pred in enumerate(predictions, 1):
            nums = pred.get('nums', [])
            match_count = 0
            for j in range(min(len(nums), 3)):
                if nums[j] == actual_nums['nums'][j]:
                    match_count += 1
            if match_count > best_match['matches']:
                best_match = {
                    'group': i,
                    'matches': match_count,
                    'nums': nums
                }
    
    compare_result = {
        'latest': {
            'code': record_code,
            'date': draw_date,
            **actual_nums
        },
        'ai_best': best_match,
        'predict_time': record.get('predict_time', ''),
        'prize_result': calculate_total_prize(lottery_type, predictions, actual_nums)
    }
    
    update_prediction_compare(lottery_type, record_code, compare_result)
    
    return compare_result


def refresh_all_prediction_compares(lottery_type: str) -> dict:
    """强制重新对比指定彩种的所有已保存预测记录（按最新中奖规则刷新奖金）。

    跳过尚未开奖的期号；已开奖但对比失败的期号会记录到 errors。

    Returns:
        {"success": int, "skipped": int, "errors": int, "details": list}
    """
    logger.info(f"[refresh_all] 开始强制刷新全量对比: lottery_type={lottery_type}")
    records = get_prediction_records(lottery_type)
    if not records:
        logger.info("[refresh_all] 无预测记录，跳过")
        return {"success": 0, "skipped": 0, "errors": 0, "details": ["无预测记录"]}

    df = _read_lottery_data(lottery_type)
    if df.empty:
        logger.warning("[refresh_all] 无历史开奖数据")
        return {"success": 0, "skipped": 0, "errors": 1, "details": ["无历史开奖数据"]}

    available_codes = set(df['code'].astype(str).tolist())

    success = 0
    skipped = 0
    errors = 0
    details = []

    for record in records:
        code = str(record.get('code', ''))
        if not code or code not in available_codes:
            skipped += 1
            details.append(f"第 {code or '?'} 期：尚未开奖，跳过")
            continue
        try:
            analyze_saved_predictions(lottery_type, target_code=code, force_refresh=True)
            success += 1
            details.append(f"第 {code} 期：已刷新")
            logger.debug(f"[refresh_all] 第 {code} 期刷新成功")
        except Exception as e:
            errors += 1
            details.append(f"第 {code} 期：失败 {e}")
            logger.error(f"[refresh_all] 第 {code} 期刷新失败: {e}", exc_info=True)

    logger.info(f"[refresh_all] 完成: success={success}, skipped={skipped}, errors={errors}")
    return {"success": success, "skipped": skipped, "errors": errors, "details": details}


SSQ_PRIZES = {
    7: {"name": "一等奖", "prize": 5000000, "desc": "6+1（浮动奖金，参考值）"},
    6: {"name": "二等奖", "prize": 500000, "desc": "6+0（浮动奖金，参考值）"},
    5: {"name": "三等奖", "prize": 3000, "desc": "5+1"},
    4: {"name": "四等奖", "prize": 200, "desc": "5+0/4+1"},
    3: {"name": "五等奖", "prize": 10, "desc": "4+0/3+1"},
    2: {"name": "六等奖", "prize": 5, "desc": "2+1/1+1/0+1"}
}


# 快乐8 官方单注固定奖金（每注2元）。key = 精确命中数；key=0 表示「全不中」奖（仅命中0个才中）。
# 数据来源：中国福利彩票官网开奖公告（选十中九=8000、全不中=2 等）。
KL8_PRIZES = {
    "pick1":  {1: {"name": "选一中一",   "prize": 4.5}},
    "pick2":  {2: {"name": "选二中二",   "prize": 19}, 0: {"name": "选二全不中", "prize": 2}},
    "pick3":  {3: {"name": "选三中三",   "prize": 52}, 2: {"name": "选三中二", "prize": 3}, 0: {"name": "选三全不中", "prize": 2}},
    "pick4":  {4: {"name": "选四中四",   "prize": 93}, 3: {"name": "选四中三", "prize": 5}, 2: {"name": "选四中二", "prize": 3}, 0: {"name": "选四全不中", "prize": 2}},
    "pick5":  {5: {"name": "选五中五",   "prize": 1000}, 4: {"name": "选五中四", "prize": 20}, 3: {"name": "选五中三", "prize": 3}, 0: {"name": "选五全不中", "prize": 2}},
    "pick6":  {6: {"name": "选六中六",   "prize": 2880}, 5: {"name": "选六中五", "prize": 30}, 4: {"name": "选六中四", "prize": 10}, 3: {"name": "选六中三", "prize": 3}, 0: {"name": "选六全不中", "prize": 2}},
    "pick7":  {7: {"name": "选七中七",   "prize": 8500}, 6: {"name": "选七中六", "prize": 300}, 5: {"name": "选七中五", "prize": 30}, 4: {"name": "选七中四", "prize": 4}, 0: {"name": "选七全不中", "prize": 2}},
    "pick8":  {8: {"name": "选八中八",   "prize": 50000}, 7: {"name": "选八中七", "prize": 800}, 6: {"name": "选八中六", "prize": 80}, 5: {"name": "选八中五", "prize": 10}, 4: {"name": "选八中四", "prize": 3}, 0: {"name": "选八全不中", "prize": 2}},
    "pick9":  {9: {"name": "选九中九",   "prize": 250000}, 8: {"name": "选九中八", "prize": 2000}, 7: {"name": "选九中七", "prize": 225}, 6: {"name": "选九中六", "prize": 22}, 5: {"name": "选九中五", "prize": 5}, 4: {"name": "选九中四", "prize": 3}, 0: {"name": "选九全不中", "prize": 2}},
    "pick10": {10: {"name": "选十中十",  "prize": 5000000}, 9: {"name": "选十中九", "prize": 8000}, 8: {"name": "选十中八", "prize": 720}, 7: {"name": "选十中七", "prize": 80}, 6: {"name": "选十中六", "prize": 5}, 5: {"name": "选十中五", "prize": 3}, 0: {"name": "选十全不中", "prize": 2}},
}


FCSD_PRIZES = {
    "straight": {3: {"name": "直选", "prize": 1040}},
    "group3": {2: {"name": "组三", "prize": 346}},
    "group6": {1: {"name": "组六", "prize": 173}}
}


def calculate_ssq_prize(reds_pred, blue_pred, reds_actual, blue_actual):
    red_match = len(set(reds_pred) & set(reds_actual))
    blue_match = 1 if blue_pred == blue_actual else 0
    
    if red_match == 6 and blue_match == 1:
        return SSQ_PRIZES[7]
    elif red_match == 6 and blue_match == 0:
        return SSQ_PRIZES[6]
    elif red_match == 5 and blue_match == 1:
        return SSQ_PRIZES[5]
    elif (red_match == 5 and blue_match == 0) or (red_match == 4 and blue_match == 1):
        return SSQ_PRIZES[4]
    elif (red_match == 4 and blue_match == 0) or (red_match == 3 and blue_match == 1):
        return SSQ_PRIZES[3]
    elif blue_match == 1:
        return SSQ_PRIZES[2]
    else:
        return {"name": "未中奖", "prize": 0, "desc": ""}


def calculate_kl8_prize(pred_nums, actual_nums, play_type="pick10"):
    match_count = len(set(pred_nums) & set(actual_nums))

    prizes = KL8_PRIZES.get(play_type, {})
    # 快乐8 各玩法按「精确命中数」定奖：命中数恰好等于某奖级 key 才中奖；
    # key=0 仅代表「全不中」（命中0个）奖，不能把「中1个」误判为中奖。
    if match_count in prizes:
        return prizes[match_count]

    return {"name": "未中奖", "prize": 0, "desc": ""}


def calculate_fcsd_prize(pred_nums, actual_nums, play_type="straight"):
    if play_type == "straight":
        if pred_nums[0] == actual_nums[0] and pred_nums[1] == actual_nums[1] and pred_nums[2] == actual_nums[2]:
            return FCSD_PRIZES["straight"][3]
    elif play_type == "group3":
        pred_sorted = sorted(pred_nums)
        actual_sorted = sorted(actual_nums)
        if pred_sorted == actual_sorted:
            return FCSD_PRIZES["group3"][2]
    elif play_type == "group6":
        pred_sorted = sorted(pred_nums)
        actual_sorted = sorted(actual_nums)
        if pred_sorted == actual_sorted:
            return FCSD_PRIZES["group6"][1]
    elif play_type is None:
        pred_sorted = sorted(pred_nums)
        actual_sorted = sorted(actual_nums)
        if pred_nums[0] == actual_nums[0] and pred_nums[1] == actual_nums[1] and pred_nums[2] == actual_nums[2]:
            return FCSD_PRIZES["straight"][3]
        elif pred_sorted == actual_sorted:
            unique_pred = len(set(pred_nums))
            if unique_pred == 2:
                return FCSD_PRIZES["group3"][2]
            elif unique_pred == 3:
                return FCSD_PRIZES["group6"][1]
    
    return {"name": "未中奖", "prize": 0, "desc": ""}


# 超级大乐透官方奖级（前区5+后区2）。数据来源：中国体彩网。
# 一等奖5+2 / 二等奖5+1（均为浮动奖，参考值）/ 三等奖5+0=10000 / 四等奖4+2=3000
# 五等奖4+1=300 / 六等奖3+2=200 / 七等奖4+0=100 / 八等奖3+1或2+2=15
# 九等奖3+0或2+1或1+2或0+2=5。注意：2+0、1+1、1+0、0+1、0+0 均不中奖。
DLT_PRIZES = {
    9: {"name": "一等奖", "prize": 10000000, "desc": "5+2（浮动奖金，参考值）"},
    8: {"name": "二等奖", "prize": 500000, "desc": "5+1（浮动奖金，参考值）"},
    7: {"name": "三等奖", "prize": 10000, "desc": "5+0"},
    6: {"name": "四等奖", "prize": 3000, "desc": "4+2"},
    5: {"name": "五等奖", "prize": 300, "desc": "4+1"},
    4: {"name": "六等奖", "prize": 200, "desc": "3+2"},
    3: {"name": "七等奖", "prize": 100, "desc": "4+0"},
    2: {"name": "八等奖", "prize": 15, "desc": "3+1/2+2"},
    1: {"name": "九等奖", "prize": 5, "desc": "3+0/2+1/1+2/0+2"},
}


def calculate_dlt_prize(fronts_pred, backs_pred, fronts_actual, backs_actual):
    front_match = len(set(fronts_pred) & set(fronts_actual))
    back_match = len(set(backs_pred) & set(backs_actual))
    # 官方大乐透中奖规则（严格按 前区命中数 + 后区命中数 组合判定）
    if front_match == 5 and back_match == 2:
        level = 9
    elif front_match == 5 and back_match == 1:
        level = 8
    elif front_match == 5 and back_match == 0:
        level = 7
    elif front_match == 4 and back_match == 2:
        level = 6
    elif front_match == 4 and back_match == 1:
        level = 5
    elif front_match == 3 and back_match == 2:
        level = 4
    elif front_match == 4 and back_match == 0:
        level = 3
    elif (front_match == 3 and back_match == 1) or (front_match == 2 and back_match == 2):
        level = 2
    elif (front_match == 3 and back_match == 0) or (front_match == 2 and back_match == 1) \
            or (front_match == 1 and back_match == 2) or (front_match == 0 and back_match == 2):
        level = 1
    else:
        return {"name": "未中奖", "prize": 0, "desc": ""}
    return DLT_PRIZES.get(level, {"name": "未中奖", "prize": 0, "desc": ""})


def calculate_qxc_prize(pred_nums, actual_nums):
    """七星彩：7 位逐位对位比较（官方规则，位置严格对应）。
    一等奖 7位全中（浮动）；二等奖 前6位全中（浮动）；三等奖 前5+后1（5+1，3000）；
    四等奖 任意5位中（500）；五等奖 任意4位中（30）；
    六等奖 任意3位中 / 前6任1+后1 / 仅后1（5元）。
    关键：三等奖要求「含后区」(5+1)，仅前6位中任意5位（不含后区）只算四等奖。
    """
    pred = list(pred_nums)
    act = list(actual_nums)
    front6_match = sum(1 for i in range(min(6, len(pred), len(act))) if pred[i] == act[i])
    last_match = 1 if (len(pred) > 6 and len(act) > 6 and pred[6] == act[6]) else 0
    total = front6_match + last_match

    if front6_match == 6 and last_match == 1:
        return {"name": "一等奖", "prize": 5000000, "desc": "7位全中（浮动奖金，参考值）"}
    elif front6_match == 6 and last_match == 0:
        return {"name": "二等奖", "prize": 0, "desc": "前6位全中（浮动奖金）"}
    elif front6_match == 5 and last_match == 1:
        return {"name": "三等奖", "prize": 3000, "desc": "前5位+后区（5+1）"}
    elif total == 5:
        return {"name": "四等奖", "prize": 500, "desc": "任意5位中"}
    elif total == 4:
        return {"name": "五等奖", "prize": 30, "desc": "任意4位中"}
    elif total == 3 or (front6_match >= 1 and last_match == 1) or last_match == 1:
        return {"name": "六等奖", "prize": 5, "desc": "任意3位中/仅后区中"}
    return {"name": "未中奖", "prize": 0, "desc": ""}


def calculate_pl3_prize(pred_nums, actual_nums, play_type="straight"):
    """排列三奖级计算（与福彩3D一致）"""
    return calculate_fcsd_prize(pred_nums, actual_nums, play_type)


def calculate_total_prize(lottery_type, predictions, actual_nums, play_type=None):
    total_prize = 0
    total_cost = len(predictions) * 2
    results = []
    
    for i, pred in enumerate(predictions, 1):
        if lottery_type == "ssq":
            reds = pred.get("red", [])
            blue = pred.get("blue", 0)
            prize_info = calculate_ssq_prize(reds, blue, actual_nums.get("reds", []), actual_nums.get("blue", 0))
        elif lottery_type == "dlt":
            nums = pred.get("nums", [])
            fronts = nums[:5] if len(nums) >= 5 else nums
            backs = nums[5:] if len(nums) > 5 else []
            prize_info = calculate_dlt_prize(fronts, backs, actual_nums.get("fronts", []), actual_nums.get("backs", []))
        elif lottery_type == "kl8":
            nums = pred.get("nums", [])
            pt = play_type or "pick10"
            prize_info = calculate_kl8_prize(nums, actual_nums.get("nums", []), pt)
        elif lottery_type == "qxc":
            nums = pred.get("nums", [])
            prize_info = calculate_qxc_prize(nums, actual_nums.get("nums", []))
        elif lottery_type == "pl3":
            nums = pred.get("nums", [])
            pt = play_type if play_type else None
            prize_info = calculate_pl3_prize(nums, actual_nums.get("nums", []), pt)
        else:
            # fcsd
            nums = pred.get("nums", [])
            pt = play_type if play_type else None
            prize_info = calculate_fcsd_prize(nums, actual_nums.get("nums", []), pt)
        
        total_prize += prize_info["prize"]
        results.append({
            "group": i,
            "prediction": pred,
            "prize_info": prize_info,
            "prize": prize_info["prize"]
        })
    
    return {
        "total_cost": total_cost,
        "total_prize": total_prize,
        "profit": total_prize - total_cost,
        "profit_rate": ((total_prize - total_cost) / total_cost * 100) if total_cost > 0 else 0,
        "results": results
    }


def get_betting_report(lottery_type: str = None) -> dict:
    records = get_prediction_records(lottery_type)
    
    if not records:
        return {"error": "暂无投注记录"}
    
    total_cost = 0
    total_prize = 0
    total_profit = 0
    total_bets = 0
    win_count = 0
    detail_records = []
    
    for record in records:
        if record.get("compared") and record.get("compare_result"):
            compare_result = record["compare_result"]
            latest_data = compare_result.get("latest", {})
            predictions = record.get("predictions", [])
            lt = record.get("lottery_type", "")
            
            prize_result = compare_result.get("prize_result", {})
            result = {
                "total_cost": prize_result.get("total_cost", 0),
                "total_prize": prize_result.get("total_prize", 0),
                "profit": prize_result.get("profit", 0),
                "profit_rate": prize_result.get("profit_rate", 0)
            }
            
            detail_records.append({
                "code": latest_data.get("code", ""),
                "date": latest_data.get("date", ""),
                "lottery_type": lt,
                "predict_time": record.get("predict_time", ""),
                "cost": result["total_cost"],
                "prize": result["total_prize"],
                "profit": result["profit"],
                "profit_rate": result["profit_rate"],
                "bets": len(predictions),
                "won": result["total_prize"] > 0
            })
            
            total_cost += result["total_cost"]
            total_prize += result["total_prize"]
            total_profit += result["profit"]
            total_bets += len(predictions)
            if result["total_prize"] > 0:
                win_count += 1
    
    return {
        "total_cost": total_cost,
        "total_prize": total_prize,
        "total_profit": total_profit,
        "total_bets": total_bets,
        "win_count": win_count,
        "win_rate": (win_count / len(detail_records) * 100) if detail_records else 0,
        "avg_profit_per_bet": (total_profit / total_bets) if total_bets > 0 else 0,
        "profit_rate": (total_profit / total_cost * 100) if total_cost > 0 else 0,
        "records": detail_records
    }


def ai_review_pool(lottery_type: str,
                   candidate_pool: List[int],
                   confidence: Dict[int, float],
                   recent_stats: str = "",
                   backtest_summary: str = "") -> dict:
    """AI 候选池审阅：算法出候选号 + 统计摘要 → AI 审阅后标记优先/回避。

    Args:
        lottery_type: 彩种代码 (ssq/dlt/kl8/qxc/pl3/fcsd)
        candidate_pool: 算法产出的候选号码列表
        confidence: 号码→置信度映射
        recent_stats: 近期统计摘要文本（可选）
        backtest_summary: 回溯验证摘要文本（可选）

    Returns:
        dict: {
            "prioritized": [号码...],   # AI 标记优先
            "avoided": [号码...],       # AI 建议回避
            "reasoning": "分析文本",    # AI 推理说明
        }
        AI 不可用时返回空标记（不阻断算法流程）。
    """
    if not is_ai_configured():
        return {"prioritized": [], "avoided": [], "reasoning": "AI 未配置，跳过审阅"}

    lottery_names = {
        "ssq": "双色球", "dlt": "大乐透", "kl8": "快乐8",
        "qxc": "七星彩", "pl3": "排列三", "fcsd": "福彩3D"
    }
    name = lottery_names.get(lottery_type, lottery_type)

    # 构建候选号置信度排序
    sorted_candidates = sorted(confidence.items(), key=lambda x: x[1], reverse=True)
    pool_str = "、".join([f"{n}(置信度{c:.1%})" for n, c in sorted_candidates[:20]])

    prompt = f"""你是一位精通概率统计的彩票数据分析顾问。请审阅以下由统计算法产出的候选号码池，给出优先/回避建议。

【彩种】{name}
【算法候选池 Top20】{pool_str}
【近期统计概况】{recent_stats if recent_stats else "无"}
【回溯验证】{backtest_summary if backtest_summary else "无"}

【审阅规则】
1. 你不是在"预测中奖号码"，而是基于统计特征对算法候选池做"增信/降权"微调
2. 标记"优先"：近期遗漏回补信号强、区间分布偏冷需要轮动、置信度被低估的号码
3. 标记"回避"：近期极热已过度释放、区间过度集中、连号/同尾数过多的号码
4. 优先+回避总数不超过候选池的40%（保守微调，不颠覆算法结果）
5. 每类至少1个，至多不超过候选池30%

【输出格式】严格JSON，不要任何解释文字：
{{
  "prioritized": [号码1, 号码2, ...],
  "avoided": [号码1, 号码2, ...],
  "reasoning": "简短分析（100字以内）"
}}"""

    messages = [
        {"role": "system", "content": "你是概率统计顾问，只基于数据做审阅，不预测中奖号码。输出严格JSON。"},
        {"role": "user", "content": prompt}
    ]

    result = _call_ai(messages, max_tokens=800, temperature=0.3)
    if "error" in result:
        return {"prioritized": [], "avoided": [], "reasoning": f"AI 审阅失败: {result['error']}"}

    try:
        parsed = _safe_json_parse(result.get("result", ""))
        if isinstance(parsed, dict):
            prioritized = [int(x) for x in parsed.get("prioritized", []) if str(x).isdigit()]
            avoided = [int(x) for x in parsed.get("avoided", []) if str(x).isdigit()]
            reasoning = str(parsed.get("reasoning", ""))
            # 安全过滤：只保留候选池中存在的号码
            pool_set = set(candidate_pool)
            prioritized = [n for n in prioritized if n in pool_set]
            avoided = [n for n in avoided if n in pool_set]
            return {
                "prioritized": prioritized,
                "avoided": avoided,
                "reasoning": reasoning
            }
    except Exception as e:
        pass

    return {"prioritized": [], "avoided": [], "reasoning": "AI 返回解析失败，使用原算法结果"}
