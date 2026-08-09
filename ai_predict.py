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
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

CONFIG_DIR = Path.home() / ".lottery_ai"
CONFIG_FILE = CONFIG_DIR / "config.json"
DATA_DIR = Path(__file__).parent / "data"

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
            return {
                "api_key": str(sec.get("api_key", "")),
                "base_url": str(sec.get("base_url", DEFAULT_CONFIG["base_url"])),
                "model": str(sec.get("model", DEFAULT_CONFIG["model"]))
            }
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
    csv_path = os.path.join("data", f"{name}.csv")
    if not os.path.exists(csv_path):
        return pd.DataFrame()
    
    try:
        df = _read_csv_file(csv_path)
        return df
    except Exception:
        return pd.DataFrame()


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
        content = response.choices[0].message.content
        return {"result": content}
    except Exception as e:
        return {"error": f"AI 调用失败: {str(e)}"}


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
    """计算尾数分布"""
    tail_counts = {}
    for _, row in df.iterrows():
        for col in cols:
            if col in df.columns:
                tail = int(row[col]) % 10
                tail_counts[tail] = tail_counts.get(tail, 0) + 1
    total = sum(tail_counts.values())
    if total == 0:
        return {}
    return {str(k): round(v / total * 100, 1) for k, v in sorted(tail_counts.items())}


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
        {"role": "system", "content": "你是专业的彩票数据分析师，擅长基于历史数据预测号码。请严格按照要求的JSON格式输出，确保号码满足所有约束条件。"},
        {"role": "user", "content": prompt}
    ]
    
    result = _call_ai(messages, max_tokens=5000, temperature=temperature)
    
    if "error" in result:
        return result
    
    try:
        import re
        json_str = result["result"]
        match = re.search(r'\{[\s\S]*\}', json_str)
        if match:
            parsed = json.loads(match.group())
            return parsed
        return {"error": "AI 返回格式异常，请重试"}
    except Exception:
        return {
            "recommendations": [],
            "analysis": result.get("result", ""),
            "note": "AI 分析文字已返回，但号码解析失败，请参考分析文字手动选号"
        }


def ai_predict_kl8(n_groups: int = 5) -> dict:
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
    
    prompt = f"""你是一个专业的彩票数据分析专家。请基于以下快乐8历史多维数据，生成 {n_groups} 组"选十"推荐号码。

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
1. 每组10个号码（1-80），不重复
2. **区间约束**：四个区间各有 2-4 个号码，不能有区间完全断档
3. **大小平衡**：大小比约 5:5 或 4:6（参考趋势）
4. **012路约束**：避免极端偏态
5. 热号与冷号的平衡（建议 6:4 或 7:3）
6. 和值范围在 {int(sum_dist['avg'])-25} 到 {int(sum_dist['avg'])+25} 之间
7. 避免与最近5期重复超过5个号码
8. 每组给出基于数据的简短分析

【输出格式】严格按以下JSON格式：
{{
  "recommendations": [
    {{
      "group": 1,
      "numbers": [号码1, ..., 号码10],
      "reason": "简短分析"
    }}
  ],
  "analysis": "整体趋势分析（300字以内）"
}}"""

    messages = [
        {"role": "system", "content": "你是专业的彩票数据分析师，擅长基于历史数据预测号码。请严格按照要求的JSON格式输出。"},
        {"role": "user", "content": prompt}
    ]
    
    result = _call_ai(messages, max_tokens=5000, temperature=temperature)
    
    if "error" in result:
        return result
    
    try:
        import re
        json_str = result["result"]
        match = re.search(r'\{[\s\S]*\}', json_str)
        if match:
            parsed = json.loads(match.group())
            return parsed
        return {"error": "AI 返回格式异常，请重试"}
    except Exception:
        return {
            "recommendations": [],
            "analysis": result.get("result", ""),
            "note": "AI 分析文字已返回，但号码解析失败"
        }


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
        {"role": "system", "content": "你是专业的彩票数据分析师，擅长基于历史数据预测号码。请严格按照要求的JSON格式输出。"},
        {"role": "user", "content": prompt}
    ]
    
    result = _call_ai(messages, max_tokens=4000, temperature=temperature)
    
    if "error" in result:
        return result
    
    try:
        import re
        json_str = result["result"]
        match = re.search(r'\{[\s\S]*\}', json_str)
        if match:
            parsed = json.loads(match.group())
            return parsed
        return {"error": "AI 返回格式异常，请重试"}
    except Exception:
        return {
            "recommendations": [],
            "analysis": result.get("result", ""),
            "note": "AI 分析文字已返回，但号码解析失败"
        }


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
【龙头凤尾】龙头: {dragon_phoenix['dragon_trend']} | 凤尾: {dragon_phoenix['phoenix_trend']}
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
        {"role": "system", "content": "你是专业的彩票数据分析师，擅长基于历史数据预测号码。请严格按照要求的JSON格式输出。"},
        {"role": "user", "content": prompt}
    ]
    
    result = _call_ai(messages, max_tokens=5000, temperature=temperature)
    
    if "error" in result:
        return result
    
    try:
        import re
        json_str = result["result"]
        match = re.search(r'\{[\s\S]*\}', json_str)
        if match:
            parsed = json.loads(match.group())
            return parsed
        return {"error": "AI 返回格式异常，请重试"}
    except Exception:
        return {
            "recommendations": [],
            "analysis": result.get("result", ""),
            "note": "AI 分析文字已返回，但号码解析失败"
        }


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
        {"role": "system", "content": "你是专业的彩票数据分析师，擅长基于历史数据预测号码。请严格按照要求的JSON格式输出。"},
        {"role": "user", "content": prompt}
    ]
    
    result = _call_ai(messages, max_tokens=4000, temperature=temperature)
    
    if "error" in result:
        return result
    
    try:
        import re
        json_str = result["result"]
        match = re.search(r'\{[\s\S]*\}', json_str)
        if match:
            parsed = json.loads(match.group())
            return parsed
        return {"error": "AI 返回格式异常，请重试"}
    except Exception:
        return {
            "recommendations": [],
            "analysis": result.get("result", ""),
            "note": "AI 分析文字已返回，但号码解析失败"
        }


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
        {"role": "system", "content": "你是专业的彩票数据分析师，擅长基于历史数据预测号码。请严格按照要求的JSON格式输出。"},
        {"role": "user", "content": prompt}
    ]
    
    result = _call_ai(messages, max_tokens=4000, temperature=temperature)
    
    if "error" in result:
        return result
    
    try:
        import re
        json_str = result["result"]
        match = re.search(r'\{[\s\S]*\}', json_str)
        if match:
            parsed = json.loads(match.group())
            return parsed
        return {"error": "AI 返回格式异常，请重试"}
    except Exception:
        return {
            "recommendations": [],
            "analysis": result.get("result", ""),
            "note": "AI 分析文字已返回，但号码解析失败"
        }


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

    else:
        recent_20 = df.head(20)
        n1_counts = recent_20['n1'].value_counts().to_dict()
        n2_counts = recent_20['n2'].value_counts().to_dict()
        n3_counts = recent_20['n3'].value_counts().to_dict()
        
        prompt = f"""请分析以下福彩3D历史数据趋势：

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
        import re
        json_str = result.get("result", "")
        match = re.search(r'\{[\s\S]*\}', json_str)
        if match:
            parsed = json.loads(match.group())
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

PREDICTION_RECORDS_FILE = DATA_DIR / 'predictions.csv'


def save_prediction_record(lottery_type: str, code: str, predictions: list, play_type: str = None):
    DATA_DIR.mkdir(exist_ok=True)
    
    import time
    new_record = {
        'lottery_type': lottery_type,
        'code': code,
        'predictions': json.dumps(predictions, ensure_ascii=False),
        'play_type': play_type or '',
        'predict_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'compared': 'false',
        'compare_result': ''
    }
    
    if PREDICTION_RECORDS_FILE.exists():
        try:
            df = _read_csv_file(PREDICTION_RECORDS_FILE, dtype={"code": str, "compared": str, "compare_result": str})
            
            mask = (df['lottery_type'] == lottery_type) & (df['code'] == code)
            pt_col = df.get('play_type', '')
            if play_type:
                mask = mask & (pt_col == play_type)
            
            if mask.any():
                for col in ['lottery_type', 'code', 'predictions', 'play_type', 'predict_time']:
                    if col in df.columns:
                        df.loc[mask, col] = new_record[col]
            else:
                df = pd.concat([pd.DataFrame([new_record]), df], ignore_index=True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            df = pd.DataFrame([new_record])
    else:
        df = pd.DataFrame([new_record])
    
    if len(df) > 100:
        df = df.head(100)
    
    df.to_csv(PREDICTION_RECORDS_FILE, index=False, encoding='utf-8-sig')


def get_prediction_records(lottery_type: str = None) -> list:
    if PREDICTION_RECORDS_FILE.exists():
        try:
            df = _read_csv_file(PREDICTION_RECORDS_FILE, dtype={"code": str, "compared": str, "compare_result": str})
            df = df.dropna(subset=['lottery_type'])
            records = []
            for _, row in df.iterrows():
                compare_result = None
                cr_val = row.get('compare_result')
                if cr_val and not pd.isna(cr_val):
                    try:
                        compare_result = json.loads(str(cr_val).strip())
                    except Exception:
                        compare_result = None
                
                predictions_str = row.get('predictions', '')
                predictions = json.loads(predictions_str) if predictions_str and not pd.isna(predictions_str) else []
                
                play_type_val = row.get('play_type', '')
                play_type = '' if pd.isna(play_type_val) else str(play_type_val)
                
                record = {
                    'lottery_type': row['lottery_type'],
                    'code': str(row['code']),
                    'predictions': predictions,
                    'play_type': play_type,
                    'predict_time': row['predict_time'],
                    'compared': str(row['compared']).lower() == 'true',
                    'compare_result': compare_result
                }
                records.append(record)
            if lottery_type:
                return [r for r in records if r.get('lottery_type') == lottery_type]
            return records
        except Exception:
            return []
    return []


def get_prediction_for_code(lottery_type: str, code: str) -> dict:
    records = get_prediction_records(lottery_type)
    for record in records:
        if record.get('code') == code:
            return record
    return None


def update_prediction_compare(lottery_type: str, code: str, compare_result: dict):
    if PREDICTION_RECORDS_FILE.exists():
        try:
            df = _read_csv_file(PREDICTION_RECORDS_FILE, dtype={"code": str, "compared": str, "compare_result": str})
            df['compared'] = df['compared'].astype(str)
            df['compare_result'] = df['compare_result'].astype(str)
            mask = (df['lottery_type'] == lottery_type) & (df['code'] == code)
            if mask.any():
                df.loc[mask, 'compared'] = 'true'
                df.loc[mask, 'compare_result'] = json.dumps(compare_result, ensure_ascii=False, default=lambda x: int(x) if isinstance(x, (int, float, np.integer, np.floating)) else bool(x) if isinstance(x, (bool, np.bool_)) else x)
                df.to_csv(PREDICTION_RECORDS_FILE, index=False, encoding='utf-8-sig')
        except Exception as e:
            import traceback
            traceback.print_exc()


def analyze_saved_predictions(lottery_type: str) -> dict:
    df = _read_lottery_data(lottery_type)
    if df.empty:
        return {'error': '暂无历史数据'}
    
    records = get_prediction_records(lottery_type)
    if not records:
        return {'error': '暂无预测记录，请先进行AI预测'}
    
    latest = df.iloc[0]
    latest_code = str(latest.get('code', ''))
    latest_date = latest.get('date', '')
    
    record = get_prediction_for_code(lottery_type, latest_code)
    
    if not record:
        next_code = str(int(latest_code) + 1)
        record = get_prediction_for_code(lottery_type, next_code)
        
        if not record:
            return {
                'latest': {
                    'code': latest_code,
                    'date': latest_date
                },
                'error': '暂无该期的预测记录',
                'available_codes': [r['code'] for r in records]
            }
    
    if int(record['code']) > int(latest_code):
        return {
            'latest': {
                'code': latest_code,
                'date': latest_date
            },
            'error': f'预测期号 {record["code"]} 尚未开奖（最新开奖期号：{latest_code}），请等待开奖后再对比',
            'available_codes': [r['code'] for r in records]
        }
    
    if record.get('compared'):
        return record['compare_result']
    
    actual_nums = []
    if lottery_type == 'ssq':
        actual_nums = {
            'reds': [latest['r1'], latest['r2'], latest['r3'], 
                     latest['r4'], latest['r5'], latest['r6']],
            'blue': latest['blue']
        }
    elif lottery_type == 'kl8':
        cols = [f'n{i:02d}' for i in range(1, 21)]
        actual_nums = {'nums': [latest[col] for col in cols if col in latest]}
    else:
        actual_nums = {'nums': [latest['n1'], latest['n2'], latest['n3']]}
    
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
    
    else:
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
            'code': latest_code,
            'date': latest_date,
            **actual_nums
        },
        'ai_best': best_match,
        'predict_time': record.get('predict_time', ''),
        'prize_result': calculate_total_prize(lottery_type, predictions, actual_nums)
    }
    
    update_prediction_compare(lottery_type, record['code'], compare_result)
    
    return compare_result


SSQ_PRIZES = {
    7: {"name": "一等奖", "prize": 5000000, "desc": "6+1（浮动奖金，参考值）"},
    6: {"name": "二等奖", "prize": 500000, "desc": "6+0（浮动奖金，参考值）"},
    5: {"name": "三等奖", "prize": 3000, "desc": "5+1"},
    4: {"name": "四等奖", "prize": 200, "desc": "5+0/4+1"},
    3: {"name": "五等奖", "prize": 10, "desc": "4+0/3+1"},
    2: {"name": "六等奖", "prize": 5, "desc": "2+1/1+1/0+1"}
}


KL8_PRIZES = {
    "pick1": {1: {"name": "一等奖", "prize": 4.6}},
    "pick2": {2: {"name": "一等奖", "prize": 46}, 0: {"name": "中零奖", "prize": 4}},
    "pick3": {3: {"name": "一等奖", "prize": 360}, 0: {"name": "中零奖", "prize": 3}},
    "pick4": {4: {"name": "一等奖", "prize": 2400}, 3: {"name": "二等奖", "prize": 20}, 0: {"name": "中零奖", "prize": 2}},
    "pick5": {5: {"name": "一等奖", "prize": 10000}, 4: {"name": "二等奖", "prize": 50}, 3: {"name": "三等奖", "prize": 5}, 0: {"name": "中零奖", "prize": 1}},
    "pick6": {6: {"name": "一等奖", "prize": 50000}, 5: {"name": "二等奖", "prize": 300}, 4: {"name": "三等奖", "prize": 10}, 0: {"name": "中零奖", "prize": 1}},
    "pick7": {7: {"name": "一等奖", "prize": 200000}, 6: {"name": "二等奖", "prize": 1000}, 5: {"name": "三等奖", "prize": 30}, 0: {"name": "中零奖", "prize": 1}},
    "pick8": {8: {"name": "一等奖", "prize": 500000}, 7: {"name": "二等奖", "prize": 5000}, 6: {"name": "三等奖", "prize": 100}, 5: {"name": "四等奖", "prize": 5}, 0: {"name": "中零奖", "prize": 1}},
    "pick9": {9: {"name": "一等奖", "prize": 1000000}, 8: {"name": "二等奖", "prize": 20000}, 7: {"name": "三等奖", "prize": 500}, 6: {"name": "四等奖", "prize": 20}, 0: {"name": "中零奖", "prize": 1}},
    "pick10": {10: {"name": "一等奖", "prize": 5000000}, 9: {"name": "二等奖", "prize": 100000}, 8: {"name": "三等奖", "prize": 2000}, 7: {"name": "四等奖", "prize": 50}, 6: {"name": "五等奖", "prize": 5}, 5: {"name": "六等奖", "prize": 10}, 0: {"name": "中零奖", "prize": 10}}
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
    for matched in sorted(prizes.keys(), reverse=True):
        if match_count >= matched:
            return prizes[matched]
    
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


def calculate_total_prize(lottery_type, predictions, actual_nums, play_type=None):
    total_prize = 0
    total_cost = len(predictions) * 2
    results = []
    
    for i, pred in enumerate(predictions, 1):
        if lottery_type == "ssq":
            reds = pred.get("red", [])
            blue = pred.get("blue", 0)
            prize_info = calculate_ssq_prize(reds, blue, actual_nums.get("reds", []), actual_nums.get("blue", 0))
        elif lottery_type == "kl8":
            nums = pred.get("nums", [])
            pt = play_type or "pick10"
            prize_info = calculate_kl8_prize(nums, actual_nums.get("nums", []), pt)
        else:
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
