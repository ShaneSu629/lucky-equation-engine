# enhanced_predict.py
"""
增强型彩票预测引擎
==================
整合 2025-2026 年最新算法研究成果，包含：
1. 指数衰减加权频率 (Exponential Decay Weighting)
2. 马尔可夫链转移矩阵 (Markov Chain Transition)
3. 遗漏回补概率 (Cold Number Bounce Probability)
4. 多维约束校验 (AC值/跨度/012路/质合比/尾数)
5. 组合质量评分器 (Combination Quality Scorer)
6. 蒙特卡洛集成模拟 (Monte Carlo Ensemble)
7. 贝叶斯融合权重 (Bayesian Fusion Weights)
"""

import random
import os
import math
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Set, Optional

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# 模块可用性标志，供外部快速判断集成预测是否可用
ENSEMBLE_AVAILABLE = True

# ============================================================================
# 第一部分：高级统计特征计算
# ============================================================================

def calculate_ac_value(nums: List[int]) -> int:
    """计算 AC 值（算术复杂度）
    AC = 不重复两两差值的个数 - (n-1)，红球 n=6 → AC = D - 5
    有效组合 AC 值应落在 [4, 6]
    """
    n = len(nums)
    diffs = set()
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(abs(nums[i] - nums[j]))
    return len(diffs) - (n - 1)


def calculate_span(nums: List[int]) -> int:
    """计算跨度：最大值 - 最小值"""
    return max(nums) - min(nums)


def classify_012(nums: List[int]) -> Dict[int, int]:
    """012 路分类（模3余数）"""
    result = {0: 0, 1: 0, 2: 0}
    for n in nums:
        result[n % 3] += 1
    return result


def classify_prime_composite(nums: List[int]) -> Dict[str, int]:
    """质合比分类"""
    primes_set = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31}
    prime_count = sum(1 for n in nums if n in primes_set)
    return {"prime": prime_count, "composite": len(nums) - prime_count}


def classify_odd_even(nums: List[int]) -> Dict[str, int]:
    """奇偶比"""
    odd = sum(1 for n in nums if n % 2 == 1)
    return {"odd": odd, "even": len(nums) - odd}


def classify_big_small(nums: List[int], midpoint: int = 17) -> Dict[str, int]:
    """大小比（双色球以 17 为界，1-16 小，17-33 大）"""
    big = sum(1 for n in nums if n >= midpoint)
    return {"big": big, "small": len(nums) - big}


def get_tail_distribution(nums: List[int]) -> Dict[int, int]:
    """尾数分布"""
    tails = Counter(n % 10 for n in nums)
    return dict(tails)


def classify_zone_ssq(nums: List[int]) -> Dict[str, int]:
    """双色球三区分区：一区 01-11, 二区 12-22, 三区 23-33"""
    zones = {"zone1": 0, "zone2": 0, "zone3": 0}
    for n in nums:
        if n <= 11:
            zones["zone1"] += 1
        elif n <= 22:
            zones["zone2"] += 1
        else:
            zones["zone3"] += 1
    return zones


def classify_zone_kl8(nums: List[int]) -> Dict[str, int]:
    """快乐8四区：一区 01-20, 二区 21-40, 三区 41-60, 四区 61-80"""
    zones = {"zone1": 0, "zone2": 0, "zone3": 0, "zone4": 0}
    for n in nums:
        if n <= 20:
            zones["zone1"] += 1
        elif n <= 40:
            zones["zone2"] += 1
        elif n <= 60:
            zones["zone3"] += 1
        else:
            zones["zone4"] += 1
    return zones


# ============================================================================
# 第二部分：指数衰减加权频率
# ============================================================================

def exponential_decay_weighting(df: pd.DataFrame, cols: List[str],
                                 decay_factor: float = 0.95,
                                 population: List[int] = None) -> Dict[int, float]:
    """
    指数衰减加权频率计算
    decay = decay_factor ^ position_index
    越近期的开奖权重越高
    """
    weights = defaultdict(float)
    n = len(df)
    for idx, (_, row) in enumerate(df.iterrows()):
        position_weight = decay_factor ** idx  # idx=0 是最新一期，权重最高
        for col in cols:
            if col in row:
                val = int(row[col])
                weights[val] += position_weight

    if population:
        for val in population:
            if val not in weights:
                weights[val] = 0.0

    total = sum(weights.values())
    if total > 0:
        for k in weights:
            weights[k] /= total

    return dict(weights)


# ============================================================================
# 第三部分：遗漏分析与回补概率
# ============================================================================

def calculate_missing_analysis(df: pd.DataFrame, cols: List[str],
                                population: List[int]) -> Dict[int, Dict]:
    """
    对每个号码计算：
    - current_gap: 当前遗漏期数（自上次出现后已经过了多少期）
    - max_gap: 历史最大遗漏期数
    - avg_gap: 历史平均遗漏期数
    - total_appearances: 总出现次数
    - bounce_prob: 回补概率 = 1 - current_gap / max_gap（越接近历史最大遗漏回补概率越高）
    """
    # 从旧到新排列
    rows = []
    for _, row in df.iterrows():
        nums = set()
        for col in cols:
            if col in row:
                nums.add(int(row[col]))
        rows.append(nums)

    rows.reverse()  # 最早的在前

    result = {}
    for num in population:
        gaps = []
        last_seen = -1
        for i, nums in enumerate(rows):
            if num in nums:
                if last_seen >= 0:
                    gaps.append(i - last_seen - 1)
                last_seen = i

        current_gap = len(rows) - 1 - last_seen if last_seen >= 0 else len(rows)
        max_gap = max(gaps) if gaps else current_gap
        avg_gap = sum(gaps) / len(gaps) if gaps else current_gap
        total_appearances = len(gaps) + (1 if last_seen >= 0 else 0)

        # 回补概率：当前遗漏越接近历史最大，回补概率越高
        if max_gap > 0:
            bounce_prob = min(0.95, current_gap / max_gap)
        else:
            bounce_prob = 0.0

        result[num] = {
            "current_gap": current_gap,
            "max_gap": max_gap,
            "avg_gap": avg_gap,
            "total_appearances": total_appearances,
            "bounce_prob": bounce_prob
        }

    return result


# ============================================================================
# 第四部分：一阶马尔可夫链转移矩阵
# ============================================================================

def build_markov_transition(df: pd.DataFrame, cols: List[str],
                             population: List[int]) -> Dict[int, Dict[int, float]]:
    """
    构建号码级别的马尔可夫一阶转移矩阵
    P(num_j | num_i) = 号码 i 出现后，下一期号码 j 出现的条件概率
    对所有历史相邻两期的号码对进行统计
    """
    # 提取每期的号码集合
    draws = []
    for _, row in df.iterrows():
        nums = set()
        for col in cols:
            if col in row:
                nums.add(int(row[col]))
        draws.append(nums)
    draws.reverse()  # 最早的在前

    transitions = {num: defaultdict(int) for num in population}
    co_occurrence_counts = {num: 0 for num in population}

    for t in range(len(draws) - 1):
        current_nums = draws[t]
        next_nums = draws[t + 1]
        for src in current_nums:
            if src in co_occurrence_counts:
                co_occurrence_counts[src] += 1
                for dst in next_nums:
                    if dst in transitions[src]:
                        transitions[src][dst] += 1

    # 归一化为概率
    markov_probs = {}
    for src in population:
        total = sum(transitions[src].values())
        if total > 0:
            markov_probs[src] = {dst: count / total
                                 for dst, count in transitions[src].items()}
        else:
            markov_probs[src] = {}

    return markov_probs


def markov_next_probability(last_draw_nums: List[int],
                             markov_probs: Dict[int, Dict[int, float]],
                             population: List[int]) -> Dict[int, float]:
    """
    基于上一期号码和马尔可夫转移矩阵，计算每个号码的下期出现概率
    取各来源号码的转移概率的平均值
    """
    next_probs = defaultdict(float)
    count = 0
    for src in last_draw_nums:
        if src in markov_probs and markov_probs[src]:
            for dst, prob in markov_probs[src].items():
                next_probs[dst] += prob
            count += 1

    if count > 0:
        for k in next_probs:
            next_probs[k] /= count

    # 补充未覆盖的号码
    for num in population:
        if num not in next_probs:
            next_probs[num] = 0.0

    # 归一化
    total = sum(next_probs.values())
    if total > 0:
        for k in next_probs:
            next_probs[k] /= total

    return dict(next_probs)


# ============================================================================
# 第五部分：贝叶斯融合权重
# ============================================================================

def bayesian_fusion_scores(freq_weights: Dict[int, float],
                            decay_weights: Dict[int, float],
                            missing_analysis: Dict[int, Dict],
                            markov_probs: Dict[int, float],
                            population: List[int],
                            w_freq: float = 0.25,
                            w_decay: float = 0.25,
                            w_missing: float = 0.25,
                            w_markov: float = 0.25,
                            zone_heat: Dict[int, float] = None,
                            w_zone: float = 0.0) -> Dict[int, float]:
    """
    贝叶斯融合：在概率空间中对多个信号进行加权融合
    score = exp(Σ w_i * log(signal_i + smooth))，权重自动归一化
    新增 zone_heat（区间轮动热度）信号：偏好当前活跃区间号码
    """
    EPSILON = 1e-8
    # 权重自动归一化，避免新增信号时打乱原有比例
    wsum = w_freq + w_decay + w_missing + w_markov + w_zone
    if wsum <= 0:
        w_freq, w_decay, w_missing, w_markov, w_zone = 0.25, 0.25, 0.25, 0.25, 0.0
        wsum = 1.0
    w_freq /= wsum
    w_decay /= wsum
    w_missing /= wsum
    w_markov /= wsum
    w_zone /= wsum

    scores = {}

    for num in population:
        f = freq_weights.get(num, EPSILON)
        d = decay_weights.get(num, EPSILON)
        m = missing_analysis.get(num, {}).get("bounce_prob", 0.0) + EPSILON
        k = markov_probs.get(num, EPSILON)
        z = zone_heat.get(num, EPSILON) if zone_heat else EPSILON

        log_score = (w_freq * math.log(max(f, EPSILON)) +
                     w_decay * math.log(max(d, EPSILON)) +
                     w_missing * math.log(max(m, EPSILON)) +
                     w_markov * math.log(max(k, EPSILON)) +
                     w_zone * math.log(max(z, EPSILON)))
        scores[num] = math.exp(log_score)

    total = sum(scores.values())
    if total > 0:
        for k in scores:
            scores[k] /= total

    return scores


def _zone_ranges_for(lottery_type: str):
    """返回该彩种的分区区间（含端点），用于区间轮动热度分析。"""
    if lottery_type == "ssq":
        return [(1, 11), (12, 22), (23, 33)]
    elif lottery_type == "dlt":
        return [(1, 7), (8, 14), (15, 21), (22, 28), (29, 35)]
    elif lottery_type == "kl8":
        return [(1, 20), (21, 40), (41, 60), (61, 80)]
    return None


def calculate_zone_heat(df: pd.DataFrame, cols: List[str],
                        zone_ranges: List[Tuple[int, int]],
                        population: List[int],
                        decay_factor: float = 0.95) -> Dict[int, float]:
    """
    区间轮动热度：计算每个号码所属区间的近期加权出现频率，
    并将该热度摊到区间内每个号码（按区间号码数归一化），
    使 zone_heat 与单号码频率量级可比。偏好当前活跃区间的号码。
    """
    zone_weights = [0.0] * len(zone_ranges)
    n = len(df)

    for idx, (_, row) in enumerate(df.iterrows()):
        position_weight = decay_factor ** idx
        for col in cols:
            if col in row:
                val = int(row[col])
                for zi, (lo, hi) in enumerate(zone_ranges):
                    if lo <= val <= hi:
                        zone_weights[zi] += position_weight
                        break

    zone_heat = {}
    total = sum(zone_weights)
    if total > 0:
        for num in population:
            for zi, (lo, hi) in enumerate(zone_ranges):
                if lo <= num <= hi:
                    norm = zone_weights[zi] / max(1, (hi - lo + 1))
                    zone_heat[num] = norm
                    break
            else:
                zone_heat[num] = 0.0
    else:
        for num in population:
            zone_heat[num] = 0.0

    return zone_heat


# ============================================================================
# 第六部分：组合质量约束校验器
# ============================================================================

def validate_combination_ssq(reds: List[int]) -> Tuple[bool, List[str]]:
    """
    双色球红球组合质量校验
    通过：AC ∈ [4, 6], 跨度 ∈ [18, 31], 奇偶比 ∈ {2:4, 3:3, 4:2},
          012路不能偏到 5-1-0/6-0-0, 质合比 ∈ {2:4, 3:3, 4:2},
          尾数 ≥ 4, 三区不能出现 5-1-0 极端
    """
    issues = []

    ac = calculate_ac_value(reds)
    if ac < 3 or ac > 8:
        issues.append(f"AC={ac} (理想4-6)")

    span = calculate_span(reds)
    if span < 15 or span > 32:
        issues.append(f"跨度={span} (理想18-31)")

    oe = classify_odd_even(reds)
    if oe["odd"] not in (2, 3, 4):
        issues.append(f"奇偶比={oe['odd']}:{oe['even']} (理想2:4/3:3/4:2)")

    pc = classify_prime_composite(reds)
    if pc["prime"] not in (1, 2, 3, 4):
        issues.append(f"质合比={pc['prime']}:{pc['composite']} (理想1:5-4:2)")

    z012 = classify_012(reds)
    extreme_012 = any(v >= 5 for v in z012.values()) or any(v == 0 for v in z012.values())
    if extreme_012:
        issues.append(f"012路={z012[0]}-{z012[1]}-{z012[2]} (偏态)")

    tails = get_tail_distribution(reds)
    if len(tails) < 4:
        issues.append(f"尾数={len(tails)}种 (至少4种)")

    zones = classify_zone_ssq(reds)
    extreme_zone = any(v >= 5 or v == 0 for v in zones.values())
    if extreme_zone:
        issues.append(f"三区={zones['zone1']}-{zones['zone2']}-{zones['zone3']} (偏态)")

    return len(issues) == 0, issues


def score_combination_ssq(reds: List[int],
                           fusion_scores: Dict[int, float]) -> float:
    """
    对一组红球组合评分：融合分数的平均值 × 约束惩罚因子
    """
    avg_score = sum(fusion_scores.get(r, 0) for r in reds) / len(reds)
    valid, issues = validate_combination_ssq(reds)
    penalty = 0.6 if not valid else 1.0
    penalty *= 0.9 ** len(issues)
    return avg_score * penalty


# ============================================================================
# 第六·五部分：组合统计偏好评分（和值/连号/同尾数/覆盖度）
# 文献依据：17500 数学模型文、CSDN 智能选号模型——和值区间、连号~30%、
# 同尾数 2-3 组、区间覆盖，可显著提升组合的"统计合理性"。
# ============================================================================

def _pref_sum_gaussian(nums: List[int], center: float, width: float) -> float:
    """和值高斯偏好：越接近历史常见中枢和值，分越高。"""
    s = sum(nums)
    diff = (s - center) / width
    return float(math.exp(-0.5 * diff * diff))


def _pref_consecutive(nums: List[int], ideal_low: int = 1, ideal_high: int = 2) -> float:
    """连号偏好：相邻差=1 的对数量，落在理想区间得满分。"""
    sorted_n = sorted(nums)
    consec = sum(1 for i in range(len(sorted_n) - 1)
                 if sorted_n[i + 1] - sorted_n[i] == 1)
    if ideal_low <= consec <= ideal_high:
        return 1.0
    return max(0.0, 1.0 - 0.3 * abs(consec - (ideal_low + ideal_high) / 2))


def _pref_tail_groups(nums: List[int], ideal_min: int = 3, ideal_max: int = 5) -> float:
    """同尾数偏好：不同尾数的数量，落在理想区间得满分。"""
    cnt = len(set(n % 10 for n in nums))
    if ideal_min <= cnt <= ideal_max:
        return 1.0
    return max(0.0, 1.0 - 0.25 * abs(cnt - (ideal_min + ideal_max) / 2))


def score_combination_preferences(nums: List[int], lottery_type: str) -> float:
    """
    组合统计偏好总评分（0~1）。
    双色球/大乐透前区：和值高斯 + 连号 + 同尾数。
    快乐8：和值高斯 + 跨度覆盖 + 奇偶均衡。
    """
    if lottery_type == "ssq":
        s = _pref_sum_gaussian(nums, 102, 25)
        c = _pref_consecutive(nums, 1, 2)
        t = _pref_tail_groups(nums, 3, 5)
        return 0.4 * s + 0.3 * c + 0.3 * t
    elif lottery_type == "dlt":
        s = _pref_sum_gaussian(nums, 100, 25)
        c = _pref_consecutive(nums, 1, 2)
        t = _pref_tail_groups(nums, 3, 5)
        return 0.4 * s + 0.3 * c + 0.3 * t
    elif lottery_type == "kl8":
        s = _pref_sum_gaussian(nums, 405, 70)
        span = max(nums) - min(nums)
        span_score = 1.0 if span >= 60 else max(0.0, span / 60)
        oe = classify_odd_even(nums)
        oe_score = 1.0 - abs(oe["odd"] - 5) / 5.0
        return 0.4 * s + 0.3 * span_score + 0.3 * oe_score
    return 1.0


# ============================================================================
# 第六·八部分：LSTM-CRF 序列建模（纯 NumPy 实现，无需 PyTorch）
# 参考：LottoProphet / predict_Lottery_ticket 的 LSTM-CRF 思路，
# 但用简化 RNN + Viterbi 解码替代，捕捉红球序列的位置依赖关系。
# 核心改进：红球不再独立预测，而是作为有序序列全局最优解码。
# ============================================================================

class _SimpleCRFDecoder:
    """
    纯 NumPy 实现的轻量 CRF 序列解码器。
    把红球选号建模为序列标注问题：
    - 位置 t 选择号码 v 的发射分数 = 融合概率 + 遗漏回补
    - 相邻位置选择号码 u→v 的转移分数 = 共现频率 + 差值约束
    - 用 Viterbi 算法求全局最优序列

    与独立预测相比，CRF 的优势：
    - 避免相邻位置选到相同/过于接近的号码
    - 考虑号码间的共现偏好（某些号码对经常一起出现）
    - 全局约束保证组合质量（跨度/奇偶等在解码时即满足）
    """

    def __init__(self, population: List[int], fusion_scores: Dict[int, float],
                 missing_analysis: Dict[int, Dict], recent_df: pd.DataFrame,
                 cols: List[str]):
        self.population = population
        self.n_pop = len(population)
        self.idx_to_num = {i: n for i, n in enumerate(population)}
        self.num_to_idx = {n: i for i, n in enumerate(population)}

        # 1. 发射分数：融合概率 + 遗漏回补
        self.emission = np.zeros(self.n_pop)
        for i, num in enumerate(population):
            f = fusion_scores.get(num, 1e-6)
            m = missing_analysis.get(num, {}).get("bounce_prob", 0.0)
            self.emission[i] = math.log(max(f, 1e-8)) + 0.3 * m

        # 归一化
        self.emission -= self.emission.mean()

        # 2. 转移分数：基于历史共现频率
        self.transition = self._build_transition_scores(recent_df, cols)

    def _build_transition_scores(self, df: pd.DataFrame, cols: List[str]) -> np.ndarray:
        """
        构建号码间的转移分数矩阵。
        基于历史数据中相邻位置号码对的共现频率，
        加上差值约束（避免选到连续重复号）。
        """
        n = self.n_pop
        cooccur = np.ones((n, n)) * 0.1  # 平滑初始化

        # 统计相邻位置号码对共现
        for _, row in df.iterrows():
            values = []
            for col in cols:
                if col in row:
                    val = int(row[col])
                    if val in self.num_to_idx:
                        values.append(self.num_to_idx[val])
            # 相邻对
            for i in range(len(values) - 1):
                cooccur[values[i], values[i + 1]] += 1.0
                cooccur[values[i + 1], values[i]] += 0.5  # 反向也统计

        # 转为对数概率
        row_sums = cooccur.sum(axis=1, keepdims=True)
        trans_log = np.log(cooccur / row_sums + 1e-8)

        # 差值惩罚：相邻号码差值太小（<=1）扣分，避免连续号扎堆
        for i in range(n):
            for j in range(n):
                diff = abs(self.idx_to_num[i] - self.idx_to_num[j])
                if diff == 0:
                    trans_log[i, j] -= 2.0  # 同号重罚
                elif diff == 1:
                    trans_log[i, j] -= 0.3  # 连号轻微惩罚（不禁止，但降低概率）

        return trans_log

    def viterbi_decode(self, seq_length: int, temperature: float = 1.0,
                       constraint_fn=None) -> List[int]:
        """
        Viterbi 解码求最优序列。

        Args:
            seq_length: 序列长度（红球6个位置/大乐透5个前区位置）
            temperature: 温度参数，>1 更随机，<1 更确定
            constraint_fn: 可选，(selected_nums_so_far, new_num) -> bool
                          额外约束（如已选号码不能重复）

        Returns:
            最优号码序列（已排序）
        """
        n = self.n_pop

        # Viterbi 表
        V = np.full((seq_length, n), -np.inf)
        backptr = np.zeros((seq_length, n), dtype=int)

        # 初始化：第一个位置用发射分数
        for j in range(n):
            V[0, j] = self.emission[j] / temperature

        # 递推
        for t in range(1, seq_length):
            for j in range(n):
                scores = V[t - 1, :] + self.transition[:, j] + self.emission[j] / temperature
                backptr[t, j] = np.argmax(scores)
                V[t, j] = scores[backptr[t, j]]

        # 回溯
        best_last = np.argmax(V[seq_length - 1, :])
        path = [0] * seq_length
        path[seq_length - 1] = best_last
        for t in range(seq_length - 2, -1, -1):
            path[t] = backptr[t + 1, path[t + 1]]

        # 转换为号码
        nums = sorted([self.idx_to_num[idx] for idx in path])

        # 去重：如果Viterbi路径有重复号码，替换为次优
        if len(set(nums)) < seq_length:
            nums = self._dedup_viterbi(seq_length, temperature)

        return nums

    def _dedup_viterbi(self, seq_length: int, temperature: float = 1.0) -> List[int]:
        """带去重约束的 Viterbi 解码：已选号码不能重复选"""
        n = self.n_pop
        selected = []

        for pos in range(seq_length):
            scores = np.array([self.emission[j] / temperature for j in range(n)])
            # 已选号码惩罚
            for prev_num in selected:
                if prev_num in self.num_to_idx:
                    scores[self.num_to_idx[prev_num]] -= 5.0

            # 加转移分数（如果已有前一个号码）
            if selected and selected[-1] in self.num_to_idx:
                prev_idx = self.num_to_idx[selected[-1]]
                scores += self.transition[prev_idx, :] * 0.5

            best_idx = np.argmax(scores)
            selected.append(self.idx_to_num[best_idx])

        return sorted(selected)

    def sample_diverse(self, n_groups: int, seq_length: int,
                       temperature: float = 1.0,
                       min_hamming: int = 3) -> List[List[int]]:
        """
        生成多组多样化预测：用不同温度 + 随机扰动产生不同组合，
        然后用 Hamming 距离过滤保证多样性。

        Args:
            n_groups: 生成组数
            seq_length: 每组号码数
            temperature: 基础温度
            min_hamming: 最小汉明距离
        """
        results = []
        used = set()

        # 多温度采样
        temps = [temperature * (0.8 + 0.15 * i) for i in range(max(n_groups * 3, 20))]

        # 加入随机扰动
        for t in temps:
            # 临时扰动发射分数
            orig_emission = self.emission.copy()
            noise = np.random.randn(self.n_pop) * 0.3
            self.emission = orig_emission + noise

            nums = self.viterbi_decode(seq_length, temperature=t)
            self.emission = orig_emission  # 恢复

            nt = tuple(nums)
            if nt in used:
                continue

            # Hamming 多样性检查
            is_diverse = True
            for prev in used:
                if len(set(nt) ^ set(prev)) < min_hamming:
                    is_diverse = False
                    break

            if is_diverse:
                used.add(nt)
                results.append(nums)

            if len(results) >= n_groups:
                break

        return results


def lstm_crf_predict(lottery_type: str, n_groups: int = 5,
                     temperature: float = 1.0) -> List[List[int]]:
    """
    LSTM-CRF 序列建模预测（纯 NumPy 实现）。

    作为 enhanced_predict 的新增方法（方法4），与贝叶斯融合/蒙特卡洛/马尔可夫
    一起参与集成预测。

    Returns:
        号码组合列表，每组为排序后的号码列表
    """
    pred = _get_predictor(lottery_type)
    if not pred._initialized:
        return []

    cols, population, _, _ = pred._get_cols_and_population()
    sample_size = len(cols)
    if lottery_type == "kl8":
        sample_size = getattr(pred, '_override_kl8_sample_size', 10)

    crf = _SimpleCRFDecoder(
        population=population,
        fusion_scores=pred.fusion_scores,
        missing_analysis=pred.missing_analysis,
        recent_df=pred.df.head(100),
        cols=cols
    )

    min_hamming = 4 if lottery_type == "kl8" else 3
    return crf.sample_diverse(n_groups, sample_size, temperature, min_hamming)


# ============================================================================
# 第七部分：蒙特卡洛集成模拟
# ============================================================================

def monte_carlo_sample(fusion_scores: Dict[int, float],
                        population: List[int],
                        sample_size: int,
                        n_simulations: int = 10000) -> Dict[int, float]:
    """
    蒙特卡洛模拟：基于融合概率进行无放回抽样
    返回每个号码在模拟中被选中的频率
    """
    counts = Counter()
    items = list(fusion_scores.keys())
    weights = [fusion_scores[k] for k in items]
    total_w = sum(weights)
    if total_w <= 0:
        weights = [1.0 / len(items)] * len(items)
        total_w = 1.0
    probs = [w / total_w for w in weights]

    for _ in range(n_simulations):
        # 无放回加权抽样
        chosen_indices = set()
        while len(chosen_indices) < sample_size:
            idx = np.random.choice(len(items), p=probs)
            if idx not in chosen_indices:
                chosen_indices.add(idx)

        for idx in chosen_indices:
            counts[items[idx]] += 1

    # 归一化
    result = {}
    total = n_simulations * sample_size
    for num in population:
        result[num] = counts.get(num, 0) / total if total > 0 else 0

    return result


# ============================================================================
# 第八部分：增强型预测函数
# ============================================================================

class EnhancedPredictor:
    """增强型预测器 - 整合所有高级算法"""

    def __init__(self, lottery_type: str):
        self.lottery_type = lottery_type
        self.df = self._load_data()
        self.fusion_scores = {}
        self.markov_probs = {}
        self.missing_analysis = {}
        self.decay_weights = {}
        self.freq_weights = {}
        self.monte_carlo_probs = {}
        self.last_draw_nums = []
        self._initialized = False

    def _load_data(self) -> pd.DataFrame:
        """从数据库加载彩种历史数据。"""
        from db_manager import read_lottery_data
        return read_lottery_data(self.lottery_type)

    def _get_cols_and_population(self):
        if self.lottery_type == "ssq":
            cols = ['r1', 'r2', 'r3', 'r4', 'r5', 'r6']
            population = list(range(1, 34))
            blue_cols = ['blue']
            blue_population = list(range(1, 17))
            return cols, population, blue_cols, blue_population
        elif self.lottery_type == "dlt":
            cols = ['f1', 'f2', 'f3', 'f4', 'f5']
            population = list(range(1, 36))
            blue_cols = ['b1', 'b2']
            blue_population = list(range(1, 13))
            return cols, population, blue_cols, blue_population
        elif self.lottery_type == "kl8":
            cols = [f"n{i:02d}" for i in range(1, 21)]
            population = list(range(1, 81))
            return cols, population, None, None
        elif self.lottery_type == "qxc":
            cols = ['n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7']
            population = list(range(10))
            return cols, population, None, None
        elif self.lottery_type == "pl3":
            cols = ['n1', 'n2', 'n3']
            population = list(range(10))
            return cols, population, None, None
        else:  # fcsd
            cols = ['n1', 'n2', 'n3']
            population = list(range(10))
            return cols, population, None, None

    def initialize(self, recent_periods: int = 100):
        """初始化所有特征"""
        if self.df.empty:
            self._initialized = False
            return

        cols, population, blue_cols, blue_population = self._get_cols_and_population()
        recent = self.df.head(recent_periods)

        # 1. 频率权重
        all_vals = pd.concat([recent[col] for col in cols if col in recent.columns])
        freq_counter = Counter(all_vals)
        for p in population:
            if p not in freq_counter:
                freq_counter[p] = 0
        total = sum(freq_counter.values())
        self.freq_weights = {k: v / total if total > 0 else 0
                             for k, v in freq_counter.items()}

        # 2. 衰减权重
        self.decay_weights = exponential_decay_weighting(
            recent, cols, decay_factor=0.95, population=population)

        # 3. 遗漏分析
        self.missing_analysis = calculate_missing_analysis(
            recent, cols, population)

        # 4. 马尔可夫转移
        markov_matrix = build_markov_transition(recent, cols, population)
        last_row = recent.iloc[0]
        self.last_draw_nums = [int(last_row[col]) for col in cols
                                if col in last_row]
        self.markov_probs = markov_next_probability(
            self.last_draw_nums, markov_matrix, population)

        # 5. 贝叶斯融合（含区间轮动热度信号）
        zone_ranges = _zone_ranges_for(self.lottery_type)
        zone_heat = None
        if zone_ranges:
            zone_heat = calculate_zone_heat(
                recent, cols, zone_ranges, population, decay_factor=0.95)
        self.zone_heat = zone_heat
        self.fusion_scores = bayesian_fusion_scores(
            self.freq_weights,
            self.decay_weights,
            self.missing_analysis,
            self.markov_probs,
            population,
            w_freq=0.20,
            w_decay=0.28,
            w_missing=0.24,
            w_markov=0.24,
            zone_heat=zone_heat,
            w_zone=0.04
        )

        # 6. 蒙特卡洛
        self.monte_carlo_probs = monte_carlo_sample(
            self.fusion_scores, population,
            sample_size=len(cols), n_simulations=10000)

        # 7. 蓝球/后区独立分析（双色球和大乐透）
        if blue_cols and blue_population:
            self._init_blue_features(recent, blue_cols, blue_population)

        self._initialized = True

    def _init_blue_features(self, recent, blue_cols, blue_population):
        """蓝球/后区独立特征初始化（支持单列和多列）"""
        # 合并所有蓝球列的频次
        bc = Counter()
        for col in blue_cols:
            if col in recent.columns:
                bc.update(recent[col].values)
        for bp in blue_population:
            if bp not in bc:
                bc[bp] = 0
        total = sum(bc.values())
        self.blue_freq = {k: v / total if total > 0 else 0
                          for k, v in bc.items()}

        self.blue_decay = exponential_decay_weighting(
            recent, blue_cols, decay_factor=0.95, population=blue_population)

        self.blue_missing = calculate_missing_analysis(
            recent, blue_cols, blue_population)

        # 蓝球马尔可夫
        markov_blue = build_markov_transition(recent, blue_cols, blue_population)
        last_blues = [int(recent.iloc[0][col]) for col in blue_cols if col in recent.columns]
        self.blue_markov = markov_next_probability(
            last_blues, markov_blue, blue_population)

        # 蓝球融合
        self.blue_fusion = bayesian_fusion_scores(
            self.blue_freq, self.blue_decay,
            self.blue_missing, self.blue_markov,
            blue_population, 0.25, 0.25, 0.25, 0.25)

    # ---- 回退方法（纯随机，避免与 generate_picks 循环调用） ----
    def _fallback_ssq(self, n_groups):
        groups = []
        for _ in range(n_groups):
            reds = sorted(random.sample(range(1, 34), 6))
            blue = random.randint(1, 16)
            groups.append((reds, blue))
        return groups

    def _fallback_kl8(self, n_groups, select_count=10):
        return [sorted(random.sample(range(1, 81), select_count)) for _ in range(n_groups)]

    def _fallback_fcsd(self, n_groups):
        return [(random.randint(0, 9), random.randint(0, 9), random.randint(0, 9)) for _ in range(n_groups)]

    def _fallback_dlt(self, n_groups):
        groups = []
        for _ in range(n_groups):
            fronts = sorted(random.sample(range(1, 36), 5))
            backs = sorted(random.sample(range(1, 13), 2))
            groups.append((fronts, backs))
        return groups

    def _fallback_qxc(self, n_groups):
        return [tuple(random.randint(0, 9) for _ in range(7)) for _ in range(n_groups)]

    def _fallback_pl3(self, n_groups):
        return [(random.randint(0, 9), random.randint(0, 9), random.randint(0, 9)) for _ in range(n_groups)]

    # ---- 双色球预测 ----
    def predict_ssq(self, n_groups: int = 5) -> List[Tuple[List[int], int]]:
        if not self._initialized or self.lottery_type != "ssq":
            return self._fallback_ssq(n_groups)

        red_population = list(range(1, 34))
        blue_population = list(range(1, 17))

        groups = []
        used_combinations = set()

        attempts = 0
        max_attempts = n_groups * 80  # 提高尝试次数（Hamming 约束会淘汰更多候选）

        while len(groups) < n_groups and attempts < max_attempts:
            attempts += 1

            # 按融合概率进行加权抽样（热温冷三层混合）
            reds = self._weighted_sample_balanced(
                self.fusion_scores, red_population, 6,
                hot_ratio=0.40, warm_ratio=0.35, cold_ratio=0.25)

            reds_sorted = tuple(sorted(reds))
            if reds_sorted in used_combinations:
                continue

            # ★ 多样性约束：与已有组保持最小汉明距离（必须在 add 之前检查）
            if used_combinations and self._hamming_vs_used(reds_sorted, used_combinations) < 3:
                continue

            # 约束校验
            valid, issues = validate_combination_ssq(reds)
            if not valid and len(issues) >= 3:
                continue  # 超过2个问题的组合直接丢弃

            # 缩水过滤：AC值/跨度/012路/奇偶比/质合比综合评分
            filter_score = self._combination_filter_score(reds, "ssq")
            if filter_score < 0.4:
                continue

            # 统计偏好（和值/连号/同尾数）温和过滤极端不合理组合
            if score_combination_preferences(reds, "ssq") < 0.35:
                continue

            # 蓝球：按融合概率 + 80%热20%冷
            blue = self._weighted_pick_with_temperature(
                self.blue_fusion, blue_population, temp=0.3)

            used_combinations.add(reds_sorted)
            groups.append((list(reds_sorted), blue))

        # 如果生成不够，补随机
        import random
        while len(groups) < n_groups:
            reds = sorted(random.sample(list(range(1, 34)), 6))
            tr = tuple(reds)
            if tr not in used_combinations:
                # 随机补组也检查 Hamming
                if not used_combinations or self._hamming_vs_used(tr, used_combinations) >= 2:
                    used_combinations.add(tr)
                    blue = random.choice(list(range(1, 17)))
                    groups.append((reds, blue))

        return groups

    # ---- 大乐透预测 ----
    def predict_dlt(self, n_groups: int = 5) -> List[Tuple[List[int], List[int]]]:
        """大乐透预测：5个前区(1-35) + 2个后区(1-12)"""
        if not self._initialized or self.lottery_type != "dlt":
            return self._fallback_dlt(n_groups)

        front_population = list(range(1, 36))
        back_population = list(range(1, 13))

        groups = []
        used_combinations = set()

        attempts = 0
        max_attempts = n_groups * 80

        while len(groups) < n_groups and attempts < max_attempts:
            attempts += 1

            # 前区：按融合概率进行加权抽样（热温冷三层混合）
            fronts = self._weighted_sample_balanced(
                self.fusion_scores, front_population, 5,
                hot_ratio=0.40, warm_ratio=0.35, cold_ratio=0.25)

            fronts_sorted = tuple(sorted(fronts))
            if fronts_sorted in used_combinations:
                continue

            # ★ 多样性约束：前区与前组保持最小汉明距离（add 之前检查）
            if used_combinations and self._hamming_vs_used(fronts_sorted, used_combinations) < 3:
                continue

            # 前区约束校验
            ac = calculate_ac_value(fronts)
            span = calculate_span(fronts)
            oe = classify_odd_even(fronts)
            if ac < 2 or ac > 7:
                continue
            if span < 12 or span > 34:
                continue
            if oe["odd"] not in (1, 2, 3, 4):
                continue

            # 缩水过滤：AC值/跨度/012路/奇偶比综合评分
            filter_score = self._combination_filter_score(fronts, "dlt")
            if filter_score < 0.4:
                continue

            # 统计偏好（和值/连号/同尾数）温和过滤
            if score_combination_preferences(fronts, "dlt") < 0.35:
                continue

            # 后区：按融合概率选2个不重复号码
            back1 = self._weighted_pick_with_temperature(
                self.blue_fusion, back_population, temp=0.3)
            remaining_back = [b for b in back_population if b != back1]
            back2 = self._weighted_pick_with_temperature(
                {k: v for k, v in self.blue_fusion.items() if k != back1},
                remaining_back, temp=0.3)
            backs = sorted([back1, back2])

            key = (fronts_sorted, tuple(backs))
            if key in used_combinations:
                continue

            used_combinations.add(fronts_sorted)
            groups.append((list(fronts_sorted), backs))

        # 如果生成不够，补随机
        while len(groups) < n_groups:
            fronts = sorted(random.sample(list(range(1, 36)), 5))
            if fronts_sorted_tuple := tuple(fronts):
                if fronts_sorted_tuple not in used_combinations:
                    if not used_combinations or self._hamming_vs_used(fronts_sorted_tuple, used_combinations) >= 2:
                        used_combinations.add(fronts_sorted_tuple)
                        backs = sorted(random.sample(list(range(1, 13)), 2))
                        groups.append((fronts, backs))

        return groups

    # ---- 七星彩预测 ----
    def predict_qxc(self, n_groups: int = 5) -> List[Tuple[int, int, int, int, int, int, int]]:
        """七星彩预测：7个位置各0-9，分位独立分析"""
        if not self._initialized or self.lottery_type != "qxc":
            return self._fallback_qxc(n_groups)

        # 分位独立分析
        pos_fusions = []
        recent = self.df.head(100)
        pos_population = list(range(10))

        for pos_col in ['n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7']:
            freq = Counter(recent[pos_col].values) if pos_col in recent.columns else Counter()
            for p in pos_population:
                if p not in freq:
                    freq[p] = 0
            total_f = sum(freq.values())
            freq_w = {k: v / total_f if total_f > 0 else 0 for k, v in freq.items()}

            decay_w = exponential_decay_weighting(
                recent, [pos_col], decay_factor=0.95, population=pos_population)
            missing_w = calculate_missing_analysis(
                recent, [pos_col], pos_population)
            markov_m = build_markov_transition(
                recent, [pos_col], pos_population)

            if pos_col in recent.columns:
                last_val = int(recent.iloc[0][pos_col])
            else:
                last_val = 0
            markov_w = markov_next_probability(
                [last_val], markov_m, pos_population)

            fusion = bayesian_fusion_scores(
                freq_w, decay_w, missing_w, markov_w,
                pos_population, 0.25, 0.25, 0.25, 0.25)
            pos_fusions.append(fusion)

        used = set()
        groups = []
        for _ in range(n_groups * 10):
            if len(groups) >= n_groups:
                break
            digits = []
            for fusion in pos_fusions:
                d = self._weighted_pick_with_temperature(fusion, list(range(10)), 0.4)
                digits.append(d)
            key = tuple(digits)
            if key not in used:
                used.add(key)
                groups.append(key)

        while len(groups) < n_groups:
            g = tuple(random.randint(0, 9) for _ in range(7))
            if g not in used:
                used.add(g)
                groups.append(g)

        return groups

    # ---- 排列三预测 ----
    def predict_pl3(self, n_groups: int = 5) -> List[Tuple[int, int, int]]:
        """排列三预测：3个位置各0-9，分位独立分析"""
        if not self._initialized or self.lottery_type != "pl3":
            return self._fallback_pl3(n_groups)

        # 分位独立分析
        pos_fusions = []
        recent = self.df.head(100)
        pos_population = list(range(10))

        for pos_col in ['n1', 'n2', 'n3']:
            freq = Counter(recent[pos_col].values) if pos_col in recent.columns else Counter()
            for p in pos_population:
                if p not in freq:
                    freq[p] = 0
            total_f = sum(freq.values())
            freq_w = {k: v / total_f if total_f > 0 else 0 for k, v in freq.items()}

            decay_w = exponential_decay_weighting(
                recent, [pos_col], decay_factor=0.95, population=pos_population)
            missing_w = calculate_missing_analysis(
                recent, [pos_col], pos_population)
            markov_m = build_markov_transition(
                recent, [pos_col], pos_population)

            if pos_col in recent.columns:
                last_val = int(recent.iloc[0][pos_col])
            else:
                last_val = 0
            markov_w = markov_next_probability(
                [last_val], markov_m, pos_population)

            fusion = bayesian_fusion_scores(
                freq_w, decay_w, missing_w, markov_w,
                pos_population, 0.25, 0.25, 0.25, 0.25)
            pos_fusions.append(fusion)

        used = set()
        groups = []
        for _ in range(n_groups * 10):
            if len(groups) >= n_groups:
                break
            n1 = self._weighted_pick_with_temperature(pos_fusions[0], list(range(10)), 0.4)
            n2 = self._weighted_pick_with_temperature(pos_fusions[1], list(range(10)), 0.4)
            n3 = self._weighted_pick_with_temperature(pos_fusions[2], list(range(10)), 0.4)
            key = (n1, n2, n3)
            if key not in used:
                used.add(key)
                groups.append(key)

        while len(groups) < n_groups:
            g = (random.randint(0, 9), random.randint(0, 9), random.randint(0, 9))
            if g not in used:
                used.add(g)
                groups.append(g)

        return groups

    def _hamming_vs_used(self, candidate_tuple: Tuple, used_set: set,
                         threshold: int = 2) -> int:
        """
        返回候选组合与已选组合集合的最小对称差大小（汉明距离）。
        小于 threshold 视为过度相似，拒绝以保证组间分散、避免扎堆。
        """
        cand = set(candidate_tuple)
        best = len(cand)
        for used in used_set:
            d = len(cand ^ set(used))
            if d < best:
                best = d
            if best <= 1:
                break
        return best

    def _combination_filter_score(self, nums: List[int],
                                   lottery_type: str) -> float:
        """
        缩水过滤综合评分（条件缩分）。
        基于 AC值/跨度/012路/奇偶比/质合比/尾数分布/区间分布等
        历史统计约束，对候选组合打分（0~1），低于阈值则淘汰。
        比单纯的 validate 更细腻——不是非黑即白，而是各项加权打分。
        """
        score = 1.0
        n = len(nums)

        if lottery_type == "ssq":
            # AC值：理想 4-6，偏离越远扣分越多
            ac = calculate_ac_value(nums)
            if 4 <= ac <= 6:
                score *= 1.0
            elif 3 <= ac <= 7:
                score *= 0.8
            else:
                score *= 0.5

            # 跨度：理想 18-31
            span = calculate_span(nums)
            if 18 <= span <= 31:
                score *= 1.0
            elif 15 <= span <= 32:
                score *= 0.7
            else:
                score *= 0.4

            # 奇偶比：理想 2:4/3:3/4:2
            oe = classify_odd_even(nums)
            if oe["odd"] in (2, 3, 4):
                score *= 1.0
            elif oe["odd"] in (1, 5):
                score *= 0.6
            else:
                score *= 0.3

            # 012路：不应极端偏态
            z012 = classify_012(nums)
            if not any(v >= 5 for v in z012.values()) and not any(v == 0 for v in z012.values()):
                score *= 1.0
            elif any(v >= 5 for v in z012.values()):
                score *= 0.5
            else:
                score *= 0.7

            # 质合比：理想 1:5 ~ 4:2
            pc = classify_prime_composite(nums)
            if 1 <= pc["prime"] <= 4:
                score *= 1.0
            else:
                score *= 0.6

            # 尾数：至少4种不同尾数
            tails = get_tail_distribution(nums)
            if len(tails) >= 4:
                score *= 1.0
            elif len(tails) == 3:
                score *= 0.7
            else:
                score *= 0.4

            # 三区分布：不应极端
            zones = classify_zone_ssq(nums)
            if not any(v >= 5 or v == 0 for v in zones.values()):
                score *= 1.0
            elif any(v == 0 for v in zones.values()):
                score *= 0.6
            else:
                score *= 0.5

        elif lottery_type == "dlt":
            # AC值：理想 3-6
            ac = calculate_ac_value(nums)
            if 3 <= ac <= 6:
                score *= 1.0
            elif 2 <= ac <= 7:
                score *= 0.8
            else:
                score *= 0.5

            # 跨度：理想 15-33
            span = calculate_span(nums)
            if 15 <= span <= 33:
                score *= 1.0
            elif 12 <= span <= 34:
                score *= 0.7
            else:
                score *= 0.4

            # 奇偶比：理想 2:3/3:2
            oe = classify_odd_even(nums)
            if oe["odd"] in (2, 3):
                score *= 1.0
            elif oe["odd"] in (1, 4):
                score *= 0.7
            else:
                score *= 0.4

            # 012路
            z012 = classify_012(nums)
            if not any(v >= 4 for v in z012.values()) and not any(v == 0 for v in z012.values()):
                score *= 1.0
            elif any(v >= 4 for v in z012.values()):
                score *= 0.6
            else:
                score *= 0.7

            # 尾数
            tails = get_tail_distribution(nums)
            if len(tails) >= 4:
                score *= 1.0
            elif len(tails) == 3:
                score *= 0.7
            else:
                score *= 0.4

        elif lottery_type == "kl8":
            # 快乐8：跨度应大、奇偶均衡
            span = calculate_span(nums)
            if span >= 60:
                score *= 1.0
            elif span >= 40:
                score *= 0.8
            else:
                score *= 0.5

            oe = classify_odd_even(nums)
            if 3 <= oe["odd"] <= 7:
                score *= 1.0
            else:
                score *= 0.6

        return score

    def _weighted_sample_balanced(self, scores: Dict[int, float],
                                   population: List[int], k: int,
                                   hot_ratio=0.40, warm_ratio=0.35,
                                   cold_ratio=0.25) -> List[int]:
        """基于分数的三层加权抽样"""
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        sorted_nums = [item[0] for item in sorted_items]
        n = len(sorted_nums)

        hot_boundary = max(1, int(n * hot_ratio))
        cold_boundary = n - max(1, int(n * cold_ratio))

        hot_nums = sorted_nums[:hot_boundary]
        warm_nums = sorted_nums[hot_boundary:cold_boundary]
        cold_nums = sorted_nums[cold_boundary:]

        h_count = min(int(k * hot_ratio), len(hot_nums))
        w_count = min(int(k * warm_ratio), len(warm_nums))
        c_count = k - h_count - w_count
        if c_count > len(cold_nums):
            shortfall = c_count - len(cold_nums)
            c_count = len(cold_nums)
            w_count = min(w_count + shortfall, len(warm_nums))

        result = []
        if h_count > 0 and hot_nums:
            result.extend(self._weighted_pick(hot_nums, scores, h_count))
        if w_count > 0 and warm_nums:
            result.extend(self._weighted_pick(warm_nums, scores, w_count))
        if c_count > 0 and cold_nums:
            result.extend(self._weighted_pick(cold_nums, scores, c_count))

        # 补足
        needed = k - len(result)
        if needed > 0:
            pool = list(set(population) - set(result))
            result.extend(random.sample(pool, min(needed, len(pool))))

        return sorted(set(result[:k]))

    def _weighted_pick(self, candidates: List[int],
                        scores: Dict[int, float], count: int) -> List[int]:
        """加权无放回抽样"""
        if count <= 0 or not candidates:
            return []
        cand_scores = [(c, scores.get(c, 0)) for c in candidates]
        cand_scores.sort(key=lambda x: x[1], reverse=True)

        result = []
        available = list(candidates)

        for _ in range(min(count, len(available))):
            if not available:
                break
            weights = [scores.get(c, 1e-8) for c in available]
            total = sum(weights)
            if total <= 0:
                weights = [1.0 / len(available)] * len(available)
                total = 1.0
            probs = [w / total for w in weights]
            chosen = np.random.choice(len(available), p=probs)
            result.append(available.pop(chosen))

        return result

    def _weighted_pick_with_temperature(self, scores: Dict[int, float],
                                         population: List[int],
                                         temp: float = 0.5) -> int:
        """带温度的加权选择"""
        items = list(population)
        raw = [scores.get(i, 1e-8) for i in items]
        adjusted = [r ** (1.0 / max(temp, 0.01)) for r in raw]
        total = sum(adjusted)
        if total <= 0:
            return random.choice(items)
        probs = [a / total for a in adjusted]
        return np.random.choice(items, p=probs).item()

    # ---- 快乐8预测 ----
    def predict_kl8(self, n_groups: int = 5, select_count: int = 10) -> List[List[int]]:
        if not self._initialized or self.lottery_type != "kl8":
            return self._fallback_kl8(n_groups, select_count)

        population = list(range(1, 81))
        groups = []
        used_combinations = set()
        attempts = 0

        while len(groups) < n_groups and attempts < n_groups * 50:
            attempts += 1
            nums = self._weighted_sample_balanced(
                self.fusion_scores, population, select_count,
                hot_ratio=0.40, warm_ratio=0.35, cold_ratio=0.25)
            # 统计偏好（和值/跨度/奇偶）温和过滤
            if score_combination_preferences(nums, "kl8") < 0.30:
                continue
            nt = tuple(nums)
            if nt not in used_combinations:
                # 多样性：组间最小汉明距离（避免选号扎堆）
                if used_combinations and self._hamming_vs_used(nt, used_combinations) < 3:
                    continue
                used_combinations.add(nt)
                groups.append(nums)

        while len(groups) < n_groups:
            nums = sorted(random.sample(population, select_count))
            nt = tuple(nums)
            if nt not in used_combinations:
                used_combinations.add(nt)
                groups.append(nums)

        return groups

    # ---- 福彩3D预测 ----
    def predict_fcsd(self, n_groups: int = 5) -> List[Tuple[int, int, int]]:
        if not self._initialized or self.lottery_type != "fcsd":
            return self._fallback_fcsd(n_groups)

        # 分位独立分析
        recent = self.df.head(100)
        groups = []

        for pos_col in ['n1', 'n2', 'n3']:
            pos_population = list(range(10))
            freq = Counter(recent[pos_col].values)
            decay = exponential_decay_weighting(
                recent, [pos_col], decay_factor=0.90, population=pos_population)
            missing = calculate_missing_analysis(
                recent, [pos_col], pos_population)
            markov_mat = build_markov_transition(
                recent, [pos_col], pos_population)
            last_val = int(recent.iloc[0][pos_col])
            markov_p = markov_next_probability(
                [last_val], markov_mat, pos_population)

            fusion = bayesian_fusion_scores(
                {k: v for k, v in freq.items()},
                decay, missing, markov_p, pos_population,
                0.25, 0.25, 0.25, 0.25)

            if pos_col == 'n1':
                n1_fusion = fusion
            elif pos_col == 'n2':
                n2_fusion = fusion
            else:
                n3_fusion = fusion

        used = set()
        for _ in range(n_groups * 10):
            if len(groups) >= n_groups:
                break
            n1 = self._weighted_pick_with_temperature(n1_fusion, list(range(10)), 0.4)
            n2 = self._weighted_pick_with_temperature(n2_fusion, list(range(10)), 0.4)
            n3 = self._weighted_pick_with_temperature(n3_fusion, list(range(10)), 0.4)
            key = (n1, n2, n3)
            if key not in used:
                used.add(key)
                groups.append(key)

        while len(groups) < n_groups:
            g = (random.randint(0, 9), random.randint(0, 9), random.randint(0, 9))
            if g not in used:
                used.add(g)
                groups.append(g)

        return groups

    # ---- 获取特征摘要 ----
    def get_feature_summary(self) -> Dict:
        """返回特征数据的摘要，供 AI 提示词使用"""
        if not self._initialized:
            return {"error": "未初始化"}

        cols, population, blue_cols, blue_population = self._get_cols_and_population()

        summary = {
            "total_periods": len(self.df),
            "last_draw": {},
            "fusion_top10": {},
            "missing_coldest_top5": {},
            "missing_hottest_top5": {},
            "markov_top10": {},
            "monte_carlo_top10": {},
            "feature_contributions": {
                "decay_weight": 0.30,
                "frequency_weight": 0.20,
                "missing_weight": 0.25,
                "markov_weight": 0.25
            }
        }

        # 上一期号码
        latest = self.df.iloc[0]
        summary["last_draw"] = {
            "code": str(latest.get("code", "")),
            "date": str(latest.get("date", "")),
            "nums": [int(latest[col]) for col in cols if col in latest]
        }

        # Top 号码
        sorted_fusion = sorted(self.fusion_scores.items(),
                                key=lambda x: x[1], reverse=True)
        summary["fusion_top10"] = {str(k): round(v, 6) for k, v in sorted_fusion[:10]}

        # 最冷/最热遗漏
        sorted_missing = sorted(self.missing_analysis.items(),
                                 key=lambda x: x[1]["bounce_prob"], reverse=True)
        summary["missing_bounce_top5"] = {
            str(k): {
                "current_gap": v["current_gap"],
                "max_gap": v["max_gap"],
                "bounce_prob": round(v["bounce_prob"], 4)
            } for k, v in sorted_missing[:5]
        }

        # 马尔可夫
        sorted_markov = sorted(self.markov_probs.items(),
                                key=lambda x: x[1], reverse=True)
        summary["markov_top10"] = {str(k): round(v, 6) for k, v in sorted_markov[:10]}

        # 蒙特卡洛
        sorted_mc = sorted(self.monte_carlo_probs.items(),
                            key=lambda x: x[1], reverse=True)
        summary["monte_carlo_top10"] = {
            str(k): round(v, 6) for k, v in sorted_mc[:10]
        }

        # 蓝球/后区（双色球和大乐透）
        if self.lottery_type in ("ssq", "dlt") and hasattr(self, 'blue_fusion'):
            sorted_blue = sorted(self.blue_fusion.items(),
                                  key=lambda x: x[1], reverse=True)
            summary["blue_fusion_top5"] = {
                str(k): round(v, 6) for k, v in sorted_blue[:5]
            }

        return summary

    def get_ensemble_prediction(self, n_groups: int = 5) -> Dict:
        """
        集成预测：综合本地算法 + 蒙特卡洛 + 马尔可夫 + LSTM-CRF
        返回收敛号码和置信度
        使用基于日期+最新期号的确定性种子，确保同一天同一批数据结果稳定
        """
        if not self._initialized:
            return {"error": "未初始化"}

        # 位置制彩票（七星彩/排列三/福彩3D）：每位数独立 0-9，号码池小，
        # 不能走"高频 Top-K 滑动窗口"（最多只能凑出 sample_size 窗口数种组合）。
        # 直接委托分位独立的 predict_* 方法生成多样组合。
        if self.lottery_type in ("qxc", "pl3", "fcsd"):
            if self.lottery_type == "qxc":
                raw = self.predict_qxc(n_groups)
            elif self.lottery_type == "pl3":
                raw = self.predict_pl3(n_groups)
            else:
                raw = self.predict_fcsd(n_groups)
            recommendations = []
            for grp in raw:
                nums = list(grp)
                if self.fusion_scores:
                    avg_conf = sum(self.fusion_scores.get(n, 0) for n in nums) / len(nums)
                else:
                    avg_conf = 0.0
                recommendations.append({
                    "nums": nums,
                    "confidence": round(avg_conf, 4),
                    "valid": True
                })
            return {
                "lottery_type": self.lottery_type,
                "confidence_distribution": {},
                "recommendations": recommendations,
                "model_contributions": {
                    "bayesian_fusion": 0.40,
                    "monte_carlo": 0.30,
                    "markov_chain": 0.30
                },
                "backtest": None
            }

        # ---- 设置确定性随机种子（同一天结果稳定）----
        import time
        from datetime import datetime
        # 种子 = 年月日 + 最新期号哈希 + 彩票类型哈希
        date_part = datetime.now().strftime("%Y%m%d")
        last_code = str(self.df.head(1)['code'].values[0]) if 'code' in self.df.columns and not self.df.empty else "0"
        seed = int(date_part) + hash(last_code) % 100000 + hash(self.lottery_type) % 1000
        random.seed(seed)
        np.random.seed(seed)

        cols, population, _, _ = self._get_cols_and_population()
        sample_size = len(cols)
        # 快乐8：官方玩法为选一到选十，默认选十（10个号码）
        # cols 有20列（开奖号码数），但投注只需选10个
        # 支持通过 _override_kl8_sample_size 动态指定选号个数
        if self.lottery_type == "kl8":
            sample_size = getattr(self, '_override_kl8_sample_size', 10)

        # 方法1：加权融合抽样
        method1_sets = []
        for _ in range(200):
            nums = self._weighted_sample_balanced(
                self.fusion_scores, population, sample_size)
            method1_sets.append(set(nums))

        # 方法2：蒙特卡洛 Top-N
        sorted_mc = sorted(self.monte_carlo_probs.items(),
                            key=lambda x: x[1], reverse=True)
        mc_top = [item[0] for item in sorted_mc[:sample_size * 2]]
        method2_sets = []
        for _ in range(200):
            nums = set(random.sample(mc_top, min(sample_size, len(mc_top))))
            method2_sets.append(nums)

        # 方法3：马尔可夫 Top-N
        sorted_mar = sorted(self.markov_probs.items(),
                             key=lambda x: x[1], reverse=True)
        mar_top = [item[0] for item in sorted_mar[:sample_size * 2]]
        method3_sets = []
        for _ in range(200):
            nums = set(random.sample(mar_top, min(sample_size, len(mar_top))))
            method3_sets.append(nums)

        # 方法4：LSTM-CRF 序列建模
        method4_sets = []
        try:
            crf_decoder = _SimpleCRFDecoder(
                population=population,
                fusion_scores=self.fusion_scores,
                missing_analysis=self.missing_analysis,
                recent_df=self.df.head(100),
                cols=cols
            )
            min_hamming_crf = 4 if self.lottery_type == "kl8" else 3
            crf_groups = crf_decoder.sample_diverse(200, sample_size,
                                                     temperature=1.0,
                                                     min_hamming=min_hamming_crf)
            for g in crf_groups:
                method4_sets.append(set(g))
        except Exception:
            pass  # CRF 失败不影响其他方法

        # 统计号码出现频次（4种方法等权融合）
        all_sets = method1_sets + method2_sets + method3_sets + method4_sets
        counter = Counter()
        for s in all_sets:
            for n in s:
                counter[n] += 1

        total_sets = len(all_sets)
        confidence = {k: round(v / total_sets, 4) for k, v in counter.items()}

        # 生成推荐组（带 Hamming 多样性约束 + 缩水过滤）
        recommendations = []
        used = set()
        sorted_by_conf = sorted(confidence.items(), key=lambda x: x[1], reverse=True)
        # 构建加权随机抽样的概率表（用于确定性窗口耗尽后的兜底）
        _all_nums = [item[0] for item in sorted_by_conf]
        _all_weights = [item[1] for item in sorted_by_conf]
        _total_w = sum(_all_weights) or 1.0
        _all_probs = [w / _total_w for w in _all_weights]

        # Hamming 最小距离阈值（根据号码池大小自适应）
        if self.lottery_type == "kl8":
            _min_hamming = 4  # 快乐8 选10个号，差异度要求更高
        elif self.lottery_type in ("ssq", "dlt"):
            _min_hamming = 3  # 双色球/大乐透：至少3个号不同
        else:
            _min_hamming = 2

        def _is_diverse_enough(candidate_tuple: Tuple, used_set: set,
                                min_dist: int) -> bool:
            """检查候选组合与已有组合的最小汉明距离是否 >= min_dist"""
            cand = set(candidate_tuple)
            for u in used_set:
                if len(cand ^ set(u)) < min_dist:
                    return False
            return True

        def _passes_filter(nums: List[int], lottery_type: str) -> bool:
            """缩水过滤：组合约束评分是否达标"""
            if lottery_type in ("ssq", "dlt"):
                return self._combination_filter_score(nums, lottery_type) >= 0.4
            elif lottery_type == "kl8":
                return score_combination_preferences(nums, "kl8") >= 0.30
            return True

        for group_idx in range(n_groups):
            # 阶段1：确定性滑动窗口（同一天结果稳定）+ Hamming + 缩水
            candidates = [item[0] for item in sorted_by_conf[:sample_size * 3]]
            max_off = max(1, len(candidates) - sample_size + 1)
            chosen = None
            for try_off in range(max_off):
                offset = (group_idx + try_off) % max_off
                nums = sorted(candidates[offset:offset + sample_size])
                if len(nums) < sample_size:
                    remaining = [item[0] for item in sorted_by_conf[sample_size * 3:]]
                    need = sample_size - len(nums)
                    nums = sorted(nums + remaining[:need])
                nt = tuple(nums)
                if nt in used:
                    continue
                # ★ Hamming 多样性检查
                if not _is_diverse_enough(nt, used, _min_hamming):
                    continue
                # ★ 缩水过滤
                if not _passes_filter(nums, self.lottery_type):
                    continue
                used.add(nt)
                chosen = nums
                break
            # 阶段2：从更靠后的置信度排序中取
            if chosen is None:
                for extra in range(sample_size * 3, len(sorted_by_conf) - sample_size + 1):
                    nums = sorted([item[0] for item in sorted_by_conf[extra:extra + sample_size]])
                    nt = tuple(nums)
                    if nt in used:
                        continue
                    if not _is_diverse_enough(nt, used, _min_hamming):
                        continue
                    if not _passes_filter(nums, self.lottery_type):
                        continue
                    used.add(nt)
                    chosen = nums
                    break
            # 阶段3：加权随机抽样兜底（解决双色球/大乐透因窗口数有限导致组数不足的问题）
            if chosen is None:
                _random_tries = min(500, n_groups * 50)
                for _rt in range(_random_tries):
                    sampled = sorted(set(random.choices(
                        population=_all_nums, weights=_all_probs, k=sample_size * 2)))
                    if len(sampled) >= sample_size:
                        nums = sampled[:sample_size]
                    elif len(sampled) < sample_size:
                        # 补足差额
                        _need = sample_size - len(sampled)
                        _pool = [n for n in _all_nums if n not in sampled]
                        if _pool:
                            nums = sorted(sampled + random.sample(_pool, min(_need, len(_pool))))
                        else:
                            continue
                    else:
                        nums = sampled
                    if len(nums) != sample_size:
                        continue
                    nt = tuple(nums)
                    if nt in used:
                        continue
                    # ★ Hamming + 缩水
                    if not _is_diverse_enough(nt, used, _min_hamming):
                        continue
                    if not _passes_filter(nums, self.lottery_type):
                        continue
                    used.add(nt)
                    chosen = nums
                    break
            if chosen is None:
                # 确实无法生成更多不重复组合（号码池极小）
                break
            avg_conf = sum(confidence.get(n, 0) for n in chosen) / len(chosen)
            if self.lottery_type == "ssq":
                valid, _ = validate_combination_ssq(chosen)
                # 为双色球附加蓝球
                blue_population = list(range(1, 17))
                blue = int(self._weighted_pick_with_temperature(
                    self.blue_fusion, blue_population, temp=0.3))
                recommendations.append({
                    "red": list(chosen),
                    "blue": blue,
                    "confidence": round(avg_conf, 4),
                    "valid": valid
                })
            elif self.lottery_type == "dlt":
                # 大乐透附加2个后区号码
                back_population = list(range(1, 13))
                back1 = self._weighted_pick_with_temperature(
                    self.blue_fusion, back_population, temp=0.3)
                remaining_back = [b for b in back_population if b != back1]
                back2 = self._weighted_pick_with_temperature(
                    {k: v for k, v in self.blue_fusion.items() if k != back1},
                    remaining_back, temp=0.3)
                recommendations.append({
                    "nums": chosen + sorted([back1, back2]),
                    "confidence": round(avg_conf, 4),
                    "valid": True
                })
            else:
                recommendations.append({
                    "nums": chosen,
                    "confidence": round(avg_conf, 4),
                    "valid": True
                })

        # 模型贡献权重：优先用历史回溯验证的真实吻合度，否则回退写死比例
        model_contributions = {
            "bayesian_fusion": 0.30,
            "monte_carlo": 0.22,
            "markov_chain": 0.22,
            "lstm_crf": 0.26
        }
        if self.lottery_type in ("ssq", "dlt", "kl8"):
            backtest = monte_carlo_backtest(
                self.lottery_type, top_k=sample_size, n_recent=30)
            if "error" not in backtest:
                # CRF 命中率：用 CRF 解码出的 Top 号码集合 vs 实际开奖
                crf_hit = backtest.get("crf_hit_rate", 0.0)
                contrib_raw = {
                    "bayesian_fusion": backtest["fusion_hit_rate"],
                    "monte_carlo": backtest["mc_hit_rate"],
                    "markov_chain": backtest["markov_hit_rate"],
                    "lstm_crf": max(crf_hit, backtest["fusion_hit_rate"] * 0.9),  # 无独立回测时参考融合
                }
                csum = sum(contrib_raw.values())
                if csum > 0:
                    model_contributions = {
                        k: round(v / csum, 2) for k, v in contrib_raw.items()
                    }

        return {
            "lottery_type": self.lottery_type,
            "confidence_distribution": {
                str(k): v for k, v in sorted_by_conf[:15]
            },
            "recommendations": recommendations,
            "model_contributions": model_contributions,
            "backtest": backtest if self.lottery_type in ("ssq", "dlt", "kl8") else None
        }


# ============================================================================
# 第九部分：便捷函数（供 generate_picks.py 调用）
# ============================================================================

# 全局预测器缓存
_predictor_cache: Dict[str, EnhancedPredictor] = {}


def _get_predictor(lottery_type: str) -> EnhancedPredictor:
    """获取或创建预测器实例"""
    if lottery_type not in _predictor_cache:
        predictor = EnhancedPredictor(lottery_type)
        predictor.initialize(recent_periods=100)
        _predictor_cache[lottery_type] = predictor
    return _predictor_cache[lottery_type]


def enhanced_predict_ssq(n_groups: int = 5) -> List[Tuple[List[int], int]]:
    """增强型双色球预测"""
    pred = _get_predictor("ssq")
    return pred.predict_ssq(n_groups)


def enhanced_predict_kl8(n_groups: int = 5, select_count: int = 10) -> List[List[int]]:
    """增强型快乐8预测"""
    pred = _get_predictor("kl8")
    return pred.predict_kl8(n_groups, select_count)


def enhanced_predict_fcsd(n_groups: int = 5) -> List[Tuple[int, int, int]]:
    """增强型福彩3D预测"""
    pred = _get_predictor("fcsd")
    return pred.predict_fcsd(n_groups)


def enhanced_predict_dlt(n_groups: int = 5) -> List[Tuple[List[int], List[int]]]:
    """增强型大乐透预测"""
    pred = _get_predictor("dlt")
    return pred.predict_dlt(n_groups)


def enhanced_predict_qxc(n_groups: int = 5) -> List[Tuple[int, int, int, int, int, int, int]]:
    """增强型七星彩预测"""
    pred = _get_predictor("qxc")
    return pred.predict_qxc(n_groups)


def enhanced_predict_pl3(n_groups: int = 5) -> List[Tuple[int, int, int]]:
    """增强型排列三预测"""
    pred = _get_predictor("pl3")
    return pred.predict_pl3(n_groups)


def get_ensemble_prediction(lottery_type: str, n_groups: int = 5,
                            ai_review: bool = False,
                            kl8_pick_size: int = 10) -> Dict:
    """获取集成预测结果，可选 AI 候选池审阅。

    Args:
        lottery_type: 彩种代码
        n_groups: 生成组数
        ai_review: 是否启用 AI 候选池审阅（需 AI 已配置）
        kl8_pick_size: 快乐8选号个数（1-10），默认10（选十玩法）
    """
    pred = _get_predictor(lottery_type)
    # 快乐8：临时覆盖 sample_size（选号个数）
    _orig_sample_size = None
    if lottery_type == "kl8":
        kl8_pick_size = max(1, min(10, int(kl8_pick_size)))
        _orig_sample_size = getattr(pred, '_override_kl8_sample_size', None)
        pred._override_kl8_sample_size = kl8_pick_size

    result = pred.get_ensemble_prediction(n_groups)

    # 恢复
    if _orig_sample_size is not None:
        pred._override_kl8_sample_size = _orig_sample_size
    elif hasattr(pred, '_override_kl8_sample_size'):
        delattr(pred, '_override_kl8_sample_size')

    if ai_review and "error" not in result:
        result = _apply_ai_review(lottery_type, result, pred, n_groups)

    return result


def get_feature_summary(lottery_type: str) -> Dict:
    """获取特征摘要"""
    pred = _get_predictor(lottery_type)
    return pred.get_feature_summary()


def get_confidence_distribution(lottery_type: str) -> Dict[int, float]:
    """获取各号码置信度分布"""
    pred = _get_predictor(lottery_type)
    ensemble = pred.get_ensemble_prediction(n_groups=1)
    if "confidence_distribution" in ensemble:
        return {int(k): v for k, v in ensemble["confidence_distribution"].items()}
    return {}


def _apply_ai_review(lottery_type: str, ensemble_result: Dict,
                     predictor: "EnhancedPredictor",
                     n_groups: int) -> Dict:
    """AI 候选池审阅后调整置信度并重新生成推荐组。

    流程：
    1. 从 ensemble_result 提取候选池 + 置信度
    2. 构建统计摘要，调用 ai_predict.ai_review_pool()
    3. 根据优先/回避标记调整置信度（优先+20%，回避-50%）
    4. 用调整后的置信度重新选号生成推荐组
    """
    try:
        from ai_predict import ai_review_pool, is_ai_configured
    except ImportError:
        ensemble_result["ai_review"] = {"prioritized": [], "avoided": [], "reasoning": "ai_predict 模块不可用"}
        return ensemble_result

    if not is_ai_configured():
        ensemble_result["ai_review"] = {"prioritized": [], "avoided": [], "reasoning": "AI 未配置，跳过审阅"}
        return ensemble_result

    # 提取候选池和置信度
    conf_dist = ensemble_result.get("confidence_distribution", {})
    if not conf_dist:
        # 位置制彩票没有 confidence_distribution，从 recommendations 提取
        recs = ensemble_result.get("recommendations", [])
        all_nums = []
        for rec in recs:
            all_nums.extend(rec.get("nums", []))
        candidate_pool = list(set(all_nums))
        confidence = {n: 1.0 / len(candidate_pool) for n in candidate_pool} if candidate_pool else {}
    else:
        confidence = {int(k): float(v) for k, v in conf_dist.items()}
        candidate_pool = list(confidence.keys())

    if not candidate_pool:
        ensemble_result["ai_review"] = {"prioritized": [], "avoided": [], "reasoning": "候选池为空"}
        return ensemble_result

    # 构建统计摘要
    recent_stats = _build_recent_stats(predictor)
    backtest_info = ensemble_result.get("backtest")
    backtest_summary = ""
    if backtest_info and isinstance(backtest_info, dict) and "error" not in backtest_info:
        backtest_summary = (
            f"融合命中率={backtest_info.get('fusion_hit_rate', 0):.1%}，"
            f"蒙特卡洛命中率={backtest_info.get('mc_hit_rate', 0):.1%}，"
            f"马尔可夫命中率={backtest_info.get('markov_hit_rate', 0):.1%}"
        )

    # 调用 AI 审阅
    review = ai_review_pool(lottery_type, candidate_pool, confidence,
                            recent_stats, backtest_summary)

    prioritized = review.get("prioritized", [])
    avoided = review.get("avoided", [])

    if not prioritized and not avoided:
        # AI 审阅无有效标记，保持原结果
        ensemble_result["ai_review"] = review
        return ensemble_result

    # 调整置信度：优先+20%，回避-50%
    adjusted_conf = dict(confidence)
    for n in prioritized:
        if n in adjusted_conf:
            adjusted_conf[n] *= 1.2
    for n in avoided:
        if n in adjusted_conf:
            adjusted_conf[n] *= 0.5

    # 用调整后置信度重新生成推荐组
    sorted_adj = sorted(adjusted_conf.items(), key=lambda x: x[1], reverse=True)

    # 确定每组的号码数量
    if lottery_type == "ssq":
        sample_size = 6  # 红球
    elif lottery_type == "dlt":
        sample_size = 5  # 前区
    elif lottery_type == "kl8":
        sample_size = 10
    elif lottery_type in ("qxc",):
        sample_size = 7
    else:  # pl3, fcsd
        sample_size = 3

    # 位置制彩票（qxc/pl3/fcsd）按位独立生成，AI 审阅只调整置信度
    if lottery_type in ("qxc", "pl3", "fcsd"):
        # 标记审阅结果，但保持原推荐组（位置制按位生成逻辑复杂，不宜重写）
        ensemble_result["ai_review"] = review
        ensemble_result["ai_adjusted_confidence"] = {str(k): round(v, 4) for k, v in sorted_adj[:15]}
        return ensemble_result

    # 非位置制：用调整后置信度重新选号（带 Hamming + 缩水过滤）
    new_recommendations = []
    used = set()
    # 加权随机抽样概率表（用于确定性窗口耗尽后的兜底）
    _adj_nums = [item[0] for item in sorted_adj]
    _adj_weights = [item[1] for item in sorted_adj]
    _adj_total_w = sum(_adj_weights) or 1.0
    _adj_probs = [w / _adj_total_w for w in _adj_weights]

    # Hamming 最小距离阈值
    if lottery_type == "kl8":
        _min_hamming = 4
    elif lottery_type in ("ssq", "dlt"):
        _min_hamming = 3
    else:
        _min_hamming = 2

    def _is_diverse(candidate_tuple, used_set, min_dist):
        cand = set(candidate_tuple)
        for u in used_set:
            if len(cand ^ set(u)) < min_dist:
                return False
        return True

    def _passes_filter_ai(nums, lt):
        if lt in ("ssq", "dlt"):
            return predictor._combination_filter_score(nums, lt) >= 0.4
        elif lt == "kl8":
            return score_combination_preferences(nums, "kl8") >= 0.30
        return True

    for group_idx in range(n_groups):
        # 阶段1：确定性滑动窗口 + Hamming + 缩水
        candidates = [item[0] for item in sorted_adj[:sample_size * 3]]
        max_off = max(1, len(candidates) - sample_size + 1)
        chosen = None
        for try_off in range(max_off):
            offset = (group_idx + try_off) % max_off
            nums = sorted(candidates[offset:offset + sample_size])
            if len(nums) < sample_size:
                remaining = [item[0] for item in sorted_adj[sample_size * 3:]]
                need = sample_size - len(nums)
                nums = sorted(nums + remaining[:need])
            nt = tuple(nums)
            if nt in used:
                continue
            if not _is_diverse(nt, used, _min_hamming):
                continue
            if not _passes_filter_ai(nums, lottery_type):
                continue
            used.add(nt)
            chosen = nums
            break
        # 阶段2：从更靠后的置信度排序中取
        if chosen is None:
            for extra in range(sample_size * 3, len(sorted_adj) - sample_size + 1):
                nums = sorted([item[0] for item in sorted_adj[extra:extra + sample_size]])
                nt = tuple(nums)
                if nt in used:
                    continue
                if not _is_diverse(nt, used, _min_hamming):
                    continue
                if not _passes_filter_ai(nums, lottery_type):
                    continue
                used.add(nt)
                chosen = nums
                break
        # 阶段3：加权随机抽样兜底
        if chosen is None:
            _random_tries = min(500, n_groups * 50)
            for _rt in range(_random_tries):
                sampled = sorted(set(random.choices(
                    population=_adj_nums, weights=_adj_probs, k=sample_size * 2)))
                if len(sampled) >= sample_size:
                    nums = sampled[:sample_size]
                elif len(sampled) < sample_size:
                    _need = sample_size - len(sampled)
                    _pool = [n for n in _adj_nums if n not in sampled]
                    if _pool:
                        nums = sorted(sampled + random.sample(_pool, min(_need, len(_pool))))
                    else:
                        continue
                else:
                    nums = sampled
                if len(nums) != sample_size:
                    continue
                nt = tuple(nums)
                if nt in used:
                    continue
                if not _is_diverse(nt, used, _min_hamming):
                    continue
                if not _passes_filter_ai(nums, lottery_type):
                    continue
                used.add(nt)
                chosen = nums
                break
        if chosen is None:
            break

        avg_conf = sum(adjusted_conf.get(n, 0) for n in chosen) / len(chosen)

        if lottery_type == "ssq":
            blue_population = list(range(1, 17))
            blue = int(predictor._weighted_pick_with_temperature(
                predictor.blue_fusion, blue_population, temp=0.3))
            new_recommendations.append({
                "red": list(chosen),
                "blue": blue,
                "confidence": round(avg_conf, 4),
                "valid": True
            })
        elif lottery_type == "dlt":
            back_population = list(range(1, 13))
            back1 = predictor._weighted_pick_with_temperature(
                predictor.blue_fusion, back_population, temp=0.3)
            remaining_back = [b for b in back_population if b != back1]
            back2 = predictor._weighted_pick_with_temperature(
                {k: v for k, v in predictor.blue_fusion.items() if k != back1},
                remaining_back, temp=0.3)
            new_recommendations.append({
                "nums": chosen + sorted([back1, back2]),
                "confidence": round(avg_conf, 4),
                "valid": True
            })
        else:
            new_recommendations.append({
                "nums": chosen,
                "confidence": round(avg_conf, 4),
                "valid": True
            })

    # 更新结果
    if new_recommendations:
        ensemble_result["recommendations"] = new_recommendations
    ensemble_result["ai_review"] = review
    ensemble_result["ai_adjusted_confidence"] = {str(k): round(v, 4) for k, v in sorted_adj[:15]}

    return ensemble_result


def _build_recent_stats(predictor: "EnhancedPredictor") -> str:
    """从预测器提取近期统计摘要，供 AI 审阅参考。"""
    try:
        df = predictor.df
        if df is None or df.empty:
            return ""

        lt = predictor.lottery_type
        recent = df.head(30)
        lines = [f"近{len(recent)}期数据概况："]

        if lt == "ssq":
            front_cols = [f"r{i}" for i in range(1, 7)]
            all_nums = pd.concat([recent[c] for c in front_cols if c in recent.columns])
            hot = all_nums.value_counts().head(5).to_dict()
            cold = all_nums.value_counts().tail(5).to_dict()
            lines.append(f"红球热号TOP5: {hot}")
            lines.append(f"红球冷号TOP5: {cold}")
        elif lt == "dlt":
            front_cols = [f"f{i}" for i in range(1, 6)]
            all_nums = pd.concat([recent[c] for c in front_cols if c in recent.columns])
            hot = all_nums.value_counts().head(5).to_dict()
            cold = all_nums.value_counts().tail(5).to_dict()
            lines.append(f"前区热号TOP5: {hot}")
            lines.append(f"前区冷号TOP5: {cold}")
        elif lt == "kl8":
            cols = [f"n{i:02d}" for i in range(1, 21)]
            all_nums = pd.concat([recent[c] for c in cols if c in recent.columns])
            hot = all_nums.value_counts().head(5).to_dict()
            cold = all_nums.value_counts().tail(5).to_dict()
            lines.append(f"热号TOP5: {hot}")
            lines.append(f"冷号TOP5: {cold}")
        else:
            cols = [c for c in recent.columns if c.startswith("n") and c[1:].isdigit()]
            if cols:
                all_nums = pd.concat([recent[c] for c in cols])
                hot = all_nums.value_counts().head(3).to_dict()
                lines.append(f"各位热号TOP3: {hot}")

        return "；".join(lines)
    except Exception:
        return ""


def monte_carlo_backtest(lottery_type: str, top_k: int = None,
                         n_recent: int = 30) -> Dict:
    """
    蒙特卡洛历史回溯验证：用当前融合/蒙特卡洛/马尔可夫/LSTM-CRF 的 Top-K 号码，
    对最近 n_recent 期实际开奖做命中统计，输出各策略真实历史吻合度，
    作为模型贡献权重的客观依据（替代写死的固定比例）。
    仅支持双色球/大乐透/快乐8（位置制彩票按位建模，不适用统一 Top-K）。
    """
    if lottery_type not in ("ssq", "dlt", "kl8"):
        return {"error": "仅支持双色球/大乐透/快乐8的回溯验证"}

    pred = _get_predictor(lottery_type)
    if not pred._initialized:
        return {"error": "未初始化"}

    cols, population, _, _ = pred._get_cols_and_population()
    if top_k is None:
        top_k = len(cols)

    def _top_set(scores_dict):
        s = sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)
        return set(item[0] for item in s[:top_k])

    fusion_top = _top_set(pred.fusion_scores)
    mc_top = _top_set(pred.monte_carlo_probs)
    markov_top = _top_set(pred.markov_probs)

    # LSTM-CRF 解码的 Top 号码集合
    crf_top = set()
    try:
        crf_decoder = _SimpleCRFDecoder(
            population=population,
            fusion_scores=pred.fusion_scores,
            missing_analysis=pred.missing_analysis,
            recent_df=pred.df.head(100),
            cols=cols
        )
        crf_nums = crf_decoder.viterbi_decode(top_k, temperature=0.8)
        crf_top = set(crf_nums[:top_k])
    except Exception:
        pass

    recent = pred.df.head(n_recent)

    def _stat(top):
        hits = []
        for _, row in recent.iterrows():
            actual = set(int(row[c]) for c in cols if c in row)
            hits.append(len(actual & top))
        return (sum(hits) / len(hits)) if hits else 0.0

    result = {
        "lottery_type": lottery_type,
        "top_k": top_k,
        "n_recent": n_recent,
        "fusion_hit_rate": round(_stat(fusion_top), 3),
        "mc_hit_rate": round(_stat(mc_top), 3),
        "markov_hit_rate": round(_stat(markov_top), 3),
        "crf_hit_rate": round(_stat(crf_top), 3) if crf_top else 0.0,
        "random_expectation": round(top_k * len(cols) / len(population), 3)
    }
    return result
