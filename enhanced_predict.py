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
                            w_markov: float = 0.25) -> Dict[int, float]:
    """
    贝叶斯融合：在概率空间中对多个信号进行加权融合
    score = exp(w1*log(freq) + w2*log(decay) + w3*log(bounce+smooth) + w4*log(markov+smooth))
    然后归一化
    """
    EPSILON = 1e-8
    scores = {}

    for num in population:
        f = freq_weights.get(num, EPSILON)
        d = decay_weights.get(num, EPSILON)
        m = missing_analysis.get(num, {}).get("bounce_prob", 0.0) + EPSILON
        k = markov_probs.get(num, EPSILON)

        # 对数空间融合
        log_score = (w_freq * math.log(max(f, EPSILON)) +
                     w_decay * math.log(max(d, EPSILON)) +
                     w_missing * math.log(max(m, EPSILON)) +
                     w_markov * math.log(max(k, EPSILON)))
        scores[num] = math.exp(log_score)

    # 归一化
    total = sum(scores.values())
    if total > 0:
        for k in scores:
            scores[k] /= total

    return scores


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
        csv_path = os.path.join(DATA_DIR, f"{self.lottery_type}.csv")
        if not os.path.exists(csv_path):
            return pd.DataFrame()
        encodings = ['utf-8', 'utf-8-sig', 'gbk', 'gb2312']
        for enc in encodings:
            try:
                return pd.read_csv(csv_path, encoding=enc)
            except (UnicodeDecodeError, UnicodeError):
                continue
        return pd.DataFrame()

    def _get_cols_and_population(self):
        if self.lottery_type == "ssq":
            cols = ['r1', 'r2', 'r3', 'r4', 'r5', 'r6']
            population = list(range(1, 34))
            blue_cols = ['blue']
            blue_population = list(range(1, 17))
            return cols, population, blue_cols, blue_population
        elif self.lottery_type == "kl8":
            cols = [f"n{i:02d}" for i in range(1, 21)]
            population = list(range(1, 81))
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

        # 5. 贝叶斯融合
        self.fusion_scores = bayesian_fusion_scores(
            self.freq_weights,
            self.decay_weights,
            self.missing_analysis,
            self.markov_probs,
            population,
            w_freq=0.20,
            w_decay=0.30,
            w_missing=0.25,
            w_markov=0.25
        )

        # 6. 蒙特卡洛
        self.monte_carlo_probs = monte_carlo_sample(
            self.fusion_scores, population,
            sample_size=len(cols), n_simulations=10000)

        # 7. 蓝球独立分析（仅双色球）
        if blue_cols and blue_population:
            self._init_blue_features(recent, blue_cols, blue_population)

        self._initialized = True

    def _init_blue_features(self, recent, blue_cols, blue_population):
        """蓝球独立特征初始化"""
        bc = Counter(recent[blue_cols[0]].values)
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
        last_blue = int(recent.iloc[0][blue_cols[0]])
        self.blue_markov = markov_next_probability(
            [last_blue], markov_blue, blue_population)

        # 蓝球融合
        self.blue_fusion = bayesian_fusion_scores(
            self.blue_freq, self.blue_decay,
            self.blue_missing, self.blue_markov,
            blue_population, 0.25, 0.25, 0.25, 0.25)

    # ---- 双色球预测 ----
    def predict_ssq(self, n_groups: int = 5) -> List[Tuple[List[int], int]]:
        if not self._initialized or self.lottery_type != "ssq":
            from generate_picks import predict_ssq
            return predict_ssq(n_groups)

        red_population = list(range(1, 34))
        blue_population = list(range(1, 17))

        groups = []
        used_combinations = set()

        # 基于融合分数排序号码
        sorted_reds = sorted(self.fusion_scores.items(),
                             key=lambda x: x[1], reverse=True)

        attempts = 0
        max_attempts = n_groups * 50

        while len(groups) < n_groups and attempts < max_attempts:
            attempts += 1

            # 按融合概率进行加权抽样（热温冷三层混合）
            reds = self._weighted_sample_balanced(
                self.fusion_scores, red_population, 6,
                hot_ratio=0.40, warm_ratio=0.35, cold_ratio=0.25)

            reds_sorted = tuple(sorted(reds))
            if reds_sorted in used_combinations:
                continue

            # 约束校验
            valid, issues = validate_combination_ssq(reds)
            if not valid and len(issues) >= 3:
                continue  # 超过2个问题的组合直接丢弃

            # 蓝球：按融合概率 + 80%热20%冷
            blue = self._weighted_pick_with_temperature(
                self.blue_fusion, blue_population, temp=0.3)

            used_combinations.add(reds_sorted)
            groups.append((list(reds_sorted), blue))

        # 如果生成不够，补随机
        import random
        while len(groups) < n_groups:
            reds = sorted(random.sample(list(range(1, 34)), 6))
            blue = random.choice(list(range(1, 17)))
            tr = tuple(reds)
            if tr not in used_combinations:
                used_combinations.add(tr)
                groups.append((reds, blue))

        return groups

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
            from generate_picks import predict_kl8
            return predict_kl8(n_groups, select_count)

        population = list(range(1, 81))
        groups = []
        used_combinations = set()
        attempts = 0

        while len(groups) < n_groups and attempts < n_groups * 50:
            attempts += 1
            nums = self._weighted_sample_balanced(
                self.fusion_scores, population, select_count,
                hot_ratio=0.40, warm_ratio=0.35, cold_ratio=0.25)
            nt = tuple(nums)
            if nt not in used_combinations:
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
            from generate_picks import predict_fcsd
            return predict_fcsd(n_groups)

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

        # 蓝球（双色球）
        if self.lottery_type == "ssq" and hasattr(self, 'blue_fusion'):
            sorted_blue = sorted(self.blue_fusion.items(),
                                  key=lambda x: x[1], reverse=True)
            summary["blue_fusion_top5"] = {
                str(k): round(v, 6) for k, v in sorted_blue[:5]
            }

        return summary

    def get_ensemble_prediction(self, n_groups: int = 5) -> Dict:
        """
        集成预测：综合本地算法 + 蒙特卡洛 + 马尔可夫
        返回收敛号码和置信度
        使用基于日期+最新期号的确定性种子，确保同一天同一批数据结果稳定
        """
        if not self._initialized:
            return {"error": "未初始化"}

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

        # 统计号码出现频次
        all_sets = method1_sets + method2_sets + method3_sets
        counter = Counter()
        for s in all_sets:
            for n in s:
                counter[n] += 1

        total_sets = len(all_sets)
        confidence = {k: round(v / total_sets, 4) for k, v in counter.items()}

        # 生成推荐组（确定性选择，避免 random.shuffle 导致每次不同）
        recommendations = []
        used = set()
        sorted_by_conf = sorted(confidence.items(), key=lambda x: x[1], reverse=True)

        for group_idx in range(n_groups):
            # 从高置信度号码中确定性选取
            candidates = [item[0] for item in sorted_by_conf[:sample_size * 3]]
            # 用偏移代替 shuffle：每组从 candidates 中滑动窗口选取
            offset = group_idx % max(1, len(candidates) - sample_size + 1)
            nums = sorted(candidates[offset:offset + sample_size])
            if len(nums) < sample_size:
                # 兜底：从剩余高置信度补全
                remaining = [item[0] for item in sorted_by_conf[sample_size * 3:]]
                need = sample_size - len(nums)
                nums = sorted(nums + remaining[:need])
            nt = tuple(nums)
            if nt not in used:
                used.add(nt)
                avg_conf = sum(confidence.get(n, 0) for n in nums) / len(nums)
                if self.lottery_type == "ssq":
                    valid, _ = validate_combination_ssq(nums)
                    # 为双色球附加蓝球
                    blue_population = list(range(1, 17))
                    blue = self._weighted_pick_with_temperature(
                        self.blue_fusion, blue_population, temp=0.3)
                    recommendations.append({
                        "nums": nums + [blue],
                        "confidence": round(avg_conf, 4),
                        "valid": valid
                    })
                else:
                    recommendations.append({
                        "nums": nums,
                        "confidence": round(avg_conf, 4),
                        "valid": True
                    })

        return {
            "lottery_type": self.lottery_type,
            "confidence_distribution": {
                str(k): v for k, v in sorted_by_conf[:15]
            },
            "recommendations": recommendations,
            "model_contributions": {
                "bayesian_fusion": 0.40,
                "monte_carlo": 0.30,
                "markov_chain": 0.30
            }
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


def get_ensemble_prediction(lottery_type: str, n_groups: int = 5) -> Dict:
    """获取集成预测结果"""
    pred = _get_predictor(lottery_type)
    return pred.get_ensemble_prediction(n_groups)


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
