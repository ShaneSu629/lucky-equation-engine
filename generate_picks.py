# generate_picks.py
import random
import os
import pandas as pd
from collections import Counter

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
# 确保数据目录自动创建，避免路径不存在报错
os.makedirs(DATA_DIR, exist_ok=True)

# 尝试导入增强预测引擎
try:
    from enhanced_predict import (
        enhanced_predict_ssq, enhanced_predict_kl8, enhanced_predict_fcsd,
        enhanced_predict_dlt, enhanced_predict_qxc, enhanced_predict_pl3,
        get_ensemble_prediction, get_feature_summary, get_confidence_distribution,
        calculate_ac_value, calculate_span, validate_combination_ssq
    )
    ENHANCED_AVAILABLE = True
except ImportError:
    ENHANCED_AVAILABLE = False


def analyze_hot_cold(series, population, hot_ratio=0.33, cold_ratio=0.33):
    # 将号码按照历史出现频次，划分为：
    # - 热码 (Hot): 出现频次最高的前 33%
    # - 冷码 (Cold): 出现频次最低的 33%
    # - 温码 (Warm): 介于两者之间的 34%
    # 返回这三类的号码列表。
    counts = Counter(series)
    # 补齐未出现的号码，频次记为 0
    for val in population:
        if val not in counts:
            counts[val] = 0

    # 按出现频次从大到小排序
    sorted_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    sorted_nums = [item[0] for item in sorted_items]

    n = len(sorted_nums)
    hot_boundary = int(n * hot_ratio)
    cold_boundary = n - int(n * cold_ratio)

    hot_nums = sorted_nums[:hot_boundary]
    warm_nums = sorted_nums[hot_boundary:cold_boundary]
    cold_nums = sorted_nums[cold_boundary:]

    return hot_nums, warm_nums, cold_nums


def pick_balanced(hot_nums, warm_nums, cold_nums, k, distribution=(3, 2, 1)):
    # 从热、温、冷码中按比例科学抽取号码。
    # 例如抽取 6 个红球，配比为 3个热码 + 2个温码 + 1个冷码。
    h_count, w_count, c_count = distribution
    # 确保抽取数量不超过各分类的实际大小
    h_count = min(h_count, len(hot_nums))
    w_count = min(w_count, len(warm_nums))
    c_count = min(c_count, len(cold_nums))

    # 抽取
    picked_hot = random.sample(hot_nums, h_count) if h_count > 0 else []
    picked_warm = random.sample(warm_nums, w_count) if w_count > 0 else []
    picked_cold = random.sample(cold_nums, c_count) if c_count > 0 else []

    result = picked_hot + picked_warm + picked_cold

    # 如果由于配比不足导致数量不够，用剩下的号码补齐
    remaining_needed = k - len(result)
    if remaining_needed > 0:
        pool = list(set(hot_nums + warm_nums + cold_nums) - set(result))
        # 修复：防止候选池不足时 sample 报错
        pick_count = min(remaining_needed, len(pool))
        if pick_count > 0:
            result += random.sample(pool, pick_count)

    return sorted(result)


def predict_ssq(n_groups=5):
    # 【增强算法 v2.0】集成：指数衰减加权 + 马尔可夫转移 + 遗漏回补 + AC/跨度/012路约束
    csv_path = os.path.join(DATA_DIR, "ssq.csv")
    red_population = list(range(1, 34))
    blue_population = list(range(1, 17))

    # 默认无数据时，进行纯随机分组
    if not os.path.exists(csv_path):
        groups = []
        for _ in range(n_groups):
            reds = sorted(random.sample(red_population, 6))
            blue = random.choice(blue_population)
            groups.append((reds, blue))
        return groups

    # 优先使用增强算法
    if ENHANCED_AVAILABLE:
        try:
            return enhanced_predict_ssq(n_groups)
        except Exception as e:
            print(f"[generate_picks] 增强算法失败，回退到经典算法：{e}")

    # === 经典算法（回退） ===
    try:
        # 指定编码，兼容 Windows 系统
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        if df.empty:
            raise ValueError("CSV file is empty")
        # 统计红球与蓝球历史频次
        all_reds = pd.concat([df['r1'], df['r2'], df['r3'], df['r4'], df['r5'], df['r6']])
        r_hot, r_warm, r_cold = analyze_hot_cold(all_reds, red_population)
        b_hot, b_warm, b_cold = analyze_hot_cold(df['blue'], blue_population, hot_ratio=0.4, cold_ratio=0.4)
    except Exception as e:
        print(f"读取双色球历史数据失败，切换为随机：{e}")
        r_hot, r_warm, r_cold = red_population[:11], red_population[11:22], red_population[22:]
        b_hot, b_warm, b_cold = blue_population[:6], blue_population[6:11], blue_population[11:]

    groups = []
    for _ in range(n_groups):
        # 经典组合比例：3个热码 + 2个温码 + 1个冷码
        reds = pick_balanced(r_hot, r_warm, r_cold, 6, distribution=(3, 2, 1))
        # 蓝球倾向于在热码中选择（80%概率选热码，20%概率选温冷码）
        if random.random() < 0.8:
            blue = random.choice(b_hot)
        else:
            blue = random.choice(b_warm + b_cold)
        groups.append((reds, blue))
    return groups


def predict_kl8(n_groups=5, select_count=10):
    # 【增强算法 v2.0】快乐8热温冷配比预测
    csv_path = os.path.join(DATA_DIR, "kl8.csv")
    population = list(range(1, 81))

    if not os.path.exists(csv_path):
        return [sorted(random.sample(population, select_count)) for _ in range(n_groups)]

    # 优先使用增强算法
    if ENHANCED_AVAILABLE:
        try:
            return enhanced_predict_kl8(n_groups, select_count)
        except Exception as e:
            print(f"[generate_picks] 增强算法失败，回退到经典算法：{e}")

    # === 经典算法（回退） ===
    try:
        # 指定编码，兼容 Windows 系统
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        if df.empty:
            raise ValueError("CSV is empty")
        cols = [f"n{i:02d}" for i in range(1, 21)]
        all_nums = pd.concat([df[col] for col in cols if col in df.columns])
        hot, warm, cold = analyze_hot_cold(all_nums, population)
    except Exception as e:
        print(f"读取快乐8历史数据失败，切换为随机：{e}")
        hot, warm, cold = population[:26], population[26:53], population[53:]

    groups = []
    for _ in range(n_groups):
        # 选十黄金分配：5个热码 + 3个温码 + 2个冷码
        nums = pick_balanced(hot, warm, cold, select_count, distribution=(5, 3, 2))
        groups.append(nums)
    return groups


def predict_fcsd(n_groups=5):
    # 【增强算法 v2.0】福彩3D分位加权直选预测
    csv_path = os.path.join(DATA_DIR, "fcsd.csv")
    population = list(range(10))

    if not os.path.exists(csv_path):
        return [(random.randint(0,9), random.randint(0,9), random.randint(0,9)) for _ in range(n_groups)]

    # 优先使用增强算法
    if ENHANCED_AVAILABLE:
        try:
            return enhanced_predict_fcsd(n_groups)
        except Exception as e:
            print(f"[generate_picks] 增强算法失败，回退到经典算法：{e}")

    # === 经典算法（回退） ===
    try:
        # 指定编码，兼容 Windows 系统
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        if df.empty:
            raise ValueError("CSV is empty")

        # 分位统计热温冷
        n1_hot, n1_warm, n1_cold = analyze_hot_cold(df['n1'], population, hot_ratio=0.4, cold_ratio=0.3)
        n2_hot, n2_warm, n2_cold = analyze_hot_cold(df['n2'], population, hot_ratio=0.4, cold_ratio=0.3)
        n3_hot, n3_warm, n3_cold = analyze_hot_cold(df['n3'], population, hot_ratio=0.4, cold_ratio=0.3)
    except Exception as e:
        print(f"读取福彩3D历史数据失败，使用纯随机：{e}")
        n1_hot, n1_warm, n1_cold = population[:4], population[4:7], population[7:]
        n2_hot, n2_warm, n2_cold = population[:4], population[4:7], population[7:]
        n3_hot, n3_warm, n3_cold = population[:4], population[4:7], population[7:]

    def pick_single_pos(hot, warm, cold):
        # 60% 概率选热码，30% 概率选温码，10% 概率选冷码
        r = random.random()
        if r < 0.6 and hot:
            return random.choice(hot)
        elif r < 0.9 and warm:
            return random.choice(warm)
        else:
            return random.choice(cold)

    groups = []
    for _ in range(n_groups):
        n1 = pick_single_pos(n1_hot, n1_warm, n1_cold)
        n2 = pick_single_pos(n2_hot, n2_warm, n2_cold)
        n3 = pick_single_pos(n3_hot, n3_warm, n3_cold)
        groups.append((n1, n2, n3))
    return groups


# ========== 以下为格式化输出函数（已按要求精简） ==========
def format_ssq(groups):
    lines = []
    for i, (reds, blue) in enumerate(groups, 1):
        red_str = " ".join(f"{x:02d}" for x in reds)
        lines.append(f"第{i:02d}注 红球：{red_str} 蓝球：{blue:02d}")
    return "\n".join(lines)


def format_kl8(groups):
    lines = []
    for i, nums in enumerate(groups, 1):
        num_str = " ".join(f"{x:02d}" for x in nums)
        lines.append(f"第{i:02d}注 {num_str}")
    return "\n".join(lines)


def format_fcsd(groups):
    lines = []
    for i, (n1, n2, n3) in enumerate(groups, 1):
        lines.append(f"第{i:02d}注 {n1} {n2} {n3}")
    return "\n".join(lines)


# ===== 体彩预测 =====

def predict_dlt(n_groups=5):
    """大乐透预测：5个前区(1-35) + 2个后区(1-12)"""
    # 优先使用增强算法
    if ENHANCED_AVAILABLE:
        try:
            return enhanced_predict_dlt(n_groups)
        except Exception as e:
            print(f"增强算法失败，回退经典算法：{e}")

    csv_path = os.path.join(DATA_DIR, "dlt.csv")
    front_population = list(range(1, 36))
    back_population = list(range(1, 13))

    if not os.path.exists(csv_path):
        groups = []
        for _ in range(n_groups):
            fronts = sorted(random.sample(front_population, 5))
            backs = sorted(random.sample(back_population, 2))
            groups.append((fronts, backs))
        return groups

    try:
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        if df.empty:
            raise ValueError("CSV file is empty")
        all_fronts = pd.concat([df['f1'], df['f2'], df['f3'], df['f4'], df['f5']])
        f_hot, f_warm, f_cold = analyze_hot_cold(all_fronts, front_population)
        all_backs = pd.concat([df['b1'], df['b2']])
        b_hot, b_warm, b_cold = analyze_hot_cold(all_backs, back_population, hot_ratio=0.4, cold_ratio=0.4)
    except Exception as e:
        print(f"读取大乐透历史数据失败，切换为随机：{e}")
        f_hot, f_warm, f_cold = front_population[:12], front_population[12:24], front_population[24:]
        b_hot, b_warm, b_cold = back_population[:5], back_population[5:9], back_population[9:]

    groups = []
    for _ in range(n_groups):
        # 前区 3:2 配比
        fronts = pick_balanced(f_hot, f_warm, f_cold, 5, distribution=(3, 2, 0))
        if len(fronts) < 5:
            fronts = sorted(random.sample(front_population, 5))
        # 后区 80% 概率选热码
        backs = []
        for _ in range(2):
            if random.random() < 0.8 and b_hot:
                backs.append(random.choice(b_hot))
            else:
                backs.append(random.choice(b_warm + b_cold))
        backs = sorted(set(backs))
        while len(backs) < 2:
            extra = random.choice(back_population)
            if extra not in backs:
                backs.append(extra)
        backs = sorted(backs)
        groups.append((sorted(fronts), backs))
    return groups


def predict_qxc(n_groups=5):
    """七星彩预测：7个位置各0-9，分位独立加权"""
    if ENHANCED_AVAILABLE:
        try:
            return enhanced_predict_qxc(n_groups)
        except Exception as e:
            print(f"增强算法失败，回退经典算法：{e}")

    csv_path = os.path.join(DATA_DIR, "qxc.csv")
    population = list(range(10))

    if not os.path.exists(csv_path):
        return [tuple(random.randint(0, 9) for _ in range(7)) for _ in range(n_groups)]

    try:
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        if df.empty:
            raise ValueError("CSV is empty")
        pos_hot = []
        pos_warm = []
        pos_cold = []
        for i in range(1, 8):
            col = f'n{i}'
            h, w, c = analyze_hot_cold(df[col], population, hot_ratio=0.4, cold_ratio=0.3)
            pos_hot.append(h)
            pos_warm.append(w)
            pos_cold.append(c)
    except Exception as e:
        print(f"读取七星彩历史数据失败，使用纯随机：{e}")
        pos_hot = [population[:4]] * 7
        pos_warm = [population[4:7]] * 7
        pos_cold = [population[7:]] * 7

    def pick_pos(hot, warm, cold):
        r = random.random()
        if r < 0.6 and hot:
            return random.choice(hot)
        elif r < 0.9 and warm:
            return random.choice(warm)
        else:
            return random.choice(cold)

    groups = []
    for _ in range(n_groups):
        nums = tuple(pick_pos(pos_hot[i], pos_warm[i], pos_cold[i]) for i in range(7))
        groups.append(nums)
    return groups


def predict_pl3(n_groups=5):
    """排列三预测：3个位置各0-9，分位独立加权"""
    if ENHANCED_AVAILABLE:
        try:
            return enhanced_predict_pl3(n_groups)
        except Exception as e:
            print(f"增强算法失败，回退经典算法：{e}")

    csv_path = os.path.join(DATA_DIR, "pl3.csv")
    population = list(range(10))

    if not os.path.exists(csv_path):
        return [(random.randint(0, 9), random.randint(0, 9), random.randint(0, 9)) for _ in range(n_groups)]

    try:
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        if df.empty:
            raise ValueError("CSV is empty")
        n1_hot, n1_warm, n1_cold = analyze_hot_cold(df['n1'], population, hot_ratio=0.4, cold_ratio=0.3)
        n2_hot, n2_warm, n2_cold = analyze_hot_cold(df['n2'], population, hot_ratio=0.4, cold_ratio=0.3)
        n3_hot, n3_warm, n3_cold = analyze_hot_cold(df['n3'], population, hot_ratio=0.4, cold_ratio=0.3)
    except Exception as e:
        print(f"读取排列三历史数据失败，使用纯随机：{e}")
        n1_hot, n1_warm, n1_cold = population[:4], population[4:7], population[7:]
        n2_hot, n2_warm, n2_cold = population[:4], population[4:7], population[7:]
        n3_hot, n3_warm, n3_cold = population[:4], population[4:7], population[7:]

    def pick_single_pos(hot, warm, cold):
        r = random.random()
        if r < 0.6 and hot:
            return random.choice(hot)
        elif r < 0.9 and warm:
            return random.choice(warm)
        else:
            return random.choice(cold)

    groups = []
    for _ in range(n_groups):
        n1 = pick_single_pos(n1_hot, n1_warm, n1_cold)
        n2 = pick_single_pos(n2_hot, n2_warm, n2_cold)
        n3 = pick_single_pos(n3_hot, n3_warm, n3_cold)
        groups.append((n1, n2, n3))
    return groups


def format_dlt(groups):
    """大乐透格式化：前区5 + 后区2"""
    lines = []
    for i, (fronts, backs) in enumerate(groups, 1):
        f_str = " ".join(f"{x:02d}" for x in fronts)
        b_str = " ".join(f"{x:02d}" for x in backs)
        lines.append(f"第{i:02d}注 前区：{f_str} 后区：{b_str}")
    return "\n".join(lines)


def format_qxc(groups):
    """七星彩格式化：7个数字"""
    lines = []
    for i, nums in enumerate(groups, 1):
        if isinstance(nums, tuple):
            num_str = " ".join(str(x) for x in nums)
        else:
            num_str = " ".join(str(x) for x in nums)
        lines.append(f"第{i:02d}注 {num_str}")
    return "\n".join(lines)


def format_pl3(groups):
    """排列三格式化：3个数字"""
    lines = []
    for i, nums in enumerate(groups, 1):
        if isinstance(nums, tuple):
            lines.append(f"第{i:02d}注 {' '.join(str(x) for x in nums)}")
        else:
            lines.append(f"第{i:02d}注 {nums}")
    return "\n".join(lines)


def format_dlt_plain(groups):
    """大乐透纯号码格式"""
    lines = []
    for i, (fronts, backs) in enumerate(groups, 1):
        f_str = " ".join(f"{x:02d}" for x in fronts)
        b_str = " ".join(f"{x:02d}" for x in backs)
        lines.append(f"{f_str} + {b_str}")
    return "\n".join(lines)


def format_qxc_plain(groups):
    """七星彩纯号码格式"""
    lines = []
    for i, nums in enumerate(groups, 1):
        if isinstance(nums, tuple):
            lines.append("".join(str(x) for x in nums))
        else:
            lines.append(str(nums))
    return "\n".join(lines)


def format_pl3_plain(groups):
    """排列三纯号码格式"""
    lines = []
    for i, nums in enumerate(groups, 1):
        if isinstance(nums, tuple):
            lines.append("".join(str(x) for x in nums))
        else:
            lines.append(str(nums))
    return "\n".join(lines)


# --- 补充被 app.py 引用的辅助方法 ---
def gen_kl8_pick1(n_groups=5):
    # 快乐 8 选一：从 1-80 中选择不同的单球
    # 增加边界校验，防止注数超过 80 时报错
    n_groups = max(1, min(n_groups, 80))
    return sorted(random.sample(range(1, 81), n_groups))


def gen_kl8_pick4(n_groups=5):
    # 快乐 8 选四：每组 4 个数字 (1-80)
    return [sorted(random.sample(range(1, 81), 4)) for _ in range(n_groups)]


def gen_3d_group6(n_groups=5):
    # 福彩 3D 组选六：选择三个互不相同的数字 (0-9)
    groups = []
    for _ in range(n_groups):
        groups.append(sorted(random.sample(range(10), 3)))
    return groups


# --- 对冲号码格式化函数 ---
def format_kl8_pick1(nums):
    lines = []
    for i, x in enumerate(nums, 1):
        lines.append(f"第{i:02d}注 {x:02d}")
    return "\n".join(lines)


def format_kl8_pick4(groups):
    lines = []
    for i, nums in enumerate(groups, 1):
        num_str = " ".join(f"{x:02d}" for x in nums)
        lines.append(f"第{i:02d}注 {num_str}")
    return "\n".join(lines)


def format_3d_group6(groups):
    lines = []
    for i, nums in enumerate(groups, 1):
        num_str = " ".join(str(x) for x in nums)
        lines.append(f"第{i:02d}注 {num_str}")
    return "\n".join(lines)


def format_ssq_plain(groups):
    """纯号码格式，不含标题，用于 AI 对比展示"""
    lines = []
    for i, (reds, blue) in enumerate(groups, 1):
        red_str = " ".join(f"{x:02d}" for x in reds)
        lines.append(f"{red_str} + {blue:02d}")
    return "\n".join(lines)


def format_kl8_plain(groups):
    """纯号码格式，不含标题，用于 AI 对比展示"""
    lines = []
    for i, nums in enumerate(groups, 1):
        num_str = " ".join(f"{x:02d}" for x in nums)
        lines.append(num_str)
    return "\n".join(lines)


def format_fcsd_plain(groups):
    """纯号码格式，不含标题，用于 AI 对比展示"""
    lines = []
    for i, (n1, n2, n3) in enumerate(groups, 1):
        lines.append(f"{n1}{n2}{n3}")
    return "\n".join(lines)