# app.py
import streamlit as st
import pandas as pd
import os
import random
import json
import time
import logging
from datetime import datetime

# 应用级日志
logger = logging.getLogger("app")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s.%(funcName)s: %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)

# 导入我们的逻辑模块
from fetch_lottery import update, DATA_DIR
from db_manager import (
    init_db, read_lottery_data as _db_read_lottery,
    get_latest_code as _db_get_latest_code,
    get_lottery_df as _db_get_lottery_df,
)
from generate_picks import (
    predict_ssq,
    predict_kl8,
    predict_fcsd,
    predict_dlt,
    predict_qxc,
    predict_pl3,
    analyze_hot_cold,
    format_ssq,
    format_kl8,
    format_fcsd,
    format_dlt,
    format_qxc,
    format_pl3,
    gen_kl8_pick1,
    gen_kl8_pick4,
    gen_3d_group6,
    gen_pl3_group6,
    gen_qxc_pick7,
    format_kl8_pick1,
    format_kl8_pick4,
    format_3d_group6,
    format_pl3_group6,
    format_qxc_pick7,
    format_ssq_plain,
    format_kl8_plain,
    format_fcsd_plain,
    format_dlt_plain,
    format_qxc_plain,
    format_pl3_plain
)
from ai_predict import (
    ai_predict_ssq,
    ai_predict_kl8,
    ai_predict_fcsd,
    ai_predict_dlt,
    ai_predict_qxc,
    ai_predict_pl3,
    ai_analyze_trend,
    ai_optimize_hedge,
    ai_optimize_hedge_sports,
    is_ai_configured,
    save_prediction_record,
    analyze_saved_predictions,
    get_prediction_records,
    get_betting_report,
    load_config
)

# 导入增强预测引擎
try:
    from enhanced_predict import (
        get_ensemble_prediction,
        get_feature_summary,
        get_confidence_distribution,
        EnhancedPredictor
    )
    ENSEMBLE_AVAILABLE = True
except ImportError:
    ENSEMBLE_AVAILABLE = False

# ===== 通用渲染辅助函数 =====

def _toast_save(msg: str, icon: str = "✅"):
    """保存操作统一反馈：toast 弹框 + 页面内 success 留痕 + 日志。"""
    logger.info(msg)
    try:
        st.toast(msg, icon=icon)
    except Exception:
        pass  # 低版本 Streamlit 无 toast
    st.success(msg)


def _toast_error(msg: str):
    """保存失败统一反馈：toast 弹框 + 页面内 error 留痕 + 日志。"""
    logger.error(msg)
    try:
        st.toast(msg, icon="❌")
    except Exception:
        pass
    st.error(msg)


def _local_ball_html(lot, item):
    """将一条本地预测结果渲染为 HTML 号码球。"""
    if lot == "ssq":
        reds, blue = item
        red_html = "".join([f"<span class='ssq-red'>{x:02d}</span>" for x in reds])
        blue_html = f"<span class='ssq-blue'>{blue:02d}</span>"
        return red_html + blue_html
    if lot == "kl8":
        return "".join([f"<span class='kl8-ball'>{x:02d}</span>" for x in item])
    if lot == "dlt":
        fronts, backs = item
        f_html = "".join([f"<span class='ssq-red'>{x:02d}</span>" for x in fronts])
        b_html = "".join([f"<span class='ssq-blue'>{x:02d}</span>" for x in backs])
        return f_html + b_html
    if lot in ("fcsd", "pl3"):
        n1, n2, n3 = item
        return (f"<span class='f3d-ball'>{n1}</span>"
                f"<span class='f3d-ball'>{n2}</span>"
                f"<span class='f3d-ball'>{n3}</span>")
    if lot == "qxc":
        return " ".join([f"<span class='f3d-ball'>{x}</span>" for x in item])
    return ""


def _validate_ai_group(lot: str, rec: dict, pick_size: int = None):
    """校验并归一化一条 AI 推荐号码，过滤非法数据。

    Args:
        lot: 彩种标识（ssq/kl8/fcsd/dlt/qxc/pl3）。
        rec: AI 返回的推荐条目（含 numbers 字段）。
        pick_size: 快乐8选号个数（1-10），默认10。

    Returns:
        (clean, ok): clean 为可入库的归一化结构；ok 表示该条是否满足基本约束。
        不满足约束的条目返回 (None, False)，避免脏数据入库后污染命中统计。
    """
    nums = rec.get("numbers", {})
    try:
        if lot == "ssq":
            reds = [int(x) for x in nums.get("red", [])]
            blue = int(nums.get("blue", 0))
            if len(reds) == 6 and len(set(reds)) == 6 and all(1 <= r <= 33 for r in reds) and 1 <= blue <= 16:
                return {"red": reds, "blue": blue}, True
        elif lot == "dlt":
            fronts = [int(x) for x in nums.get("front", [])]
            backs = [int(x) for x in nums.get("back", [])]
            if (len(fronts) == 5 and len(set(fronts)) == 5 and all(1 <= f <= 35 for f in fronts)
                    and len(backs) == 2 and len(set(backs)) == 2 and all(1 <= b <= 12 for b in backs)):
                return {"nums": fronts + backs}, True
        else:
            # kl8 / fcsd / pl3 / qxc：numbers 为列表
            balls = [int(x) for x in nums if str(x).isdigit()]
            _kl8_n = pick_size if pick_size else 10
            if lot == "kl8" and len(balls) == _kl8_n and len(set(balls)) == _kl8_n and all(1 <= b <= 80 for b in balls):
                return {"nums": balls}, True
            if lot in ("fcsd", "pl3") and len(balls) == 3 and all(0 <= b <= 9 for b in balls):
                return {"nums": balls}, True
            if lot == "qxc" and len(balls) == 7 and all(0 <= b <= 9 for b in balls):
                return {"nums": balls}, True
    except (ValueError, TypeError, AttributeError):
        pass
    return None, False


def _ai_ball_html(lot, rec):
    """将一条 AI 推荐结果渲染为 HTML 号码球。"""
    nums = rec.get("numbers", {})
    if lot == "ssq":
        reds = nums.get("red", [])
        blue = nums.get("blue", 0)
        red_html = "".join([f"<span class='ssq-red'>{x:02d}</span>" for x in reds])
        blue_html = f"<span class='ssq-blue'>{blue:02d}</span>"
        return red_html + blue_html
    if lot == "kl8":
        return "".join([f"<span class='kl8-ball'>{x:02d}</span>" for x in nums]) if isinstance(nums, list) else ""
    if lot == "dlt":
        fronts = nums.get("front", [])
        backs = nums.get("back", [])
        f_html = "".join([f"<span class='ssq-red'>{x:02d}</span>" for x in fronts])
        b_html = "".join([f"<span class='ssq-blue'>{x:02d}</span>" for x in backs])
        return f_html + b_html
    if lot in ("fcsd", "pl3"):
        if isinstance(nums, list) and len(nums) >= 3:
            return (f"<span class='f3d-ball'>{nums[0]}</span>"
                    f"<span class='f3d-ball'>{nums[1]}</span>"
                    f"<span class='f3d-ball'>{nums[2]}</span>")
        return ""
    if lot == "qxc":
        if isinstance(nums, list):
            return " ".join([f"<span class='f3d-ball'>{x}</span>" for x in nums])
        return ""
    return ""


# 页面配置：美化并设置标题与布局
st.set_page_config(
    page_title="幸运方程式 · 数字推理引擎",
    page_icon="🎲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 页面顶部系统名称
st.title("🎲 幸运方程式 · 数字推理引擎")
st.caption("基于历史数据的数字推理与组合优化平台")

# 理性购彩声明：开奖为独立随机事件，工具仅作统计娱乐参考
st.info(
    "⚠️ **理性购彩提示**：彩票开奖为独立随机事件，任何历史数据分析都无法预测结果。"
    "本工具仅提供**统计规律的娱乐参考**，不构成中奖承诺或购彩建议。请量力而行、理性投注，切勿沉迷。",
    icon="🎯",
)

# ===== 顶部：彩票大类总开关（福利彩票 / 体育彩票）=====
LOT_CATS = {
    "welfare": {"label": "🟢 福利彩票", "lots": ["ssq", "kl8", "fcsd"],
                "names": {"ssq": "🔴 双色球", "kl8": "🟡 快乐8", "fcsd": "🟢 福彩3D"}},
    "sports":  {"label": "🔵 体育彩票", "lots": ["dlt", "qxc", "pl3"],
                "names": {"dlt": "🔵 大乐透", "qxc": "🟣 七星彩", "pl3": "🟤 排列三"}},
}

if 'lottery_category' not in st.session_state:
    st.session_state['lottery_category'] = 'welfare'
if 'selected_lottery' not in st.session_state:
    st.session_state['selected_lottery'] = 'ssq'


def _on_category_change():
    """切换大类时，自动选中该大类下的第一个彩种，避免跨类错乱。"""
    new_cat = st.session_state.get('lottery_category')
    if new_cat in LOT_CATS and st.session_state.get('selected_lottery') not in LOT_CATS[new_cat]['lots']:
        st.session_state['selected_lottery'] = LOT_CATS[new_cat]['lots'][0]
    # 清空待保存的 AI 预测，避免切换后残留旧彩种的保存按钮
    for _k in [k for k in st.session_state if k.startswith('pending_') and k.endswith('_predictions')]:
        del st.session_state[_k]


st.markdown("**🎯 彩票大类**")
st.segmented_control(
    "彩票大类（福利彩票 / 体育彩票）",
    options=["welfare", "sports"],
    format_func=lambda c: LOT_CATS[c]["label"],
    key="lottery_category",
    on_change=_on_category_change,
    help="在顶部切换大类，下方所有页面与侧边栏彩种选择将只显示该大类的彩种。",
)
# 防御：保证当前彩种始终属于当前大类
if st.session_state.get('lottery_category') not in LOT_CATS:
    st.session_state['lottery_category'] = 'welfare'
if st.session_state.get('selected_lottery') not in LOT_CATS[st.session_state['lottery_category']]['lots']:
    st.session_state['selected_lottery'] = LOT_CATS[st.session_state['lottery_category']]['lots'][0]

# 自定义样式：美化卡片和彩票球
st.markdown("""
<style>
    .stApp {
        padding-top: 0 !important;
    }
    section.main {
        padding-top: 0 !important;
        margin-top: 0 !important;
    }
    .stSubheader {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }
    .stMarkdown {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }
    .ssq-red {
        background-color: #ff4d4f;
        color: white;
        border-radius: 50%;
        margin: 4px;
        display: inline-block;
        font-weight: bold;
        font-size: 16px;
        width: 44px;
        height: 44px;
        text-align: center;
        line-height: 44px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        box-sizing: border-box;
    }
    .ssq-blue {
        background-color: #1890ff;
        color: white;
        border-radius: 50%;
        margin: 4px;
        display: inline-block;
        font-weight: bold;
        font-size: 16px;
        width: 44px;
        height: 44px;
        text-align: center;
        line-height: 44px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        box-sizing: border-box;
    }
    .kl8-ball {
        background-color: #fa8c16;
        color: white;
        border-radius: 50%;
        margin: 4px;
        display: inline-block;
        font-weight: bold;
        font-size: 14px;
        width: 40px;
        height: 40px;
        text-align: center;
        line-height: 40px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        box-sizing: border-box;
    }
    .f3d-ball {
        background-color: #13c2c2;
        color: white;
        border-radius: 4px;
        padding: 8px 15px;
        margin: 4px;
        display: inline-block;
        font-weight: bold;
        font-size: 18px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .card {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        border: 1px solid #f0f0f0;
    }
    section.main [data-testid="stButton"] > button {
        background: linear-gradient(135deg, #f97316 0%, #ea580c 100%) !important;
        border: none !important;
        border-radius: 8px !important;
        color: #ffffff !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 2px 8px rgba(249, 115, 22, 0.3) !important;
    }
    section.main [data-testid="stButton"] > button:hover {
        background: linear-gradient(135deg, #fb923c 0%, #f97316 100%) !important;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(249, 115, 22, 0.4) !important;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 50%, #f8fafc 100%) !important;
        padding: 6px !important;
        margin: 0 !important;
    }
    [data-testid="stSidebar"] > div:first-child {
        padding: 0 !important;
        margin: 0 !important;
    }
    section[data-testid="stSidebar"] {
        padding: 6px !important;
    }
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] .stSubheader {
        color: #1e293b;
        padding: 0 !important;
        margin: 0 !important;
    }
    [data-testid="stSidebar"] .stExpander {
        border-color: rgba(0,0,0,0.1);
        width: 100% !important;
        margin: 0 !important;
    }
    [data-testid="stSidebar"] .stExpander > div:first-child {
        background: rgba(255,255,255,0.8);
        border-radius: 8px;
        width: 100% !important;
    }
    [data-testid="stSidebar"] .stButton {
        margin: 2px 0 !important;
        padding: 0 !important;
    }
    [data-testid="stSidebar"] .stButton > button {
        width: calc(100% - 8px) !important;
        padding: 10px 16px !important;
        margin: 2px 4px !important;
        border-radius: 8px !important;
        font-size: 14px !important;
        font-weight: 500 !important;
        text-align: left !important;
        cursor: pointer !important;
        transition: all 0.3s ease !important;
        border: 1px solid transparent !important;
        min-height: 40px !important;
        line-height: 20px !important;
    }
    [data-testid="stSidebar"] .stButton > button:not([kind="primary"]) {
        background: #ffffff !important;
        color: #475569 !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06) !important;
    }
    [data-testid="stSidebar"] .stButton > button:not([kind="primary"]):hover {
        background: #fff7ed !important;
        border-color: #fdba74 !important;
        transform: translateX(3px) !important;
        box-shadow: 0 2px 8px rgba(249, 115, 22, 0.15) !important;
        color: #ea580c !important;
    }
    [data-testid="stSidebar"] .stButton > button[kind="primary"] {
        background: #3b82f6 !important;
        color: #ffffff !important;
        border-color: #3b82f6 !important;
        box-shadow: 0 2px 8px rgba(59, 130, 246, 0.3) !important;
        transform: translateX(3px) !important;
    }
    [data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
        background: #60a5fa !important;
        box-shadow: 0 3px 12px rgba(59, 130, 246, 0.4) !important;
    }
    [data-testid="stSidebar"] .stTextInput > div > div > input {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        color: #1e293b !important;
        border-radius: 8px !important;
    }
    [data-testid="stSidebar"] .stTextInput > div > div > input::placeholder {
        color: #94a3b8 !important;
    }
    [data-testid="stSidebar"] .stInfo {
        background: rgba(59, 130, 246, 0.08) !important;
        border: 1px solid rgba(59, 130, 246, 0.2) !important;
        color: #1e40af !important;
    }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    if 'selected_page' not in st.session_state:
        st.session_state['selected_page'] = 'dashboard'
    
    nav_items = [
        ("dashboard", "📈", "历史数据看板"),
        ("predict", "🎯", "智能号码预测"),
        ("hedge", "🛡️", "组合配比策略"),
        ("ai", "🤖", "AI 智能分析"),
        ("config", "⚙️", "配置中心")
    ]
    
    for key, icon, label in nav_items:
        is_selected = st.session_state['selected_page'] == key
        btn_type = "primary" if is_selected else "secondary"
        
        if st.button(f"{icon} {label}", key=f"nav_btn_{key}", width="stretch", type=btn_type):
            st.session_state['selected_page'] = key
            st.rerun()
    
    selected_page = st.session_state['selected_page']

    # ===== 彩种选择（跟随顶部大类，仅显示当前大类）=====
    st.subheader("🎰 选择彩种")
    cur_cat = st.session_state.get('lottery_category', 'welfare')
    for lot_key in LOT_CATS[cur_cat]['lots']:
        lot_label = LOT_CATS[cur_cat]['names'][lot_key]
        is_sel = st.session_state.get('selected_lottery') == lot_key
        btype = "primary" if is_sel else "secondary"
        if st.button(lot_label, key=f"lot_btn_{lot_key}", width="stretch", type=btype):
            st.session_state['selected_lottery'] = lot_key
            for _k in [k for k in st.session_state if k.startswith('pending_') and k.endswith('_predictions')]:
                del st.session_state[_k]
            st.rerun()

    st.markdown("---")


    st.subheader("🔄 数据同步与更新")
    if st.button("立即同步最新数据", width="stretch", type="primary"):
        st.session_state.sync_step = 0
        st.session_state.normal_sync = True
        st.rerun()
    
    if 'normal_sync' in st.session_state:
        with st.spinner("⏳ 正在同步最新数据..."):
            from fetch_lottery import update
            sync_order = [("ssq", "双色球"), ("kl8", "快乐8"), ("fcsd", "福彩3D"),
                          ("dlt", "大乐透"), ("qxc", "七星彩"), ("pl3", "排列三")]
            
            if st.session_state.sync_step < len(sync_order):
                lot_name, display_name = sync_order[st.session_state.sync_step]
                force_full = _db_get_latest_code(lot_name) is None
                
                if force_full:
                    st.info(f"📊 {display_name}：首次同步，全量获取历史数据...")
                else:
                    st.info(f"📊 {display_name}：增量同步，只获取最新数据...")
                
                st.caption(f"📥 正在同步 {display_name}...")
                
                update(lot_name, force_full=force_full)
                st.caption(f"✅ {display_name} 同步成功")
                
                st.session_state.sync_step += 1
                st.rerun()
            else:
                st.success("🎉 数据同步成功！")
                
                del st.session_state.sync_step
                del st.session_state.normal_sync
                st.rerun()


def _render_sports_hedge():
    """体彩版组合配比策略：核心=大乐透，对冲=排列三/七星彩（与福彩版对称）。"""
    st.subheader("🛡️ 大乐透「智能配比」组合优化工具")
    st.write("""
    大乐透单期中奖率低，很多彩友连续多期无法回血。
    本工具旨在将您的投注资金**科学分流分配**，组合购买**「高频、高中奖概率」**的排列三（每日开奖）或七星彩。
    **目的**：利用高概率小奖返还的资金，平扣大乐透不中奖的成本，实现总体账户避险。
    ⚠️ 彩票开奖完全随机，本工具仅作娱乐参考，请理性购彩、量力而行。
    """)

    st.write("---")

    col_input, col_math = st.columns([2, 3])

    with col_input:
        st.write("### 💰 第一步：输入你的投注配置")
        dlt_bets = st.number_input("核心：大乐透单期计划投注（注数）", min_value=1, max_value=50, value=10)
        dlt_cost = dlt_bets * 2
        st.markdown(f"🔴 **大乐透主投注额：** **{dlt_cost} 元**")

        hedge_strategy = st.selectbox(
            "第二步：选择搭配的对冲避险方案",
            [
                "🛡️ 方案 A（极速回血）：搭配 排列三 组选六（每日开奖，约0.6%中奖率，中奖得 173 元）",
                "🛡️ 方案 B（低频大回血）：搭配 七星彩 七位直选（超低概率、高奖级，中一次覆盖多期亏损）"
            ]
        )

        hedge_bets = st.slider("对冲单注数", min_value=1, max_value=20, value=5)
        hedge_cost = hedge_bets * 2
        total_cost = dlt_cost + hedge_cost

        st.write("---")
        st.metric("💳 总体组合投资预算", f"{total_cost} 元", f"其中大乐透 {dlt_cost} 元，对冲彩 {hedge_cost} 元")

    with col_math:
        st.write("### 📊 方案数学期望与回血率分析")

        if "方案 A" in hedge_strategy:
            st.markdown(f"""
            #### 🛡️ 方案 A 排列三「组选六」
            * **中奖率**：单注约 **0.6%**（1/167）。
            * **奖级设定**：中奖得固定奖金 **173 元**。
            * **回血期望**：
              - 排列三每日开奖，属于高频玩法。
              - 一旦中奖一次，得奖金 **173 元**，可直接平扣您过去多期买大乐透的部分累计不中亏损。
              - 建议作为日常"小额高频"回血主力。
            """)
        else:
            st.markdown(f"""
            #### 🛡️ 方案 B 七星彩「七位直选」
            * **中奖率**：单注极低（约千万分之一），属于**低频高奖级**玩法。
            * **奖级设定**：直选为浮动高等奖级，中一次可覆盖数十期大乐透未中亏损。
            * **回血期望**：
              - 属于"搏大奖"型对冲补充，不宜重仓。
              - 与排列三搭配使用，兼顾"日常回血 + 偶发大回血"。
            """)

    st.write("---")
    st.write("### 🎫 您的对冲投资组合号码推荐：")

    col_out_dlt, col_out_hedge = st.columns(2)

    with col_out_dlt:
        st.markdown(f"#### 🔴 核心主推：大乐透 {dlt_bets} 注")
        dlt_groups = predict_dlt(dlt_bets)
        for i, (fronts, backs) in enumerate(dlt_groups, 1):
            front_html = "".join([f"<span class='ssq-red'>{x:02d}</span>" for x in fronts])
            back_html = "".join([f"<span class='ssq-blue'>{x:02d}</span>" for x in backs])
            st.markdown(f"**第 {i:02d} 注**： {front_html}{back_html}", unsafe_allow_html=True)

        st.markdown("**📋 复制大乐透号码**")
        dlt_hedge_text = "-----大乐透主投-----\n" + format_dlt(dlt_groups)
        st.code(dlt_hedge_text, language="text")

    with col_out_hedge:
        if "方案 A" in hedge_strategy:
            st.markdown(f"#### 🟢 组合配比：排列三 组选六 {hedge_bets} 注")
            groups = gen_pl3_group6(hedge_bets)
            for i, nums in enumerate(groups, 1):
                nums_html = "".join([f"<span class='f3d-ball'>{x}</span>" for x in nums])
                st.markdown(f"**第 {i:02d} 注**： {nums_html} (组选六)", unsafe_allow_html=True)

            st.markdown("**📋 复制对冲号码**")
            hedge_text = "----排列三 组选六-----\n" + format_pl3_group6(groups)
            st.code(hedge_text, language="text")
        else:
            st.markdown(f"#### 🟣 组合配比：七星彩 七位直选 {hedge_bets} 注")
            groups = gen_qxc_pick7(hedge_bets)
            for i, nums in enumerate(groups, 1):
                nums_html = "".join([f"<span class='f3d-ball'>{x}</span>" for x in nums])
                st.markdown(f"**第 {i:02d} 注**： {nums_html}", unsafe_allow_html=True)

            st.markdown("**📋 复制对冲号码**")
            hedge_text = "----七星彩 七位-----\n" + format_qxc_pick7(groups)
            st.code(hedge_text, language="text")

    st.write("---")
    st.markdown("### 📋 一键复制完整对冲组合")
    full_text = f"{dlt_hedge_text}\n\n{hedge_text}"
    st.code(full_text, language="text")

    # ===== AI 智能分析对冲策略（体彩版） =====
    st.write("---")
    st.subheader("🤖 AI 智能对冲策略分析")
    if is_ai_configured():
        if st.button("🔍 AI 分析当前方案并推荐最优策略", type="primary", width="stretch"):
            with st.spinner("AI 正在分析历史数据，生成最优对冲策略..."):
                try:
                    hedge_opt = ai_optimize_hedge_sports(dlt_bets, hedge_strategy)
                    if "error" in hedge_opt:
                        st.error(hedge_opt["error"])
                    else:
                        st.markdown(hedge_opt.get("advice", ""))

                        dlt_groups_ai = hedge_opt.get("dlt_groups", [])
                        hedge_groups_ai = hedge_opt.get("hedge_groups", [])
                        hedge_type_ai = hedge_opt.get("hedge_type", "")
                        hedge_name_ai = hedge_opt.get("hedge_name", "")

                        if dlt_groups_ai or hedge_groups_ai:
                            st.markdown("### 🎯 AI 推荐号码组合")

                            col_dlt_ai, col_hedge_ai = st.columns(2)

                            dlt_text_ai = ""
                            dlt_predictions_to_save = []
                            if dlt_groups_ai:
                                with col_dlt_ai:
                                    st.markdown("#### 🔴 大乐透推荐")
                                    dlt_lines = []
                                    for i, item in enumerate(dlt_groups_ai, 1):
                                        fronts = item.get("front", [])
                                        backs = item.get("back", [])
                                        # save_prediction_record("dlt") 期望扁平 {"nums": 前区+后区}
                                        dlt_predictions_to_save.append({"nums": fronts + backs})
                                        front_str = " ".join(f"{x:02d}" for x in fronts)
                                        back_str = " ".join(f"{x:02d}" for x in backs)
                                        dlt_lines.append(f"第{i:02d}注 前区：{front_str} 后区：{back_str}")
                                        front_html = "".join([f"<span class='ssq-red'>{x:02d}</span>" for x in fronts])
                                        back_html = "".join([f"<span class='ssq-blue'>{x:02d}</span>" for x in backs])
                                        st.markdown(f"**第 {i:02d} 注**： {front_html}{back_html}", unsafe_allow_html=True)
                                    dlt_text_ai = "-----AI大乐透推荐-----\n" + "\n".join(dlt_lines)

                            hedge_text_ai = ""
                            sports_hedge_predictions_to_save = []
                            if hedge_groups_ai:
                                with col_hedge_ai:
                                    st.markdown(f"#### 🟢 {hedge_name_ai}")
                                    hedge_lines = []
                                    for i, nums in enumerate(hedge_groups_ai, 1):
                                        nums = list(nums)
                                        sports_hedge_predictions_to_save.append({"nums": nums})
                                        nums_str = " ".join(str(x) for x in nums)
                                        hedge_lines.append(f"第{i:02d}注 {nums_str}")
                                        nums_html = "".join([f"<span class='f3d-ball'>{x}</span>" for x in nums])
                                        st.markdown(f"**第 {i:02d} 注**： {nums_html}", unsafe_allow_html=True)
                                    hedge_text_ai = f"-----AI{hedge_name_ai}推荐-----\n" + "\n".join(hedge_lines)

                            if dlt_text_ai or hedge_text_ai:
                                st.markdown("### 📋 一键复制 AI 推荐号码")
                                full_text_ai = f"{dlt_text_ai}\n\n{hedge_text_ai}".strip()
                                st.code(full_text_ai, language="text")

                            if dlt_predictions_to_save:
                                st.session_state['pending_hedge_dlt_predictions'] = dlt_predictions_to_save

                            if sports_hedge_predictions_to_save and hedge_type_ai:
                                st.session_state['pending_hedge_companion_predictions'] = sports_hedge_predictions_to_save
                                st.session_state['pending_hedge_companion_type'] = hedge_type_ai
                                st.session_state['pending_hedge_companion_name'] = hedge_name_ai
                except Exception as e:
                    st.error(f"AI 分析失败: {e}")

        if 'pending_hedge_dlt_predictions' in st.session_state and st.session_state['pending_hedge_dlt_predictions']:
            if st.button("💾 保存大乐透预测记录", type="secondary", width="stretch"):
                try:
                    latest_code = _db_get_latest_code("dlt")
                    if latest_code:
                        next_code = str(int(latest_code) + 1)
                        _n = len(st.session_state['pending_hedge_dlt_predictions'])
                        logger.info(f"[对冲] 保存大乐透: 期号={next_code}, 组数={_n}")
                        save_prediction_record("dlt", next_code, st.session_state['pending_hedge_dlt_predictions'])
                        _toast_save(f"✅ 大乐透预测记录已保存（第 {next_code} 期）")
                    else:
                        logger.warning("[对冲] 保存大乐透: 无法获取最新期号")
                except Exception as e:
                    _toast_error(f"保存失败: {e}")

        if 'pending_hedge_companion_predictions' in st.session_state and st.session_state['pending_hedge_companion_predictions']:
            hedge_type = st.session_state['pending_hedge_companion_type']
            hedge_name = st.session_state['pending_hedge_companion_name']
            if st.button(f"💾 保存{hedge_name}预测记录", type="secondary", width="stretch"):
                try:
                    latest_code = _db_get_latest_code(hedge_type)
                    if latest_code:
                        next_code = str(int(latest_code) + 1)
                        _n = len(st.session_state['pending_hedge_companion_predictions'])
                        logger.info(f"[对冲] 保存{hedge_name}: 彩种={hedge_type}, 期号={next_code}, 组数={_n}")
                        save_prediction_record(hedge_type, next_code, st.session_state['pending_hedge_companion_predictions'])
                        _toast_save(f"✅ {hedge_name}预测记录已保存（第 {next_code} 期）")
                    else:
                        logger.warning(f"[对冲] 保存{hedge_name}: 无法获取最新期号")
                except Exception as e:
                    _toast_error(f"保存失败: {e}")
    else:
        st.info("💡 未检测到 AI 配置，体彩 AI 对冲分析不可用。请在设置中配置后重试。")


def _render_welfare_hedge():
    st.subheader("🛡️ 双色球「智能配比」组合优化工具")
    st.write("""
    双色球单期中奖率低，很多彩民连续多期无法回血。
    本工具旨在将您的投注资金**科学分流分配**，组合购买**「高频、高中奖概率」**的快乐 8 选一、选四或 3D 组选六。
    **目的**：利用高概率小奖返还的资金，平扣双色球不中奖的成本，实现总体账户避险。
    """)
    
    st.write("---")
    
    col_input, col_math = st.columns([2, 3])
    
    with col_input:
        st.write("### 💰 第一步：输入你的投注配置")
        ssq_bets = st.number_input("核心：双色球单期计划投注（注数）", min_value=1, max_value=50, value=10)
        ssq_cost = ssq_bets * 2
        st.markdown(f"🔴 **双色球主投注额：** **{ssq_cost} 元**")
        
        hedge_strategy = st.selectbox(
            "第二步：选择搭配的对冲避险方案",
            [
                "🛡️ 方案 A（极速回血）：搭配 快乐8 选一（25%中奖率，稳健返还 4.6元/注）",
                "🛡️ 方案 B（阳光普照）：搭配 快乐8 选四（25.89%中奖率，中4个得100元）",
                "🛡️ 方案 C（低频大回血）：搭配 福彩3D 组选六（1/167中奖率，中奖得 173 元）"
            ]
        )
        
        hedge_bets = st.slider("对冲单注数", min_value=1, max_value=20, value=5)
        hedge_cost = hedge_bets * 2
        total_cost = ssq_cost + hedge_cost
        
        st.write("---")
        st.metric("💳 总体组合投资预算", f"{total_cost} 元", f"其中双色球 {ssq_cost} 元，对冲彩 {hedge_cost} 元")
        
    with col_math:
        st.write("### 📊 方案数学期望与回血率分析")
        
        if "方案 A" in hedge_strategy:
            st.markdown(f"""
            #### 🛡️ 方案 A 快乐8「选一」
            * **科学计算中奖率**：单注 **25.00%** 的超高中奖率。
            * **投产分析**：你买了 {hedge_bets} 注（花费 {hedge_cost} 元）。
            * **回血期望**：
              - 预计 **100% 能够中得 1注以上**，至少回血 **4.6 元**。
              - 运气较好中 2 注可回血 **9.2 元**。
              - 回血可直接抵消双色球约 **23% ~ 46%** 的成本！
            """)
        elif "方案 B" in hedge_strategy:
            st.markdown(f"""
            #### 🛡️ 方案 B 快乐8「选四」
            * **科学计算中奖率**：单注中奖率 **25.89%**。
            * **奖级设定**：
              - 4中2 奖金 3 元
              - 4中3 奖金 5 元
              - 4中4 奖金 **100 元**
            * **回血期望**：中 2~3 个球的概率极高，能返还 3~5 元阳光普照奖。一旦中 4 个，立刻净赚 100 元，**不仅全额覆盖双色球未中损失，还能净赚 80 元以上**。
            """)
        else:
            st.markdown(f"""
            #### 🛡️ 方案 C 福彩3D「组选六」
            * **科学计算中奖率**：单注中奖率 **1/167 (约 0.6%)**。
            * **奖级设定**：中奖得固定奖金 **173 元**。
            * **回血期望**：
              - 属于低频高返还方案。
              - 一旦中奖一次，得奖金 **173 元**，可以直接**全额平扣您过去连续 8 期买 10注双色球（共160元）的全部累计不中亏损**！
            """)
            
    st.write("---")
    st.write("### 🎫 您的对冲投资组合号码推荐：")
    
    col_out_ssq, col_out_hedge = st.columns(2)
    
    with col_out_ssq:
        st.markdown(f"#### 🔴 核心主推：双色球 {ssq_bets} 注")
        ssq_groups = predict_ssq(ssq_bets)
        for i, (reds, blue) in enumerate(ssq_groups, 1):
            red_html = "".join([f"<span class='ssq-red'>{x:02d}</span>" for x in reds])
            blue_html = f"<span class='ssq-blue'>{blue:02d}</span>"
            st.markdown(f"**第 {i:02d} 注**： {red_html}{blue_html}", unsafe_allow_html=True)
    
        # 复制双色球号码（自带标题）
        st.markdown("**📋 复制双色球号码**")
        ssq_hedge_text = "-----双色球主投-----\n" + format_ssq(ssq_groups)
        st.code(ssq_hedge_text, language="text")
            
    with col_out_hedge:
        if "方案 A" in hedge_strategy:
            st.markdown(f"#### 🟡 组合配比：快乐8 选一 {hedge_bets} 注")
            nums = gen_kl8_pick1(hedge_bets)
            for i, x in enumerate(nums, 1):
                st.markdown(f"**第 {i:02d} 注**： <span class='kl8-ball'>{x:02d}</span>", unsafe_allow_html=True)
            
            # 复制对冲号码（自带标题）
            st.markdown("**📋 复制对冲号码**")
            hedge_text = "----快乐8 选一-----\n" + format_kl8_pick1(nums)
            st.code(hedge_text, language="text")
                
        elif "方案 B" in hedge_strategy:
            st.markdown(f"#### 🟡 组合配比：快乐8 选四 {hedge_bets} 注")
            groups = gen_kl8_pick4(hedge_bets)
            for i, nums in enumerate(groups, 1):
                nums_html = "".join([f"<span class='kl8-ball'>{x:02d}</span>" for x in nums])
                st.markdown(f"**第 {i:02d} 注**： {nums_html}", unsafe_allow_html=True)
            
            # 复制对冲号码（自带标题）
            st.markdown("**📋 复制对冲号码**")
            hedge_text = "----快乐8 选四-----\n" + format_kl8_pick4(groups)
            st.code(hedge_text, language="text")
                
        else:
            st.markdown(f"#### 🟢 组合配比：福彩3D 组选六 {hedge_bets} 注")
            groups = gen_3d_group6(hedge_bets)
            for i, nums in enumerate(groups, 1):
                nums_html = "".join([f"<span class='f3d-ball'>{x}</span>" for x in nums])
                st.markdown(f"**第 {i:02d} 注**： {nums_html} (组选六)", unsafe_allow_html=True)
            
            # 复制对冲号码（自带标题）
            st.markdown("**📋 复制对冲号码**")
            hedge_text = "----福彩3D 组选六-----\n" + format_3d_group6(groups)
            st.code(hedge_text, language="text")
    
    # ===== 底部：一键复制完整对冲组合 =====
    st.write("---")
    st.markdown("### 📋 一键复制完整对冲组合")
    full_text = f"{ssq_hedge_text}\n\n{hedge_text}"
    st.code(full_text, language="text")
    
    # ===== AI 智能分析对冲策略 =====
    st.write("---")
    st.subheader("🤖 AI 智能对冲策略分析")
    
    if is_ai_configured():
        if st.button("🔍 AI 分析当前方案并推荐最优策略", type="primary", width="stretch"):
            with st.spinner("AI 正在分析历史数据，生成最优对冲策略..."):
                try:
                    hedge_opt = ai_optimize_hedge(ssq_bets, hedge_strategy)
                    if "error" in hedge_opt:
                        st.error(hedge_opt["error"])
                    else:
                        st.markdown(hedge_opt.get("advice", ""))
                        
                        ssq_groups_ai = hedge_opt.get("ssq_groups", [])
                        hedge_groups_ai = hedge_opt.get("hedge_groups", [])
                        hedge_type_ai = hedge_opt.get("hedge_type", "")
                        hedge_name_ai = hedge_opt.get("hedge_name", "")
                        
                        if ssq_groups_ai or hedge_groups_ai:
                            st.markdown("### 🎯 AI 推荐号码组合")
                            
                            col_ssq_ai, col_hedge_ai = st.columns(2)
                            
                            ssq_text_ai = ""
                            ssq_predictions_to_save = []
                            if ssq_groups_ai:
                                with col_ssq_ai:
                                    st.markdown("#### 🔴 双色球推荐")
                                    ssq_lines = []
                                    for i, item in enumerate(ssq_groups_ai, 1):
                                        reds = item.get("red", [])
                                        blue = item.get("blue", 0)
                                        ssq_predictions_to_save.append({"red": reds, "blue": blue})
                                        red_str = " ".join(f"{x:02d}" for x in reds)
                                        ssq_lines.append(f"第{i:02d}注 红球：{red_str} 蓝球：{blue:02d}")
                                        red_html = "".join([f"<span class='ssq-red'>{x:02d}</span>" for x in reds])
                                        blue_html = f"<span class='ssq-blue'>{blue:02d}</span>"
                                        st.markdown(f"**第 {i:02d} 注**： {red_html}{blue_html}", unsafe_allow_html=True)
                                    ssq_text_ai = "-----AI双色球推荐-----\n" + "\n".join(ssq_lines)
                            
                            hedge_text_ai = ""
                            hedge_predictions_to_save = []
                            if hedge_groups_ai:
                                with col_hedge_ai:
                                    if hedge_type_ai == "kl8":
                                        st.markdown(f"#### 🟡 {hedge_name_ai}")
                                        hedge_lines = []
                                        for i, nums in enumerate(hedge_groups_ai, 1):
                                            hedge_predictions_to_save.append({"nums": nums})
                                            nums_str = " ".join(f"{x:02d}" for x in nums)
                                            hedge_lines.append(f"第{i:02d}注 {nums_str}")
                                            nums_html = "".join([f"<span class='kl8-ball'>{x:02d}</span>" for x in nums])
                                            st.markdown(f"**第 {i:02d} 注**： {nums_html}", unsafe_allow_html=True)
                                        hedge_text_ai = f"-----AI{hedge_name_ai}推荐-----\n" + "\n".join(hedge_lines)
                                    else:
                                        st.markdown(f"#### 🟢 {hedge_name_ai}")
                                        hedge_lines = []
                                        for i, nums in enumerate(hedge_groups_ai, 1):
                                            hedge_predictions_to_save.append({"nums": nums})
                                            nums_str = " ".join(str(x) for x in nums)
                                            hedge_lines.append(f"第{i:02d}注 {nums_str}")
                                            nums_html = "".join([f"<span class='f3d-ball'>{x}</span>" for x in nums])
                                            st.markdown(f"**第 {i:02d} 注**： {nums_html}", unsafe_allow_html=True)
                                        hedge_text_ai = f"-----AI{hedge_name_ai}推荐-----\n" + "\n".join(hedge_lines)
                            
                            if ssq_text_ai or hedge_text_ai:
                                st.markdown("### 📋 一键复制 AI 推荐号码")
                                full_text_ai = f"{ssq_text_ai}\n\n{hedge_text_ai}".strip()
                                st.code(full_text_ai, language="text")
                                
                            if ssq_predictions_to_save:
                                st.session_state['pending_ssq_predictions'] = ssq_predictions_to_save
                            
                            if hedge_predictions_to_save and hedge_type_ai:
                                st.session_state['pending_hedge_predictions'] = hedge_predictions_to_save
                                st.session_state['pending_hedge_type'] = hedge_type_ai
                                st.session_state['pending_hedge_name'] = hedge_name_ai
                except Exception as e:
                    st.error(f"AI 分析失败: {e}")
        
        if 'pending_ssq_predictions' in st.session_state and st.session_state['pending_ssq_predictions']:
            if st.button("💾 保存双色球预测记录", type="secondary", width="stretch"):
                try:
                    latest_code = _db_get_latest_code("ssq")
                    if latest_code:
                        next_code = str(int(latest_code) + 1)
                        _n = len(st.session_state['pending_ssq_predictions'])
                        logger.info(f"[对冲] 保存双色球: 期号={next_code}, 组数={_n}")
                        save_prediction_record("ssq", next_code, st.session_state['pending_ssq_predictions'])
                        _toast_save(f"✅ 双色球预测记录已保存（第 {next_code} 期）")
                    else:
                        logger.warning("[对冲] 保存双色球: 无法获取最新期号")
                except Exception as e:
                    _toast_error(f"保存失败: {e}")
        
        if 'pending_hedge_predictions' in st.session_state and st.session_state['pending_hedge_predictions']:
            hedge_type = st.session_state['pending_hedge_type']
            hedge_name = st.session_state['pending_hedge_name']
            if st.button(f"💾 保存{hedge_name}预测记录", type="secondary", width="stretch"):
                try:
                    latest_code = _db_get_latest_code(hedge_type)
                    if latest_code:
                        next_code = str(int(latest_code) + 1)
                        _n = len(st.session_state['pending_hedge_predictions'])
                        logger.info(f"[对冲] 保存{hedge_name}: 彩种={hedge_type}, 期号={next_code}, 组数={_n}")
                        save_prediction_record(hedge_type, next_code, st.session_state['pending_hedge_predictions'])
                        _toast_save(f"✅ {hedge_name}预测记录已保存（第 {next_code} 期）")
                    else:
                        logger.warning(f"[对冲] 保存{hedge_name}: 无法获取最新期号")
                except Exception as e:
                    _toast_error(f"保存失败: {e}")
    else:
        st.info("💡 请在侧边栏配置 AI API Key，启用 AI 智能分析功能")
    
if selected_page == "config":
    st.subheader("⚙️ AI 模型配置中心")
    
    with st.container():
        # 检测云部署环境
        from ai_predict import is_cloud_deployed
        cloud_mode = is_cloud_deployed()
        
        if cloud_mode:
            st.success("☁️ 检测到 Streamlit Secrets 配置，AI 已就绪（云端模式）")
            st.info("💡 云端部署通过 Settings → Secrets 配置，无需在此手动输入")
        else:
            st.info("💡 配置后可使用 AI 深度分析功能，提升预测科学性（本地模式）")
        
        # 读取当前配置作为默认值
        _current_cfg = load_config()
        
        api_key = st.text_input(
            "LLM API Key",
            value="" if cloud_mode else "",
            type="password",
            placeholder="sk-xxxxxxxxxxxxxxxx（云端已通过 Secrets 配置，可留空）" if cloud_mode else "sk-xxxxxxxxxxxxxxxx",
            help="OpenAI 兼容接口的 API Key",
            disabled=cloud_mode
        )
        base_url = st.text_input(
            "Base URL",
            value=_current_cfg.get("base_url", "https://api.siliconflow.cn/v1"),
            placeholder="https://api.openai.com/v1",
            help="OpenAI 兼容的 API 地址",
            disabled=cloud_mode
        )
        model_name = st.text_input(
            "模型名称",
            value=_current_cfg.get("model", "deepseek-ai/DeepSeek-R1"),
            placeholder="deepseek-ai/DeepSeek-R1",
            help="支持的模型：DeepSeek-R1, Qwen, GLM 等",
            disabled=cloud_mode
        )
        
        if not cloud_mode:
            col_test, col_save = st.columns(2)
            with col_test:
                if st.button("🧪 测试连接", width="stretch"):
                    if not api_key:
                        st.error("请先输入 API Key")
                    else:
                        try:
                            from ai_predict import test_ai_connection
                            msg = test_ai_connection(api_key, base_url, model_name)
                            if msg.startswith("✅"):
                                st.success(msg)
                            else:
                                st.error(msg)
                        except Exception as e:
                            st.error(f"测试失败: {e}")
            with col_save:
                if st.button("💾 保存配置", width="stretch"):
                    if not api_key:
                        st.error("API Key 不能为空")
                    else:
                        try:
                            from ai_predict import save_ai_config
                            save_ai_config(api_key, base_url, model_name)
                            st.success("✅ AI 配置已保存！正在刷新页面...")
                            st.rerun()
                        except Exception as e:
                            st.error(f"保存失败: {e}")
        else:
            st.write("---")
            st.markdown("**📋 Streamlit Secrets 配置格式**")
            st.code('''[ai_config]
api_key = "sk-xxxxxxxxxxxxxxxx"
base_url = "https://api.siliconflow.cn/v1"
model = "deepseek-ai/DeepSeek-R1"''', language="toml")

elif selected_page == "dashboard":
    st.subheader("📊 自发行开奖以来完整开奖数据总览")
    
    # 彩种数据加载（跟随左侧菜单选择）
    current_name = st.session_state.get('selected_lottery', 'ssq')
    lot_display = {
        "ssq": "双色球", "kl8": "快乐8", "fcsd": "福彩3D",
        "dlt": "大乐透", "qxc": "七星彩", "pl3": "排列三"
    }
    view_name = f"{lot_display.get(current_name, current_name)} ({current_name})"
    st.info(f"📊 当前查看彩种：**{lot_display.get(current_name, current_name)}**（可在左侧「🎰 选择彩种」菜单切换）")

    df = _db_get_lottery_df(current_name, dtype={"code": str})

    if not df.empty:
        
        # 指标展示
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🏆 本地收录期数", f"{len(df)} 期")
        with col2:
            st.metric("🕒 最新开奖期号", f"{df.iloc[0]['code']} 期")
        with col3:
            st.metric("📅 最新开奖日期", f"{df.iloc[0]['date']}")
            
        # ===== 最新开奖详情 =====
        st.write("---")
        st.subheader("🎰 最新开奖详情")
        
        latest = df.iloc[0]
        
        if current_name == "ssq":
            reds = [latest['r1'], latest['r2'], latest['r3'], 
                    latest['r4'], latest['r5'], latest['r6']]
            blue = latest['blue']
            red_html = "".join([f"<span class='ssq-red'>{x:02d}</span>" for x in reds])
            blue_html = f"<span class='ssq-blue'>{blue:02d}</span>"
            st.markdown(f"**第 {latest['code']} 期** | 日期：{latest['date']}", unsafe_allow_html=True)
            st.markdown(f"红球：{red_html} 蓝球：{blue_html}", unsafe_allow_html=True)
            
        elif current_name == "kl8":
            cols = [f"n{i:02d}" for i in range(1, 21)]
            nums = [latest[col] for col in cols if col in latest]
            nums_html = "".join([f"<span class='kl8-ball'>{x:02d}</span>" for x in nums])
            st.markdown(f"**第 {latest['code']} 期** | 日期：{latest['date']}", unsafe_allow_html=True)
            st.markdown(f"开奖号码：{nums_html}", unsafe_allow_html=True)
            
        elif current_name == "dlt":
            fronts = [latest['f1'], latest['f2'], latest['f3'], latest['f4'], latest['f5']]
            backs = [latest['b1'], latest['b2']]
            f_html = "".join([f"<span class='ssq-red'>{x:02d}</span>" for x in fronts])
            b_html = "".join([f"<span class='ssq-blue'>{x:02d}</span>" for x in backs])
            st.markdown(f"**第 {latest['code']} 期** | 日期：{latest['date']}", unsafe_allow_html=True)
            st.markdown(f"前区：{f_html} 后区：{b_html}", unsafe_allow_html=True)
            
        elif current_name == "qxc":
            qxc_cols = ['n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7']
            nums = [latest[col] for col in qxc_cols if col in latest]
            nums_html = " ".join([f"<span class='f3d-ball'>{x}</span>" for x in nums])
            st.markdown(f"**第 {latest['code']} 期** | 日期：{latest['date']}", unsafe_allow_html=True)
            st.markdown(f"开奖号码：{nums_html}", unsafe_allow_html=True)
            
        elif current_name == "pl3":
            n1, n2, n3 = latest['n1'], latest['n2'], latest['n3']
            st.markdown(f"**第 {latest['code']} 期** | 日期：{latest['date']}", unsafe_allow_html=True)
            st.markdown(
                f"百位：<span class='f3d-ball'>{n1}</span> "
                f"十位：<span class='f3d-ball'>{n2}</span> "
                f"个位：<span class='f3d-ball'>{n3}</span>",
                unsafe_allow_html=True
            )
            
        else:
            n1, n2, n3 = latest['n1'], latest['n2'], latest['n3']
            st.markdown(f"**第 {latest['code']} 期** | 日期：{latest['date']}", unsafe_allow_html=True)
            st.markdown(
                f"百位：<span class='f3d-ball'>{n1}</span> "
                f"十位：<span class='f3d-ball'>{n2}</span> "
                f"个位：<span class='f3d-ball'>{n3}</span>",
                unsafe_allow_html=True
            )
            
        # ===== AI 对比中奖数据 =====
        st.write("---")
        st.subheader("🤖 AI 预测 vs 实际开奖对比")
        
        records = get_prediction_records(current_name)
        if records:
            st.info(f"📊 已保存 {len(records)} 期预测记录")
            
            if st.button("📋 查看已保存的预测号码", width="stretch"):
                filtered_records = records
                
                if not filtered_records:
                    st.warning("未找到预测记录")
                else:
                    st.info(f"� 共找到 {len(filtered_records)} 条预测记录")
                    for record in filtered_records:
                        st.write(f"---")
                        st.markdown(f"**🎟️ 第 {record['code']} 期** | 🕐 {record['predict_time']}")
                        if record.get('compared'):
                            st.markdown(f"✅ 已对比开奖结果")
                        else:
                            st.markdown(f"⏳ 等待开奖")
                        
                        predictions = record.get('predictions', [])
                        if record['lottery_type'] == 'ssq':
                            for i, pred in enumerate(predictions, 1):
                                # 容错：历史旧数据可能用 nums 存红球
                                if 'red' in pred:
                                    reds = pred.get('red', [])
                                    blue = pred.get('blue', 0)
                                elif 'nums' in pred and isinstance(pred.get('nums'), list):
                                    _tmp = pred.get('nums', [])
                                    reds = _tmp[:6]
                                    blue = pred.get('blue', _tmp[6] if len(_tmp) > 6 else 0)
                                else:
                                    reds = []
                                    blue = 0
                                red_html = "".join([f"<span class='ssq-red'>{x:02d}</span>" for x in reds])
                                blue_html = f"<span class='ssq-blue'>{blue:02d}</span>"
                                st.markdown(f"第 {i:02d} 组： {red_html} {blue_html}", unsafe_allow_html=True)
                        elif record['lottery_type'] == 'kl8':
                            for i, pred in enumerate(predictions, 1):
                                nums = pred.get('nums', [])
                                nums_html = "".join([f"<span class='kl8-ball'>{x:02d}</span>" for x in nums])
                                st.markdown(f"第 {i:02d} 组： {nums_html}", unsafe_allow_html=True)
                        else:
                            for i, pred in enumerate(predictions, 1):
                                nums = pred.get('nums', [])
                                nums_str = " ".join(f"{x:02d}" for x in nums)
                                st.markdown(f"第 {i:02d} 组： {nums_str}")
        
        if not is_ai_configured():
            st.warning("⚠️ AI 功能尚未配置，请在左侧边栏配置 API Key 后使用此功能。")
        elif not records:
            st.info("ℹ️ 暂无该彩种的 AI 预测记录。请先在「智能号码预测」页生成并保存预测号码，再回来对比开奖结果。")
        else:
            st.markdown("#### 📅 选择要对比的预测期号")
            available_codes = [str(r['code']) for r in records]
            selected_code = st.selectbox(
                "从已保存的预测记录中选择期号：",
                options=available_codes,
                index=0,
                key=f"compare_code_{current_name}",
                help="选择后点击下方的对比按钮，即可查看该期预测与实际开奖的对照结果。"
            )
            if st.button("🔍 对比已保存的预测记录与实际开奖", type="primary", width="stretch"):
                with st.spinner("正在分析已保存的预测记录与实际开奖的对比..."):
                    try:
                        compare_result = analyze_saved_predictions(current_name, selected_code)
                        
                        if "error" in compare_result:
                            st.error(compare_result["error"])
                            if "available_codes" in compare_result:
                                st.info(f"可用预测记录期号：{', '.join(compare_result['available_codes'][:10])}")
                        else:
                            latest_data = compare_result["latest"]
                            ai_best = compare_result["ai_best"]
                            predict_time = compare_result.get("predict_time", "")
                            
                            st.markdown(f"**开奖号码：第 {latest_data['code']} 期 ({latest_data['date']})**")
                            if predict_time:
                                st.markdown(f"🕐 预测时间：{predict_time}")
                            
                            col_actual, col_ai = st.columns(2)
                            
                            with col_actual:
                                st.markdown("#### 🎯 实际开奖号码")
                                if current_name == "ssq":
                                    red_html = "".join([f"<span class='ssq-red'>{x:02d}</span>" for x in latest_data['reds']])
                                    blue_html = f"<span class='ssq-blue'>{latest_data['blue']:02d}</span>"
                                    st.markdown(f"红球：{red_html}", unsafe_allow_html=True)
                                    st.markdown(f"蓝球：{blue_html}", unsafe_allow_html=True)
                                elif current_name == "dlt":
                                    front_html = "".join([f"<span class='ssq-red'>{x:02d}</span>" for x in latest_data['fronts']])
                                    back_html = "".join([f"<span class='ssq-blue'>{x:02d}</span>" for x in latest_data['backs']])
                                    st.markdown(f"前区：{front_html}", unsafe_allow_html=True)
                                    st.markdown(f"后区：{back_html}", unsafe_allow_html=True)
                                elif current_name == "kl8":
                                    nums_html = "".join([f"<span class='kl8-ball'>{x:02d}</span>" for x in latest_data['nums']])
                                    st.markdown(f"开奖号码：{nums_html}", unsafe_allow_html=True)
                                elif current_name == "qxc":
                                    nums_html = " ".join([f"<span class='f3d-ball'>{x}</span>" for x in latest_data['nums']])
                                    st.markdown(f"开奖号码：{nums_html}", unsafe_allow_html=True)
                                else:
                                    # fcsd / pl3
                                    st.markdown(
                                        f"<span class='f3d-ball'>{latest_data['nums'][0]}</span>"
                                        f"<span class='f3d-ball'>{latest_data['nums'][1]}</span>"
                                        f"<span class='f3d-ball'>{latest_data['nums'][2]}</span>",
                                        unsafe_allow_html=True
                                    )
                            
                            with col_ai:
                                st.markdown("#### 🤖 AI 预测最佳命中")
                                if current_name == "ssq":
                                    ai_nums = ai_best["nums"]
                                    ai_reds = ai_nums.get("red", [])
                                    ai_blue = ai_nums.get("blue", 0)
                                    red_html = "".join([f"<span class='ssq-red'>{x:02d}</span>" for x in ai_reds])
                                    blue_html = f"<span class='ssq-blue'>{ai_blue:02d}</span>"
                                    st.markdown(f"红球：{red_html}", unsafe_allow_html=True)
                                    st.markdown(f"蓝球：{blue_html}", unsafe_allow_html=True)
                                    st.markdown(f"✅ 红球命中：{ai_best['red_matches']} 个")
                                    st.markdown(f"✅ 蓝球命中：{'是' if ai_best['blue_match'] else '否'}")
                                elif current_name == "dlt":
                                    ai_nums = ai_best["nums"]
                                    ai_fronts = ai_nums[:5] if len(ai_nums) >= 5 else ai_nums
                                    ai_backs = ai_nums[5:] if len(ai_nums) > 5 else []
                                    front_html = "".join([f"<span class='ssq-red'>{x:02d}</span>" for x in ai_fronts])
                                    back_html = "".join([f"<span class='ssq-blue'>{x:02d}</span>" for x in ai_backs])
                                    st.markdown(f"前区：{front_html}", unsafe_allow_html=True)
                                    st.markdown(f"后区：{back_html}", unsafe_allow_html=True)
                                    st.markdown(f"✅ 前区命中：{ai_best['front_matches']} 个")
                                    st.markdown(f"✅ 后区命中：{ai_best['back_matches']} 个")
                                elif current_name == "kl8":
                                    ai_nums = ai_best["nums"]
                                    nums_html = "".join([f"<span class='kl8-ball'>{x:02d}</span>" for x in ai_nums])
                                    st.markdown(f"预测号码：{nums_html}", unsafe_allow_html=True)
                                    st.markdown(f"✅ 命中号码：{ai_best['matches']} 个")
                                elif current_name == "qxc":
                                    ai_nums = ai_best["nums"]
                                    nums_html = " ".join([f"<span class='f3d-ball'>{x}</span>" for x in ai_nums])
                                    st.markdown(f"预测号码：{nums_html}", unsafe_allow_html=True)
                                    st.markdown(f"✅ 命中位数：{ai_best['matches']} 位")
                                else:
                                    # fcsd / pl3
                                    ai_nums = ai_best["nums"]
                                    if len(ai_nums) >= 3:
                                        st.markdown(
                                            f"<span class='f3d-ball'>{ai_nums[0]}</span>"
                                            f"<span class='f3d-ball'>{ai_nums[1]}</span>"
                                            f"<span class='f3d-ball'>{ai_nums[2]}</span>",
                                            unsafe_allow_html=True
                                        )
                                    st.markdown(f"✅ 命中位数：{ai_best['matches']} 位")
                                
                    except Exception as e:
                        st.error(f"分析失败: {e}")
        
        # ===== 投注报表 =====
        st.write("---")
        st.subheader("💰 AI 投注报表")
        
        report = get_betting_report(current_name)
        
        if "error" in report:
            st.info(report["error"])
        else:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🎯 总投注注数", f"{report['total_bets']} 注")
            with col2:
                st.metric("💸 总投入", f"¥{report['total_cost']}")
            with col3:
                st.metric("🎁 总奖金", f"¥{report['total_prize']}")
            
            col4, col5, col6 = st.columns(3)
            with col4:
                profit_color = "green" if report["total_profit"] >= 0 else "red"
                st.metric("📈 总盈亏", f"¥{report['total_profit']}", 
                         delta=f"{report['avg_profit_per_bet']:.2f} 元/注")
            with col5:
                st.metric("🏆 中奖率", f"{report['win_rate']:.1f}%",
                         delta=f"{report['win_count']} 期中奖")
            with col6:
                st.metric("📊 收益率", f"{report['profit_rate']:.1f}%")
            
            if report["records"]:
                st.write("### 📋 投注明细")
                report_df = pd.DataFrame(report["records"])
                report_df["profit"] = report_df["profit"].apply(lambda x: f"¥{x}" if x >= 0 else f"-¥{abs(x)}")
                report_df["profit_rate"] = report_df["profit_rate"].apply(lambda x: f"{x:.1f}%" if x >= 0 else f"-{abs(x):.1f}%")
                report_df["won"] = report_df["won"].apply(lambda x: "✅" if x else "❌")
                report_df = report_df[["code", "date", "lottery_type", "bets", "cost", "prize", "profit", "profit_rate", "won"]]
                report_df.columns = ["期号", "日期", "彩种", "注数", "投入", "奖金", "盈亏", "收益率", "中奖"]
                st.dataframe(report_df, width="stretch")
            
            if report["total_bets"] > 0:
                chart_data = pd.DataFrame(report["records"])
                chart_data["cumulative_profit"] = chart_data["profit"].cumsum()
                st.write("### 📈 累计盈亏走势图")
                st.line_chart(chart_data[["date", "cumulative_profit"]].set_index("date"))

        st.write("### 📌 历史开奖明细")
        st.dataframe(df, width="stretch")
        
        # 数据可视化
        st.write("### 📊 历史出号频率热度图 (走势热图)")
        if current_name == "ssq":
            all_reds = pd.concat([df['r1'], df['r2'], df['r3'], df['r4'], df['r5'], df['r6']])
            red_counts = all_reds.value_counts().sort_index()
            blue_counts = df['blue'].value_counts().sort_index()
            
            st.write("**红球历史出号频次分布：**")
            st.bar_chart(red_counts)
            
            st.write("**蓝球历史出号频次分布：**")
            st.bar_chart(blue_counts)
            
        elif current_name == "kl8":
            cols = [f"n{i:02d}" for i in range(1, 21)]
            all_nums = pd.concat([df[col] for col in cols if col in df.columns])
            kl8_counts = all_nums.value_counts().sort_index()
            st.write("**快乐 8 各号码历史出现总频次：**")
            st.bar_chart(kl8_counts)
            
        elif current_name == "fcsd":
            n1_counts = df['n1'].value_counts().sort_index()
            n2_counts = df['n2'].value_counts().sort_index()
            n3_counts = df['n3'].value_counts().sort_index()
            
            c_pos = st.radio("选择查看的数位", ["百位", "十位", "个位"], horizontal=True)
            if c_pos == "百位":
                st.bar_chart(n1_counts)
            elif c_pos == "十位":
                st.bar_chart(n2_counts)
            else:
                st.bar_chart(n3_counts)
                
        # 下载数据按钮
        st.download_button(
            label=f"💾 下载 {view_name} 完整历史数据",
            data=df.to_csv(index=False).encode('utf-8-sig'),
            file_name=f"{current_name}_history.csv",
            mime="text/csv",
            width="stretch"
        )
    else:
        st.warning("⚠️ 暂无本地历史数据，请点击左侧控制中心的“立即同步最新数据”进行拉取。")

elif selected_page == "predict":
    st.subheader("🎯 智能预测模型（热温冷概率配比抽样）")
    cur_cat = st.session_state.get('lottery_category', 'welfare')
    ratio_desc = {
        "ssq": "双色球采用 3:2:1 比例组合",
        "kl8": "快乐 8 选十采用 5:3:2 组合",
        "fcsd": "福彩 3D 按位热温冷配比",
        "dlt": "大乐透前区 3:2 搭配后区热温冷",
        "qxc": "七星彩按位热温冷配比",
        "pl3": "排列三按位热温冷配比",
    }
    cat_lots = LOT_CATS[cur_cat]['lots']
    ratio_text = "，".join(ratio_desc[l] for l in cat_lots if l in ratio_desc)
    st.info(f"💡 **科学选号原理（{LOT_CATS[cur_cat]['label']}）**：根据自首发开奖至今的历史数据频次，自动划分为「热码」、「温码」、「冷码」。{ratio_text}，有效规避不平衡选号！")
    
    col_cnt, _ = st.columns([2, 4])
    with col_cnt:
        n_groups = st.slider("每种彩票生成组数", min_value=1, max_value=10, value=5)
    
    # ===== AI 智能预测（按顶部大类动态显示对应彩种）=====
    if is_ai_configured():
        st.markdown(f"### 🤖 AI 智能预测（{LOT_CATS[cur_cat]['label']}）")

        if cur_cat == 'welfare':
            col_ssq_ai, col_kl8_ai, col_f3d_ai = st.columns(3)

            with col_ssq_ai:
                st.markdown("#### 🔴 双色球")
                if st.button("🔮 AI预测双色球", width="stretch"):
                    with st.spinner("AI正在分析双色球趋势..."):
                        ai_ssq = ai_predict_ssq(n_groups)
                        if "error" in ai_ssq:
                            st.error(ai_ssq["error"])
                        else:
                            ai_numbers = ai_ssq.get("recommendations", [])
                            if ai_numbers:
                                ssq_lines = []
                                predictions_to_save = []
                                _dropped = 0
                                for i, rec in enumerate(ai_numbers[:n_groups], 1):
                                    reds = rec.get("numbers", {}).get("red", [])
                                    blue = rec.get("numbers", {}).get("blue", 0)
                                    clean, ok = _validate_ai_group("ssq", rec)
                                    if reds and blue:
                                        red_str = " ".join(f"{x:02d}" for x in reds)
                                        ssq_lines.append(f"第{i:02d}注 红球：{red_str} 蓝球：{blue:02d}")
                                        red_html = "".join([f"<span class='ssq-red'>{x:02d}</span>" for x in reds])
                                        blue_html = f"<span class='ssq-blue'>{blue:02d}</span>"
                                        st.markdown(f"**第 {i:02d} 组**： {red_html} {blue_html}", unsafe_allow_html=True)
                                    if ok:
                                        predictions_to_save.append(clean)
                                    else:
                                        _dropped += 1
                                if _dropped:
                                    st.warning(f"⚠️ 已忽略 {_dropped} 组不符合规则的号码（范围/个数/重复校验未通过）")
                                if ssq_lines:
                                    st.code("\n".join(ssq_lines), language="text")
                                    st.session_state['pending_ssq_predictions'] = predictions_to_save

            if 'pending_ssq_predictions' in st.session_state and st.session_state['pending_ssq_predictions']:
                if st.button("💾 保存双色球预测", width="stretch"):
                    try:
                        latest_code = _db_get_latest_code("ssq")
                        if latest_code:
                            next_code = str(int(latest_code) + 1)
                            _n = len(st.session_state['pending_ssq_predictions'])
                            logger.info(f"[AI预测] 保存双色球: 期号={next_code}, 组数={_n}")
                            save_prediction_record("ssq", next_code, st.session_state['pending_ssq_predictions'])
                            _toast_save(f"✅ 已保存（第 {next_code} 期）")
                        else:
                            logger.warning("[AI预测] 保存双色球: 无法获取最新期号")
                    except Exception as e:
                        _toast_error(f"保存失败: {e}")

            with col_kl8_ai:
                st.markdown("#### 🟡 快乐8")
                _kl8_play_options = {"选十 (10个号)": 10, "选九 (9个号)": 9, "选八 (8个号)": 8,
                                     "选七 (7个号)": 7, "选六 (6个号)": 6, "选五 (5个号)": 5,
                                     "选四 (4个号)": 4, "选三 (3个号)": 3, "选二 (2个号)": 2, "选一 (1个号)": 1}
                _kl8_play_sel = st.selectbox("玩法", list(_kl8_play_options.keys()),
                                             index=0, key="kl8_play_type",
                                             help="快乐8可选选一~选十，选十最主流（中10个=500万，综合中奖率11%）")
                _kl8_pick_size = _kl8_play_options[_kl8_play_sel]
                st.caption(f"当前：{_kl8_play_sel}，每组选 {_kl8_pick_size} 个号码")
                if st.button("🔮 AI预测快乐8", width="stretch"):
                    with st.spinner(f"AI正在分析快乐8({_kl8_play_sel})趋势..."):
                        ai_kl8 = ai_predict_kl8(n_groups, pick_size=_kl8_pick_size)
                        if "error" in ai_kl8:
                            st.error(ai_kl8["error"])
                        else:
                            _kl8_play_name = ai_kl8.get("play_name", _kl8_play_sel)
                            ai_numbers = ai_kl8.get("recommendations", [])
                            if ai_numbers:
                                kl8_lines = []
                                predictions_to_save = []
                                _dropped = 0
                                for i, rec in enumerate(ai_numbers[:n_groups], 1):
                                    nums = rec.get("numbers", [])
                                    clean, ok = _validate_ai_group("kl8", rec, pick_size=_kl8_pick_size)
                                    if nums:
                                        nums_str = " ".join(f"{x:02d}" for x in nums)
                                        kl8_lines.append(f"第{i:02d}注 [{_kl8_play_name}] {nums_str}")
                                        nums_html = "".join([f"<span class='kl8-ball'>{x:02d}</span>" for x in nums])
                                        st.markdown(f"**第 {i:02d} 组** [{_kl8_play_name}]： {nums_html}", unsafe_allow_html=True)
                                    if ok:
                                        predictions_to_save.append(clean)
                                    else:
                                        _dropped += 1
                                if _dropped:
                                    st.warning(f"⚠️ 已忽略 {_dropped} 组不符合规则的号码（范围/个数/重复校验未通过）")
                                if kl8_lines:
                                    st.code("\n".join(kl8_lines), language="text")
                                    st.session_state['pending_kl8_predictions'] = predictions_to_save
                                    st.session_state['pending_kl8_play_name'] = _kl8_play_name

            if 'pending_kl8_predictions' in st.session_state and st.session_state['pending_kl8_predictions']:
                _kl8_pn = st.session_state.get('pending_kl8_play_name', '选十')
                if st.button(f"💾 保存快乐8预测({_kl8_pn})", width="stretch"):
                    try:
                        latest_code = _db_get_latest_code("kl8")
                        if latest_code:
                            next_code = str(int(latest_code) + 1)
                            _n = len(st.session_state['pending_kl8_predictions'])
                            logger.info(f"[AI预测] 保存快乐8: 玩法={_kl8_pn}, 期号={next_code}, 组数={_n}")
                            save_prediction_record("kl8", next_code, st.session_state['pending_kl8_predictions'])
                            _toast_save(f"✅ 已保存 {_kl8_pn} 预测（第 {next_code} 期）")
                        else:
                            logger.warning("[AI预测] 保存快乐8: 无法获取最新期号")
                    except Exception as e:
                        _toast_error(f"保存失败: {e}")

            with col_f3d_ai:
                st.markdown("#### 🟢 福彩3D")
                if st.button("🔮 AI预测福彩3D", width="stretch"):
                    with st.spinner("AI正在分析福彩3D趋势..."):
                        ai_fcsd = ai_predict_fcsd(n_groups)
                        if "error" in ai_fcsd:
                            st.error(ai_fcsd["error"])
                        else:
                            ai_numbers = ai_fcsd.get("recommendations", [])
                            if ai_numbers:
                                fcsd_lines = []
                                predictions_to_save = []
                                _dropped = 0
                                for i, rec in enumerate(ai_numbers[:n_groups], 1):
                                    nums = rec.get("numbers", [])
                                    clean, ok = _validate_ai_group("fcsd", rec)
                                    if nums and len(nums) >= 3:
                                        nums_str = " ".join(str(x) for x in nums)
                                        fcsd_lines.append(f"第{i:02d}注 {nums_str}")
                                        st.markdown(
                                            f"**第 {i:02d} 组**： "
                                            f"<span class='f3d-ball'>{nums[0]}</span>"
                                            f"<span class='f3d-ball'>{nums[1]}</span>"
                                            f"<span class='f3d-ball'>{nums[2]}</span>",
                                            unsafe_allow_html=True
                                        )
                                    if ok:
                                        predictions_to_save.append(clean)
                                    else:
                                        _dropped += 1
                                if _dropped:
                                    st.warning(f"⚠️ 已忽略 {_dropped} 组不符合规则的号码（范围/个数校验未通过）")
                                if fcsd_lines:
                                    st.code("\n".join(fcsd_lines), language="text")
                                    st.session_state['pending_fcsd_predictions'] = predictions_to_save

            if 'pending_fcsd_predictions' in st.session_state and st.session_state['pending_fcsd_predictions']:
                if st.button("💾 保存福彩3D预测", width="stretch"):
                    try:
                        latest_code = _db_get_latest_code("fcsd")
                        if latest_code:
                            next_code = str(int(latest_code) + 1)
                            _n = len(st.session_state['pending_fcsd_predictions'])
                            logger.info(f"[AI预测] 保存福彩3D: 期号={next_code}, 组数={_n}")
                            save_prediction_record("fcsd", next_code, st.session_state['pending_fcsd_predictions'])
                            _toast_save(f"✅ 已保存（第 {next_code} 期）")
                        else:
                            logger.warning("[AI预测] 保存福彩3D: 无法获取最新期号")
                    except Exception as e:
                        _toast_error(f"保存失败: {e}")
        else:
            col_dlt_ai, col_qxc_ai, col_pl3_ai = st.columns(3)

            with col_dlt_ai:
                st.markdown("#### 🔵 大乐透")
                if st.button("🔮 AI预测大乐透", width="stretch"):
                    with st.spinner("AI正在分析大乐透趋势..."):
                        ai_dlt = ai_predict_dlt(n_groups)
                        if "error" in ai_dlt:
                            st.error(ai_dlt["error"])
                        else:
                            ai_numbers = ai_dlt.get("recommendations", [])
                            if ai_numbers:
                                dlt_lines = []
                                predictions_to_save = []
                                _dropped = 0
                                for i, rec in enumerate(ai_numbers[:n_groups], 1):
                                    fronts = rec.get("numbers", {}).get("front", [])
                                    backs = rec.get("numbers", {}).get("back", [])
                                    clean, ok = _validate_ai_group("dlt", rec)
                                    if fronts and backs:
                                        f_str = " ".join(f"{x:02d}" for x in fronts)
                                        b_str = " ".join(f"{x:02d}" for x in backs)
                                        dlt_lines.append(f"第{i:02d}注 前区：{f_str} 后区：{b_str}")
                                        f_html = "".join([f"<span class='ssq-red'>{x:02d}</span>" for x in fronts])
                                        b_html = "".join([f"<span class='ssq-blue'>{x:02d}</span>" for x in backs])
                                        st.markdown(f"**第 {i:02d} 组**： {f_html} {b_html}", unsafe_allow_html=True)
                                    if ok:
                                        predictions_to_save.append(clean)
                                    else:
                                        _dropped += 1
                                if _dropped:
                                    st.warning(f"⚠️ 已忽略 {_dropped} 组不符合规则的号码（范围/个数/重复校验未通过）")
                                if dlt_lines:
                                    st.code("\n".join(dlt_lines), language="text")
                                    st.session_state['pending_dlt_predictions'] = predictions_to_save

            if 'pending_dlt_predictions' in st.session_state and st.session_state['pending_dlt_predictions']:
                if st.button("💾 保存大乐透预测", width="stretch"):
                    try:
                        latest_code = _db_get_latest_code("dlt")
                        if latest_code:
                            next_code = str(int(latest_code) + 1)
                            _n = len(st.session_state['pending_dlt_predictions'])
                            logger.info(f"[AI预测] 保存大乐透: 期号={next_code}, 组数={_n}")
                            save_prediction_record("dlt", next_code, st.session_state['pending_dlt_predictions'])
                            _toast_save(f"✅ 已保存（第 {next_code} 期）")
                        else:
                            logger.warning("[AI预测] 保存大乐透: 无法获取最新期号")
                    except Exception as e:
                        _toast_error(f"保存失败: {e}")

            with col_qxc_ai:
                st.markdown("#### 🟣 七星彩")
                if st.button("🔮 AI预测七星彩", width="stretch"):
                    with st.spinner("AI正在分析七星彩趋势..."):
                        ai_qxc = ai_predict_qxc(n_groups)
                        if "error" in ai_qxc:
                            st.error(ai_qxc["error"])
                        else:
                            ai_numbers = ai_qxc.get("recommendations", [])
                            if ai_numbers:
                                qxc_lines = []
                                predictions_to_save = []
                                _dropped = 0
                                for i, rec in enumerate(ai_numbers[:n_groups], 1):
                                    nums = rec.get("numbers", [])
                                    clean, ok = _validate_ai_group("qxc", rec)
                                    if nums:
                                        nums_str = " ".join(str(x) for x in nums)
                                        qxc_lines.append(f"第{i:02d}注 {nums_str}")
                                        nums_html = " ".join([f"<span class='f3d-ball'>{x}</span>" for x in nums])
                                        st.markdown(f"**第 {i:02d} 组**： {nums_html}", unsafe_allow_html=True)
                                    if ok:
                                        predictions_to_save.append(clean)
                                    else:
                                        _dropped += 1
                                if _dropped:
                                    st.warning(f"⚠️ 已忽略 {_dropped} 组不符合规则的号码（个数/范围校验未通过）")
                                if qxc_lines:
                                    st.code("\n".join(qxc_lines), language="text")
                                    st.session_state['pending_qxc_predictions'] = predictions_to_save

            if 'pending_qxc_predictions' in st.session_state and st.session_state['pending_qxc_predictions']:
                if st.button("💾 保存七星彩预测", width="stretch"):
                    try:
                        latest_code = _db_get_latest_code("qxc")
                        if latest_code:
                            next_code = str(int(latest_code) + 1)
                            _n = len(st.session_state['pending_qxc_predictions'])
                            logger.info(f"[AI预测] 保存七星彩: 期号={next_code}, 组数={_n}")
                            save_prediction_record("qxc", next_code, st.session_state['pending_qxc_predictions'])
                            _toast_save(f"✅ 已保存（第 {next_code} 期）")
                        else:
                            logger.warning("[AI预测] 保存七星彩: 无法获取最新期号")
                    except Exception as e:
                        _toast_error(f"保存失败: {e}")

            with col_pl3_ai:
                st.markdown("#### 🟤 排列三")
                if st.button("🔮 AI预测排列三", width="stretch"):
                    with st.spinner("AI正在分析排列三趋势..."):
                        ai_pl3 = ai_predict_pl3(n_groups)
                        if "error" in ai_pl3:
                            st.error(ai_pl3["error"])
                        else:
                            ai_numbers = ai_pl3.get("recommendations", [])
                            if ai_numbers:
                                pl3_lines = []
                                predictions_to_save = []
                                _dropped = 0
                                for i, rec in enumerate(ai_numbers[:n_groups], 1):
                                    nums = rec.get("numbers", [])
                                    clean, ok = _validate_ai_group("pl3", rec)
                                    if nums and len(nums) >= 3:
                                        pl3_lines.append(f"第{i:02d}注 {nums[0]} {nums[1]} {nums[2]}")
                                        st.markdown(
                                            f"**第 {i:02d} 组**： "
                                            f"<span class='f3d-ball'>{nums[0]}</span>"
                                            f"<span class='f3d-ball'>{nums[1]}</span>"
                                            f"<span class='f3d-ball'>{nums[2]}</span>",
                                            unsafe_allow_html=True
                                        )
                                    if ok:
                                        predictions_to_save.append(clean)
                                    else:
                                        _dropped += 1
                                if _dropped:
                                    st.warning(f"⚠️ 已忽略 {_dropped} 组不符合规则的号码（个数/范围校验未通过）")
                                if pl3_lines:
                                    st.code("\n".join(pl3_lines), language="text")
                                    st.session_state['pending_pl3_predictions'] = predictions_to_save

            if 'pending_pl3_predictions' in st.session_state and st.session_state['pending_pl3_predictions']:
                if st.button("💾 保存排列三预测", width="stretch"):
                    try:
                        latest_code = _db_get_latest_code("pl3")
                        if latest_code:
                            next_code = str(int(latest_code) + 1)
                            _n = len(st.session_state['pending_pl3_predictions'])
                            logger.info(f"[AI预测] 保存排列三: 期号={next_code}, 组数={_n}")
                            save_prediction_record("pl3", next_code, st.session_state['pending_pl3_predictions'])
                            _toast_save(f"✅ 已保存（第 {next_code} 期）")
                        else:
                            logger.warning("[AI预测] 保存排列三: 无法获取最新期号")
                    except Exception as e:
                        _toast_error(f"保存失败: {e}")
    else:
        st.warning("⚠️ AI功能尚未配置，请在左侧边栏配置API Key")
    
    # ===== v2.0 集成预测模式（本地+蒙特卡洛+马尔可夫三路投票） =====
    if ENSEMBLE_AVAILABLE:
        st.write("---")
        st.markdown("### 🧬 集成预测（贝叶斯融合 + 蒙特卡洛 + 马尔可夫链三路投票）")
        st.info("🔬 **算法原理**：对历史数据进行**指数衰减加权**、**马尔可夫转移矩阵**、**遗漏回补概率**、**贝叶斯融合**四维建模后，通过 **10,000 次蒙特卡洛模拟** 投票选出高置信度号码。同时校验 AC值、跨度、012路、质合比、尾数等数学约束。")
        
        col_ens_type, col_ens_cnt = st.columns([2, 3])
        with col_ens_type:
            current_name = st.session_state.get('selected_lottery', 'ssq')
            lot_display = {
                "ssq": "双色球", "kl8": "快乐8", "fcsd": "福彩3D",
                "dlt": "大乐透", "qxc": "七星彩", "pl3": "排列三"
            }
            st.info(f"当前彩种：**{lot_display.get(current_name, current_name)}**（顶部切换大类 / 左侧选择具体彩种）")
        with col_ens_cnt:
            ensemble_n = st.slider("生成组数", 1, 10, 5, key="ensemble_n")

        # 快乐8玩法选择（仅快乐8时显示）
        _ens_kl8_pick_size = 10
        if current_name == "kl8":
            _kl8_ens_options = {"选十 (10个号)": 10, "选九 (9个号)": 9, "选八 (8个号)": 8,
                                "选七 (7个号)": 7, "选六 (6个号)": 6, "选五 (5个号)": 5,
                                "选四 (4个号)": 4, "选三 (3个号)": 3, "选二 (2个号)": 2, "选一 (1个号)": 1}
            _kl8_ens_sel = st.selectbox("快乐8玩法", list(_kl8_ens_options.keys()),
                                        index=0, key="kl8_ens_play_type",
                                        help="快乐8可选选一~选十，选十最主流")
            _ens_kl8_pick_size = _kl8_ens_options[_kl8_ens_sel]

        # AI 候选池审阅开关
        _use_ai_review = is_ai_configured() and st.checkbox(
            "🤖 AI 候选池审阅", value=True,
            help="开启后，AI 将审阅算法候选池，标记优先/回避号码，微调置信度后重新生成推荐",
            key="chk_ai_review"
        )

        if st.button("🧬 运行集成预测", type="primary", width="stretch", key="btn_ensemble"):
            spinner_msg = "🧬 正在运行四维建模 + 10,000次蒙特卡洛模拟..."
            if _use_ai_review:
                spinner_msg = "🧬 算法建模 → 🤖 AI 审阅候选池 → 生成推荐..."
            with st.spinner(spinner_msg):
                lt = st.session_state.get('selected_lottery', 'ssq')
                
                # 获取特征摘要
                try:
                    summary = get_feature_summary(lt)
                except Exception:
                    summary = {}
                
                # 获取集成预测（可选 AI 审阅）
                try:
                    ensemble = get_ensemble_prediction(lt, ensemble_n, ai_review=_use_ai_review,
                                                      kl8_pick_size=_ens_kl8_pick_size)
                except Exception as e:
                    st.error(f"集成预测失败: {e}")
                    ensemble = {}
                
                if ensemble and "error" not in ensemble:
                    # 显示 AI 审阅结果
                    ai_review_info = ensemble.get("ai_review", {})
                    if ai_review_info and (ai_review_info.get("prioritized") or ai_review_info.get("avoided")):
                        st.markdown("#### 🤖 AI 候选池审阅")
                        _pri = ai_review_info.get("prioritized", [])
                        _avd = ai_review_info.get("avoided", [])
                        _reason = ai_review_info.get("reasoning", "")
                        if _pri:
                            pri_html = " ".join([f"<span style='display:inline-block;background:#52c41a;color:#fff;border-radius:50%;width:28px;height:28px;line-height:28px;text-align:center;margin:2px;font-size:13px;font-weight:bold'>{n:02d}</span>" for n in _pri])
                            st.markdown(f"**🔺 优先（遗漏回补/区间轮动）**：{pri_html}", unsafe_allow_html=True)
                        if _avd:
                            avd_html = " ".join([f"<span style='display:inline-block;background:#ff4d4f;color:#fff;border-radius:50%;width:28px;height:28px;line-height:28px;text-align:center;margin:2px;font-size:13px;font-weight:bold'>{n:02d}</span>" for n in _avd])
                            st.markdown(f"**🔻 回避（极热释放/过度集中）**：{avd_html}", unsafe_allow_html=True)
                        if _reason:
                            st.caption(f"💡 {_reason}")
                        st.markdown("---")

                    # 显示置信度分布
                    conf_dist = ensemble.get("confidence_distribution", {})
                    if conf_dist:
                        st.markdown("#### 📊 号码置信度分布（三路投票收敛结果）")
                        # 转成排序列表
                        sorted_conf = sorted(conf_dist.items(), key=lambda x: float(x[1]), reverse=True)
                        
                        # 显示为图表
                        import plotly.graph_objects as go
                        nums = [x[0] for x in sorted_conf[:20]]
                        confs = [x[1] for x in sorted_conf[:20]]
                        
                        fig = go.Figure(data=[
                            go.Bar(x=nums, y=confs, 
                                   marker_color=['#ff4d4f' if float(c) >= 0.3 else '#fa8c16' if float(c) >= 0.15 else '#d9d9d9' for c in confs],
                                   text=[f'{c:.1%}' for c in confs],
                                   textposition='auto')
                        ])
                        fig.update_layout(
                            title="号码置信度 TOP20（红色≥30% 高置信）",
                            xaxis_title="号码",
                            yaxis_title="置信度",
                            height=400,
                            margin=dict(l=20, r=20, t=40, b=20)
                        )
                        st.plotly_chart(fig, width="stretch")
                        
                        # 高置信号码
                        high_conf = [(n, c) for n, c in sorted_conf if float(c) >= 0.10]
                        if high_conf:
                            st.markdown("**🌟 热门号码池**：")
                            if lt == "ssq":
                                high_nums_html = " ".join([f"<span class='ssq-red'>{n}</span>" for n, c in high_conf])
                            elif lt == "kl8":
                                high_nums_html = " ".join([f"<span class='kl8-ball'>{n}</span>" for n, c in high_conf])
                            else:
                                high_nums_html = " ".join([f"<span class='f3d-ball'>{n}</span>" for n, c in high_conf])
                            st.markdown(high_nums_html, unsafe_allow_html=True)
                    
                    # 显示推荐组
                    recs = ensemble.get("recommendations", [])
                    if recs:
                        st.markdown("#### 🎯 集成预测推荐号码")
                        
                        if lt == "ssq":
                            cols_display = st.columns(min(len(recs), 3))
                            for i, rec in enumerate(recs):
                                with cols_display[i % 3]:
                                    nums = rec.get("nums", [])
                                    conf = rec.get("confidence", 0)
                                    valid = rec.get("valid", True)
                                    valid_icon = "✅" if valid else "⚠️"
                                    red_html = "".join([f"<span class='ssq-red'>{x:02d}</span>" for x in nums[:6]])
                                    if len(nums) > 6:
                                        blue_html = f"<span class='ssq-blue'>{nums[6]:02d}</span>"
                                    else:
                                        blue_html = ""
                                    st.markdown(f"**组 {i+1}** {valid_icon} 置信度 {conf:.1%}")
                                    st.markdown(red_html + blue_html, unsafe_allow_html=True)
                        elif lt == "dlt":
                            cols_display = st.columns(min(len(recs), 3))
                            for i, rec in enumerate(recs):
                                with cols_display[i % 3]:
                                    nums = rec.get("nums", [])
                                    conf = rec.get("confidence", 0)
                                    valid = rec.get("valid", True)
                                    valid_icon = "✅" if valid else "⚠️"
                                    # 大乐透：5前区+2后区
                                    if len(nums) > 5:
                                        f_html = "".join([f"<span class='ssq-red'>{x:02d}</span>" for x in nums[:5]])
                                        b_html = "".join([f"<span class='ssq-blue'>{x:02d}</span>" for x in nums[5:7]])
                                    else:
                                        f_html = " ".join([f"<span class='ssq-red'>{x:02d}</span>" for x in nums])
                                        b_html = ""
                                    st.markdown(f"**组 {i+1}** {valid_icon} 置信度 {conf:.1%}")
                                    st.markdown(f_html + b_html, unsafe_allow_html=True)
                        elif lt == "kl8":
                            _kl8_ens_names = {1:"选一",2:"选二",3:"选三",4:"选四",5:"选五",
                                              6:"选六",7:"选七",8:"选八",9:"选九",10:"选十"}
                            _kl8_ens_pn = _kl8_ens_names.get(_ens_kl8_pick_size, f"选{_ens_kl8_pick_size}")
                            cols_display = st.columns(min(len(recs), 3))
                            for i, rec in enumerate(recs):
                                with cols_display[i % 3]:
                                    nums = rec.get("nums", [])
                                    conf = rec.get("confidence", 0)
                                    nums_html = "".join([f"<span class='kl8-ball'>{x:02d}</span>" for x in nums])
                                    st.markdown(f"**组 {i+1}** [{_kl8_ens_pn}] 置信度 {conf:.1%}")
                                    st.markdown(nums_html, unsafe_allow_html=True)
                        elif lt in ("qxc",):
                            cols_display = st.columns(min(len(recs), 3))
                            for i, rec in enumerate(recs):
                                with cols_display[i % 3]:
                                    nums = rec.get("nums", [])
                                    conf = rec.get("confidence", 0)
                                    nums_html = " ".join([f"<span class='f3d-ball'>{x}</span>" for x in nums])
                                    st.markdown(f"**组 {i+1}** 置信度 {conf:.1%}")
                                    st.markdown(nums_html, unsafe_allow_html=True)
                        else:
                            cols_display = st.columns(min(len(recs), 3))
                            for i, rec in enumerate(recs):
                                with cols_display[i % 3]:
                                    nums = rec.get("nums", [])
                                    conf = rec.get("confidence", 0)
                                    st.markdown(f"**组 {i+1}** 置信度 {conf:.1%}")
                                    st.markdown(
                                        f"<span class='f3d-ball'>{nums[0]}</span>"
                                        f"<span class='f3d-ball'>{nums[1]}</span>"
                                        f"<span class='f3d-ball'>{nums[2]}</span>",
                                        unsafe_allow_html=True
                                    )
                        
                        # 一键复制推荐号码（投注用）
                        st.write("---")
                        st.markdown("**📋 一键复制推荐号码（娱乐参考）**")
                        copy_lines = []
                        if lt == "ssq":
                            for i, rec in enumerate(recs, 1):
                                nums = rec.get("nums", [])
                                if len(nums) >= 7:
                                    reds = " ".join(f"{x:02d}" for x in nums[:6])
                                    blue = f"{nums[6]:02d}"
                                    copy_lines.append(f"第{i:02d}注 红球：{reds} 蓝球：{blue}")
                            copy_text = "-----集成预测 双色球-----\n" + "\n".join(copy_lines)
                        elif lt == "dlt":
                            for i, rec in enumerate(recs, 1):
                                nums = rec.get("nums", [])
                                if len(nums) >= 7:
                                    fronts = " ".join(f"{x:02d}" for x in nums[:5])
                                    backs = " ".join(f"{x:02d}" for x in nums[5:7])
                                    copy_lines.append(f"第{i:02d}注 前区：{fronts} 后区：{backs}")
                            copy_text = "-----集成预测 大乐透-----\n" + "\n".join(copy_lines)
                        elif lt == "kl8":
                            for i, rec in enumerate(recs, 1):
                                nums = rec.get("nums", [])
                                nums_str = " ".join(f"{x:02d}" for x in nums)
                                copy_lines.append(f"第{i:02d}注 [{_kl8_ens_pn}] {nums_str}")
                            copy_text = f"-----集成预测 快乐8({_kl8_ens_pn})-----\n" + "\n".join(copy_lines)
                        elif lt == "qxc":
                            for i, rec in enumerate(recs, 1):
                                nums = rec.get("nums", [])
                                nums_str = " ".join(str(x) for x in nums)
                                copy_lines.append(f"第{i:02d}注 {nums_str}")
                            copy_text = "-----集成预测 七星彩-----\n" + "\n".join(copy_lines)
                        elif lt == "pl3":
                            for i, rec in enumerate(recs, 1):
                                nums = rec.get("nums", [])
                                if len(nums) >= 3:
                                    copy_lines.append(f"第{i:02d}注 {nums[0]} {nums[1]} {nums[2]}")
                            copy_text = "-----集成预测 排列三-----\n" + "\n".join(copy_lines)
                        else:
                            for i, rec in enumerate(recs, 1):
                                nums = rec.get("nums", [])
                                if len(nums) >= 3:
                                    copy_lines.append(f"第{i:02d}注 {nums[0]} {nums[1]} {nums[2]}")
                            copy_text = "-----集成预测-----\n" + "\n".join(copy_lines)
                        if copy_lines:
                            st.code(copy_text, language="text")
                        
                        # 显示模型贡献
                        contributions = ensemble.get("model_contributions", {})
                        if contributions:
                            st.caption(f"模型权重: 贝叶斯融合 {contributions.get('bayesian_fusion', 0)*100:.0f}% | "
                                      f"蒙特卡洛 {contributions.get('monte_carlo', 0)*100:.0f}% | "
                                      f"马尔可夫链 {contributions.get('markov_chain', 0)*100:.0f}%")

                        # --- 保存集成预测记录 ---
                        ensemble_predictions_to_save = []
                        _ens_dropped = 0
                        for rec in recs:
                            nums = rec.get("nums", [])
                            try:
                                if lt == "ssq":
                                    if len(nums) >= 7:
                                        reds = [int(x) for x in nums[:6]]
                                        blue = int(nums[6])
                                        if (len(set(reds)) == 6 and all(1 <= r <= 33 for r in reds)
                                                and 1 <= blue <= 16):
                                            ensemble_predictions_to_save.append({"red": reds, "blue": blue})
                                        else:
                                            _ens_dropped += 1
                                    else:
                                        _ens_dropped += 1
                                elif lt == "dlt":
                                    if len(nums) >= 7:
                                        fronts = [int(x) for x in nums[:5]]
                                        backs = [int(x) for x in nums[5:7]]
                                        if (len(set(fronts)) == 5 and all(1 <= f <= 35 for f in fronts)
                                                and len(set(backs)) == 2 and all(1 <= b <= 12 for b in backs)):
                                            ensemble_predictions_to_save.append({"nums": fronts + backs})
                                        else:
                                            _ens_dropped += 1
                                    else:
                                        _ens_dropped += 1
                                elif lt == "kl8":
                                    balls = [int(x) for x in nums]
                                    _n = _ens_kl8_pick_size
                                    if len(balls) == _n and len(set(balls)) == _n and all(1 <= b <= 80 for b in balls):
                                        ensemble_predictions_to_save.append({"nums": balls})
                                    else:
                                        _ens_dropped += 1
                                elif lt == "qxc":
                                    balls = [int(x) for x in nums]
                                    if len(balls) == 7 and all(0 <= b <= 9 for b in balls):
                                        ensemble_predictions_to_save.append({"nums": balls})
                                    else:
                                        _ens_dropped += 1
                                elif lt in ("fcsd", "pl3"):
                                    balls = [int(x) for x in nums]
                                    if len(balls) == 3 and all(0 <= b <= 9 for b in balls):
                                        ensemble_predictions_to_save.append({"nums": balls})
                                    else:
                                        _ens_dropped += 1
                            except (ValueError, TypeError):
                                _ens_dropped += 1

                        if _ens_dropped:
                            st.warning(f"⚠️ 已忽略 {_ens_dropped} 组不符合规则的号码")

                        if ensemble_predictions_to_save:
                            st.session_state['pending_ensemble_predictions'] = ensemble_predictions_to_save
                            st.session_state['pending_ensemble_lottery_type'] = lt
                            # 快乐8玩法信息
                            if lt == "kl8":
                                _kl8_ens_names = {1:"选一",2:"选二",3:"选三",4:"选四",5:"选五",
                                                  6:"选六",7:"选七",8:"选八",9:"选九",10:"选十"}
                                st.session_state['pending_ensemble_kl8_play'] = _kl8_ens_names.get(_ens_kl8_pick_size, f"选{_ens_kl8_pick_size}")

                            # ===== 运行集成预测后自动保存（所有彩种统一逻辑，无需手动点击）=====
                            try:
                                _auto_latest = _db_get_latest_code(lt)
                                if _auto_latest:
                                    _auto_next = str(int(_auto_latest) + 1)
                                    _auto_n = len(ensemble_predictions_to_save)
                                    _auto_label = lot_display.get(lt, lt)
                                    if lt == "kl8":
                                        _auto_label = f"快乐8({st.session_state.get('pending_ensemble_kl8_play', '选十')})"
                                    logger.info(f"[集成预测] 自动保存: 彩种={lt}, 期号={_auto_next}, 组数={_auto_n}, play_type=ensemble")
                                    save_prediction_record(lt, _auto_next, ensemble_predictions_to_save, play_type="ensemble")
                                    st.session_state['_ensemble_auto_msg'] = f"✅ 已自动保存 {_auto_label} 集成预测（第 {_auto_next} 期），共 {_auto_n} 组，可在「AI 分析」页查看对比"
                                else:
                                    logger.warning(f"[集成预测] 自动保存: 彩种={lt}, 无法获取最新期号，跳过自动保存")
                                    st.session_state['_ensemble_auto_msg'] = "⚠️ 自动保存跳过：无法获取最新期号"
                            except Exception as _ae:
                                logger.error(f"[集成预测] 自动保存失败: 彩种={lt}, error={_ae}", exc_info=True)
                                st.session_state['_ensemble_auto_msg'] = f"⚠️ 自动保存失败: {_ae}"

                    else:
                        st.warning("集成预测需要历史数据支持，请先同步数据")

            # 集成预测运行后即自动保存（所有彩种统一），此处仅展示反馈提示
            # 注：原「手动保存」按钮使用 on_click 回调并在回调内清空 pending，
            # 导致主循环中显示提示的 if 块永远不执行（只有日志、界面无提示）。
            # 现改为「运行即保存 + 当前渲染周期直接 toast」，彻底解决无提示问题。
            _auto_msg = st.session_state.pop('_ensemble_auto_msg', None)
            if _auto_msg:
                if _auto_msg.startswith("✅"):
                    _toast_save(_auto_msg)
                else:
                    _toast_error(_auto_msg)



elif selected_page == "hedge":
    if st.session_state.get('lottery_category') == 'sports':
        _render_sports_hedge()
    else:
        _render_welfare_hedge()
elif selected_page == "ai":
    st.subheader("🤖 AI 大模型深度分析与预测")
    
    # AI 配置检查
    if not is_ai_configured():
        st.warning("⚠️ **AI 功能尚未配置**")
        st.info("请在左侧边栏点击「🔑 配置 AI 模型」输入您的 API Key 和 Base URL，即可启用 AI 深度分析功能。")
        st.stop()
    
    st.info("💡 **AI 分析原理**：基于本地历史数据，通过大语言模型进行多维度趋势分析，提供可解释的预测建议和科学依据。")
    
    # AI 参数设置
    col_ai_cnt, col_ai_type = st.columns([2, 3])
    with col_ai_cnt:
        ai_n_groups = st.slider("AI 生成组数", min_value=1, max_value=20, value=5)
    with col_ai_type:
        cur_lot = st.session_state.get('selected_lottery', 'ssq')
        lot_display = {
            "ssq": "双色球", "kl8": "快乐8", "fcsd": "福彩3D",
            "dlt": "大乐透", "qxc": "七星彩", "pl3": "排列三"
        }
        st.info(f"当前分析彩种：**{lot_display.get(cur_lot, cur_lot)}**（顶部切换大类 / 左侧选择具体彩种）")
    
    st.write("---")
    
    # AI 预测按钮
    if st.button("🚀 启动 AI 深度分析", type="primary", width="stretch"):
        with st.spinner("🤖 AI 正在分析历史数据并生成预测报告..."):
            try:
                cur_lot = st.session_state.get('selected_lottery', 'ssq')
                lot_display = {
                    "ssq": "双色球", "kl8": "快乐8", "fcsd": "福彩3D",
                    "dlt": "大乐透", "qxc": "七星彩", "pl3": "排列三"
                }
                local_funcs = {
                    "ssq": lambda n: predict_ssq(n),
                    "kl8": lambda n: predict_kl8(n, 10),
                    "fcsd": lambda n: predict_fcsd(n),
                    "dlt": lambda n: predict_dlt(n),
                    "qxc": lambda n: predict_qxc(n),
                    "pl3": lambda n: predict_pl3(n),
                }
                ai_funcs = {
                    "ssq": ai_predict_ssq, "kl8": ai_predict_kl8, "fcsd": ai_predict_fcsd,
                    "dlt": ai_predict_dlt, "qxc": ai_predict_qxc, "pl3": ai_predict_pl3,
                }
                fmt_funcs = {
                    "ssq": format_ssq, "kl8": format_kl8, "fcsd": format_fcsd,
                    "dlt": format_dlt, "qxc": format_qxc, "pl3": format_pl3,
                }

                col_local, col_ai = st.columns(2)

                with col_local:
                    st.markdown(f"### 📊 本地算法预测（{lot_display[cur_lot]}）")
                    local_res = local_funcs[cur_lot](ai_n_groups)
                    for i, item in enumerate(local_res, 1):
                        st.markdown(f"**第 {i:02d} 组**： {_local_ball_html(cur_lot, item)}", unsafe_allow_html=True)
                    st.code(fmt_funcs[cur_lot](local_res), language="text")

                with col_ai:
                    st.markdown(f"### 🤖 AI 算法预测（{lot_display[cur_lot]}）")
                    ai_res = ai_funcs[cur_lot](ai_n_groups)
                    if "error" in ai_res:
                        st.error(ai_res["error"])
                    else:
                        recs = ai_res.get("recommendations", [])
                        for i, rec in enumerate(recs[:ai_n_groups], 1):
                            st.markdown(f"**第 {i:02d} 组**： {_ai_ball_html(cur_lot, rec)}", unsafe_allow_html=True)

                        copy_lines = []
                        for i, rec in enumerate(recs[:ai_n_groups], 1):
                            nums = rec.get("numbers", {})
                            if cur_lot == "ssq":
                                reds = nums.get("red", [])
                                blue = nums.get("blue", 0)
                                if reds and blue:
                                    copy_lines.append(f"第{i:02d}注 红球：{' '.join(f'{x:02d}' for x in reds)} 蓝球：{blue:02d}")
                            elif cur_lot == "dlt":
                                fronts = nums.get("front", [])
                                backs = nums.get("back", [])
                                if fronts and backs:
                                    copy_lines.append(f"第{i:02d}注 前区：{' '.join(f'{x:02d}' for x in fronts)} 后区：{' '.join(f'{x:02d}' for x in backs)}")
                            elif cur_lot == "kl8":
                                if isinstance(nums, list):
                                    copy_lines.append(f"第{i:02d}注 {' '.join(f'{x:02d}' for x in nums)}")
                            elif cur_lot == "qxc":
                                if isinstance(nums, list):
                                    copy_lines.append(f"第{i:02d}注 {' '.join(str(x) for x in nums)}")
                            elif cur_lot in ("fcsd", "pl3"):
                                if isinstance(nums, list) and len(nums) >= 3:
                                    copy_lines.append(f"第{i:02d}注 {nums[0]} {nums[1]} {nums[2]}")
                        if copy_lines:
                            st.markdown("📋 复制 AI 推荐号码")
                            st.code(f"-----AI{lot_display[cur_lot]}推荐-----\n" + "\n".join(copy_lines), language="text")

                        analysis = ai_res.get("analysis", "")
                        if analysis:
                            st.markdown("### 📝 AI 分析报告")
                            st.markdown(analysis)

                # 显示趋势分析（仅当前大类）
                st.write("---")
                st.markdown("### 📈 历史趋势深度分析")
                _cat = st.session_state['lottery_category']
                cur_cat_lots = LOT_CATS[_cat]['lots']
                trend_cols = st.columns(len(cur_cat_lots))
                for i, lot_key in enumerate(cur_cat_lots):
                    lot_short = LOT_CATS[_cat]['names'][lot_key].split(' ', 1)[-1]
                    with trend_cols[i]:
                        if st.button(f"🔍 分析{lot_short}趋势", width="stretch", key=f"trend_{lot_key}"):
                            trend = ai_analyze_trend(lot_key)
                            if "error" not in trend:
                                st.markdown(trend)
                            else:
                                st.error(trend["error"])

            except Exception as e:
                st.error(f"AI 分析失败: {e}")
    else:
        # 默认展示：趋势分析按钮
        st.markdown("### 📈 快捷趋势分析")
        st.info("点击上方「🚀 启动 AI 深度分析」按钮，获取完整的 AI 预测报告")

        _cat = st.session_state['lottery_category']
        st.markdown(f"**{LOT_CATS[_cat]['label']}**")
        cat_lots = LOT_CATS[_cat]['lots']
        q_cols = st.columns(len(cat_lots))
        for i, lot_key in enumerate(cat_lots):
            lot_short = LOT_CATS[_cat]['names'][lot_key].split(' ', 1)[-1]
            with q_cols[i]:
                if st.button(f"🔍 {lot_short}趋势", width="stretch", key=f"q_trend_{lot_key}"):
                    trend = ai_analyze_trend(lot_key)
                    if "error" not in trend:
                        st.markdown(trend)
                    else:
                        st.error(trend["error"])

    st.write("---")
    if st.session_state.get('lottery_category') == 'sports':
        st.info("🛡️ AI 对冲优化建议是基于「双色球」的福利彩票专属功能。当前已选择「体育彩票」，如需使用请切换顶部为「🟢 福利彩票」。")
        st.stop()
    st.markdown("### 🛡️ AI 对冲优化建议")
    st.info("💡 AI 会根据当前双色球趋势，智能推荐最优对冲方案和投注比例")
    
    col_hedge_ssq, col_hedge_kl8 = st.columns(2)
    with col_hedge_ssq:
        ssq_bets_opt = st.number_input("双色球计划投注（注数）", min_value=1, max_value=50, value=10)
    with col_hedge_kl8:
        hedge_strategy_opt = st.selectbox(
            "选择对冲方案",
            [
                "🛡️ 方案 A（极速回血）：快乐8 选一",
                "🛡️ 方案 B（阳光普照）：快乐8 选四",
                "🛡️ 方案 C（低频大回血）：福彩3D 组选六"
            ]
        )
    
    if st.button("🤖 获取 AI 对冲优化建议", type="primary", width="stretch"):
        with st.spinner("🤖 AI 正在分析最优对冲策略..."):
            try:
                hedge_opt = ai_optimize_hedge(ssq_bets_opt, hedge_strategy_opt)
                if "error" in hedge_opt:
                    st.error(hedge_opt["error"])
                else:
                    st.markdown(hedge_opt.get("advice", ""))
                    
                    # 显示优化后的推荐号码
                    if "optimized_numbers" in hedge_opt:
                        st.markdown("### 🎫 AI 优化后的推荐号码")
                        st.code(hedge_opt["optimized_numbers"], language="text")
            except Exception as e:
                st.error(f"对冲优化失败: {e}")
            

