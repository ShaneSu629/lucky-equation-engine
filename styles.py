# styles.py
"""
统一 CSS 样式管理 — 喜庆风格
主色调：中国红 #E53935 / 金色 #FFD700 / 暖橙 #FF8F00
"""

LOTTERY_META = {
    "ssq":  {"label": "双色球", "icon": "🔴", "cat": "welfare",
             "ball_cols": ["r1","r2","r3","r4","r5","r6"], "special_col": "blue",
             "ball_type": "dual",  # dual = 红球+蓝球
             "primary_range": (1, 33), "special_range": (1, 16)},
    "kl8":  {"label": "快乐8", "icon": "🟡", "cat": "welfare",
             "ball_cols": [f"n{i:02d}" for i in range(1, 21)],
             "ball_type": "single", "primary_range": (1, 80)},
    "fcsd": {"label": "福彩3D", "icon": "🟢", "cat": "welfare",
             "ball_cols": ["n1", "n2", "n3"],
             "ball_type": "positional", "primary_range": (0, 9)},
    "dlt":  {"label": "大乐透", "icon": "🔵", "cat": "sports",
             "ball_cols": ["f1","f2","f3","f4","f5"], "special_col": ["b1","b2"],
             "ball_type": "dual",
             "primary_range": (1, 35), "special_range": (1, 12)},
    "qxc":  {"label": "七星彩", "icon": "🟣", "cat": "sports",
             "ball_cols": ["n1","n2","n3","n4","n5","n6","n7"],
             "ball_type": "positional", "primary_range": (0, 9)},
    "pl3":  {"label": "排列三", "icon": "🟤", "cat": "sports",
             "ball_cols": ["n1", "n2", "n3"],
             "ball_type": "positional", "primary_range": (0, 9)},
}

LOT_CATS = {
    "welfare": {"label": "🟢 福利彩票", "lots": ["ssq", "kl8", "fcsd"],
                "names": {"ssq": "🔴 双色球", "kl8": "🟡 快乐8", "fcsd": "🟢 福彩3D"}},
    "sports":  {"label": "🔵 体育彩票", "lots": ["dlt", "qxc", "pl3"],
                "names": {"dlt": "🔵 大乐透", "qxc": "🟣 七星彩", "pl3": "🟤 排列三"}},
}

NAV_ITEMS = [
    ("dashboard", "📈", "历史数据看板"),
    ("predict",   "🎯", "智能号码预测"),
    ("hedge",     "🛡️", "组合配比策略"),
    ("ai",        "🤖", "AI 智能分析"),
    ("config",    "⚙️", "配置中心"),
]

FESTIVE_CSS = """
<style>
    /* ===== 全局基础 ===== */
    .stApp {
        background: linear-gradient(135deg, #FFF8F0 0%, #FFF5F5 50%, #FFF8F0 100%);
    }
    section.main {
        padding-top: 1rem !important;
    }

    /* ===== 号码球 — 红球（双色球/大乐透前区）===== */
    .ball-red {
        background: linear-gradient(145deg, #EF5350 0%, #C62828 100%);
        color: #FFF;
        border: 2px solid #FFD700;
        border-radius: 50%;
        margin: 3px;
        display: inline-block;
        font-weight: 700;
        font-size: 15px;
        width: 42px;
        height: 42px;
        text-align: center;
        line-height: 38px;
        box-shadow: 0 3px 8px rgba(198,40,40,0.35), inset 0 1px 0 rgba(255,255,255,0.25);
        box-sizing: border-box;
    }
    /* ===== 号码球 — 蓝球（双色球蓝球/大乐透后区）===== */
    .ball-blue {
        background: linear-gradient(145deg, #42A5F5 0%, #1565C0 100%);
        color: #FFF;
        border: 2px solid #FFD700;
        border-radius: 50%;
        margin: 3px;
        display: inline-block;
        font-weight: 700;
        font-size: 15px;
        width: 42px;
        height: 42px;
        text-align: center;
        line-height: 38px;
        box-shadow: 0 3px 8px rgba(21,101,192,0.35), inset 0 1px 0 rgba(255,255,255,0.25);
        box-sizing: border-box;
    }
    /* ===== 号码球 — 橙球（快乐8）===== */
    .ball-orange {
        background: linear-gradient(145deg, #FFA726 0%, #E65100 100%);
        color: #FFF;
        border: 2px solid #FFD700;
        border-radius: 50%;
        margin: 3px;
        display: inline-block;
        font-weight: 700;
        font-size: 13px;
        width: 38px;
        height: 38px;
        text-align: center;
        line-height: 34px;
        box-shadow: 0 3px 8px rgba(230,81,0,0.35), inset 0 1px 0 rgba(255,255,255,0.25);
        box-sizing: border-box;
    }
    /* ===== 号码球 — 青球（3D/排列三/七星彩 位置制）===== */
    .ball-teal {
        background: linear-gradient(145deg, #26A69A 0%, #00695C 100%);
        color: #FFF;
        border: 2px solid #FFD700;
        border-radius: 6px;
        padding: 6px 12px;
        margin: 3px;
        display: inline-block;
        font-weight: 700;
        font-size: 17px;
        text-align: center;
        box-shadow: 0 3px 8px rgba(0,105,92,0.3), inset 0 1px 0 rgba(255,255,255,0.2);
    }

    /* ===== 喜庆卡片 ===== */
    .festive-card {
        background: linear-gradient(135deg, #FFFFFF 0%, #FFFAF0 100%);
        border: 1px solid #FFE0B2;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 4px 16px rgba(229,57,53,0.08), 0 1px 4px rgba(255,215,0,0.1);
    }

    /* ===== 按钮全局 — 喜庆红金 ===== */
    section.main [data-testid="stButton"] > button[kind="primary"] {
        background: linear-gradient(135deg, #E53935 0%, #C62828 100%) !important;
        border: 2px solid #FFD700 !important;
        border-radius: 10px !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        transition: all 0.25s ease !important;
        box-shadow: 0 3px 12px rgba(229,57,53,0.3) !important;
    }
    section.main [data-testid="stButton"] > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #EF5350 0%, #E53935 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(229,57,53,0.4) !important;
    }
    section.main [data-testid="stButton"] > button:not([kind="primary"]) {
        background: #FFFFFF !important;
        border: 1px solid #FFD700 !important;
        border-radius: 10px !important;
        color: #C62828 !important;
        font-weight: 500 !important;
        transition: all 0.25s ease !important;
        box-shadow: 0 2px 8px rgba(255,215,0,0.15) !important;
    }
    section.main [data-testid="stButton"] > button:not([kind="primary"]):hover {
        background: #FFF8E1 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(255,215,0,0.25) !important;
    }

    /* ===== 侧边栏 — 白底 ===== */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #FFFFFF 0%, #FFF6EF 100%) !important;
        padding: 8px !important;
        border-right: 2px solid rgba(230,57,53,0.18) !important;
    }
    [data-testid="stSidebar"] > div:first-child {
        padding: 4px !important;
    }
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] .stSubheader {
        color: #8E0000 !important;
    }
    [data-testid="stSidebar"] .stCaption,
    [data-testid="stSidebar"] .stInfo {
        color: #B71C1C !important;
    }
    /* 侧边栏按钮 */
    [data-testid="stSidebar"] .stButton > button {
        width: calc(100% - 8px) !important;
        padding: 10px 14px !important;
        margin: 3px 4px !important;
        border-radius: 10px !important;
        font-size: 14px !important;
        font-weight: 500 !important;
        text-align: left !important;
        transition: all 0.25s ease !important;
        min-height: 40px !important;
    }
    [data-testid="stSidebar"] .stButton > button:not([kind="primary"]) {
        background: #FFFFFF !important;
        color: #8E0000 !important;
        border: 1px solid rgba(230,57,53,0.35) !important;
        box-shadow: none !important;
    }
    [data-testid="stSidebar"] .stButton > button:not([kind="primary"]):hover {
        background: #FFF0E0 !important;
        border-color: #E63935 !important;
        color: #E63935 !important;
        transform: translateX(4px) !important;
    }
    [data-testid="stSidebar"] .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #E63935 0%, #C62828 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        font-weight: 700 !important;
        box-shadow: 0 2px 10px rgba(230,57,53,0.35) !important;
    }
    [data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
        box-shadow: 0 4px 16px rgba(230,57,53,0.5) !important;
        transform: translateX(4px) !important;
    }

    /* ===== st.metric 金色高亮 ===== */
    [data-testid="stMetricValue"] {
        color: #C62828 !important;
        font-weight: 700 !important;
    }

    /* ===== Tab 标签页 ===== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px !important;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px 10px 0 0 !important;
        padding: 10px 20px !important;
        font-weight: 500 !important;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #E53935, #C62828) !important;
        color: #FFD700 !important;
    }

    /* ===== 理性购彩提示缩小 ===== */
    .disclaimer-tip {
        font-size: 12px;
        color: #9E9E9E;
        text-align: center;
        padding: 4px 0;
    }

    /* ===== 分隔线 — 金色渐变 ===== */
    hr {
        border: none !important;
        height: 1px !important;
        background: linear-gradient(90deg, transparent, #FFD700, transparent) !important;
        margin: 16px 0 !important;
    }
</style>
"""


def inject_styles():
    """注入喜庆风格 CSS。"""
    import streamlit as st
    st.markdown(FESTIVE_CSS, unsafe_allow_html=True)
