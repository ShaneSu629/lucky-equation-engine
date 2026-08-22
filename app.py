# app.py — 幸运方程式 · 数字推理引擎（重构版）
"""
入口文件：全局配置 + 页面路由 + 通用辅助函数
业务逻辑拆分到 pages/ 目录和 ui_components.py
"""
import streamlit as st
import pandas as pd
import logging
from datetime import datetime

# 应用级日志
logger = logging.getLogger("app")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s.%(funcName)s: %(message)s",
        datefmt="%H:%M:%S"))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)

# ===== 导入 =====
from db_manager import (
    init_db, read_lottery_data as _db_read_lottery,
    get_latest_code as _db_get_latest_code,
    get_lottery_df as _db_get_lottery_df,
)
from styles import inject_styles, LOT_CATS, LOTTERY_META, NAV_ITEMS
from ui_components import render_balls_html

# 页面模块
from views.dashboard import render as render_dashboard
from views.predict import render as render_predict
from views.hedge import render as render_hedge
from views.ai import render as render_ai
from views.config import render as render_config


# ===== 页面配置 =====
st.set_page_config(
    page_title="幸运方程式 · 数字推理引擎",
    page_icon="🎲",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 注入喜庆风格
inject_styles()

# ===== 顶部 =====
st.title("🎲 幸运方程式 · 数字推理引擎")
st.caption(
    "⚠️ 理性购彩提示：彩票开奖为独立随机事件，本工具仅提供统计规律的娱乐参考，"
    "不构成中奖承诺或购彩建议。请量力而行、理性投注。"
)

# ===== 顶部：彩票大类切换 =====
if 'lottery_category' not in st.session_state:
    st.session_state['lottery_category'] = 'welfare'
if 'selected_lottery' not in st.session_state:
    st.session_state['selected_lottery'] = 'ssq'
if 'selected_page' not in st.session_state:
    st.session_state['selected_page'] = 'dashboard'


def _on_category_change():
    """切换大类时自动选中该大类第一个彩种。"""
    new_cat = st.session_state.get('lottery_category')
    if new_cat in LOT_CATS and st.session_state.get('selected_lottery') not in LOT_CATS[new_cat]['lots']:
        st.session_state['selected_lottery'] = LOT_CATS[new_cat]['lots'][0]
    for _k in [k for k in st.session_state if k.startswith('pending_') and k.endswith('_predictions')]:
        del st.session_state[_k]


st.segmented_control(
    "彩票大类",
    options=["welfare", "sports"],
    format_func=lambda c: LOT_CATS[c]["label"],
    key="lottery_category",
    on_change=_on_category_change,
)

# 防御
cur_cat = st.session_state.get('lottery_category', 'welfare')
if cur_cat not in LOT_CATS:
    st.session_state['lottery_category'] = 'welfare'
    cur_cat = 'welfare'
if st.session_state.get('selected_lottery') not in LOT_CATS[cur_cat]['lots']:
    st.session_state['selected_lottery'] = LOT_CATS[cur_cat]['lots'][0]

# ===== 侧边栏 =====
with st.sidebar:
    # 导航
    st.subheader("📌 页面导航")
    for key, icon, label in NAV_ITEMS:
        is_selected = st.session_state['selected_page'] == key
        btn_type = "primary" if is_selected else "secondary"
        if st.button(f"{icon} {label}", key=f"nav_btn_{key}", use_container_width=True, type=btn_type):
            st.session_state['selected_page'] = key
            st.rerun()

    # 彩种选择
    st.markdown("---")
    st.subheader("🎰 选择彩种")
    for lot_key in LOT_CATS[cur_cat]['lots']:
        lot_label = LOT_CATS[cur_cat]['names'][lot_key]
        is_sel = st.session_state.get('selected_lottery') == lot_key
        btype = "primary" if is_sel else "secondary"
        if st.button(lot_label, key=f"lot_btn_{lot_key}", use_container_width=True, type=btype):
            st.session_state['selected_lottery'] = lot_key
            for _k in [k for k in st.session_state if k.startswith('pending_') and k.endswith('_predictions')]:
                del st.session_state[_k]
            st.rerun()

    # 数据同步
    st.markdown("---")
    st.subheader("🔄 数据同步")
    if st.button("📥 立即同步最新数据", type="primary", use_container_width=True):
        st.session_state.sync_step = 0
        st.session_state.normal_sync = True
        st.rerun()

    if 'normal_sync' in st.session_state:
        with st.spinner("⏳ 正在同步..."):
            from fetch_lottery import update
            sync_order = [
                ("ssq", "双色球"), ("kl8", "快乐8"), ("fcsd", "福彩3D"),
                ("dlt", "大乐透"), ("qxc", "七星彩"), ("pl3", "排列三"),
            ]
            if st.session_state.sync_step < len(sync_order):
                lot_name, display_name = sync_order[st.session_state.sync_step]
                force_full = _db_get_latest_code(lot_name) is None
                st.caption(f"📥 同步 {display_name}...")
                update(lot_name, force_full=force_full)
                st.caption(f"✅ {display_name} 同步成功")
                st.session_state.sync_step += 1
                st.rerun()
            else:
                st.success("🎉 数据同步成功！")
                del st.session_state.sync_step
                del st.session_state.normal_sync
                st.rerun()

# ===== 页面路由 =====
selected_page = st.session_state['selected_page']
current_name = st.session_state.get('selected_lottery', 'ssq')

# 初始化数据库
init_db()

if selected_page == "dashboard":
    df = _db_get_lottery_df(current_name, dtype={"code": str})
    render_dashboard(current_name, df)

elif selected_page == "predict":
    render_predict(current_name, cur_cat)

elif selected_page == "hedge":
    render_hedge(cur_cat)

elif selected_page == "ai":
    render_ai(current_name)

elif selected_page == "config":
    render_config()
