# pages/hedge.py
"""
组合配比策略页面
福利版（双色球+快乐8/3D）+ 体彩版（大乐透+排列三/七星彩）
"""
import streamlit as st
import logging
from styles import LOTTERY_META, LOT_CATS
from ui_components import render_balls_html, render_local_balls_html, render_save_button

logger = logging.getLogger("app")


def render(cur_cat: str):
    """渲染对冲策略页面。"""
    if cur_cat == "welfare":
        _render_welfare_hedge()
    else:
        _render_sports_hedge()


def _render_welfare_hedge():
    """福利版对冲：核心=双色球，对冲=快乐8/福彩3D。"""
    st.subheader("🛡️ 双色球「智能配比」组合优化工具")
    st.caption(
        "双色球中奖率低 → 搭配高频高中奖率玩法（快乐8/3D），用小奖回扣主投成本。"
        "⚠️ 彩票开奖完全随机，本工具仅作娱乐参考。"
    )

    tab_config, tab_result = st.tabs(["⚙️ 投注配置", "🎫 号码推荐"])

    with tab_config:
        st.markdown("### 💰 第一步：输入投注配置")
        ssq_bets = st.number_input("核心：双色球单期计划投注（注数）",
                                   min_value=1, max_value=100, value=10)
        ssq_cost = ssq_bets * 2
        st.markdown(f"🔴 **双色球主投注额：** **{ssq_cost} 元**")

        hedge_strategy = st.selectbox(
            "第二步：选择搭配的对冲方案",
            [
                "🛡️ 方案 A（极速回血）：搭配 快乐8 选一（25%中奖率，稳健返还 4.6元/注）",
                "🛡️ 方案 B（阳光普照）：搭配 快乐8 选四（25.89%中奖率，中4个得100元）",
                "🛡️ 方案 C（低频大回血）：搭配 福彩3D 组选六（1/167中奖率，中奖得 173 元）",
            ]
        )

        hedge_bets = st.number_input("对冲单注数", min_value=1, max_value=100, value=5)
        hedge_cost = hedge_bets * 2
        total_cost = ssq_cost + hedge_cost

        st.markdown("---")
        st.metric("💳 总体组合投资预算", f"{total_cost} 元",
                  delta=f"双色球 {ssq_cost} 元 + 对冲 {hedge_cost} 元")

        # 方案数学分析
        with st.expander("📊 方案数学期望与回血率分析", expanded=False):
            if "方案 A" in hedge_strategy:
                st.markdown(f"""
                **快乐8「选一」**：中奖率 **25%**，中奖得 4.6 元。
                - 买 {hedge_bets} 注 → 预计至少中 1 注，回血 4.6~9.2 元
                - 可抵消双色球约 **23%~46%** 成本
                """)
            elif "方案 B" in hedge_strategy:
                st.markdown(f"""
                **快乐8「选四」**：中奖率 **25.89%**。
                - 4中2 → 3元 | 4中3 → 5元 | 4中4 → **100元**
                - 阳光普照 + 偶发大回血
                """)
            else:
                st.markdown(f"""
                **福彩3D「组选六」**：中奖率 **0.6%**，中奖得 **173 元**。
                - 中一次可直接覆盖约 8 期 10 注双色球的累计亏损
                """)

    with tab_result:
        _render_welfare_numbers(ssq_bets, hedge_strategy, hedge_bets)

    # AI 对冲分析
    _render_welfare_ai_analysis(ssq_bets, hedge_strategy)


def _render_welfare_numbers(ssq_bets, hedge_strategy, hedge_bets):
    """渲染福利版号码推荐。"""
    from generate_picks import (
        predict_ssq, gen_kl8_pick1, gen_kl8_pick4, gen_3d_group6,
        format_ssq, format_kl8_pick1, format_kl8_pick4, format_3d_group6,
    )

    col_ssq, col_hedge = st.columns(2)

    with col_ssq:
        st.markdown(f"#### 🔴 双色球 {ssq_bets} 注")
        ssq_groups = predict_ssq(ssq_bets)
        for i, (reds, blue) in enumerate(ssq_groups, 1):
            html = render_balls_html("ssq", {"red": reds, "blue": blue})
            st.markdown(f"**第 {i:02d} 注**：{html}", unsafe_allow_html=True)

        ssq_text = "-----双色球主投-----\n" + format_ssq(ssq_groups)
        with st.expander("📋 复制双色球号码"):
            st.code(ssq_text, language="text")

    with col_hedge:
        if "方案 A" in hedge_strategy:
            st.markdown(f"#### 🟡 快乐8 选一 {hedge_bets} 注")
            nums = gen_kl8_pick1(hedge_bets)
            for i, x in enumerate(nums, 1):
                html = render_balls_html("kl8", {"nums": [x]})
                st.markdown(f"**第 {i:02d} 注**：{html}", unsafe_allow_html=True)
            hedge_text = "----快乐8 选一-----\n" + format_kl8_pick1(nums)
        elif "方案 B" in hedge_strategy:
            st.markdown(f"#### 🟡 快乐8 选四 {hedge_bets} 注")
            groups = gen_kl8_pick4(hedge_bets)
            for i, nums in enumerate(groups, 1):
                html = render_balls_html("kl8", {"nums": nums})
                st.markdown(f"**第 {i:02d} 注**：{html}", unsafe_allow_html=True)
            hedge_text = "----快乐8 选四-----\n" + format_kl8_pick4(groups)
        else:
            st.markdown(f"#### 🟢 福彩3D 组选六 {hedge_bets} 注")
            groups = gen_3d_group6(hedge_bets)
            for i, nums in enumerate(groups, 1):
                html = render_balls_html("fcsd", {"nums": nums})
                st.markdown(f"**第 {i:02d} 注**：{html}", unsafe_allow_html=True)
            hedge_text = "----福彩3D 组选六-----\n" + format_3d_group6(groups)

        with st.expander("📋 复制对冲号码"):
            st.code(hedge_text, language="text")

    # 一键复制
    st.markdown("---")
    with st.expander("📋 一键复制完整对冲组合"):
        st.code(f"{ssq_text}\n\n{hedge_text}", language="text")


def _render_welfare_ai_analysis(ssq_bets, hedge_strategy):
    """福利版 AI 对冲分析。"""
    from ai_predict import ai_optimize_hedge, is_ai_configured
    from app_utils import _toast_save, _toast_error
    from db_manager import get_latest_code as _db_get_latest_code
    from ai_predict import save_prediction_record

    st.markdown("---")
    st.markdown("### 🤖 AI 智能对冲策略分析")

    if not is_ai_configured():
        st.info("💡 未检测到 AI 配置，对冲分析不可用。请在配置中心设置后重试。")
        return

    if st.button("🔍 AI 分析当前方案并推荐最优策略", type="primary",
                 use_container_width=True, key="ai_hedge_welfare"):
        with st.spinner("AI 正在分析历史数据..."):
            try:
                hedge_opt = ai_optimize_hedge(ssq_bets, hedge_strategy)
                if "error" in hedge_opt:
                    st.error(hedge_opt["error"])
                    return

                st.markdown(hedge_opt.get("advice", ""))

                ssq_groups_ai = hedge_opt.get("ssq_groups", [])
                hedge_groups_ai = hedge_opt.get("hedge_groups", [])
                hedge_type_ai = hedge_opt.get("hedge_type", "")
                hedge_name_ai = hedge_opt.get("hedge_name", "")

                if ssq_groups_ai or hedge_groups_ai:
                    col_ssq_ai, col_hedge_ai = st.columns(2)

                    if ssq_groups_ai:
                        with col_ssq_ai:
                            st.markdown("#### 🔴 AI 双色球推荐")
                            predictions = []
                            for i, item in enumerate(ssq_groups_ai, 1):
                                reds = item.get("red", [])
                                blue = item.get("blue", 0)
                                html = render_balls_html("ssq", {"red": reds, "blue": blue})
                                st.markdown(f"**第 {i:02d} 注**：{html}", unsafe_allow_html=True)
                                predictions.append({"red": reds, "blue": blue})
                            st.session_state['pending_ssq_predictions'] = predictions

                    if hedge_groups_ai:
                        with col_hedge_ai:
                            st.markdown(f"#### AI {hedge_name_ai}推荐")
                            predictions = []
                            for i, nums in enumerate(hedge_groups_ai, 1):
                                nums = list(nums)
                                html = render_balls_html(hedge_type_ai, {"nums": nums})
                                st.markdown(f"**第 {i:02d} 注**：{html}", unsafe_allow_html=True)
                                predictions.append({"nums": nums})
                            st.session_state['pending_hedge_predictions'] = predictions
                            st.session_state['pending_hedge_type'] = hedge_type_ai
                            st.session_state['pending_hedge_name'] = hedge_name_ai

            except Exception as e:
                st.error(f"AI 分析失败: {e}")

    # 保存按钮
    render_save_button("ssq", "pending_ssq_predictions", source_label="对冲AI")
    if 'pending_hedge_predictions' in st.session_state and st.session_state.get('pending_hedge_predictions'):
        hedge_type = st.session_state.get('pending_hedge_type', '')
        render_save_button(hedge_type, "pending_hedge_predictions", source_label="对冲AI")


def _render_sports_hedge():
    """体彩版对冲：核心=大乐透，对冲=排列三/七星彩。"""
    st.subheader("🛡️ 大乐透「智能配比」组合优化工具")
    st.caption(
        "大乐透中奖率低 → 搭配排列三（每日开奖）或七星彩，用小奖回扣主投成本。"
        "⚠️ 彩票开奖完全随机，本工具仅作娱乐参考。"
    )

    tab_config, tab_result = st.tabs(["⚙️ 投注配置", "🎫 号码推荐"])

    with tab_config:
        st.markdown("### 💰 第一步：输入投注配置")
        dlt_bets = st.number_input("核心：大乐透单期计划投注（注数）",
                                   min_value=1, max_value=100, value=10)
        dlt_cost = dlt_bets * 2
        st.markdown(f"🔵 **大乐透主投注额：** **{dlt_cost} 元**")

        hedge_strategy = st.selectbox(
            "第二步：选择搭配的对冲方案",
            [
                "🛡️ 方案 A（极速回血）：搭配 排列三 组选六（每日开奖，约0.6%中奖率，中奖得 173 元）",
                "🛡️ 方案 B（低频大回血）：搭配 七星彩 七位直选（超低概率、高奖级）",
            ]
        )

        hedge_bets = st.number_input("对冲单注数", min_value=1, max_value=100, value=5)
        hedge_cost = hedge_bets * 2
        total_cost = dlt_cost + hedge_cost

        st.markdown("---")
        st.metric("💳 总体组合投资预算", f"{total_cost} 元",
                  delta=f"大乐透 {dlt_cost} 元 + 对冲 {hedge_cost} 元")

    with tab_result:
        _render_sports_numbers(dlt_bets, hedge_strategy, hedge_bets)

    # AI 对冲分析
    _render_sports_ai_analysis(dlt_bets, hedge_strategy)


def _render_sports_numbers(dlt_bets, hedge_strategy, hedge_bets):
    """渲染体彩版号码推荐。"""
    from generate_picks import (
        predict_dlt, gen_pl3_group6, gen_qxc_pick7,
        format_dlt, format_pl3_group6, format_qxc_pick7,
    )

    col_dlt, col_hedge = st.columns(2)

    with col_dlt:
        st.markdown(f"#### 🔵 大乐透 {dlt_bets} 注")
        dlt_groups = predict_dlt(dlt_bets)
        for i, (fronts, backs) in enumerate(dlt_groups, 1):
            html = render_balls_html("dlt", {"front": fronts, "back": backs})
            st.markdown(f"**第 {i:02d} 注**：{html}", unsafe_allow_html=True)

        dlt_text = "-----大乐透主投-----\n" + format_dlt(dlt_groups)
        with st.expander("📋 复制大乐透号码"):
            st.code(dlt_text, language="text")

    with col_hedge:
        if "方案 A" in hedge_strategy:
            st.markdown(f"#### 🟤 排列三 组选六 {hedge_bets} 注")
            groups = gen_pl3_group6(hedge_bets)
            for i, nums in enumerate(groups, 1):
                html = render_balls_html("pl3", {"nums": nums})
                st.markdown(f"**第 {i:02d} 注**：{html}", unsafe_allow_html=True)
            hedge_text = "----排列三 组选六-----\n" + format_pl3_group6(groups)
        else:
            st.markdown(f"#### 🟣 七星彩 七位直选 {hedge_bets} 注")
            groups = gen_qxc_pick7(hedge_bets)
            for i, nums in enumerate(groups, 1):
                html = render_balls_html("qxc", {"nums": nums})
                st.markdown(f"**第 {i:02d} 注**：{html}", unsafe_allow_html=True)
            hedge_text = "----七星彩 七位-----\n" + format_qxc_pick7(groups)

        with st.expander("📋 复制对冲号码"):
            st.code(hedge_text, language="text")

    st.markdown("---")
    with st.expander("📋 一键复制完整对冲组合"):
        st.code(f"{dlt_text}\n\n{hedge_text}", language="text")


def _render_sports_ai_analysis(dlt_bets, hedge_strategy):
    """体彩版 AI 对冲分析。"""
    from ai_predict import ai_optimize_hedge_sports, is_ai_configured, save_prediction_record
    from app_utils import _toast_save, _toast_error
    from db_manager import get_latest_code as _db_get_latest_code

    st.markdown("---")
    st.markdown("### 🤖 AI 智能对冲策略分析")

    if not is_ai_configured():
        st.info("💡 未检测到 AI 配置，对冲分析不可用。请在配置中心设置后重试。")
        return

    if st.button("🔍 AI 分析当前方案并推荐最优策略", type="primary",
                 use_container_width=True, key="ai_hedge_sports"):
        with st.spinner("AI 正在分析历史数据..."):
            try:
                hedge_opt = ai_optimize_hedge_sports(dlt_bets, hedge_strategy)
                if "error" in hedge_opt:
                    st.error(hedge_opt["error"])
                    return

                st.markdown(hedge_opt.get("advice", ""))

                dlt_groups_ai = hedge_opt.get("dlt_groups", [])
                hedge_groups_ai = hedge_opt.get("hedge_groups", [])
                hedge_type_ai = hedge_opt.get("hedge_type", "")
                hedge_name_ai = hedge_opt.get("hedge_name", "")

                if dlt_groups_ai or hedge_groups_ai:
                    col_dlt_ai, col_hedge_ai = st.columns(2)

                    if dlt_groups_ai:
                        with col_dlt_ai:
                            st.markdown("#### 🔵 AI 大乐透推荐")
                            predictions = []
                            for i, item in enumerate(dlt_groups_ai, 1):
                                fronts = item.get("front", [])
                                backs = item.get("back", [])
                                html = render_balls_html("dlt", {"front": fronts, "back": backs})
                                st.markdown(f"**第 {i:02d} 注**：{html}", unsafe_allow_html=True)
                                predictions.append({"nums": fronts + backs})
                            st.session_state['pending_hedge_dlt_predictions'] = predictions

                    if hedge_groups_ai:
                        with col_hedge_ai:
                            st.markdown(f"#### AI {hedge_name_ai}推荐")
                            predictions = []
                            for i, nums in enumerate(hedge_groups_ai, 1):
                                nums = list(nums)
                                html = render_balls_html(hedge_type_ai, {"nums": nums})
                                st.markdown(f"**第 {i:02d} 注**：{html}", unsafe_allow_html=True)
                                predictions.append({"nums": nums})
                            st.session_state['pending_hedge_companion_predictions'] = predictions
                            st.session_state['pending_hedge_companion_type'] = hedge_type_ai
                            st.session_state['pending_hedge_companion_name'] = hedge_name_ai

            except Exception as e:
                st.error(f"AI 分析失败: {e}")

    render_save_button("dlt", "pending_hedge_dlt_predictions", source_label="对冲AI")
    if 'pending_hedge_companion_predictions' in st.session_state and st.session_state.get('pending_hedge_companion_predictions'):
        hedge_type = st.session_state.get('pending_hedge_companion_type', '')
        render_save_button(hedge_type, "pending_hedge_companion_predictions", source_label="对冲AI")
