# pages/predict.py
"""
智能号码预测页面
本地算法 + AI 预测 + 集成预测
"""
import streamlit as st
import logging
from styles import LOTTERY_META, LOT_CATS
from ui_components import render_balls_html, render_local_balls_html

logger = logging.getLogger("app")


def render(current_name: str, cur_cat: str):
    """渲染预测页面。"""
    st.subheader("🎯 智能号码预测")

    # 科学选号原理
    cat_lots = LOT_CATS[cur_cat]['lots']
    ratio_desc = {
        "ssq": "双色球采用 3:2:1 比例组合",
        "kl8": "快乐 8 选十采用 5:3:2 组合",
        "fcsd": "福彩 3D 按位热温冷配比",
        "dlt": "大乐透前区 3:2 搭配后区热温冷",
        "qxc": "七星彩按位热温冷配比",
        "pl3": "排列三按位热温冷配比",
    }
    ratio_text = "，".join(ratio_desc[l] for l in cat_lots if l in ratio_desc)
    st.info(
        f"💡 **科学选号原理（{LOT_CATS[cur_cat]['label']}）**："
        f"根据自首发开奖至今的历史数据频次，自动划分为「热码」「温码」「冷码」。"
        f"{ratio_text}，有效规避不平衡选号！"
    )

    # 生成组数
    col_cnt, _ = st.columns([2, 4])
    with col_cnt:
        n_groups = st.number_input("每种彩票生成组数", min_value=1, max_value=100, value=5, step=1)

    # ===== Tab 分区 =====
    tab_local, tab_ai, tab_ensemble = st.tabs(
        ["🎲 本地算法", "🤖 AI 预测", "🔬 集成预测"]
    )

    with tab_local:
        _render_local_predict(current_name, n_groups)

    with tab_ai:
        _render_ai_predict(current_name, n_groups, cur_cat)

    with tab_ensemble:
        _render_ensemble_predict(current_name, n_groups)


def _render_local_predict(current_name: str, n_groups: int):
    """本地算法预测。"""
    from generate_picks import (
        predict_ssq, predict_kl8, predict_fcsd,
        predict_dlt, predict_qxc, predict_pl3,
        format_ssq, format_kl8, format_fcsd,
        format_dlt, format_qxc, format_pl3,
    )

    meta = LOTTERY_META[current_name]
    label = meta["label"]
    icon = meta["icon"]

    if st.button(f"🎲 生成 {label} 本地推荐号码", type="primary", use_container_width=True,
                 key=f"local_predict_{current_name}"):
        predict_fn = {
            "ssq": predict_ssq, "kl8": predict_kl8, "fcsd": predict_fcsd,
            "dlt": predict_dlt, "qxc": predict_qxc, "pl3": predict_pl3,
        }[current_name]
        format_fn = {
            "ssq": format_ssq, "kl8": format_kl8, "fcsd": format_fcsd,
            "dlt": format_dlt, "qxc": format_qxc, "pl3": format_pl3,
        }[current_name]

        groups = predict_fn(n_groups)
        for i, item in enumerate(groups, 1):
            html = render_local_balls_html(current_name, item)
            st.markdown(f"**第 {i:02d} 注**：{html}", unsafe_allow_html=True)

        plain_text = format_fn(groups)
        st.code(plain_text, language="text")


def _render_ai_predict(current_name: str, n_groups: int, cur_cat: str):
    """AI 智能预测。"""
    from ai_predict import (
        ai_predict_ssq, ai_predict_kl8, ai_predict_fcsd,
        ai_predict_dlt, ai_predict_qxc, ai_predict_pl3,
        is_ai_configured, save_prediction_record,
    )
    from app_utils import _validate_ai_group, _toast_save, _toast_error
    from ui_components import render_ai_prediction, render_save_button
    from db_manager import get_latest_code as _db_get_latest_code

    if not is_ai_configured():
        st.warning("⚠️ AI 功能尚未配置，请在配置中心设置 API Key。")
        return

    # 当前大类下的所有彩种 AI 预测面板
    cat_lots = LOT_CATS[cur_cat]['lots']

    cols = st.columns(len(cat_lots))
    for idx, lot_key in enumerate(cat_lots):
        with cols[idx]:
            _render_single_ai_panel(lot_key, n_groups)

    # 保存按钮（独立区域）
    st.markdown("---")
    st.markdown("### 💾 保存预测记录")
    for lot_key in cat_lots:
        pending_key = f"pending_{lot_key}_predictions"
        render_save_button(lot_key, pending_key, source_label="AI预测")


def _render_single_ai_panel(lot_key: str, n_groups: int):
    """渲染单个彩种的 AI 预测面板。"""
    from ai_predict import (
        ai_predict_ssq, ai_predict_kl8, ai_predict_fcsd,
        ai_predict_dlt, ai_predict_qxc, ai_predict_pl3,
    )
    from ui_components import render_ai_prediction

    meta = LOTTERY_META[lot_key]
    label = meta["label"]
    icon = meta["icon"]

    st.markdown(f"#### {icon} {label}")

    # 快乐8玩法选择
    pick_size = None
    if lot_key == "kl8":
        play_options = {
            "选十 (10个号)": 10, "选九 (9个号)": 9, "选八 (8个号)": 8,
            "选七 (7个号)": 7, "选六 (6个号)": 6, "选五 (5个号)": 5,
            "选四 (4个号)": 4, "选三 (3个号)": 3, "选二 (2个号)": 2, "选一 (1个号)": 1,
        }
        play_sel = st.selectbox("玩法", list(play_options.keys()), index=0,
                                key=f"ai_kl8_play_{lot_key}")
        pick_size = play_options[play_sel]
        st.caption(f"每组选 {pick_size} 个号码")

    if st.button(f"🔮 AI预测{label}", key=f"ai_predict_{lot_key}", use_container_width=True):
        predict_fn = {
            "ssq": ai_predict_ssq, "kl8": ai_predict_kl8, "fcsd": ai_predict_fcsd,
            "dlt": ai_predict_dlt, "qxc": ai_predict_qxc, "pl3": ai_predict_pl3,
        }[lot_key]

        with st.spinner(f"AI 正在分析{label}趋势..."):
            if lot_key == "kl8":
                result = predict_fn(n_groups, pick_size=pick_size)
            else:
                result = predict_fn(n_groups)

            if "error" in result:
                st.error(result["error"])
            else:
                predictions = render_ai_prediction(lot_key, result, n_groups, pick_size=pick_size)
                if predictions:
                    pending_key = f"pending_{lot_key}_predictions"
                    st.session_state[pending_key] = predictions


def _render_ensemble_predict(current_name: str, n_groups: int):
    """集成预测。"""
    try:
        from enhanced_predict import get_ensemble_prediction, ENSEMBLE_AVAILABLE
    except ImportError:
        ENSEMBLE_AVAILABLE = False

    if not ENSEMBLE_AVAILABLE:
        st.warning("⚠️ 增强预测模块不可用")
        return

    from ai_predict import is_ai_configured, save_prediction_record
    from app_utils import _toast_save, _toast_error
    from db_manager import get_latest_code as _db_get_latest_code

    meta = LOTTERY_META[current_name]
    label = meta["label"]

    with st.form("ensemble_form"):
        st.markdown(f"#### 🔬 {label} 集成预测")
        ensemble_n = st.number_input("生成组数", min_value=1, max_value=100, value=5, step=1,
                                     key="ensemble_n")

        _use_ai = False
        if is_ai_configured():
            _use_ai = st.checkbox("🤖 启用 AI 审阅增强", value=False,
                                  help="让 AI 审阅并调整集成预测的置信度分布")

        # 快乐8选号个数
        _kl8_pick = 10
        if current_name == "kl8":
            _kl8_opts = {"选十": 10, "选九": 9, "选八": 8, "选七": 7,
                         "选六": 6, "选五": 5, "选四": 4}
            _kl8_sel = st.selectbox("快乐8玩法", list(_kl8_opts.keys()), index=0)
            _kl8_pick = _kl8_opts[_kl8_sel]

        submitted = st.form_submit_button("🚀 运行集成预测", type="primary",
                                          use_container_width=True)

    if submitted:
        with st.spinner(f"正在运行{label}集成预测引擎..."):
            try:
                result = get_ensemble_prediction(
                    current_name, n_groups=ensemble_n,
                    ai_review=_use_ai,
                    kl8_pick_size=_kl8_pick if current_name == "kl8" else None,
                )
                recs = result.get("recommendations", [])
                if not recs:
                    st.warning("⚠️ 未生成推荐号码")
                    return

                # 渲染推荐组
                predictions_to_save = []
                for i, rec in enumerate(recs, 1):
                    html = render_balls_html(current_name, rec)
                    st.markdown(f"**第 {i:02d} 组**：{html}", unsafe_allow_html=True)
                    predictions_to_save.append(rec)

                # 特征摘要
                feature = result.get("feature_summary")
                if feature:
                    with st.expander("📊 特征分析摘要", expanded=False):
                        st.markdown(feature)

                # 自动保存
                latest_code = _db_get_latest_code(current_name)
                if latest_code:
                    next_code = str(int(latest_code) + 1)
                    save_prediction_record(current_name, next_code, predictions_to_save)
                    _toast_save(f"✅ {label}集成预测已保存（第 {next_code} 期，{len(recs)} 组）")

            except Exception as e:
                logger.error(f"集成预测失败: {e}", exc_info=True)
                _toast_error(f"集成预测失败: {e}")
