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


def _parse_must_numbers(raw: str, num_range: tuple) -> list:
    """解析用户输入的必含号码字符串。

    支持格式：
      - 连写数字：389 → [3, 8, 9]（单数字优先）
      - 逗号/空格分隔：3,8,9 或 3 8 9 → [3, 8, 9]
      - 两位数：12,3,28 → [12, 3, 28]
      - 混合：3 12 28 → [3, 12, 28]

    逻辑：先尝试按分隔符拆分，若无分隔符则逐字符解析单数字。
    """
    lo, hi = num_range
    # 尝试按分隔符拆分（逗号、空格、中文逗号）
    import re
    parts = re.split(r'[,，\s]+', raw.strip())
    parts = [p for p in parts if p]

    if len(parts) > 1:
        # 有分隔符 → 每个部分是一个完整数字
        result = []
        for p in parts:
            if p.isdigit():
                n = int(p)
                if lo <= n <= hi:
                    result.append(n)
        return list(dict.fromkeys(result))  # 去重保序

    # 无分隔符 → 逐字符解析（单数字模式）
    result = []
    for ch in raw:
        if ch.isdigit():
            n = int(ch)
            if lo <= n <= hi:
                result.append(n)
    return list(dict.fromkeys(result))

# 各彩种单注价格与官方理论返奖率（用于投注成本风险提示）
# 返奖率为官方公开值，是长期数学期望，与选号方式无关
BET_META = {
    "ssq":  {"price": 2, "return_rate": 0.50, "name": "双色球"},
    "dlt":  {"price": 2, "return_rate": 0.50, "name": "大乐透"},
    "kl8":  {"price": 2, "return_rate": 0.58, "name": "快乐8"},
    "fcsd": {"price": 2, "return_rate": 0.53, "name": "福彩3D"},
    "pl3":  {"price": 2, "return_rate": 0.53, "name": "排列三"},
    "qxc":  {"price": 2, "return_rate": 0.50, "name": "七星彩"},
}


def _render_cost_risk(current_name: str, n_groups: int):
    """
    投注成本与风险提示。

    为什么加这个（2026-08-29）：
    回测已证明算法无 alpha、买多不改变收益率（ROI 恒定），
    因此最有价值的信息不是「选什么号」，而是让用户清楚
    「这笔投入的数学期望是多少」。纯算法计算，不调用 AI。
    """
    meta = BET_META.get(current_name)
    if not meta:
        return

    price = meta["price"]
    rate = meta["return_rate"]
    cost = n_groups * price
    expected_return = cost * rate
    expected_loss = cost - expected_return

    st.markdown("---")
    st.markdown("#### 💰 本期成本与期望")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("本期投入", f"¥{cost}",
                  help=f"{n_groups} 组 × ¥{price}/注")
    with col2:
        st.metric("理论期望回报", f"¥{expected_return:.1f}",
                  delta=f"-¥{expected_loss:.1f}",
                  delta_color="off",
                  help=f"按官方返奖率 {rate:.0%} 计算，与选号方式无关")
    with col3:
        st.metric("期望亏损", f"-¥{expected_loss:.1f}",
                  delta=f"{rate - 1:.0%} 收益率",
                  delta_color="off",
                  help="长期重复投注的数学期望")

    # 历史实际 ROI（从数据库统计）
    try:
        from ai_predict import get_betting_report
        report = get_betting_report(current_name)
        if "error" not in report and report.get("total_bets"):
            hist_roi = report.get("profit_rate", 0)
            st.caption(
                f"📊 本彩种历史实际：已投 {report['total_bets']} 注、"
                f"投入 ¥{report['total_cost']}、回收 ¥{report['total_prize']}、"
                f"收益率 {hist_roi:.1f}%"
            )
    except Exception:
        pass

    st.caption(
        "⚠️ 增加组数只会提高「至少中一次」的概率，**不改变每元期望回报**"
        "（投入翻倍、期望回报同样翻倍，亏损率恒定）。"
        "彩票为独立随机事件，任何选号方法都无法改变上述数学期望。"
    )


def render(current_name: str, cur_cat: str):
    """渲染预测页面。"""
    st.subheader("🎯 智能号码预测")

    # 直接渲染集成预测（2026-08-29 精简）
    # 原本地算法 tab 已被集成预测完整覆盖（贝叶斯/蒙特卡洛/马尔可夫/LSTM-CRF）
    # 原 AI 预测 tab 实测号码有系统性偏差（χ² 显著偏离均匀），不比随机数更准
    _render_ensemble_predict(current_name)


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

        # 成本与风险提示
        _render_cost_risk(current_name, len(groups))


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

                    # 成本与风险提示
                    _render_cost_risk(lot_key, len(predictions))

                    # AI 局限性说明（2026-08-29 实测）
                    st.caption(
                        "🤖 **关于 AI 预测**：实测 AI 生成的号码存在系统性偏差"
                        "（χ² 检验显著偏离均匀，偏好 31/32/33 等大号码、"
                        "回避 6~11），属于「有偏的随机源」，"
                        "不比随机数生成器更准，仅供娱乐参考。"
                    )


def _render_ensemble_predict(current_name: str):
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
        ensemble_n = st.number_input("生成组数", min_value=1, max_value=100,
                                     value=15, step=1,
                                     key="ensemble_n",
                                     help="默认 15 组。回测显示组数只影响「至少中一次」"
                                          "的概率，不改变每元期望回报")

        _refine = st.checkbox(
            "🎯 置信度精选（实验）", value=False,
            help="只留模型投票最一致的组，实际效果是号码往热号扎堆，"
                 "15组只覆盖约20个号码（关则29个），和最大覆盖优化反着来。"
                 "回测：不提高命中，反而降覆盖，故默认关。")

        # 必含号码输入
        _must_raw = ""
        _positional = current_name in ("fcsd", "pl3", "qxc")
        if _positional:
            _must_raw = st.text_input(
                "✨ 指定位数字（可选）",
                placeholder="如：389 表示百位3、十位8、个位9",
                key="must_include_pos",
                help="从左到右依次对应各位数字，每位0-9。"
                     "不足3位的在右边补0，超过3位截断。"
                     "留空=不指定。")
        else:
            _range = LOTTERY_META[current_name].get("primary_range", (1, 33))
            _must_raw = st.text_input(
                "✨ 必含号码（可选）",
                placeholder=f"如：389 → 每组必含3、8、9号",
                key="must_include_num",
                help=f"填入的数字会强制出现在每组推荐中，"
                     f"其余位置由算法填充。号码范围 {_range[0]}-{_range[1]}，"
                     f"最多填{6 if current_name == 'ssq' else 5}个。"
                     f"留空=不指定。不影响命中率，仅满足偏好。")

        st.caption(
            "📐 号码已启用**最大覆盖优化**：15 组会尽量铺开到更多不同号码。"
            "实测覆盖度 +24.5%、跨期波动 −25.9%、最差情况命中 +40%，"
            "而平均命中基本不变（期望守恒）。")

        # 快乐8选号个数
        _kl8_pick = 10
        if current_name == "kl8":
            _kl8_opts = {"选十": 10, "选九": 9, "选八": 8, "选七": 7,
                         "选六": 6, "选五": 5, "选四": 4}
            _kl8_sel = st.selectbox("快乐8玩法", list(_kl8_opts.keys()), index=0)
            _kl8_pick = _kl8_opts[_kl8_sel]

        submitted = st.form_submit_button("🚀 运行集成预测", type="primary",
                                          use_container_width=True)

    # 解析必含号码
    _must_nums = None
    if _must_raw.strip():
        raw = _must_raw.strip()
        if _positional:
            # 位置制：每个字符对应一个位，0-9
            digits = [int(ch) for ch in raw if ch.isdigit()]
            if digits:
                _must_nums = digits
        else:
            # 非位置制：用解析函数处理（支持连写/逗号/空格/两位数）
            _range = LOTTERY_META[current_name].get("primary_range", (1, 33))
            nums = _parse_must_numbers(raw, _range)
            if nums:
                _must_nums = nums

    if submitted:
        with st.spinner(f"正在运行{label}集成预测引擎..."):
            try:
                result = get_ensemble_prediction(
                    current_name, n_groups=ensemble_n,
                    kl8_pick_size=_kl8_pick if current_name == "kl8" else None,
                    refine=_refine,
                    must_include=_must_nums,
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

                # 成本与风险提示
                _render_cost_risk(current_name, len(recs))

                # 自动保存
                latest_code = _db_get_latest_code(current_name)
                if latest_code:
                    next_code = str(int(latest_code) + 1)
                    save_prediction_record(current_name, next_code, predictions_to_save)
                    _toast_save(f"✅ {label}集成预测已保存（第 {next_code} 期，{len(recs)} 组）")

            except Exception as e:
                logger.error(f"集成预测失败: {e}", exc_info=True)
                _toast_error(f"集成预测失败: {e}")
