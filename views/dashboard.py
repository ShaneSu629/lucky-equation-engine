# pages/dashboard.py
"""
历史数据看板页面
使用 st.tabs 拆分：最新开奖 | 预测对比 | 投注报表 | 历史数据
"""
import streamlit as st
import pandas as pd
from styles import LOTTERY_META
from ui_components import (
    render_latest_draw, render_balls_html,
    render_compare_result,
)


def render(current_name: str, df: pd.DataFrame):
    """渲染 dashboard 页面。"""
    meta = LOTTERY_META[current_name]
    label = meta["label"]
    icon = meta["icon"]

    st.subheader(f"📊 {icon} {label} 数据看板")

    if df.empty:
        st.warning("⚠️ 暂无本地历史数据，请在侧边栏点击「立即同步最新数据」拉取。")
        return

    # ===== 顶部指标 =====
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🏆 本地收录期数", f"{len(df)} 期")
    with col2:
        st.metric("🕒 最新开奖期号", f"{df.iloc[0]['code']} 期")
    with col3:
        st.metric("📅 最新开奖日期", f"{df.iloc[0]['date']}")

    # ===== Tab 分区 =====
    tab_latest, tab_compare, tab_report, tab_history = st.tabs(
        ["🎰 最新开奖", "🤖 预测对比", "💰 投注报表", "📋 历史数据"]
    )

    # ---------- Tab 1: 最新开奖 ----------
    with tab_latest:
        _render_latest_tab(current_name, df)

    # ---------- Tab 2: 预测对比 ----------
    with tab_compare:
        _render_compare_tab(current_name)

    # ---------- Tab 3: 投注报表 ----------
    with tab_report:
        _render_report_tab(current_name)

    # ---------- Tab 4: 历史数据 ----------
    with tab_history:
        _render_history_tab(current_name, df)


def _render_latest_tab(current_name: str, df: pd.DataFrame):
    """最新开奖 + 频率图。"""
    from ai_predict import get_prediction_records, is_ai_configured

    latest = df.iloc[0]
    meta = LOTTERY_META[current_name]

    # 最新开奖号码
    st.markdown(f"**第 {latest['code']} 期** | 日期：{latest['date']}")
    html = render_latest_draw(current_name, latest)
    st.markdown(html, unsafe_allow_html=True)

    # 频率图
    st.markdown("---")
    st.markdown("#### 📊 出号频率分布")

    if current_name == "ssq":
        all_reds = pd.concat([df['r1'], df['r2'], df['r3'], df['r4'], df['r5'], df['r6']])
        st.caption("红球频次")
        st.bar_chart(all_reds.value_counts().sort_index())
        st.caption("蓝球频次")
        st.bar_chart(df['blue'].value_counts().sort_index())

    elif current_name == "dlt":
        all_fronts = pd.concat([df['f1'], df['f2'], df['f3'], df['f4'], df['f5']])
        st.caption("前区频次")
        st.bar_chart(all_fronts.value_counts().sort_index())
        all_backs = pd.concat([df['b1'], df['b2']])
        st.caption("后区频次")
        st.bar_chart(all_backs.value_counts().sort_index())

    elif current_name == "kl8":
        cols = [f"n{i:02d}" for i in range(1, 21)]
        all_nums = pd.concat([df[c] for c in cols if c in df.columns])
        st.bar_chart(all_nums.value_counts().sort_index())

    elif current_name in ("fcsd", "pl3"):
        c_pos = st.radio("选择查看的数位", ["百位", "十位", "个位"], horizontal=True,
                         key=f"freq_pos_{current_name}")
        col_idx = ["百位", "十位", "个位"].index(c_pos) + 1
        st.bar_chart(df[f'n{col_idx}'].value_counts().sort_index())

    elif current_name == "qxc":
        c_pos = st.radio("选择查看的数位", [f"第{i}位" for i in range(1, 8)], horizontal=True,
                         key=f"freq_pos_{current_name}")
        col_idx = [f"第{i}位" for i in range(1, 8)].index(c_pos) + 1
        st.bar_chart(df[f'n{col_idx}'].value_counts().sort_index())


def _render_single_record(lot_type: str, record: dict):
    """渲染单条预测记录，折叠显示各组号码（用于分页浏览）。"""
    compared = "✅ 已对比" if record.get('compared') else "⏳ 待开奖"
    title = f"🎟️ 第 {record['code']} 期 | 🕐 {record.get('predict_time', '')} | {compared}"
    with st.expander(title, expanded=False):
        predictions = record.get('predictions', [])
        if not predictions:
            st.caption("（该期无预测号码）")
            return
        for i, pred in enumerate(predictions, 1):
            html = render_balls_html(lot_type, pred)
            if html:
                st.markdown(f"第 {i:02d} 组：{html}", unsafe_allow_html=True)
            else:
                nums = pred.get("nums", [])
                st.markdown(f"第 {i:02d} 组：{' '.join(str(x) for x in nums)}")


def _render_compare_tab(current_name: str):
    """AI 预测对比 — 数据库级筛选 + 分页浏览历史预测 + 单期详细对比。"""
    from ai_predict import (
        analyze_saved_predictions,
        refresh_all_prediction_compares, is_ai_configured,
    )
    from db_manager import (
        count_prediction_records, get_prediction_records_paged, get_prediction_codes,
    )
    from app_utils import _toast_save, _toast_error

    meta = LOTTERY_META[current_name]

    if not is_ai_configured():
        st.warning("⚠️ AI 功能尚未配置，请在配置中心设置 API Key。")
        return

    # ===== 筛选条件 =====
    fcol1, fcol2, fcol3, fcol4 = st.columns(4)
    with fcol1:
        code_start = st.number_input("起始期号", min_value=0, value=0, step=1,
                                     key=f"f_code_start_{current_name}",
                                     help="填 0 表示不限制起始期")
    with fcol2:
        code_end = st.number_input("结束期号", min_value=0, value=0, step=1,
                                   key=f"f_code_end_{current_name}",
                                   help="填 0 表示不限制结束期")
    with fcol3:
        status = st.selectbox("对比状态", ["全部", "已对比", "待开奖"],
                              index=0, key=f"f_status_{current_name}")
    with fcol4:
        win = st.selectbox("中奖状态", ["全部", "已中奖", "未中奖"],
                           index=0, key=f"f_win_{current_name}")

    status_map = {"全部": "all", "已对比": "compared", "待开奖": "pending"}
    win_map = {"全部": "all", "已中奖": "won", "未中奖": "lost"}
    status_v = status_map[status]
    win_v = win_map[win]
    cs = code_start if code_start and code_start > 0 else None
    ce = code_end if code_end and code_end > 0 else None

    total = count_prediction_records(current_name, cs, ce, status_v, win_v)
    if total == 0:
        st.info(f"ℹ️ 没有符合筛选条件的{meta['label']}预测记录。"
                f"请调整筛选条件，或先在「智能号码预测」页生成并保存。")
        return

    st.caption(f"📊 符合条件 {total} 期（数据库分页，仅加载当前页）")

    # ===== 分页（page_key 含筛选签名，换筛选自动回到第 1 页）=====
    PAGE_SIZE = 10
    sig = f"{status_v}|{win_v}|{cs}|{ce}"
    page_key = f"compare_page_{current_name}_{sig}"
    if page_key not in st.session_state:
        st.session_state[page_key] = 0
    total_pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)
    if st.session_state[page_key] >= total_pages:
        st.session_state[page_key] = total_pages - 1
    if st.session_state[page_key] < 0:
        st.session_state[page_key] = 0

    start = st.session_state[page_key] * PAGE_SIZE
    page_records = get_prediction_records_paged(
        current_name, offset=start, limit=PAGE_SIZE,
        code_start=cs, code_end=ce, status=status_v, win=win_v
    )

    st.markdown(
        f"#### 📋 历史预测记录（第 {start + 1}-{start + len(page_records)} 期 / 共 {total} 期）"
    )

    nav1, nav2, nav3 = st.columns([1, 2, 1])
    with nav1:
        if st.button("⬅️ 上一页", key=f"prev_{page_key}",
                     disabled=st.session_state[page_key] <= 0, width="stretch"):
            st.session_state[page_key] -= 1
            st.rerun()
    with nav2:
        st.markdown(
            f"<div style='text-align:center;padding-top:6px'>"
            f"第 {st.session_state[page_key] + 1} / {total_pages} 页</div>",
            unsafe_allow_html=True,
        )
    with nav3:
        if st.button("下一页 ➡️", key=f"next_{page_key}",
                     disabled=st.session_state[page_key] >= total_pages - 1, width="stretch"):
            st.session_state[page_key] += 1
            st.rerun()

    for record in page_records:
        _render_single_record(current_name, record)

    st.markdown("---")

    # ===== 单期详细对比 =====
    st.markdown("#### 📅 选择要对比的预测期号")
    all_codes = get_prediction_codes(current_name)
    if not all_codes:
        st.info(f"ℹ️ 暂无{meta['label']}的预测记录。请先在「智能号码预测」页生成并保存。")
        return
    selected_code = st.selectbox(
        "从已保存的预测记录中选择期号：",
        options=all_codes, index=0,
        key=f"compare_code_{current_name}",
    )

    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        run_compare = st.button("🔍 对比预测与开奖", type="primary", width="stretch")
    with btn_col2:
        force_compare = st.button("🔄 强制重新对比", width="stretch",
                                  help="忽略缓存，按最新中奖规则重新计算")

    if run_compare or force_compare:
        with st.spinner("正在分析对比..."):
            try:
                result = analyze_saved_predictions(current_name, selected_code,
                                                   force_refresh=force_compare)
                render_compare_result(current_name, result)
            except Exception as e:
                st.error(f"分析失败: {e}")

    # 刷新全部
    st.markdown("---")
    if st.button("🔄 重新计算全部历史对比", key=f"refresh_all_{current_name}",
                 help="按最新中奖规则刷新该彩种所有历史对比"):
        with st.spinner("正在刷新全部历史对比..."):
            try:
                r = refresh_all_prediction_compares(current_name)
                msg = (f"✅ 刷新完成：成功 {r['success']} 期，"
                       f"跳过 {r['skipped']} 期，失败 {r['errors']} 期")
                _toast_save(msg)
                if r['errors'] > 0:
                    for d in r['details'][-3:]:
                        if '失败' in d:
                            st.caption(d)
            except Exception as e:
                _toast_error(f"刷新失败: {e}")


def _render_report_tab(current_name: str):
    """投注报表。"""
    from ai_predict import get_betting_report

    meta = LOTTERY_META[current_name]
    report = get_betting_report(current_name)

    if "error" in report:
        st.info(report["error"])
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🎯 总投注注数", f"{report['total_bets']} 注")
    with col2:
        st.metric("💸 总投入", f"¥{report['total_cost']}")
    with col3:
        st.metric("🎁 总奖金", f"¥{report['total_prize']}")

    col4, col5, col6 = st.columns(3)
    with col4:
        st.metric("📈 总盈亏", f"¥{report['total_profit']}",
                  delta=f"{report['avg_profit_per_bet']:.2f} 元/注")
    with col5:
        st.metric("🏆 中奖率", f"{report['win_rate']:.1f}%",
                  delta=f"{report['win_count']} 期中奖")
    with col6:
        st.metric("📊 收益率", f"{report['profit_rate']:.1f}%")

    if report["records"]:
        st.markdown("### 📋 投注明细")
        report_df = pd.DataFrame(report["records"])
        report_df["profit"] = report_df["profit"].apply(
            lambda x: f"¥{x}" if x >= 0 else f"-¥{abs(x)}")
        report_df["profit_rate"] = report_df["profit_rate"].apply(
            lambda x: f"{x:.1f}%" if x >= 0 else f"-{abs(x):.1f}%")
        report_df["won"] = report_df["won"].apply(lambda x: "✅" if x else "❌")
        report_df = report_df[["code", "date", "lottery_type", "bets", "cost",
                               "prize", "profit", "profit_rate", "won"]]
        report_df.columns = ["期号", "日期", "彩种", "注数", "投入",
                             "奖金", "盈亏", "收益率", "中奖"]
        st.dataframe(report_df, use_container_width=True)

        # 累计盈亏图
        chart_data = pd.DataFrame(report["records"])
        chart_data["cumulative_profit"] = chart_data["profit"].cumsum()
        st.markdown("### 📈 累计盈亏走势")
        st.line_chart(chart_data[["date", "cumulative_profit"]].set_index("date"))


def _render_history_tab(current_name: str, df: pd.DataFrame):
    """历史数据表 + 下载。"""
    meta = LOTTERY_META[current_name]

    st.dataframe(df, use_container_width=True)

    st.download_button(
        label=f"💾 下载 {meta['label']} 完整历史数据",
        data=df.to_csv(index=False).encode('utf-8-sig'),
        file_name=f"{current_name}_history.csv",
        mime="text/csv",
        use_container_width=True,
    )
