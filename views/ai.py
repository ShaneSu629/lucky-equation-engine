# pages/ai.py
"""
AI 智能分析页面 — 趋势分析
"""
import streamlit as st
import logging
from styles import LOTTERY_META

logger = logging.getLogger("app")


def render(current_name: str):
    """渲染 AI 智能分析页面。"""
    from ai_predict import ai_analyze_trend, is_ai_configured

    meta = LOTTERY_META[current_name]
    label = meta["label"]

    st.subheader(f"🤖 AI 智能趋势分析 — {label}")

    if not is_ai_configured():
        st.warning("⚠️ AI 功能尚未配置，请在配置中心设置 API Key。")
        return

    if st.button(f"🔍 AI 分析{label}历史趋势", type="primary", use_container_width=True,
                 key=f"ai_trend_{current_name}"):
        with st.spinner(f"AI 正在深度分析{label}历史趋势..."):
            try:
                trend = ai_analyze_trend(current_name)
                st.markdown(trend)
            except Exception as e:
                logger.error(f"AI趋势分析失败: {e}", exc_info=True)
                st.error(f"AI 分析失败: {e}")

    # 历史分析记录（可扩展）
    st.markdown("---")
    st.caption("💡 AI 分析基于历史数据统计规律，不构成中奖承诺或购彩建议。")
