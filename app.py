# app.py
import streamlit as st
import pandas as pd
import os
import random
import json
import time
from datetime import datetime

# 导入我们的逻辑模块
from fetch_lottery import update, DATA_DIR
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
    format_kl8_pick1,
    format_kl8_pick4,
    format_3d_group6,
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
        
        if st.button(f"{icon} {label}", key=f"nav_btn_{key}", use_container_width=True, type=btn_type):
            st.session_state['selected_page'] = key
            st.rerun()
    
    selected_page = st.session_state['selected_page']
    
    st.markdown("---")
    
    
    st.subheader("🔄 数据同步与更新")
    if st.button("立即同步最新数据", use_container_width=True, type="primary"):
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
                csv_path = os.path.join(DATA_DIR, f"{lot_name}.csv")
                force_full = not os.path.exists(csv_path)
                
                if force_full:
                    st.info(f"📊 {display_name}：首次同步，全量获取历史数据...")
                else:
                    st.info(f"📊 {display_name}：增量同步，只获取最新数据...")
                
                st.caption(f"📥 正在同步 {display_name}...")
                
                update(lot_name, force_full=force_full)
                st.caption(f"✅ {display_name} 同步成功")
                
                st.session_state.sync_step += 1
                time.sleep(10)
                st.rerun()
            else:
                st.success("🎉 数据同步成功！")
                
                del st.session_state.sync_step
                del st.session_state.normal_sync
                st.rerun()

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
                if st.button("🧪 测试连接", use_container_width=True):
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
                if st.button("💾 保存配置", use_container_width=True):
                    if not api_key:
                        st.error("API Key 不能为空")
                    else:
                        try:
                            from ai_predict import save_ai_config
                            save_ai_config(api_key, base_url, model_name)
                            st.success("✅ AI 配置已保存！下次启动自动加载")
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
    
    # 彩种数据加载
    col_sel, _ = st.columns([2, 4])
    with col_sel:
        view_name = st.selectbox("选择要查看的历史彩种", [
            "双色球 (ssq)", "快乐 8 (kl8)", "福彩 3D (fcsd)",
            "大乐透 (dlt)", "七星彩 (qxc)", "排列三 (pl3)"
        ])
        
    name_map = {
        "双色球 (ssq)": "ssq", "快乐 8 (kl8)": "kl8", "福彩 3D (fcsd)": "fcsd",
        "大乐透 (dlt)": "dlt", "七星彩 (qxc)": "qxc", "排列三 (pl3)": "pl3"
    }
    current_name = name_map[view_name]
    csv_path = os.path.join(DATA_DIR, f"{current_name}.csv")
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, dtype={"code": str})
        
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
            
            if st.button("📋 查看已保存的预测号码", use_container_width=True):
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
                                reds = pred.get('red', [])
                                blue = pred.get('blue', 0)
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
        
        if is_ai_configured():
            if st.button("🔍 对比已保存的预测记录与实际开奖", type="primary", use_container_width=True):
                with st.spinner("正在分析已保存的预测记录与实际开奖的对比..."):
                    try:
                        compare_result = analyze_saved_predictions(current_name)
                        
                        if "error" in compare_result:
                            st.error(compare_result["error"])
                            if "available_codes" in compare_result:
                                st.info(f"可用预测记录期号：{', '.join(compare_result['available_codes'][:10])}")
                        else:
                            latest_data = compare_result["latest"]
                            ai_best = compare_result["ai_best"]
                            predict_time = compare_result.get("predict_time", "")
                            
                            st.markdown(f"**最新开奖：第 {latest_data['code']} 期 ({latest_data['date']})**")
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
                                elif current_name == "kl8":
                                    nums_html = "".join([f"<span class='kl8-ball'>{x:02d}</span>" for x in latest_data['nums']])
                                    st.markdown(f"开奖号码：{nums_html}", unsafe_allow_html=True)
                                else:
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
                                elif current_name == "kl8":
                                    ai_nums = ai_best["nums"]
                                    nums_html = "".join([f"<span class='kl8-ball'>{x:02d}</span>" for x in ai_nums])
                                    st.markdown(f"预测号码：{nums_html}", unsafe_allow_html=True)
                                    st.markdown(f"✅ 命中号码：{ai_best['matches']} 个")
                                else:
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
        else:
            st.warning("⚠️ AI 功能尚未配置，请在左侧边栏配置 API Key 后使用此功能。")
        
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
                st.dataframe(report_df, use_container_width=True)
            
            if report["total_bets"] > 0:
                chart_data = pd.DataFrame(report["records"])
                chart_data["cumulative_profit"] = chart_data["profit"].cumsum()
                st.write("### 📈 累计盈亏走势图")
                st.line_chart(chart_data[["date", "cumulative_profit"]].set_index("date"))

        st.write("### 📌 历史开奖明细")
        st.dataframe(df, use_container_width=True)
        
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
            use_container_width=True
        )
    else:
        st.warning("⚠️ 暂无本地历史数据，请点击左侧控制中心的“立即同步最新数据”进行拉取。")

elif selected_page == "predict":
    st.subheader("🎯 智能预测模型（热温冷概率配比抽样）")
    st.info("💡 **科学选号原理**：根据自首发开奖至今的历史数据频次，自动划分为「热码」、「温码」、「冷码」。使用 “黄金配比”算法，双色球采用 3:2:1 比例组合，快乐 8 选十采用 5:3:2 组合，有效规避不平衡选号！")
    
    col_cnt, _ = st.columns([2, 4])
    with col_cnt:
        n_groups = st.slider("每种彩票生成组数", min_value=1, max_value=10, value=5)
    
    st.write("---")
    
    st.markdown("### 🤖 AI 智能预测")
    if is_ai_configured():
        col_ssq_ai, col_kl8_ai, col_f3d_ai = st.columns(3)
        
        with col_ssq_ai:
            st.markdown("#### 🔴 双色球")
            if st.button("🔮 AI预测双色球", use_container_width=True):
                with st.spinner("AI正在分析双色球趋势..."):
                    ai_ssq = ai_predict_ssq(n_groups)
                    if "error" in ai_ssq:
                        st.error(ai_ssq["error"])
                    else:
                        ai_numbers = ai_ssq.get("recommendations", [])
                        if ai_numbers:
                            ssq_lines = []
                            predictions_to_save = []
                            for i, rec in enumerate(ai_numbers[:n_groups], 1):
                                reds = rec.get("numbers", {}).get("red", [])
                                blue = rec.get("numbers", {}).get("blue", 0)
                                if reds and blue:
                                    predictions_to_save.append({"red": reds, "blue": blue})
                                    red_str = " ".join(f"{x:02d}" for x in reds)
                                    ssq_lines.append(f"第{i:02d}注 红球：{red_str} 蓝球：{blue:02d}")
                                    red_html = "".join([f"<span class='ssq-red'>{x:02d}</span>" for x in reds])
                                    blue_html = f"<span class='ssq-blue'>{blue:02d}</span>"
                                    st.markdown(f"**第 {i:02d} 组**： {red_html} {blue_html}", unsafe_allow_html=True)
                            if ssq_lines:
                                st.code("\n".join(ssq_lines), language="text")
                                st.session_state['pending_ssq_predictions'] = predictions_to_save
            
            if 'pending_ssq_predictions' in st.session_state and st.session_state['pending_ssq_predictions']:
                if st.button("💾 保存双色球预测", use_container_width=True):
                    try:
                        df_ssq = pd.read_csv(os.path.join(DATA_DIR, "ssq.csv"))
                        if not df_ssq.empty:
                            latest_code = str(df_ssq.iloc[0]['code'])
                            next_code = str(int(latest_code) + 1)
                            save_prediction_record("ssq", next_code, st.session_state['pending_ssq_predictions'])
                            st.success(f"✅ 已保存（第 {next_code} 期）")
                    except Exception as e:
                        st.error(f"保存失败: {e}")
        
        with col_kl8_ai:
            st.markdown("#### 🟡 快乐8")
            if st.button("🔮 AI预测快乐8", use_container_width=True):
                with st.spinner("AI正在分析快乐8趋势..."):
                    ai_kl8 = ai_predict_kl8(n_groups)
                    if "error" in ai_kl8:
                        st.error(ai_kl8["error"])
                    else:
                        ai_numbers = ai_kl8.get("recommendations", [])
                        if ai_numbers:
                            kl8_lines = []
                            predictions_to_save = []
                            for i, rec in enumerate(ai_numbers[:n_groups], 1):
                                nums = rec.get("numbers", [])
                                if nums:
                                    predictions_to_save.append({"nums": nums})
                                    nums_str = " ".join(f"{x:02d}" for x in nums)
                                    kl8_lines.append(f"第{i:02d}注 {nums_str}")
                                    nums_html = "".join([f"<span class='kl8-ball'>{x:02d}</span>" for x in nums])
                                    st.markdown(f"**第 {i:02d} 组**： {nums_html}", unsafe_allow_html=True)
                            if kl8_lines:
                                st.code("\n".join(kl8_lines), language="text")
                                st.session_state['pending_kl8_predictions'] = predictions_to_save
            
            if 'pending_kl8_predictions' in st.session_state and st.session_state['pending_kl8_predictions']:
                if st.button("💾 保存快乐8预测", use_container_width=True):
                    try:
                        df_kl8 = pd.read_csv(os.path.join(DATA_DIR, "kl8.csv"))
                        if not df_kl8.empty:
                            latest_code = str(df_kl8.iloc[0]['code'])
                            next_code = str(int(latest_code) + 1)
                            save_prediction_record("kl8", next_code, st.session_state['pending_kl8_predictions'])
                            st.success(f"✅ 已保存（第 {next_code} 期）")
                    except Exception as e:
                        st.error(f"保存失败: {e}")
        
        with col_f3d_ai:
            st.markdown("#### 🟢 福彩3D")
            if st.button("🔮 AI预测福彩3D", use_container_width=True):
                with st.spinner("AI正在分析福彩3D趋势..."):
                    ai_fcsd = ai_predict_fcsd(n_groups)
                    if "error" in ai_fcsd:
                        st.error(ai_fcsd["error"])
                    else:
                        ai_numbers = ai_fcsd.get("recommendations", [])
                        if ai_numbers:
                            fcsd_lines = []
                            predictions_to_save = []
                            for i, rec in enumerate(ai_numbers[:n_groups], 1):
                                nums = rec.get("numbers", [])
                                if nums and len(nums) >= 3:
                                    predictions_to_save.append({"nums": nums})
                                    nums_str = " ".join(str(x) for x in nums)
                                    fcsd_lines.append(f"第{i:02d}注 {nums_str}")
                                    st.markdown(
                                        f"**第 {i:02d} 组**： "
                                        f"<span class='f3d-ball'>{nums[0]}</span>"
                                        f"<span class='f3d-ball'>{nums[1]}</span>"
                                        f"<span class='f3d-ball'>{nums[2]}</span>",
                                        unsafe_allow_html=True
                                    )
                            if fcsd_lines:
                                st.code("\n".join(fcsd_lines), language="text")
                                st.session_state['pending_fcsd_predictions'] = predictions_to_save
            
            if 'pending_fcsd_predictions' in st.session_state and st.session_state['pending_fcsd_predictions']:
                if st.button("💾 保存福彩3D预测", use_container_width=True):
                    try:
                        df_fcsd = pd.read_csv(os.path.join(DATA_DIR, "fcsd.csv"))
                        if not df_fcsd.empty:
                            latest_code = str(df_fcsd.iloc[0]['code'])
                            next_code = str(int(latest_code) + 1)
                            save_prediction_record("fcsd", next_code, st.session_state['pending_fcsd_predictions'])
                            st.success(f"✅ 已保存（第 {next_code} 期）")
                    except Exception as e:
                        st.error(f"保存失败: {e}")
    
    else:
        st.warning("⚠️ AI功能尚未配置，请在左侧边栏配置API Key")
    
    # ===== 体彩 AI 预测 =====
    if is_ai_configured():
        st.write("---")
        st.markdown("### 🏅 体彩 AI 智能预测")
        col_dlt_ai, col_qxc_ai, col_pl3_ai = st.columns(3)
        
        with col_dlt_ai:
            st.markdown("#### 🔵 大乐透")
            if st.button("🔮 AI预测大乐透", use_container_width=True):
                with st.spinner("AI正在分析大乐透趋势..."):
                    ai_dlt = ai_predict_dlt(n_groups)
                    if "error" in ai_dlt:
                        st.error(ai_dlt["error"])
                    else:
                        ai_numbers = ai_dlt.get("recommendations", [])
                        if ai_numbers:
                            dlt_lines = []
                            for i, rec in enumerate(ai_numbers[:n_groups], 1):
                                fronts = rec.get("numbers", {}).get("front", [])
                                backs = rec.get("numbers", {}).get("back", [])
                                if fronts and backs:
                                    f_str = " ".join(f"{x:02d}" for x in fronts)
                                    b_str = " ".join(f"{x:02d}" for x in backs)
                                    dlt_lines.append(f"第{i:02d}注 前区：{f_str} 后区：{b_str}")
                                    f_html = "".join([f"<span class='ssq-red'>{x:02d}</span>" for x in fronts])
                                    b_html = "".join([f"<span class='ssq-blue'>{x:02d}</span>" for x in backs])
                                    st.markdown(f"**第 {i:02d} 组**： {f_html} {b_html}", unsafe_allow_html=True)
                            if dlt_lines:
                                st.code("\n".join(dlt_lines), language="text")
        
        with col_qxc_ai:
            st.markdown("#### 🟣 七星彩")
            if st.button("🔮 AI预测七星彩", use_container_width=True):
                with st.spinner("AI正在分析七星彩趋势..."):
                    ai_qxc = ai_predict_qxc(n_groups)
                    if "error" in ai_qxc:
                        st.error(ai_qxc["error"])
                    else:
                        ai_numbers = ai_qxc.get("recommendations", [])
                        if ai_numbers:
                            qxc_lines = []
                            for i, rec in enumerate(ai_numbers[:n_groups], 1):
                                nums = rec.get("numbers", [])
                                if nums:
                                    nums_str = " ".join(str(x) for x in nums)
                                    qxc_lines.append(f"第{i:02d}注 {nums_str}")
                                    nums_html = " ".join([f"<span class='f3d-ball'>{x}</span>" for x in nums])
                                    st.markdown(f"**第 {i:02d} 组**： {nums_html}", unsafe_allow_html=True)
                            if qxc_lines:
                                st.code("\n".join(qxc_lines), language="text")
        
        with col_pl3_ai:
            st.markdown("#### 🟤 排列三")
            if st.button("🔮 AI预测排列三", use_container_width=True):
                with st.spinner("AI正在分析排列三趋势..."):
                    ai_pl3 = ai_predict_pl3(n_groups)
                    if "error" in ai_pl3:
                        st.error(ai_pl3["error"])
                    else:
                        ai_numbers = ai_pl3.get("recommendations", [])
                        if ai_numbers:
                            pl3_lines = []
                            for i, rec in enumerate(ai_numbers[:n_groups], 1):
                                nums = rec.get("numbers", [])
                                if nums and len(nums) >= 3:
                                    pl3_lines.append(f"第{i:02d}注 {nums[0]} {nums[1]} {nums[2]}")
                                    st.markdown(
                                        f"**第 {i:02d} 组**： "
                                        f"<span class='f3d-ball'>{nums[0]}</span>"
                                        f"<span class='f3d-ball'>{nums[1]}</span>"
                                        f"<span class='f3d-ball'>{nums[2]}</span>",
                                        unsafe_allow_html=True
                                    )
                            if pl3_lines:
                                st.code("\n".join(pl3_lines), language="text")
    
    # ===== v2.0 集成预测模式（本地+蒙特卡洛+马尔可夫三路投票） =====
    if ENSEMBLE_AVAILABLE:
        st.write("---")
        st.markdown("### 🧬 集成预测（贝叶斯融合 + 蒙特卡洛 + 马尔可夫链三路投票）")
        st.info("🔬 **算法原理**：对历史数据进行**指数衰减加权**、**马尔可夫转移矩阵**、**遗漏回补概率**、**贝叶斯融合**四维建模后，通过 **10,000 次蒙特卡洛模拟** 投票选出高置信度号码。同时校验 AC值、跨度、012路、质合比、尾数等数学约束。")
        
        col_ens_type, col_ens_cnt = st.columns([2, 3])
        with col_ens_type:
            ensemble_lottery = st.selectbox(
                "选择彩种",
                ["双色球 (SSQ)", "快乐8 (KL8)", "福彩3D (FCSD)",
                 "大乐透 (DLT)", "七星彩 (QXC)", "排列三 (PL3)"],
                key="ensemble_lottery"
            )
        with col_ens_cnt:
            ensemble_n = st.slider("生成组数", 1, 10, 5, key="ensemble_n")
        
        if st.button("🧬 运行集成预测", type="primary", use_container_width=True, key="btn_ensemble"):
            with st.spinner("🧬 正在运行四维建模 + 10,000次蒙特卡洛模拟..."):
                lt_map = {
                    "双色球 (SSQ)": "ssq", "快乐8 (KL8)": "kl8", "福彩3D (FCSD)": "fcsd",
                    "大乐透 (DLT)": "dlt", "七星彩 (QXC)": "qxc", "排列三 (PL3)": "pl3"
                }
                lt = lt_map[ensemble_lottery]
                
                # 获取特征摘要
                try:
                    summary = get_feature_summary(lt)
                except Exception:
                    summary = {}
                
                # 获取集成预测
                try:
                    ensemble = get_ensemble_prediction(lt, ensemble_n)
                except Exception as e:
                    st.error(f"集成预测失败: {e}")
                    ensemble = {}
                
                if ensemble and "error" not in ensemble:
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
                        st.plotly_chart(fig, use_container_width=True)
                        
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
                            cols_display = st.columns(min(len(recs), 3))
                            for i, rec in enumerate(recs):
                                with cols_display[i % 3]:
                                    nums = rec.get("nums", [])
                                    conf = rec.get("confidence", 0)
                                    nums_html = "".join([f"<span class='kl8-ball'>{x:02d}</span>" for x in nums])
                                    st.markdown(f"**组 {i+1}** 置信度 {conf:.1%}")
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
                        st.markdown("**📋 一键复制推荐号码（投注用）**")
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
                                copy_lines.append(f"第{i:02d}注 {nums_str}")
                            copy_text = "-----集成预测 快乐8-----\n" + "\n".join(copy_lines)
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
                else:
                    st.warning("集成预测需要历史数据支持，请先同步数据")

elif selected_page == "hedge":
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
        if st.button("🔍 AI 分析当前方案并推荐最优策略", type="primary", use_container_width=True):
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
            if st.button("💾 保存双色球预测记录", type="secondary", use_container_width=True):
                try:
                    df_ssq = pd.read_csv(os.path.join(DATA_DIR, "ssq.csv"))
                    if not df_ssq.empty:
                        latest_code = str(df_ssq.iloc[0]['code'])
                        next_code = str(int(latest_code) + 1)
                        save_prediction_record("ssq", next_code, st.session_state['pending_ssq_predictions'])
                        st.success(f"✅ 双色球预测记录已保存（第 {next_code} 期）")
                except Exception as e:
                    st.error(f"保存失败: {e}")
        
        if 'pending_hedge_predictions' in st.session_state and st.session_state['pending_hedge_predictions']:
            hedge_type = st.session_state['pending_hedge_type']
            hedge_name = st.session_state['pending_hedge_name']
            if st.button(f"💾 保存{hedge_name}预测记录", type="secondary", use_container_width=True):
                try:
                    df_file = os.path.join(DATA_DIR, f"{hedge_type}.csv")
                    df = pd.read_csv(df_file)
                    if not df.empty:
                        latest_code = str(df.iloc[0]['code'])
                        next_code = str(int(latest_code) + 1)
                        save_prediction_record(hedge_type, next_code, st.session_state['pending_hedge_predictions'])
                        st.success(f"✅ {hedge_name}预测记录已保存（第 {next_code} 期）")
                except Exception as e:
                    st.error(f"保存失败: {e}")
    else:
        st.info("💡 请在侧边栏配置 AI API Key，启用 AI 智能分析功能")

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
        ai_lottery_type = st.radio(
            "选择彩种",
            ["双色球 (SSQ)", "快乐 8 (KL8)", "福彩 3D (FCSD)"],
            horizontal=True
        )
    
    st.write("---")
    
    # AI 预测按钮
    if st.button("🚀 启动 AI 深度分析", type="primary", use_container_width=True):
        with st.spinner("🤖 AI 正在分析历史数据并生成预测报告..."):
            try:
                if "双色球" in ai_lottery_type:
                    # 并行执行：本地预测 + AI 预测 + 趋势分析
                    col_local, col_ai = st.columns(2)
                    
                    with col_local:
                        st.markdown("### 📊 本地算法预测（热温冷配比）")
                        local_ssq = predict_ssq(ai_n_groups)
                        for i, (reds, blue) in enumerate(local_ssq, 1):
                            red_html = "".join([f"<span class='ssq-red'>{x:02d}</span>" for x in reds])
                            blue_html = f"<span class='ssq-blue'>{blue:02d}</span>"
                            st.markdown(f"**第 {i:02d} 组**： {red_html} {blue_html}", unsafe_allow_html=True)
                        st.code(format_ssq(local_ssq), language="text")
                    
                    with col_ai:
                        st.markdown("### 🤖 AI 算法预测（大模型推理）")
                        ai_ssq = ai_predict_ssq(ai_n_groups)
                        if "error" in ai_ssq:
                            st.error(ai_ssq["error"])
                        else:
                            ai_numbers = ai_ssq.get("recommendations", [])
                            if ai_numbers:
                                ssq_lines = []
                                predictions_to_save = []
                                for i, rec in enumerate(ai_numbers[:ai_n_groups], 1):
                                    reds = rec.get("numbers", {}).get("red", [])
                                    blue = rec.get("numbers", {}).get("blue", 0)
                                    if reds and blue:
                                        predictions_to_save.append({"red": reds, "blue": blue})
                                        red_str = " ".join(f"{x:02d}" for x in reds)
                                        ssq_lines.append(f"第{i:02d}注 红球：{red_str} 蓝球：{blue:02d}")
                                        red_html = "".join([f"<span class='ssq-red'>{x:02d}</span>" for x in reds])
                                        blue_html = f"<span class='ssq-blue'>{blue:02d}</span>"
                                        st.markdown(f"**第 {i:02d} 组**： {red_html} {blue_html}", unsafe_allow_html=True)
                                if ssq_lines:
                                    st.markdown("📋 复制 AI 推荐号码")
                                    st.code("-----AI双色球推荐-----\n" + "\n".join(ssq_lines), language="text")
                            
                            analysis = ai_ssq.get("analysis", "")
                            if analysis:
                                st.markdown("### 📝 AI 分析报告")
                                st.markdown(analysis)
                
                elif "快乐 8" in ai_lottery_type:
                    col_local, col_ai = st.columns(2)
                    
                    with col_local:
                        st.markdown("### 📊 本地算法预测（快乐8 选十）")
                        local_kl8 = predict_kl8(ai_n_groups, 10)
                        for i, nums in enumerate(local_kl8, 1):
                            nums_html = "".join([f"<span class='kl8-ball'>{x:02d}</span>" for x in nums])
                            st.markdown(f"**第 {i:02d} 组**： {nums_html}", unsafe_allow_html=True)
                        st.code(format_kl8(local_kl8), language="text")
                    
                    with col_ai:
                        st.markdown("### 🤖 AI 算法预测（快乐8 选十）")
                        ai_kl8 = ai_predict_kl8(ai_n_groups)
                        if "error" in ai_kl8:
                            st.error(ai_kl8["error"])
                        else:
                            ai_numbers = ai_kl8.get("recommendations", [])
                            if ai_numbers:
                                kl8_lines = []
                                predictions_to_save = []
                                for i, rec in enumerate(ai_numbers[:ai_n_groups], 1):
                                    nums = rec.get("numbers", [])
                                    if nums:
                                        predictions_to_save.append({"nums": nums})
                                        nums_str = " ".join(f"{x:02d}" for x in nums)
                                        kl8_lines.append(f"第{i:02d}注 {nums_str}")
                                        nums_html = "".join([f"<span class='kl8-ball'>{x:02d}</span>" for x in nums])
                                        st.markdown(f"**第 {i:02d} 组**： {nums_html}", unsafe_allow_html=True)
                                if kl8_lines:
                                    st.markdown("📋 复制 AI 推荐号码")
                                    st.code("-----AI快乐8推荐-----\n" + "\n".join(kl8_lines), language="text")
                            
                            analysis = ai_kl8.get("analysis", "")
                            if analysis:
                                st.markdown("### 📝 AI 分析报告")
                                st.markdown(analysis)
                
                else:  # 福彩3D
                    col_local, col_ai = st.columns(2)
                    
                    with col_local:
                        st.markdown("### 📊 本地算法预测（福彩3D）")
                        local_fcsd = predict_fcsd(ai_n_groups)
                        for i, (n1, n2, n3) in enumerate(local_fcsd, 1):
                            st.markdown(
                                f"**第 {i:02d} 组**： "
                                f"<span class='f3d-ball'>{n1}</span>"
                                f"<span class='f3d-ball'>{n2}</span>"
                                f"<span class='f3d-ball'>{n3}</span>",
                                unsafe_allow_html=True
                            )
                        st.code(format_fcsd(local_fcsd), language="text")
                    
                    with col_ai:
                        st.markdown("### 🤖 AI 算法预测（福彩3D）")
                        ai_fcsd = ai_predict_fcsd(ai_n_groups)
                        if "error" in ai_fcsd:
                            st.error(ai_fcsd["error"])
                        else:
                            ai_numbers = ai_fcsd.get("recommendations", [])
                            if ai_numbers:
                                fcsd_lines = []
                                predictions_to_save = []
                                for i, rec in enumerate(ai_numbers[:ai_n_groups], 1):
                                    nums = rec.get("numbers", [])
                                    if nums and len(nums) >= 3:
                                        predictions_to_save.append({"nums": nums})
                                        nums_str = " ".join(str(x) for x in nums)
                                        fcsd_lines.append(f"第{i:02d}注 {nums_str}")
                                        st.markdown(
                                            f"**第 {i:02d} 组**： "
                                            f"<span class='f3d-ball'>{nums[0]}</span>"
                                            f"<span class='f3d-ball'>{nums[1]}</span>"
                                            f"<span class='f3d-ball'>{nums[2]}</span>",
                                            unsafe_allow_html=True
                                        )
                                if fcsd_lines:
                                    st.markdown("📋 复制 AI 推荐号码")
                                    st.code("-----AI福彩3D推荐-----\n" + "\n".join(fcsd_lines), language="text")
                            
                            analysis = ai_fcsd.get("analysis", "")
                            if analysis:
                                st.markdown("### 📝 AI 分析报告")
                                st.markdown(analysis)
                
                # 显示趋势分析
                st.write("---")
                st.markdown("### 📈 历史趋势深度分析")
                trend_col1, trend_col2 = st.columns(2)
                with trend_col1:
                    if st.button("🔍 分析双色球趋势", use_container_width=True):
                        trend_ssq = ai_analyze_trend("ssq")
                        if "error" not in trend_ssq:
                            st.markdown(trend_ssq)
                        else:
                            st.error(trend_ssq["error"])
                with trend_col2:
                    if st.button("🔍 分析快乐8趋势", use_container_width=True):
                        trend_kl8 = ai_analyze_trend("kl8")
                        if "error" not in trend_kl8:
                            st.markdown(trend_kl8)
                        else:
                            st.error(trend_kl8["error"])
                
                st.markdown("---")
                with trend_col1:
                    if st.button("🔍 分析福彩3D趋势", use_container_width=True):
                        trend_fcsd = ai_analyze_trend("fcsd")
                        if "error" not in trend_fcsd:
                            st.markdown(trend_fcsd)
                        else:
                            st.error(trend_fcsd["error"])
                
            except Exception as e:
                st.error(f"AI 分析失败: {e}")
    else:
        # 默认展示：趋势分析按钮
        st.markdown("### 📈 快捷趋势分析")
        st.info("点击上方「🚀 启动 AI 深度分析」按钮，获取完整的 AI 预测报告")
        
        col_quick1, col_quick2, col_quick3 = st.columns(3)
        with col_quick1:
            if st.button("🔍 双色球趋势", use_container_width=True):
                trend = ai_analyze_trend("ssq")
                if "error" not in trend:
                    st.markdown(trend)
                else:
                    st.error(trend["error"])
        with col_quick2:
            if st.button("🔍 快乐8趋势", use_container_width=True):
                trend = ai_analyze_trend("kl8")
                if "error" not in trend:
                    st.markdown(trend)
                else:
                    st.error(trend["error"])
        with col_quick3:
            if st.button("🔍 福彩3D趋势", use_container_width=True):
                trend = ai_analyze_trend("fcsd")
                if "error" not in trend:
                    st.markdown(trend)
                else:
                    st.error(trend["error"])
    
    st.write("---")
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
    
    if st.button("🤖 获取 AI 对冲优化建议", type="primary", use_container_width=True):
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
            

