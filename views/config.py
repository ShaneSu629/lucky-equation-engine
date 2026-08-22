# pages/config.py
"""
配置中心页面
"""
import streamlit as st


def render():
    """渲染配置中心页面。"""
    st.subheader("⚙️ AI 模型配置中心")

    with st.container():
        from ai_predict import is_cloud_deployed, load_config
        cloud_mode = is_cloud_deployed()

        if cloud_mode:
            st.success("☁️ 检测到 Streamlit Secrets 配置，AI 已就绪（云端模式）")
            st.info("💡 云端部署通过 Settings → Secrets 配置，无需在此手动输入")
        else:
            st.info("💡 配置后可使用 AI 深度分析功能，提升预测科学性（本地模式）")

        _current_cfg = load_config()

        api_key = st.text_input(
            "LLM API Key",
            value="" if cloud_mode else "",
            type="password",
            placeholder="sk-xxxxxxxxxxxxxxxx（云端已通过 Secrets 配置，可留空）" if cloud_mode else "sk-xxxxxxxxxxxxxxxx",
            help="OpenAI 兼容接口的 API Key",
            disabled=cloud_mode,
        )
        base_url = st.text_input(
            "Base URL",
            value=_current_cfg.get("base_url", "https://api.siliconflow.cn/v1"),
            placeholder="https://api.openai.com/v1",
            help="OpenAI 兼容的 API 地址",
            disabled=cloud_mode,
        )
        model_name = st.text_input(
            "模型名称",
            value=_current_cfg.get("model", "deepseek-ai/DeepSeek-R1"),
            placeholder="deepseek-ai/DeepSeek-R1",
            help="支持的模型：DeepSeek-R1, Qwen, GLM 等",
            disabled=cloud_mode,
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
                            st.success("✅ AI 配置已保存！正在刷新页面...")
                            st.rerun()
                        except Exception as e:
                            st.error(f"保存失败: {e}")
        else:
            st.markdown("---")
            st.markdown("**📋 Streamlit Secrets 配置格式**")
            st.code('''[ai_config]
api_key = "sk-xxxxxxxxxxxxxxxx"
base_url = "https://api.siliconflow.cn/v1"
model = "deepseek-ai/DeepSeek-R1"''', language="toml")
