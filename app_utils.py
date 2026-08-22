# app_utils.py
"""
集中放置被多个视图/组件复用的辅助函数，
避免子模块 `from app import ...` 引发的循环导入
（app.py 顶层直接渲染 UI，被重复 import 会触发 StreamlitDuplicateElementKey）。
"""
import streamlit as st
import logging

logger = logging.getLogger("app")


def _toast_save(msg: str, icon: str = "✅"):
    """保存操作统一反馈：toast + success + 日志。"""
    logger.info(msg)
    try:
        st.toast(msg, icon=icon)
    except Exception:
        pass
    st.success(msg)


def _toast_error(msg: str):
    """保存失败统一反馈。"""
    logger.error(msg)
    try:
        st.toast(msg, icon="❌")
    except Exception:
        pass
    st.error(msg)


def _validate_ai_group(lot: str, rec: dict, pick_size: int = None):
    """校验并归一化一条 AI 推荐号码。"""
    nums = rec.get("numbers", {})
    try:
        if lot == "ssq":
            reds = [int(x) for x in nums.get("red", [])]
            blue = int(nums.get("blue", 0))
            if (len(reds) == 6 and len(set(reds)) == 6
                    and all(1 <= r <= 33 for r in reds) and 1 <= blue <= 16):
                return {"red": reds, "blue": blue}, True
        elif lot == "dlt":
            fronts = [int(x) for x in nums.get("front", [])]
            backs = [int(x) for x in nums.get("back", [])]
            if (len(fronts) == 5 and len(set(fronts)) == 5 and all(1 <= f <= 35 for f in fronts)
                    and len(backs) == 2 and len(set(backs)) == 2 and all(1 <= b <= 12 for b in backs)):
                return {"nums": fronts + backs}, True
        else:
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
