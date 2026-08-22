# ui_components.py
"""
通用 UI 组件 — 号码球渲染、保存按钮、开奖展示等
消除每个彩种重复的 AI 预测/保存/渲染代码
"""
import streamlit as st
import logging
from styles import LOTTERY_META, LOT_CATS

logger = logging.getLogger("app")


# ===== 号码球 HTML 渲染 =====

def render_balls_html(lot_type: str, data: dict) -> str:
    """通用号码球 HTML 渲染。

    Args:
        lot_type: 彩种标识 (ssq/dlt/kl8/fcsd/qxc/pl3)
        data: 号码数据，格式取决于彩种：
            - ssq: {"red": [1,2,3,4,5,6], "blue": 7}
            - dlt: {"front": [1,2,3,4,5], "back": [1,2]}
            - kl8: {"nums": [1,2,...10]}
            - fcsd/pl3: {"nums": [1,2,3]}
            - qxc: {"nums": [1,2,3,4,5,6,7]}

    Returns:
        HTML 字符串
    """
    if lot_type == "ssq":
        # 兼容两种上游结构：{"red":[...], "blue":x} 或 {"reds":[...], "blue":x}
        # 以及嵌套 {"nums": {"red":[...], "blue":x}}
        if isinstance(data.get("nums"), dict):
            reds = data["nums"].get("red", [])
            blue = data["nums"].get("blue", 0)
        else:
            reds = data.get("red") or data.get("reds") or (data.get("nums", [])[:6] if isinstance(data.get("nums"), list) else [])
            blue = data.get("blue", 0)
        red_html = "".join(f'<span class="ball-red">{int(x):02d}</span>' for x in reds)
        blue_html = f'<span class="ball-blue">{int(blue):02d}</span>'
        return red_html + blue_html

    if lot_type == "dlt":
        # 兼容 {"front":[...], "back":[...]} / {"fronts":[...], "backs":[...]}
        # 以及嵌套 {"nums": {"front":[...], "back":[...]}}
        if isinstance(data.get("nums"), dict):
            fronts = data["nums"].get("front", [])
            backs = data["nums"].get("back", [])
        else:
            fronts = data.get("front") or data.get("fronts") or (data.get("nums", [])[:5] if isinstance(data.get("nums"), list) else [])
            backs = data.get("back") or data.get("backs") or (data.get("nums", [])[-2:] if isinstance(data.get("nums"), list) and len(data.get("nums", [])) > 5 else [])
        f_html = "".join(f'<span class="ball-red">{int(x):02d}</span>' for x in fronts)
        b_html = "".join(f'<span class="ball-blue">{int(x):02d}</span>' for x in backs)
        return f_html + b_html

    if lot_type == "kl8":
        nums = data.get("nums", [])
        return "".join(f'<span class="ball-orange">{x:02d}</span>' for x in nums)

    # fcsd / pl3 / qxc — 位置制
    nums = data.get("nums", [])
    return "".join(f'<span class="ball-teal">{x}</span>' for x in nums)


def render_latest_draw(lot_type: str, row) -> str:
    """从 DataFrame 行渲染最新开奖号码球。

    Returns:
        HTML 字符串
    """
    meta = LOTTERY_META[lot_type]
    if lot_type == "ssq":
        reds = [row[c] for c in meta["ball_cols"]]
        blue = row[meta["special_col"]]
        return render_balls_html("ssq", {"red": reds, "blue": blue})

    if lot_type == "dlt":
        fronts = [row[c] for c in meta["ball_cols"]]
        backs = [row[c] for c in meta["special_col"]]
        return render_balls_html("dlt", {"front": fronts, "back": backs})

    if lot_type == "kl8":
        nums = [row[c] for c in meta["ball_cols"] if c in row.index]
        return render_balls_html("kl8", {"nums": nums})

    # fcsd / pl3 / qxc
    nums = [row[c] for c in meta["ball_cols"] if c in row.index]
    return render_balls_html(lot_type, {"nums": nums})


# ===== AI 预测结果渲染 =====

def render_ai_prediction(lot_type: str, ai_result: dict, n_groups: int, pick_size: int = None) -> list:
    """渲染 AI 预测结果，返回可保存的 predictions 列表。

    通用函数，替代每个彩种独立的 AI 预测渲染逻辑。
    """
    from app_utils import _validate_ai_group, _toast_error

    meta = LOTTERY_META[lot_type]
    label = meta["label"]
    icon = meta["icon"]

    ai_numbers = ai_result.get("recommendations", [])
    if not ai_numbers:
        st.warning(f"⚠️ AI 未返回有效的{label}推荐号码")
        return []

    predictions_to_save = []
    _dropped = 0

    for i, rec in enumerate(ai_numbers[:n_groups], 1):
        clean, ok = _validate_ai_group(lot_type, rec, pick_size=pick_size)
        # 构造展示数据
        if lot_type == "ssq":
            nums_data = rec.get("numbers", {})
            reds = nums_data.get("red", [])
            blue = nums_data.get("blue", 0)
            display_data = {"red": reds, "blue": blue}
        elif lot_type == "dlt":
            nums_data = rec.get("numbers", {})
            fronts = nums_data.get("front", [])
            backs = nums_data.get("back", [])
            display_data = {"front": fronts, "back": backs}
        else:
            nums = rec.get("numbers", [])
            if isinstance(nums, dict):
                nums = nums.get("nums", [])
            display_data = {"nums": nums}

        html = render_balls_html(lot_type, display_data)
        if html:
            st.markdown(f"**第 {i:02d} 组**：{html}", unsafe_allow_html=True)

        if ok:
            predictions_to_save.append(clean)
        else:
            _dropped += 1

    if _dropped:
        st.warning(f"⚠️ 已忽略 {_dropped} 组不符合规则的号码")

    return predictions_to_save


# ===== 保存按钮 =====

def render_save_button(lot_type: str, pending_key: str, source_label: str = ""):
    """通用保存预测按钮。

    Args:
        lot_type: 彩种标识
        pending_key: session_state 中待保存数据的 key
        source_label: 来源标签（如"AI预测"/"对冲"），用于日志和提示
    """
    from app_utils import _toast_save, _toast_error
    from db_manager import get_latest_code as _db_get_latest_code
    from ai_predict import save_prediction_record

    meta = LOTTERY_META[lot_type]
    label = meta["label"]

    if pending_key not in st.session_state or not st.session_state[pending_key]:
        return

    btn_label = f"💾 保存{label}预测"
    if st.button(btn_label, key=f"save_{pending_key}", width="stretch"):
        try:
            latest_code = _db_get_latest_code(lot_type)
            if latest_code:
                next_code = str(int(latest_code) + 1)
                _n = len(st.session_state[pending_key])
                logger.info(f"[{source_label}] 保存{label}: 期号={next_code}, 组数={_n}")
                save_prediction_record(lot_type, next_code, st.session_state[pending_key])
                _toast_save(f"✅ {label}预测已保存（第 {next_code} 期，{_n} 组）")
                st.session_state[pending_key] = []
            else:
                logger.warning(f"[{source_label}] 保存{label}: 无法获取最新期号")
                _toast_error(f"保存失败：无法获取{label}最新期号")
        except Exception as e:
            _toast_error(f"保存失败: {e}")


# ===== 对比结果渲染 =====

def render_compare_result(lot_type: str, compare_result: dict):
    """渲染预测 vs 实际开奖对比结果。"""
    meta = LOTTERY_META[lot_type]
    label = meta["label"]

    if "error" in compare_result:
        st.error(compare_result["error"])
        return

    latest_data = compare_result["latest"]
    ai_best = compare_result["ai_best"]
    predict_time = compare_result.get("predict_time", "")

    st.markdown(f"**开奖号码：第 {latest_data['code']} 期 ({latest_data['date']})**")
    if predict_time:
        st.caption(f"🕐 预测时间：{predict_time}")

    col_actual, col_ai = st.columns(2)

    with col_actual:
        st.markdown("#### 🎯 实际开奖")
        actual_html = render_balls_html(lot_type, latest_data)
        st.markdown(actual_html, unsafe_allow_html=True)

    with col_ai:
        st.markdown("#### 🤖 最佳命中")
        ai_nums_raw = ai_best.get("nums", {})
        # ai_best['nums'] 在各彩种中格式不同：ssq 是 dict，其余是 list
        if lot_type == "ssq":
            ai_nums = ai_nums_raw if isinstance(ai_nums_raw, dict) else {}
        elif lot_type == "dlt":
            if isinstance(ai_nums_raw, list) and len(ai_nums_raw) >= 7:
                ai_nums = {"front": ai_nums_raw[:5], "back": ai_nums_raw[-2:]}
            else:
                ai_nums = ai_nums_raw if isinstance(ai_nums_raw, dict) else {}
        else:
            ai_nums = {"nums": ai_nums_raw} if isinstance(ai_nums_raw, list) else ai_nums_raw
        ai_html = render_balls_html(lot_type, ai_nums)
        st.markdown(ai_html, unsafe_allow_html=True)

        # 命中信息
        if "red_matches" in ai_best:
            st.markdown(f"✅ 红球命中：{ai_best['red_matches']} 个")
            st.markdown(f"✅ 蓝球命中：{'是' if ai_best.get('blue_match') else '否'}")
        elif "front_matches" in ai_best:
            st.markdown(f"✅ 前区命中：{ai_best['front_matches']} 个")
            st.markdown(f"✅ 后区命中：{ai_best['back_matches']} 个")
        elif "matches" in ai_best:
            match_label = "位" if lot_type in ("fcsd", "pl3", "qxc") else "个"
            st.markdown(f"✅ 命中：{ai_best['matches']} {match_label}")


# ===== 预测记录列表渲染 =====

def render_prediction_records(lot_type: str, records: list):
    """渲染已保存的预测记录列表。"""
    meta = LOTTERY_META[lot_type]

    for record in records:
        st.markdown(f"**🎟️ 第 {record['code']} 期** | 🕐 {record['predict_time']}")
        if record.get('compared'):
            st.markdown("✅ 已对比开奖结果")
        else:
            st.markdown("⏳ 等待开奖")

        predictions = record.get('predictions', [])
        for i, pred in enumerate(predictions, 1):
            html = render_balls_html(lot_type, pred)
            if html:
                st.markdown(f"第 {i:02d} 组：{html}", unsafe_allow_html=True)
            else:
                nums = pred.get("nums", [])
                st.markdown(f"第 {i:02d} 组：{' '.join(str(x) for x in nums)}")


# ===== 对冲号码渲染 =====

def render_local_balls_html(lot_type: str, item) -> str:
    """将本地预测结果（tuple 格式）渲染为 HTML 号码球。"""
    if lot_type == "ssq":
        reds, blue = item
        return render_balls_html("ssq", {"red": reds, "blue": blue})
    if lot_type == "dlt":
        fronts, backs = item
        return render_balls_html("dlt", {"front": fronts, "back": backs})
    if lot_type == "kl8":
        return render_balls_html("kl8", {"nums": item})
    # fcsd / pl3 / qxc
    return render_balls_html(lot_type, {"nums": list(item)})
