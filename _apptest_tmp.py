"""临时 AppTest 渲染验证：5 页面 x 福彩/体彩，捕获异常与警告。"""
import sys
import warnings
from streamlit.testing.v1 import AppTest

PAGES = ["dashboard", "predict", "hedge", "ai", "config"]
CATS = {
    "welfare": ["ssq", "kl8", "fcsd"],
    "sports": ["dlt", "qxc", "pl3"],
}

failures = []
warned = []

for page in PAGES:
    for cat, lots in CATS.items():
        for lot in lots:
            try:
                at = AppTest.from_file("app.py", default_timeout=60)
                at.session_state["selected_page"] = page
                at.session_state["lottery_category"] = cat
                at.session_state["selected_lottery"] = lot
                at.run()
                # 捕获脚本内 st.exception 显示的异常
                if at.exception:
                    failures.append((page, cat, lot, "exception", str(at.exception)[:300]))
                # 捕获 st.error 文本（可能含运行错误）
                errs = [e.value for e in at.error]
                for e in errs:
                    if "失败" in e or "Error" in e or "Traceback" in e or "异常" in e:
                        failures.append((page, cat, lot, "error_widget", e[:200]))
            except Exception as ex:  # AppTest 自身抛错（如脚本崩溃）
                failures.append((page, cat, lot, "crash", f"{type(ex).__name__}: {ex}")[:4] if False else (page, cat, lot, "crash", f"{type(ex).__name__}: {str(ex)[:300]}"))

print("=== RESULT ===")
if not failures:
    print("ALL PAGES RENDER OK (0 failures)")
else:
    print(f"TOTAL FAILURES: {len(failures)}")
    for f in failures:
        print(f)

sys.exit(1 if failures else 0)
