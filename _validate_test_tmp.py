"""单元测试 _validate_ai_group：合法通过、非法拦截。"""
import sys
sys.path.insert(0, ".")
import importlib
import app as app_mod
importlib.reload(app_mod)

V = app_mod._validate_ai_group

cases = [
    # (lot, rec, 期望 ok)
    ("ssq", {"numbers": {"red": [1,2,3,4,5,6], "blue": 7}}, True),
    ("ssq", {"numbers": {"red": [1,2,3,4,5,5], "blue": 7}}, False),   # 重复
    ("ssq", {"numbers": {"red": [1,2,3,4,5], "blue": 7}}, False),     # 数量不足
    ("ssq", {"numbers": {"red": [1,2,3,4,5,99], "blue": 7}}, False),  # 越界
    ("ssq", {"numbers": {"red": [1,2,3,4,5,6], "blue": 99}}, False),  # 蓝球越界
    ("kl8", {"numbers": list(range(1,11))}, True),
    ("kl8", {"numbers": list(range(1,11)) + [11]}, False),            # 11个
    ("kl8", {"numbers": [1,1,2,3,4,5,6,7,8,9]}, False),               # 重复
    ("fcsd", {"numbers": [0,5,9]}, True),
    ("fcsd", {"numbers": [0,5]}, False),                              # 数量不足
    ("fcsd", {"numbers": [0,5,10]}, False),                           # 越界
    ("pl3", {"numbers": [3,5,3]}, True),                              # 允许各位重复
    ("dlt", {"numbers": {"front": [1,2,3,4,5], "back": [1,2]}}, True),
    ("dlt", {"numbers": {"front": [1,2,3,4,5], "back": [1,1]}}, False),  # 后区重复
    ("dlt", {"numbers": {"front": [1,2,3,4,99], "back": [1,2]}}, False), # 前区越界
    ("qxc", {"numbers": [0,1,2,3,4,5,6]}, True),
    ("qxc", {"numbers": [0,1,2,3,4,5]}, False),                       # 7个不足
]

fails = 0
for lot, rec, exp in cases:
    clean, ok = V(lot, rec)
    status = "OK" if ok == exp else "FAIL"
    if ok != exp:
        fails += 1
        print(f"  [{status}] {lot}: got ok={ok}, expected {exp} | clean={clean}")
    else:
        print(f"  [{status}] {lot}: ok={ok}")

print("\n=== RESULT ===")
print("ALL VALIDATION TESTS PASS" if fails == 0 else f"{fails} FAILED")
sys.exit(1 if fails else 0)
