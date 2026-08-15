"""核心逻辑功能测试：本地预测 + 集成预测 + AI prompt 构建（mock AI）。"""
import sys, json, traceback
sys.path.insert(0, ".")

import generate_picks as gp
import ai_predict as ap

# mock AI 调用，让 prompt 构建（特征工程）全程跑通
FAKE = ({"recommendations": [{"group": 1, "numbers": {}, "reason": "mock"}],
         "analysis": "mock analysis"}, None)
ap._call_ai_json = lambda *a, **k: FAKE
# 顺便确保 get_ensemble 不依赖 AI
import enhanced_predict as ep

LOTS = {
    "ssq": gp.predict_ssq, "kl8": gp.predict_kl8, "fcsd": gp.predict_fcsd,
    "dlt": gp.predict_dlt, "qxc": gp.predict_qxc, "pl3": gp.predict_pl3,
}
AI_FUNCS = {
    "ssq": ap.ai_predict_ssq, "kl8": ap.ai_predict_kl8, "fcsd": ap.ai_predict_fcsd,
    "dlt": ap.ai_predict_dlt, "qxc": ap.ai_predict_qxc, "pl3": ap.ai_predict_pl3,
}

problems = []

print("=== 1) 本地预测 generate_picks.predict_* ===")
for name, fn in LOTS.items():
    try:
        res = fn(3)
        assert isinstance(res, list) and len(res) == 3, f"len!={len(res)}"
        print(f"  {name}: OK -> {res[0]}")
    except Exception as e:
        problems.append(f"predict_{name}: {type(e).__name__}: {e}")
        traceback.print_exc()

print("=== 2) 集成预测 enhanced_predict.get_ensemble_prediction ===")
for name in LOTS:
    try:
        res = ep.get_ensemble_prediction(name, 3)
        if isinstance(res, dict) and "error" in res:
            problems.append(f"ensemble_{name}: {res['error']}")
            print(f"  {name}: ERROR -> {res['error']}")
        else:
            print(f"  {name}: OK")
    except Exception as e:
        problems.append(f"ensemble_{name}: {type(e).__name__}: {e}")
        traceback.print_exc()

print("=== 3) AI prompt 构建 ai_predict_*（mock AI）===")
for name, fn in AI_FUNCS.items():
    try:
        res = fn(3)
        if isinstance(res, dict) and "error" in res:
            problems.append(f"ai_{name}: {res['error']}")
            print(f"  {name}: ERROR -> {res['error']}")
        else:
            print(f"  {name}: OK (no crash in feature engineering)")
    except Exception as e:
        problems.append(f"ai_{name}: {type(e).__name__}: {e}")
        traceback.print_exc()

print("\n=== RESULT ===")
if not problems:
    print("ALL CORE LOGIC OK")
else:
    print(f"PROBLEMS ({len(problems)}):")
    for p in problems:
        print("  -", p)
sys.exit(1 if problems else 0)
