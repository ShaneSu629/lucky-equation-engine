path = "app.py"
with open(path, encoding="utf-8") as f:
    lines = f.readlines()

# 福利对冲正文：1-indexed 1557..1792  -> 0-indexed [1556:1792]
ws, we = 1556, 1792
welfare = lines[ws:we]

new_welfare = []
for i, ln in enumerate(welfare):
    stripped = ln.lstrip(" ")
    cur = len(ln) - len(stripped)
    ded = 8 if i == 0 else 4   # 首行原在 else: 下(8空格)，其余在 elif 体(4空格)
    new_indent = max(0, cur - ded)
    new_welfare.append(" " * new_indent + stripped)

func = ["\n", "def _render_welfare_hedge():\n"] + ["    " + ln for ln in new_welfare]

# 用 else 分支替换原福利正文
else_branch = ["    else:\n", "        _render_welfare_hedge()\n"]
rest = lines[:ws] + else_branch + lines[we:]

# 在 if selected_page == "config": 之前插入函数
insert_at = None
for i, ln in enumerate(rest):
    if ln.strip().startswith('if selected_page == "config":'):
        insert_at = i
        break
if insert_at is None:
    raise SystemExit("anchor not found")

new_lines = rest[:insert_at] + func + rest[insert_at:]
with open(path, "w", encoding="utf-8") as f:
    f.writelines(new_lines)
print("welfare func inserted at", insert_at, "; total lines", len(new_lines))
