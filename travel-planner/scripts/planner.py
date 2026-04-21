#!/usr/bin/env python3
# 旅行计划持久化管理器

import json
import argparse
from pathlib import Path
from datetime import datetime, date

MEMORY_ROOT = Path("memory/travel")


def get_plan_path(name):
    return MEMORY_ROOT / name.replace(" ", "_")


def load_json(path):
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def detect_holiday(start_str):
    """检测是否为中国主要节假日"""
    try:
        d = datetime.strptime(start_str, "%Y-%m-%d").date()
    except ValueError:
        return None

    holidays = {
        (4, 30): "五一黄金周", (5, 1): "五一黄金周", (5, 2): "五一黄金周",
        (5, 3): "五一黄金周", (5, 4): "五一黄金周", (5, 5): "五一黄金周",
        (10, 1): "国庆黄金周", (10, 2): "国庆黄金周", (10, 3): "国庆黄金周",
        (10, 4): "国庆黄金周", (10, 5): "国庆黄金周", (10, 6): "国庆黄金周",
        (10, 7): "国庆黄金周",
        (4, 4): "清明节", (4, 5): "清明节", (4, 6): "清明节",
        (6, 22): "端午节", (6, 23): "端午节", (6, 24): "端午节",
    }
    return holidays.get((d.month, d.day))


def cmd_init(args):
    path = get_plan_path(args.name)
    meta_file = path / "plan.json"

    if meta_file.exists():
        print(f"计划 '{args.name}' 已存在，使用 view 查看或直接修改")
        return

    holiday = detect_holiday(args.start)
    plan = {
        "name": args.name,
        "from": args.from_city,
        "to": args.to,
        "start": args.start,
        "end": args.end,
        "people": args.people,
        "group_type": args.type,
        "budget": args.budget,
        "pace": args.pace,
        "holiday": holiday,
        "created_at": datetime.now().isoformat(),
        "days": {},
        "notes": [],
    }
    save_json(meta_file, plan)

    if holiday:
        print(f"✅ 计划 '{args.name}' 已创建")
        print(f"⚠️  检测到节假日：{holiday}，注意提前订票和预约景区！")
    else:
        print(f"✅ 计划 '{args.name}' 已创建，保存于 {path}")


def cmd_add_day(args):
    meta_file = get_plan_path(args.name) / "plan.json"
    plan = load_json(meta_file)

    if not plan:
        print(f"❌ 未找到计划 '{args.name}'，请先运行 init")
        return

    plan.setdefault("days", {})[str(args.day)] = {
        "summary": args.summary,
        "note": args.note or "",
        "updated_at": datetime.now().isoformat(),
    }
    save_json(meta_file, plan)
    print(f"✅ 第 {args.day} 天行程已保存：{args.summary}")


def cmd_view(args):
    meta_file = get_plan_path(args.name) / "plan.json"
    plan = load_json(meta_file)

    if not plan:
        print(f"❌ 未找到计划 '{args.name}'")
        return

    print(f"\n📋 旅行计划：{plan['name']}")
    print(f"   {plan['from']} → {plan['to']}")
    print(f"   {plan['start']} 至 {plan['end']}（{plan['people']}人，{plan['group_type']}）")
    print(f"   预算：{plan['budget']} | 节奏：{plan['pace']}")
    if plan.get("holiday"):
        print(f"   ⚠️  节假日：{plan['holiday']}")

    days = plan.get("days", {})
    if days:
        print("\n每日安排：")
        for day_num in sorted(days.keys(), key=int):
            d = days[day_num]
            print(f"  第{day_num}天: {d['summary']}")
            if d.get("note"):
                print(f"        💡 {d['note']}")
    else:
        print("\n（尚未添加每日行程）")


def cmd_list(args):
    if not MEMORY_ROOT.exists():
        print("暂无已保存的旅行计划")
        return

    plans = []
    for p in MEMORY_ROOT.iterdir():
        meta = load_json(p / "plan.json")
        if meta:
            plans.append(meta)

    if not plans:
        print("暂无已保存的旅行计划")
        return

    print(f"\n已保存的旅行计划（共 {len(plans)} 个）：")
    for plan in plans:
        holiday_tag = f" [{plan['holiday']}]" if plan.get("holiday") else ""
        days_count = len(plan.get("days", {}))
        print(f"  • {plan['name']}{holiday_tag} — {plan['from']}→{plan['to']} "
              f"({plan['start']}~{plan['end']}, 已规划{days_count}天)")


def cmd_export(args):
    meta_file = get_plan_path(args.name) / "plan.json"
    plan = load_json(meta_file)

    if not plan:
        print(f"❌ 未找到计划 '{args.name}'")
        return

    lines = [
        f"# {plan['name']}",
        f"",
        f"- **出发地**：{plan['from']}",
        f"- **目的地**：{plan['to']}",
        f"- **时间**：{plan['start']} 至 {plan['end']}",
        f"- **人员**：{plan['people']} 人（{plan['group_type']}）",
        f"- **预算**：{plan['budget']} | **节奏**：{plan['pace']}",
    ]

    if plan.get("holiday"):
        lines.append(f"- ⚠️ **节假日**：{plan['holiday']}，提前预订交通和住宿")

    days = plan.get("days", {})
    if days:
        lines.append(f"\n## 行程安排\n")
        for day_num in sorted(days.keys(), key=int):
            d = days[day_num]
            lines.append(f"### 第 {day_num} 天")
            lines.append(d["summary"])
            if d.get("note"):
                lines.append(f"> 💡 {d['note']}")
            lines.append("")

    output_path = get_plan_path(args.name) / f"{args.name}.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"✅ 已导出到：{output_path}")


def main():
    parser = argparse.ArgumentParser(description="旅行计划管理器")
    sub = parser.add_subparsers(dest="cmd")

    p_init = sub.add_parser("init", help="初始化新旅行计划")
    p_init.add_argument("-n", "--name", required=True, help="计划名称")
    p_init.add_argument("--from", dest="from_city", required=True, help="出发城市")
    p_init.add_argument("--to", required=True, help="目的地")
    p_init.add_argument("--start", required=True, help="出发日期 YYYY-MM-DD")
    p_init.add_argument("--end", required=True, help="返回日期 YYYY-MM-DD")
    p_init.add_argument("--people", type=int, required=True, help="出行人数")
    p_init.add_argument("--type", required=True, help="出行类型（情侣/家庭/朋友/独行）")
    p_init.add_argument("--budget", required=True, help="预算档次（经济/中等/舒适/奢华）")
    p_init.add_argument("--pace", required=True, help="节奏（休闲/标准/冲刺）")

    p_add = sub.add_parser("add_day", help="添加某天行程摘要")
    p_add.add_argument("-n", "--name", required=True)
    p_add.add_argument("--day", type=int, required=True, help="第几天")
    p_add.add_argument("--summary", required=True, help="行程摘要")
    p_add.add_argument("--note", help="当日提示")

    p_view = sub.add_parser("view", help="查看计划详情")
    p_view.add_argument("-n", "--name", required=True)

    sub.add_parser("list", help="列出所有计划")

    p_export = sub.add_parser("export", help="导出为 Markdown 文件")
    p_export.add_argument("-n", "--name", required=True)

    args = parser.parse_args()

    dispatch = {
        "init": cmd_init,
        "add_day": cmd_add_day,
        "view": cmd_view,
        "list": cmd_list,
        "export": cmd_export,
    }

    if args.cmd in dispatch:
        dispatch[args.cmd](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
