#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RPG 上下文管理器 - RPG 故事引擎
用于管理游戏状态、角色数据、物品栏和事件日志
"""

import json
import os
import argparse
from pathlib import Path
from datetime import datetime

# 存档根目录
MEMORY_ROOT = Path("memory/rpg")


def get_campaign_path(campaign):
    """获取战役存档路径"""
    return MEMORY_ROOT / campaign


def load_json(path):
    """
    加载 JSON 文件
    
    参数:
        path: 文件路径
    返回:
        解析后的字典，文件不存在则返回空字典
    """
    if not path.exists():
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(path, data):
    """
    保存数据到 JSON 文件
    
    参数:
        path: 文件路径
        data: 要保存的数据
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def init_campaign(campaign, system, setting, tone, char_name, archetype):
    """
    初始化新战役
    
    参数:
        campaign: 战役名称
        system: 规则系统 (d20/pbta/freeform)
        setting: 世界设定
        tone: 基调
        char_name: 角色名称
        archetype: 角色职业/原型
    """
    path = get_campaign_path(campaign)
    path.mkdir(parents=True, exist_ok=True)
    
    # 创建世界状态文件
    world = {
        "campaign": campaign,
        "system": system,
        "setting": setting,
        "tone": tone,
        "location": "starting_area",
        "time": "08:00",
        "weather": "clear",
        "flags": {}
    }
    save_json(path / "world.json", world)
    
    # 创建角色数据文件
    char = {
        "name": char_name,
        "archetype": archetype,
        "stats": {
            "hp": {"current": 20, "max": 20},
            "sanity": {"current": 50, "max": 50}
        },
        "inventory": [],
        "quests": []
    }
    save_json(path / "character.json", char)

    
    # 创建 NPC 数据文件
    npcs = {}
    save_json(path / "npcs.json", npcs)
    
    # 创建事件日志文件
    journal_path = path / "journal.md"
    date_str = datetime.now().strftime("%Y-%m-%d")
    with open(journal_path, 'w', encoding='utf-8') as f:
        f.write(f"# {campaign} - 冒险日志\n\n")
        f.write(f"- **战役开始**: {date_str}\n")
        f.write(f"- **世界设定**: {setting}\n")
        f.write(f"- **基调**: {tone}\n")
        f.write(f"- **规则系统**: {system}\n")
        f.write(f"- **主角**: {char_name}，{archetype}\n\n## 故事开始\n")
    
    print(f"战役 '{campaign}' 初始化成功，存档位置: {path}")


def get_state(campaign):
    """
    获取当前世界状态
    
    参数:
        campaign: 战役名称
    """
    path = get_campaign_path(campaign)
    world = load_json(path / "world.json")
    print(json.dumps(world, indent=2, ensure_ascii=False))


def set_flag(campaign, key, value):
    """
    设置世界标记
    
    参数:
        campaign: 战役名称
        key: 标记名称
        value: 标记值
    """
    path = get_campaign_path(campaign) / "world.json"
    world = load_json(path)
    
    # 自动类型转换：布尔 → 整数 → 浮点数 → 字符串
    if value.lower() == 'true':
        value = True
    elif value.lower() == 'false':
        value = False
    else:
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                pass  # 保持为字符串
    
    world.setdefault("flags", {})[key] = value
    save_json(path, world)
    print(f"已设置标记 '{key}' 为 {value}")


def update_char(campaign, stat, amount):
    """
    更新角色属性
    
    参数:
        campaign: 战役名称
        stat: 属性名称（任意名称，如 hp、生命值、理智、信用点等）
        amount: 变化量（正数增加，负数减少）
    
    说明:
        - 如果属性值是 {"current": X, "max": Y} 结构，则自动处理上下限
        - 如果属性值是简单数值，则直接加减
        - 属性统一存储在 stats 字典中
    """
    path = get_campaign_path(campaign) / "character.json"
    char = load_json(path)
    char.setdefault("stats", {})
    
    amount = int(amount)
    current_value = char["stats"].get(stat)
    
    if isinstance(current_value, dict) and "current" in current_value:
        # 有上限的资源类型（如 {"current": 20, "max": 20}）
        current_value["current"] += amount
        # 不超过上限
        if "max" in current_value and current_value["current"] > current_value["max"]:
            current_value["current"] = current_value["max"]
        # 不低于 0
        if current_value["current"] < 0:
            current_value["current"] = 0
        char["stats"][stat] = current_value
        print(f"已更新 {stat}: {current_value['current']}/{current_value.get('max', '∞')}")
    else:
        # 简单数值类型
        old_value = current_value if isinstance(current_value, (int, float)) else 0
        char["stats"][stat] = old_value + amount
        print(f"已更新 {stat}: {char['stats'][stat]}")
        
    save_json(path, char)


def inventory(campaign, action, item):
    """
    管理物品栏
    
    参数:
        campaign: 战役名称
        action: 操作类型 (add/remove)
        item: 物品名称
    """
    path = get_campaign_path(campaign) / "character.json"
    char = load_json(path)
    
    if "inventory" not in char:
        char["inventory"] = []
        
    if action == "add":
        char["inventory"].append(item)
        print(f"已添加到物品栏: {item}")
    elif action == "remove":
        if item in char["inventory"]:
            char["inventory"].remove(item)
            print(f"已从物品栏移除: {item}")
        else:
            print(f"物品栏中未找到: {item}")
            return
            
    save_json(path, char)


def log_journal(campaign, entry):
    """
    记录事件日志
    
    参数:
        campaign: 战役名称
        entry: 日志内容
    """
    path = get_campaign_path(campaign) / "journal.md"
    world = load_json(get_campaign_path(campaign) / "world.json")
    
    # 添加时间和天气前缀
    time_str = world.get("time", "")
    weather_str = world.get("weather", "")
    prefix = f"[{time_str} | {weather_str}] " if time_str or weather_str else ""
    
    with open(path, 'a', encoding='utf-8') as f:
        f.write(f"- {prefix}{entry}\n")
    print("日志已记录。")


def main():
    """主函数 - 解析命令行参数"""
    parser = argparse.ArgumentParser(description="RPG 上下文管理器 2.0")
    subparsers = parser.add_subparsers(dest="command")

    # 初始化战役命令
    init_p = subparsers.add_parser("init", help="初始化新战役")
    init_p.add_argument("-c", "--campaign", required=True, help="战役名称")
    init_p.add_argument("--system", required=True, help="规则系统")
    init_p.add_argument("--setting", required=True, help="世界设定")
    init_p.add_argument("--tone", required=True, help="基调")
    init_p.add_argument("--char", required=True, help="角色名称")
    init_p.add_argument("--archetype", required=True, help="角色职业")

    # 获取状态命令
    get_p = subparsers.add_parser("get_state", help="获取世界状态")
    get_p.add_argument("-c", "--campaign", required=True, help="战役名称")

    # 设置标记命令
    flag_p = subparsers.add_parser("set_flag", help="设置世界标记")
    flag_p.add_argument("-c", "--campaign", required=True, help="战役名称")
    flag_p.add_argument("-k", "--key", required=True, help="标记名称")
    flag_p.add_argument("-v", "--value", required=True, help="标记值")

    # 更新角色属性命令
    char_p = subparsers.add_parser("update_char", help="更新角色属性")
    char_p.add_argument("-c", "--campaign", required=True, help="战役名称")
    char_p.add_argument("-s", "--stat", required=True, help="属性名称")
    char_p.add_argument("-a", "--amount", required=True, type=int, help="变化量")

    # 物品栏管理命令
    inv_p = subparsers.add_parser("inventory", help="管理物品栏")
    inv_p.add_argument("-c", "--campaign", required=True, help="战役名称")
    inv_p.add_argument("-a", "--action", choices=["add", "remove"], required=True, help="操作类型")
    inv_p.add_argument("-i", "--item", required=True, help="物品名称")

    # 日志记录命令
    log_p = subparsers.add_parser("log", help="记录事件日志")
    log_p.add_argument("-c", "--campaign", required=True, help="战役名称")
    log_p.add_argument("-e", "--entry", required=True, help="日志内容")

    args = parser.parse_args()

    # 根据命令执行对应函数
    if args.command == "init":
        init_campaign(args.campaign, args.system, args.setting, args.tone, args.char, args.archetype)
    elif args.command == "get_state":
        get_state(args.campaign)
    elif args.command == "set_flag":
        set_flag(args.campaign, args.key, args.value)
    elif args.command == "update_char":
        update_char(args.campaign, args.stat, args.amount)
    elif args.command == "inventory":
        inventory(args.campaign, args.action, args.item)
    elif args.command == "log":
        log_journal(args.campaign, args.entry)


if __name__ == "__main__":
    main()