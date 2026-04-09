#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
骰子系统 - RPG 故事引擎
支持 D20、PbtA、优势骰和劣势骰
"""

import argparse
import random
import re


def roll(expression, advantage=False, disadvantage=False):
    """
    执行骰子投掷
    
    参数:
        expression: 骰子表达式 (如 "1d20+5" 或 "pbta+2")
        advantage: 是否使用优势骰（投两次取高）
        disadvantage: 是否使用劣势骰（投两次取低）
    """
    # 支持标准骰子格式，如 1d20+5
    match = re.match(r'(\d+)d(\d+)([+-]\d+)?', expression.lower())
    
    # 检查 PbtA 格式 "pbta+X"
    pbta_match = re.match(r'pbta([+-]\d+)?', expression.lower())
    
    if pbta_match:
        # PbtA 系统：2d6 + 修正值
        count = 2
        sides = 6
        modifier = int(pbta_match.group(1)) if pbta_match.group(1) else 0
    elif match:
        # 标准骰子格式
        count = int(match.group(1))      # 骰子数量
        sides = int(match.group(2))      # 骰子面数
        modifier = int(match.group(3)) if match.group(3) else 0  # 修正值
    else:
        print("格式错误。请使用 XdY+Z（如 1d20+5）或 pbta+Z（如 pbta+2）")
        return

    def do_roll():
        """执行一次投骰"""
        return [random.randint(1, sides) for _ in range(count)]
    
    # 如果同时有优势和劣势，则相互抵消
    if advantage and disadvantage:
        advantage = False
        disadvantage = False

    # 第一次投骰
    rolls1 = do_roll()
    total1 = sum(rolls1) + modifier
    
    if advantage or disadvantage:
        # 需要投两次
        rolls2 = do_roll()
        total2 = sum(rolls2) + modifier
        print(f"第一次投骰: {rolls1} + {modifier} = {total1}")
        print(f"第二次投骰: {rolls2} + {modifier} = {total2}")
        
        if advantage:
            final_total = max(total1, total2)
            print(f"优势骰！取较高值: {final_total}")
        else:
            final_total = min(total1, total2)
            print(f"劣势骰！取较低值: {final_total}")
        
        total = final_total
        rolls = rolls1 if total == total1 else rolls2
    else:
        # 普通投骰
        rolls = rolls1
        total = total1
        print(f"表达式: {expression}")
        print(f"骰子结果: {rolls}")
        print(f"修正值: {modifier:+}")
        print(f"总计: {total}")

    # 特殊结果输出
    if pbta_match:
        # PbtA 系统结果判定
        if total >= 10:
            print("PbtA 结果: 完全成功 (10+)")
        elif total >= 7:
            print("PbtA 结果: 部分成功 (7-9) - 成功但有代价")
        else:
            print("PbtA 结果: 失败 (6-) - GM 做出强硬反应")
    elif not pbta_match and sides == 20 and count == 1:
        # D20 系统的大成功/大失败
        if rolls[0] == 20:
            print("大成功！(自然20)")
        elif rolls[0] == 1:
            print("大失败！(自然1)")


def main():
    """主函数 - 解析命令行参数"""
    parser = argparse.ArgumentParser(description="RPG 骰子系统")
    parser.add_argument("expression", help="骰子表达式（如 1d20+5 或 pbta+2）")
    parser.add_argument("-a", "--advantage", action="store_true", 
                        help="优势骰：投两次取较高值")
    parser.add_argument("-d", "--disadvantage", action="store_true", 
                        help="劣势骰：投两次取较低值")
    args = parser.parse_args()
    roll(args.expression, args.advantage, args.disadvantage)


if __name__ == "__main__":
    main()