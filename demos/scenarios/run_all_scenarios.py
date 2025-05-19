#!/usr/bin/env python
# run_all_scenarios.py

import os
import argparse
from scenario1_no_data_no_examples import run_scenario1
from scenario2_no_data_synthetic_examples import run_scenario2
from scenario3_with_data_with_examples import run_scenario3

def main():
    parser = argparse.ArgumentParser(description="运行PromptWizard场景示例")
    
    parser.add_argument(
        "--scenario", 
        type=int, 
        choices=[1, 2, 3, 0], 
        default=0,
        help="要运行的场景编号：1=无数据无示例，2=无数据有合成示例，3=有数据有示例，0=全部场景"
    )
    
    parser.add_argument(
        "--model", 
        type=str, 
        default="gpt-4o",
        help="要使用的模型ID"
    )
    
    parser.add_argument(
        "--rounds", 
        type=int, 
        default=3,
        help="变异轮数"
    )
    
    args = parser.parse_args()
    
    # 设置环境变量
    os.environ["OPENAI_MODEL_NAME"] = args.model
    
    if args.scenario == 0 or args.scenario == 1:
        print("\n" + "="*50)
        print("场景1：无训练数据，不使用样例")
        print("="*50)
        run_scenario1()
        
    if args.scenario == 0 or args.scenario == 2:
        print("\n" + "="*50)
        print("场景2：无训练数据，使用合成样例")
        print("="*50)
        run_scenario2()
        
    if args.scenario == 0 or args.scenario == 3:
        print("\n" + "="*50)
        print("场景3：有训练数据，使用样例")
        print("="*50)
        run_scenario3()
        
    print("\n所有选定场景执行完成!")

if __name__ == "__main__":
    main()