#!/usr/bin/env python
# scenario1_no_data_no_examples.py

import sys
import os
sys.path.insert(0, "../../")  # 调整为正确的项目根路径
import promptwizard
from promptwizard.glue.promptopt.instantiate import GluePromptOpt
from promptwizard.glue.promptopt.techniques.common_logic import DatasetSpecificProcessing
from dotenv import load_dotenv

def run_scenario1():
    """执行场景1：没有训练数据，不使用样例"""
    print("执行场景1：没有训练数据，不使用样例")
    
    # 加载环境变量
    load_dotenv(override=True)
    
    # 定义配置参数，替代YAML文件
    promptopt_config = {
        "task_description": "You are a mathematics expert. You will be given a mathematics problem which you need to solve",
        "base_instruction": "Lets think step by step.",
        "mutation_rounds": 5,
        "max_history_length": 10,
        "use_examples_in_context": False,
        "prompt_templates_file": "configs/prompt_library.yaml"  # 仍需要模板库
    }
    
    setup_config = {
        "experiment_name": "no_data_no_examples",
        "experiment_id": "scenario1",
        "model_id": "gpt-4o"  # 确保这个模型ID在环境变量或系统中已配置
    }
    
    # 创建临时配置文件或直接传入参数
    print("初始化PromptOpt...")
    gp = GluePromptOpt(
        promptopt_config=promptopt_config,  # 直接传入字典而非文件路径
        setup_config=setup_config,
        dataset_jsonl=None,
        data_processor=None
    )
    
    # 调用优化函数
    print("开始优化提示...")
    best_prompt, expert_profile = gp.get_best_prompt(
        use_examples=False, 
        run_without_train_examples=True,
        generate_synthetic_examples=False
    )
    
    print("\n=== 最佳专家档案 ===")
    print(expert_profile)
    
    print("\n=== 最佳提示 ===")
    print(best_prompt)
    
    return best_prompt, expert_profile

if __name__ == "__main__":
    run_scenario1()