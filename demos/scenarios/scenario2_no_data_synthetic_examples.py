#!/usr/bin/env python
# scenario2_no_data_synthetic_examples.py

import sys
import os
sys.path.insert(0, "../../")  # 调整为正确的项目根路径
import promptwizard
from promptwizard.glue.promptopt.instantiate import GluePromptOpt
from dotenv import load_dotenv
from tqdm import tqdm

# 导入通用数据处理器
from generic_data_processor import GenericDataProcessor

def run_scenario2():
    """执行场景2：没有训练数据，使用合成样例"""
    print("执行场景2：没有训练数据，使用合成样例")
    
    # 步骤1：生成合成数据
    print("步骤1：生成合成数据")
    
    # 加载环境变量
    load_dotenv(override=True)
    
    # 定义配置参数
    promptopt_config = {
        "task_description": "You are a mathematics expert. You will be given a mathematics problem which you need to solve",
        "base_instruction": "Lets think step by step.",
        "answer_format": "For each question present the reasoning followed by the correct answer.",
        "mutation_rounds": 2,
        "few_shot_count": 5,
        "generate_expert_identity": True,
        "generate_intent_keywords": False,
        "generate_reasoning": True,
        "mutate_refine_iterations": 3,
        "min_correct_count": 3,
        "max_eval_batches": 6,
        "num_train_examples": 20,
        "prompt_technique_name": "critique_n_refine",
        "questions_batch_size": 1,
        "refine_instruction": True,
        "refine_task_eg_iterations": 3,
        "seen_set_size": 20,
        "style_variation": 5,
        "top_n": 1,
        "unique_model_id": "Pro/deepseek-ai/DeepSeek-V3",
        "max_history_length": 10,
        "prompt_templates_file": "configs/prompt_library.yaml"
    }
    
    setup_config = {
        "experiment_name": "no_data_synthetic_examples",
        "experiment_id": "scenario2_step1",
        "model_id": "gpt-4o"
    }
    
    # 创建PromptOpt对象并生成合成数据
    print("初始化PromptOpt用于生成合成数据...")
    gp = GluePromptOpt(
        promptopt_config=promptopt_config,
        setup_config=setup_config,
        dataset_jsonl=None,
        data_processor=None
    )
    
    # 生成合成样例
    print("生成合成训练样例...")
    gp.get_best_prompt(
        use_examples=False, 
        run_without_train_examples=False, 
        generate_synthetic_examples=True
    )
    
    # 步骤2：使用合成数据优化提示
    print("\n步骤2：使用合成数据优化提示")
    
    # 更新配置
    promptopt_config.update({
        "few_shot_count": 5,
        "generate_reasoning": True,
        "mutate_refine_iterations": 3,
        "seen_set_size": 20
    })
    
    setup_config["experiment_id"] = "scenario2_step2"
    
    # 初始化数据处理器 - 使用通用处理器替代GSM8k
    math_processor = GenericDataProcessor().configure_for_task('math')
    
    # 使用合成数据优化提示
    print("初始化PromptOpt用于优化提示...")
    gp = GluePromptOpt(
        prompt_config=promptopt_config,
        setup_config=setup_config,
        dataset_jsonl="train_synthetic.jsonl",  # 使用生成的合成数据
        data_processor=math_processor
    )
    
    # 调用优化函数
    print("开始使用合成数据优化提示...")
    best_prompt, expert_profile = gp.get_best_prompt(
        use_examples=True, 
        run_without_train_examples=False, 
        generate_synthetic_examples=False
    )
    
    print("\n=== 最佳专家档案 ===")
    print(expert_profile)
    
    print("\n=== 最佳提示 ===")
    print(best_prompt)
    
    return best_prompt, expert_profile

if __name__ == "__main__":
    run_scenario2()