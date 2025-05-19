#!/usr/bin/env python
# scenario3_with_data_with_examples.py

import sys
import os
sys.path.insert(0, "../../")  # 调整为正确的项目根路径
import promptwizard
from promptwizard.glue.promptopt.instantiate import GluePromptOpt
from dotenv import load_dotenv
from tqdm import tqdm

# 导入通用数据处理器
from generic_data_processor import GenericDataProcessor

def run_scenario3():
    """执行场景3：有训练数据，使用样例"""
    print("执行场景3：有训练数据，使用样例")
    
    # 加载环境变量
    load_dotenv(override=True)
    
    # 初始化处理器 - 使用通用处理器替代GSM8k
    math_processor = GenericDataProcessor().configure_for_task('math')
    
    # 创建数据目录
    if not os.path.exists("data"):
        os.mkdir("data")
    
    # 加载和保存数据集
    print("准备训练数据...")
    
    # try:
    #     from datasets import load_dataset
    #     dataset = load_dataset("openai/gsm8k", "main")
        
    #     num_samples = 0
    #     for dataset_type in ['train', 'test']:
    #         data_list = []
    #         for data in dataset[dataset_type]:
    #             data_list.append({"question": data['question'], "answer": data['answer']})
    #             if num_samples == 100 and dataset_type == 'train':  # 只取100个训练样例
    #                 break
    #             num_samples += 1
                
    #         # 保存为JSONL格式
    #         math_processor.dataset_to_jsonl(f"data/{dataset_type}.jsonl", dataset=data_list)
            
    #     print(f"数据已保存到 data/ 目录，共处理 {num_samples} 个样本")
    # except Exception as e:
    #     print(f"无法加载数据集: {e}")
    #     print("正在创建示例数据...")
    #     # 创建一些示例数据用于演示
    #     example_data = [
    #         {"question": "2+2=?", "answer": "#### 4"},
    #         {"question": "3*4=?", "answer": "#### 12"},
    #         {"question": "10-5=?", "answer": "#### 5"}
    #     ]
    #     math_processor.dataset_to_jsonl("data/train.jsonl", dataset=example_data)
    #     print("已创建示例数据")
    
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
        "experiment_name": "with_data_with_examples",
        "experiment_id": "scenario3",
        "model_id": "Pro/deepseek-ai/DeepSeek-V3"
    }
    
    # 创建PromptOpt对象
    print("初始化PromptOpt...")
    gp = GluePromptOpt(
        prompt_config=promptopt_config,
        setup_config=setup_config,
        dataset_jsonl="C:/Users/liz119/newer/AIGC/PromptWizard/demos/scenarios/train_synthetic.jsonl",
        data_processor=math_processor  # 使用通用处理器
    )
    
    # 调用优化函数
    print("开始优化提示...")
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
    run_scenario3()