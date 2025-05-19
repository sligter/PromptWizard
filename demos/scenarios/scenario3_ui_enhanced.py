#!/usr/bin/env python
# scenario3_ui_enhanced.py

import sys
import os
sys.path.insert(0, "../../")
# import promptwizard # Potentially unused
from promptwizard.glue.promptopt.instantiate import GluePromptOpt
from dotenv import load_dotenv
# from tqdm import tqdm # Potentially unused

from generic_data_processor import GenericDataProcessor # Ensure this file exists

def run_scenario3_enhanced(
    task_description="你是一个数学专家，你将得到一个数学问题，你需要解决它",
    base_instruction="让我们一步一步地思考。",
    answer_format="对于每个问题，先进行推理，然后给出正确答案。",
    experiment_name="with_data_with_examples",
    model_id="Pro/deepseek-ai/DeepSeek-V3",
    dataset_path="train_synthetic.jsonl", # 默认使用场景2可能生成的文件
    data_task_type="math",
    mutation_rounds=2,
    few_shot_count=5,
    generate_expert_identity=True,
    generate_intent_keywords=False,
    generate_reasoning=True,
    mutate_refine_iterations=3,
    min_correct_count=3,
    max_eval_batches=6,
    num_train_examples=20, # 从数据集中加载的样例数
    prompt_technique_name="critique_n_refine",
    questions_batch_size=1,
    refine_instruction=True,
    refine_task_eg_iterations=3,
    seen_set_size=20,
    style_variation=5,
    top_n=1,
    max_history_length=10
):
    """
    执行场景3：有训练数据，使用样例进行提示词优化。

    参数说明:
        task_description (str): 任务的核心描述，告知模型其角色和基本任务。
        base_instruction (str): 提供给模型的基础指令或思考引导。
        answer_format (str): 期望模型输出答案时遵循的格式规范。
        experiment_name (str): 本次实验的名称，用于结果保存和区分不同运行。
        model_id (str): 使用的大语言模型的标识符。
        dataset_path (str): 训练数据集的路径（JSONL格式）。此数据集将用于提示优化。
        data_task_type (str): 数据集的任务类型，如 'math', 'code', 'text'。用于选择合适的数据处理器。
        mutation_rounds (int): 提示词进行变异操作的轮数。
        few_shot_count (int): 构建提示时用作少样本示例的数量。
        generate_expert_identity (bool): 是否让模型为任务生成一个“专家身份”描述。
        generate_intent_keywords (bool): 是否尝试从任务描述中提取意图关键词。
        generate_reasoning (bool): 是否在生成的提示中包含引导模型进行推理的指令。
        mutate_refine_iterations (int): 变异和精炼过程的迭代次数。
        min_correct_count (int): 评估时认为提示有效的最小正确样例数。
        max_eval_batches (int): 最大评估批次数。
        num_train_examples (int): 从提供的`dataset_path`中加载并用于优化的样例数量。如果设为-1或一个非常大的数，通常表示使用数据集中的所有样例。
        prompt_technique_name (str): 使用的核心提示工程技术名称。
        questions_batch_size (int): 处理问题时的批次大小。
        refine_instruction (bool): 是否对基础指令进行精炼优化。
        refine_task_eg_iterations (int): 对任务样例进行精炼的迭代次数。
        seen_set_size (int): 用于避免重复生成相似提示的已见集合的大小。
        style_variation (int): 生成提示时风格变化的程度或数量。
        top_n (int): 从候选项中选择前N个最佳结果。
        max_history_length (int): 保持的优化历史记录的最大长度。
    
    返回:
        tuple: (best_prompt, expert_profile)
               best_prompt (str): 优化得到的最佳提示。
               expert_profile (str): 优化得到的专家画像。
    """
    print("执行场景3：有训练数据，使用样例 (V2 with docstrings)")
    load_dotenv(override=True)
    
    if not os.path.exists(dataset_path):
        print(f"错误: 数据集文件 {dataset_path} 不存在!")
        return f"数据集文件 {dataset_path} 未找到", "请检查路径或先生成数据"

    data_processor = GenericDataProcessor().configure_for_task(data_task_type)
    
    promptopt_config = {
        "task_description": task_description,
        "base_instruction": base_instruction,
        "answer_format": answer_format,
        "mutation_rounds": int(mutation_rounds),
        "few_shot_count": int(few_shot_count),
        "generate_expert_identity": bool(generate_expert_identity),
        "generate_intent_keywords": bool(generate_intent_keywords),
        "generate_reasoning": bool(generate_reasoning),
        "mutate_refine_iterations": int(mutate_refine_iterations),
        "min_correct_count": int(min_correct_count),
        "max_eval_batches": int(max_eval_batches),
        "num_train_examples": int(num_train_examples),
        "prompt_technique_name": prompt_technique_name,
        "questions_batch_size": int(questions_batch_size),
        "refine_instruction": bool(refine_instruction),
        "refine_task_eg_iterations": int(refine_task_eg_iterations),
        "seen_set_size": int(seen_set_size),
        "style_variation": int(style_variation),
        "top_n": int(top_n),
        "unique_model_id": model_id,
        "max_history_length": int(max_history_length),
    }
    
    setup_config = {
        "experiment_name": experiment_name,
        "experiment_id": f"scenario3_{experiment_name}",
        "model_id": model_id
    }
    
    print("初始化PromptOpt...")
    gp = GluePromptOpt(
        prompt_config=promptopt_config,
        setup_config=setup_config,
        dataset_jsonl=dataset_path,
        data_processor=data_processor
    )
    
    print("开始优化提示...")
    best_prompt, expert_profile = gp.get_best_prompt(
        use_examples=True, 
        run_without_train_examples=False, 
        generate_synthetic_examples=False
    )
    
    print("\n=== 最佳专家画像 (场景3) ===")
    print(expert_profile)
    print("\n=== 最佳提示 (场景3) ===")
    print(best_prompt)
    
    return best_prompt if best_prompt else "未能生成最佳提示", \
           expert_profile if expert_profile else "未能生成专家画像"

if __name__ == "__main__":
    # 假设 train_synthetic.jsonl 存在 (例如由场景2生成)
    if not os.path.exists("train_synthetic.jsonl"):
        print("场景3测试需要 train_synthetic.jsonl 文件。请先运行场景2或提供一个有效的数据集。")
    else:
        best_prompt, expert_profile = run_scenario3_enhanced()
        print(f"\n最终结果:\n最佳提示: {best_prompt}\n专家画像: {expert_profile}")