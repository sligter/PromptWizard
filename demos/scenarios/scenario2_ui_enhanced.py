#!/usr/bin/env python
# scenario2_ui_enhanced.py

import sys
import os
sys.path.insert(0, "../../")
# import promptwizard # Potentially unused
from promptwizard.glue.promptopt.instantiate import GluePromptOpt
from dotenv import load_dotenv
# from tqdm import tqdm # Potentially unused in this script directly

# 导入通用数据处理器
from generic_data_processor import GenericDataProcessor # Ensure this file exists

def run_scenario2_enhanced(
    task_description="你是一个数学专家，你将得到一个数学问题，你需要解决它",
    base_instruction="让我们一步一步地思考。",
    answer_format="对于每个问题，先进行推理，然后给出正确答案。",
    experiment_name="no_data_synthetic_examples",
    model_id="Pro/deepseek-ai/DeepSeek-V3",
    num_train_examples=20, # This is for synthetic generation
    mutation_rounds=2,
    few_shot_count=5,
    generate_expert_identity=True,
    generate_intent_keywords=False,
    generate_reasoning=True,
    mutate_refine_iterations=3,
    min_correct_count=3,
    max_eval_batches=6,
    # num_train_examples for optimization is taken from promptopt_config,
    # if different from synthetic generation count, it could be confusing.
    # For clarity, the optimization phase will use examples from synthetic_data_path
    prompt_technique_name="critique_n_refine",
    questions_batch_size=1,
    refine_instruction=True,
    refine_task_eg_iterations=3,
    seen_set_size=20,
    style_variation=5,
    top_n=1,
    max_history_length=10,
    synthetic_data_path="train_synthetic.jsonl"
):
    """
    执行场景2：没有训练数据，先生成合成样例，然后使用这些合成样例进行提示词优化。

    参数说明:
        task_description (str): 任务的核心描述，告知模型其角色和基本任务。
        base_instruction (str): 提供给模型的基础指令或思考引导，用于提示优化阶段。
        answer_format (str): 期望模型输出答案时遵循的格式规范，用于提示优化阶段。
        experiment_name (str): 本次实验的名称，用于结果保存和区分不同运行。
        model_id (str): 使用的大语言模型的标识符。
        num_train_examples (int): 要生成的合成训练样例的数量。这个数量用于第一步的数据生成。
        mutation_rounds (int): (优化阶段) 提示词进行变异操作的轮数。
        few_shot_count (int): (优化阶段) 构建提示时用作少样本示例的数量。
        generate_expert_identity (bool): (优化阶段) 是否让模型为任务生成一个“专家身份”描述。
        generate_intent_keywords (bool): (优化阶段) 是否尝试从任务描述中提取意图关键词。
        generate_reasoning (bool): (优化阶段) 是否在生成的提示中包含引导模型进行推理的指令。
        mutate_refine_iterations (int): (优化阶段) 变异和精炼过程的迭代次数。
        min_correct_count (int): (优化阶段) 评估时认为提示有效的最小正确样例数。
        max_eval_batches (int): (优化阶段) 最大评估批次数。
        prompt_technique_name (str): (优化阶段) 使用的核心提示工程技术名称。
        questions_batch_size (int): (优化阶段) 处理问题时的批次大小。
        refine_instruction (bool): (优化阶段) 是否对基础指令进行精炼优化。
        refine_task_eg_iterations (int): (优化阶段) 对任务样例进行精炼的迭代次数。
        seen_set_size (int): (优化阶段) 用于避免重复生成相似提示的已见集合的大小。
        style_variation (int): (优化阶段) 生成提示时风格变化的程度或数量。
        top_n (int): (优化阶段) 从候选项中选择前N个最佳结果。
        max_history_length (int): (优化阶段) 保持的优化历史记录的最大长度。
        synthetic_data_path (str): 生成的合成数据保存的路径（.jsonl格式）。此文件将被用于后续的提示优化。
    
    返回:
        tuple: (best_prompt, expert_profile, synthetic_data_path)
               best_prompt (str): 优化得到的最佳提示。
               expert_profile (str): 优化得到的专家画像。
               synthetic_data_path (str): 合成数据实际保存的路径。
    """
    print("执行场景2：没有训练数据，使用合成样例 (V2 with docstrings)")
    load_dotenv(override=True)
    
    # 配置用于第一步：生成合成数据
    # 注意: gp.get_best_prompt的generate_synthetic_examples=True模式可能不使用所有promptopt_config参数
    # 它主要依赖 task_description 和 num_train_examples 来指导数据生成。
    # base_instruction 和 answer_format 在此阶段会被内部生成逻辑覆盖或不使用。
    promptopt_config_step1 = {
        "task_description": task_description,
        "base_instruction": "请为我生成相关的训练数据。", # 一个通用的数据生成指令
        "answer_format": "每个样例包含'question'和'answer'字段。", # 描述生成数据的格式
        "num_train_examples": int(num_train_examples), # 关键：用于生成多少数据
        "unique_model_id": model_id,
        # 以下参数可能在纯数据生成模式下不被 PromptOpt 的 generate_synthetic_examples 使用，
        # 但为保持结构完整性而包含。PromptOpt 内部会决定哪些适用。
        "mutation_rounds": 1, "few_shot_count": 0, "generate_expert_identity": False,
        "generate_intent_keywords": False, "generate_reasoning": False, "mutate_refine_iterations": 1,
        "min_correct_count": 1, "max_eval_batches": 1,
        "prompt_technique_name": "critique_n_refine", # 可能不适用
        "questions_batch_size": 1, "refine_instruction": False, "refine_task_eg_iterations": 1,
        "seen_set_size": 5, "style_variation": 1, "top_n": 1, "max_history_length": 1,
    }
    
    setup_config_step1 = {
        "experiment_name": experiment_name,
        "experiment_id": f"scenario2_step1_datagen_{experiment_name}",
        "model_id": model_id
    }
    
    print("\n步骤1：生成合成数据")
    print("初始化PromptOpt用于生成合成数据...")
    gp_step1 = GluePromptOpt(
        prompt_config=promptopt_config_step1,
        setup_config=setup_config_step1,
        dataset_jsonl=None, # 没有输入数据集
        data_processor=None # 不需要数据处理器来生成数据
    )
    
    print(f"生成{num_train_examples}个合成训练样例到 {synthetic_data_path}...")
    # get_best_prompt 在 generate_synthetic_examples=True 时，主要作用是生成数据
    # 其返回的 best_prompt, expert_profile 在此模式下可能不是我们最终要的提示
    # 它会将生成的数据保存到 prompt_config 中指定的 synthetic_data_path (如果内部逻辑支持)
    # 或者默认保存到与 experiment_id 相关的文件。我们需要确保它保存到我们指定的 synthetic_data_path
    # 这通常通过 PromptOpt 内部配置或其使用的 DataSynthesizer 的 output_path 来控制。
    # GluePromptOpt 可能将 synthetic_data_path 传递给底层的合成器。
    # 如果不是，则合成的数据可能在 gp_step1.synthesizer.output_path 或类似属性中。
    # 为确保数据保存到指定路径，PromptOpt的配置可能需要一个明确的 synthetic_data_output_path 字段。
    # 假设 PromptOpt 会使用 prompt_config 里的 "synthetic_data_path" (如果它设计如此)
    # 或者其内部默认路径是可预测的。
    # 此处我们将 num_train_examples 传入 prompt_config，并期望它能生成数据。

    # Hack: PromptWizard的get_best_prompt可能不直接接受synthetic_data_path作为输出参数
    # 它通常将合成数据保存在 self.synthesizer.output_path
    # 我们需要确保这个路径是可控的，或者在之后重命名/移动文件。
    # 假设其默认输出在实验目录下名为 "train_synthetic.jsonl"
    # 我们在构造 gp_step1 时，其 setup_config.experiment_id 决定了输出目录。
    # 我们期望它写入 synthetic_data_path。如果不是，需要检查PromptWizard的合成逻辑。
    # 临时修改：直接设置合成器输出路径（如果PromptWizard结构允许）
    if hasattr(gp_step1, 'synthesizer') and gp_step1.synthesizer is not None:
         gp_step1.synthesizer.output_path = synthetic_data_path
         print(f"DEBUG: Set synthesizer output path to {synthetic_data_path}")
    elif hasattr(gp_step1, 'prompt_config') and 'synthetic_data_path' not in gp_step1.prompt_config:
        # 如果PromptOpt通过prompt_config来获取输出路径
        gp_step1.prompt_config['synthetic_data_path'] = synthetic_data_path # 尝试设置
        print(f"DEBUG: Added synthetic_data_path to prompt_config: {synthetic_data_path}")


    gp_step1.get_best_prompt(
        use_examples=False, 
        run_without_train_examples=False, # 必须为False才能触发合成或使用数据
        generate_synthetic_examples=True  # 关键：指示生成合成数据
    )

    # 验证合成数据文件是否已生成到期望路径
    # PromptWizard 内部合成数据时，可能保存到 self.synthesizer.output_path
    # 我们需要获取这个实际路径
    actual_synthetic_data_path = synthetic_data_path # 默认为我们期望的路径
    if hasattr(gp_step1, 'synthesizer') and gp_step1.synthesizer is not None and hasattr(gp_step1.synthesizer, 'output_path'):
        actual_synthetic_data_path = gp_step1.synthesizer.output_path
        print(f"合成数据实际保存路径 (来自synthesizer): {actual_synthetic_data_path}")
        if not os.path.exists(synthetic_data_path) and os.path.exists(actual_synthetic_data_path) and actual_synthetic_data_path != synthetic_data_path:
            print(f"警告: 合成数据未在期望路径 {synthetic_data_path} 生成, 而是在 {actual_synthetic_data_path}。")
            # 可以选择复制或重命名文件到 synthetic_data_path，或直接使用 actual_synthetic_data_path
            # 为简单起见，如果期望路径不存在但实际路径存在，我们使用实际路径
            if not os.path.exists(synthetic_data_path):
                 synthetic_data_path = actual_synthetic_data_path


    if not os.path.exists(synthetic_data_path):
        print(f"错误: 合成数据文件 {synthetic_data_path} 在步骤1后未找到!")
        # 尝试在默认实验目录查找
        default_synth_path = os.path.join("experiments", setup_config_step1["experiment_id"], "train_synthetic.jsonl")
        if os.path.exists(default_synth_path):
            print(f"提示: 在默认路径 {default_synth_path} 找到了合成数据。将使用此路径。")
            synthetic_data_path = default_synth_path
        else:
            return "合成数据生成失败", f"未能找到文件 {synthetic_data_path} 或 {default_synth_path}", synthetic_data_path
    
    print(f"合成数据已确认/生成于: {synthetic_data_path}")

    # 步骤2：使用合成数据优化提示
    print("\n步骤2：使用合成数据优化提示")
    
    # 配置用于第二步：使用合成数据进行优化
    # 这里的 num_train_examples 指的是从数据集中加载多少样例用于优化，
    # 而不是再次生成。它应该与上面生成的数量或我们希望用于优化的数量一致。
    # 如果想使用所有生成的样例，可以设置一个较大的数或让PromptOpt内部逻辑处理。
    # 通常，PromptOpt会使用 dataset_jsonl 中的所有数据，num_train_examples可能用于截断。
    promptopt_config_step2 = {
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
        "num_train_examples": int(num_train_examples), # 用于从数据集中选取样本，若为-1或大数则可能用全部
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

    setup_config_step2 = {
        "experiment_name": experiment_name,
        "experiment_id": f"scenario2_step2_optimize_{experiment_name}",
        "model_id": model_id
    }
    
    math_processor = GenericDataProcessor().configure_for_task('math') # 假设是数学任务
    
    print("初始化PromptOpt用于优化提示...")
    gp_step2 = GluePromptOpt(
        prompt_config=promptopt_config_step2,
        setup_config=setup_config_step2,
        dataset_jsonl=synthetic_data_path, # 使用生成的合成数据
        data_processor=math_processor
    )
    
    print("开始使用合成数据优化提示...")
    best_prompt, expert_profile = gp_step2.get_best_prompt(
        use_examples=True, # 使用数据集中的样例
        run_without_train_examples=False, # 因为我们有数据
        generate_synthetic_examples=False # 不再生成数据
    )
    
    print("\n=== 最佳专家画像 (场景2) ===")
    print(expert_profile)
    print("\n=== 最佳提示 (场景2) ===")
    print(best_prompt)
    
    return best_prompt if best_prompt else "未能生成最佳提示", \
           expert_profile if expert_profile else "未能生成专家画像", \
           synthetic_data_path

if __name__ == "__main__":
    # 运行示例
    best_prompt, expert_profile, out_path = run_scenario2_enhanced(num_train_examples=5) # 测试时生成少量样例
    print(f"\n最终结果:\n最佳提示: {best_prompt}\n专家画像: {expert_profile}\n合成数据路径: {out_path}")