#!/usr/bin/env python
# synthetic_data_generator.py

import sys
import os
sys.path.insert(0, "../../")
# import promptwizard # Potentially unused
from promptwizard.glue.promptopt.instantiate import GluePromptOpt
from dotenv import load_dotenv
# import json # Potentially unused
import time

def create_synthetic_data(
    task_description="你是一个数学专家，你将得到一个数学问题，你需要解决它",
    task_type="math",
    model_id="Pro/deepseek-ai/DeepSeek-V3",
    num_examples=20,
    output_path="train_synthetic.jsonl",
    temperature=0.7,
    max_tokens=1000,
    example_complexity="medium"
):
    """
    创建合成训练数据并保存到指定文件。

    参数说明:
        task_description (str): 任务的核心描述，用于指导模型生成相关数据。例如："你是一个代码生成器，根据需求生成Python函数。"
        task_type (str): 生成数据的任务类型，如 'math', 'code', 'text'。这会影响内部用于生成数据的引导提示。
        model_id (str): 用于生成合成数据的大语言模型的标识符。
        num_examples (int): 要生成的合成训练样例的数量。
        output_path (str): 生成的合成数据保存的文件路径（.jsonl格式）。
        temperature (float): 模型生成文本时的温度参数，控制随机性。较高值更随机，较低值更确定。
        max_tokens (int): 模型为每个样例生成的最大token数量。
        example_complexity (str): 生成样例的复杂度级别，可选值为 'simple', 'medium', 'complex'。
    
    返回:
        str or None: 如果成功，返回输出文件的路径；如果失败，返回None。
    """
    print(f"开始为'{task_type}'任务生成{num_examples}个合成样例... (V2 with docstrings)")
    load_dotenv(override=True)
    
    # PromptOpt 在数据生成模式下，主要依赖 task_description, num_train_examples, unique_model_id。
    # 其他参数如 base_instruction, answer_format 等会被内部生成逻辑调整。
    # 此处我们通过修改 base_instruction 来传递 task_type 和 example_complexity 的意图。
    
    complexity_prompts = {
        "simple": "生成简单难度的问题和答案。问题应该简洁明了，答案应该简短直接。",
        "medium": "生成中等难度的问题和答案。问题应该有一定的思考空间，答案应该包含推理过程。",
        "complex": "生成高难度的问题和答案。问题应该具有挑战性，答案应该包含详细的推理过程和多个步骤。"
    }
    
    task_specific_guidance = ""
    if task_type == "math":
        task_specific_guidance = f"生成{num_examples}个数学问题及其对应的详细解答。"
    elif task_type == "code":
        task_specific_guidance = f"生成{num_examples}个编程问题（例如，用Python实现特定功能），并提供正确的代码和解释。"
    elif task_type == "text":
        task_specific_guidance = f"生成{num_examples}个文本写作相关的任务（例如，写一段关于特定主题的描述，或总结一段文字），并提供高质量的范例答案。"
    else: # general
        task_specific_guidance = f"基于任务描述，生成{num_examples}个相关的问答对或任务样例。"

    # 结合复杂度提示
    full_generation_instruction = f"{task_specific_guidance} {complexity_prompts.get(example_complexity, complexity_prompts['medium'])} "
    full_generation_instruction += "确保每个生成的样例都包含一个'question'字段和一个'answer'字段。"


    promptopt_config = {
        "task_description": task_description, # 核心任务描述
        "base_instruction": full_generation_instruction, # 指导生成具体类型和复杂度的数据
        "answer_format": "输出JSONL格式，每行一个JSON对象，包含 'question' 和 'answer'。", # 对LLM的格式要求
        "num_train_examples": int(num_examples), # 告诉PromptOpt要生成多少
        "unique_model_id": model_id,
        "max_tokens": int(max_tokens), # 控制LLM生成长度
        "temperature": float(temperature), # 控制LLM生成多样性
        "synthetic_data_path": output_path # 期望PromptOpt使用此路径保存
    }
    
    setup_config = {
        "experiment_name": f"synthetic_data_{task_type}_{example_complexity}",
        "experiment_id": f"data_gen_{task_type}_{int(time.time())}",
        "model_id": model_id
    }
    
    print("初始化PromptOpt用于生成合成数据...")
    gp = GluePromptOpt(
        prompt_config=promptopt_config,
        setup_config=setup_config,
        dataset_jsonl=None,
        data_processor=None
    )

    # 再次尝试确保输出路径被正确设置
    # PromptWizard内部可能将合成数据保存在self.synthesizer.output_path
    # 我们需要确保这个路径是我们期望的output_path
    if hasattr(gp, 'synthesizer') and gp.synthesizer is not None:
        gp.synthesizer.output_path = output_path # 覆盖内部合成器的输出路径
        print(f"DEBUG: Explicitly set synthesizer output path to {output_path}")
    elif 'synthetic_data_path' not in gp.prompt_config:
         gp.prompt_config['synthetic_data_path'] = output_path
         print(f"DEBUG: Added synthetic_data_path to prompt_config for generation: {output_path}")


    try:
        print(f"正在通过PromptOpt的get_best_prompt (generate_synthetic_examples=True) 生成样例到 {output_path}...")
        # 在此模式下，get_best_prompt 主要作用是触发数据合成
        gp.get_best_prompt(
            use_examples=False, 
            run_without_train_examples=False, # 必须为False才能触发合成或使用数据
            generate_synthetic_examples=True  # 关键：指示生成合成数据
        )
        
        actual_saved_path = output_path # 默认为期望路径
        if hasattr(gp, 'synthesizer') and gp.synthesizer is not None and hasattr(gp.synthesizer, 'output_path'):
            actual_saved_path = gp.synthesizer.output_path
        
        if os.path.exists(actual_saved_path):
            if actual_saved_path != output_path:
                print(f"数据实际保存在: {actual_saved_path}。将其移动/重命名到期望路径: {output_path}")
                # 确保目标目录存在
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                os.rename(actual_saved_path, output_path) # 或者 shutil.move
            
            with open(output_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                actual_examples_count = len(lines)
            print(f"成功生成 {actual_examples_count} 个样例到 {output_path}")
            return output_path
        else:
            # 尝试在默认实验目录查找（作为备用方案）
            default_exp_output_path = os.path.join("experiments", setup_config["experiment_id"], "train_synthetic.jsonl")
            if os.path.exists(default_exp_output_path):
                print(f"警告: 文件未在 {actual_saved_path} 生成, 但在默认实验路径 {default_exp_output_path} 中找到。")
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                os.rename(default_exp_output_path, output_path)
                print(f"文件已移动到 {output_path}。")
                return output_path
            else:
                print(f"错误: 预期的输出文件 {actual_saved_path} (或 {default_exp_output_path}) 未找到。")
                return None
    except Exception as e:
        print(f"生成合成数据时出错: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    output_file = create_synthetic_data(num_examples=3, task_type="code", example_complexity="simple")
    if output_file:
        print(f"数据已保存到: {output_file}")
        # 打印文件内容的前几行以供检查
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i < 5: print(line.strip())
                    else: break
        except Exception as e:
            print(f"读取生成的文件时出错: {e}")
    else:
        print("数据生成失败")