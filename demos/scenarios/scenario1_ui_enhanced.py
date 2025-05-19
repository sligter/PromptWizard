#!/usr/bin/env python
# scenario1_ui_enhanced.py

import sys
import os
sys.path.insert(0, "../../")
from promptwizard.glue.promptopt.instantiate import GluePromptOpt
from dotenv import load_dotenv
from io import StringIO
import contextlib
import re

def parse_variations_from_output(output_text):
    """
    从捕获的 stdout 文本中解析出提示变体。
    返回一个包含所有变体的格式化字符串。
    """
    print_debug = False # 控制是否打印此函数的调试信息

    if print_debug:
        print(f"\n--- DEBUG: parse_variations_from_output ---")
        print(f"--- Raw input text to parse (first 500 chars): ---\n{output_text[:500]}\n--- END RAW ---")

    variations = []
    pattern = re.compile(
        r"Variations\s*(\d+):\s*Expert Profile:\s*(.*?)\s*Prompt:\s*(.*?)(?:\s*Keywords:.*?|\s* വിശദാംശങ്ങൾ:.*?)?(?=\n\s*Variations\s*\d+:|\Z|_Variations_End_)",
        re.DOTALL | re.IGNORECASE
    )
    
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    cleaned_output_text = ansi_escape.sub('', output_text)

    cleaned_output_text = re.sub(r'\s*\d+%\|[█\s]*\|\s*\d+/\d+\s*\[\d{2}:\d{2}.*?\]', '', cleaned_output_text)
    cleaned_output_text = re.sub(r'Iterations completed:.*?\n', '', cleaned_output_text, flags=re.IGNORECASE)
    cleaned_output_text = re.sub(r'Time taken to find best prompt:.*?\n', '', cleaned_output_text, flags=re.IGNORECASE)
    cleaned_output_text = cleaned_output_text.replace("_______________________________________________________________________", "_Variations_End_")

    if print_debug:
        print(f"--- Cleaned text for parsing (first 500 chars): ---\n{cleaned_output_text[:500]}\n--- END CLEANED ---")

    found_variations_header = "Possible prompt variations:" in cleaned_output_text
    search_start_pos = 0
    if found_variations_header:
        header_pos = cleaned_output_text.find("Possible prompt variations:")
        if header_pos != -1:
            first_separator_pos = cleaned_output_text.find("_Variations_End_", header_pos)
            if first_separator_pos != -1:
                search_start_pos = first_separator_pos + len("_Variations_End_")
    
    if print_debug: print(f"--- Search start position for regex: {search_start_pos} ---")

    matches_found = 0
    for match in pattern.finditer(cleaned_output_text, pos=search_start_pos):
        matches_found += 1
        var_num = match.group(1)
        expert_profile = match.group(2).strip()
        prompt_text = match.group(3).strip()
        variations.append({"number": var_num, "expert_profile": expert_profile, "prompt": prompt_text})
        if print_debug:
            print(f"--- Matched Variation {var_num} ---")
            print(f"Expert Profile (first 50): {expert_profile[:50]}...")
            print(f"Prompt (first 50): {prompt_text[:50]}...")

    if print_debug:
        print(f"--- Total matches found by regex: {matches_found} ---")
        if not variations: print(f"--- Regex did not find any variations. Found header: {found_variations_header} ---")

    if not variations:
        if found_variations_header:
            relevant_output = cleaned_output_text.split("Possible prompt variations:", 1)[-1].strip()
            if "_Variations_End_" in relevant_output: relevant_output = relevant_output.split("_Variations_End_",1)[-1].strip()
            if not relevant_output.strip(): return "找到了'Possible prompt variations:'头部，但未能解析到具体变体内容，或内容为空。"
            return f"未能按预期格式解析变体。捕获到的相关内容（已清理）如下：\n\n{relevant_output}"
        elif output_text.strip() and not cleaned_output_text.strip().startswith("Found 0 prompts that satisfy the conditions"):
             return f"未找到'Possible prompt variations:'头部标记。原始输出（已清理）如下：\n\n{cleaned_output_text.strip()}"
        return "未解析到任何提示变体，或输出为空。"

    formatted_string = "发现的提示变体：\n\n"
    for var in variations:
        formatted_string += f"--- 变体 {var['number']} ---\n"
        formatted_string += f"专家画像:\n{var['expert_profile']}\n\n"
        formatted_string += f"提示内容:\n{var['prompt']}\n"
        formatted_string += "-----------------------------------\n\n"
    return formatted_string.strip()

def run_scenario1_enhanced(
    task_description="你是一个数学专家，你将得到一个数学问题，你需要解决它",
    base_instruction="让我们一步一步地思考。",
    answer_format="对于每个问题，先进行推理，然后给出正确答案。",
    experiment_name="no_data_no_examples",
    model_id="Pro/deepseek-ai/DeepSeek-V3",
    mutation_rounds=2,
    few_shot_count=5,
    generate_expert_identity=True,
    generate_intent_keywords=False,
    generate_reasoning=True,
    mutate_refine_iterations=3,
    min_correct_count=3,
    max_eval_batches=6,
    num_train_examples=20,
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
    执行场景1：没有训练数据，不使用样例。
    该函数通过捕获和解析PromptWizard核心库的输出，来展示所有生成的提示变体。

    参数说明:
        task_description (str): 任务的核心描述，告知模型其角色和基本任务。例如："你是一个Python编程助手。"
        base_instruction (str): 提供给模型的基础指令或思考引导。例如："让我们一步一步地思考。"
        answer_format (str): 期望模型输出答案时遵循的格式规范。例如："请先分析问题，然后给出代码和解释。"
        experiment_name (str): 本次实验的名称，用于结果保存和区分不同运行。
        model_id (str): 使用的大语言模型的标识符，例如 "Pro/deepseek-ai/DeepSeek-V3"。
        mutation_rounds (int): 提示词进行变异操作的轮数。更多轮数可能产生更多样化的提示。
        few_shot_count (int): (此场景不直接使用) 构建提示时用作少样本示例的数量。
        generate_expert_identity (bool): 是否让模型为任务生成一个“专家身份”描述。
        generate_intent_keywords (bool): (此场景不直接使用) 是否尝试从任务描述中提取意图关键词。
        generate_reasoning (bool): 是否在生成的提示中包含引导模型进行推理的指令。
        mutate_refine_iterations (int): 变异和精炼过程的迭代次数。
        min_correct_count (int): (此场景不直接使用评估) 评估时认为提示有效的最小正确样例数。
        max_eval_batches (int): (此场景不直接使用评估) 最大评估批次数。
        num_train_examples (int): (此场景不直接使用) 理论上的训练样例数。
        prompt_technique_name (str): 使用的核心提示工程技术名称，例如 "critique_n_refine"。
        questions_batch_size (int): (此场景不直接使用) 处理问题时的批次大小。
        refine_instruction (bool): 是否对基础指令进行精炼优化。
        refine_task_eg_iterations (int): (此场景不直接使用) 对任务样例进行精炼的迭代次数。
        seen_set_size (int): 用于避免重复生成相似提示的已见集合的大小。
        style_variation (int): 生成提示时风格变化的程度或数量。
        top_n (int): (此场景不直接使用选择) 从候选项中选择前N个最佳结果。
        max_history_length (int): 保持的优化历史记录的最大长度。
    
    返回:
        str: 一个包含所有生成提示变体的格式化字符串。
    """
    print("执行场景1：没有训练数据，不使用样例 (V3 with docstrings)")
    load_dotenv(override=True)
    
    promptopt_config = {
        "task_description": task_description, "base_instruction": base_instruction,
        "answer_format": answer_format, "mutation_rounds": int(mutation_rounds),
        "few_shot_count": int(few_shot_count), "generate_expert_identity": bool(generate_expert_identity),
        "generate_intent_keywords": bool(generate_intent_keywords), "generate_reasoning": bool(generate_reasoning),
        "mutate_refine_iterations": int(mutate_refine_iterations), "min_correct_count": int(min_correct_count),
        "max_eval_batches": int(max_eval_batches), "num_train_examples": int(num_train_examples),
        "prompt_technique_name": prompt_technique_name, "questions_batch_size": int(questions_batch_size),
        "refine_instruction": bool(refine_instruction), "refine_task_eg_iterations": int(refine_task_eg_iterations),
        "seen_set_size": int(seen_set_size), "style_variation": int(style_variation),
        "top_n": int(top_n), "unique_model_id": model_id, "max_history_length": int(max_history_length),
    }
    setup_config = {
        "experiment_name": experiment_name, "experiment_id": f"scenario1_{experiment_name}", "model_id": model_id
    }
    
    print("初始化PromptOpt (场景1)...")
    gp = GluePromptOpt(prompt_config=promptopt_config, setup_config=setup_config, dataset_jsonl=None, data_processor=None)
    
    print("开始优化提示 (场景1，准备捕获输出)...")
    all_variations_formatted = "解析过程中出现意外错误。"
    output_text_for_debug = "未能捕获到任何输出。"

    try:
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()
        returned_prompt, returned_profile = gp.get_best_prompt(
            use_examples=False, run_without_train_examples=True, generate_synthetic_examples=False
        )
        sys.stdout = old_stdout
        output_text = captured_output.getvalue()
        output_text_for_debug = output_text

        print(f"\n--- DEBUG (scenario1_ui_enhanced.py): gp.get_best_prompt returned: prompt='{returned_prompt}', profile='{returned_profile}' ---")
        print(f"--- DEBUG (scenario1_ui_enhanced.py): 开始解析捕获到的输出 (长度: {len(output_text)}) ---")
        if not output_text.strip():
            print("--- DEBUG (scenario1_ui_enhanced.py): 捕获到的输出为空或仅包含空白。---")
            all_variations_formatted = "未能从PromptWizard核心库捕获到任何输出文本。"
        else:
            all_variations_formatted = parse_variations_from_output(output_text)
            print(f"--- DEBUG (scenario1_ui_enhanced.py): parse_variations_from_output 返回 (长度: {len(all_variations_formatted)}): ---\n{all_variations_formatted[:300]}...\n--- END PARSED PREVIEW ---")
    except Exception as e:
        sys.stdout = old_stdout
        print(f"--- ERROR (scenario1_ui_enhanced.py): 在捕获或解析过程中发生错误: {e} ---")
        all_variations_formatted = f"处理场景1输出时发生错误: {e}\n捕获到的部分输出 (可能不完整):\n{output_text_for_debug[:500]}..."
        import traceback
        traceback.print_exc()

    print(f"\n=== 最终将返回给Gradio的解析结果 (场景1) (长度: {len(all_variations_formatted)}) ===")
    print(f"{all_variations_formatted[:500]}{'...' if len(all_variations_formatted) > 500 else ''}")
    return all_variations_formatted

if __name__ == "__main__":
    result = run_scenario1_enhanced()
    print("\n--- run_scenario1_enhanced 返回结果 ---")
    print(result)