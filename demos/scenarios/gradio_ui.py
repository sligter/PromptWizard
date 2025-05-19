#!/usr/bin/env python
# gradio_ui.py

import sys
import os
import time
import datetime
sys.path.insert(0, "../../")  # 调整为正确的项目根路径
import gradio as gr
import yaml
import json
from dotenv import load_dotenv

# 导入三个场景
from scenario1_no_data_no_examples import run_scenario1
from scenario2_no_data_synthetic_examples import run_scenario2 
from scenario3_with_data_with_examples import run_scenario3

# 加载配置文件
def load_yaml_config(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    except Exception as e:
        print(f"加载配置文件出错: {e}")
        return {}

# 将字典转换为YAML格式字符串
def dict_to_yaml_str(config_dict):
    if not config_dict:
        return ""
    try:
        return yaml.dump(config_dict, allow_unicode=True)
    except Exception as e:
        print(f"转换YAML出错: {e}")
        return ""

def save_yaml_config(file_path, config_dict):
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            yaml.dump(config_dict, file, allow_unicode=True)
        return True, f"配置已保存到 {file_path}"
    except Exception as e:
        return False, f"保存配置失败: {e}"

def load_configs():
    promptopt_config = load_yaml_config("configs/promptopt_config.yaml")
    llm_config = load_yaml_config("configs/llm_config.yaml")
    setup_config = load_yaml_config("configs/setup_config.yaml")
    return promptopt_config, llm_config, setup_config

# 结果保存函数
def save_result(best_prompt, expert_profile, experiment_name="experiment"):
    """保存优化结果"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = "results"
    
    # 如果目录不存在，创建它
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # 文件名格式
    filename = f"{results_dir}/{experiment_name}_{timestamp}.json"
    
    # 保存结果
    result_data = {
        "timestamp": timestamp,
        "experiment_name": experiment_name,
        "best_prompt": best_prompt,
        "expert_profile": expert_profile
    }
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)
        return f"结果已保存到 {filename}"
    except Exception as e:
        return f"保存结果失败: {e}"

# 日志更新函数
def update_log(log_box, message):
    """更新日志显示"""
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    return log_box + f"\n[{timestamp}] {message}"

# 创建合成数据
def create_synthetic_data(log_output):
    """创建合成训练数据"""
    log_output = update_log(log_output, "开始创建合成数据...")
    
    try:
        # 这里可以调用创建合成数据的函数
        # 假设使用场景2的部分功能
        log_output = update_log(log_output, "正在调用合成数据生成函数...")
        # TODO: 实现合成数据生成功能
        
        time.sleep(1)  # 模拟处理时间
        log_output = update_log(log_output, "合成数据创建完成")
        return log_output, "train_synthetic.jsonl"
    except Exception as e:
        log_output = update_log(log_output, f"创建合成数据时出错: {e}")
        return log_output, None

# 运行场景1函数
def run_scenario1_ui(task_description, base_instruction, model_id, mutation_rounds, use_expert_identity, log_output):
    # 记录开始
    log_output = update_log(log_output, f"开始运行场景1: 没有训练数据，不使用样例")
    log_output = update_log(log_output, f"使用模型: {model_id}")
    
    try:
        # 设置配置
        os.environ["OPENAI_API_KEY"] = api_key
        
        # 定义配置参数
        promptopt_config = {
            "task_description": task_description,
            "base_instruction": base_instruction,
            "mutation_rounds": int(mutation_rounds),
            "max_history_length": 10,
            "use_examples_in_context": False,
            "generate_expert_identity": use_expert_identity,
            "prompt_templates_file": "configs/prompt_library.yaml"
        }
        
        setup_config = {
            "experiment_name": "no_data_no_examples",
            "experiment_id": f"scenario1_{int(time.time())}",
            "model_id": model_id
        }
        
        # 记录配置
        log_output = update_log(log_output, f"配置已设置，开始优化提示...")
        
        # 执行场景1
        best_prompt, expert_profile = run_scenario1()
        
        # 保存结果
        save_msg = save_result(best_prompt, expert_profile, setup_config["experiment_name"])
        log_output = update_log(log_output, save_msg)
        
        return f"### 专家档案\n{expert_profile}\n\n### 最佳提示\n{best_prompt}", log_output
    except Exception as e:
        error_msg = f"场景1执行出错: {e}"
        log_output = update_log(log_output, error_msg)
        return error_msg, log_output

# 运行场景2函数
def run_scenario2_ui(task_description, base_instruction, model_id, mutation_rounds, num_examples, log_output):
    # 记录开始
    log_output = update_log(log_output, f"开始运行场景2: 没有训练数据，使用合成样例")
    log_output = update_log(log_output, f"使用模型: {model_id}, 生成{num_examples}个样例")
    
    try:
        # 设置配置
        os.environ["OPENAI_API_KEY"] = api_key
        
        # 执行场景2
        best_prompt, expert_profile = run_scenario2()
        
        # 保存结果
        save_msg = save_result(best_prompt, expert_profile, "no_data_synthetic_examples")
        log_output = update_log(log_output, save_msg)
        
        return f"### 专家档案\n{expert_profile}\n\n### 最佳提示\n{best_prompt}", log_output
    except Exception as e:
        error_msg = f"场景2执行出错: {e}"
        log_output = update_log(log_output, error_msg)
        return error_msg, log_output

# 运行场景3函数
def run_scenario3_ui(task_description, base_instruction, model_id, dataset_path, few_shot_count, log_output):
    # 记录开始
    log_output = update_log(log_output, f"开始运行场景3: 有训练数据，使用样例")
    log_output = update_log(log_output, f"使用模型: {model_id}, 数据集: {dataset_path}")
    
    try:
        # 设置配置
        os.environ["OPENAI_API_KEY"] = api_key
        
        # 执行场景3
        best_prompt, expert_profile = run_scenario3()
        
        # 保存结果
        save_msg = save_result(best_prompt, expert_profile, "with_data_with_examples")
        log_output = update_log(log_output, save_msg)
        
        return f"### 专家档案\n{expert_profile}\n\n### 最佳提示\n{best_prompt}", log_output
    except Exception as e:
        error_msg = f"场景3执行出错: {e}"
        log_output = update_log(log_output, error_msg)
        return error_msg, log_output

# 保存API密钥
api_key = ""
def save_api_key(key):
    global api_key
    api_key = key
    return "API密钥已保存"

# 保存配置文件
def save_promptopt_config(config_content):
    try:
        # 解析YAML内容
        config_dict = yaml.safe_load(config_content)
        
        # 保存到文件
        success, message = save_yaml_config("configs/promptopt_config.yaml", config_dict)
        
        return message
    except Exception as e:
        return f"保存PromptOpt配置文件失败: {e}"

def save_llm_config(config_content):
    try:
        # 解析YAML内容
        config_dict = yaml.safe_load(config_content)
        
        # 保存到文件
        success, message = save_yaml_config("configs/llm_config.yaml", config_dict)
        
        return message
    except Exception as e:
        return f"保存LLM配置文件失败: {e}"

def save_prompt_library(config_content):
    try:
        # 解析YAML内容
        config_dict = yaml.safe_load(config_content)
        
        # 保存到文件
        success, message = save_yaml_config("configs/prompt_library.yaml", config_dict)
        
        return message
    except Exception as e:
        return f"保存提示词模板文件失败: {e}"

# 创建Gradio界面
with gr.Blocks(title="PromptWizard优化器", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🧙‍♂️ PromptWizard提示词优化系统")
    gr.Markdown("提示词优化系统可以帮助优化大语言模型的提示词，提升模型的表现和输出质量。")
    
    # API密钥设置
    with gr.Accordion("🔑 API设置", open=True):
        api_key_input = gr.Textbox(label="OpenAI API密钥", type="password")
        save_key_btn = gr.Button("保存API密钥")
        key_status = gr.Textbox(label="状态")
        save_key_btn.click(save_api_key, inputs=api_key_input, outputs=key_status)
    
    # 日志输出
    log_output = gr.Textbox(label="运行日志", lines=10, max_lines=30, interactive=False)
    
    # 场景选择标签页
    with gr.Tabs():
        # 场景1：没有训练数据，不使用样例
        with gr.Tab("场景1: 没有训练数据，不使用样例"):
            with gr.Row():
                with gr.Column(scale=2):
                    task_desc1 = gr.Textbox(
                        label="任务描述", 
                        value="You are a mathematics expert. You will be given a mathematics problem which you need to solve",
                        lines=3
                    )
                    base_instr1 = gr.Textbox(
                        label="基础指令", 
                        value="Lets think step by step.",
                        lines=2
                    )
                    
                    with gr.Row():
                        model_id1 = gr.Dropdown(
                            label="模型ID", 
                            choices=["gpt-4o", "gpt-3.5-turbo", "Pro/deepseek-ai/DeepSeek-V3"], 
                            value="gpt-4o"
                        )
                        mutation_rounds1 = gr.Slider(
                            label="变异轮次", 
                            minimum=1, 
                            maximum=10, 
                            value=5, 
                            step=1
                        )
                    
                    expert_identity1 = gr.Checkbox(
                        label="生成专家身份", 
                        value=True
                    )
                    
                    run_btn1 = gr.Button("运行场景1", variant="primary")
                
                with gr.Column(scale=3):
                    result1 = gr.Markdown(label="结果")
            
            run_btn1.click(
                run_scenario1_ui, 
                inputs=[task_desc1, base_instr1, model_id1, mutation_rounds1, expert_identity1, log_output], 
                outputs=[result1, log_output]
            )
        
        # 场景2：没有训练数据，使用合成样例
        with gr.Tab("场景2: 没有训练数据，使用合成样例"):
            with gr.Row():
                with gr.Column(scale=2):
                    task_desc2 = gr.Textbox(
                        label="任务描述", 
                        value="You are a mathematics expert. You will be given a mathematics problem which you need to solve",
                        lines=3
                    )
                    base_instr2 = gr.Textbox(
                        label="基础指令", 
                        value="Lets think step by step.",
                        lines=2
                    )
                    
                    with gr.Row():
                        model_id2 = gr.Dropdown(
                            label="模型ID", 
                            choices=["gpt-4o", "gpt-3.5-turbo", "Pro/deepseek-ai/DeepSeek-V3"], 
                            value="gpt-4o"
                        )
                        mutation_rounds2 = gr.Slider(
                            label="变异轮次", 
                            minimum=1, 
                            maximum=10, 
                            value=2, 
                            step=1
                        )
                    
                    num_examples2 = gr.Slider(
                        label="生成样例数量", 
                        minimum=5, 
                        maximum=30, 
                        value=20, 
                        step=5
                    )
                    
                    create_data_btn = gr.Button("创建合成数据")
                    synthetic_data_path = gr.Textbox(label="合成数据路径", value="")
                    
                    run_btn2 = gr.Button("运行场景2", variant="primary")
                
                with gr.Column(scale=3):
                    result2 = gr.Markdown(label="结果")
            
            create_data_btn.click(
                create_synthetic_data,
                inputs=[log_output],
                outputs=[log_output, synthetic_data_path]
            )
            
            run_btn2.click(
                run_scenario2_ui, 
                inputs=[task_desc2, base_instr2, model_id2, mutation_rounds2, num_examples2, log_output], 
                outputs=[result2, log_output]
            )
        
        # 场景3：有训练数据，使用样例
        with gr.Tab("场景3: 有训练数据，使用样例"):
            with gr.Row():
                with gr.Column(scale=2):
                    task_desc3 = gr.Textbox(
                        label="任务描述", 
                        value="You are a mathematics expert. You will be given a mathematics problem which you need to solve",
                        lines=3
                    )
                    base_instr3 = gr.Textbox(
                        label="基础指令", 
                        value="Lets think step by step.",
                        lines=2
                    )
                    
                    with gr.Row():
                        model_id3 = gr.Dropdown(
                            label="模型ID", 
                            choices=["gpt-4o", "gpt-3.5-turbo", "Pro/deepseek-ai/DeepSeek-V3"], 
                            value="Pro/deepseek-ai/DeepSeek-V3"
                        )
                        few_shot_count3 = gr.Slider(
                            label="少样本数量", 
                            minimum=1, 
                            maximum=10, 
                            value=5, 
                            step=1
                        )
                    
                    dataset_path3 = gr.Textbox(
                        label="训练数据路径", 
                        value="train_synthetic.jsonl"
                    )
                    
                    run_btn3 = gr.Button("运行场景3", variant="primary")
                
                with gr.Column(scale=3):
                    result3 = gr.Markdown(label="结果")
            
            run_btn3.click(
                run_scenario3_ui, 
                inputs=[task_desc3, base_instr3, model_id3, dataset_path3, few_shot_count3, log_output], 
                outputs=[result3, log_output]
            )
    
    # 高级配置标签页
    with gr.Tabs():
        with gr.Tab("PromptOpt配置"):
            # 加载配置并转换为YAML字符串
            promptopt_config_yaml_dict = load_yaml_config("configs/promptopt_config.yaml")
            promptopt_config_yaml_value = dict_to_yaml_str(promptopt_config_yaml_dict)
            if not promptopt_config_yaml_value:
                promptopt_config_yaml_value = """# 高级PromptOpt配置
task_description: "You are a mathematics expert. You will be given a mathematics problem which you need to solve"
base_instruction: "Lets think step by step."
answer_format: "For each question present the reasoning followed by the correct answer."
mutation_rounds: 2
few_shot_count: 5
generate_expert_identity: true
generate_intent_keywords: false
generate_reasoning: true
mutate_refine_iterations: 3
min_correct_count: 3
max_eval_batches: 6
num_train_examples: 20
prompt_technique_name: "critique_n_refine"
questions_batch_size: 1
refine_instruction: true
refine_task_eg_iterations: 3
seen_set_size: 20
style_variation: 5
top_n: 1
unique_model_id: "Pro/deepseek-ai/DeepSeek-V3"
max_history_length: 10"""
            
            promptopt_config_yaml = gr.Code(
                label="PromptOpt配置", 
                language="yaml", 
                value=promptopt_config_yaml_value
            )
            save_promptopt_btn = gr.Button("保存PromptOpt配置")
            promptopt_save_status = gr.Textbox(label="保存状态")
            
            save_promptopt_btn.click(
                save_promptopt_config,
                inputs=promptopt_config_yaml,
                outputs=promptopt_save_status
            )
            
        with gr.Tab("LLM配置"):
            # 加载配置并转换为YAML字符串
            llm_config_yaml_dict = load_yaml_config("configs/llm_config.yaml")
            llm_config_yaml_value = dict_to_yaml_str(llm_config_yaml_dict)
            if not llm_config_yaml_value:
                llm_config_yaml_value = """# LLM配置
model_id: "gpt-4o"
temperature: 0.7
max_tokens: 1000"""
            
            llm_config_yaml = gr.Code(
                label="LLM配置", 
                language="yaml",
                value=llm_config_yaml_value
            )
            save_llm_btn = gr.Button("保存LLM配置")
            llm_save_status = gr.Textbox(label="保存状态")
            
            save_llm_btn.click(
                save_llm_config,
                inputs=llm_config_yaml,
                outputs=llm_save_status
            )
            
        with gr.Tab("提示词模板"):
            # 加载配置并转换为YAML字符串
            prompt_library_yaml_dict = load_yaml_config("configs/prompt_library.yaml")
            prompt_library_yaml_value = dict_to_yaml_str(prompt_library_yaml_dict)
            if not prompt_library_yaml_value:
                prompt_library_yaml_value = """# 提示词模板库
system_prompts: |
  You are a helpful assistant that assists research students in understanding research papers.
system_guidelines: |
  Guidelines 
  - Your role must always be a helpful assistant that assists students in understanding research papers.
  - Only answer questions that are directly or indirectly related to the referenced paper(s)."""
            
            prompt_library_yaml = gr.Code(
                label="提示词模板库", 
                language="yaml",
                value=prompt_library_yaml_value
            )
            save_prompt_library_btn = gr.Button("保存提示词模板")
            prompt_library_save_status = gr.Textbox(label="保存状态")
            
            save_prompt_library_btn.click(
                save_prompt_library,
                inputs=prompt_library_yaml,
                outputs=prompt_library_save_status
            )
    
    # 帮助和说明
    with gr.Accordion("📖 使用说明", open=False):
        gr.Markdown("""
        ## PromptWizard 使用说明
        
        ### 场景1：没有训练数据，不使用样例
        此场景适用于您没有任何训练数据，也不希望生成合成样例的情况。系统将直接优化提示词。
        
        ### 场景2：没有训练数据，使用合成样例
        此场景适用于您没有训练数据，但希望使用合成样例的情况。系统将首先生成合成样例，然后基于这些样例优化提示词。
        
        ### 场景3：有训练数据，使用样例
        此场景适用于您已有训练数据，并希望使用这些数据进行提示词优化的情况。
        
        ### 参数说明
        - **任务描述**：描述要执行的任务
        - **基础指令**：提供给模型的基本指令
        - **变异轮次**：指定提示词变异的轮数
        - **生成专家身份**：是否生成专家身份
        - **少样本数量**：在提示词中使用的示例数量
        
        ### 功能按钮
        - **创建合成数据**：在场景2中，用于生成合成训练数据
        - **运行场景X**：执行相应场景的提示词优化
        - **保存XX配置**：保存对应的配置文件
        """)

    # 运行日志和版本信息
    gr.Markdown("### 🌟 PromptWizard v1.0.0 | 提示词优化系统")

if __name__ == "__main__":
    # 创建results目录
    if not os.path.exists("results"):
        os.makedirs("results")
    
    # 加载环境变量
    load_dotenv(override=True)
    # 启动Gradio界面
    demo.launch(server_port=7860) 