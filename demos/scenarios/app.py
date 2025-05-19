#!/usr/bin/env python
# promptwizard_app_cn.py (重要：确保文件名不是 gradio.py)

import gradio as gr
import inspect
import os
import sys
from dotenv import load_dotenv
import re # 确保 re 被导入，get_function_params 中可能用到

# --- 重要 ---
# 确保此脚本与场景脚本和 generic_data_processor.py 在同一目录
# 场景脚本中的 sys.path.insert(0, "../../") 应能正确找到 promptwizard 库

# 导入脚本中的函数
from scenario1_ui_enhanced import run_scenario1_enhanced
from scenario2_ui_enhanced import run_scenario2_enhanced
from scenario3_ui_enhanced import run_scenario3_enhanced
from synthetic_data_generator import create_synthetic_data

# 加载 .env 文件
load_dotenv(override=True)

# --- 辅助函数：提取参数和描述 (微调默认描述) ---
def get_function_params(func):
    sig = inspect.signature(func)
    params_info = {}
    docstring = inspect.getdoc(func)
    param_descriptions = {}

    if docstring:
        lines = docstring.split('\n')
        param_section = False
        param_headers = [
            "parameters:", "params:", "args:", "arguments:", "parameter description:",
            "参数说明:", "参数列表:", "参数:", "形参列表:", "形参:"
        ]
        section_enders = [
            "returns:", "yields:", "raises:", "attributes:",
            "返回:", "返回值:", "产生:", "异常:", "属性:"
        ]
        current_param_name_for_multiline_desc = None
        current_desc_lines = []

        for line_num, raw_line in enumerate(lines):
            stripped_line = raw_line.strip()
            is_header = False
            if not param_section:
                for header_text in param_headers:
                    if stripped_line.lower().startswith(header_text) and (len(stripped_line) > len(header_text) or ':' in stripped_line):
                        param_section = True; is_header = True
                        current_param_name_for_multiline_desc = None; current_desc_lines = []
                        break
            if is_header: continue

            if param_section:
                is_ender = False
                for ender_text in section_enders:
                    if stripped_line.lower().startswith(ender_text):
                        param_section = False; is_ender = True; break
                if is_ender:
                    if current_param_name_for_multiline_desc and current_desc_lines:
                        param_descriptions[current_param_name_for_multiline_desc] = " ".join(current_desc_lines).strip()
                    current_param_name_for_multiline_desc = None; current_desc_lines = []
                    continue
                
                # 使用原始行来匹配，以保留缩进信息判断是否是参数定义行
                # ^\s*([\w_]+)\s*(?:\((.*?)\))?:\s*(.*)
                # 匹配行首的参数名，可选的类型，冒号，然后是描述的开始
                # 我们需要一个更简单的方法：如果行包含冒号，并且冒号前是有效的参数名
                match_param_def = None
                potential_param_part = stripped_line.split(":",1)[0]
                param_name_match = re.match(r"^\s*([\w_]+)", potential_param_part)

                if ":" in stripped_line and param_name_match:
                    param_name_candidate = param_name_match.group(1)
                    if param_name_candidate in sig.parameters:
                         # 这是一个新的参数定义行
                        if current_param_name_for_multiline_desc and current_desc_lines: # 保存上一个
                            param_descriptions[current_param_name_for_multiline_desc] = " ".join(current_desc_lines).strip()
                        
                        current_param_name_for_multiline_desc = param_name_candidate
                        description_start = stripped_line.split(":",1)[1].strip()
                        current_desc_lines = [description_start] if description_start else []
                    elif current_param_name_for_multiline_desc and stripped_line: # 可能是多行描述的一部分
                        current_desc_lines.append(stripped_line)
                    else: # 不是新参数，也不是旧参数的延续
                        current_param_name_for_multiline_desc = None; current_desc_lines = []

                elif current_param_name_for_multiline_desc and stripped_line: 
                    current_desc_lines.append(stripped_line)
                elif not stripped_line and current_param_name_for_multiline_desc: 
                     if current_param_name_for_multiline_desc and current_desc_lines:
                        param_descriptions[current_param_name_for_multiline_desc] = " ".join(current_desc_lines).strip()
                     current_param_name_for_multiline_desc = None; current_desc_lines = []
        
        if current_param_name_for_multiline_desc and current_desc_lines:
            param_descriptions[current_param_name_for_multiline_desc] = " ".join(current_desc_lines).strip()

    for name, param in sig.parameters.items():
        param_type = param.annotation if param.annotation != inspect.Parameter.empty else str
        default_value = param.default if param.default != inspect.Parameter.empty else None
        if isinstance(default_value, bool): param_type = bool
        elif isinstance(default_value, int): param_type = int
        elif isinstance(default_value, float): param_type = float
        
        final_description = param_descriptions.get(name, f"关于参数 '{name}' 的配置。请在脚本的函数文档字符串中为此参数添加详细说明。")
        params_info[name] = {"default": default_value, "type": param_type, "description": final_description}
    return params_info

# --- UI 元素创建函数（中文版） ---
def create_ui_elements_cn(params_info, is_output_path_related=False):
    elements = {}
    param_name_to_cn_label = {
        "task_description": "任务描述", "base_instruction": "基础指令", "answer_format": "答案格式",
        "experiment_name": "实验名称", "model_id": "模型ID", "num_train_examples": "训练样例数量",
        "mutation_rounds": "变异轮次", "few_shot_count": "少样本数量", "generate_expert_identity": "生成专家身份",
        "generate_intent_keywords": "生成意图关键词", "generate_reasoning": "生成推理过程",
        "mutate_refine_iterations": "变异-精炼迭代次数", "min_correct_count": "最小正确数量",
        "max_eval_batches": "最大评估批次数", "prompt_technique_name": "提示技术名称",
        "questions_batch_size": "问题批次大小", "refine_instruction": "精炼指令",
        "refine_task_eg_iterations": "精炼任务示例迭代次数", "seen_set_size": "已见样本集大小",
        "style_variation": "风格变异度", "top_n": "保留Top-N结果", "max_history_length": "最大历史长度",
        "synthetic_data_path": "合成数据路径", "task_type": "任务类型", "num_examples": "生成样例数量",
        "output_path": "输出文件路径", "temperature": "温度参数", "max_tokens": "最大Token数",
        "example_complexity": "样例复杂度", "dataset_path": "数据集路径", "data_task_type": "数据任务类型",
    }
    for name, info in params_info.items():
        label = param_name_to_cn_label.get(name, name.replace("_", " ").title())
        description = info["description"] 
        value = info["default"]
        elem = None
        if name == "model_id":
            elem = gr.Dropdown(choices=["Pro/deepseek-ai/DeepSeek-V3", "Pro/mistralai/Mixtral-8x22B-Instruct-v0.1", "Pro/google/Gemini-Pro-1.5", "HuggingFaceH4/zephyr-7b-beta"], value=value, label=label, info=description, container=False)
        elif info["type"] == bool: elem = gr.Checkbox(value=bool(value), label=label, info=description, container=False)
        elif info["type"] == int: elem = gr.Number(value=int(value) if value is not None else 0, label=label, precision=0, info=description, container=False)
        elif info["type"] == float: elem = gr.Number(value=float(value) if value is not None else 0.0, label=label, info=description, container=False)
        elif name == "example_complexity": elem = gr.Dropdown(choices=["simple", "medium", "complex"], value=value, label=label, info=description, container=False)
        elif name == "task_type" or name == "data_task_type": elem = gr.Dropdown(choices=["math", "code", "text", "general"], value=value, label=label, info=description, container=False)
        elif name == "prompt_technique_name": elem = gr.Dropdown(choices=["critique_n_refine", "analogical_reasoning", "self_discover"], value=value, label=label, info=description, container=False)
        elif "path" in name or "dataset" in name:
            path_info_cn = " 相对路径或绝对路径。"
            if is_output_path_related and "output_path" in name: elem = gr.Textbox(value=value, label=label, info=description + " 若不存在则会被创建。", container=False)
            else: elem = gr.Textbox(value=value, label=label, info=description + path_info_cn, container=False)
        else: elem = gr.Textbox(value=str(value) if value is not None else "", label=label, info=description, container=False)
        if elem is not None: elements[name] = elem
    return elements

# --- 获取各函数参数 ---
params_data_gen_all = get_function_params(create_synthetic_data)
params_scen1_all = get_function_params(run_scenario1_enhanced)
params_scen2_all = get_function_params(run_scenario2_enhanced)
params_scen3_all = get_function_params(run_scenario3_enhanced)

# --- Gradio 界面 ---
with gr.Blocks(theme=gr.themes.Soft(), title="PromptWizard 可视化工具") as demo:
    gr.Markdown(
        """
        <div style="text-align: center; margin-bottom: 20px;">
            <h1 style="color: #2c3e50; font-size: 2.5em;">🚀 PromptWizard 可视化向导</h1>
            <p style="color: #555; font-size: 1.1em;">一个用于配置、运行提示词优化场景及生成合成数据的交互界面。</p>
        </div>
        """
    )

    with gr.Tabs():
        with gr.TabItem("📊 合成数据生成器", id="tab_data_gen"):
            with gr.Column(variant="panel"): 
                gr.Markdown("### <center>创建合成训练数据</center>\n<p style='text-align:center; color:gray;'>根据任务描述和参数，生成用于模型训练或评估的合成样例。</p>", elem_classes="tab-title")
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### 主要配置")
                        ui_elements_data_gen_main = {}
                        main_params_data_gen = ["task_description", "task_type", "model_id", "num_examples", "example_complexity"]
                        for p_name in main_params_data_gen:
                            if p_name in params_data_gen_all:
                                ui_elements_data_gen_main.update(create_ui_elements_cn({p_name: params_data_gen_all[p_name]}))
                        
                        gr.Markdown("#### 高级配置")
                        ui_elements_data_gen_adv = {}
                        adv_params_data_gen = ["temperature", "max_tokens"]
                        for p_name in adv_params_data_gen:
                             if p_name in params_data_gen_all:
                                ui_elements_data_gen_adv.update(create_ui_elements_cn({p_name: params_data_gen_all[p_name]}))
                        
                        output_path_ui_data_gen = create_ui_elements_cn({'output_path': params_data_gen_all['output_path']}, is_output_path_related=True)['output_path']

                    with gr.Column(scale=1): 
                        gr.Markdown("#### 执行与结果")
                        run_button_data_gen = gr.Button("生成数据", variant="primary", scale=0)
                        with gr.Group(): 
                            output_status_data_gen = gr.Textbox(label="状态 / 输出文件路径", lines=2, interactive=False)
                            output_file_data_gen = gr.File(label="下载生成的数据文件 (.jsonl)")
                
                ordered_input_keys_data_gen = [name for name in params_data_gen_all if name != 'output_path'] + ['output_path']
                all_ui_elements_data_gen = {**ui_elements_data_gen_main, **ui_elements_data_gen_adv, 'output_path': output_path_ui_data_gen}
                ordered_ui_inputs_data_gen = [all_ui_elements_data_gen[key] for key in ordered_input_keys_data_gen]

                def handle_data_generation(*args_from_ui):
                    kwargs = {name: args_from_ui[i] for i, name in enumerate(ordered_input_keys_data_gen)}
                    try:
                        for name, info in params_data_gen_all.items():
                            if name in kwargs:
                                if info['type'] == int: kwargs[name] = int(kwargs[name])
                                elif info['type'] == float: kwargs[name] = float(kwargs[name])
                                elif info['type'] == bool: kwargs[name] = bool(kwargs[name])
                        output_file = create_synthetic_data(**kwargs)
                        if output_file and os.path.exists(output_file): return f"数据生成成功: {output_file}", output_file
                        else: return f"数据生成失败或未找到文件 (预期: {kwargs.get('output_path', 'N/A')})。\n请检查控制台日志获取详细错误信息。", None
                    except Exception as e: return f"数据生成时发生严重错误: {str(e)}\n请检查控制台日志。", None
                run_button_data_gen.click(handle_data_generation, inputs=ordered_ui_inputs_data_gen, outputs=[output_status_data_gen, output_file_data_gen])

        with gr.TabItem("💡 场景1：无数据，无样例", id="tab_scen1"):
            with gr.Column(variant="panel"):
                gr.Markdown("### <center>无训练数据和样例的提示词优化 (显示所有变体)</center>\n<p style='text-align:center; color:gray;'>此场景直接基于任务描述和基础指令生成提示变体，不依赖外部数据。</p>", elem_classes="tab-title")
                with gr.Row():
                    with gr.Column(scale=1): 
                        gr.Markdown("#### 参数配置")
                        basic_params_scen1 = ["task_description", "base_instruction", "answer_format", "model_id"]
                        adv_params_scen1 = [p for p in params_scen1_all.keys() if p not in basic_params_scen1]

                        with gr.Accordion("基础参数", open=True):
                            ui_elements_scen1_basic = create_ui_elements_cn({k: params_scen1_all[k] for k in basic_params_scen1 if k in params_scen1_all})
                        with gr.Accordion("高级参数", open=False):
                            ui_elements_scen1_adv = create_ui_elements_cn({k: params_scen1_all[k] for k in adv_params_scen1 if k in params_scen1_all})
                        
                        all_ui_elements_scen1 = {**ui_elements_scen1_basic, **ui_elements_scen1_adv}
                        ordered_input_keys_scen1 = list(params_scen1_all.keys()) 
                        ordered_ui_inputs_scen1 = [all_ui_elements_scen1[key] for key in ordered_input_keys_scen1]

                    with gr.Column(scale=1): 
                        gr.Markdown("#### 执行与结果")
                        run_button_scen1 = gr.Button("运行场景1", variant="primary", scale=0)
                        with gr.Group(elem_classes="output-box"):
                            output_variations_scen1 = gr.Textbox(label="生成的提示变体", lines=25, interactive=False, show_copy_button=True)
                
                def handle_scenario1(*args_from_ui):
                    kwargs = {name: args_from_ui[i] for i, name in enumerate(ordered_input_keys_scen1)}
                    try:
                        for name, info in params_scen1_all.items():
                            if name in kwargs:
                                if info['type'] == int: kwargs[name] = int(kwargs[name])
                                elif info['type'] == bool: kwargs[name] = bool(kwargs[name])
                                elif info['type'] == float: kwargs[name] = float(kwargs[name])
                        all_variations_str = run_scenario1_enhanced(**kwargs)
                        return all_variations_str
                    except Exception as e: return f"运行场景1时发生错误: {str(e)}\n请检查控制台日志。"
                run_button_scen1.click(handle_scenario1, inputs=ordered_ui_inputs_scen1, outputs=[output_variations_scen1])

        def create_scenario_tab_cn_beautified(tab_id, tab_title_cn, scenario_md_cn, params_dict, run_function, output_labels_cn_map):
            with gr.TabItem(tab_title_cn, id=tab_id):
                with gr.Column(variant="panel"):
                    gr.Markdown(f"### <center>{scenario_md_cn['title']}</center>\n<p style='text-align:center; color:gray;'>{scenario_md_cn['description']}</p>", elem_classes="tab-title")
                    with gr.Row():
                        with gr.Column(scale=1): 
                            gr.Markdown("#### 参数配置")
                            # 为参数提供一个可折叠区域
                            with gr.Accordion("点击展开/折叠所有参数", open=len(params_dict) < 10): # 参数少于10个时默认展开
                                ui_elements = create_ui_elements_cn(params_dict)
                            ordered_ui_inputs = [ui_elements[name] for name in params_dict.keys()]

                        with gr.Column(scale=1): 
                            gr.Markdown("#### 执行与结果")
                            run_button_label = f"运行{tab_title_cn.split('：')[0]}"
                            run_button = gr.Button(run_button_label, variant="primary", scale=0)
                            
                            with gr.Group(elem_classes="output-box"): 
                                outputs_ui = []
                                for key, label_text in output_labels_cn_map.items():
                                    lines = 10 if "提示" in label_text or "变体" in label_text else 5
                                    if "路径" in label_text: lines = 2
                                    outputs_ui.append(gr.Textbox(label=label_text, lines=lines, interactive=False, show_copy_button=True))
                    
                    def generic_handler_flexible_outputs(*args_from_ui):
                        kwargs = {name: args_from_ui[i] for i, name in enumerate(list(params_dict.keys()))}
                        try:
                            for name, info in params_dict.items():
                                if name in kwargs:
                                    if info['type'] == int: kwargs[name] = int(kwargs[name])
                                    elif info['type'] == bool: kwargs[name] = bool(kwargs[name])
                                    elif info['type'] == float: kwargs[name] = float(kwargs[name])
                            results = run_function(**kwargs)
                            if not isinstance(results, tuple): results = (results,)
                            num_expected_outputs = len(outputs_ui)
                            if len(results) == num_expected_outputs: return results
                            elif len(results) < num_expected_outputs:
                                processed_results = list(results)
                                while len(processed_results) < num_expected_outputs:
                                    processed_results.append(f"{run_function.__name__} 未返回足够输出 (预期{num_expected_outputs}, 得到{len(results)})")
                                return tuple(processed_results)
                            else: return results[:num_expected_outputs]
                        except Exception as e:
                            error_msg = f"运行 {tab_title_cn} 时发生错误: {str(e)}\n请检查控制台日志。"
                            return tuple([error_msg] * len(outputs_ui))
                    run_button.click(generic_handler_flexible_outputs, inputs=ordered_ui_inputs, outputs=outputs_ui)
        
        create_scenario_tab_cn_beautified(
            tab_id="tab_scen2",
            tab_title_cn="🤖 场景2：无数据，合成样例",
            scenario_md_cn={
                "title": "使用合成样例进行提示词优化",
                "description": "此场景首先生成合成数据 (若指定路径文件不存在或脚本决定重新生成), 然后用其优化提示词。"
            },
            params_dict=params_scen2_all,
            run_function=run_scenario2_enhanced,
            output_labels_cn_map={"best_prompt": "最佳提示", "expert_profile": "专家画像", "synthetic_data_path": "使用/生成的合成数据路径"}
        )

        with gr.TabItem("💾 场景3：有数据，有样例", id="tab_scen3"):
            with gr.Column(variant="panel"):
                gr.Markdown("### <center>使用现有训练数据和样例进行提示词优化</center>\n<p style='text-align:center; color:gray;'>利用您提供的数据集来优化提示词，适用于已有标注数据的场景。</p>", elem_classes="tab-title")
                with gr.Row():
                    with gr.Column(scale=1): 
                        gr.Markdown("#### 参数配置")
                        params_scen3_no_dataset = {k: v for k,v in params_scen3_all.items() if k != 'dataset_path'}
                        with gr.Accordion("点击展开/折叠所有参数", open=len(params_scen3_no_dataset) < 10):
                             ui_elements_scen3_part1 = create_ui_elements_cn(params_scen3_no_dataset)
                        
                        gr.Markdown("#### 数据集提供方式")
                        dataset_path_file_scen3 = gr.File(label="上传数据集文件 (.jsonl)", file_types=[".jsonl"], scale=0)
                        dataset_path_text_scen3 = gr.Textbox(
                            value=params_scen3_all['dataset_path']['default'], 
                            label="或 输入数据集本地路径 (.jsonl)", 
                            info=params_scen3_all['dataset_path']['description'] + " 上传文件优先。",
                            container=False
                        )

                        ordered_input_names_scen3_no_dataset = [name for name in params_scen3_all.keys() if name != 'dataset_path']
                        all_inputs_scen3_for_click = [ui_elements_scen3_part1[name] for name in ordered_input_names_scen3_no_dataset]
                        all_inputs_scen3_for_click.extend([dataset_path_file_scen3, dataset_path_text_scen3])

                    with gr.Column(scale=1): 
                        gr.Markdown("#### 执行与结果")
                        run_button_scen3 = gr.Button("运行场景3", variant="primary", scale=0)
                        with gr.Group(elem_classes="output-box"):
                            output_prompt_scen3 = gr.Textbox(label="最佳提示", lines=15, interactive=False, show_copy_button=True)
                            output_profile_scen3 = gr.Textbox(label="专家画像", lines=8, interactive=False, show_copy_button=True)
                
                def handle_scenario3(*args_from_ui):
                    kwargs = {name: args_from_ui[i] for i, name in enumerate(ordered_input_names_scen3_no_dataset)}
                    uploaded_file_obj = args_from_ui[len(ordered_input_names_scen3_no_dataset)]
                    text_path = args_from_ui[len(ordered_input_names_scen3_no_dataset) + 1]
                    dataset_actual_path = text_path
                    if uploaded_file_obj is not None:
                        dataset_actual_path = uploaded_file_obj.name
                        gr.Info(f"正在使用上传的文件: {dataset_actual_path}")
                    kwargs['dataset_path'] = dataset_actual_path
                    if not kwargs['dataset_path'] or not os.path.exists(kwargs['dataset_path']):
                        return (f"错误：数据集路径 '{kwargs['dataset_path']}' 未提供或文件不存在。", "请提供有效的数据集。")
                    try:
                        for name, info in params_scen3_all.items():
                            if name in kwargs and name != 'dataset_path':
                                if info['type'] == int: kwargs[name] = int(kwargs[name])
                                elif info['type'] == bool: kwargs[name] = bool(kwargs[name])
                                elif info['type'] == float: kwargs[name] = float(kwargs[name])
                        best_prompt, expert_profile = run_scenario3_enhanced(**kwargs)
                        return best_prompt, expert_profile
                    except Exception as e: return f"运行场景3时发生错误: {str(e)}\n请检查控制台日志。", f"错误: {str(e)}\n请检查控制台日志。"
                run_button_scen3.click(handle_scenario3, inputs=all_inputs_scen3_for_click, outputs=[output_prompt_scen3, output_profile_scen3])

if __name__ == "__main__":
    if os.path.basename(__file__).lower() == "gradio.py":
        print("错误：此脚本名为 'gradio.py'。")
        print("请将其重命名 (例如 'promptwizard_app_cn.py') 以避免导入冲突。")
        sys.exit(1)
    demo.launch(share=False)