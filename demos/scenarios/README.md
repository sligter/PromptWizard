# PromptWizard 提示词优化系统

PromptWizard是一个用于大语言模型提示词优化的系统，可以帮助用户自动化优化提示词，提升模型的回答质量。

## 功能特点

- 🧙‍♂️ **三种优化场景**：支持"无数据无样例"、"无数据合成样例"、"有数据有样例"三种场景
- 📊 **自动优化**：自动进行提示词变异、评估和优化
- 💾 **结果保存**：自动保存优化结果到文件
- 🛠️ **配置灵活**：支持多种参数配置和调整
- 🔄 **合成数据**：能够生成合成训练数据用于优化

## 安装依赖

```bash
pip install gradio pyyaml python-dotenv tqdm
```

## 目录结构

```
demos/scenarios/
├── configs/                   # 配置文件目录
│   ├── llm_config.yaml        # LLM配置
│   ├── prompt_library.yaml    # 提示词模板库
│   ├── promptopt_config.yaml  # 提示词优化配置
│   └── setup_config.yaml      # 系统设置配置
├── results/                   # 结果保存目录
├── gradio_ui.py               # Gradio UI界面
├── generic_data_processor.py  # 通用数据处理器
├── scenario1_no_data_no_examples.py       # 场景1实现
├── scenario2_no_data_synthetic_examples.py # 场景2实现
├── scenario3_with_data_with_examples.py    # 场景3实现
└── README.md                  # 本文件
```

## 使用方法

1. 确保安装了所有依赖
2. 设置环境变量（如`OPENAI_API_KEY`）或在UI中填入API密钥
3. 运行Gradio UI

```bash
cd demos/scenarios
python gradio_ui.py
```

4. 打开浏览器访问 `http://127.0.0.1:7860`

## 三种场景介绍

### 场景1：没有训练数据，不使用样例

适用于没有任何训练数据，也不希望生成合成样例的情况。系统将直接优化提示词。

使用步骤：
1. 输入任务描述和基础指令
2. 选择模型和参数
3. 点击"运行场景1"按钮

### 场景2：没有训练数据，使用合成样例

适用于没有训练数据，但希望使用合成样例的情况。系统将首先生成合成样例，然后基于这些样例优化提示词。

使用步骤：
1. 输入任务描述和基础指令
2. 点击"创建合成数据"按钮
3. 等待合成数据创建完成
4. 点击"运行场景2"按钮

### 场景3：有训练数据，使用样例

适用于已有训练数据，并希望使用这些数据进行提示词优化的情况。

使用步骤：
1. 准备训练数据（JSONL格式）
2. 输入任务描述和基础指令
3. 指定数据文件路径
4. 点击"运行场景3"按钮

## 配置文件说明

系统使用四个主要配置文件：

1. `promptopt_config.yaml` - 提示词优化参数配置
2. `llm_config.yaml` - 语言模型配置
3. `setup_config.yaml` - 系统设置配置
4. `prompt_library.yaml` - 提示词模板库

这些配置可以通过UI界面的"高级配置"标签页进行查看和修改。

## 提示词优化参数说明

- **task_description**: 描述要执行的任务
- **base_instruction**: 提供给模型的基本指令
- **mutation_rounds**: 指定提示词变异的轮数
- **few_shot_count**: 在提示词中使用的示例数量
- **generate_expert_identity**: 是否生成专家身份
- **generate_intent_keywords**: 是否生成意图关键词
- **generate_reasoning**: 是否在提示词中包含推理过程
- **mutate_refine_iterations**: 变异-精炼迭代次数

## 结果输出

每次运行优化后，系统会自动保存结果到`results`目录，包括：

- 专家档案（Expert Profile）
- 最佳提示词（Best Prompt）
- 时间戳和实验名称

## 问题排查

如果遇到问题，请检查：

1. API密钥是否正确设置
2. 依赖是否完整安装
3. 配置文件是否正确
4. 日志输出内容

## 示例数据格式

JSONL格式的训练数据示例：

```json
{"question": "问题1", "answer": "答案1"}
{"question": "问题2", "answer": "答案2"}
```

## 版本信息

PromptWizard v1.0.0 