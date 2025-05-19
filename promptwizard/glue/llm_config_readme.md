# PromptWizard中使用自定义模型提供商和温度参数

本文档介绍如何在PromptWizard中配置和使用自定义模型提供商(BASE_URL)和温度参数。

## 配置文件说明

PromptWizard的配置主要通过以下几个文件实现：

1. **llm_config.yaml** - 定义LLM模型配置和自定义模型配置
2. **.env** - 环境变量配置，包括API密钥、BASE_URL和温度参数等
3. **setup_config.yaml** - 设置使用的模型ID和实验配置
4. **promptopt_config.yaml** - Prompt优化相关配置

## 步骤1：配置 llm_config.yaml

创建或编辑 `llm_config.yaml`，参考以下模板：

```yaml
# Azure OpenAI配置
azure_open_ai:
  api_key: "" # 可以留空并使用环境变量
  azure_endpoint: "" # 可以留空并使用环境变量
  api_version: "2023-05-15"
  use_azure_ad: false
  azure_oai_models:
    - unique_model_id: "gpt-4o" # 唯一标识符，在setup_config.yaml中引用
      model_type: "chat" # 模型类型: chat, completion, embeddings, multimodal
      model_name_in_azure: "gpt-4o" # 模型名称
      deployment_name_in_azure: "your-deployment-name" # 部署名称
      track_tokens: true # 是否追踪token使用量
      req_per_min: 100 # 每分钟请求限制
      tokens_per_min: 100000 # 每分钟token限制
      error_backoff_in_seconds: 30 # 错误后重试间隔时间(秒)

# 自定义模型配置（使用自己的模型服务）
custom_models:
  - unique_model_id: "custom-model" # 唯一标识符
    model_type: "chat" # 模型类型
    path_to_py_file: "path/to/your/custom_llm.py" # 自定义LLM实现文件路径
    class_name: "CustomLLM" # 自定义LLM类名
    track_tokens: true
    req_per_min: 60
    tokens_per_min: 60000
    error_backoff_in_seconds: 30

# 用户和调度器限制
user_limits:
  max_wait_time_in_secs: 300
  timeout_waiting_for_other_reqs_in_secs: 60
scheduler_limits:
  check_for_cancellation_in_secs: 1
```

## 步骤2：创建环境变量文件 .env

将以下内容复制到 `.env` 文件中，并根据需要修改：

```
# 模型类型设置
# 选项: AzureOpenAI, OpenAI, CustomModel
MODEL_TYPE=OpenAI

# === OpenAI API 配置 ===
USE_OPENAI_API_KEY=True
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL_NAME=gpt-4o
# 自定义API基础URL
OPENAI_BASE_URL=https://your-custom-api-url.com/v1
OPENAI_API_VERSION=2023-05-15
# 温度参数设置
OPENAI_TEMPERATURE=0.7

# === Azure OpenAI 配置 ===
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT_NAME=your-deployment-name
AZURE_OPENAI_API_KEY=your_azure_api_key
# 温度参数设置
AZURE_OPENAI_TEMPERATURE=0.5

# === 自定义模型配置 ===
CUSTOM_BASE_URL=http://localhost:8000
CUSTOM_API_KEY=your_custom_api_key
CUSTOM_TEMPERATURE=0.8
```

## 步骤3：创建自定义LLM实现（如果使用自定义模型）

如果需要使用自定义模型，需要创建自定义的LLM实现类。创建一个Python文件（如`custom_llm.py`），参考以下模板：

```python
from typing import Any, List, Optional
from llama_index.core.llms.types import CompletionResponse, ChatMessage, ChatResponse
from llama_index.core.llms.llm import LLM
import os
import requests

class CustomLLM(LLM):
    def __init__(
        self,
        base_url: Optional[str] = None,
        temperature: Optional[float] = None,
        api_key: Optional[str] = None,
        callback_manager: Optional[Any] = None,
    ) -> None:
        self.base_url = base_url or os.environ.get("CUSTOM_BASE_URL")
        self.temperature = temperature or float(os.environ.get("CUSTOM_TEMPERATURE", 0.0))
        self.api_key = api_key or os.environ.get("CUSTOM_API_KEY")
        
        super().__init__(callback_manager=callback_manager)
    
    @classmethod
    def get_tokenizer(cls):
        return None
    
    def chat(self, messages: List[ChatMessage], **kwargs: Any) -> ChatResponse:
        # 从kwargs中获取温度参数，如果没有则使用默认值
        temperature = kwargs.get("temperature", self.temperature)
        
        # 将ChatMessage转换为API期望的格式
        formatted_messages = []
        for message in messages:
            formatted_messages.append({
                "role": message.role,
                "content": message.content,
            })
        
        # 构建请求
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "messages": formatted_messages,
            "temperature": temperature,
            # 其他所需参数
        }
        
        # 发送请求到自定义API
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data
            )
            response.raise_for_status()
            result = response.json()
            
            # 从响应中提取文本
            message_content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            return ChatResponse(
                message=ChatMessage(
                    role="assistant", 
                    content=message_content
                )
            )
        
        except Exception as e:
            print(f"Error calling custom LLM API: {e}")
            return ChatResponse(
                message=ChatMessage(
                    role="assistant", 
                    content="Error: Unable to get chat response from API"
                )
            )
```

## 步骤4：修改 llm_mgr.py 以支持温度参数

需要修改 `promptwizard/glue/common/llm/llm_mgr.py` 中的 `call_api` 函数，使其支持从环境变量获取温度参数：

```python
def call_api(messages, temperature=None):
    from openai import OpenAI
    from azure.identity import get_bearer_token_provider, AzureCliCredential
    from openai import AzureOpenAI

    # 如果没有提供温度参数，则从环境变量获取
    if temperature is None:
        if os.environ['USE_OPENAI_API_KEY'] == "True":
            temperature = float(os.environ.get("OPENAI_TEMPERATURE", 0.0))
        else:
            temperature = float(os.environ.get("AZURE_OPENAI_TEMPERATURE", 0.0))

    if os.environ['USE_OPENAI_API_KEY'] == "True":
        client = OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=os.environ.get("OPENAI_BASE_URL")  # 支持自定义BASE_URL
        )

        response = client.chat.completions.create(
            model=os.environ["OPENAI_MODEL_NAME"],
            messages=messages,
            temperature=temperature,  # 使用温度参数
        )
    else:
        token_provider = get_bearer_token_provider(
                AzureCliCredential(), "https://cognitiveservices.azure.com/.default"
            )
        client = AzureOpenAI(
            api_version=os.environ["OPENAI_API_VERSION"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            azure_ad_token_provider=token_provider
            )
        response = client.chat.completions.create(
            model=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
            messages=messages,
            temperature=temperature,  # 使用温度参数
        )

    prediction = response.choices[0].message.content
    return prediction
```

## 步骤5：更新 chat_completion 方法

同时更新 `LLMMgr` 类的 `chat_completion` 方法，使其可以接受温度参数：

```python
@staticmethod
def chat_completion(messages: Dict, temperature=None):
    llm_handle = os.environ.get("MODEL_TYPE", "AzureOpenAI")
    try:
        if(llm_handle == "AzureOpenAI"): 
            # 传递温度参数
            return call_api(messages, temperature)
        elif(llm_handle == "CustomModel"):
            # 使用自定义模型
            # 这里需要实现自定义模型的调用逻辑
            return custom_api_call(messages, temperature)
        else:
            # 默认调用方式
            return call_api(messages, temperature)
    except Exception as e:
        print(e)
        return "Sorry, I am not able to understand your query. Please try again."
```

## 使用示例

在Python代码中调用LLM时，可以指定温度参数：

```python
from promptwizard.glue.common.llm.llm_mgr import LLMMgr

# 准备消息
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Tell me a story."}
]

# 调用API时指定温度参数
response = LLMMgr.chat_completion(messages, temperature=0.8)
print(response)
```

## 注意事项

1. 自定义BASE_URL主要适用于调用OpenAI API兼容的其他模型服务，如本地部署的模型服务器。
2. 温度参数(temperature)控制生成文本的随机性，值越高结果越随机，值越低结果越确定。
3. 在使用自定义模型时，需要确保您的模型API与标准接口兼容，或相应修改自定义LLM类。
4. 请确保所有敏感信息（API密钥等）都存放在`.env`文件中，切勿直接写入代码或配置文件。 