from typing import Any, Dict, List, Optional
import os
from llama_index.core.llms.types import CompletionResponse, ChatMessage, ChatResponse, ChatResponseGen
from llama_index.core.llms.llm import LLM

class CustomLLM(LLM):
    """自定义LLM实现，支持BASE_URL和温度参数设置。
    
    这个类可以作为创建自定义LLM的模板。
    继承自llama_index的LLM类，提供了与其他LLM相兼容的接口。
    """
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        temperature: Optional[float] = None,
        api_key: Optional[str] = None,
        callback_manager: Optional[Any] = None,
    ) -> None:
        """初始化自定义LLM。

        Args:
            base_url: API基础URL，默认从环境变量获取
            temperature: 生成温度，值越高随机性越大，默认从环境变量获取
            api_key: API密钥，默认从环境变量获取
            callback_manager: 回调管理器
        """
        self.base_url = base_url or os.environ.get("CUSTOM_BASE_URL")
        self.temperature = temperature or float(os.environ.get("CUSTOM_TEMPERATURE", 0.0))
        self.api_key = api_key or os.environ.get("CUSTOM_API_KEY")
        
        super().__init__(callback_manager=callback_manager)
    
    @classmethod
    def get_tokenizer(cls):
        """返回tokenizer，用于token计数。
        
        返回None代表不支持token计数。
        """
        return None
    
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """完成文本生成任务。
        
        Args:
            prompt: 输入提示文本
            **kwargs: 其他参数
        
        Returns:
            CompletionResponse对象
        """
        # 这里实现调用自定义API的逻辑
        # 示例实现:
        import requests
        
        # 从kwargs中获取温度，如果没有则使用默认值
        temperature = kwargs.get("temperature", self.temperature)
        
        # 构建请求
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "prompt": prompt,
            "temperature": temperature,
            # 其他所需参数
        }
        
        # 发送请求到自定义API
        try:
            response = requests.post(
                f"{self.base_url}/completions",
                headers=headers,
                json=data
            )
            response.raise_for_status()
            result = response.json()
            
            # 从响应中提取文本
            text = result.get("choices", [{}])[0].get("text", "")
            
            return CompletionResponse(text=text)
        
        except Exception as e:
            print(f"Error calling custom LLM API: {e}")
            return CompletionResponse(text="Error: Unable to get completion from API")
    
    def chat(self, messages: List[ChatMessage], **kwargs: Any) -> ChatResponse:
        """聊天接口实现。
        
        Args:
            messages: 聊天消息列表
            **kwargs: 其他参数
            
        Returns:
            ChatResponse对象
        """
        # 从kwargs中获取温度，如果没有则使用默认值
        temperature = kwargs.get("temperature", self.temperature)
        
        # 实现调用聊天API的逻辑
        import requests
        
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
            print(f"Error calling custom LLM chat API: {e}")
            return ChatResponse(
                message=ChatMessage(
                    role="assistant", 
                    content="Error: Unable to get chat response from API"
                )
            )
    
    def stream_chat(self, messages: List[ChatMessage], **kwargs: Any) -> ChatResponseGen:
        """流式聊天接口实现，如果API支持流式返回可以实现此方法。
        
        Args:
            messages: 聊天消息列表
            **kwargs: 其他参数
            
        Returns:
            ChatResponseGen生成器
        """
        # 这里是个简单实现，实际应用中可以实现流式API调用
        response = self.chat(messages, **kwargs)
        
        def gen():
            yield response
        
        return gen() 