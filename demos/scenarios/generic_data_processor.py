#!/usr/bin/env python
# generic_data_processor.py

import json
from typing import List, Dict, Any, Optional

class GenericDataProcessor:
    """通用数据处理器类，可处理不同类型的任务数据"""
    
    def __init__(self):
        """初始化处理器"""
        self.task_type = None
        self.input_key = "question"
        self.output_key = "answer"
    
    def configure_for_task(self, task_type: str):
        """配置处理器以处理特定任务类型
        
        Args:
            task_type: 任务类型 ('math', 'qa', 'summarization', 等)
        
        Returns:
            self: 返回处理器实例，以便链式调用
        """
        self.task_type = task_type
        
        # 根据任务类型配置处理器
        if task_type == 'math':
            self.input_key = "question"
            self.output_key = "answer"
        elif task_type == 'qa':
            self.input_key = "question"
            self.output_key = "answer"
        elif task_type == 'summarization':
            self.input_key = "text"
            self.output_key = "summary"
        # 可添加更多任务类型
        
        return self
    
    def preprocess_input(self, input_data: str) -> str:
        """前处理输入数据
        
        Args:
            input_data: 原始输入数据
        
        Returns:
            str: 处理后的输入数据
        """
        # 根据任务类型执行不同的预处理
        if self.task_type == 'math':
            # 数学问题预处理
            return input_data.strip()
        elif self.task_type == 'qa':
            # QA问题预处理
            return input_data.strip()
        else:
            return input_data.strip()
    
    def postprocess_output(self, output_data: str) -> str:
        """后处理输出数据
        
        Args:
            output_data: 模型生成的原始输出
        
        Returns:
            str: 处理后的输出数据
        """
        # 根据任务类型执行不同的后处理
        if self.task_type == 'math':
            # 提取数学问题的最终答案
            # 通常在GSM8k格式中，答案格式为 "#### <answer>"
            lines = output_data.strip().split("\n")
            for line in lines:
                if line.startswith("####"):
                    return line.strip()
            return output_data  # 如果没有标准格式，返回原始输出
        else:
            return output_data.strip()
    
    def dataset_to_jsonl(self, output_path: str, dataset: List[Dict[str, Any]]):
        """将数据集转换为JSONL格式并保存
        
        Args:
            output_path: 输出文件路径
            dataset: 数据集，列表中的每个元素都是一个字典 
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in dataset:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    def read_jsonl_dataset(self, file_path: str) -> List[Dict[str, Any]]:
        """读取JSONL格式的数据集
        
        Args:
            file_path: JSONL文件路径
        
        Returns:
            List: 数据集列表
        """
        dataset = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    dataset.append(json.loads(line))
        return dataset
    
    def extract_examples(self, dataset: List[Dict[str, Any]], 
                        num_examples: int = 5) -> List[Dict[str, Any]]:
        """从数据集中提取指定数量的样例
        
        Args:
            dataset: 数据集
            num_examples: 要提取的样例数量
        
        Returns:
            List: 提取的样例列表
        """
        return dataset[:min(num_examples, len(dataset))]
    
    def get_input_output_keys(self):
        """获取当前配置的输入和输出字段名
        
        Returns:
            tuple: (input_key, output_key)
        """
        return self.input_key, self.output_key