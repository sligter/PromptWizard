from os.path import dirname, join
import pickle
import time
import os
from typing import Any, Dict, Union

from ..common.base_classes import LLMConfig, SetupConfig
from ..common.constants.log_strings import CommonLogsStr
from ..common.llm.llm_mgr import LLMMgr
from ..common.utils.logging import get_glue_logger, set_logging_config
from ..common.utils.file import read_jsonl, yaml_to_class, yaml_to_dict, read_jsonl_row
from ..paramlogger import ParamLogger
from ..promptopt.constants import PromptOptimizationLiterals
from ..promptopt.techniques.common_logic import DatasetSpecificProcessing
from ..promptopt.utils import get_promptopt_class


class GluePromptOpt:
    """
    This class is trigger point for any prompt optimization method. Different prompt optimization techniques are
    represented by different classes. This class collates all the user configs present in different yaml files or
    dictionaries and other boilerplate code. Any of supported prompt optimization techniques can be triggered by this class.
    """
    BEST_PROMPT = None
    EXPERT_PROFILE = None
    data_processor = None
    iolog = ParamLogger()

    class EvalLiterals:
        IS_CORRECT = "is_correct"
        PREDICTED_ANS = "predicted_ans"
        LLM_OUTPUT = "llm_output"

    def __init__(self,
                 prompt_config: Union[str, Dict],
                 setup_config: Union[str, Dict],
                 dataset_jsonl: str = None,
                 data_processor: DatasetSpecificProcessing = None,
                 dataset_processor_pkl_path: str = None,
                 prompt_pool_path: str = None):
        """
        Collates all the configs from files or dictionaries. Initialize logger, de-serialize pickle file that has
        class/method for dataset processing (for given dataset).

        Args:
            prompt_config: Path to yaml file or dictionary that has prompt templates for the given techniques.
            setup_config: Path to yaml file or dictionary that has user preferences.
            dataset_jsonl: Path to jsonl file that has dataset present in jsonl format.
            data_processor: Object of DatasetSpecificProcessing class, which has data handling methods specific to dataset.
            dataset_processor_pkl_path: Path to pickle file that has object of class DatasetSpecificProcessing serialized.
            prompt_pool_path: Path to yaml file that has prompts
        """
        # 处理数据处理器
        if dataset_jsonl is not None:
            if data_processor:
                self.data_processor = data_processor
            elif dataset_processor_pkl_path:
                with open(dataset_processor_pkl_path, "rb") as file:
                    self.data_processor = pickle.load(file)  # datatype: class DatasetSpecificProcessing
                    
        # 处理prompt配置
        if isinstance(prompt_config, dict):
            prompt_config_dict = prompt_config
        else:
            prompt_config_dict = yaml_to_dict(prompt_config)

        # 获取提示优化类
        prompt_opt_cls, prompt_opt_hyperparam_cls, promptpool_cls = get_promptopt_class(
            prompt_config_dict[PromptOptimizationLiterals.PROMPT_TECHNIQUE_NAME])

        # 处理setup配置
        if isinstance(setup_config, dict):
            # 直接将字典转换为SetupConfig对象
            from types import SimpleNamespace
            self.setup_config = SimpleNamespace()
            for key, value in setup_config.items():
                setattr(self.setup_config, key, value)
        else:
            # 从YAML文件加载
            self.setup_config = yaml_to_class(setup_config, SetupConfig)
            
        # 处理prompt参数配置
        if isinstance(prompt_config, dict):
            # 直接将字典转换为prompt_opt_hyperparam_cls对象
            from types import SimpleNamespace
            self.prompt_opt_param = SimpleNamespace()
            for key, value in prompt_config.items():
                setattr(self.prompt_opt_param, key, value)
        else:
            # 从YAML文件加载
            self.prompt_opt_param = yaml_to_class(prompt_config, prompt_opt_hyperparam_cls)
        
        # 处理prompt_pool
        current_dir = dirname(__file__)
        default_yaml_path = join(current_dir,
                               "techniques",
                               prompt_config_dict[PromptOptimizationLiterals.PROMPT_TECHNIQUE_NAME],
                               "prompt_pool.yaml")

        self.prompt_pool = yaml_to_class(prompt_pool_path, promptpool_cls, default_yaml_path)

        # 加载数据集
        if dataset_jsonl is not None:
            dataset = read_jsonl(dataset_jsonl)
        else:
            dataset = None

        # 配置答案格式
        if hasattr(self.prompt_opt_param, "answer_format") and hasattr(self.prompt_pool, "ans_delimiter_instruction"):
            self.prompt_opt_param.answer_format += self.prompt_pool.ans_delimiter_instruction
            
        # 配置日志
        try:
            base_path = join(self.setup_config.dir_info.base_dir, self.setup_config.experiment_name)
            set_logging_config(join(base_path, self.setup_config.dir_info.log_dir_name),
                             self.setup_config.mode)
        except AttributeError:
            base_path = "experiments"
            os.makedirs(base_path, exist_ok=True)
            set_logging_config(base_path)
            
        self.logger = get_glue_logger(__name__)

        # 检查数据集大小和参数设置
        if dataset_jsonl is not None and dataset:
            if hasattr(self.prompt_opt_param, "seen_set_size") and len(dataset) < self.prompt_opt_param.seen_set_size:
                self.prompt_opt_param.seen_set_size = len(dataset)
                self.logger.info(f"Dataset has {len(dataset)} samples. However values for seen_set_size is "
                                f"{self.prompt_opt_param.seen_set_size}. Hence resetting seen_set_size"
                                f" to {len(dataset)}")

            if hasattr(self.prompt_opt_param, "few_shot_count") and hasattr(self.prompt_opt_param, "seen_set_size") and \
                    self.prompt_opt_param.few_shot_count > self.prompt_opt_param.seen_set_size:
                self.prompt_opt_param.few_shot_count = self.prompt_opt_param.seen_set_size
                self.logger.info(f"Value set for few_shot_count is {self.prompt_opt_param.few_shot_count}. "
                               f"However values for seen_set_size is {self.prompt_opt_param.seen_set_size}. "
                               f"Hence resetting few_shot_count to {self.prompt_opt_param.few_shot_count}")

        # 准备训练数据集
        if dataset_jsonl is not None and dataset and hasattr(self.prompt_opt_param, "seen_set_size"):
            training_dataset = dataset[:self.prompt_opt_param.seen_set_size]
        else:
            training_dataset = None
            
        # 记录配置信息
        self.logger.info(f"Setup configurations parameters: {self.setup_config} \n{CommonLogsStr.LOG_SEPERATOR}")
        self.logger.info(f"Prompt Optimization parameters: {self.prompt_opt_param} \n{CommonLogsStr.LOG_SEPERATOR}")

        # 初始化日志
        try:
            self.iolog.reset_eval_glue(join(base_path, "evaluation"))
        except:
            self.iolog.reset_eval_glue("evaluation")

        # 初始化prompt优化器
        self.prompt_opt = prompt_opt_cls(training_dataset, base_path, self.setup_config,
                                       self.prompt_pool, self.data_processor, self.logger)

    # 保持其余方法不变
    def get_best_prompt(self,use_examples=False,run_without_train_examples=False,generate_synthetic_examples=False) -> (str, Any):
        """
        Call get_best_prompt() method of class PromptOptimizer & return its value.
        :return: (best_prompt, expert_profile)
            best_prompt-> Best prompt for a given task description
            expert_profile-> Description of an expert who is apt to solve the task at hand.
        """
        start_time = time.time()
        self.BEST_PROMPT, self.EXPERT_PROFILE = self.prompt_opt.get_best_prompt(self.prompt_opt_param,use_examples=use_examples,run_without_train_examples=run_without_train_examples,generate_synthetic_examples=generate_synthetic_examples)

        self.logger.info(f"Time taken to find best prompt: {(time.time() - start_time)} sec")
        return self.BEST_PROMPT, self.EXPERT_PROFILE

    def evaluate(self, test_dataset_jsonl: str) -> float:
        """
        Evaluate the performance of self.BEST_PROMPT over test dataset. Return the accuracy.

        :param test_dataset_jsonl: Path to jsonl file that has test dataset
        :return: Percentage accuracy
        """
        # 方法内容保持不变
        start_time = time.time()
        self.logger.info(f"Evaluation started {CommonLogsStr.LOG_SEPERATOR}")
        if not self.BEST_PROMPT:
            self.logger.error("BEST_PROMPT attribute is not set. Please set self.BEST_PROMPT attribute of this object, "
                              "either manually or by calling get_best_prompt() method.")
            return

        total_correct = 0
        total_count = 0
        for json_obj in read_jsonl_row(test_dataset_jsonl):
            answer = self.predict_and_access(json_obj[DatasetSpecificProcessing.QUESTION_LITERAL],
                                             json_obj[DatasetSpecificProcessing.FINAL_ANSWER_LITERAL])
      
            total_correct += answer[self.EvalLiterals.IS_CORRECT]
            total_count += 1
            result = {"accuracy": f"{total_correct}/{total_count} : {total_correct/total_count}%",
                      "predicted": answer[self.EvalLiterals.PREDICTED_ANS],
                      "actual": json_obj[DatasetSpecificProcessing.FINAL_ANSWER_LITERAL]}
            self.iolog.append_dict_to_chained_logs(result)
            self.logger.info(result)

        self.iolog.dump_chained_log_to_file(file_name=f"eval_result_{self.setup_config.experiment_name}")
        self.logger.info(f"Time taken for evaluation: {(time.time() - start_time)} sec")
        return total_correct / total_count

    @iolog.log_io_params
    def predict_and_access(self, question: str, gt_answer: str) -> (bool, str, str):
        """
        For the given input question, get answer to it from LLM, using the BEST_PROMPT & EXPERT_PROFILE
        computes earlier.

        :param question: Question to be asked to LLM, to solve
        :param gt_answer: Ground truth, final answer.
        :return:  (is_correct, predicted_ans, llm_output)
        :rtype: (bool, str, str)
        """
        # 方法内容保持不变
        final_prompt = self.prompt_pool.eval_prompt.format(instruction=self.BEST_PROMPT,
                                                           question=question)
        llm_output = self.prompt_opt.chat_completion(user_prompt=final_prompt, system_prompt=self.EXPERT_PROFILE)
        
        is_correct, predicted_ans = self.data_processor.access_answer(llm_output, gt_answer)
        return {self.EvalLiterals.IS_CORRECT: is_correct,
                self.EvalLiterals.PREDICTED_ANS: predicted_ans,
                self.EvalLiterals.LLM_OUTPUT: llm_output}