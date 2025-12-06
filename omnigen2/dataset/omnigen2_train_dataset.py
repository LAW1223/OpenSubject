from typing import Optional, Union, List

import os
import random
import yaml
import glob
from PIL import Image

import torch
from torchvision import transforms

from datasets import load_dataset, concatenate_datasets

from ..pipelines.omnigen2.pipeline_omnigen2 import OmniGen2ImageProcessor

class OmniGen2TrainDataset(torch.utils.data.Dataset):
    """
    OmniGen2 训练数据集类
    
    支持多种任务类型的多模态数据集：
    - t2i (text-to-image): 文本生成图像任务，只需要instruction和output_image
    - edit (image-editing): 图像编辑任务，需要instruction、input_images和output_image
    - ic (image-conditioning): 图像条件生成任务，需要instruction、input_images和output_image
    
    数据格式示例：
    - t2i: {"task_type": "t2i", "instruction": "A big tree is in the forest", "output_image": "/path/to/image.png"}
    - edit: {"task_type": "edit", "instruction": "add a hat to the person", "input_images": ["/path/to/input.png"], "output_image": "/path/to/output.png"}
    - ic: {"task_type": "ic", "instruction": "A big tree is in the forest", "input_images": ["/path/to/input1.png", "/path/to/input2.png"], "output_image": "/path/to/output.png"}
    """
    
    # 完整的系统提示词，用于指导模型生成高质量图像
    SYSTEM_PROMPT = "You are a helpful assistant that generates high-quality images based on user instructions."
    # 简化的系统提示词，用于prompt dropout时
    SYSTEM_PROMPT_DROP = "You are a helpful assistant that generates images."

    def __init__(
        self,
        config_path: str,                                               # 数据配置文件路径，支持yml/yaml格式
        tokenizer,                                                      # 分词器对象，用于处理文本指令
        use_chat_template: bool,                                        # 是否使用聊天模板格式化输入
        max_input_pixels: Optional[Union[int, List[int]]] = None,       # 输入图像最大像素数，可以是单个值或列表（针对不同数量的输入图像）
        max_output_pixels: Optional[int] = None,                        # 输出图像最大像素数
        max_side_length: Optional[int] = None,                          # 图像最大边长限制
        img_scale_num: int = 16,                                        # 图像缩放因子（VAE缩放因子）
        prompt_dropout_prob: float = 0.0,                              # 指令dropout概率，用于数据增强
        ref_img_dropout_prob: float = 0.0,                             # 参考图像dropout概率，用于数据增强
    ):
        # 保存图像处理相关参数
        self.max_input_pixels = max_input_pixels               # 输入图像最大像素数限制
        self.max_output_pixels = max_output_pixels             # 输出图像最大像素数限制
        self.max_side_length = max_side_length                 # 图像最大边长限制
        self.img_scale_num = img_scale_num                     # VAE缩放因子，通常为16
        
        # 保存数据增强相关参数
        self.prompt_dropout_prob = prompt_dropout_prob         # 指令dropout概率，训练时随机丢弃指令
        self.ref_img_dropout_prob = ref_img_dropout_prob       # 参考图像dropout概率，训练时随机丢弃输入图像

        # 加载数据配置文件（支持嵌套的yml配置）
        with open(config_path, "r") as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

        # 保存文本处理相关参数
        self.use_chat_template = use_chat_template             # 是否使用聊天模板格式
        
        # 初始化图像处理器，用于预处理输入和输出图像
        self.image_processor = OmniGen2ImageProcessor(vae_scale_factor=img_scale_num, do_resize=True)

        # 收集并加载所有数据注释
        data = self._collect_annotations(self.config)

        # 保存数据集和分词器
        self.data = data
        self.tokenizer = tokenizer
        
    def _collect_annotations(self, config):
        """
        收集和处理数据注释的核心方法
        
        支持的配置格式：
        - 直接的jsonl/json文件路径
        - 包含多个数据文件的目录路径
        - 嵌套的yml/yaml配置文件
        
        配置中的ratio字段用于控制各数据源的采样比例
        
        Args:
            config: 配置字典，包含data字段，每个数据源有path、type、ratio等属性
            
        Returns:
            合并后的数据集对象
        """
        total_samples = 0      # 总样本数统计
        total_ratio = 0        # 总比例统计
        json_datasets = []     # 存储所有数据集的列表
        
        # 遍历配置中的每个数据源
        for data in config['data']:
            data_path, data_type = data['path'], data.get("type", "default")
            
            # 处理目录路径：递归查找所有json/jsonl文件
            if os.path.isdir(data_path):
                jsonl_files = list(glob.glob(os.path.join(data_path, "**/*.jsonl"), recursive=True)) + \
                             list(glob.glob(os.path.join(data_path, "**/*.json"), recursive=True))
                json_dataset = load_dataset('json', data_files=jsonl_files, cache_dir=None)['train']
            else:
                # 处理单个文件路径
                data_ext = os.path.splitext(data_path)[-1]
                if data_ext in [".json", ".jsonl"]:
                    # 直接加载json/jsonl数据文件
                    json_dataset = load_dataset('json', data_files=data_path, cache_dir=None)['train']
                elif data_ext in [".yml", ".yaml"]:
                    # 递归加载嵌套的yaml配置文件
                    with open(data_path, "r") as f:
                        sub_config = yaml.load(f, Loader=yaml.FullLoader)
                        json_dataset = self._collect_annotations(sub_config)
                else:
                    # 不支持的文件格式
                    raise NotImplementedError(
                        f'Unknown data file extension: "{data_ext}". '
                        f"Currently, .json, .jsonl .yml .yaml are supported. "
                        "If you are using a supported format, please set the file extension so that the proper parsing "
                        "routine can be called."
                    )
            
            # 累计比例和样本数
            total_ratio += data['ratio']
            total_samples += len(json_dataset)
            json_datasets.append(json_dataset)
        
        # 根据比例重新采样各个数据集
        for json_dataset in json_datasets:
            # 计算每个数据集的目标大小（归一化比例）
            target_size = int(len(json_dataset) * data['ratio'] / total_ratio)
            
            if target_size <= len(json_dataset):
                # 无替换随机采样（下采样）
                indices = random.sample(range(len(json_dataset)), target_size)
            else:
                # 有替换随机采样（上采样）
                indices = random.choices(range(len(json_dataset)), k=target_size)
            json_dataset = json_dataset.select(indices)
            
        # 合并所有数据集
        json_dataset = concatenate_datasets(json_datasets)
        return json_dataset
    
    def clean_data_item(self, data_item):
        """
        清理和预处理数据项
        
        主要功能：
        - 对于文本生成图像任务(t2i)，随机移除指令中的描述性前缀
        - 这有助于提高模型对不同指令格式的鲁棒性
        
        Args:
            data_item: 单个数据项字典
            
        Returns:
            清理后的数据项
        """
        task_type = data_item['task_type']
        
        # 定义常见的描述性前缀（中英文）
        prefixs = ["The image portrays ", "The image depicts ", "The image captures ", 
                  "The image highlights ", "The image shows ", "这张图片展示了"]
        
        # 只对文本生成图像任务进行前缀清理
        if "text_to_image" in task_type or "t2i" in task_type:
            if random.random() < 0.5:  # 50%概率进行清理
                for p in prefixs:
                    if p in data_item['instruction']:
                        data_item['instruction'] = data_item['instruction'].replace(p, "")
                        break  # 只移除第一个匹配的前缀
        return data_item
    
    def apply_chat_template(self, instruction, system_prompt):
        """
        应用聊天模板格式化指令
        
        将用户指令和系统提示词组合成聊天格式，这在多轮对话模型中很常见
        
        Args:
            instruction: 用户指令文本
            system_prompt: 系统提示词
            
        Returns:
            格式化后的指令文本
        """
        if self.use_chat_template:
            # 构建聊天格式的对话
            prompt = [
                {
                    "role": "system",        # 系统角色，定义助手的行为
                    "content": system_prompt,
                },
                {"role": "user", "content": instruction},  # 用户角色，包含具体指令
            ]
            # 使用分词器的聊天模板进行格式化
            instruction = self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=False)
        return instruction
    
    def process_item(self, data_item):
        """
        处理单个数据项的核心方法
        
        功能包括：
        1. 数据清理和预处理
        2. 随机dropout处理（prompt和reference image）
        3. 图像加载和预处理
        4. 文本指令格式化
        
        支持的数据格式：
        - t2i: {"task_type": "t2i", "instruction": "...", "output_image": "..."}
        - edit/ic: {"task_type": "edit/ic", "instruction": "...", "input_images": [...], "output_image": "..."}
        
        Args:
            data_item: 原始数据项字典
            
        Returns:
            处理后的数据字典，包含所有必要字段
        """
        assert data_item['instruction'] is not None
        
        # 1. 清理数据项（移除描述性前缀等）
        data_item = self.clean_data_item(data_item)

        # 2. 随机dropout处理 - 用于数据增强
        drop_prompt = random.random() < self.prompt_dropout_prob      # 是否丢弃提示词
        drop_ref_img = drop_prompt and random.random() < self.ref_img_dropout_prob  # 是否丢弃参考图像

        # 3. 处理文本指令
        if drop_prompt:
            # 使用空指令和简化系统提示词
            instruction = self.apply_chat_template("", self.SYSTEM_PROMPT_DROP)
        else:
            # 使用完整指令和完整系统提示词
            instruction = self.apply_chat_template(data_item['instruction'], self.SYSTEM_PROMPT)

        # 4. 处理输入图像（用于edit和ic任务）
        if not drop_ref_img and 'input_images' in data_item and data_item['input_images'] is not None:
            input_images_path = data_item['input_images']
            input_images = []

            # 根据输入图像数量选择合适的像素限制
            max_input_pixels = self.max_input_pixels[len(input_images_path) - 1] if isinstance(self.max_input_pixels, list) else self.max_input_pixels

            # 加载和预处理每张输入图像
            for input_image_path in input_images_path:
                input_image = Image.open(input_image_path).convert("RGB")
                input_image = self.image_processor.preprocess(input_image, max_pixels=max_input_pixels, max_side_length=self.max_side_length)
                input_images.append(input_image)
        else:
            # 没有输入图像或被dropout
            input_images_path, input_images = None, None

        # 5. 处理输出图像（所有任务都需要）
        output_image_path = data_item['output_image']
        output_image = Image.open(output_image_path).convert("RGB")
        output_image = self.image_processor.preprocess(output_image, max_pixels=self.max_output_pixels, max_side_length=self.max_side_length)

        # 6. 构建最终的数据字典
        data = {
            'task_type': data_item['task_type'],              # 任务类型
            'instruction': instruction,                       # 格式化后的指令
            'input_images_path': input_images_path,           # 输入图像路径列表
            'input_images': input_images,                     # 预处理后的输入图像张量
            'output_image': output_image,                     # 预处理后的输出图像张量
            'output_image_path': output_image_path,           # 输出图像路径
        }
        return data

    def __getitem__(self, index):
        """
        获取指定索引的数据项
        
        实现了重试机制，当某个数据项处理失败时（如图像文件损坏），
        会随机选择其他数据项进行重试，确保训练过程的稳定性
        
        Args:
            index: 数据项索引
            
        Returns:
            处理后的数据字典
            
        Raises:
            Exception: 当重试次数达到上限仍失败时抛出最后一次的异常
        """
        max_retries = 12      # 最大重试次数

        current_index = index
        for attempt in range(max_retries):
            try:
                data_item = self.data[current_index]
                return self.process_item(data_item)
            except Exception as e:
                if attempt == max_retries - 1:
                    # 达到最大重试次数，抛出异常
                    raise e
                else:
                    # 随机选择一个新的索引进行重试
                    current_index = random.randint(0, len(self.data) - 1)
                    continue
        
    def __len__(self):
        """返回数据集的总长度"""
        return len(self.data)

class OmniGen2Collator():
    """
    OmniGen2 数据批处理器
    
    负责将多个数据样本组织成训练批次，主要功能：
    1. 收集批次中所有样本的各个字段
    2. 对文本指令进行批量分词和填充
    3. 组织成模型训练所需的格式
    
    用于DataLoader的collate_fn参数
    """
    
    def __init__(self, tokenizer, max_token_len):
        """
        初始化批处理器
        
        Args:
            tokenizer: 分词器对象，用于处理文本
            max_token_len: 文本序列的最大长度限制
        """
        self.tokenizer = tokenizer           # 分词器
        self.max_token_len = max_token_len   # 最大token长度

    def __call__(self, batch):
        """
        批处理函数，将一个batch的数据样本组织成训练格式
        
        Args:
            batch: 数据样本列表，每个样本是由OmniGen2TrainDataset.__getitem__返回的字典
            
        Returns:
            批处理后的数据字典，包含模型训练所需的所有字段
        """
        # 从批次中提取各个字段
        task_type = [data['task_type'] for data in batch]                    # 任务类型列表
        instruction = [data['instruction'] for data in batch]                # 指令文本列表
        input_images_path = [data['input_images_path'] for data in batch]    # 输入图像路径列表
        input_images = [data['input_images'] for data in batch]              # 输入图像张量列表
        output_image = [data['output_image'] for data in batch]              # 输出图像张量列表
        output_image_path = [data['output_image_path'] for data in batch]    # 输出图像路径列表

        # 批量分词处理指令文本
        text_inputs = self.tokenizer(
            instruction,
            padding="longest",      # 填充到批次中最长的序列
            max_length=self.max_token_len,  # 最大长度限制
            truncation=True,        # 超长截断
            return_tensors="pt",    # 返回PyTorch张量
        )

        # 组织最终的批处理数据
        data = {
            "task_type": task_type,                      # 任务类型列表
            "text_ids": text_inputs.input_ids,           # 分词后的文本ID张量 [batch_size, seq_len]
            "text_mask": text_inputs.attention_mask,     # 注意力掩码张量 [batch_size, seq_len]
            "input_images": input_images,                # 输入图像列表（可能包含None）
            "input_images_path": input_images_path,      # 输入图像路径列表
            "output_image": output_image,                # 输出图像张量列表
            "output_image_path": output_image_path,      # 输出图像路径列表
        }
        return data
