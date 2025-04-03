#!/usr/bin/env python
import os
import torch
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
import logging
import sys

# 设置日志记录，打印带有时间戳的消息（INFO级及以上）
logging.basicConfig(
    level=logging.INFO,  # 如需更详细日志，请设置为 DEBUG 级别
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def log_gpu_status():
    """记录当前 GPU 状态"""
    if torch.cuda.is_available():
        logging.info("GPU 可用：%s", torch.cuda.get_device_name(0))
        logging.info("当前 GPU 已分配内存：%s 字节", torch.cuda.memory_allocated(0))
        logging.info("当前 GPU 缓存内存：%s 字节", torch.cuda.memory_reserved(0))
    else:
        logging.info("GPU 不可用；使用 CPU。")

def main():
    logging.info("开始运行训练脚本。")
    logging.info("Python 版本：%s", sys.version.replace("\n", " "))
    logging.info("PyTorch 版本：%s", torch.__version__)
    logging.info("Transformers 版本：%s", transformers.__version__)
    
    # 设置 GPU 内存分配的环境变量（如果适用）
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # 在加载数据前记录 GPU 状态
    log_gpu_status()

    # === 2. 数据加载：读取 dataset/jssp_3m3j.json 文件 ===
    data_file = "dataset/jssp_3m3j.json"
    logging.info("从 %s 加载数据集", data_file)
    dataset = load_dataset("json", data_files=data_file)
    logging.info("数据集加载完成。总样本数：%d", len(dataset["train"]))
    logging.info("第一个样本：%s", dataset["train"][0])
    
    # === 3. 划分训练集和测试集 ===
    logging.info("将数据集划分为训练集（80%%）和评估集（20%%）。")
    split_dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    logging.info("训练集大小：%d", len(train_dataset))
    logging.info("评估集大小：%d", len(eval_dataset))
    
    # === 4. 加载模型和 Tokenizer ===
    model_name = "microsoft/Phi-3-mini-4k-instruct"
    logging.info("加载模型和 Tokenizer：%s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 尝试先在 GPU 上加载模型；如果内存不足，则切换到 CPU
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        logging.info("模型已在 GPU 上加载。")
    except RuntimeError as e:
        if "out of memory" in str(e):
            logging.warning("加载模型时 GPU 内存不足。切换到 CPU。")
            torch.cuda.empty_cache()
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="cpu"
            )
        else:
            logging.error("加载模型时出错：%s", e)
            raise e
    model.train()  # 设置模型为训练模式
    logging.info("模型和 Tokenizer 加载成功！")
    
    # === 5. 数据预处理：分词 ===
    def tokenize_function(example):
        # 拼接 input 和 output 字段
        text = example["input"] + example["output"]
        return tokenizer(text, truncation=True, max_length=512)
    
    logging.info("对训练集和评估集进行分词...")
    train_dataset = train_dataset.map(tokenize_function, batched=False)
    eval_dataset = eval_dataset.map(tokenize_function, batched=False)
    logging.info("分词完成。训练集示例：%s", train_dataset[0])
    
    # === 6. 构造 Data Collator ===
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    logging.info("Data Collator 构造成功。")
    # 如有必要，确保模型参数为 float 类型
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.float()
    
    # === 7. 配置训练参数并初始化 Trainer ===
    training_args = TrainingArguments(
        output_dir="./finetuned_model",
        per_device_train_batch_size=1,      # 根据硬件配置调整 batch size
        num_train_epochs=3,                   # 训练轮数
        logging_steps=10,                     # 每 10 步打印日志
        eval_steps=50,                        # 每 50 步进行评估
        evaluation_strategy="steps",          # 按步数进行评估
        save_steps=100,                       # 每 100 步保存检查点
        fp16=True,                            # 如果支持则启用混合精度
        save_total_limit=2,                   # 限制最多保存 2 个检查点
    )
    logging.info("训练参数：%s", training_args)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    logging.info("Trainer 初始化成功。")
    torch.cuda.empty_cache()
    
    # === 8. 开始训练并监控进度 ===
    logging.info("开始训练...")
    log_gpu_status()
    
    try:
        train_result = trainer.train()
    except RuntimeError as e:
        if "out of memory" in str(e):
            logging.warning("训练过程中遇到 GPU 内存错误。记录 GPU 状态并切换到 CPU。")
            log_gpu_status()
            torch.cuda.empty_cache()
            model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cpu")
            model.train()
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
            )
            train_result = trainer.train()
        else:
            logging.error("训练错误：%s", e, exc_info=True)
            raise e

    logging.info("训练完成！")
    logging.info("最终 GPU 状态：")
    log_gpu_status()

    trainer.save_model()  # 保存微调后的模型
    logging.info("微调模型已保存到 ./finetuned_model")

if __name__ == "__main__":
    main()
