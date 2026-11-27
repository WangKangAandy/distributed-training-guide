"""数据集加载、分词、切块。与第 1 章 data.py 同构；假定 tokenizer / 原始数据已在本地就绪。"""

from itertools import chain
import multiprocessing

import datasets
from transformers import AutoTokenizer


def load_and_preprocess_data(args, config, *, num_proc=None):
    """
    Function created using code found in
    https://github.com/huggingface/transformers/blob/v4.45.1/examples/pytorch/language-modeling/run_clm_no_trainer.py

    local_files_only：不从网络下载缺失文件。
    load_from_cache_file：若 tokenization / grouping 缓存已存在则复用；
    缓存不存在时，仍会在本次启动中执行 map（不是「数据已完全预处理好」）。
    num_proc：datasets.map 的并行进程数；多卡时由 train_llm 按 world_size 均分传入。
    """
    # 未指定时退回「占满本机 CPU」；DDP 下应由调用方传入更小的值
    if num_proc is None:
        num_proc = max(1, multiprocessing.cpu_count())

    # 分词器：文本 → token id，需与模型词表匹配；local_files_only 禁止启动时下载
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, local_files_only=True
    )

    # 从本地缓存加载数据集；subset 是子集名（有的数据集需要，如 glue/mrpc）
    data = datasets.load_dataset(
        args.dataset_name,
        args.dataset_subset,
        download_config=datasets.DownloadConfig(local_files_only=True),
    )

    # 找出文本所在的列名（常见叫 "text"，否则用第一列）
    column_names = data["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    # 对一批样本文本做分词（batched=True 时 examples 是列式 dict）
    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    # map：对整份数据集并行分词；去掉原始文本列，只保留 token 相关字段
    tokenized_datasets = data.map(
        tokenize_function,
        batched=True,  # 按批调用，比逐条快
        remove_columns=column_names,  # 去掉原文列，减小缓存体积
        num_proc=num_proc,  # 并行进程数（多卡时勿再用满核）
        load_from_cache_file=True,  # 有缓存则直接读，避免重复分词
        desc="Running tokenizer on dataset",
    )

    # 训练序列长度：命令行指定，或退回 tokenizer 默认最大长度
    seq_length = args.seq_length or tokenizer.model_max_length
    # 不能超过模型位置编码上限；过大时压到 min(1024, max_position_embeddings)
    if seq_length > config.max_position_embeddings:
        seq_length = min(1024, config.max_position_embeddings)

    # 拼接全部文本，再按 seq_length 切成固定长度块（因果语言建模常用 packing）
    def group_texts(examples):
        # 把本批里各字段的列表首尾相接成一条长序列
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # 总长度超过 seq_length 时，截去不足整块的尾部；正长度不足一块时，当前实现仍保留短块
        # 若模型支持 padding，也可改成填充而不是丢弃
        if total_length > seq_length:
            total_length = (total_length // seq_length) * seq_length
        # 按 seq_length 切成块
        result = {
            k: [t[i : i + seq_length] for i in range(0, total_length, seq_length)]
            for k, t in concatenated_examples.items()
        }
        # Causal LM 的标签通常就是 input_ids 本身（预测下一个 token）
        result["labels"] = result["input_ids"].copy()
        return result

    # 再 map 一次：把 token 流切成固定长度块，供 DataLoader 直接用
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=num_proc,
        load_from_cache_file=True,  # grouping 结果也可缓存
        desc=f"Grouping texts in chunks of {seq_length}",
    )

    # 本脚本只用训练集（验证集可按需另取 lm_datasets["validation"]）
    return lm_datasets["train"]
