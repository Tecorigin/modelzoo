import dataclasses
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
import torch_sdaa
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    T5Tokenizer,
    BartTokenizer,
    HfArgumentParser,
    DataCollator,
    TrainingArguments,
    set_seed,
)

from trainer import Trainer
from data_collator import T2TDataCollator
from utils import freeze_embeds, assert_not_all_frozen

# 自定义 Dataset 类
from torch.utils.data import Dataset

class QGDataset(Dataset):
    def __init__(self, hf_dataset):
        self.dataset = hf_dataset
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        example = self.dataset[idx]
        # 确保返回正确的格式
        source_ids = example['source_ids']
        target_ids = example['target_ids']
        attention_mask = example['attention_mask']
        
        # 如果不是 tensor 类型，转换为 tensor
        if not isinstance(source_ids, torch.Tensor):
            source_ids = torch.tensor(source_ids)
        if not isinstance(target_ids, torch.Tensor):
            target_ids = torch.tensor(target_ids)
        if not isinstance(attention_mask, torch.Tensor):
            attention_mask = torch.tensor(attention_mask)
            
        # 返回元组格式，确保不会被过滤
        return (source_ids, target_ids, attention_mask)

MODEL_TYPE_TO_TOKENIZER = {
    "t5": T5Tokenizer,
    "bart": BartTokenizer,
}

os.environ['MASTER_ADDR'] = 'localhost'
local_rank = int(os.environ.get("LOCAL_RANK", -1))

# local_rank = int(os.environ['LOCAL_RANK'])
# master_addr = os.environ['MASTER_ADDR']
# master_port = os.environ['MASTER_PORT']
# torch.sdaa.set_device(local_rank)
# dist.init_process_group('tccl', init_method='env://') 
# rank = dist.get_rank()
# print(f"rank = {rank} is initialized in {master_addr}:{master_port}; local_rank = {local_rank}")
# torch.sdaa.set_device(rank)
# tensor = torch.tensor([1, 2]).sdaa()
# print(tensor)

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    model_type: str = field(metadata={"help": "One of 't5', 'bart'"})
    tokenizer_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    label_smoothing: Optional[float] = field(
        default=0,
        metadata={"help": "label smoothing rate, set to > 0 if you want to enable lable smoothing"}
    )
    freeze_embeds: bool = field(
        default=False,
        metadata={"help": "Freeze token embeddings and positional embeddings for bart, just token embeddings for t5."}
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    train_file_path: str = field(
        metadata={"help": "Path for cached train dataset"},
    )
    valid_file_path: str = field(
        metadata={"help": "Path for cached valid dataset"},
    )
    data_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path for data files"}, 
    )
    task: Optional[str] = field(
        default=None,
        metadata={"help": "Which task 'qa', 'qg', 'e2e_qg', 'ans_ext', 'multi'. 'multi' means 'qa', 'qg', 'ans_ext' tasks"}, 
    )
    qg_format: Optional[str] = field(
        default='prepend_qg_format',
        metadata={"help": "How to format inputs for que generation, 'highlight_qg_format' or 'prepend_qg_format'"}, 
    )
    max_source_length: Optional[int] = field(
        default=512,
        metadata={"help": "Max input length for the source text"},
    )
    max_target_length: Optional[int] = field(
        default=32,
        metadata={"help": "Max input length for the target text"},
    )


def main(args_file=None):
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    # device = torch.device(f"sdaa:{local_rank}")
    # torch.sdaa.set_device(device)
    # torch.distributed.init_process_group(backend="tccl", init_method="env://")

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    if (len(sys.argv) == 2 and sys.argv[1].endswith(".json")) or args_file is not None:
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        args_file_path = os.path.abspath(sys.argv[1]) if args_file is None else args_file
        model_args, data_args, training_args = parser.parse_json_file(json_file=args_file_path)
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    assert model_args.model_type in list(MODEL_TYPE_TO_TOKENIZER.keys()), "model type should be 't5' or 'bart'"

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    # Set project name
    os.environ["WANDB_PROJECT"] = "question-generation"

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    tokenizer_cls = MODEL_TYPE_TO_TOKENIZER[model_args.model_type]
    tokenizer = tokenizer_cls.from_pretrained(
        model_args.tokenizer_name_or_path if model_args.tokenizer_name_or_path else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        "t5-small",
        cache_dir=model_args.cache_dir,
    )

    # ddp_model = DDP(model)

    model.resize_token_embeddings(len(tokenizer))

    if model_args.freeze_embeds:
        logger.info("freezing embeddings of the model")
        freeze_embeds(model)
        assert_not_all_frozen(model)

    # Get datasets
    logger.info('loading dataset')
    
    train_dataset = torch.load(data_args.train_file_path) if training_args.do_train else None
    valid_dataset = torch.load(data_args.valid_file_path) if training_args.do_eval else None

    # 包装 Dataset
    if train_dataset is not None:
        train_dataset = QGDataset(train_dataset)
        
    if valid_dataset is not None:
        valid_dataset = QGDataset(valid_dataset)

    logger.info('finished loading dataset')

    # Initialize data_collator
    data_collator = T2TDataCollator(
        tokenizer=tokenizer,
        model_type=model_args.model_type,
        mode="training",
        using_tpu=training_args.tpu_num_cores is not None
    )

    # print('train dataset :', train_dataset)
    # print('data collator :', data_collator)
    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        # prediction_loss_only=True,
        label_smoothing=model_args.label_smoothing
    )

    # disable wandb console logs
    logging.getLogger('wandb.run_manager').setLevel(logging.WARNING)

    # Training
    if training_args.do_train:
        trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    results = {}
    if training_args.do_eval and training_args.local_rank in [-1, 0]:
        logger.info("*** Evaluate ***")

        eval_output = trainer.evaluate()

        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(eval_output.keys()):
                logger.info("  %s = %s", key, str(eval_output[key]))
                writer.write("%s = %s\n" % (key, str(eval_output[key])))
    
        results.update(eval_output)
    
    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()

def run_qg(args_dict):
    with open("args.json", 'w') as f:
        json.dump(args_dict, f)
    
    main(args_file="args.json")

if __name__ == "__main__":
    main()