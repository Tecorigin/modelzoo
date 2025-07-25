import logging
import os
from dataclasses import dataclass, field
from typing import Optional
from transformers.trainer_callback import TrainerCallback
import datasets
from datasets import load_dataset

import transformers
from transformers import (
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

logger = logging.getLogger(__name__)

class IterationLogCallback(TrainerCallback):
    def __init__(self, log_file):
        self.log_file = log_file
        self.log_handle = open(log_file, "w")

    def on_step_end(self, args, state, control, **kwargs):
        current_step = state.global_step
        if state.log_history and "loss" in state.log_history[-1]:
            log_message = f"{current_step}iter - "
            log_message += " - ".join([f"{key}: {value}" for key, value in state.log_history[-1].items()])
            log_message += "\n"
            self.log_handle.write(log_message)
            self.log_handle.flush()
        else:
            log_message = f"{current_step}iter - no loss logged\n"
            self.log_handle.write(log_message)
            self.log_handle.flush()

    def on_train_end(self, args, state, control, **kwargs):
        self.log_handle.close()

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "The model checkpoint for weights initialization."}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Optional config path if not same as model_name."}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Optional tokenizer path if not same as model_name."}
    )

@dataclass
class DataTrainingArguments:
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "Dataset name from the HuggingFace hub."}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "Path to the training data file (txt/json)." }
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "Path to the validation data file (txt/json)." }
    )
    line_by_line: bool = field(
        default=False, metadata={"help": "Whether each line is a distinct sequence."}
    )
    max_seq_length: int = field(
        default=128, metadata={"help": "Maximum input sequence length."}
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Masked LM probability."}
    )

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detect last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)

    set_seed(training_args.seed)

    # Load dataset
    data_files = {}
    if data_args.train_file is not None:
        data_args.train_file = "/data/bigc-data/zjh/new/copy1/transformers1/examples/pytorch/language-modeling/train.txt"
        data_files["train"] = data_args.train_file
    if data_args.validation_file is not None:
        data_files["validation"] = data_args.validation_file

    extension = data_args.train_file.split(".")[-1]
    raw_datasets = load_dataset("text", data_files=data_files)

    # Set local paths
    model_path = "/data/bigc-data/zjh/new/copy1/transformers1/examples/pytorch/language-modeling/model"
    model_args.model_name_or_path = model_path
    model_args.config_name = os.path.join(model_path, "config.json")
    model_args.tokenizer_name = model_path  # 修改这里

    # Load tokenizer and model from local paths
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name,  # 使用正确的文件夹路径
        use_fast=True,
        local_files_only=True  # 禁止联网
    )
    config = AutoConfig.from_pretrained(
        model_args.config_name,  # 使用正确的配置文件路径
        local_files_only=True # 禁止联网
    )
    print("当前使用的模型路径：", model_args.model_name_or_path)
    assert os.path.exists(os.path.join(model_args.model_name_or_path, "pytorch_model.bin")), "模型参数文件未找到"
 
    model = AutoModelForMaskedLM.from_pretrained(
        model_args.model_name_or_path,  # 使用模型文件夹路径
        config=config,
        local_files_only=True  # 禁止联网
    )

    # Preprocessing
    if data_args.line_by_line:
        def tokenize_function(examples):
            return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=data_args.max_seq_length)

        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=1,
            remove_columns=["text"],
            desc="Tokenizing line by line",
        )
    else:
        def tokenize_function(examples):
            return tokenizer(examples["text"])

        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=1,
            remove_columns=["text"],
            desc="Tokenizing",
        )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=data_args.mlm_probability
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[IterationLogCallback(os.path.join(training_args.output_dir, "sdaa.log"))], 
    )

    # Training
    if training_args.do_train:
        checkpoint = last_checkpoint if last_checkpoint is not None else None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

if __name__ == "__main__":
    main()