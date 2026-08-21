import os
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from datasets import load_dataset, Split, Dataset
from transformers import T5Tokenizer, BartTokenizer, HfArgumentParser


logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    task: str = field(
        metadata={"help": "Which task 'qa', 'qg', 'e2e_qg', 'ans_ext', 'multi'. 'multi' means 'qa', 'qg', 'ans_ext' tasks"}, 
    )
    model_type: str = field(metadata={"help": "One of 't5', 'bart'"})
    dataset_path: Optional[str] = field(
        default="data/squad_multitask",
        metadata={"help": "Path for dataset directory"}, 
    )
    train_file_name: Optional[str] = field(
        default=None,
        metadata={"help": "name for cached train dataset"},
    )
    valid_file_name: Optional[str] = field(
        default=None,
        metadata={"help": "name for cached valid dataset"},
    )
    valid_for_qg_only: bool = field(
        default=False,
        metadata={"help": "For multitask dataset valid split should contain only qg task or all tasks."}
    )
    qg_format: Optional[str] = field(
        default='highlight_qg_format',
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
    download_squad: bool = field(
        default=False,
        metadata={"help": "Download SQuAD dataset if local dataset not found"},
    )

class DataProcessor:
    def __init__(self, tokenizer, model_type="t5", max_source_length=512, max_target_length=32):
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.model_type = model_type
        self.hl_token = "<hl>"
        
        if model_type == "t5":
            self.sep_token = "<sep>"
        elif model_type == "bart":
            self.sep_token = "<sep>"
        else:
            self.sep_token = "[SEP]"
  
    def process(self, dataset):
        if self.model_type == "t5":
            dataset = dataset.map(self._add_eos_examples)
        
        dataset = dataset.map(self._add_special_tokens)
        dataset = dataset.map(self._convert_to_features, batched=True)
        
        return dataset
  
    def _add_eos_examples(self, example):
        example['source_text'] = example['source_text'] + " </s>"
        example['target_text'] = example['target_text'] + " </s>"
        return example
  
    def _add_special_tokens(self, example):
        example['source_text'] = example['source_text'].replace("{hl_token}", self.hl_token)    
        example['target_text'] = example['target_text'].replace("{sep_token}", self.sep_token)
        return example
  
    # tokenize the examples
    def _convert_to_features(self, example_batch):
        source_encoding = self.tokenizer.batch_encode_plus(
            example_batch['source_text'],
            max_length=self.max_source_length,
            padding='max_length',
            pad_to_max_length=True,
            truncation=True, 
        )
        target_encoding = self.tokenizer.batch_encode_plus(
            example_batch['target_text'],
            max_length=self.max_target_length,
            padding='max_length',
            pad_to_max_length=True,
            truncation=True, 
        )

        encodings = {
            'source_ids': source_encoding['input_ids'], 
            'target_ids': target_encoding['input_ids'],
            'attention_mask': source_encoding['attention_mask'],
        }

        return encodings


def filter_qa(example):
    return example['task'] == 'qa'

def filter_qg(example):
    return example['task'] == 'qg'

def filter_e2e_qg(example):
    return example['task'] == 'e2e_qg'

def filter_ans_ext(example):
    return example['task'] == 'ans_ext'

def filter_multi(example):
    return example['task'] != 'e2e_qg'


TASK_TO_FILTER_FN = {
    'qa': filter_qa,
    'qg': filter_qg,
    'e2e_qg': filter_e2e_qg,
    'ans_ext': filter_ans_ext,
    'multi': filter_multi
}


def create_sample_dataset():
    sample_data = {
        'source_text': [
            "Python is a high-level programming language. {hl_token}",
            "Machine learning is a subset of artificial intelligence. {hl_token}"
        ],
        'target_text': [
            "What is Python? {sep_token}",
            "What is machine learning? {sep_token}"
        ],
        'task': ['qg', 'qg']
    }
    return Dataset.from_dict(sample_data)


def main():
    parser = HfArgumentParser((DataTrainingArguments,))

    data_args = parser.parse_args_into_dataclasses()[0]

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

    if data_args.model_type == 't5':
        tokenizer = T5Tokenizer.from_pretrained("t5-base", legacy=False)
    else:
        tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    
    tokenizer.add_tokens(['<sep>', '<hl>'])
    
    dataset_exists = os.path.exists(data_args.dataset_path) and (
        os.path.isdir(data_args.dataset_path) or os.path.isfile(data_args.dataset_path)
    )
    
    if not dataset_exists and not data_args.download_squad:
        logger.warning(f"Local dataset path does not exist: {data_args.dataset_path}")
        logger.info("Creating a sample dataset for testing...")
        train_dataset = create_sample_dataset()
        valid_dataset = create_sample_dataset()
    elif data_args.download_squad or not dataset_exists:
        logger.info("Downloading SQuAD dataset...")
        try:
            train_dataset = load_dataset("squad", split="train")
            valid_dataset = load_dataset("squad", split="validation")
            
            def convert_squad_format(example):
                return {
                    'source_text': f"{example['context']} {tokenizer.sep_token} {example['question']}",
                    'target_text': example['answers']['text'][0] if example['answers']['text'] else "",
                    'task': 'qg'
                }
            
            train_dataset = train_dataset.map(convert_squad_format)
            valid_dataset = valid_dataset.map(convert_squad_format)
            
            logger.info(f"Downloaded SQuAD train dataset with {len(train_dataset)} examples")
            logger.info(f"Downloaded SQuAD validation dataset with {len(valid_dataset)} examples")
        except Exception as e:
            logger.error(f"Failed to download SQuAD dataset: {str(e)}")
            logger.info("Creating a sample dataset for testing...")
            train_dataset = create_sample_dataset()
            valid_dataset = create_sample_dataset()
    else:
        logger.info(f"Loading dataset from: {data_args.dataset_path}")
        logger.info(f"Dataset format: {data_args.qg_format}")
        
        try:
            if os.path.isdir(data_args.dataset_path):
                train_dataset = load_dataset(
                    data_args.dataset_path, 
                    data_args.qg_format, 
                    split=Split.TRAIN,
                    data_dir=data_args.dataset_path
                )
                valid_dataset = load_dataset(
                    data_args.dataset_path, 
                    data_args.qg_format, 
                    split=Split.VALIDATION,
                    data_dir=data_args.dataset_path
                )
            else:
                train_dataset = load_dataset(
                    data_args.dataset_path, 
                    name=data_args.qg_format, 
                    split=Split.TRAIN
                )
                valid_dataset = load_dataset(
                    data_args.dataset_path, 
                    name=data_args.qg_format, 
                    split=Split.VALIDATION
                )
        except Exception as e:
            logger.error(f"Failed to load dataset: {str(e)}")
            logger.info("Trying to load without format specification...")
            
            try:
                train_dataset = load_dataset(data_args.dataset_path, split=Split.TRAIN)
                valid_dataset = load_dataset(data_args.dataset_path, split=Split.VALIDATION)
            except Exception as e2:
                logger.error(f"Alternative loading also failed: {str(e2)}")
                logger.info("Creating a sample dataset for testing...")
                train_dataset = create_sample_dataset()
                valid_dataset = create_sample_dataset()
    
    logger.info(f"Train dataset type: {type(train_dataset)}")
    logger.info(f"Validation dataset type: {type(valid_dataset)}")
    
    if hasattr(train_dataset, '__len__'):
        logger.info(f"Successfully loaded train dataset with {len(train_dataset)} examples")
    if hasattr(valid_dataset, '__len__'):
        logger.info(f"Successfully loaded validation dataset with {len(valid_dataset)} examples")
    
    processor = DataProcessor(
        tokenizer,
        model_type=data_args.model_type,
        max_source_length=data_args.max_source_length,
        max_target_length=data_args.max_target_length
    )

    if hasattr(train_dataset, '__len__') and len(train_dataset) > 0:
        try:
            logger.info(f"Sample train example keys: {list(train_dataset[0].keys())}")
        except:
            logger.info("Could not access sample example keys")
    
    try:
        train_dataset = train_dataset.filter(TASK_TO_FILTER_FN[data_args.task])
        if data_args.task == 'multi' and data_args.valid_for_qg_only:
            logger.info("processing valid data only for qg task")
            valid_dataset = valid_dataset.filter(filter_qg)
        else:
            valid_dataset = valid_dataset.filter(TASK_TO_FILTER_FN[data_args.task])
    except Exception as e:
        logger.warning(f"Filtering failed: {str(e)}, continuing without filtering")
    
    train_dataset = processor.process(train_dataset)
    valid_dataset = processor.process(valid_dataset)

    columns = ["source_ids", "target_ids", "attention_mask"]
    train_dataset.set_format(type='torch', columns=columns)
    valid_dataset.set_format(type='torch', columns=columns)

    if data_args.train_file_name is None:
        train_file_name = f"train_data_{data_args.task}_{data_args.qg_format}_{data_args.model_type}.pt"
        train_path = os.path.join("data", train_file_name)

        valid_file_name = f"valid_data_{data_args.task}_{data_args.qg_format}_{data_args.model_type}.pt"
        valid_path = os.path.join("data", valid_file_name)
    else:
        train_path = os.path.join("data", data_args.train_file_name)
        valid_path = os.path.join("data", data_args.valid_file_name)
    
    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    
    torch.save(train_dataset, train_path)
    logger.info(f"saved train dataset at {train_path}")
    
    torch.save(valid_dataset, valid_path)
    logger.info(f"saved validation dataset at {valid_path}")
    
    tokenizer_path = f"{data_args.model_type}_qg_tokenizer"
    if not os.path.exists(tokenizer_path):
        os.makedirs(tokenizer_path, exist_ok=True)
    tokenizer.save_pretrained(tokenizer_path)
    logger.info(f"saved tokenizer at {tokenizer_path}")


if __name__ == "__main__":
    main()