# run_electra.py
from transformers import ElectraTokenizerFast, ElectraForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
from argument import parse_args

def main():
    args = parse_args()

    # 加载 SST-2 数据集（HuggingFace 内置）
    dataset = load_dataset("glue", "sst2")
    tokenizer = ElectraTokenizerFast.from_pretrained(args.model_name_or_path)

    def preprocess(example):
        return tokenizer(example['sentence'], truncation=True, padding="max_length", max_length=args.max_seq_length)

    encoded = dataset.map(preprocess, batched=True)

    model = ElectraForSequenceClassification.from_pretrained(args.model_name_or_path)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        num_train_epochs=args.num_train_epochs,
        evaluation_strategy="epoch",
        logging_dir="./logs",
        logging_steps=50
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded["train"],
        eval_dataset=encoded["validation"]
    )

    if args.do_train:
        trainer.train()

    if args.do_eval:
        trainer.evaluate()

if __name__ == "__main__":
    main()
