# run_electra.py
#运行命令 python run_electra.py   --model_name_or_path ../configs/electra-small-generator   --train_file ../configs/train.txt   --do_train --do_eval   --output_dir electra_out   --overwrite_output_dir   --per_device_train_batch_size 2   --max_seq_length 32   --line_by_line

from argument import parse_args
import subprocess

def build_cmd(args):
    cmd = [
        "python", "../configs/run_mlm.py",
        "--model_name_or_path", args.model_name_or_path,
        "--train_file", args.train_file,
        "--validation_file", args.validation_file,
        "--output_dir", args.output_dir,
        "--per_device_train_batch_size", str(args.per_device_train_batch_size),
        "--per_device_eval_batch_size", str(args.per_device_eval_batch_size),
        "--max_seq_length", str(args.max_seq_length),
        "--num_train_epochs", str(args.num_train_epochs),
        "--learning_rate", str(args.learning_rate),
        "--logging_steps", str(args.logging_steps),

    ]

    if args.line_by_line:
        cmd.append("--line_by_line")
    if args.mlm_probability:
        cmd += ["--mlm_probability", str(args.mlm_probability)]
    if args.overwrite_output_dir:
        cmd.append("--overwrite_output_dir")
    if args.do_train:
        cmd.append("--do_train")
    if args.do_eval:
        cmd.append("--do_eval")
    if args.fp16:
        cmd.append("--fp16")

    return cmd

def main():
    args = parse_args()
    cmd = build_cmd(args)

    print("🚀 Running command:")
    print(" ".join(cmd))

    subprocess.run(cmd)

if __name__ == "__main__":
    main()
