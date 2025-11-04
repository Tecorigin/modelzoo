from argument import parse_args, check_argument
import os
from pathlib import Path

if __name__ == "__main__":
    args = parse_args()
    nproc_per_node = args.nproc_per_node
    train_file = args.train_file
    eval_file = args.eval_file
    outputs_dir = args.outputs_dir
    scale = args.scale
    lr = args.lr
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    num_workers = args.num_workers
    seed = args.seed
    use_amp = args.use_amp
    use_ddp = args.use_ddp
    if use_ddp:
        local_rank = args.local_rank
    else:
        local_rank = 0

    project_path = str(Path(__file__).resolve().parents[1])

    if nproc_per_node>1:
        cmd = f"python -m torch.distributed.launch \
            --nproc_per_node={nproc_per_node} \
            --master_port=25641 {project_path}/train.py \
            --train_file={train_file} \
            --eval_file={eval_file} \
            --outputs_dir={outputs_dir} \
            --scale={scale} \
            --lr={lr} \
            --batch_size={batch_size} \
            --num_epochs={num_epochs} \
            --num_workers={num_workers} \
            --seed={seed} \
            --use_amp={use_amp} \
            --use_ddp={use_ddp}"
    else:
        cmd = f"python {project_path}/train.py \
            --nproc_per_node={nproc_per_node} \
            --train_file={train_file} \
            --eval_file={eval_file} \
            --outputs_dir={outputs_dir} \
            --scale={scale} \
            --lr={lr} \
            --batch_size={batch_size} \
            --num_epochs={num_epochs} \
            --num_workers={num_workers} \
            --seed={seed} \
            --use_amp={use_amp} \
            --use_ddp={use_ddp}"

    print(cmd)
    
    if local_rank == 0:
        os.system(cmd)