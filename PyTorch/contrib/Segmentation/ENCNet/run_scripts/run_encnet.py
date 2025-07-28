from argument import parse_args


def build_hyper_parameters(args):
    # 提取并存储参数
    launcher = args.launcher
    amp = args.amp
    config = args.config
    cfg_options = args.cfg_options

    hyper_parameters = f"{config} --launcher {launcher} "

    if args.amp:
        hyper_parameters += f" --amp"
    if args.cfg_options:
        cfg_str = ""
        for k, v in args.cfg_options.items():
            # 确保值被正确格式化
            if isinstance(v, str) and ' ' in v:
                # 如果值包含空格，用引号包裹
                cfg_str += f"{k}='{v}' "
            else:
                cfg_str += f"{k}={v} "       
        hyper_parameters += f" --cfg-options {cfg_str.strip()} "

    return hyper_parameters

def build_command(args,hyper_parameters):

    nnodesx = args.nnodes
    node_rank = args.node_rank
    master_addr = args.master_addr
    master_port = args.master_port
    nproc_per_node = args.nproc_per_node
    cmd = ""
    cmd = f'python -m torch.distributed.launch --nnodes={nnodesx} \
        --node_rank={node_rank} \
        --master_addr={master_addr} \
        --nproc_per_node={nproc_per_node} \
        --master_port={master_port} \
        ../tools/train.py {hyper_parameters}'
    print("cmd--->>>>>:\n{}\n".format(cmd))
    return cmd

def excute_command(cmd):
    import subprocess
    try:
        subprocess.check_call(cmd, shell=True)
    except subprocess.CalledProcessError as e:
        exit_code = e.returncode
        print("Command failed with exit code:", exit_code)
        exit(exit_code)

if __name__ == "__main__":
    args = parse_args()
    hyper_parameters = build_hyper_parameters(args)
    cmd = build_command(args,hyper_parameters)
    excute_command(cmd)

