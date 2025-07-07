import os
import subprocess
from argument import parse_args

if __name__ == '__main__':
    args = parse_args()

    command = f"python mainpro_FER.py --model {args.model_name} --bs {args.batchsize} --lr {args.lr}"
    if args.resume:
        command += " --resume"

    print("[RUNNING]", command)
    os.system(command)