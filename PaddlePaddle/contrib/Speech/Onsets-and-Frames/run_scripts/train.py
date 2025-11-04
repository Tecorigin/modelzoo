#encoding=utf-8
import os
import sys
from argument import config

if __name__ == "__main__":
    cmd = config()
    assert (npn := cmd.nproc_per_node) >= 1, "nproc_per_node argument shouldn't less than 1"
    if npn == 1:
        # 这里避免用多卡启动，可以让控制台log更加友好
        exit(os.system("python run_scripts/ori_train.py "
                 + " ".join(sys.argv[1:])))
    else:
        device = ",".join([str(i) for i in range(npn)])
        exit(os.system(f"python -m paddle.distributed.launch --device {device} run_scripts/ori_train.py "
                      + " ".join(sys.argv[1:])))