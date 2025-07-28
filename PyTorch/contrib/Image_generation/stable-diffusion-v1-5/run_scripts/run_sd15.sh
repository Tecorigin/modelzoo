#!/bin/bash

current_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
train_dir="$(realpath "$current_dir/../")"

cd "$train_dir" || { echo "进入 train.py 目录失败"; exit 1; }

echo "运行训练命令: python train.py $*"
python train.py "$@"

ret=$?
if [ $ret -ne 0 ]; then
    echo "训练命令执行失败"
    exit $ret
else
    echo "训练命令执行完成"
fi
