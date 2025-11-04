#!/bin/bash

pip install -r ../requirements.txt

python ../preprocess/process_data.py
if [ $? -eq 0 ]; then
    python ../preprocess/get_list.py
else
    echo "process_data.py 执行失败，停止运行 get_list.py"
    exit 1
fi