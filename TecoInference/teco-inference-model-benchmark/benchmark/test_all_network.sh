# BSD 3- Clause License Copyright (c) 2023, Tecorigin Co., Ltd. All rights
# reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software
# without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY
# WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
# OF SUCH DAMAGE.

# !/bin/bash

#set -euo pipefail

if [ -z $1 ];then
echo "Specify json files to run "
echo "e.g. bash test_all_network.sh json "
exit -1
fi

files=()
# 定义函数来递归遍历目录
function traverse_directory() {
    for file in "$1"/*; do
        if [ -d "$file" ]; then
            # 如果是目录，则递归调用自身
            traverse_directory "$file"
        elif [ "${file##*.}" = "json" ]; then
            # 如果是JSON文件，则打印完整路径
            files+=(${file})
        fi
    done
}

function traverse_file() {
    while IFS= read -r line; do
        files+=("$line")
    done < "$1"
}

# 主程序
directory=$1  # jsons目录路径，请根据实际情况修改
if [ -d "$directory" ]; then
    traverse_directory "$directory"
elif [ -f "$directory" ]; then
    traverse_file "$directory"
else
    echo "目录不存在：$directory"
fi

mkdir -p logs
for file in ${files[@]}
do
    echo "${file}"
    python3 main.py --json  ${file} 2>&1 | tee logs/$(basename "$file")
done

pip install openpyxl
python3 process_log.py

