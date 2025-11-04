#!/bin/bash
################基础配置参数，需要模型审视修改##################
# 必选字段(必须在此处定义的参数): Network batch_size RANK_SIZE
# 网络名称，同目录名称
cur_path=`pwd`
Network="PnasNet5Large"
# 训练batch_size
batch_size=8
# 训练使用的sdaa卡数
RANK_SIZE=""
# 数据集路径,保持为空,不需要修改
data_path=""
cur_path=`pwd`
cd ..
test_path_dir=$cur_path


# 参数校验，data_path为必传参数，其他参数的增删由模型自身决定；此处新增参数需在上面有定义并赋值
for para in $*
do
    if [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    elif [[ $para == --rank_size* ]];then
        RANK_SIZE=`echo ${para#*=}`
        export RANK_SIZE=$RANK_SIZE
    fi
done


RANK_ID_START=0
max_step=100
if [ -d ${test_path_dir}/output ];then
    rm -rf ${test_path_dir}/output
    mkdir -p ${test_path_dir}/output
else
    mkdir -p ${test_path_dir}/output
fi

pip3 install -r ./requirements.txt

for((RANK_ID=$RANK_ID_START;RANK_ID<$((RANK_ID_START+RANK_SIZE));RANK_ID++));
do
nohup python ./imagenet_fast.py --start-epoch 0 --data $data_path --max_step ${max_step}  --epochs 1 --wd 4e-5 --gamma 0.97  --world_size $RANK_SIZE \
     --train-batch $batch_size --test-batch $batch_size --wd-all --warmup 0  --workers 8 --lr 0.4 --print-freq 1  --use_aux  --device sdaa --local_rank $RANK_ID --loss_scale 'dynamic' \
     > ${test_path_dir}/output/${RANK_ID}_nohup.out &
done
