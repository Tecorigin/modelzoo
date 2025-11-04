注释掉check.check_gpu(cfg.use_gpu)这行代码

将if cfg.use_gpu:
        place = paddle.set_device('cuda')改为：
 if cfg.use_gpu:
        place = paddle.set_device('sdaa')

配置环境
导入sdaa：
export PADDLE_XCCL_BACKEND=sdaa

数据集：
dataset/mot/MOT17
                --annotations/train_half.json
                --images/train
                --
其中，annotations/train_half.json需要用create_json.py去创建

运行指令：
python -m paddle.distributed.launch --log_dir=ppyoloe --gpus 0,1 tools/train.py -c configs/mot/bytetrack/detector/ppyoloe_crn_l_36e_640x640_mot17half.yml --eval --amp