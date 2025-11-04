参数说明
| 参数名             | 类型  | 默认值                            | 说明             |
| --------------- | --- | ------------------------------ | -------------- |
| `--model_name`  | str | `controlnet`                   | 模型名称（记录用途）     |
| `--batch_size`  | int | `1`                            | 训练批大小          |
| `--max_epochs`  | int | `1`                            | 最大训练轮数         |
| `--max_iters`   | int | `100`                          | 最多迭代步数（控制训练长度） |
| `--log_file`    | str | `sdaa.log`                     | 日志文件输出路径       |
| `--resume_path` | str | `models/control_sd15_ini.ckpt` | 模型加载路径         |
启动训练
```
cd ./ControlNet
python run_scripts/run_controlnet.py \
    --model_name controlnet \
    --batch_size 1 \
    --max_epochs 1 \
    --max_iters 100 \
    --resume_path models/control_sd15_ini.ckpt \
    --log_file sdaa.log
```