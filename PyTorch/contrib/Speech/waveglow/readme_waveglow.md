# 拷贝已有的torch环境
    conda create --name waveglow  --clone torch_env
# 下载原项目代码包
    git clone https://github.com/NVIDIA/waveglow.git

# 安装依赖库函数
    pip install scipy
    pip install librosa==0.10.2

# 数据集准备：
    # 数据集格式
    # 修改config.json中的training_files为对应的训练数据文件
# 在tacotron2文件夹下放置layers.py文件、audio_processing.py文件、stft.py

    mel2samp.py的89行修改为：filename = self.audio_files[index].split('|')[0]

# 在train.py同级别目录下放置multiproc.py，并且创建一个logs文件夹
# 在train.py下添加参数n_gpus:parser.add_argument('--n_gpus', type=int, default=4,required=False, help='number of gpus')
    增加一个to_gpu函数