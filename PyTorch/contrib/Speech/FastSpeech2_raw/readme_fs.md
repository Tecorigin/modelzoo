数据预处理
1.修改config文件
2.python3 prepare_align.py config/LJSpeech/preprocess.yaml
文件目录结构：
|--lispeech_raw
        |-- LJSpeech-1.1
        |-- output

3.运行：python3 preprocess.py config/LJSpeech/preprocess.yaml

cd hifigan
unzip generator_LJSpeech.pth.tar.zip
unzip generator_universal.pth.tar.zip 