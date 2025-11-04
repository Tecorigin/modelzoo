## 模型简介：
该项目旨在通过训练生成对抗网络 (GAN) 来去除视频中电视徽标的干扰。

## 数据集下载：

您可以从 [🤗 Hugging Face - tv-logo](URL_ADDRESS您可以从 [🤗 Hugging Face - tv-logo](https://huggingface.co/datasets/nssharmaofficial/tv-logo) 预览和/或下载图像。
- 带有徽标的图像位于文件夹 `'images/logo'` 中，文件名模式为 `'i-j-k.jpg'`
- 没有徽标的图像位于文件夹 `'images/clean'` 中，文件名模式为 `'i-j.jpg'`
- **注意**：一个干净的图像有多个对应的徽标图像
路径到图像首先被分为训练和验证集（70/30）
```python
# 获取对应图像的路径列表（拆分 70/30）
train_logo_paths, val_logo_paths, train_clean_paths, val_clean_paths = get_paths()
```
在 `Dataset` 类中，参数 `patches` 可以定义图像是否被划分为补丁。
```python
train_dataset = Dataset(train_logo_paths, train_clean_paths, patches = True)
```

## 开始训练
python main_generator_only.py
