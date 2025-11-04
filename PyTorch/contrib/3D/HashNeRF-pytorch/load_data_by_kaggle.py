import kagglehub

# Download latest version
path = kagglehub.dataset_download("sauravmaheshkar/nerf-dataset")

print("Path to dataset files:", path)
#下载的数据集会存放在/root/.cache/kagglehub/datasets/arenagrenade/llff-dataset-full/versions/1