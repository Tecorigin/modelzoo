import kagglehub

# Download latest version
path = kagglehub.dataset_download("arenagrenade/llff-dataset-full")

print("Path to dataset files:", path)