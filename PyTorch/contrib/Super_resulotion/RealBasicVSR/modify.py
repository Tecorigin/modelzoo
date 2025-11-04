import os

def add_import_to_files(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                with open(file_path, "r") as f:
                    lines = f.readlines()
                
                # 检查是否已经有 `import torch_sdaa`
                if any("import torch_sdaa" in line for line in lines):
                    continue
                
                # 修改文件内容
                modified = False
                with open(file_path, "w") as f:
                    for line in lines:
                        f.write(line)
                        # 在 `import torch` 后插入 `import torch_sdaa`
                        if "import torch" in line and not modified:
                            f.write("import torch_sdaa\n")
                            modified = True

if __name__ == "__main__":
    directory = "."  # 替换为你的代码目录
    add_import_to_files(directory)
