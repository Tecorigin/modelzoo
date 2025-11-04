#encoding=utf-8
import importlib
import os
import warnings
import sys
from pathlib import Path


root_dir = Path(__file__).parent.parent.resolve()
os.environ['PYTHONPATH'] = str(root_dir)
sys.path.insert(0, str(root_dir))


import os,shutil
url = r'https://ai-studio-online.bj.bcebos.com/v1/a00e9ca2ddc44de4b82229134f1e345d029fb30cbbfc4b629998ebf79c96810e?responseContentDisposition=attachment%3B%20filename%3DMIR-1K_4_RMVPE.zip&authorization=bce-auth-v1%2F5cfe9a5e1454405eb2a975c43eace6ec%2F2024-06-04T12%3A27%3A26Z%2F-1%2F%2Ff603ff0d9f8029fd1ed811fab957326a32b130b1a7727feb6f78b939b291c84c'

if __name__ == "__main__":
    if os.path.isdir('Hybrid'):
        shutil.rmtree('Hybrid')
    os.system("wget \"{}\" --no-check-certificate -O dashichang.zip".format(url))
    os.system("unzip dashichang")
    os.system("rm dashichang.zip")