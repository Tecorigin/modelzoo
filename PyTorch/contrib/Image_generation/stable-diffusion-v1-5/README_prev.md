---
frameworks:
- Pytorch
license: Apache License 2.0
tasks:
- image-style-transfer
---

Copied from: https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5

SDK下载
```bash
#安装ModelScope
pip install modelscope
```
```python
#SDK模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('songkey/stable-diffusion-v1-5')
```
Git下载
```
#Git模型下载
git clone https://www.modelscope.cn/songkey/stable-diffusion-v1-5.git
```