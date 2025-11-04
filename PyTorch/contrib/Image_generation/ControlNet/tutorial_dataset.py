import json
import cv2
import numpy as np

from torch.utils.data import Dataset
from torch_sdaa.utils.cuda_migrate import apply_monkey_patches
apply_monkey_patches()

class MyDataset(Dataset):
    def __init__(self, max_samples=200):  # ✅ 加上 max_samples 参数，默认2500张
        self.data = []
        with open('./training/fill50k/prompt.json', 'rt') as f:
            for i, line in enumerate(f):
                if i >= max_samples:  # 只加载前 max_samples 条数据
                    break
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        source = cv2.imread('./training/fill50k/' + source_filename)
        target = cv2.imread('./training/fill50k/' + target_filename)

        if source is None or target is None:
            raise FileNotFoundError(f"File not found: {source_filename} or {target_filename}")

    # 降低分辨率，比如缩放到 256x256
        new_size = (256, 256)
        source = cv2.resize(source, new_size, interpolation=cv2.INTER_AREA)
        target = cv2.resize(target, new_size, interpolation=cv2.INTER_AREA)

        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        source = source.astype(np.float32) / 255.0
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)
