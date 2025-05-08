"""
1. 运行demo.py,获取demo.log
cd ~/PyTorch/contrib/Classification/Demo/run_scripts
python run_demo.py --nproc_per_node 4 --model_name demo --epoch 2 --batch_size 32 --device sdaa | tee demo.log

2. 运行plt_loss.py,生成loss曲线
cd ~/PyTorch/contrib/Classification/Demo/tools
python plt_loss.py
"""


import matplotlib.pyplot as plt

import tqdm
import re
with open('../run_scripts/demo.log', 'r') as file:
    contents = file.readlines()

loss = []
for content in tqdm.tqdm(contents[:]):
    loss_values = re.findall(r'train.loss\s*:\s*(\d+(?:\.\d+)?)', content)
    if loss_values is []:
        content
    else:
        for loss_value in loss_values:
            loss.append(float(loss_value))

from matplotlib import pyplot as plt
# 创建画布
fig, ax = plt.subplots(figsize=(8, 6))

# 绘制曲线,并指定颜色
ax.plot(loss[:], color='red', label='loss')

# 添加标题和坐标轴标签
ax.set_title('Loss Curves')
ax.set_xlabel('Iteration')
ax.set_ylabel('Loss')

# 添加图例
ax.legend()

# 保存图像
plt.savefig('loss_curves.png')

# 打印最小和最后的loss值
print('Minimum loss:', min(loss))
print('Last loss:', loss[-1])