# 1.模型概述
GAT（Graph Attention Networks）是一种专门设计用于处理图结构数据的深度学习模型，它通过引入注意力机制来增强图神经网络的能力。GAT的核心是图注意力层，这种结构允许模型在处理节点间的信息传递时动态地为不同邻居节点分配不同的权重，从而有效地解决了图卷积网络中可能遇到的问题，如依赖于全局图结构和固定的邻域聚合策略。这种方法不仅提升了模型对于复杂图结构数据的学习能力，还特别适用于存在高度异质性和多尺度特征的任务。
GAT模型因其创新性的方法，在多种涉及图数据的应用中表现出色，例如社交网络分析、推荐系统、引文网络分类等。其灵活性和高效性使得GAT成为处理非欧几里得数据（即图数据）的重要工具之一，并且在图表示学习领域占据了关键位置。当前支持的GAT模型变体主要包括基础的GAT架构以及针对特定任务或优化目的而设计的各种扩展版本，这些扩展通常集中在改进注意力机制、增强模型表达能力或是提高训练效率等方面。

# 2.快速开始
使用本模型执行训练的主要流程如下：
    基础环境安装：介绍训练前需要完成的基础环境检查和安装。
    获取数据集：介绍如何获取训练所需的数据集。
    启动训练：介绍如何运行训练。

2.1 基础环境安装
    从已有的torch_env拷贝环境，设虚拟环境名称为gat
    conda create --name gat --clone torch_env
    安装相关库函数：
    conda install --file conda_requirements.txt
2.2 数据集获取
    gat数据集例如cora等数据都很小，可以直接在训练命令行中指定并下载

2.3 启动训练
由于sdaa不支持gat中涉及的sparse数据，因此训练方式有两种，分别是用cpu训练和sdaa训练（修改sparse网络为dense网络，具体实现方式采用linear层来实现）
2.3.1 cpu训练方式
    python3 train_cpu.py --dataset cora
2.3.2 sdaa训练方式
    python3 train.py --dataset cora




创建环境

pip install dgl==1.0

# 由于sdaa不支持稀疏图的运算，因此第一个方法是将原模型放在cpu上运行
    进入目录：cd /dgl/examples/pytorch/gcn
    运行：python3 train_cpu.py --dataset cora

# 方法二：采用linear和torch.mm的方式替代原来的稀疏网络，其次原来sparse图数据转为dense数据，见train.py的转换：
    进入目录：cd /dgl/examples/pytorch/gcn
    运行：python3 train.py --dataset cora
    # 多卡+amp方式：torchrun --nproc_per_node 4 train.py --use_amp True --use_ddp True



Test accuracy 0.8190