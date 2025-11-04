## 内嵌物理知识神经网络 Physics Informed Neural Network（PINN）

内嵌物理知识神经网络（Physics Informed Neural Network，简称PINN）是一种将机器学习领域与传统数值计算领域相结合形成的一种理论与应用方法，特别是用于解决与偏微分方程（PDE）相关的各种问题，包括方程求解、参数反演、模型发现、控制与优化等。

PINN 的原理简单概括来说就是通过训练神经网络，最小化其损失函数来近似 PDE 的求解，所谓的损失函数项包括了初值和边界条件的残差项，以及给定区域中选定点（按传统应该称为“配点”）处的偏微分方程残差项，使用训练完成后的模型进行推断（Inference）即可得到时空点上的值。

如果把 PINN 当作是单纯的数值求解器，无论从速度或者精度上来说，通常 PINN 在性能上并不能跟传统方法（有限差分、有限元、有限体积等大类方法）抗衡。但 PINN 的优势在于这种方法或者思想可以弥补科学机器学习领域中单纯数据驱动的弱点。如果把传统数值格式认为是单纯物理知识驱动，那么 PINN 或者更广义一点的内嵌物理知识机器学习就是数据驱动与知识驱动的融合方法。同时，PINN 也相对降低了求解 PDE 的难度，使用者无需深入学习更多 PDE 的相关知识便可以使用和构建 PINN。另外，相关科学领域的使用者可以通过 PINN 进一步弥合数据与知识之间的差距，从而进一步补充或是发现相应的科学定律。

### 1. 偏微分方程概述

一个偏微分方程求解问题通常包括如下内容

*   由算子 ${\mathcal F}$ 描述的控制方程 ${\mathcal F}(u(\boldsymbol{x}, t), \gamma) = f(\boldsymbol{x}, t), (\boldsymbol{x}, t) \in \Omega$，其中 $\boldsymbol{x}$ 为空间坐标，$t$ 为时间坐标，$\gamma$ 为控制方程参数，$u(\boldsymbol{x}, t)$ 为方程的解，$\Omega$ 为方程所在的区域

*   由算子 ${\mathcal B}$ 描述的边界条件 ${\mathcal B}(u(\boldsymbol{x}, t)) = b(\boldsymbol{x}, t), (\boldsymbol{x}, t) \in \partial \Omega$

*   由算子 ${\mathcal I}$ 描述的初值条件 ${\mathcal I}(u(\boldsymbol{x_{0}}, t_{0})) = i(\boldsymbol{x_{0}}, t_{0})$，这一项不一定是必须的

如经典的波动方程

*    $① \qquad u_{tt}(x, t) = (2\pi)^{2}u_{xx}(x, t)$
*    $② \qquad u(x, 0) = \cos(2\pi x)$
*    $③ \qquad u_{t}(x, 0) = -\sin(2\pi x)$

其中①是控制方程，②③是边界条件，其解析解为 $u(x, t) = \cos(2\pi x + t)$

### 2. PINN 原理概述

通常在研究实际问题中，会得到一些具体的观测数据，即

*   由算子 ${\mathcal D}$ 描述为 ${\mathcal D}(u(\boldsymbol{x_{i}}, t_{i})) = d(\boldsymbol{x_{i}}, t_{i}), i \in D$

现在针对求解所包含的信息为控制方程项 ${\mathcal F}$，边界条件项 ${\mathcal B}$，初值条件项 ${\mathcal I}$ 以及具体观测数据项 ${\mathcal D}$，相比传统微分方程数值求解的描述，这里多出对于具体观测数据的使用，通过数据进行驱动是 PINN 的特点之一。

PINN 所要做的即是对解 $u(\boldsymbol{x}, t)$ 用神经网络进行逼近，将这个解用神经网络参数化表达为 ${\cal U_{\theta}}$，那么需要寻找一组参数 $\theta$ 使得 ${\cal U_{\theta}}(\boldsymbol{x}, t) \approx u(\boldsymbol{x}, t)$。

通常 ${\cal U_{\theta}}$ 具有多层感知机或者加入特殊结构的变种，这里以多层感知机（MLP）为例，即

*   ${\cal U_{\theta}}(\boldsymbol{x}, t) = f_{n}(\boldsymbol{x}, t) \circ f_{n - 1}(\boldsymbol{x}, t) \circ ... \circ f_{1}(\boldsymbol{x}, t)$

除最后一层外，其余的各层都是“线性变换 + 激活函数”层。这里的时空信息都被包含在其中，便可以通过自动微分运算来表达这些微分算子，这只是个最基本的模型，首先由下文提出，目前也被使用得最多。其他加入了 Resnet、soft-attention、Echo State Network 的结构也不鲜见，总之，这类结构的共同特点是可以对 $(\boldsymbol{x}, t)$ 求自动微分。

> Raissi, Maziar, Paris Perdikaris, and George Em Karniadakis. "Physics informed deep learning (part i): Data-driven solutions of nonlinear partial differential equations." arXiv preprint arXiv:1711.10561 (2017).

神经网络 ${\cal U_{\theta}}(\boldsymbol{x}, t)$ 在定义方程四个公式中的残差，就可以引出由四项损失加权得到的总损失，即

*   $L = w_{{\mathcal F}}L_{{\mathcal F}} + w_{{\mathcal B}}L_{{\mathcal B}} + w_{{\mathcal I}}L_{{\mathcal I}} + w_{{\mathcal D}}L_{{\mathcal D}}$

这四项其实比较笼统，还可以加入正则化项，以及其他各种先验信息。对于具体问题，细分下来可以有十多项。最后变成了这样一个优化问题，即

*   $\theta^{*} \in argmin_{\theta} L[{\cal U_{\theta}}(\boldsymbol{x}, t)]$

此时便回归到了传统的机器学习领域中。      

### 3. PINN 求解 PDE 实例

**例 1** 求如下波动方程

*    $ u_{tt}(x, t) = u_{xx}(x, t) $
*    $ u(x, 0) = \cos(x) $
*    $ u_{t}(x, 0) = -\sin(x) $

在 $x, t \in [0, \frac{\pi}{2}]$ 的数值解

首先在区域 $[0, \frac{\pi}{2}] \times [0, \frac{\pi}{2}]$ 上分别针对上述三个方程采集样本点 $(x_{i}, t_{i})$，随后针对上述三个方程构造损失函数，即

*   $L_{{\cal F}} = \frac{1}{N_{{\cal F}}}\Sigma_{i=1}^{N_{{\cal F}}} || {\cal U_{\theta tt}}(x_{i}, t_{i}) - {\cal U_{\theta xx}}(x_{i}, t_{i}) ||^2$

*   $L_{{\cal B_{1}}} = \frac{1}{N_{{\cal B_{1}}}}\Sigma_{i=1}^{N_{{\cal B_{1}}}} || {\cal U_{\theta}}(x_{i}, 0) - \cos(x_{i}) ||^2$

*   $L_{{\cal B_{2}}} = \frac{1}{N_{{\cal B_{2}}}}\Sigma_{i=1}^{N_{{\cal B_{2}}}} || {\cal U_{\theta t}}(x_{i}, 0) + \sin(x_{i}) ||^2$ 

总的损失函数构造为

*   $L = L_{{\cal F}} + L_{{\cal B_{1}}} + L_{{\cal B_{2}}}$

仍采用多层感知机结构，所设计的网络结构如下所示

```python
class PhysicsInformedNeuralNetwork(torch.nn.Module):
    def __init__(self):
        super(PhysicsInformedNeuralNetwork, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(2, 16),
            torch.nn.Tanh(),
            torch.nn.Linear(16, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 16),
            torch.nn.Tanh(),
            torch.nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x)
```

可使用 `pytorch` 中的自动微分函数求方程中所包含的微分计算，即如下函数

```python
torch.autograd.grad(outputs, inputs, grad_outputs=None, retain_graph=None, create_graph=False, only_inputs=True, allow_unused=None, is_grads_batched=False, materialize_grads=False)
```

PINN 的求解结果如下图所示

![PINN 求解波动方程结果](https://gitee.com/ThreeJoe/modelzoo/raw/contrib/ThreeJoe/PINN/PyTorch/contrib/AI4Science/PINN/png/Figure_1.png)

该波动方程的解析解为 $u(x, y) = \cos(x + t)$，其图像如下图所示

![波动方程解析解](https://gitee.com/ThreeJoe/modelzoo/raw/contrib/ThreeJoe/PINN/PyTorch/contrib/AI4Science/PINN/png/Figure_2.png)

**例 2** 求如下 Laplace 方程

*    $ u_{xx}(x, y) + u_{yy}(x, y) = 0 $
*    $ u(x, 0) = \sin(x) $
*    $ u(x, \pi) = \sin(x)\frac{e^{\pi} + e^{-\pi}}{2} $
*    $ u(0, y) = 0 $
*    $ u(\pi, y) = 0 $

在 $x, y \in [0, \pi]$ 的数值解

与例 1 类似，在区域 $[0, \pi] \times [0, \pi]$ 上分别针对上述五个方程采集样本点 $(x_{i}, y_{i})$，同时构造如下损失函数

*   $L_{{\cal F}} = \frac{1}{N_{{\cal F}}}\Sigma_{i=1}^{N_{{\cal F}}} || {\cal U_{\theta yy}}(x_{i}, y_{i}) + {\cal U_{\theta xx}}(x_{i}, y_{i}) ||^2$

*   $L_{{\cal B_{1}}} = \frac{1}{N_{{\cal B_{1}}}}\Sigma_{i=1}^{N_{{\cal B_{1}}}} || {\cal U_{\theta}}(x_{i}, 0) - \sin(x_{i}) ||^2$

*   $L_{{\cal B_{2}}} = \frac{1}{N_{{\cal B_{2}}}}\Sigma_{i=1}^{N_{{\cal B_{2}}}} || {\cal U_{\theta}}(x_{i}, 0) - \sin(x_{i})\frac{e^{\pi} + e^{-\pi}}{2} ||^2$ 

*   $L_{{\cal B_{3}}} = \frac{1}{N_{{\cal B_{3}}}}\Sigma_{i=1}^{N_{{\cal B_{3}}}} || {\cal U_{\theta}}(0, y_{i}) ||^2$ 

*   $L_{{\cal B_{4}}} = \frac{1}{N_{{\cal B_{4}}}}\Sigma_{i=1}^{N_{{\cal B_{4}}}} || {\cal U_{\theta}}(\pi, y_{i}) ||^2$     

总的损失函数构造为

*   $L = L_{{\cal F}} + L_{{\cal B_{1}}} + L_{{\cal B_{2}}} + L_{{\cal B_{3}}} + L_{{\cal B_{4}}}$    

神经网络仍选用例 1 中所述的，PINN 的求解结果如下图所示

![PINN 求解 Laplace 方程结果](https://gitee.com/ThreeJoe/modelzoo/raw/contrib/ThreeJoe/PINN/PyTorch/contrib/AI4Science/PINN/png/Figure_3.png)

该 Laplace 方程的解析解为 $u(x, y) = \sin(x)\cosh(y)$，其图像如下图所示

![波动方程解析解](https://gitee.com/ThreeJoe/modelzoo/raw/contrib/ThreeJoe/PINN/PyTorch/contrib/AI4Science/PINN/png/Figure_4.png)

### 4. 快速开始运行实例

#### 4.1 环境准备

##### 4.1.1 拉取代码仓库

```bash
git clone https://gitee.com/tecorigin/modelzoo.git
```

##### 4.1.2 Docker 环境准备

###### 4.1.2.1 获取 SDAA TecoPytorch 基础 docker 环境

SDAA 提供了支持 TecoPytorch 的 docker 镜像，请参考 [Teco文档中心的教程->安装指南->Docker安装](http://docs.tecorigin.com/release/tecopytorch_course/v1.0.0/#37d7c9feecc811eea206024214151608) 中的内容进行 TecoPytorch 基础 docker 镜像的部署。

环境搭建完成后运行以下命令：

```bash
cd Pytorch/contrib/AI4Science/PINN
```

进入模型所在文件夹

###### 4.1.2.1 创建 Teco 虚拟环境

```bash
cd Pytorch/contrib/AI4Science/PINN
conda activate teco-pytorch

# 依次执行如下命令确认正常开启虚拟环境
python
import torch
```

#### 4.2 数据集准备

上述实例中仅包含在特定区域内随机采样得到的数据集，未包含具体的观测数据

#### 4.3 启动训练

进入如下目录

```bash
cd Pytorch/contrib/AI4Science/PINN/run_scripts
```

选择所要运行的实例以及相应的运行参数，如

```bash
python run_pytorch_pinn.py --model_name=wave --nnode=1 --nproc_per_node=1 --device=sdaa --epoch=4096 --batch_size=512 --show_graph=False
```

出现 `training model finished` 则表示程序正常运行完毕，模型随后保存在当前目录下。

**模型训练脚本参数说明如下**

| 参数名 | 解释 | 样例 |
|--|--|--|
| model_name | 模型名称，目前为 wave 或 laplace | --model_name=wave |
| epoch | 训练轮次，和训练步数冲突，默认为 8192 | --epoch=4096 |
| batch_size | 每一轮的批次大小，默认为 1024 | --batch_size=512 |
| device | 指定运行设备，默认为 sdaa | --device=sdaa |
| show_graph | 查看偏微分方程解的图像，默认为 False | --show_graph=False |
| nnode | 用于执行分布式数据并行训练时 node 的数量，默认为 1 | --nnode=1 |
| nproc_per_node | 在执行分布式数据并行时每个 node 上的 rank 数量, 不输入时默认为 1, 表示单核执行 | --nproc_per_node=1 |

注：若需要查看偏微分方程解的图像，请确认当前环境支持图像显示功能。

#### 4.3 训练结果

训练结果如例 1 和例 2 中图像显示。