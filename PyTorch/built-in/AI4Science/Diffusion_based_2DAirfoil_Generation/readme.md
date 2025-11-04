## 模型简介：
该项目探索了如何使用条件生成模型来生成逼真的 2D 翼型样本以进行空气动力学优化。训练基于神经网络的代理模型来评估翼型。PCA 用于编码 2D 翼型到潜在空间的几何表示。

该项目展示了该模型在指定升力系数阈值内生成可行解决方案的能力，以及在优化问题中应用条件生成模型的可能性。

## 数据预处理
```shell
python preprocess.py
```

## 模型训练
Surrogate model training:
```
python -m models.airfoil_surrogate
```
Diffusion model training:
```
python -m models.airfoil_MLP 
```