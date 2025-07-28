# EncNet
## 1. Model Overview
Recent work has made significant progress in improving spatial resolution for pixelwise labeling with Fully Convolutional Network (FCN) framework by employing Dilated/Atrous convolution, utilizing multi-scale features and refining boundaries. In this paper, we explore the impact of global contextual information in semantic segmentation by introducing the Context Encoding Module, which captures the semantic context of scenes and selectively highlights class-dependent featuremaps. The proposed Context Encoding Module significantly improves semantic segmentation results with only marginal extra computation cost over FCN. Our approach has achieved new state-of-the-art results 51.7% mIoU on PASCAL-Context, 85.9% mIoU on PASCAL VOC 2012. Our single model achieves a final score of 0.5567 on ADE20K test set, which surpass the winning entry of COCO-Place Challenge in 2017. In addition, we also explore how the Context Encoding Module can improve the feature representation of relatively shallow networks for the image classification on CIFAR-10 dataset. Our 14 layer network has achieved an error rate of 3.45%, which is comparable with state-of-the-art approaches with over 10 times more layers. The source code for the complete system are publicly available.
- Paper Link: [Context Encoding for Semantic Segmentation](https://arxiv.org/abs/1803.08904)
- Code Link: [Code](https://github.com/zhanghang1989/PyTorch-Encoding)
## 2. Quick Start
The main steps for training with this model are as follows:
1. In the configured environment, navigate to the training script directory.
   ```
   cd <ModelZoo_path>/PyTorch/contrib/Segmentation/encnet/run_scripts
   ```
2. Run the training script.
  ```
   python run_encnet.py --config ../configs/encnet/encnet_r50-d8_4xb2-80k_cityscapes-512x1024.py \
    --launcher pytorch --nproc-per-node 1 --amp 2>&1 | tee sdaa.log
   ```

![loss](./run_scripts/loss.jpg)

MeanRelativeError:-0.19116692150429956

MeanAbsoluteError:-0.49424196600914

Rule,mean absolute error=-0.49424196600914

pass mean relative error=-0.19116692150429956 <= 0.05 or mean absolute error=-0.49424196600914 <= 0.0002