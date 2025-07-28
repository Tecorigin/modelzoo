# EncNet
## 1. Model Overview
Semantic segmentation is a key technology for autonomous vehicles to understand the surrounding scenes. The appealing performances of contemporary models usually come at the expense of heavy computations and lengthy inference time, which is intolerable for self-driving. Using light-weight architectures (encoder-decoder or two-pathway) or reasoning on low-resolution images, recent methods realize very fast scene parsing, even running at more than 100 FPS on a single 1080Ti GPU. However, there is still a signiﬁcant gap in performance between these real-time methods and the models based on dilation backbones. To tackle this problem, we proposed a family of efﬁcient backbones specially designed for real-time semantic segmentation. The proposed deep dual-resolution networks (DDRNets) are composed of two deep branches between which multiple bilateral fusions are performed. Additionally, we design a new contextual information extractor named Deep Aggregation Pyramid Pooling Module (DAPPM) to enlarge effective receptive ﬁelds and fuse multi-scale context based on low-resolution feature maps. Our method achieves a new state-of-the-art trade-off between accuracy and speed on both Cityscapes and CamVid dataset. In particular, on a single 2080Ti GPU, DDRNet-23-slim yields 77.4% mIoU at 102 FPS on Cityscapes test set and 74.7% mIoU at 230 FPS on CamVid test set. With widely used test augmentation, our method is superior to most state-of-the-art models and requires much less computation. Codes and trained models are available online.
- Paper Link: [Deep Dual-resolution Networks for Real-time and Accurate Semantic Segmentation of Road Scenes](http://arxiv.org/abs/2101.06085)
- Code Link: [Code](https://github.com/ydhongHIT/DDRNet)
## 2. Quick Start
The main steps for training with this model are as follows:
1. In the configured environment, navigate to the training script directory.
   ```
   cd <ModelZoo_path>/PyTorch/contrib/Segmentation/ddrnet_cityscapes/run_scripts
   ```
2. Run the training script.
  ```
   python run_ddrnet.py --config ../configs/ddrnet/ddrnet_23-slim_in1k-pre_2xb6-120k_cityscapes-1024x1024.py \
    --launcher pytorch --nproc-per-node 1 --amp 2>&1 | tee sdaa.log
   ```

![loss](./run_scripts/loss.jpg)

MeanRelativeError: -0.4552674768603906

MeanAbsoluteError: -0.9562658238410949

Rule,mean absolute error -0.9562658238410949

pass mean relative error=-0.4552674768603906 <= 0.05 or mean absolute error=-0.9562658238410949 <= 0.0002
