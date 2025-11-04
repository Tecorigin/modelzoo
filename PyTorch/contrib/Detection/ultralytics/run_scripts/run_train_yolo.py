# BSD 3- Clause License Copyright (c) 2023, Tecorigin Co., Ltd. All rights
# reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software
# without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY
# WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
# OF SUCH DAMAGE.


import os
import argparse


parser = argparse.ArgumentParser()

parser_mode = parser.add_argument_group('Mode Settings')
parser_mode.add_argument('--mode', type=str, default='detect', help='detect, segment, classify, pose, obb')

parser_train = parser.add_argument_group('Train Settings')
parser_train.add_argument('--model', type=str, default=None, help='Specifies the model file for training. Accepts a path to either a .pt pretrained model or a .yaml configuration file. Essential for defining the model structure or initializing weights.')
parser_train.add_argument('--data', type=str, default=None, help='Path to the dataset configuration file (e.g., coco8.yaml). This file contains dataset-specific parameters, including paths to training and validation data, class names, and number of classes.')
parser_train.add_argument('--epochs', type=int, default=100, help='Total number of training epochs. Each epoch represents a full pass over the entire dataset. Adjusting this value can affect training duration and model performance.')
parser_train.add_argument('--time', type=float, default=None, help='Maximum training time in hours. If set, this overrides the epochs argument, allowing training to automatically stop after the specified duration. Useful for time-constrained training scenarios.')
parser_train.add_argument('--patience', type=int, default=100, help='Number of epochs to wait without improvement in validation metrics before early stopping the training. Helps prevent overfitting by stopping training when performance plateaus.')
parser_train.add_argument('--batch', type=int, default=16, help='Batch size for training, indicating how many images are processed before the model\'s internal parameters are updated. AutoBatch (batch=-1) dynamically adjusts the batch size based on GPU memory availability.')
parser_train.add_argument('--imgsz', type=int, default=640, help='Target image size for training. All images are resized to this dimension before being fed into the model. Affects model accuracy and computational complexity.')
parser_train.add_argument('--save', type=str, default='True', help='Enables saving of training checkpoints and final model weights. Useful for resuming training or model deployment.')
parser_train.add_argument('--save_period', type=int, default=-1, help='Frequency of saving model checkpoints, specified in epochs. A value of -1 disables this feature. Useful for saving interim models during long training sessions.')
parser_train.add_argument('--cache', type=str, default='False', help='Enables caching of dataset images in memory (True/ram), on disk (disk), or disables it (False). Improves training speed by reducing disk I/O at the cost of increased memory usage.')
parser_train.add_argument('--device', type=str, default=None, help='Specifies the computational device(s) for training: a single GPU (device=0), multiple GPUs (device=0,1), CPU (device=cpu), or MPS for Apple silicon (device=mps).')
parser_train.add_argument('--workers', type=int, default=8, help='Number of worker threads for data loading (per RANK if Multi-GPU training). Influences the speed of data preprocessing and feeding into the model, especially useful in multi-GPU setups.')
parser_train.add_argument('--project', type=str, default=None, help='Name of the project directory where training outputs are saved. Allows for organized storage of different experiments.')
parser_train.add_argument('--name', type=str, default=None, help='Name of the training run. Used for creating a subdirectory within the project folder, where training logs and outputs are stored.')
parser_train.add_argument('--exist_ok', type=str, default='False', help='If True, allows overwriting of an existing project/name directory. Useful for iterative experimentation without needing to manually clear previous outputs.')
parser_train.add_argument('--pretrained', type=str, default='True', help='Determines whether to start training from a pretrained model. Can be a boolean value or a string path to a specific model from which to load weights. Enhances training efficiency and model performance.')
parser_train.add_argument('--optimizer', type=str, default='auto', help='Choice of optimizer for training. Options include SGD, Adam, AdamW, NAdam, RAdam, RMSProp etc., or auto for automatic selection based on model configuration. Affects convergence speed and stability.')
parser_train.add_argument('--verbose', type=str, default='False', help='Enables verbose output during training, providing detailed logs and progress updates. Useful for debugging and closely monitoring the training process.')
parser_train.add_argument('--seed', type=int, default=0, help='Sets the random seed for training, ensuring reproducibility of results across runs with the same configurations.')
parser_train.add_argument('--deterministic', type=str, default='True', help='Forces deterministic algorithm use, ensuring reproducibility but may affect performance and speed due to the restriction on non-deterministic algorithms.')
parser_train.add_argument('--single_cls', type=str, default='False', help='Treats all classes in multi-class datasets as a single class during training. Useful for binary classification tasks or when focusing on object presence rather than classification.')
parser_train.add_argument('--rect', type=str, default='False', help='Enables rectangular training, optimizing batch composition for minimal padding. Can improve efficiency and speed but may affect model accuracy.')
parser_train.add_argument('--cos_lr', type=str, default='False', help='Utilizes a cosine learning rate scheduler, adjusting the learning rate following a cosine curve over epochs. Helps in managing learning rate for better convergence.')
parser_train.add_argument('--close_mosaic', type=int, default=10, help='Disables mosaic data augmentation in the last N epochs to stabilize training before completion. Setting to 0 disables this feature.')
parser_train.add_argument('--resume', type=str, default='False', help='Resumes training from the last saved checkpoint. Automatically loads model weights, optimizer state, and epoch count, continuing training seamlessly.')
parser_train.add_argument('--amp', type=str, default='True', help='Enables Automatic Mixed Precision (AMP) training, reducing memory usage and possibly speeding up training with minimal impact on accuracy.')
parser_train.add_argument('--fraction', type=float, default=1.0, help='Specifies the fraction of the dataset to use for training. Allows for training on a subset of the full dataset, useful for experiments or when resources are limited.')
parser_train.add_argument('--profile', type=str, default='False', help='Enables profiling of ONNX and TensorRT speeds during training, useful for optimizing model deployment.')
parser_train.add_argument('--freeze', type=str, default=None, help='Freezes the first N layers of the model or specified layers by index, reducing the number of trainable parameters. Useful for fine-tuning or transfer learning.')
parser_train.add_argument('--lr0', type=float, default=0.01, help='Initial learning rate (i.e. SGD=1E-2, Adam=1E-3). Adjusting this value is crucial for the optimization process, influencing how rapidly model weights are updated.')
parser_train.add_argument('--lrf', type=float, default=0.01, help='Final learning rate as a fraction of the initial rate = (lr0 * lrf), used in conjunction with schedulers to adjust the learning rate over time.')
parser_train.add_argument('--momentum', type=float, default=0.937, help='Momentum factor for SGD or beta1 for Adam optimizers, influencing the incorporation of past gradients in the current update.')
parser_train.add_argument('--weight_decay', type=float, default=0.0005, help='L2 regularization term, penalizing large weights to prevent overfitting.')
parser_train.add_argument('--warmup_epochs', type=float, default=3.0, help='Number of epochs for learning rate warmup, gradually increasing the learning rate from a low value to the initial learning rate to stabilize training early on.')
parser_train.add_argument('--warmup_momentum', type=float, default=0.8, help='Initial momentum for warmup phase, gradually adjusting to the set momentum over the warmup period.')
parser_train.add_argument('--warmup_bias_lr', type=float, default=0.1, help='Learning rate for bias parameters during the warmup phase, helping stabilize model training in the initial epochs.')
parser_train.add_argument('--box', type=float, default=7.5, help='Weight of the box loss component in the loss function, influencing how much emphasis is placed on accurately predicting bounding box coordinates.')
parser_train.add_argument('--cls', type=float, default=0.5, help='Weight of the classification loss in the total loss function, affecting the importance of correct class prediction relative to other components.')
parser_train.add_argument('--dfl', type=float, default=1.5, help='Weight of the distribution focal loss, used in certain YOLO versions for fine-grained classification.')
parser_train.add_argument('--pose', type=float, default=12.0, help='Weight of the pose loss in models trained for pose estimation, influencing the emphasis on accurately predicting pose keypoints.')
parser_train.add_argument('--kobj', type=float, default=2.0, help='Weight of the keypoint objectness loss in pose estimation models, balancing detection confidence with pose accuracy.')
parser_train.add_argument('--label_smoothing', type=float, default=0.0, help='Applies label smoothing, softening hard labels to a mix of the target label and a uniform distribution over labels, can improve generalization.')
parser_train.add_argument('--nbs', type=int, default=64, help='Nominal batch size for normalization of loss.')
parser_train.add_argument('--overlap_mask', type=str, default='True', help='Determines whether segmentation masks should overlap during training, applicable in instance segmentation tasks.')
parser_train.add_argument('--mask_ratio', type=int, default=4, help='Downsample ratio for segmentation masks, affecting the resolution of masks used during training.')
parser_train.add_argument('--dropout', type=float, default=0.0, help='Dropout rate for regularization in classification tasks, preventing overfitting by randomly omitting units during training.')
parser_train.add_argument('--val', type=str, default='True', help='Enables validation during training, allowing for periodic evaluation of model performance on a separate dataset.')
parser_train.add_argument('--plots', type=str, default='False', help='Generates and saves plots of training and validation metrics, as well as prediction examples, providing visual insights into model performance and learning progression.')
                    
parser_augment = parser.add_argument_group('Augmentation Settings')
parser_augment.add_argument('--hsv_h', type=float, default=0.015, help='Adjusts the hue of the image by a fraction of the color wheel, introducing color variability. Helps the model generalize across different lighting conditions.')
parser_augment.add_argument('--hsv_s', type=float, default=0.7, help='Alters the saturation of the image by a fraction, affecting the intensity of colors. Useful for simulating different environmental conditions.')
parser_augment.add_argument('--hsv_v', type=float, default=0.4, help='Modifies the value (brightness) of the image by a fraction, helping the model to perform well under various lighting conditions.')
parser_augment.add_argument('--degrees', type=float, default=0.0, help='Rotates the image randomly within the specified degree range (-180 to +180), improving the model\'s ability to recognize objects at various orientations.')
parser_augment.add_argument('--translate', type=float, default=0.1, help='Translates the image horizontally and vertically by a fraction (0.0 - 1.0) of the image size, aiding in learning to detect partially visible objects.')
parser_augment.add_argument('--scale', type=float, default=0.5, help='Scales the image by a gain factor (>=0.0), simulating objects at different distances from the camera.')
parser_augment.add_argument('--shear', type=float, default=0.0, help='Shears the image by a specified degree range (-180 to +180), mimicking the effect of objects being viewed from different angles.')
parser_augment.add_argument('--perspective', type=float, default=0.0, help='Applies a random perspective transformation to the image (0.0 - 0.001), enhancing the model\'s ability to understand objects in 3D space.')
parser_augment.add_argument('--flipud', type=float, default=0.0, help='Flips the image upside down with the specified probability (0.0 - 1.0), increasing the data variability without affecting the object\'s characteristics.')
parser_augment.add_argument('--fliplr', type=float, default=0.5, help='Flips the image left to right with the specified probability (0.0 - 1.0), useful for learning symmetrical objects and increasing dataset diversity.')
parser_augment.add_argument('--bgr', type=float, default=0.0, help='Flips the image channels from RGB to BGR with the specified probability (0.0 - 1.0), useful for increasing robustness to incorrect channel ordering.')
parser_augment.add_argument('--mosaic', type=float, default=1.0, help='Combines four training images into one (0.0 - 1.0), simulating different scene compositions and object interactions. Highly effective for complex scene understanding.')
parser_augment.add_argument('--mixup', type=float, default=0.0, help='Blends two images and their labels (0.0 - 1.0), creating a composite image. Enhances the model\'s ability to generalize by introducing label noise and visual variability.')
parser_augment.add_argument('--copy_paste', type=float, default=0.0, help='Copies objects from one image and pastes them onto another (0.0 - 1.0), useful for increasing object instances and learning object occlusion.')
parser_augment.add_argument('--auto_augment', type=str, default='randaugment', help='Automatically applies a predefined augmentation policy (randaugment, autoaugment, augmix), optimizing for classification tasks by diversifying the visual features.')
parser_augment.add_argument('--erasing', type=float, default=0.4, help='Randomly erases a portion of the image during classification training (0.0 - 0.9), encouraging the model to focus on less obvious features for recognition.')
parser_augment.add_argument('--crop_fraction', type=float, default=1.0, help='Crops the classification image to a fraction (0.1 - 1.0) of its size to emphasize central features and adapt to object scales, reducing background distractions.')


if __name__ == '__main__':
    args, _ = parser.parse_known_args()

    mode = args.mode

    args_dict = vars(args)
    args_dict.pop('mode')

    os.system(f'yolo {mode} train ' + ' '.join([f'{k}={v}' for k, v in args_dict.items() if parser.get_default(k) != v]))
