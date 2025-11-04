#!/usr/bin/env bash

DOWNLOAD_DIR=$1
DATA_ROOT=$2

# unzip $DOWNLOAD_DIR/OpenDataLab___COCO_2017/raw/Images/val2017.zip -d $DATA_ROOT
# unzip $DOWNLOAD_DIR/OpenDataLab___COCO_2017/raw/Images/train2017.zip -d $DATA_ROOT
# unzip $DOWNLOAD_DIR/OpenDataLab___COCO_2017/raw/Images/test2017.zip -d $DATA_ROOT/
# unzip $DOWNLOAD_DIR/OpenDataLab___COCO_2017/raw/Images/unlabeled2017.zip -d $DATA_ROOT
# unzip $DOWNLOAD_DIR/OpenDataLab___COCO_2017/raw/Annotations/stuff_annotations_trainval2017.zip -d $DATA_ROOT/
# unzip $DOWNLOAD_DIR/OpenDataLab___COCO_2017/raw/Annotations/panoptic_annotations_trainval2017.zip -d $DATA_ROOT/
# unzip $DOWNLOAD_DIR/OpenDataLab___COCO_2017/raw/Annotations/image_info_unlabeled2017.zip -d $DATA_ROOT/
# unzip $DOWNLOAD_DIR/OpenDataLab___COCO_2017/raw/Annotations/image_info_test2017.zip -d $DATA_ROOT/
# unzip $DOWNLOAD_DIR/OpenDataLab___COCO_2017/raw/Annotations/annotations_trainval2017.zip -d $DATA_ROOT
unzip $DOWNLOAD_DIR/val2017.zip -d $DATA_ROOT
unzip $DOWNLOAD_DIR/train2017.zip -d $DATA_ROOT
unzip $DOWNLOAD_DIR/test2017.zip -d $DATA_ROOT
unzip $DOWNLOAD_DIR/unlabeled2017.zip -d $DATA_ROOT
unzip $DOWNLOAD_DIR/stuff_annotations_trainval2017.zip -d $DATA_ROOT
unzip $DOWNLOAD_DIR/panoptic_annotations_trainval2017.zip -d $DATA_ROOT
unzip $DOWNLOAD_DIR/image_info_unlabeled2017.zip -d $DATA_ROOT
unzip $DOWNLOAD_DIR/image_info_test2017.zip -d $DATA_ROOT
unzip $DOWNLOAD_DIR/annotations_trainval2017.zip -d $DATA_ROOT
