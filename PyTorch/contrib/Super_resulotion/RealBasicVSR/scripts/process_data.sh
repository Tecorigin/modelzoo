cd ..
python tools/dataset_converters/reds/preprocess_reds_dataset.py --root-path ./data/REDS
python tools/dataset_converters/reds/crop_sub_images.py --data-root ./data/REDS  -scales 4