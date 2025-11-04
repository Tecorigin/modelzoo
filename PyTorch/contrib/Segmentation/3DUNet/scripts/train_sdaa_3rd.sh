cd ..
pip install -e .
pip install -r requirements.txt
SDAA_VISIBLE_DEVICES=0,1,2,3 
train3dunet --config resources/3DUnet_confocal_boundary/train_config.yml