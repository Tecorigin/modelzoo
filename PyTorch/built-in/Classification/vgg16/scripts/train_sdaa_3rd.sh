export TORCH_SDAA_AUTOLOAD=cuda_migrate
cd ..
python train.py -net vgg16 -gpu
