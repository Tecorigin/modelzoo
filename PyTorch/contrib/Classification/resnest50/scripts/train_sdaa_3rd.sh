cd ..
pip install resnest --pre
pip install -r requirements.txt
python ./scripts/torch/train.py --config-file ./configs/config50.yaml