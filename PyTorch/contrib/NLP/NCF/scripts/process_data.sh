cd ../v0.5.0/nvidia/submission/code/recommendation
# Creates ml-20.zip
source recommendation/download_dataset.sh
# Confirms the MD5 checksum of ml-20.zip
source recommendation/verify_dataset.sh
# Unzip
unzip ml-20m.zip
# Preprocessing
python convert.py --path ./ml-20m/ratings.csv --output /data/ml-20m