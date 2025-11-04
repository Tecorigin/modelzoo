
#!/bin/bash
cd ..
LOG_FILE="scripts/train_sdaa_3rd.log"

torchrun --nproc_per_node 4 train.py --config config_v1.json > "${LOG_FILE}" 2>&1