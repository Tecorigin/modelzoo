export TORCH_SDAA_LOG_LEVEL=debug
export SDAA_LAUNCH_BLOCKING=1
LOG_FILE="debuglog.log"
python train.py -c config.json > "${LOG_FILE}" 2>&1