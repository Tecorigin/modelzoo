pip install -r requirements.txt
cd ../s0

sh run_u2++_conformer.sh --stage 0 --stop_stage 0 # AIShell-1数据组织成两个文件：wav.scp和text（位于s0/data/目录下）

sh run_u2++_conformer.sh --stage 1 --stop_stage 1

sh run_u2++_conformer.sh --stage 2 --stop_stage 2

sh run_u2++_conformer.sh --stage 3 --stop_stage 3

sh run_u2++_conformer.sh --stage 4 --stop_stage 4

# 制作log文件
python3 create_log.py