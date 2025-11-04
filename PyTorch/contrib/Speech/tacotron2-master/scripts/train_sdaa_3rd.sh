#!/bin/bash
cd ..

python -m multiproc train.py --output_directory=outdir --log_directory=logdir --hparams=distributed_run=True,use_amp=True
