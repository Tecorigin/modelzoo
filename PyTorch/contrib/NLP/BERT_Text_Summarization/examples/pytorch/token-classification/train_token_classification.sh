# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
export SDAA_VISIBLE_DEVICES=1,2
export HF_ENDPOINT=https://hf-mirror.com
python -m torch.distributed.launch --nproc_per_node=2 examples/pytorch/token-classification/run_ner.py \
  --model_name_or_path google-bert/bert-base-uncased \
  --dataset_name conll2003/conll2003.py \
  --output_dir tmp/test-ner \
  --do_train \
  --do_eval \
  --trust_remote_code True \
  --overwrite_output_dir \
  --fp16
