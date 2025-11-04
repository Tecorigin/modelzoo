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

export TASK_NAME=ner
export SDAA_VISIBLE_DEVICES=1
export HF_ENDPOINT=https://hf-mirror.com
python examples/pytorch/token-classification/run_ner_no_trainer.py \
  --model_name_or_path google-bert/bert-base-cased \
  --dataset_name conll2003/conll2003.py \
  --task_name $TASK_NAME \
  --max_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir tmp/$TASK_NAME/
