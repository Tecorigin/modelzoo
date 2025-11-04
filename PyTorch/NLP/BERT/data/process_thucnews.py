# BSD 3- Clause License Copyright (c) 2023, Tecorigin Co., Ltd. All rights
# reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software
# without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY
# WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
# OF SUCH DAMAGE.

import os
import random
import pandas as pd
import re
import sys
vocab=[]  #词表
"""
将原始的文件处理成 评论列表和标签列表
"""

# input_dir = '/path-to-dataset/THUCNews' # 修改为数据集存放路径
# output_dir = '/path-to-dataset/processed_THUCNews' # 修改为数据集存放路径
input_dir = sys.argv[1] # 修改为数据集存放路径
output_dir = sys.argv[2] # 修改为数据集存放路径
train_dir=input_dir


categories = ['财经', '彩票', '房产', '股票', '家居', '教育', '科技', 
              '社会', '时尚', '时政', '体育', '星座', '游戏', '娱乐']

def file_list(f_dir):
    labels=[];texts=[]
    for label_type in categories:
        dir_name=os.path.join(f_dir,label_type)
        file_cnt = 0
        for fname in os.listdir(dir_name):
            if fname[-4:] =='.txt':
                fo=open(os.path.join(dir_name,fname))
                texts.append(fo.read())
                fo.close()
                print('read file ' + label_type + ' ' + fname)
                if label_type=='财经':
                    labels.append(0)
                elif label_type=='彩票':
                    labels.append(1)
                elif label_type=='房产':
                    labels.append(2)
                elif label_type=='股票':
                    labels.append(3)
                elif label_type=='家居':
                    labels.append(4)
                elif label_type=='教育':
                    labels.append(5)
                elif label_type=='科技':
                    labels.append(6)
                elif label_type=='社会':
                    labels.append(7)
                elif label_type=='时尚':
                    labels.append(8)
                elif label_type=='时政':
                    labels.append(9)
                elif label_type=='体育':
                    labels.append(10)
                elif label_type=='星座':
                    labels.append(11)
                elif label_type=='游戏':
                    labels.append(12)
                else:
                    labels.append(13)
            file_cnt += 1
            if file_cnt == 5000:
                break
    return texts,labels

train_texts,train_labels=file_list(train_dir)
random.seed(1)
idx=[i for i in range(len(train_texts))]
random.shuffle(idx)

x=[]    #打乱后的文本列表
y=[]    #打乱后对应的标签列表
#x,y对应评论和标签的列表，已打乱
for id in idx:
    x.append(train_texts[id])
    y.append(train_labels[id])

print(train_texts[-1:],train_labels[-1:])
TNEWS_SIZE = len(train_texts)
print(TNEWS_SIZE)

TRAINSET_SIZE = int(TNEWS_SIZE * 0.9)
DEVSET_SIZE = TNEWS_SIZE - TRAINSET_SIZE

train_samples = x[:TRAINSET_SIZE]
train_labels = y[:TRAINSET_SIZE]

dev_samples = x[TRAINSET_SIZE:TRAINSET_SIZE+DEVSET_SIZE]
dev_labels = y[TRAINSET_SIZE:TRAINSET_SIZE+DEVSET_SIZE]
print(len(dev_labels))

path = output_dir
folder = os.path.exists(path)
if not folder:
    os.makedirs(path)

def save_process(output_dir, samples, labels, type):
    datasets, labels = samples, labels
    sentences = []
    punc = '[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\n。！，]'
    for sen in datasets:
        # sen = sen.replace('\n', '')
        # sen = sen.replace('<br /><br />', ' ')
        sen = sen.replace('\t', '')
        sen = sen.replace('\n', '')
        # sen = sen.replace('\u3000', '')
        # sen = re.sub(punc, '', sen)
        sentences.append(sen)
    
    name = "train"
    if type == 1:
        name = "dev"
    df = pd.DataFrame({'sentence': sentences, 'label': labels})
    df.to_csv(os.path.join(output_dir, f"{name}.tsv"), index=False, sep='\t') # 修改为您的数据集存放路径

save_process(output_dir, train_samples, train_labels, 0)
save_process(output_dir, dev_samples, dev_labels, 1)
