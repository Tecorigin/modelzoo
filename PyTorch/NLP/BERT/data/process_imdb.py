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
# 从pos以及neg样例中共抽取25000个样本
# imdb_dir = '/path/to/dataset/aclImdb/' # 修改为数据集存放路径
# output_dir = '/path/to/dataset/IMDB/' # 修改为数据集存放路径

imdb_dir = sys.argv[1] # 修改为数据集存放路径
output_dir = sys.argv[2] # 修改为数据集存放路径
train_dir=os.path.join(imdb_dir,'train')
test_dir=os.path.join(imdb_dir,'test')

def file_list(f_dir):
    labels=[];texts=[]
    for label_type in ['neg','pos']:
        dir_name=os.path.join(f_dir,label_type)
        for fname in os.listdir(dir_name):
            if fname[-4:] =='.txt':
                fo=open(os.path.join(dir_name,fname))
                texts.append(fo.read())
                fo.close()
                if label_type=='pos':
                    labels.append(1)
                else:
                    labels.append(0)
    return texts,labels

train_texts,train_labels=file_list(train_dir)
test_texts,test_labels=file_list(test_dir)

random.seed(1)
idx=[i for i in range(len(train_texts))]
random.shuffle(idx)

x=[]    #打乱后的文本列表
y=[]    #打乱后对应的标签列表
#x,y对应评论和标签的列表，已打乱
for id in idx:
    x.append(train_texts[id])
    y.append(train_labels[id])

TRAINSET_SIZE = 25000
TESTSET_SIZE = 25000

train_samples = x[:TRAINSET_SIZE]
train_labels = y[:TRAINSET_SIZE]

test_samples = test_texts[:TESTSET_SIZE]  #测试集不用打乱
test_labels = test_labels[:TESTSET_SIZE]
# print(eval_labels)

path = output_dir
folder = os.path.exists(path)
if not folder:
    os.makedirs(path)

def save_process(output_dir, samples, labels, type):
    datasets, labels = samples, labels
    sentences = []
    punc = '[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\n。！，]'
    for sen in datasets:
        sen = sen.replace('\n', '')
        sen = sen.replace('<br /><br />', ' ')
        sen = sen.replace('\t', '')
        sen = re.sub(punc, '', sen)
        sentences.append(sen)
    
    name = "train"
    if type == 1:
        name = "test"
    elif type == 2:
        name = "dev"
    df = pd.DataFrame({'sentence': sentences, 'label': labels})
    df.to_csv(os.path.join(output_dir, f"{name}.tsv"), index=False, sep='\t')


save_process(output_dir, train_samples, train_labels, 0)
save_process(output_dir, test_samples, test_labels, 2)