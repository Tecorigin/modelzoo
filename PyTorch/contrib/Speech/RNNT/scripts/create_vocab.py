from transformers import AutoTokenizer

# 加载预训练的 tokenizer 和词汇表
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# 获取词汇表
vocab = tokenizer.get_vocab()

# 将词汇表保存到文件
vocab_path = 'vocab.txt'
with open(vocab_path, 'w') as f:
    for word in vocab.keys():
        f.write(word + '\n')