import json
import matplotlib.pyplot as plt

def extract_loss_from_log_bart(log_file):
    losses = []  # 存储所有的loss值
    epochs = []
    with open(log_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        try:
            # 尝试将每行作为JSON来解析
            log_data = json.loads(line.strip())
            # 检查是否包含"loss"字段
            if 'loss' in log_data:
                loss_value = float(log_data['loss'])
                losses.append(loss_value)

                epoch = float(log_data['update'])
                epochs.append(epoch)
        except json.JSONDecodeError:
            # 如果遇到不能解析的行，则跳过
            continue
    return epochs,losses
    
def draw_bart():
    # 设置日志文件路径
    log_file = 'fairseq/checkpoints/bart/log.log'  
    # 提取loss
    epochs,losses = extract_loss_from_log_bart(log_file)
    # 绘制loss变化图
    plt.plot(epochs,losses)
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.title('Loss over iterations')
    plt.grid(True)
    plt.savefig('README/BART/loss.png')



def extract_loss_from_log_conv(log_file):
    losses = []  # 存储所有的loss值
    epochs = []
    with open(log_file, 'r') as f:
        lines = f.readlines()
    epoch_last = -1.0
    for line in lines:
        try:
            # 尝试将每行作为JSON来解析
            log_data = json.loads(line.strip())
            # 检查是否包含"loss"字段
            if 'loss' in log_data:
                epoch = float(log_data['num_updates'])
                if abs(epoch-epoch_last) <= 1e-6:
                    continue
                else:
                    epoch_last = epoch
                    epochs.append(epoch)
                    loss_value = float(log_data['loss'])
                    losses.append(loss_value)

                
        except json.JSONDecodeError:
            # 如果遇到不能解析的行，则跳过
            continue
    return epochs,losses

def draw_convolutional():
    log_file = 'fairseq/checkpoints/fconv_wmt_en_de/log.log'  
    # 提取loss
    epochs,losses = extract_loss_from_log_conv(log_file)
    # 绘制loss变化图
    plt.plot(epochs,losses)
    plt.xlabel('Update')
    plt.ylabel('Loss')
    plt.title('Loss over iterations')
    plt.grid(True)
    plt.savefig('README/Convolutional/loss.png')
    plt.close()

def draw_transformer():
    log_file = 'fairseq/checkpoints/transformer_iwslt_de_en/log.log'  
    # 提取loss
    epochs,losses = extract_loss_from_log_conv(log_file)
    # 绘制loss变化图
    plt.plot(epochs,losses)
    plt.xlabel('Update')
    plt.ylabel('Loss')
    plt.title('Loss over iterations')
    plt.grid(True)
    plt.savefig('README/Transformer/loss.png')
    plt.close()

def extract_loss_from_log_roberta(log_file):
    losses = []  # 存储所有的loss值
    epochs = []
    with open(log_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.split(sep=" - ")[-1]
        try:
            # 尝试将每行作为JSON来解析
            log_data = json.loads(line.strip())
            # 检查是否包含"loss"字段
            if 'train_loss' in log_data:
                loss_value = float(log_data['train_loss'])
                losses.append(loss_value)

                epoch = float(log_data['epoch'])
                epochs.append(epoch)
        except json.JSONDecodeError:
            # 如果遇到不能解析的行，则跳过
            continue
    return epochs,losses
    
def draw_roberta():
    # 设置日志文件路径
    log_file = 'fairseq/outputs/2024-11-25/09-01-54/hydra_train.log'  
    # 提取loss
    epochs,losses = extract_loss_from_log_roberta(log_file)
    # 绘制loss变化图
    plt.plot(epochs,losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over iterations')
    plt.grid(True)
    plt.savefig('README/RoBerta/loss.png')

import re
def extract_losses_from_log_esrgan(log_file):
    losses = {
        "iter": [],
        "loss_pix": [],
        "loss_perceptual": [],
        "loss_gan": [],
        "loss_d_real": [],
        "loss_d_fake": []
    }
    
    # 定义一个正则表达式来匹配损失值和迭代次数
    # loss_pattern = r"Iter\(train\).*loss_pix: (\S+) .* loss_perceptual: (\S+) .* loss_gan: (\S+) .* loss_d_real: (\S+) .* loss_d_fake: (\S+)"
    iter_pattern = r"Iter\(train\) \[\s*(\d+)\s*/\s*(\d+)\s*\]"
    loss_pattern = r"loss_pix: (\S+) .* loss_perceptual: (\S+) .* loss_gan: (\S+) .* loss_d_real: (\S+) .* loss_d_fake: (\S+)"
    with open(log_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        # 使用正则表达式提取损失值
        # print(line)
        match_iter = re.search(iter_pattern,line)
        match = re.search(loss_pattern, line)
        if match:
            # print(match.group())
            iter_info = match_iter.group(1)
            losses["iter"].append(int(iter_info.split('/')[0]))

            losses["loss_pix"].append(float(match.groups(0)[0]))
            losses["loss_perceptual"].append(float(match.groups(0)[1]))
            losses["loss_gan"].append(float(match.groups(0)[2]))
            losses["loss_d_real"].append(float(match.groups(0)[3]))
            losses["loss_d_fake"].append(float(match.groups(0)[4]))

    return losses

def draw_esrgan():
    log_file = "mmagic/work_dirs/esrgan_x4c64b23g32_1xb16-400k_div2k/20241203_084222/20241203_084222.log"
    losses = extract_losses_from_log_esrgan(log_file)

    # plt.plot(losses["loss_pix"], label="loss_pix")
    plt.plot(losses["iter"],losses["loss_perceptual"], label="loss_perceptual")
    # plt.plot(losses["iter"],losses["loss_gan"], label="loss_gan")
    # plt.plot(losses["iter"],losses["loss_d_real"], label="loss_d_real")
    # plt.plot(losses["iter"],losses["loss_d_fake"], label="loss_d_fake")
    
    # 添加标签和标题
    plt.xlabel('Iteration')
    plt.ylabel('Perceptual Loss')
    plt.title('Losses over Iterations')
    plt.legend()
    plt.grid(True)
    
    # 保存图像
    plt.savefig('README/ESRGAN/loss.png')

def extract_losses_from_log_liif(log_file):
    losses = {
        "iter": [],
        "loss": [],
    }
    
    # 定义一个正则表达式来匹配损失值和迭代次数
    # loss_pattern = r"Iter\(train\).*loss_pix: (\S+) .* loss_perceptual: (\S+) .* loss_gan: (\S+) .* loss_d_real: (\S+) .* loss_d_fake: (\S+)"
    iter_pattern = r"Iter\(train\) \[\s*(\d+)\s*/\s*(\d+)\s*\]"
    loss_pattern = r"loss: (\S+)"
    with open(log_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        # 使用正则表达式提取损失值
        # print(line)
        match_iter = re.search(iter_pattern,line)
        match = re.search(loss_pattern, line)
        if match:
            iter_info = match_iter.group(1)
            losses["iter"].append(int(iter_info.split('/')[0]))

            losses["loss"].append(float(match.groups(0)[0]))

    return losses

def draw_liif():
    log_file = "train_sdaa_3rd.log"
    losses = extract_losses_from_log_liif(log_file)

    # plt.plot(losses["loss_pix"], label="loss_pix")
    plt.plot(losses["iter"],losses["loss"])
    # plt.plot(losses["iter"],losses["loss_gan"], label="loss_gan")
    # plt.plot(losses["iter"],losses["loss_d_real"], label="loss_d_real")
    # plt.plot(losses["iter"],losses["loss_d_fake"], label="loss_d_fake")
    
    # 添加标签和标题
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Losses over Iterations')
    # plt.legend()
    plt.grid(True)
    
    # 保存图像
    plt.savefig('train_sdaa_3rd.png')

def draw_realbasicvsr():
    log_file = "mmagic/work_dirs/realbasicvsr_c64b20-1x30x8_8xb1-lr5e-5-150k_reds/20250106_094932/20250106_094932.log"
    losses = extract_losses_from_log_esrgan(log_file)

    # plt.plot(losses["iter"],losses["loss_pix"], label="loss_pix")
    # plt.plot(losses["iter"],losses["loss_perceptual"], label="loss_perceptual")
    # plt.plot(losses["iter"],losses["loss_gan"], label="loss_gan")
    plt.plot(losses["iter"],losses["loss_d_real"], label="loss_d_real")
    plt.plot(losses["iter"],losses["loss_d_fake"], label="loss_d_fake")
    
    # 添加标签和标题
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Losses over Iterations')
    plt.legend()
    plt.grid(True)
    
    # 保存图像
    plt.savefig('README/RealBasicVSR/loss.png')

def extract_losses_from_log_bertner(log_file):
    epochs = []
    losses = []
    with open(log_file,'r') as f:
        file = json.load(f)
        for log in file["log_history"]:
            # print(log)
            
            if "loss" in log:
                epochs.append(log["epoch"])
                losses.append(log["loss"])
            # elif "train_loss" in log:
            #     epochs.append(log["epoch"])
            #     losses.append(log["train_loss"])
    return epochs,losses

def draw_bertner():
    log_file = "transformers/tmp/test-ner/trainer_state.json"
    epochs,losses = extract_losses_from_log_bertner(log_file)

    plt.plot(epochs,losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Losses over Iterations') 
    plt.grid(True)
    plt.savefig('README/BERT_NER/loss.png')

def draw_berttextclassification():
    log_file = "transformers/tmp/mrpc/trainer_state.json"
    epochs,losses = extract_losses_from_log_bertner(log_file)

    plt.plot(epochs,losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Losses over Iterations') 
    plt.grid(True)
    plt.savefig('README/BERT_Text_Classification/loss.png')

def draw_qa():
    log_file = "transformers/tmp/qa_squad/checkpoint-8000/trainer_state.json"
    epochs,losses = extract_losses_from_log_bertner(log_file)

    plt.plot(epochs,losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Losses over Iterations') 
    plt.grid(True)
    plt.savefig('README/BERT_Question_Answering/loss.png')



def draw_gru():
    log_file = "fairseq/checkpoints/gru_iwslt_de_en/log.log"
    epochs,losses = extract_loss_from_log_bart(log_file)
    plt.plot(epochs,losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over iterations')
    plt.grid(True)
    plt.savefig('README/GRU/loss.png')
    plt.close()

def draw_lstm():
    log_file = "fairseq/checkpoints/lstm_iwslt_de_en/log.log"
    epochs,losses = extract_loss_from_log_bart(log_file)
    plt.plot(epochs,losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over iterations')
    plt.grid(True)
    plt.savefig('README/LSTM/loss.png')
    plt.close()

def draw_gpt2():
    log_file = "transformers/tmp/test-clm/trainer_state.json"
    epochs,losses = extract_losses_from_log_bertner(log_file)

    plt.plot(epochs,losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Losses over Iterations') 
    plt.grid(True)
    plt.savefig('README/GPT2/loss.png')

def draw_edsr():
    log_file = "train_sdaa_3rd.log"
    losses = extract_losses_from_log_liif(log_file)
    plt.plot(losses["iter"],losses["loss"])
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Losses over Iterations')
    plt.grid(True)
    plt.savefig('train_sdaa_3rd.png')
    plt.close()

if __name__ == "__main__":
    draw_edsr()