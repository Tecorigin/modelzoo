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
    log_file = "mmagic/work_dirs/liif-edsr-norm_c64b16_1xb16-1000k_div2k/20241204_013519/20241204_013519.log"
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
    plt.savefig('README/LIIF/loss.png')

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
    log_file = "mmagic/work_dirs/edsr_x4c64b16_1xb16-300k_div2k/20250209_023427/20250209_023427.log"
    losses = extract_losses_from_log_liif(log_file)
    plt.plot(losses["iter"],losses["loss"])
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Losses over Iterations')
    plt.grid(True)
    plt.savefig('README/EDSR/loss.png')
    plt.close()

def draw_t5():
    log_file = "transformers/tmp/test-translation/checkpoint-47500/trainer_state.json"
    epochs,losses = extract_losses_from_log_bertner(log_file)

    plt.plot(epochs,losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Losses over Iterations') 
    plt.grid(True)
    plt.savefig('README/T5/loss.png')
    plt.close()
    
import ast
def extract_losses_from_log_summ(log_file):
    losses = []  # 存储所有的loss值
    epochs = []
    with open(log_file, 'r') as f:
        lines = f.readlines()
    for line in lines:
        try:
            # 尝试将每行作为JSON来解析
            python_dict = ast.literal_eval(line)  # 转换为 Python 字典
            json_str = json.dumps(python_dict)   # 转换为 JSON 字符串
            log_data = json.loads(json_str)
            # log_data = json.loads(line.strip())
            # 检查是否包含"loss"字段
            if 'loss' in log_data:
                epoch = float(log_data['epoch'])

                epochs.append(epoch)
                loss_value = float(log_data['loss'])
                losses.append(loss_value)

                
        except Exception as e:
            # 如果遇到不能解析的行，则跳过
            continue
    return epochs,losses

def draw_summ():
    log_file = "transformers/tmp/test-summarization/checkpoint-5500/trainer_state.json"
    epochs,losses = extract_losses_from_log_bertner(log_file)

    plt.plot(epochs,losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Losses over Iterations') 
    plt.grid(True)
    plt.savefig('README/BERT_Text_Summarization/loss.png')
    plt.close()

def extract_losses_from_log_realsr(log_file):
    pattern = r"""
        (?P<date>\d{2}-\d{2}-\d{2})\s+          # 日期 (25-03-05)
        (?P<time>\d{2}:\d{2}:\d{2}\.\d{3})\s+   # 时间 (09:36:46.396)
        -\s+
        (?P<level>\w+):\s+                      # 日志级别 (INFO)
        <epoch:\s+(?P<epoch>\d+),\s+            # epoch (0)
        iter:\s+(?P<iter>\d{1,3}(,\d{3})*),\s+               # iter (20)
        lr:(?P<lr>[\d\.e+-]+)>\s+               # lr (1.000e-04)
        l_g_pix:\s+(?P<l_g_pix>[\d\.e+-]+)\s+   # l_g_pix (5.1971e-03)
        l_g_fea:\s+(?P<l_g_fea>[\d\.e+-]+)\s+   # l_g_fea (2.7953e+00)
        l_g_gan:\s+(?P<l_g_gan>[\d\.e+-]+)\s+   # l_g_gan (6.0978e-02)
        l_d_real:\s+(?P<l_d_real>[\d\.e+-]+)\s+ # l_d_real (1.8487e-05)
        l_d_fake:\s+(?P<l_d_fake>[\d\.e+-]+)\s+ # l_d_fake (1.1683e-04)
        D_real:\s+(?P<D_real>[\d\.e+-]+)\s+     # D_real (2.2987e+00)
        D_fake:\s+(?P<D_fake>[\d\.e+-]+)        # D_fake (-9.8968e+00)
    """
    
    # 初始化存储列表
    dates, times, levels, epochs, iters, lrs = [], [], [], [], [], []
    l_g_pixs, l_g_feas, l_g_gans = [], [], []
    l_d_reals, l_d_fakes, D_reals, D_fakes = [], [], [], []
    
    # 打开日志文件并逐行解析
    with open(log_file, 'r') as file:
        for line in file:
            match = re.match(pattern, line.strip(), re.VERBOSE)
            if match:
                # 提取匹配的字段
                
                dates.append(match.group('date'))
                times.append(match.group('time'))
                levels.append(match.group('level'))
                epochs.append(int(match.group('epoch')))
                iters.append(int(match.group('iter').replace(",","")))
                lrs.append(float(match.group('lr')))
                l_g_pixs.append(float(match.group('l_g_pix')))
                l_g_feas.append(float(match.group('l_g_fea')))
                l_g_gans.append(float(match.group('l_g_gan')))
                l_d_reals.append(float(match.group('l_d_real')))
                l_d_fakes.append(float(match.group('l_d_fake')))
                D_reals.append(float(match.group('D_real')))
                D_fakes.append(float(match.group('D_fake')))
    return iters,l_g_pixs,l_g_feas,l_g_gans,l_d_reals,l_d_fakes,D_reals,D_fakes


def draw_realsr():
    log_file = "train_sdaa_3rd.log"
    iters,l_g_pixs,l_g_feas,l_g_gans,l_d_reals,l_d_fakes,D_reals,D_fakes = extract_losses_from_log_realsr(log_file)

    plt.plot(iters,l_g_pixs,label="l_g_pixs")
    plt.plot(iters,l_g_feas,label="l_g_feas")
    plt.plot(iters,l_g_gans,label="l_g_gans")
    plt.plot(iters,l_d_reals,label="l_d_reals")
    plt.plot(iters,l_d_fakes,label="l_d_fakes")
    plt.plot(iters,D_reals,label="D_reals")
    plt.plot(iters,D_fakes,label="D_fakes")
    
    # 添加标签和标题
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Losses over Iterations')
    plt.legend()
    plt.grid(True)
    
    # 保存图像
    plt.savefig('train_sdaa_3rd.png')



if __name__ == "__main__":
    draw_realsr()