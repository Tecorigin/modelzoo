import matplotlib.pyplot as plt

def read_log_file(file_path):
    with open(file_path, 'r') as file:
        return file.readlines()

def parse_logs(lines):
    epochs = {}
    for line in lines:
        parts = line.split()
        epoch = int(parts[0].split(':')[1])
        d_loss_a = float(parts[2].split(':')[1])
        d_loss_b = float(parts[3].split(':')[1])
        
        if epoch not in epochs:
            epochs[epoch] = {'D_loss_A': [], 'D_loss_B': []}
        epochs[epoch]['D_loss_A'].append(d_loss_a)
        epochs[epoch]['D_loss_B'].append(d_loss_b)
    return epochs

def calculate_means(epochs):
    return {epoch: {'D_loss_A': sum(values['D_loss_A'])/len(values['D_loss_A']),
                    'D_loss_B': sum(values['D_loss_B'])/len(values['D_loss_B'])} 
            for epoch, values in epochs.items()}

def plot_loss(epochs_means):
    epochs_list = list(epochs_means.keys())
    d_loss_a_means = [epochs_means[epoch]['D_loss_A'] for epoch in epochs_list]
    d_loss_b_means = [epochs_means[epoch]['D_loss_B'] for epoch in epochs_list]

    plt.figure(figsize=(10, 5))
    plt.plot(epochs_list, d_loss_a_means, label='D_loss_A', marker='o')
    plt.plot(epochs_list, d_loss_b_means, label='D_loss_B', marker='x')
    plt.title('Average D_loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('train_loss.png')

# 设置你的日志文件路径
log_file_path = 'train_sdaa_3rd.log'  # 请替换为你的实际文件路径

# 读取并处理日志数据
lines = read_log_file(log_file_path)
epochs_data = parse_logs(lines)
epochs_means = calculate_means(epochs_data)

# 绘制图形
plot_loss(epochs_means)