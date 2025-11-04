import re
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def parse_log(log_file):
    pattern = re.compile(r'Test set: Average loss: ([\d.]+), Accuracy: \d+/10000 \(([\d.]+)%\)')
    datasets = []
    current_data = []

    with open(log_file, 'r') as f:
        for line in f:
            if 'Test set: Average loss:' in line:
                match = pattern.search(line)
                if match:
                    loss = float(match.group(1))
                    acc = float(match.group(2))
                    current_data.append((loss, acc))
            elif line.startswith('Best accuracy:'):
                if current_data:
                    datasets.append(current_data)
                    current_data = []

    if current_data:
        datasets.append(current_data)

    return datasets[:3]  # Ensure we only get 3 datasets


def plot_curves(datasets, output_file):
    titles = ['Baseline', 'Sparsity (1e-4)', 'Fine-tune-160(70%)']
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, (ax, data) in enumerate(zip(axes, datasets)):
        epochs = list(range(len(data)))
        losses = [d[0] for d in data]
        accuracies = [d[1] for d in data]

        # 设置紧凑的x轴
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))

        # 绘制loss曲线
        loss_line = ax.plot(epochs, losses, 'b-', label='Loss', linewidth=2)[0]
        ax.set_xlabel('Epoch', fontsize=10)
        ax.set_ylabel('Loss', color='b', fontsize=10)
        ax.tick_params(axis='y', labelcolor='b', labelsize=8)
        ax.tick_params(axis='x', labelsize=8, rotation=45)
        ax.grid(True, alpha=0.3)

        # 绘制accuracy曲线
        ax2 = ax.twinx()
        acc_line = ax2.plot(epochs, accuracies, 'r-', label='Accuracy', linewidth=2)[0]
        ax2.set_ylabel('Accuracy (%)', color='r', fontsize=10)
        ax2.tick_params(axis='y', labelcolor='r', labelsize=8)
        ax2.set_ylim(40, 100)  # 固定accuracy范围

        # 统一y轴范围
        ax.set_ylim(0, max(losses) * 1.1)

        # 添加图例
        lines = [loss_line, acc_line]
        ax.legend(lines, ['Loss', 'Accuracy'],
                  loc='lower right' if idx == 2 else 'upper right',
                  fontsize=8,
                  framealpha=0.9)

        ax.set_title(titles[idx], fontsize=12, pad=12)

    plt.tight_layout(pad=2.0)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    log_data = parse_log('train_sdaa_3rd.log')
    plot_curves(log_data, 'train_sdaa_3rd.png')