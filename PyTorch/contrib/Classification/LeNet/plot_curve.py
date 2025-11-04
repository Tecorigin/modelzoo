import matplotlib.pyplot as plt
import numpy as np

# 读取训练结果
epochs, losses, accuracies = [], [], []
with open("train_results.txt", "r") as f:
    for line in f:
        epoch, loss, acc = line.strip().split(",")
        epochs.append(int(epoch))
        losses.append(float(loss))
        accuracies.append(float(acc))

# 绘制 Loss 曲线
plt.figure(figsize=(10,5))
plt.plot(epochs, losses, label="Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.legend()
plt.savefig("training_loss.png")

# 绘制 Accuracy 曲线
plt.figure(figsize=(10,5))
plt.plot(epochs, accuracies, label="Training Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.title("Training Accuracy Curve")
plt.legend()
plt.savefig("training_accuracy.png")

plt.show()
