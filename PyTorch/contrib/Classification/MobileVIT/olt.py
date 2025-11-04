import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

# 加载TensorBoard日志数据
ea = event_accumulator.EventAccumulator('./runs/').Reload()

# 获取损失和准确率数据
train_loss = ea.Scalars('train_loss')
train_acc = ea.Scalars('train_acc')
val_loss = ea.Scalars('val_loss')
val_acc = ea.Scalars('val_acc')

# 提取epoch和对应的值
epochs = [x.step for x in train_loss]
train_loss_values = [x.value for x in train_loss]
train_acc_values = [x.value for x in train_acc]
val_loss_values = [x.value for x in val_loss]
val_acc_values = [x.value for x in val_acc]

# 绘制损失曲线
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss_values, label='Train Loss')
plt.plot(epochs, val_loss_values, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Curves')
plt.legend()

# 绘制准确率曲线
plt.subplot(1, 2, 2)
plt.plot(epochs, train_acc_values, label='Train Accuracy')
plt.plot(epochs, val_acc_values, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy Curves')
plt.legend()

plt.tight_layout()

# 保存为jpg格式
plt.savefig('training_validation_curves.jpg', format='jpg')

plt.show()
