import matplotlib.pyplot as plt
import json
import os


log_file_name = '../log.jsonl'

img_save_dir = '../plot_figures'
os.makedirs(img_save_dir, exist_ok=True)
loss_curve_name = 'loss_curve.jpg'
# acc_curve_name = 'accuracy_curve.jpg'

iterations = []
loss = []
# acc = []

with open(log_file_name, 'r') as f:
	lines = f.readlines()
	for i, line in enumerate(lines):
		line = line.strip()
		tmp_json = json.loads(line)
		# iterations.append(tmp_json['Iteration'])
		iterations.append(i)
		loss.append(tmp_json['Loss'])
		# acc.append(tmp_json['Acc'])

plt.figure()
plt.plot(iterations, loss, label='Train Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Curve')
plt.savefig(os.path.join(img_save_dir, loss_curve_name))

# plt.figure()
# plt.plot(iterations, acc, label='Train Accuracy')
# plt.xlabel('Iterations')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.title('Accuracy Curve')
# plt.savefig(os.path.join(img_save_dir, acc_curve_name))
