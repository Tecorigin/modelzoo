import os
import yaml
f_dir = "exp/u2++_conformer/"

content_list = []
for f_name in os.listdir(f_dir):
    if "yaml" in f_name and "epoch" in f_name:
        f_path = os.path.join(f_dir, f_name)
        with open(f_path, 'r') as f:
            config_data = yaml.safe_load(f)
        acc = config_data['loss_dict']['acc']
        loss = config_data['loss_dict']['loss']
        content = f"acc:{acc} loss:{loss}"
        content_list.append(content)
log_name = "../../../scripts/train_sdaa_3rd.log"

with open(log_name, 'w') as file:
        for item in content_list:
            file.write(f"{item}\n")
        