"""Get the binarized MNIST dataset and convert to hdf5.
From https://github.com/yburda/iwae/blob/master/datasets.py
"""
import urllib.request
import os
import numpy as np
import h5py
import torch


def parse_binary_mnist(data_dir):
    def lines_to_np_array(lines):
        # 过滤空行并确保每行都有正确的元素数量
        processed_lines = []
        expected_length = 784  # MNIST 图像应该是 28*28=784 像素
        
        for line in lines:
            line = line.strip()
            if not line:  # 跳过空行
                continue
            elements = line.split()
            if len(elements) == expected_length:
                processed_lines.append([int(i) for i in elements])
            # 可选：打印有问题的行用于调试
            # elif len(elements) > 0:
            #     print(f"Skipping line with {len(elements)} elements: {line[:50]}...")
        
        return np.array(processed_lines).astype("float32")

    with open(os.path.join(data_dir, "binarized_mnist_train.amat")) as f:
        lines = f.readlines()
    train_data = lines_to_np_array(lines)
    
    with open(os.path.join(data_dir, "binarized_mnist_valid.amat")) as f:
        lines = f.readlines()
    validation_data = lines_to_np_array(lines)
    
    with open(os.path.join(data_dir, "binarized_mnist_test.amat")) as f:
        lines = f.readlines()
    test_data = lines_to_np_array(lines)
    
    print(f"Train data shape: {train_data.shape}")
    print(f"Validation data shape: {validation_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    
    return train_data, validation_data, test_data


def download_binary_mnist(fname):
    data_dir = "/data/application/huangyun/variational-autoencoder/tmp/"
    subdatasets = ["train", "valid", "test"]
    
    # 检查文件是否已存在，如果存在则跳过下载
    files_exist = True
    for subdataset in subdatasets:
        filename = "binarized_mnist_{}.amat".format(subdataset)
        local_filename = os.path.join(data_dir, filename)
        if not os.path.exists(local_filename):
            files_exist = False
            break
    
    if not files_exist:
        print("Downloading binary MNIST data...")
        for subdataset in subdatasets:
            filename = "binarized_mnist_{}.amat".format(subdataset)
            url = "http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_{}.amat".format(
                subdataset
            )
            local_filename = os.path.join(data_dir, filename)
            urllib.request.urlretrieve(url, local_filename)
    else:
        print("Using existing binary MNIST data files.")

    train, validation, test = parse_binary_mnist(data_dir)

    data_dict = {"train": train, "valid": validation, "test": test}
    f = h5py.File(fname, "w")
    f.create_dataset("train", data=data_dict["train"])
    f.create_dataset("valid", data=data_dict["valid"])
    f.create_dataset("test", data=data_dict["test"])
    f.close()
    print(f"Saved binary MNIST data to: {fname}")


def load_binary_mnist(fname, batch_size, test_batch_size, use_gpu):
    f = h5py.File(fname, "r")
    x_train = f["train"][::]
    x_val = f["valid"][::]
    x_test = f["test"][::]
    train = torch.utils.data.TensorDataset(torch.from_numpy(x_train))
    kwargs = {"num_workers": 4, "pin_memory": True} if use_gpu else {}
    train_loader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=True, **kwargs
    )
    validation = torch.utils.data.TensorDataset(torch.from_numpy(x_val))
    val_loader = torch.utils.data.DataLoader(
        validation, batch_size=test_batch_size, shuffle=False, **kwargs
    )
    test = torch.utils.data.TensorDataset(torch.from_numpy(x_test))
    test_loader = torch.utils.data.DataLoader(
        test, batch_size=test_batch_size, shuffle=False, **kwargs
    )
    return train_loader, val_loader, test_loader
