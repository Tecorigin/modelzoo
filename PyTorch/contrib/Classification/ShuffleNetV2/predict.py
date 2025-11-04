import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from model import shufflenet_v2_x1_5 as create_model


def main():
    # device = torch.device(f"sdaa:{0}")
    device = torch.device("cpu")
    img_size = 224
    data_transform = transforms.Compose(
        [transforms.Resize(256),
                                   transforms.CenterCrop(img_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # load image
    img_path = "/mnt/nvme/common/train_dataset/mini-imagenet/mini_image_classification/mini_train/n04604644/n04604644_1758.JPEG"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)


    # create model
    model = create_model(num_classes=10).to(device)
    # load model weights
    model_weight_path = "./weights/best_model.pth"
    weights_dict = torch.load(model_weight_path, map_location=device)
    model.load_state_dict(weights_dict, strict=False)
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = "class: {}   prob: {:.3}".format(str(predict_cla),
                                                 predict[predict_cla].numpy())
    print(print_res)
    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3}".format(str(i),
                                                  predict[i].numpy()))


if __name__ == '__main__':
    main()
