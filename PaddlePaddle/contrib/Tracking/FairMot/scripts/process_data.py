import os
path = "MOT17/images/train"
wrt_path = "image_lists/mot17.train"
with open(wrt_path, 'w') as f:
    for cls_name in os.listdir(path):
        if "MOT17" in cls_name:
            cls_path = os.path.join(path, cls_name)+"/"+"img1"
            for img_name in os.listdir(cls_path):
                img_path = os.path.join(cls_path, img_name)
                f.write(img_path+"\n")
            
        