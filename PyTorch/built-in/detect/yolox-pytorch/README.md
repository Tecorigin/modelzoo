数据集路径：/mnt/nvme1/dataset/datasets/yolox
数据集预处理：
在完成数据集的摆放之后，我们需要利用voc_annotation.py获得训练用的2007_train.txt和2007_val.txt。   
修改voc_annotation.py里面的参数。第一次训练可以仅修改classes_path，classes_path用于指向检测类别所对应的txt。   
与训练权重：
## 文件下载
训练所需的权值可在百度网盘中下载。  
链接: https://pan.baidu.com/s/1bi2UBwwIHES0OpLeyYuBFg    
提取码: f4ni    
VOC数据集下载地址如下，里面已经包括了训练集、测试集、验证集（与测试集一样），无需再次划分：  
链接: https://pan.baidu.com/s/19Mw2u_df_nBzsC2lg20fQA   
提取码: j5ge   

运行：
python train.py








