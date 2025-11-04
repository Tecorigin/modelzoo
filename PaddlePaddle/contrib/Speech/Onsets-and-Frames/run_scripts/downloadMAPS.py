url = r'''https://ai-studio-online.bj.bcebos.com/v1/fae18ff3b4fc4d528f9ad6025ed1bf76634ad0cf9fb746c7a839216cad6047c9?responseContentDisposition=attachment%3B%20filename%3Dofall.zip&authorization=bce-auth-v1%2F5cfe9a5e1454405eb2a975c43eace6ec%2F2024-05-27T14%3A02%3A15Z%2F-1%2F%2F0a4b68c6edfe154d372ec383d5666c4cd97ee70186a4d088c0901bab52119561'''

if __name__ == "__main__":
    import os
    # 下载ai studio上面的项目
    os.system(f"wget \"{url}\" --no-check-certificate -O of.zip")
    # 解压项目压缩包
    os.system("unzip of.zip -d of")
    # 准备datas文件夹
    if not os.path.isdir("datas"):
        # 没有就创建文件夹
        os.system("mkdir datas")
    # 移动数据文件夹，不会覆盖已有的文件夹
    os.system("mv of/datas/MAPS datas")
    # 创建模型文件夹
    if not os.path.isdir("model"):
        os.system("mkdir model")
    # 移动预训练模型
    os.system("mv of/model-300000.pdparams model/model-300000.pdparams")
    # 删除ai studio项目文件夹
    os.system("rm -rf of")
    # 删除ai studio项目压缩包
    os.system("rm of.zip")