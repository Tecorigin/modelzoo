#encoding=utf-8
data_url = r"https://ai-studio-online.bj.bcebos.com/v1/ae11a2e0a4f145269e1385d54a2c9e3b849334b770e44f9cbd09213423f84b3e?responseContentDisposition=attachment%3B%20filename%3DSpleeter%E6%B5%8B%E8%AF%95%E7%9B%B8%E5%85%B3%E6%95%B0%E6%8D%AE.zip&authorization=bce-auth-v1%2F5cfe9a5e1454405eb2a975c43eace6ec%2F2024-06-21T18%3A02%3A55Z%2F-1%2F%2F823ecc63558b0402bf8604889e7536f10aa74a0edacf029ea8a1124dcc5e9354"
from typing import List
import os

track_list: list = ["instrumental.wav", "mixture.wav", "vocal.wav"]

def place_data() -> None:
    os.system("wget \"{}\" --no-check-certificate -O 摘苹果.zip".format(data_url))
    os.system("unzip 摘苹果.zip -d .")
    os.system("rm 摘苹果.zip")

def data_fix() -> None:
    from run_scripts.argument import config
    import paddle
    from tqdm import tqdm
    import librosa
    paddle.set_device("sdaa")
    cmd = config()
    sr = cmd.sample_rate
    dataset_path = cmd.train_dataset
    for name in tqdm(os.listdir(dataset_path)): # 对于每首歌
        data_list: List[paddle.Tensor] = []
        folder_path = os.path.join(dataset_path, name)
        for track in track_list: # 对于每个轨道
            path = os.path.join(folder_path,track)
            data, ori_sr = paddle.audio.load(path,channels_first=True) # data : [2, n]
            data = paddle.to_tensor(librosa.resample(data.numpy(), orig_sr=ori_sr, target_sr=sr))
            data_list.append(data)

        shape = min([i.shape[-1] for i in data_list])
        assert shape > 2
        data_list = [i[:,:shape] for i in data_list]
        for i in range(3): # 对于每个轨道
            data = data_list[i]
            track = track_list[i]
            path = os.path.join(folder_path,track)
            paddle.audio.save(filepath=path, src=data, sample_rate=sr, channels_first=True)

if __name__ == "__main__":
    place_data()
    data_fix()