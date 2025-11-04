#encoding=utf-8
urls = r'''https://ai-studio-online.bj.bcebos.com/v1/88a1e77d205f4b4da85ff528527940736fbaddca4ff24902a37ddbfa5c796afd?responseContentDisposition=attachment%3B%20filename%3Dmusdb18_all1.zip&authorization=bce-auth-v1%2F5cfe9a5e1454405eb2a975c43eace6ec%2F2023-07-05T10%3A43%3A53Z%2F-1%2F%2Ffab0ac83821ac97ff25d35c9b92707c4d2d21f10cff2f8b23ded2c75883fc0af
https://ai-studio-online.bj.bcebos.com/v1/3a4c73a91650440ea56707c34bfe2d4ebcc6304d03274107a1cdb83fa6016b50?responseContentDisposition=attachment%3B%20filename%3Dmusdb18_all2.zip&authorization=bce-auth-v1%2F5cfe9a5e1454405eb2a975c43eace6ec%2F2023-07-05T11%3A06%3A52Z%2F-1%2F%2F4e7af474f40bcba173bbaf59d7cd65a632f821044dafdcbe415d550721783ae2
https://ai-studio-online.bj.bcebos.com/v1/92c083de551944ce9967c11f469c75638c760a9e13544f85b2c43ee7513783c6?responseContentDisposition=attachment%3B%20filename%3Dmusdb18_all3.zip&authorization=bce-auth-v1%2F5cfe9a5e1454405eb2a975c43eace6ec%2F2023-07-05T13%3A41%3A48Z%2F-1%2F%2Fae51482cb731342c7bab6ea7f91f22a168fa21016f2f092b4fd4a612ab770e51
https://ai-studio-online.bj.bcebos.com/v1/b8f7d9e9143548b7b8bdf87e7c596482c484b2e9bf7a43e985cc94ffb5a04418?responseContentDisposition=attachment%3B%20filename%3Dmusdb18_all4.zip&authorization=bce-auth-v1%2F5cfe9a5e1454405eb2a975c43eace6ec%2F2023-07-05T10%3A22%3A34Z%2F-1%2F%2F8cd054ecab7a61d2a92ff4736a4f47f067e88b55dea87b4687f24b97c7c4bd4a
https://ai-studio-online.bj.bcebos.com/v1/6f0aee44e8ef4e71b05b756081ae5ab321df2df7b51f44818ce0f063f0308d28?responseContentDisposition=attachment%3B%20filename%3Dmusdb18_all5.zip&authorization=bce-auth-v1%2F5cfe9a5e1454405eb2a975c43eace6ec%2F2023-07-05T09%3A34%3A09Z%2F-1%2F%2Fee422d04dffa3f8d01da7a87eb86eaf9127d1b079be50ff76f48e467b4e6409a'''.split('\n')
import os
from place_data import place_data, data_fix
if __name__ == "__main__":
    count = 1
    usage = 5
    for i in range(usage):#range(len(urls)):
        i = urls[i]
        os.system("wget \"{}\" --no-check-certificate -O dashichang{}.zip".format(i,count))
        count += 1
    for i in range(1,usage+1):
        os.system(f"unzip dashichang{i}.zip -d dataset")
        os.system(f"rm dashichang{i}.zip")

    place_data()
    data_fix()