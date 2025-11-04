import os
import cv2 as cv
from PIL import Image
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio

from tqdm import tqdm
import argparse

def parse_opt():
    # 设置评估选项
    parser = argparse.ArgumentParser(description='Evaluate options')
    parser.add_argument('--result_path', type=str, default='./results/pyramidpix2pix', help='results saved path')
    parser.add_argument('--lpips_net', type=str, default='vgg', help='lpips model [alex, vgg] (default: vgg)')
    opt = parser.parse_args()
    return opt

opt = parse_opt()

def psnr_ssim_lpips(result_path):
    psnr = []
    ssim = []
    lpips_score = []
    dr ='images'

    for i in tqdm(os.listdir(os.path.join(result_path, dr))):
        if 'fake_B' in i:
            try:
                fake = cv.imread(os.path.join(result_path, dr, i))
                real = cv.imread(os.path.join(result_path, dr, i.replace('fake_B', 'real_B')))
                PSNR = peak_signal_noise_ratio(fake, real)
                psnr.append(PSNR)
                SSIM = structural_similarity(fake, real, channel_axis=2, multichannel=True)
                ssim.append(SSIM)
                #fake_tensor = lpips.im2tensor(fake)
                #real_tensor = lpips.im2tensor(real)
                #score = lpips_loss.forward(fake_tensor, real_tensor).item()
                #lpips_score.append(score)
            except:
                print("there is something wrong with " + i)
        else:
            continue
    average_psnr = sum(psnr) / len(psnr)
    average_ssim = sum(ssim) / len(psnr)
    print("The average PSNR is " + str(average_psnr))
    print("The average SSIM is " + str(average_ssim))


psnr_ssim_lpips(opt.result_path, opt.lpips_net)