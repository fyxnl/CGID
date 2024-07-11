import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# from traindata import GivenData
from view_testdata1 import TestData#from test_data_real import TestData
from CGID import fusion_net
import math
import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'

import os
import time
import numpy as np
from Utils import save_image_,PSNR_cal,SSIM_cal,psnr_,ssim_
# from sklearn.metrics.pairwise import rbf_kernel
# from sklearn.metrics.pairwise import cosine_similarity


def PSNR(img1, img2):
    b,_,_,_=img1.shape
    img1=np.clip(img1,0,255)
    img2=np.clip(img2,0,255)
    mse = np.mean((img1/ 255. - img2/ 255.) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
def ensure_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
test_dir=r'G:\dataset\work4ceshi/'
test_store=r'**/'#E:/mengxzh/GP/review/view_test/sots_out/'
expname='**/'
ensure_dir(test_store)
# ots_train_Dehaze_82833_224_AECRNet_3_5_71_828outdoor11,,ots_train_Dehaze_81933_224_AECRNet_3_5_73_819outdoor11_28
device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")#11,
model_dir=r'/**.pk'

device='cuda' if torch.cuda.is_available() else 'cpu'
ckp = torch.load(model_dir,map_location=device)
net = fusion_net().to('cuda')
net = nn.DataParallel(net)
net.load_state_dict(ckp['model'])
net.eval()
batch_size=1
nyuhaze=False
synTest=True
indoor=False
test_loader=DataLoader(TestData(test_dir,synTest,indoor),batch_size=1)
psnr=[]
ssim=[]
start=time.time()

for j,data_ in enumerate(test_loader):
    with torch.no_grad():
        if synTest:
            wsz=32
            test_x,test_img_name= data_
            test_x=test_x.to(device)
            b,c,h,w=test_x.size()
            max_size = 1500 ** 2
            if h * w < max_size:
                try:
                    h_pad= (h // wsz + 1) * wsz - h
                    w_pad = (w // wsz + 1) * wsz - w
                    test_x = torch.cat([test_x, torch.flip(test_x, [2])], 2)[:, :, :h + h_pad, :]
                    test_x = torch.cat([test_x, torch.flip(test_x, [3])], 3)[:, :, :, :w + w_pad]
            # print(test_x.shape)
            # test_x=input
                    predict_y,_,_=net(test_x)
                except:
                    continue
            else:
                down_img = torch.nn.UpsamplingBilinear2d((h // 2, w // 2))(test_x)
                try:
                    h_pad = (h // wsz + 1) * wsz - h
                    w_pad = (w // wsz + 1) * wsz - w
                    test_x = torch.cat([down_img, torch.flip(down_img, [2])], 2)[:, :, :h + h_pad, :]
                    test_x = torch.cat([down_img, torch.flip(down_img, [3])], 3)[:, :, :, :w + w_pad]
                    predict_y, _, _ = net(test_x)
                    predict_y=torch.nn.UpsamplingBilinear2d((h, w))(predict_y)
                except:
                    continue



        else:
            wsz = 32
            test_x, test_img_name = data_
            test_x = test_x.to(device)
            b, c, h, w = test_x.size()
            max_size = 1500 ** 2
            if h * w < max_size:
                try:
                    h_pad = (h // wsz + 1) * wsz - h
                    w_pad = (w // wsz + 1) * wsz - w
                    test_x = torch.cat([test_x, torch.flip(test_x, [2])], 2)[:, :, :h + h_pad, :]
                    test_x = torch.cat([test_x, torch.flip(test_x, [3])], 3)[:, :, :, :w + w_pad]
                    # print(test_x.shape)
                    # test_x=input
                    predict_y, _, _ = net(test_x)
                except:
                    continue
            else:
                down_img = torch.nn.UpsamplingBilinear2d((h // 2, w // 2))(test_x)
                try:
                    h_pad = (h // wsz + 1) * wsz - h
                    w_pad = (w // wsz + 1) * wsz - w
                    test_x = torch.cat([down_img, torch.flip(down_img, [2])], 2)[:, :, :h + h_pad, :]
                    test_x = torch.cat([down_img, torch.flip(down_img, [3])], 3)[:, :, :, :w + w_pad]
                    predict_y, _, _ = net(test_x)
                    predict_y = torch.nn.UpsamplingBilinear2d((h, w))(predict_y)
                except:
                    continue
    save_image_(predict_y,expname,test_img_name)
    # print(j)
end=time.time()-start
print(end)

