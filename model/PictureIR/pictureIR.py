import numpy as np
import pandas as pd
import os
import pickle
import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import random

def picIR():
    # 各种文件路径...
    df_path = r'F:\desktop\各种课件\经管\大四上\大数据分析技术\大作业\data_df.csv'
    pic_path = r'F:\desktop\各种课件\经管\大四上\大数据分析技术\大作业\图片'
    model_params_path = r'F:\desktop\各种课件\经管\大四上\大数据分析技术\大作业\ImageAE_params.pth'

    input_path = input()
    number = int(input())

    df = pd.read_csv(df_path, engine='python', encoding='utf-8')

    class MyModel(nn.Module):
        def __init__(self):
            super(MyModel, self).__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(3, 6, 20),
                nn.Tanh(),
                nn.MaxPool2d(4, 4),
                nn.Conv2d(6, 16, 20),
                nn.Tanh(),
                nn.MaxPool2d(4, 4),
            )
            self.img_fc = nn.Sequential(
                nn.Linear(64, 32),
                nn.Tanh(),
                nn.Linear(32, 32),
                nn.Tanh(),
            )
            self.img_decoder = nn.Sequential(
                nn.Linear(32, 32),
                nn.Tanh(),
                nn.Linear(32, 64),   
            )
        
        def forward(self, x):
            img_feature = self.conv(x)
            img_feature = img_feature.view(-1, 64)
            img_embedding = self.img_fc(img_feature)
            img_decode = self.img_decoder(img_embedding)
            return img_feature, img_embedding, img_decode

    testmodel = MyModel()
    testmodel.load_state_dict(torch.load(model_params_path)) 
    testmodel.eval()

    with open('F:\desktop\各种课件\经管\大四上\大数据分析技术\大作业\pic_vec_lis.txt', 'rb') as f:
        pic_vec_lis = pickle.load(f)

    top_lis = []
    totensor = transforms.Compose([transforms.Resize((128,128)), transforms.ToTensor()])
    with torch.no_grad():
        test_img = Image.open(input_path).convert('RGB')
        test_img = totensor(test_img).unsqueeze(dim=0)
        _, test_vec, _ = testmodel(test_img.float())
        test_vec /= (test_vec.square().sum()**0.5)
        test_vec = test_vec.squeeze(dim=0)
        for p in pic_vec_lis:
            top_lis.append([p[0], torch.dot(test_vec, p[1].squeeze(dim=0))])

    top_lis.sort(key= lambda x: x[1], reverse=True)
    res_lis = []
    for i in range(number):
        for p in ['21a', '161', '162', '239']:
            if os.path.exists(pic_path + '\\' + p + '\\' + top_lis[i][0]):
                res_lis.append(pic_path + '\\' + p + '\\' + top_lis[i][0])

    return res