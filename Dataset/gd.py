import os
import numpy as np
from PIL import Image

PATH = 'AID'

dirs = os.listdir(PATH)

label = 0
xtrain = []
xtest = []
ytrain = []
ytest = []

'''读取图片数据并处理成300x300的灰度图数据'''
for dir in dirs:
    print(dir)
    img_list = os.listdir(PATH+'/'+dir)
    count = 1
    for img_name in img_list:
        if img_name[-3:] == 'jpg':
            img = Image.open(PATH+'/'+dir+'/'+img_name).convert('L')
            if img.size == (600, 600):
                rs_img = img.resize((300, 300), Image.ANTIALIAS)
                rs_img = np.asarray(rs_img)
                if count <= 120:
                    xtrain.append(rs_img.reshape((1, 90000)).tolist()[0])
                    ytrain.append(label)
                    count += 1
                elif 120 < count <= 150:
                    xtest.append(rs_img.reshape((1, 90000)).tolist()[0])
                    ytest.append(label)
                    count += 1
                else:
                    break
    print(label)
    label += 1

'''对训练集数据打乱顺序'''
arr = np.arange((len(xtrain)))

np.random.shuffle(arr)

s_xtrain = []
s_ytrain = []

for a in arr:
    s_xtrain.append(xtrain[a])
    s_ytrain.append(ytrain[a])
    print(s_ytrain.__len__())

'''将数据存储为npz文件'''
np.savez('300x300Data.npz', x_train=s_xtrain, y_train=s_ytrain, x_test=xtest, y_test=ytest)
