#coding: utf-8
import numpy as np
import chainer
from chainer import cuda
import chainer.functions as F
from chainer import optimizers
import time
import pylab
import matplotlib.pyplot as plt
import chainer
import chainer.links as L
import os
from PIL import Image
import cPickle
import xlrd
import cv2
#0サケ、１ブリ、２イワシ、３イカ、４マグロ

gpu_flag = 2

if gpu_flag >= 0:
    cuda.check_cuda_available()
xp = cuda.cupy if gpu_flag >= 0 else np

val_batchsize=1
n_epoch = 1
tate=165
yoko=25
with open('model.pkl', 'rb') as i:
    model = cPickle.load(i)

# ピクセルの値を0.0-1.0に正規化

# 訓練データとテストデータに分割


# 画像を (nsample, channel, height, width) の4次元テンソルに変換
# MNISTはチャンネル数が1なのでreshapeだけでOK

# plt.imshow(X_train[1][0], cmap=pylab.cm.gray_r, interpolation='nearest')
# plt.show()
#, stride=1,pad=2

if gpu_flag >= 0:
    cuda.get_device(gpu_flag).use()
    model.to_gpu()

def forward(x_data, y_data, train=True):
    x, t = chainer.Variable(x_data), chainer.Variable(y_data)
    h = F.max_pooling_2d(F.relu(
        F.local_response_normalization(model.conv1(x))), 2,stride=2)
    h = F.max_pooling_2d(F.relu(
        F.local_response_normalization(model.conv2(h))), 3,stride=2)
    h = F.dropout(F.relu(model.fc6(h)))
    h = F.dropout(F.relu(model.fc7(h)))
    h = model.fc8(h)
    if train:

        return F.softmax_cross_entropy(h, t)

    else:
        return F.accuracy(h, t)

optimizer = optimizers.Adam()
optimizer.setup(model)

#fp1 = open("miss.txt", "w")

fg = [[0 for i in range(5)] for j in range(5)]


# 訓練ループ
start_time = time.clock()
val_list = []

count =0
error=0
all_data=[]
ans_data=[]
gyosyu_list=[]


op=(1,3)
for al1,al in enumerate(op):
    insert = 0

    print str(al)+"test.xls open"
    book = xlrd.open_workbook(str(al)+'test.xls')
    sheet_1 = book.sheet_by_index(0)
    for cell in range(4600):
        test_list = np.ndarray((1,125, 25), dtype=np.uint8)

        if int(sheet_1.cell(cell, 0).value)>insert:
            for col in range(7, 132):
                for row in range(cell, cell+25):
                    test_list[0][col-7][row-cell]=int(sheet_1.cell(row, col).value)
            size = (25,165)
            resize=cv2.resize(test_list[0],size, interpolation = cv2.INTER_CUBIC)
            all_data.append((resize,int(al)))
            #gyosyu_ans[al][count] = int(al)
            count+=1
        else:
            error+=1
        insert=int(sheet_1.cell(cell, 0).value)
    print count,error
    gyosyu_list.append(int(count))
    count=0
    error=0



N_test = gyosyu_list[0]+gyosyu_list[1]

for epoch in range(1, n_epoch + 1):


    print "epoch: %d" % epoch

    count=0

    sum_accuracy = 0
    for i in range(0, N_test, val_batchsize):
        print i
        val_x_batch = np.ndarray(
            (val_batchsize, 1, tate, yoko), dtype=np.float32)
        val_y_batch = np.ndarray((val_batchsize,), dtype=np.int32)
        val_batch_pool = [None] * val_batchsize

        for zz in range(val_batchsize):
            path, label = all_data[count]
            val_batch_pool[zz] = path
            val_x_batch[zz]=val_batch_pool[zz]
            val_y_batch[zz] = label
            count+=1
        x_batch = xp.asarray(val_x_batch)
        y_batch = xp.asarray(val_y_batch)

        acc = forward(x_batch, y_batch, train=False)
        list = []
        print acc.data
        for c in range(len(acc.data[0])):
            list.append(acc.data[c])
        n_ans = 0

        #for idx, value in enumerate(list):
            #if value == max(list):
                #n_ans = idx
        if y_batch == 0:
            for i in range(5):
                buri.write(str(list[i]))
                buri.write(",")
            buri.write("\n")
            buri.flush()

        if y_batch == 1:
            for i in range(5):
                maguro.write(str(list[i]))
                maguro.write(",")
            maguro.write("\n")
            maguro.flush()


        if y_batch == 2:
            for i in range(5):
                ika.write(str(list[i]))
                ika.write(",")
            ika.write("\n")
            ika.flush()

        if y_batch == 3:
            for i in range(5):
                iwasi.write(str(list[i]))
                iwasi.write(",")
            iwasi.write("\n")
            iwasi.flush()

        if y_batch == 4:
            for i in range(5):
                sake.write(str(list[i]))
                sake.write(",")
            sake.write("\n")
            sake.flush()



        '''
        fg[y_batch][n_ans] = fg[y_batch][n_ans] + 1  # visibledeprecationwarning とでるが、intにfloatがはいっている？ということらしい
        if y_batch!=n_ans:
            fp1.write(str(n_ans))
            fp1.write(",")
            fp1.write(str(val_list[count-1]))
            fp1.write("\n")
        '''

        # f[n_ans] = f[n_ans] + 1
    print "buri maguro ika iwasi sake"
    #print fg
    #fp1.write(str(fg))
    #fp1.flush()
    count=0



end_time = time.clock()
print end_time - start_time

#fp1.close()
sake.close()
buri.close()
iwasi.close()
ika.close()
maguro.close()

