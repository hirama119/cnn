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
import cv2


#0サケ、１ブリ、２イワシ、３イカ、４マグロ

gpu_flag = 0

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
        return h

optimizer=optimizers.RMSpropGraves()
optimizer.setup(model)

fp1 = open("miss.txt", "w")

fg = [[0 for i in range(5)] for j in range(5)]


# 訓練ループ
start_time = time.clock()
val_list = []

count =0
error=0
all_data=[]
ans_data=[]
gyosyu_list=[]

up=[]
for al1,al in enumerate(up):
    insert = 0

    print str(al) + "test.csv open"
    data = np.genfromtxt(str(al[0]) + "test.csv", delimiter=",", dtype=np.int32)
    for cell in range(4600):
        test_list = np.ndarray((165, 25), dtype=np.int32)
        if int(data[cell, 0]) > insert:
            for row in range(cell, cell + 25):
                for col in range(7, 172):
                    test_list[col - 7][row - cell] = float(data[row, col])

            all_data.append((test_list, int(al[1])))
            # print test_list,al
            # break
            count += 1
        else:
            error += 1

        insert = int(data[cell, 0])

    gyosyu_list.append(int(count))
    count = 0
    error = 0

up=[(921,1)]
for al1,al in enumerate(up):
    insert = 0

    print str(al) + ".csv open"
    data = np.genfromtxt(str(al[0]) + ".csv", delimiter=",", dtype=np.int32)
    for cell in range(4600):
        test_list = np.ndarray((1,125, 25), dtype=np.uint8)
        if int(data[cell, 0]) > insert:
            for row in range(cell, cell + 25):
                for col in range(7, 132):
                    test_list[0][col - 7][row - cell] = float(data[row, col])
            size = (25,165)
            resize = cv2.resize(test_list[0],size,interpolation=cv2.INTER_CUBIC)
            all_data.append((resize, int(al[1])))
            # print test_list,al
            # break
            count += 1
        else:
            error += 1

        insert = int(data[cell, 0])

    gyosyu_list.append(int(count))
    count = 0
    error = 0
N_test = sum(gyosyu_list)
print N_test
for epoch in range(1, n_epoch + 1):


    print "epoch: %d" % epoch

    count=0

    sum_accuracy = 0
    for i in range(0, N_test, val_batchsize):
        if i%1000==0:
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
        #list = np.ndarray(5,dtype=np.float16)
        list=[]
        for c in range(len(acc.data[0])):
            list.append(acc.data[0][c])
        n_ans = 0
       # print "list"
       # print list
        for idx, value in enumerate(list):
            if value == max(list):
                n_ans = idx



        fg[label][n_ans] = fg[label][n_ans] + 1  # visibledeprecationwarning とでるが、intにfloatがはいっている？ということらしい
       # if label!=n_ans:
        #    fp1.write(str(n_ans))
        #    fp1.write(",")
        #    fp1.write(str(val_list[count-1]))
        #    fp1.write("\n")


        # f[n_ans] = f[n_ans] + 1
    print "サケ ブリ イワシ イカ マグロ"
    print fg
    #fp1.write(str(fg))
    #fp1.flush()
    count=0



end_time = time.clock()
print end_time - start_time

fp1.close()


