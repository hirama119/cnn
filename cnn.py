# coding: utf-8
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
gpu_flag = -1

if gpu_flag >= 0:
    cuda.check_cuda_available()
xp = cuda.cupy if gpu_flag >= 0 else np

batchsize = 13
val_batchsize = 6
n_epoch = 20
gyosyu=5


tate = 165
yoko = 25

model = chainer.FunctionSet(conv1=L.Convolution2D(1,  40, 2),
                            conv2=L.Convolution2D(40, 20,  2),
                            fc6=L.Linear(4000, 2048),
                            fc7=L.Linear(2048, 512),
                            fc8=L.Linear(512, 5),
                            )

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


optimizer=optimizers.RMSpropGraves()
optimizer.setup(model)
start_time = time.clock()


count =0
error=0



#test_list = np.ndarray((125, 25), dtype=np.float32)
all_data=[]
ans_data=[]

train_data=[]
val_data=[]
gyosyu_list=[]

for al in range(gyosyu):
    insert = 0

    print str(al)+".xls open"
    book = xlrd.open_workbook(str(al)+'.xls')
    sheet_1 = book.sheet_by_index(0)
    for cell in range(4600):
        test_list = np.ndarray((1,125, 25), dtype=np.uint8)
        if int(sheet_1.cell(cell, 0).value)>insert:
            for row in range(cell, cell+25):
                for col in range(7, 132):
                    test_list[0][col-7][row-cell]=float(sheet_1.cell(row, col).value)

            size = (25, 165)
            resize = cv2.resize(test_list[0], size, interpolation=cv2.INTER_CUBIC)
            all_data.append((resize,int(al)))
            #print test_list,al
            #break
            count+=1
        else:
            error+=1

        insert=int(sheet_1.cell(cell, 0).value)

    gyosyu_list.append(int(count))
    count=0
    error=0

np.random.seed(100)#シード値固定
perm=[0]*gyosyu
#gyosyu_list=[3799,4600,4600,4600,4599]
#up=[0,3799,8399,12999,17599]

up=[]
upp=0
for g in range(gyosyu):
    up.append(upp)
    upp=upp+gyosyu_list[g]


for pe in range(gyosyu):#各魚種のデータ数に応じた乱数作成
#    perm[pe] = np.random.permutation(gyosyu_list[pe])
    perm[pe] = np.arange(gyosyu_list[pe])

for al in range(gyosyu):#データを8:２に分けてる
    indeces = [int(perm[al].size*n) for n in [0.6,0.6+0.2]]
    train, val,train1 = np.split(perm[al], indeces)
    for i,tra in enumerate(train):
        train_data.append(tra+up[al])
    for i,tra in enumerate(train1):
        train_data.append(tra+up[al])
    for i,va in enumerate(val):
        val_data.append(va+up[al])


N = len(train_data)
N_test = len(val_data)
print N,N_test,len(all_data)


fp1 = open("accuracy.txt", "w")
fp2 = open("loss.txt", "w")
fp1.write("epoch\ttest_accuracy\n")
fp2.write("epoch\ttrain_loss\n")
#np.random.shuffle(all_data)
np.random.shuffle(train_data)
np.random.shuffle(val_data)
#for i in range(N, N + N_test):
#    path, label = all_list[perm[i]], ans_data[[perm[count]]]
#    fp3.write("%s" % path)
#    fp3.write(",%s\n" % label)
#    print path
#    fp3.flush()
    #fp3 = open(str(modelname)+str(heikin)"test_list.txt", "w")
for epoch in range(1, n_epoch + 1):
    print "epoch: %d" % epoch

    sum_loss = 0
    count = 0
    for i in range(0, N, batchsize):
        x_batch1 = np.ndarray((batchsize, 1, tate,yoko), dtype=np.float32)
        y_batch1 = np.ndarray((batchsize,), dtype=np.int32)
        batch_pool = [None] * batchsize

        for z in range(batchsize):
            path, label = all_data[train_data[count]]#, ans_data[train_data[count]]
            #print path,label
#            print train_data[count]
            batch_pool[z] = path
            x_batch1[z] = batch_pool[z]
            y_batch1[z] = label
            count += 1

        x_batch = xp.asarray(x_batch1)
        y_batch = xp.asarray(y_batch1)

        optimizer.zero_grads()
        loss = forward(x_batch, y_batch, train=True)
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * len(y_batch)

    print "train mean loss: %f" % (sum_loss / N)
    fp2.write("%d\t%f\n" % (epoch, sum_loss / N))
    fp2.flush()

    sum_accuracy = 0
    count =0
    for i in range(0, N_test, val_batchsize):
        val_x_batch = np.ndarray((val_batchsize, 1, tate,yoko), dtype=np.float32)
        val_y_batch = np.ndarray((val_batchsize,), dtype=np.int32)
        val_batch_pool = [None] * val_batchsize

        for zz in range(val_batchsize):
            path, label = all_data[val_data[count]]#, ans_data[val_data[count]]
            val_batch_pool[zz] = path
            val_x_batch[zz] = val_batch_pool[zz]
            val_y_batch[zz] = label
            count += 1
        x_batch = xp.asarray(val_x_batch)
        y_batch = xp.asarray(val_y_batch)

        acc = forward(x_batch, y_batch, train=False)
        sum_accuracy += float(acc.data) * len(y_batch)
    count = 0

    print "test accuracy: %f" % (sum_accuracy / N_test)
    fp1.write("%d\t%f\n" % (epoch, sum_accuracy / N_test))
    fp1.flush()

#end_time = time.clock()
#print end_time - start_time

#fp3.close()
fp1.close()
fp2.close()
'''
if __name__ == "__main__":
            fp1 = open(str(modelname1[h])+str(i)+"accuracy.txt", "w")
            fp2 = open(str(modelname1[h])+str(i)+"loss.txt", "w")
            fp1.write("epoch\ttest_accuracy\n")
            fp2.write("epoch\ttrain_loss\n")
    modelname=[optimizers.Adam(),optimizers.SGD(),optimizers.RMSpropGraves(),optimizers.RMSprop(),optimizers.AdaDelta(),optimizers.AdaGrad(),optimizers.MomentumSGD(),optimizers.NesterovAG()]
    modelname1=["Adam","SGD","RMSpropGraves","RMSprop","AdaDelta","AdaGrad","MomentumSGD","NesterovAG"]
    for i in range(10):
        for h in range(8):
            a=MyClass(modelname[h],i,modelname1[h])
            #main(modelname[h],i,modelname1[h])
            a.main()
        # CPU環境でも学習済みモデルを読み込めるようにCPUに移してからダンプ
        #model.to_cpu()
        #cPickle.dump(model, open("model.pkl", "wb"), -1)
'''






