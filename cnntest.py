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

gpu_flag = 3

if gpu_flag >= 0:
    cuda.check_cuda_available()
xp = cuda.cupy if gpu_flag >= 0 else np

batchsize = 13
val_batchsize = 6
n_epoch = 1


tate = 125
yoko = 25
with open('RMSpropGraves_model.pkl','rb') as i:
    model=cPickle.load(i)

if gpu_flag >= 0:
    cuda.get_device(gpu_flag).use()
    model.to_gpu()


def forward(x_data, y_data, train=True):
    x, t = chainer.Variable(x_data), chainer.Variable(y_data)
    h = F.max_pooling_2d(F.relu(F.local_response_normalization(model.conv1(x))), 2, stride=2)
    h = F.max_pooling_2d(F.relu(F.local_response_normalization(model.conv2(h))), 2, stride=2)
    h = F.dropout(F.relu(model.fc6(h)))
    h = F.dropout(F.relu(model.fc7(h)))
    h = model.fc8(h)
    if train:

        return F.softmax_cross_entropy(h, t)

    else:
        return h


optimizer=optimizers.RMSpropGraves()
optimizer.setup(model)
start_time = time.clock()

'''
sake_list = np.ndarray((3799, 125, 25), dtype=np.int32)
buri_list = np.ndarray((4600, 125, 25), dtype=np.int32)
iwasi_list = np.ndarray((4600, 125, 25), dtype=np.int32)
ika_list = np.ndarray((4600, 125, 25), dtype=np.int32)
maguro_list = np.ndarray((4599, 125, 25), dtype=np.int32)

sake_ans = np.ndarray(3799, dtype=np.int32)
buri_ans = np.ndarray(4600, dtype=np.int32)
iwasi_ans = np.ndarray(4600, dtype=np.int32)
ika_ans = np.ndarray(4600, dtype=np.int32)
maguro_ans = np.ndarray(4599, dtype=np.int32)

gyosyu_list=[sake_list,buri_list,iwasi_list,ika_list,maguro_list]
gyosyu_ans=[sake_ans,buri_ans,iwasi_ans,ika_ans,maguro_ans]
'''

count =0
error=0



#test_list = np.ndarray((125, 25), dtype=np.float32)
all_data=[]
ans_data=[]

train_data=[]
val_data=[]
gyosyu=5
for al in range(gyosyu):
    insert = 0

    print str(al)+".xls open"
    book = xlrd.open_workbook(str(al)+'.xls')
    sheet_1 = book.sheet_by_index(0)
    for cell in range(4600):
        test_list = np.ndarray((125, 25), dtype=np.float32)
        if int(sheet_1.cell(cell, 0).value)>insert:
            for row in range(cell, cell+25):
                for col in range(7, 132):
                    test_list[col-7][row-cell]=float(sheet_1.cell(row, col).value)
            all_data.append((test_list,int(al)))
            #print test_list,al
            #break
            count+=1
        else:
            error+=1

        insert=int(sheet_1.cell(cell, 0).value)
    count=0
    error=0

np.random.seed(100)#シード値固定
perm=[0,0,0,0,0]
gyosyu_list=[3799,4600,4600,4600,4599]
up=[0,3799,8399,12999,17599]
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


#fp1 = open(str(argvs[1])+str(argvs[2])+"accuracy.txt", "w")
#fp2 = open(str(argvs[1])+str(argvs[2])+"loss.txt", "w")
#fp1.write("epoch\ttest_accuracy\n")
#fp2.write("epoch\ttrain_loss\n")
#np.random.shuffle(all_data)
#np.random.shuffle(train_data)
#np.random.shuffle(val_data)

#fp3 = open("test_list.txt", "w")
fg=[[0 for i in range(5)] for j in range(5)]
f=open("test_list.txt")
data1=f.readlines()
val_list=[]
for r in range(N_test):
    data2=data1[r].split(',')
    data2[1]=data2[1].rstrip('\n')
    val_list.append(int(data2[1]))

print len(val_list)

for epoch in range(1, n_epoch + 1):
    print "epoch: %d" % epoch

    sum_accuracy = 0
    count =0
    for i in range(N_test):
        val_x_batch = np.ndarray((val_batchsize, 1, tate,yoko), dtype=np.float32)
        val_y_batch = np.ndarray((val_batchsize,), dtype=np.int32)
        val_batch_pool = [None] * val_batchsize

        for zz in range(1):
            path, label = all_data[val_list[count]]#, ans_data[val_data[count]]
            val_batch_pool[zz] = path
            val_x_batch[zz] = val_batch_pool[zz]
            val_y_batch[zz] = label
            count += 1
        x_batch = xp.asarray(val_x_batch)
        y_batch = xp.asarray(val_y_batch)
        list1=[]
        acc = forward(x_batch, y_batch, train=False)
        for c in range(len(acc.data[0])):
            list1.append(acc.data[0][c])
        n_ans = 0
        
        for idx, value in enumerate(list1):
            if value == max(list1):
                n_ans = idx
        fg[val_y_batch[0]][n_ans] = fg[val_y_batch[0]][n_ans] + 1
    
    
    print "buri maguro ika iwasi sake"
    print fg
    count = 0

#end_time = time.clock()
#print end_time - start_time


