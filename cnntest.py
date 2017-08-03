<<<<<<< HEAD
<<<<<<< HEAD
# coding: utf-8
=======
#coding: utf-8
>>>>>>> 6f804c952fd3a869a45b3416636c33c249f0c62e
=======
#coding: utf-8
>>>>>>> 9316e2f5cc21765b5c44f38aa3dd82a7c5cb8b6d
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
<<<<<<< HEAD
<<<<<<< HEAD
import xlrd

gpu_flag = 3
=======


#0サケ、１ブリ、２イワシ、３イカ、４マグロ

gpu_flag = -1
>>>>>>> 6f804c952fd3a869a45b3416636c33c249f0c62e
=======
import xlrd
import cv2
#0サケ、１ブリ、２イワシ、３イカ、４マグロ

gpu_flag = -1
>>>>>>> 9316e2f5cc21765b5c44f38aa3dd82a7c5cb8b6d

if gpu_flag >= 0:
    cuda.check_cuda_available()
xp = cuda.cupy if gpu_flag >= 0 else np

<<<<<<< HEAD
<<<<<<< HEAD
batchsize = 13
val_batchsize = 6
n_epoch = 1


tate = 125
yoko = 25
with open('RMSpropGraves_model.pkl','rb') as i:
    model=cPickle.load(i)
=======
=======
>>>>>>> 9316e2f5cc21765b5c44f38aa3dd82a7c5cb8b6d
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
<<<<<<< HEAD
>>>>>>> 6f804c952fd3a869a45b3416636c33c249f0c62e
=======
>>>>>>> 9316e2f5cc21765b5c44f38aa3dd82a7c5cb8b6d

if gpu_flag >= 0:
    cuda.get_device(gpu_flag).use()
    model.to_gpu()

<<<<<<< HEAD
<<<<<<< HEAD

def forward(x_data, y_data, train=True):
    x, t = chainer.Variable(x_data), chainer.Variable(y_data)
    h = F.max_pooling_2d(F.relu(F.local_response_normalization(model.conv1(x))), 2, stride=2)
    h = F.max_pooling_2d(F.relu(F.local_response_normalization(model.conv2(h))), 3, stride=2)
=======
=======
>>>>>>> 9316e2f5cc21765b5c44f38aa3dd82a7c5cb8b6d
def forward(x_data, y_data, train=True):
    x, t = chainer.Variable(x_data), chainer.Variable(y_data)
    h = F.max_pooling_2d(F.relu(
        F.local_response_normalization(model.conv1(x))), 2,stride=2)
    h = F.max_pooling_2d(F.relu(
        F.local_response_normalization(model.conv2(h))), 3,stride=2)
<<<<<<< HEAD
>>>>>>> 6f804c952fd3a869a45b3416636c33c249f0c62e
=======
>>>>>>> 9316e2f5cc21765b5c44f38aa3dd82a7c5cb8b6d
    h = F.dropout(F.relu(model.fc6(h)))
    h = F.dropout(F.relu(model.fc7(h)))
    h = model.fc8(h)
    if train:

        return F.softmax_cross_entropy(h, t)

    else:
<<<<<<< HEAD
<<<<<<< HEAD
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

=======
        return F.accuracy(h, t)

=======
        return h

>>>>>>> 9316e2f5cc21765b5c44f38aa3dd82a7c5cb8b6d
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


<<<<<<< HEAD
for al in range(2):
    insert = 0

    print str(al)+"test.xls open"
    book = xlrd.open_workbook(str(al)+'test.xls')
=======
op=(1,3)
for al1,al in enumerate(op):
    insert = 0

    print str(al)+"test.xls open"
    book = xlrd.open_workbook(str(al)+'.xls')
>>>>>>> 9316e2f5cc21765b5c44f38aa3dd82a7c5cb8b6d
    sheet_1 = book.sheet_by_index(0)
    for cell in range(4600):
        test_list = np.ndarray((1,125, 25), dtype=np.uint8)

        if int(sheet_1.cell(cell, 0).value)>insert:
            for col in range(7, 132):
                for row in range(cell, cell+25):
                    test_list[0][col-7][row-cell]=int(sheet_1.cell(row, col).value)
            size = (25,165)
<<<<<<< HEAD
            resize=cv2.resize(test_list,size[0], interpolation = cv2.INTER_CUBIC)
=======
            resize=cv2.resize(test_list[0],size, interpolation = cv2.INTER_CUBIC)
>>>>>>> 9316e2f5cc21765b5c44f38aa3dd82a7c5cb8b6d
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

<<<<<<< HEAD
        for zz in range(val_batchsize):
            path, label = alldata[count]
            val_batch_pool[zz] = path
            val_x_batch[zz]=val_batch_pool[zz]
            val_y_batch[zz] = label
            count+=1
=======
        path, label = all_data[count]
        val_batch_pool = path
        val_x_batch = val_batch_pool
        val_y_batch = label
        count+=1
>>>>>>> 9316e2f5cc21765b5c44f38aa3dd82a7c5cb8b6d
        x_batch = xp.asarray(val_x_batch)
        y_batch = xp.asarray(val_y_batch)

        acc = forward(x_batch, y_batch, train=False)
        list = []
<<<<<<< HEAD

        for c in range(len(acc.data[0])):
            list.append(acc.data[0][c])
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
=======
        print acc.data
        for c in range(len(acc.data[0])):
            list.append(acc.data[c])
        n_ans = 0

        for idx, value in enumerate(list):
            if value == max(list):
                n_ans = idx



        fg[label][n_ans] = fg[label][n_ans] + 1  # visibledeprecationwarning とでるが、intにfloatがはいっている？ということらしい
        if label!=n_ans:
>>>>>>> 9316e2f5cc21765b5c44f38aa3dd82a7c5cb8b6d
            fp1.write(str(n_ans))
            fp1.write(",")
            fp1.write(str(val_list[count-1]))
            fp1.write("\n")
<<<<<<< HEAD
        '''
=======

>>>>>>> 9316e2f5cc21765b5c44f38aa3dd82a7c5cb8b6d

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
<<<<<<< HEAD
>>>>>>> 6f804c952fd3a869a45b3416636c33c249f0c62e
=======
>>>>>>> 9316e2f5cc21765b5c44f38aa3dd82a7c5cb8b6d

