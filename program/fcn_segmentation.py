import tensorflow as tf
import numpy as np
import PIL
from PIL import Image,ImageOps
import os
import sys
import time
import matplotlib.pyplot as plt
tf.compat.v1.disable_eager_execution()
tf.debugging.set_log_device_placement(True)
def file_zero(sayi):
        if sayi<10:
                return "00"+str(sayi)
        elif sayi<100:
                return "0"+str(sayi)
        else:
                return str(sayi)
def file_zero2(sayi):
        if sayi<10:
                return "0"+str(sayi)
        else:
                return str(sayi)
def reshape(liste):#(None,channel,height,width) datayı (None ,height,width,channel) dataya çevirir
    shp=liste.shape
    boş_data=np.empty((shp[0],shp[2],shp[3],shp[1]),np.float32)
    index_ornek=0
    for ornek in liste:
        index_resim=0
        for resim in ornek:
            index_height=0
            for i in resim:
                index_width=0
                for q in i:
                    boş_data[index_ornek][index_height][index_width][index_resim]=q/255
                    index_width+=1
                index_height+=1
            index_resim+=1
        index_ornek+=1
    return boş_data
#def conv_1_by_1(x, num_classes, 
#        kernel_regularizer = tf.keras.regularizers.l2(l=0.01),
#        init = tf.compat.v1.truncated_normal_initializer(stddev = 0.01)):
#        return tf.compat.v1.layers.conv2d(x, num_classes, 1,1, padding = 'same', kernel_regularizer =  kernel_regularizer, kernel_initializer = init)
def conv_1_by_1(x,kernel):
        return tf.nn.conv2d(x,kernel, [1, 1, 1, 1], "SAME")
def upsample(x,kernel,stride):
        shapes=x.shape
        ret=tf.nn.conv2d_transpose(x,kernel,[2,shapes[1]*2,shapes[2]*2,shapes[3]],[1,2,2,1],"SAME")
        return ret
width=320
num_classes=3
height=240
kernel_size=3
train_data=np.empty((0,width),np.float32)
konumlar_train=[]
konumlar_iris=[]
konumlar_pupil=[]
for q in range(len(os.listdir("ground_truth"))):
        for i in os.listdir("ground_truth/"+str(q)):
               if "iris" in i:
                    konumlar_iris.append( "ground_truth/"+str(q)+"/"+i)
               elif "pupil" in i:
                    konumlar_pupil.append( "ground_truth/"+str(q)+"/"+i)
               if "image" in i:
                    konumlar_train.append( "ground_truth/"+str(q)+"/"+i)  
for i in konumlar_train:
        train_data=np.append(train_data,np.asarray(PIL.Image.open(i).convert("L").resize((320,240))),axis=0)

#test_data=np.empty((0,width),np.float32)
test_data=np.empty((0,3,height,width),np.float32)
yeni_test=np.empty((1,3,height,width),np.float32)

index=0
for i in konumlar_iris:
        yeni_test[0][0]=np.asarray(PIL.Image.open(konumlar_iris[index]).convert("L").resize((320,240)))
        yeni_test[0][1]=np.asarray(PIL.Image.open(konumlar_pupil[index]).convert("L").resize((320,240)))
        test_data=np.append(test_data,yeni_test,axis=0)
        index+=1
for i in range(1):

    for q in test_data:#q iris ve göz bebeği resimlerini barındırır
      for t in q[:-1]:#t her bir resmi ifade eder
       index=0
       for v in t:
        index2=0
        for i in v:
                if i>128:
                        t[index,index2]=255
                else:
                        t[index,index2]=0
                index2+=1
        index+=1

    #buişlem esnasında hem iris hemde göz bebeği ile eşleşmeyen pixeller ayrı bir class içerisinde tutulur # üçüncü görüntü else durumunu ifade eder
    for q in test_data:
        for i in range(q[0].shape[0]):
                for t in range(q[0].shape[1]):
                        if q[0][i,t]==q[1][i,t] and q[0][i,t]<128:
                                q[2][i,t]=255
    image_reshape=(tf.reshape(train_data,[-1,height,width,1]))/255
    reshaped_test_data=reshape(test_data)
    x=tf.compat.v1.placeholder(tf.float32,[None,height,width,1])
    y=tf.compat.v1.placeholder(tf.float32,[None,height,width,3])
    kernel1=tf.Variable(tf.compat.v1.random_normal([11, 11,1,8], stddev=0.03,dtype=tf.float32),name="kernel1")
    kernel_ara1=tf.Variable(tf.compat.v1.random_normal([5, 5,8,8], stddev=0.03,dtype=tf.float32),name="kernel_ara1")
    kernel_ara2=tf.Variable(tf.compat.v1.random_normal([kernel_size, kernel_size,8,12], stddev=0.03,dtype=tf.float32),name="kernel_ara2")
    kernel_ara3=tf.Variable(tf.compat.v1.random_normal([kernel_size, kernel_size,12,12], stddev=0.03,dtype=tf.float32),name="kernel_ara3")
    kernel2=tf.Variable(tf.compat.v1.random_normal([kernel_size, kernel_size,12,16], stddev=0.03,dtype=tf.float32),name="kernel2")
    kernel_ara4=tf.Variable(tf.compat.v1.random_normal([kernel_size, kernel_size,16,16], stddev=0.03,dtype=tf.float32),name="kernel_ara4")
    kernel_ara5=tf.Variable(tf.compat.v1.random_normal([kernel_size, kernel_size,16,24], stddev=0.03,dtype=tf.float32),name="kernel_ara5")
    kernel_ara6=tf.Variable(tf.compat.v1.random_normal([kernel_size, kernel_size,24,32], stddev=0.03,dtype=tf.float32),name="kernel_ara6")
    kernel3=tf.Variable(tf.compat.v1.random_normal([kernel_size, kernel_size,32,32], stddev=0.03,dtype=tf.float32),name="kernel3")
    kernel_ara7=tf.Variable(tf.compat.v1.random_normal([kernel_size, kernel_size,32,32], stddev=0.03,dtype=tf.float32),name="kernel_ara7")
    kernel_ara8=tf.Variable(tf.compat.v1.random_normal([kernel_size, kernel_size,32,48], stddev=0.03,dtype=tf.float32),name="kernel_ara8")
    kernel_ara9=tf.Variable(tf.compat.v1.random_normal([kernel_size, kernel_size,48,48], stddev=0.03,dtype=tf.float32),name="kernel_ara9")
    kernel4=tf.Variable(tf.compat.v1.random_normal([kernel_size, kernel_size,48,64], stddev=0.03,dtype=tf.float32),name="kernel4")
    kernel5=tf.Variable(tf.compat.v1.random_normal([kernel_size, kernel_size,64,96], stddev=0.03,dtype=tf.float32),name="kernel5")
    kernel6=tf.Variable(tf.compat.v1.random_normal([kernel_size, kernel_size,96,96], stddev=0.03,dtype=tf.float32),name="kernel6")
    kernel7=tf.Variable(tf.compat.v1.random_normal([kernel_size,kernel_size,96,96], stddev=0.03,dtype=tf.float32),name="kernel7")
    kernel8=tf.Variable(tf.compat.v1.random_normal([kernel_size,kernel_size,96,128], stddev=0.03,dtype=tf.float32),name="kernel8")

    ##
    kernel_1x1_1=tf.Variable(tf.compat.v1.random_normal([1, 1,128,3], stddev=0.01,dtype=tf.float32),name="kernel_1x1")
    kernel_1x1_2=tf.Variable(tf.compat.v1.random_normal([1, 1,96,3], stddev=0.01,dtype=tf.float32),name="kernel_1x2")
    kernel_1x1_3=tf.Variable(tf.compat.v1.random_normal([1, 1,64,3], stddev=0.01,dtype=tf.float32),name="kernel_1x3")
    kernel_1x1_4=tf.Variable(tf.compat.v1.random_normal([1, 1,32,3], stddev=0.01,dtype=tf.float32),name="kernel_1x4")
    ##
    kernel_trans_1=tf.Variable(tf.compat.v1.random_normal([4, 4,3,3], stddev=0.01,dtype=tf.float32),name="kernel_trans_1")
    kernel_trans_2=tf.Variable(tf.compat.v1.random_normal([4, 4,3,3], stddev=0.01,dtype=tf.float32),name="kernel_trans_2")
    kernel_trans_3=tf.Variable(tf.compat.v1.random_normal([4, 4,3,3], stddev=0.01,dtype=tf.float32),name="kernel_trans_3")
    kernel_trans_4=tf.Variable(tf.compat.v1.random_normal([4, 4,3,3], stddev=0.01,dtype=tf.float32),name="kernel_trans_4")
    #model
    # 4 adet conv uygulandı
    conv1=tf.nn.conv2d(x, kernel1, [1, 1, 1, 1], "SAME",name="conv1")
    conv1=tf.nn.relu(conv1,name="relu1")
    conv1=tf.nn.conv2d(conv1, kernel_ara1, [1, 1, 1, 1], "SAME",name="conv_ara1")
    conv1=tf.nn.relu(conv1,name="relu_ara1")
    conv1=tf.nn.conv2d(conv1, kernel_ara2, [1, 1, 1, 1], "SAME",name="conv_ara2")
    conv1=tf.nn.relu(conv1,name="relu_ara2")
    conv1=tf.nn.conv2d(conv1, kernel_ara3, [1, 1, 1, 1], "SAME",name="conv_ara3")
    conv1=tf.nn.relu(conv1,name="relu_ara3")
    # 1. max pool
    max_pool=tf.nn.max_pool2d(conv1,[1,2,2,1],[1,2,2,1],"SAME")
    # 4 adet conv uygulandı
    conv2=tf.nn.conv2d(max_pool,  kernel2, [1, 1, 1, 1], "SAME",name="conv2")
    conv2=tf.nn.relu(conv2,name="relu2")
    conv2=tf.nn.conv2d(conv2,  kernel_ara4, [1, 1, 1, 1], "SAME",name="conv_ara4")
    conv2=tf.nn.relu(conv2,name="relu_ara4")
    conv2=tf.nn.conv2d(conv2,  kernel_ara5, [1, 1, 1, 1], "SAME",name="conv_ara5")
    conv2=tf.nn.relu(conv2,name="relu_ara5")
    conv2=tf.nn.conv2d(conv2,  kernel_ara6, [1, 1, 1, 1], "SAME",name="conv_ara6")
    conv2=tf.nn.relu(conv2,name="relu_ara6")
    # 2. max pool
    max_pool=tf.nn.max_pool2d(conv2,[1,2,2,1],[1,2,2,1],"SAME")
    conv3=tf.nn.conv2d(max_pool,  kernel3, [1, 1, 1, 1], "SAME",name="conv3")
    conv3=tf.nn.relu(conv3,name="relu3")
    conv3=tf.nn.conv2d(conv3,  kernel_ara7, [1, 1, 1, 1], "SAME",name="conv_ara7")
    conv3=tf.nn.relu(conv3,name="relu_ara7")
    conv3=tf.nn.conv2d(conv3,  kernel_ara8, [1, 1, 1, 1], "SAME",name="conv_ara8")
    conv3=tf.nn.relu(conv3,name="relu_ara8")
    conv3=tf.nn.conv2d(conv3,  kernel_ara9, [1, 1, 1, 1], "SAME",name="conv_ara9")
    conv3=tf.nn.relu(conv3,name="relu_ara9")
    #max_pool=tf.nn.max_pool2d(conv3,[1,2,2,1],[1,2,2,1],"SAME")
    conv4=tf.nn.conv2d(conv3,  kernel4, [1, 1, 1, 1], "SAME",name="conv4")
    conv4=tf.nn.relu(conv4,name="relu4")
    max_pool=tf.nn.max_pool2d(conv4,[1,2,2,1],[1,2,2,1],"SAME")
    conv5=tf.nn.conv2d(max_pool,  kernel5, [1, 1, 1, 1], "SAME",name="conv5")
    conv5=tf.nn.relu(conv5,name="relu5")
    max_pool=tf.nn.max_pool2d(conv5,[1,2,2,1],[1,2,2,1],"SAME")
    conv6=tf.nn.conv2d(max_pool,  kernel6, [1, 1, 1, 1], "SAME",name="conv6")
    conv6=tf.nn.relu(conv6,name="relu6")
    #max_pool=tf.nn.max_pool2d(conv6,[1,2,2,1],[1,2,2,1],"SAME")
    conv7=tf.nn.conv2d(conv6,  kernel7, [1, 1, 1, 1], "SAME",name="conv7")
    conv7=tf.nn.relu(conv7,name="relu7")
    conv8=tf.nn.conv2d(conv7,  kernel8, [1, 1, 1, 1], "SAME",name="conv8")
    conv8=tf.nn.relu(conv8,name="relu8")

    
    #conv10 15x20
    
    #conv7_1x1 = conv_1_by_1(conv7, num_classes)
    conv5_1x1 = conv_1_by_1(conv5, kernel_1x1_2)
    conv4_1x1 = conv_1_by_1(conv4, kernel_1x1_3)
    #conv3_1x1 = conv_1_by_1(conv3, num_classes)
    conv2_1x1 = conv_1_by_1(conv2, kernel_1x1_4)
    conv9_1x1 = conv_1_by_1(conv8, kernel_1x1_1)
    #upsample l7 by 2
    #skip1=tf.add(conv9_1x1,conv7_1x1)
    l9_upsample = upsample(conv9_1x1,kernel_trans_1, 2)
    
    #add skip connection from  l4_1x1    
    l9l5_skip = tf.add(l9_upsample, conv5_1x1)

    #implement the another transposed convolution layer
    l9l5_upsample = upsample(l9l5_skip,kernel_trans_2, 2)
    #l7l4_upsample = tf.layers.batch_normalization(l7l4_upsample)

    l9l5l3_skip = tf.add(l9l5_upsample, conv4_1x1)
    last_up=upsample(l9l5l3_skip,kernel_trans_3, 2)
    
    l9l5l3l2_skip = tf.add(last_up, conv2_1x1)
    last_up=upsample(l9l5l3l2_skip,kernel_trans_4, 2)

    y_ = tf.nn.softmax(last_up)
    y_clipped = tf.clip_by_value(y_, 1e-10, 0.9999999)
    learning_rate=tf.compat.v1.placeholder(tf.float32,[])
    loss = tf.reduce_mean(-(tf.reduce_sum(y * tf.math.log(y_clipped)
                                    + (1 - y) * tf.math.log(1 - y_clipped), axis=3)))
    #compute_grad=tf.compat.v1.train.AdamOptimizer(learning_rate=0.1).compute_gradients(y_,[kernel1,kernel2,kernel3,kernel4])
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)
init = tf.compat.v1.global_variables_initializer()
saver = tf.compat.v1.train.Saver(save_relative_paths=True)
learn=0.0001
first=True
first2=True
epoch=10
with tf.compat.v1.Session() as sess:
    sess.run(init)
    i=0
    while i<epoch:
        if i%20==0 and i!=0:
                time.sleep(120)
        data=sess.run(image_reshape)

        size=data.shape[0]
        batch_size=2
        
        cost=0
        for q in range(int(data.shape[0]/batch_size)):
                _,c=sess.run([optimizer,loss],feed_dict={x:data[q*batch_size:(q+1)*batch_size],y:reshaped_test_data[q*batch_size:(q+1)*batch_size],learning_rate:learn})
                cost+=c/int(data.shape[0]/batch_size)

        print("epoch:",i,"cost:",cost)
        i+=1
        if cost<0.2 and first==True:
                learn=learn/10
                first=False

        if cost<0.03 and first2==True:
                learn=learn/10
                first2=False

        if i==epoch-1:
                girdi=input("devam etsin mi : ")
                if girdi=="e":
                        break
                else:
                        try:
                                epoch=epoch+int(girdi)
                        except:
                                girdi=input()
                                epoch=epoch+int(girdi)
    save_path = saver.save(sess, "fcn_model/başarı1.ckpt",global_step=1)
