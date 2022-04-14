import tensorflow as tf
import numpy as np
import PIL
from PIL import Image,ImageOps
import os
import sys
import time
import matplotlib.pyplot as plt
import gc

def conv_1_by_1(x,kernel):
        return tf.nn.conv2d(x,kernel, [1, 1, 1, 1], "SAME")
def upsample(x,kernel,batch_size):
        shapes=x.shape
        return tf.nn.conv2d_transpose(x,kernel,[batch_size,shapes[1]*2,shapes[2]*2,shapes[3]],[1,2,2,1],"SAME")

class fcn_segmentation(object):
        def __init__(self):
                tf.compat.v1.disable_eager_execution()
                tf.debugging.set_log_device_placement(True)
                self.restorer()
                self.y_=self.structure()
                #self.saver = tf.compat.v1.train.Saver()
                
        def clear(self):
                tf.compat.v1.reset_default_graph()
        #@profile
        def run(self,images,parameter_type="resim"):
                if parameter_type=="resim":

                        kontrol_img=np.asarray(images[0].resize((320,240)))
                        resized_images=np.zeros((len(images),kontrol_img.shape[0],kontrol_img.shape[1]),np.uint8)
                        data_size=len(images)
                        for i in range(len(images)):
                                resized_images[i]=np.asarray(images[i].resize((320,240)))
                elif parameter_type=="konum":
                        kontrol_img=np.asarray(PIL.Image.open(images[0]).convert("L").resize((320,240)))
                        resized_images=np.zeros((len(images),kontrol_img.shape[0],kontrol_img.shape[1]),np.uint8)
                        data_size=len(images)
                        for i in range(len(images)):
                                resized_images[i]=np.asarray(PIL.Image.open(images[i]).convert("L").resize((320,240)))
                batch_size=50
                artan=len(resized_images)-(int(len(resized_images)/batch_size)*batch_size)
                images=[]
                width=320
                num_classes=3
                height=240
                kernel_size=3
                #received data is reshaped to [-1,width,height,1]
                trans_feed=tf.compat.v1.placeholder(tf.float32,[None,None,None,None])
                batch_x=tf.compat.v1.placeholder(tf.float32,[None,kontrol_img.shape[0],kontrol_img.shape[1]])
                resize_place=tf.compat.v1.placeholder(tf.float32,[None,None,None,None])
                image_reshape=(tf.reshape(batch_x,[-1,height,width,1]))/255
                data_index=0
                
                irises=np.empty((data_size,360,480),np.uint8)
                pupils=np.empty((data_size,360,480),np.uint8)
                #irises=np.empty((data_size,480,640),np.uint8)
                #pupils=np.empty((data_size,480,640),np.uint8)
                veri=np.empty((batch_size,height,width,3),np.float32)
                
                adim_sayisi=int(data_size/batch_size)
                resize_images=tf.image.resize(self.y_*255,[360,480])
                trans=tf.transpose(resize_images,perm=[0,3,1,2])
                image_reshaped=resized_images.reshape((-1,height,width,1))
                with tf.compat.v1.Session() as self.sess:

                        feed={  self.kernel1_place:self.kernel1,
                                self.kernel_ara1_place:self.kernel_ara1,
                                self.kernel_ara2_place:self.kernel_ara2,
                                self.kernel_ara3_place:self.kernel_ara3,
                                self.kernel2_place:self.kernel2,
                                self.kernel_ara4_place:self.kernel_ara4,
                                self.kernel_ara5_place:self.kernel_ara5,
                                self.kernel_ara6_place:self.kernel_ara6,
                                self.kernel3_place:self.kernel3,
                                self.kernel_ara7_place:self.kernel_ara7,
                                self.kernel_ara8_place:self.kernel_ara8,
                                self.kernel_ara9_place:self.kernel_ara9,
                                self.kernel4_place:self.kernel4,
                                self.kernel5_place:self.kernel5,
                                self.kernel6_place:self.kernel6,
                                self.kernel7_place:self.kernel7,
                                self.kernel8_place:self.kernel8,
                                self.kernel9_place:self.kernel9,
                                self.kernel10_place:self.kernel10,
                                self.k1x1_1:self.kernel_1x1_1,
                                self.k1x1_2:self.kernel_1x1_2,
                                self.k1x1_3:self.kernel_1x1_3,
                                self.k1x1_4:self.kernel_1x1_4,
                                self.trans_1:self.kernel_trans_1,
                                self.trans_2:self.kernel_trans_2,
                                self.trans_3:self.kernel_trans_3,
                                self.trans_4:self.kernel_trans_4}
                        print("fcn başlangıç:",time.localtime())
                        for i in range(adim_sayisi):
                                feed[self.x]=image_reshaped[i*batch_size:(i+1)*batch_size]
                                feed[self.batch_size]=batch_size
                                #print("hesaplama başlangıç:",time.localtime())
                                veri=self.sess.run(trans,feed_dict=feed)
                                #print("hesaplama bitiş:",time.localtime())
                                veri2 = veri.astype(np.uint8)
                                for i in range(veri.shape[0]):
                                        
                                        #image=Image.fromarray(veri2[i][0])
                                        #image2=Image.fromarray(veri2[i][1])
                                        #image.save("C:/Users/ibrahim eyyüp inan/Desktop/segmented/fcn/iris "+str(data_index)+".jpg")
                                        #image2.save("C:/Users/ibrahim eyyüp inan/Desktop/segmented/fcn/pupil "+str(data_index)+".jpg")
                                        irises[data_index]=veri2[i][0]
                                        pupils[data_index]=veri2[i][1]
                                        data_index+=1

                                veri=[]
                                veri2=[]
                        if artan>0:
                                feed[self.x]=image_reshaped[adim_sayisi*batch_size:]
                                feed[self.batch_size]=artan
                                veri=self.sess.run(trans,feed_dict=feed)
                                veri2 = veri.astype(np.uint8)
                                for i in range(veri.shape[0]):                                        
                                        #image=Image.fromarray(veri2[i][0])
                                        #image2=Image.fromarray(veri2[i][1])
                                        #image.save("C:/Users/ibrahim eyyüp inan/Desktop/segmented/fcn/iris "+str(data_index)+".jpg")
                                        #image2.save("C:/Users/ibrahim eyyüp inan/Desktop/segmented/fcn/pupil "+str(data_index)+".jpg")
                                        irises[data_index]=veri2[i][0]
                                        pupils[data_index]=veri2[i][1]
                                        data_index+=1
                                veri=[]
                                veri2=[]
                        print("fcn sonu:",time.localtime())
                        #input()
                        self.sess.close()
                image_reshaped=None
                del image_reshaped

                gc.collect()
                resized_images=[]
                #irises=[]
                #pupils=[]
                #tf.compat.v1.reset_default_graph()
                return irises,pupils#iris,pupil
        #@profile
        def restorer(self):
                kernel_size=3
                kernel1=tf.Variable(tf.compat.v1.random_normal([kernel_size, kernel_size,1,8], stddev=0.03),name="kernel1")
                kernel_ara1=tf.Variable(tf.compat.v1.random_normal([kernel_size, kernel_size,8,8], stddev=0.03),name="kernel_ara1")
                kernel_ara2=tf.Variable(tf.compat.v1.random_normal([kernel_size, kernel_size,8,12], stddev=0.03),name="kernel_ara2")
                kernel_ara3=tf.Variable(tf.compat.v1.random_normal([kernel_size, kernel_size,12,12], stddev=0.03),name="kernel_ara3")
                kernel2=tf.Variable(tf.compat.v1.random_normal([kernel_size, kernel_size,12,16], stddev=0.03),name="kernel2")
                kernel_ara4=tf.Variable(tf.compat.v1.random_normal([kernel_size, kernel_size,16,16], stddev=0.03),name="kernel_ara4")
                kernel_ara5=tf.Variable(tf.compat.v1.random_normal([kernel_size, kernel_size,16,24], stddev=0.03),name="kernel_ara5")
                kernel_ara6=tf.Variable(tf.compat.v1.random_normal([kernel_size, kernel_size,24,32], stddev=0.03),name="kernel_ara6")
                kernel3=tf.Variable(tf.compat.v1.random_normal([kernel_size, kernel_size,32,32], stddev=0.03),name="kernel3")
                kernel_ara7=tf.Variable(tf.compat.v1.random_normal([kernel_size, kernel_size,32,32], stddev=0.03),name="kernel_ara7")
                kernel_ara8=tf.Variable(tf.compat.v1.random_normal([kernel_size, kernel_size,32,48], stddev=0.03),name="kernel_ara8")
                kernel_ara9=tf.Variable(tf.compat.v1.random_normal([kernel_size, kernel_size,48,48], stddev=0.03),name="kernel_ara9")
                kernel4=tf.Variable(tf.compat.v1.random_normal([kernel_size, kernel_size,48,64], stddev=0.03),name="kernel4")
                kernel5=tf.Variable(tf.compat.v1.random_normal([kernel_size, kernel_size,64,96], stddev=0.03),name="kernel5")
                kernel6=tf.Variable(tf.compat.v1.random_normal([kernel_size, kernel_size,96,96], stddev=0.03),name="kernel6")
                kernel7=tf.Variable(tf.compat.v1.random_normal([kernel_size,kernel_size,96,96], stddev=0.03),name="kernel7")
                kernel8=tf.Variable(tf.compat.v1.random_normal([kernel_size,kernel_size,96,128], stddev=0.03),name="kernel8")
                kernel9=tf.Variable(tf.compat.v1.random_normal([kernel_size,kernel_size,128,196], stddev=0.03),name="kernel9")
                kernel10=tf.Variable(tf.compat.v1.random_normal([kernel_size,kernel_size,196,256], stddev=0.03),name="kernel10")
                ##
                kernel_1x1_1=tf.Variable(tf.compat.v1.random_normal([1, 1,128,3], stddev=0.01),name="kernel_1x1")
                kernel_1x1_2=tf.Variable(tf.compat.v1.random_normal([1, 1,96,3], stddev=0.01),name="kernel_1x2")
                kernel_1x1_3=tf.Variable(tf.compat.v1.random_normal([1, 1,64,3], stddev=0.01),name="kernel_1x3")
                kernel_1x1_4=tf.Variable(tf.compat.v1.random_normal([1, 1,32,3], stddev=0.01),name="kernel_1x4")
                ##
                kernel_trans_1=tf.Variable(tf.compat.v1.random_normal([4, 4,3,3], stddev=0.01),name="kernel_trans_1")
                kernel_trans_2=tf.Variable(tf.compat.v1.random_normal([4, 4,3,3], stddev=0.01),name="kernel_trans_2")
                kernel_trans_3=tf.Variable(tf.compat.v1.random_normal([4, 4,3,3], stddev=0.01),name="kernel_trans_3")
                kernel_trans_4=tf.Variable(tf.compat.v1.random_normal([4, 4,3,3], stddev=0.01),name="kernel_trans_4")
                #self.kernel1=tf.constant(kernel1)
                restore_list=[kernel_1x1_1,kernel_1x1_2,kernel_1x1_3,kernel_1x1_4,kernel_trans_1,kernel_trans_2,kernel_trans_3,
                              kernel_trans_4,kernel1,kernel_ara1,kernel_ara2,kernel_ara3,kernel2,kernel_ara4,kernel_ara5,kernel_ara6,kernel3,
                              kernel_ara7,kernel_ara8,kernel_ara9,kernel4,kernel5,kernel6,kernel7,kernel8,kernel9,kernel10]
                self.saver = tf.compat.v1.train.Saver(restore_list)
                self.init = tf.compat.v1.global_variables_initializer()             
                with tf.compat.v1.Session() as self.sess:
                        self.sess.run(self.init)
                        self.saver.restore(self.sess,tf.train.latest_checkpoint('fcn_model/'))

                        self.kernel1=(kernel1.eval().astype(np.float32))
                        #print("self.kernel1 shape:",self.kernel1.shape)
                        #print("self.kernel1 size:",sys.getsizeof(self.kernel1))
                        self.kernel_ara1=(kernel_ara1.eval().astype(np.float32))
                        self.kernel_ara2=(kernel_ara2.eval().astype(np.float32))
                        self.kernel_ara3=(kernel_ara3.eval().astype(np.float32))
                        self.kernel2=(kernel2.eval().astype(np.float32))
                        self.kernel_ara4=(kernel_ara4.eval().astype(np.float32))
                        self.kernel_ara5=(kernel_ara5.eval().astype(np.float32))
                        self.kernel_ara6=(kernel_ara6.eval().astype(np.float32))
                        self.kernel3=(kernel3.eval().astype(np.float32))
                        self.kernel_ara7=(kernel_ara7.eval().astype(np.float32))
                        self.kernel_ara8=(kernel_ara8.eval().astype(np.float32))
                        self.kernel_ara9=(kernel_ara9.eval().astype(np.float32))
                        self.kernel4=(kernel4.eval().astype(np.float32))
                        self.kernel5=(kernel5.eval().astype(np.float32))
                        self.kernel6=(kernel6.eval().astype(np.float32))
                        self.kernel7=(kernel7.eval().astype(np.float32))
                        self.kernel8=(kernel8.eval().astype(np.float32))
                        self.kernel9=(kernel9.eval().astype(np.float32))
                        self.kernel10=(kernel10.eval().astype(np.float32))

                        self.kernel_1x1_1=(kernel_1x1_1.eval().astype(np.float32))
                        self.kernel_1x1_2=(kernel_1x1_2.eval().astype(np.float32))
                        self.kernel_1x1_3=(kernel_1x1_3.eval().astype(np.float32))
                        self.kernel_1x1_4=(kernel_1x1_4.eval().astype(np.float32))
                        self.kernel_trans_1=kernel_trans_1.eval().astype(np.float32)
                        self.kernel_trans_2=kernel_trans_2.eval().astype(np.float32)
                        self.kernel_trans_3=kernel_trans_3.eval().astype(np.float32)
                        self.kernel_trans_4=kernel_trans_4.eval().astype(np.float32)
                #print("restorer sonu")
                #input()
        #@profile
        def structure(self):
                width=320
                num_classes=3
                height=240
                kernel_size=3
                self.kernel1_place=tf.compat.v1.placeholder(tf.float32,[11,11,1,8])
                self.kernel_ara1_place=tf.compat.v1.placeholder(tf.float32,[5,5,8,8])
                self.kernel_ara2_place=tf.compat.v1.placeholder(tf.float32,[kernel_size,kernel_size,8,12])
                self.kernel_ara3_place=tf.compat.v1.placeholder(tf.float32,[kernel_size,kernel_size,12,12])
                self.kernel2_place=tf.compat.v1.placeholder(tf.float32,[kernel_size,kernel_size,12,16])
                self.kernel_ara4_place=tf.compat.v1.placeholder(tf.float32,[kernel_size,kernel_size,16,16])
                self.kernel_ara5_place=tf.compat.v1.placeholder(tf.float32,[kernel_size,kernel_size,16,24])
                self.kernel_ara6_place=tf.compat.v1.placeholder(tf.float32,[kernel_size,kernel_size,24,32])
                self.kernel3_place=tf.compat.v1.placeholder(tf.float32,[kernel_size,kernel_size,32,32])
                self.kernel_ara7_place=tf.compat.v1.placeholder(tf.float32,[kernel_size,kernel_size,32,32])
                self.kernel_ara8_place=tf.compat.v1.placeholder(tf.float32,[kernel_size,kernel_size,32,48])
                self.kernel_ara9_place=tf.compat.v1.placeholder(tf.float32,[kernel_size,kernel_size,48,48])
                self.kernel4_place=tf.compat.v1.placeholder(tf.float32,[kernel_size,kernel_size,48,64])
                self.kernel5_place=tf.compat.v1.placeholder(tf.float32,[kernel_size,kernel_size,64,96])
                self.kernel6_place=tf.compat.v1.placeholder(tf.float32,[kernel_size,kernel_size,96,96])
                self.kernel7_place=tf.compat.v1.placeholder(tf.float32,[kernel_size,kernel_size,96,96])
                self.kernel8_place=tf.compat.v1.placeholder(tf.float32,[kernel_size,kernel_size,96,128])
                self.kernel9_place=tf.compat.v1.placeholder(tf.float32,[kernel_size,kernel_size,128,196])
                self.kernel10_place=tf.compat.v1.placeholder(tf.float32,[kernel_size,kernel_size,196,256])
                self.k1x1_1=tf.compat.v1.placeholder(tf.float32,[1,1,256,3])
                self.k1x1_2=tf.compat.v1.placeholder(tf.float32,[1,1,96,3])
                self.k1x1_3=tf.compat.v1.placeholder(tf.float32,[1,1,64,3])
                self.k1x1_4=tf.compat.v1.placeholder(tf.float32,[1,1,32,3])
                self.trans_1=tf.compat.v1.placeholder(tf.float32,[4,4,3,3])
                self.trans_2=tf.compat.v1.placeholder(tf.float32,[4,4,3,3])
                self.trans_3=tf.compat.v1.placeholder(tf.float32,[4,4,3,3])
                self.trans_4=tf.compat.v1.placeholder(tf.float32,[4,4,3,3])
                self.train_data=np.empty((0,width),np.float32)
                self.batch_size=tf.compat.v1.placeholder(tf.int32,[])
                self.x=tf.compat.v1.placeholder(tf.float32,[None,height,width,1])
                self.y=tf.compat.v1.placeholder(tf.float32,[None,height,width,3])
                #kernel11=tf.Variable(tf.compat.v1.random_normal([kernel_size,kernel_size,512,1024], stddev=0.03),name="kernel11")
                #model
                # 4 adet conv uygulandı
                conv1=tf.nn.conv2d(self.x, self.kernel1_place, [1, 1, 1, 1], "SAME",name="conv1")
                conv1=tf.nn.relu(conv1,name="relu1")
                conv1=tf.nn.conv2d(conv1, self.kernel_ara1_place, [1, 1, 1, 1], "SAME",name="conv_ara1")
                conv1=tf.nn.relu(conv1,name="relu_ara1")
                conv1=tf.nn.conv2d(conv1, self.kernel_ara2_place, [1, 1, 1, 1], "SAME",name="conv_ara2")
                conv1=tf.nn.relu(conv1,name="relu_ara2")
                conv1=tf.nn.conv2d(conv1, self.kernel_ara3_place, [1, 1, 1, 1], "SAME",name="conv_ara3")
                conv1=tf.nn.relu(conv1,name="relu_ara3")
                # 1. max pool
                max_pool=tf.nn.max_pool2d(conv1,[1,2,2,1],[1,2,2,1],"SAME")
                # 4 adet conv uygulandı
                conv2=tf.nn.conv2d(max_pool,  self.kernel2_place, [1, 1, 1, 1], "SAME",name="conv2")
                conv2=tf.nn.relu(conv2,name="relu2")
                conv2=tf.nn.conv2d(conv2,  self.kernel_ara4_place, [1, 1, 1, 1], "SAME",name="conv_ara4")
                conv2=tf.nn.relu(conv2,name="relu_ara4")
                conv2=tf.nn.conv2d(conv2,  self.kernel_ara5_place, [1, 1, 1, 1], "SAME",name="conv_ara5")
                conv2=tf.nn.relu(conv2,name="relu_ara5")
                conv2=tf.nn.conv2d(conv2,  self.kernel_ara6_place, [1, 1, 1, 1], "SAME",name="conv_ara6")
                conv2=tf.nn.relu(conv2,name="relu_ara6")
                # 2. max pool
                max_pool=tf.nn.max_pool2d(conv2,[1,2,2,1],[1,2,2,1],"SAME")
                conv3=tf.nn.conv2d(max_pool,  self.kernel3_place, [1, 1, 1, 1], "SAME",name="conv3")
                conv3=tf.nn.relu(conv3,name="relu3")
                conv3=tf.nn.conv2d(conv3,  self.kernel_ara7_place, [1, 1, 1, 1], "SAME",name="conv_ara7")
                conv3=tf.nn.relu(conv3,name="relu_ara7")
                conv3=tf.nn.conv2d(conv3,  self.kernel_ara8_place, [1, 1, 1, 1], "SAME",name="conv_ara8")
                conv3=tf.nn.relu(conv3,name="relu_ara8")
                conv3=tf.nn.conv2d(conv3,  self.kernel_ara9_place, [1, 1, 1, 1], "SAME",name="conv_ara9")
                conv3=tf.nn.relu(conv3,name="relu_ara9")
                #max_pool=tf.nn.max_pool2d(conv3,[1,2,2,1],[1,2,2,1],"SAME")
                conv4=tf.nn.conv2d(conv3,  self.kernel4_place, [1, 1, 1, 1], "SAME",name="conv4")
                conv4=tf.nn.relu(conv4,name="relu4")
                max_pool=tf.nn.max_pool2d(conv4,[1,2,2,1],[1,2,2,1],"SAME")
                conv5=tf.nn.conv2d(max_pool,  self.kernel5_place, [1, 1, 1, 1], "SAME",name="conv5")
                conv5=tf.nn.relu(conv5,name="relu5")
                max_pool=tf.nn.max_pool2d(conv5,[1,2,2,1],[1,2,2,1],"SAME")
                conv6=tf.nn.conv2d(max_pool,  self.kernel6_place, [1, 1, 1, 1], "SAME",name="conv6")
                conv6=tf.nn.relu(conv6,name="relu6")
                #max_pool=tf.nn.max_pool2d(conv6,[1,2,2,1],[1,2,2,1],"SAME")
                conv7=tf.nn.conv2d(conv6,  self.kernel7_place, [1, 1, 1, 1], "SAME",name="conv7")
                conv7=tf.nn.relu(conv7,name="relu7")
                conv8=tf.nn.conv2d(conv7,  self.kernel8_place, [1, 1, 1, 1], "SAME",name="conv8")
                conv8=tf.nn.relu(conv8,name="relu8")
                conv9=tf.nn.conv2d(conv8,  self.kernel9_place, [1, 1, 1, 1], "SAME",name="conv9")
                conv9=tf.nn.relu(conv9,name="relu9")
                conv10=tf.nn.conv2d(conv9, self.kernel10_place, [1, 1, 1, 1], "SAME",name="conv10")
                conv10=tf.nn.relu(conv10,name="relu10")

                                    
                conv5_1x1 = conv_1_by_1(conv5, self.k1x1_2)
                conv4_1x1 = conv_1_by_1(conv4, self.k1x1_3)
                conv2_1x1 = conv_1_by_1(conv2, self.k1x1_4)
                conv9_1x1 = conv_1_by_1(conv10, self.k1x1_1)
                    
                l9_upsample = upsample(conv9_1x1,self.trans_1, self.batch_size)
                    
                l9l5_skip = tf.add(l9_upsample, conv5_1x1)

                l9l5_upsample = upsample(l9l5_skip,self.trans_2, self.batch_size)

                l9l5l3_skip = tf.add(l9l5_upsample, conv4_1x1)
                last_up=upsample(l9l5l3_skip,self.trans_3, self.batch_size)
                    

                l9l5l3l2_skip = tf.add(last_up, conv2_1x1)
                last_up=upsample(l9l5l3l2_skip,self.trans_4, self.batch_size)


                y_ = tf.nn.softmax(last_up)
                return y_
if __name__=="__main__":
        obj=fcn_segmentation()
        iris,pupil=obj.run(["casia_dataset_examples/001/R/S2001R18.jpg"],"konum")
        for i in iris:
                arr=i.astype(np.uint8)
                img=Image.fromarray(arr)
                img.save("fcn_deneme_iris.jpg")
        for i in pupil:
                arr=i.astype(np.uint8)
                img=Image.fromarray(arr)
                img.save("fcn_deneme_pupil.jpg")
