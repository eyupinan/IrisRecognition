import tensorflow as tf
import numpy as np
import PIL
from PIL import Image,ImageOps
import os
import sys
import time
import matplotlib.pyplot as plt
import daugmans_rubber as dg
from multiprocessing import Process,Queue
import gc
import multiprocessing
import math
import threading
#print(tf.__version__)
def initializer(kernel_sizes,init="random_normal",activation="relu"):
        if init=="random_normal":
            std=0.03
        if init=="xavier_normal":
            fan_in=np.prod(kernel_sizes[:-1])
            fan_out=kernel_sizes[-1]
            if activation=="softmax" or activation=="sigmoid" :
                std = np.sqrt(2. / (fan_in + fan_out))
            elif activation=="relu":
                std = np.sqrt(2. / (fan_in))
        if init=="xavier_uniform":
            fan_in=np.prod(kernel_sizes[:-1])
            fan_out=kernel_sizes[-1]
            if activation=="softmax" or activation=="sigmoid":
                std = np.sqrt(6. / (fan_in + fan_out))
            elif activation=="relu":
                std = np.sqrt(6. / (fan_in))
        return std

class cnn_architecture(object):
    def __init__(self):
        self.sess=None
        self.initializer="xavier_normal"
        tf.compat.v1.disable_eager_execution()
        tf.debugging.set_log_device_placement(True)        
        self.is_training=tf.compat.v1.placeholder(tf.bool,[])
        self.data_place_list=[]
        self.layer_count=0
        self.load_pre=False
        self.input_count=0
        self.hidden_count=0,
        self.regularization_type="l2"
        self.trainable_list=[]
        self.freeze_list=[]
        self.warm_list=[]
        self.regularization=0
        self.alpha=0.002
        self.kernel1=None
        self.bias1=None
        self.conv1=None
        self.pre_trained=False
        self.restored=False
        self.keep_prob=tf.compat.v1.placeholder(tf.float32,[])
        self.learning_rate=tf.compat.v1.placeholder(tf.float32,[])
        self.test_data=[]
        self.correct_label_test=[]
        self.original_training_dataset=[]
        self.original_correct_label=[]
        self.last_weights=[]
        self.last_biases=[]
        self.restore_list=[]
        self.model_outputs=[]
        self.middle_layer=-1
        self.ara_place=[]
        self.new_list=[]
        self.kesik_kisim=[]
        self.kernels=[]
        self.daugman=dg.daugman()
        self.mean_and_var=[]
        self.kernel_numpy_arrays=[]
        self.outlier_list=[]
        self.alt_limit=0.95
        self.epoch_count=-1
        self.last_ep=-2
        self.last_kernels=[]
        self.garbage_dataset=[]
        self.path="trained_model/"
    def get_num_class(self):
            try:
                    arr=np.load("train_model/correct.npy")
                    return arr.shape[1]
            except:
                    return None
    def set_save_path(self,path):
            if (os.path.exists(path)!=True):
                        os.mkdir(path)
            self.path=path
    def draw_info(self):
        while True:
                if self.last_ep!=self.epoch_count:
                        plt.clf()
                        plt.plot(self.test_accuracy_list)
                        #plt.ylim(0, 1)
                        plt.draw()
                        plt.pause(0.0001)
                        self.last_ep=self.epoch_count
    def restoration(self):
        init = tf.compat.v1.global_variables_initializer()
        with tf.compat.v1.Session() as sess:
                sess.run(init)
                if os.path.exists('trained_model/model.ckpt-1.data-00000-of-00001')==True:
                    saver = tf.compat.v1.train.Saver(self.kernels)
                    saver.restore(sess,tf.train.latest_checkpoint('trained_model/'))
                    for i in self.kernels:
                            self.kernel_numpy_arrays.append(sess.run(i))
    def set_konumlar(self,konumlar):
        self.konumlar=konumlar
    def set_initializer(self,init_name):
        self.initializer=init_name
    def regularizator(self,regulatization_opt):
        self.regularization_type=regulatization_opt
    def set_optimizer(self,opt):
        self.opt_name=opt
    def get_test_accuracy_list(self):
        return self.test_accuracy
    def set_model_name(self,name):
        self.model_name=name
    def save_name(self,name):
            self.save_name=name
    def save_condition_function(self,func):
            self.cond_func=func
    def convolutional_layer(self,input_conv,kernel_sizes,strides=[1,1,1,1],padding="SAME",activation="relu",state="freeze",layer=0):
        std=initializer(kernel_sizes,self.initializer,activation) 
        kernel=tf.Variable(tf.compat.v1.random_normal(kernel_sizes, stddev=std),name="kernel"+self.model_name+str(self.layer_count))
        bias=tf.Variable(tf.compat.v1.random_normal([kernel_sizes[3]],stddev=0.03),name="bias"+self.model_name+str(self.layer_count))
        if self.pre_trained==True:
                if layer==self.middle_layer+1:
                        place=tf.compat.v1.placeholder(tf.float32,self.kesik_kisim[-1].shape)
                        self.ara_place.append(place)
                        input_conv=[place,input_conv]
        self.kernels.append(kernel)
        self.kernels.append(bias)   
        if state=="freeze":
            self.freeze_list.append(kernel)
            self.freeze_list.append(bias)
        if state=="trainable":
            self.trainable_list.append(kernel)
            self.trainable_list.append(bias)

        if layer<=self.middle_layer:
                self.restore_list.append(kernel)
                self.restore_list.append(bias)
        if layer>self.middle_layer:
                self.new_list.append(kernel)
                self.new_list.append(bias)
        if activation=="relu":
            if type(input_conv)!=list:
                    conv=tf.nn.conv2d(input_conv, kernel, strides, padding,name="conv"+str(self.layer_count))
                    conv=tf.nn.bias_add(conv,bias,name="bias_add"+self.model_name+str(self.layer_count))
                    conv=self.relu_layer(conv)
            elif type(input_conv==list):
                    conv=[]
                    for i in input_conv:
                            ara_conv=tf.nn.conv2d(i, kernel, strides, padding,name="conv"+str(self.layer_count))
                            ara_conv=tf.nn.bias_add(ara_conv,bias,name="bias_add"+self.model_name+str(self.layer_count))
                            conv.append(self.relu_layer(ara_conv))
        elif activation=="softmax":
            if type(input_conv)!=list:
                    conv=tf.nn.conv2d(input_conv, kernel, strides, padding,name="conv"+str(self.layer_count))
                    conv=tf.nn.bias_add(conv,bias,name="bias_add"+self.model_name+str(self.layer_count))
                    conv=self.softmax_layer(conv)
            elif type(input_conv==list):
                    conv=[]
                    for i in input_conv:
                            ara_conv=tf.nn.conv2d(i, kernel, strides, padding,name="conv"+str(self.layer_count))
                            ara_conv=tf.nn.bias_add(ara_conv,bias,name="bias_add"+self.model_name+str(self.layer_count))
                            conv.append(self.softmax_layer(ara_conv))

            
        self.layer_count+=1
        #if self.conv1==None:
        #        self.conv1=conv
        self.model_outputs.append(conv)
        if self.pre_trained==True:
                if layer==self.middle_layer:
                        self.kesik_kisim.append(conv)
                
        return conv
    def max_pooling_layer(self,_input,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME",layer=0):
        if type(_input)!=list:
                pool=tf.nn.max_pool2d(_input,ksize,strides,padding)
        if type(_input)==list:
                pool=[]
                for i in _input:
                        pool.append(tf.nn.max_pool2d(i,ksize,strides,padding))
        if self.pre_trained==True:
                if layer==self.middle_layer:
                        self.kesik_kisim.append(pool)             
        return pool        
    def relu_layer(self,input_conv):
        return tf.nn.relu(input_conv,name="relu"+str(self.layer_count))
    def softmax_layer(self,input_conv):
        print(self.temp)
        return tf.nn.softmax(input_conv/self.temp,name="softmax"+str(self.layer_count))
    def sigmoid_layer(self,input_conv):
        return tf.nn.sigmoid(input_conv,name="sigmoid"+str(self.layer_count))
    def flattening_layer(self,_input,layer=0):
        if type(_input)!=list:
                shapes=_input.shape
        if type(_input)==list:
                shapes=_input[0].shape
        self.input_count=1
        for shape in shapes:
            if shape!=None:
                self.input_count*=shape
        self.hidden_count=int(self.input_count/7*4)
        if type(_input)!=list:
                flat=tf.reshape(_input,[-1,self.input_count])
        else:
                flat=[]
                for i in _input:
                        flat.append(tf.reshape(i,[-1,self.input_count]))
        if self.pre_trained==True:
                if layer==self.middle_layer:
                        self.kesik_kisim.append(flat)
        return flat
    def hidden_layer(self,_input,activation="relu",input_count=0,hidden_count=0,state="trainable",restore=True,layer=0):                
        if hidden_count==0:
            hidden_count=self.hidden_count
        if input_count==0:
            input_count=self.input_count
        if state=="warm" or True:
                if type(_input)==list:
                        shapes=_input[0].shape
                else:
                        shapes=_input.shape
                input_count=1
                for shape in shapes:
                    if shape!=None:
                        input_count*=shape
                #if activation=="softmax":
                #        print("input size:",input_count)
        kernel_sizes=[input_count, hidden_count]
        std=initializer(kernel_sizes,self.initializer,activation)
        W = tf.Variable(tf.compat.v1.random_normal([input_count, hidden_count], stddev=std), name="W"+self.model_name+str(self.layer_count))
        b = tf.Variable(tf.compat.v1.random_normal([hidden_count],stddev=0.01), name="b"+self.model_name+str(self.layer_count))
        if state!="outlier":
                self.kernels.append(W)
                self.kernels.append(b)
        if self.pre_trained==True:
                if layer==self.middle_layer+1:
                        place=tf.compat.v1.placeholder(tf.float32,self.kesik_kisim[-1].shape)
                        self.ara_place.append(place)
                        _input=[place,_input]
        if layer<=self.middle_layer and self.pre_trained==True:
                self.restore_list.append(W)
                self.restore_list.append(b)
        if layer>self.middle_layer and self.pre_trained==True:
                self.new_list.append(W)
                self.new_list.append(b)
        self.layer_count+=1
        if state=="trainable":
            self.trainable_list.append(W)
            self.trainable_list.append(b)
        elif state=="freeze":
            self.freeze_list.append(W)
            self.freeze_list.append(b)
        elif state=="warm":
            self.warm_list.append(W)
            self.warm_list.append(b)
        elif state=="outlier":
            self.outlier_list.append(W)
            self.outlier_list.append(b)
        elif state=="last_hidden":
            self.last_kernels.append(W)
            self.last_kernels.append(b)
        if self.regularization_type=="l2" and (self.pre_trained==False or layer>self.middle_layer):
                self.regularization=self.regularization+tf.reduce_sum(tf.square(W))+tf.reduce_sum(tf.square(b))
        self.input_count=hidden_count
        
        if activation=="relu":
            if type(_input)!=list:
                    hidden_out = tf.add(tf.matmul(_input, W), b)
                    hidden_out=self.relu_layer(hidden_out)
                    self.model_outputs.append(hidden_out)
                    ret=hidden_out
            elif type(_input)==list:
                    output_list=[]
                    for i in _input:
                            hidden_out = tf.add(tf.matmul(i, W), b)
                            hidden_out=self.relu_layer(hidden_out)
                            output_list.append(hidden_out)
                    ret=output_list
        elif activation=="softmax":
            self.last_weights.append(W)
            self.last_biases.append(b)
            if type(_input)!=list:
                    hidden_out = tf.add(tf.matmul(_input, W), b)
                    hidden_out=self.softmax_layer(hidden_out)
                    self.model_outputs.append(hidden_out)
                    ret=hidden_out
            elif type(_input)==list:
                    output_list=[]
                    for i in _input:
                          hidden_out = tf.add(tf.matmul(i, W), b)  
                          hidden_out=self.softmax_layer(hidden_out)
                          output_list.append(hidden_out)
                    ret=output_list
        elif activation=="sigmoid":
            self.last_weights.append(W)
            self.last_biases.append(b)
            if type(_input)!=list:
                    hidden_out = tf.add(tf.matmul(_input, W), b)
                    hidden_out=self.sigmoid_layer(hidden_out)
                    self.model_outputs.append(hidden_out)
                    ret=hidden_out
            elif type(_input)==list:
                    output_list=[]
                    for i in _input:
                          hidden_out = tf.add(tf.matmul(i, W), b)  
                          hidden_out=self.sigmoid_layer(hidden_out)
                          output_list.append(hidden_out)
                    ret=output_list
        if self.pre_trained==True:
                if layer==self.middle_layer:
                        self.kesik_kisim.append(ret)
                
        return ret
    def dropout_layer(self,_input,layer=0):
        if type(_input)!=list:
                drop=tf.nn.dropout(_input,self.keep_prob)
        else:
                drop=[]
                for i in _input:
                     drop.append(tf.nn.dropout(i,self.keep_prob))
        if layer==self.middle_layer and self.pre_trained==True:
                self.kesik_kisim.append(drop)
        return drop
    def merge_layer(self,_input_list,layer=0):
        if type(_input_list[0])!=list:
                
                merged=tf.concat(_input_list,axis=1)
        else:
                merged=[]
                for r in range(2):
                        out=tf.concat([_input_list[0][r],_input_list[1][r]],axis=1)
                        merged.append(out)
        self.merge_layer_output=merged
        if layer==self.middle_layer:
                self.kesik_kisim.append(merged) 
        return merged
    def new_last_layer(self):
            W,b=self.merge_weights()
            self.regularization=self.regularization+tf.reduce_sum(tf.square(W))+tf.reduce_sum(tf.square(b))
            self.trainable_list.append(W)
            self.trainable_list.append(b)
            hidden_out = tf.add(tf.matmul(self.last_hidden, W), b)
            hidden_out=self.softmax_layer(hidden_out)
            return hidden_out
    def loss_function(self,y_,y=None,loss_type="categori"):#y_ prediction ,y ground truth
        if y==None:
                y=self.y
        y_clipped = tf.clip_by_value(y_, 1e-10, 0.9999999)
        if loss_type=="categori":
                
                arg=tf.cast(tf.argmax(y,1),tf.int32)
                arg=tf.expand_dims(arg,axis=1)
                y_indexes=tf.concat([tf.expand_dims(tf.range(tf.shape(y)[0]),axis=1),arg],axis=1)
                gathered=tf.gather_nd(y_clipped,y_indexes)
                log_likelihood = -tf.math.log(gathered)/tf.cast(tf.shape(y)[0],tf.float32)
                cross_entropy = tf.reduce_sum(log_likelihood)
        if loss_type=="binary":
                cross_entropy = tf.reduce_mean(-(tf.reduce_sum(y * tf.math.log(y_clipped)
                                    + (1 - y) * tf.math.log(1 - y_clipped), axis=1)))
        ##
        return cross_entropy
    def probability_limit(self,y_):
        unknown_samples=tf.where(tf.less(tf.reduce_max(y_,1),self.alt_limit))
        return unknown_samples
    def accuracy(self,y_,y=None):#y_ prediction ,y ground truth
        if y==None:
                y=self.y
        correct_prediction = tf.equal(tf.argmax(y_, 1),tf.argmax(y, 1))
        wrong_prediction=tf.not_equal(tf.argmax(y_, 1),tf.argmax(y, 1))
        self.correct_indexes=tf.where(correct_prediction)
        self.wrong_indexes=tf.where(wrong_prediction)
        self.gather_correct=tf.gather_nd(y_,self.correct_indexes)
        self.gather_wrong=tf.gather_nd(y_,self.wrong_indexes)
        self.correct_prob=tf.reduce_max(self.gather_correct,1)
        self.wrong_prob=tf.reduce_max(self.gather_wrong,1)
        self.correct_mean=tf.cond(
              tf.equal(tf.size(self.correct_prob), 0), 
              lambda : tf.constant(0.0), lambda: tf.reduce_mean(self.correct_prob)
            )
        self.wrong_mean=tf.cond(
              tf.equal(tf.size(self.wrong_prob), 0), 
              lambda : tf.constant(0.0), lambda: tf.reduce_mean(self.wrong_prob)
            )
        self.probability_difference=self.correct_mean-self.wrong_mean
        self.limit=self.wrong_mean+(self.probability_difference/4*3)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return accuracy
    def cnn_arc(self,cnn_size,num_classes):
            data_place=tf.compat.v1.placeholder(tf.float32,[None,cnn_size[0],cnn_size[1],1])
            self.data_place_list.append(data_place)
            kernel_size=3
            conv1=self.convolutional_layer(data_place, [11,11,1,16],layer=0)
            conv=self.convolutional_layer(conv1, [3,3,16,32],layer=1)
            conv=self.max_pooling_layer(conv,layer=2)
            conv=self.convolutional_layer(conv, [3,3,32,48],layer=3)
            conv=self.max_pooling_layer(conv,layer=4)
            conv=self.convolutional_layer(conv, [kernel_size,kernel_size,48,64],layer=5)
            conv=self.max_pooling_layer(conv,layer=6)
            conv=self.convolutional_layer(conv, [kernel_size,kernel_size,64,96],layer=7)
            conv=self.max_pooling_layer(conv,layer=8)
            conv2=self.convolutional_layer(conv, [kernel_size,kernel_size,96,128],layer=9)
            conv=self.max_pooling_layer(conv2,layer=10)
            conv=self.convolutional_layer(conv, [kernel_size,kernel_size,128,196],layer=11)
            out=self.flattening_layer(conv,layer=12)
            out=self.hidden_layer(out,"relu",state="freeze",layer=13)
            out=self.dropout_layer(out,layer=14)
            return out
    def fcn_arc(self,_input,num_class):
            self.y=tf.compat.v1.placeholder(tf.float32,[None,num_class])
            #shp=tf.shape(_input)
            if type(_input)!=list:
                    shapes=_input.shape
            else:
                    shapes=_input[0].shape
            input_count=1
            for shape in shapes:
                    if shape!=None:
                        input_count*=shape
                
            hidden=self.hidden_layer(_input,"relu",input_count,int(input_count/3*2),layer=16)
            self.last_hidden=hidden
            output=self.hidden_layer(hidden,"softmax",hidden_count=num_class,layer=17)
            return output,self.y
    def outlier(self,_input):
            self.outlier_place=tf.compat.v1.placeholder(tf.float32,[None,2])
            if type(_input)!=list:
                    shapes=_input.shape
            else:
                    shapes=_input[0].shape
            input_count=1
            for shape in shapes:
                    if shape!=None:
                        input_count*=shape
            hidden=self.hidden_layer(_input,"relu",input_count,int(input_count/3*2),state="outlier",layer=-1)
            output=self.hidden_layer(hidden,"sigmoid",hidden_count=2,state="outlier",layer=-1)
            return output
    def optimizer(self,nn_last_layer,trainable_list=[],correct_place=None,loss_type="categori"):
        self.last_layer=nn_last_layer
        
        if correct_place==None:
                loss=self.loss_function(nn_last_layer,loss_type=loss_type)
        else:
                loss=self.loss_function(nn_last_layer,correct_place,loss_type=loss_type)
        if trainable_list==[]:
                trainable_list=self.kernels
        if self.pre_trained==False:
                if self.opt_name=="adam":
                    if self.regularization_type=="l2":
                            opt = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss+self.regularization*self.alpha,var_list=trainable_list)
                    elif self.regularization_type=="off":
                            opt = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)
                elif self.opt_name=="momentum":
                    if self.regularization_type=="l2":
                            opt = tf.compat.v1.train.MomentumOptimizer(self.learning_rate,0.9).minimize(loss+self.regularization*self.alpha)
                    elif self.regularization_type=="off":
                            opt = tf.compat.v1.train.MomentumOptimizer(self.learning_rate,0.9).minimize(loss)
        else:
                if self.opt_name=="adam":
                        if self.regularization_type=="l2":
                            opt = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss+self.regularization*self.alpha,var_list=trainable_list)
                        if self.regularization_type=="off":
                            opt = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss,var_list=trainable_list)
                elif self.opt_name=="momentum":
                    opt = tf.compat.v1.train.MomentumOptimizer(self.learning_rate,0.9).minimize(loss,var_list=trainable_list)
        if correct_place==None:
                acc=self.accuracy(nn_last_layer)
        else:
                acc=self.accuracy(nn_last_layer,correct_place)
        return loss,opt,acc
    def argmax(self,_input):
            return tf.argmax(_input,1)
    def reduce_max(self,_input):
            return tf.reduce_max(_input,1)
    def get_random_data(self,size,data=None,label=None):
            stc_train=[]
            if data==None:
                    sample_count=self.original_training_dataset[0].shape[0]
                    indices=np.random.randint(0,sample_count,size)
                    stc_train.append(self.original_training_dataset[0][indices])
                    stc_train.append(self.original_training_dataset[1][indices])
                    #stc_train.append(self.original_training_dataset[2][indices])
                    correct_list=self.original_correct_label[indices]
                    return stc_train,correct_list
            elif data!=None:
                    sample_count=data[0].shape[0]
                    indices=np.random.randint(0,sample_count,size)
                    stc_train.append(data[0][indices])
                    stc_train.append(data[1][indices])
                    #stc_train.append(data[2][indices])
                    correct_list=label[indices]
                    return stc_train,correct_list
    #@profile
    def merge_datasets(self):
        #### bu kısımda orjinal task'ın correct label ve yeni task'ın correct label değerleri birleştiriliyor. Bu şekilde yeni sınıf sayısına uygun bir one hot
        # array'i oluşturuluyor. Yeni data set ve orjinal dataset birleştiriliyor. Aynı işlemler test aşaması içinde gerçekleştiriliyor       
        merged_dataset=[]
        new_sample=self.correct_label.shape[0]
        old_sample=self.original_correct_label.shape[0]
        new_class_num=self.correct_label.shape[1]+self.original_correct_label.shape[1]
        new_correct_label=np.zeros((new_sample,new_class_num),np.float32)
        old_correct_label=np.zeros((old_sample,new_class_num),np.float32)
        
        new_correct_label[:,new_class_num-self.correct_label.shape[1]:]=self.correct_label
        old_correct_label[:,:self.original_correct_label.shape[1]]=self.original_correct_label
        for i in range(2):
                merged_dataset.append(np.concatenate([self.original_training_dataset[i],self.train_data[i]],axis=0))
        merged_correct_label=np.concatenate([old_correct_label,new_correct_label],axis=0)
        merged_dataset,merged_correct_label=self.shuffling(merged_dataset,merged_correct_label)
        self.train_data=merged_dataset
        self.correct_label=merged_correct_label
        merged_test_data=[]
        merged_test_correct_label=[]
        for data_index in range(len(self.test_data)):
                test_data=self.test_data[data_index]
                correct_label_test=self.correct_label_test[data_index]
                test_correct_label=np.zeros((correct_label_test.shape[0],new_class_num),np.float32)
                if data_index==0:
                        test_correct_label[:,new_class_num-correct_label_test.shape[1]:]=correct_label_test
                elif data_index==1:
                        test_correct_label[:,:correct_label_test.shape[1]]=correct_label_test
                merged_test_correct_label.append(test_correct_label)
        merged_test_dataset=[]
        for i in range(2):
                merged_test_dataset.append(np.concatenate([self.test_data[0][i],self.test_data[1][i]],axis=0))
        self.test_data=[merged_test_dataset]
        self.correct_label_test=[np.concatenate(merged_test_correct_label,axis=0)]
        ###### Bu kısımda sistem yeni dataset ve correct_label dizisine göre yeniden eğitiliyor eğitilecek olan ağırlık değişkenlerinin bazıları sabitleniyor(feature extraction)
        # ağırlık değerleri sisteme aşına olduğu için yeni dataset'in tamamı değil sadece içerisinden rastgele seçilen görüntüler eğitiliyor bu şekilde sistemin hızlı çalışması sağlanıyor
        ## ilk olarak dondurulmuş modelin oluşturduğu output değeri hesaplanıp saklanıyor
    def _middleware_output(self,sess,batch_size):
        ##bu fonksiyon modelin ikiye ayrıldığı katmanı kullanarak bir output oluşturur. Bu output değerleri
        ## modelin geriye kalan kısmı eğitilmek istendiği zaman tekrar tekrar kullanılabilmesi için gerekmektedir.
        ## bu işlemin ayrı bir şekilde yapılmasının sebebi  her zaman aynı sonucu oluşturacak dondurulmuş katmanları defalarca hesaplamak yerine
        ## bir sefer hesaplayıp defalarca kullanmaktır.Bu sayede zaman kaybı engellenmiş olur.
        step_size=int(self.train_data[0].shape[0]/batch_size)
        step_size_test=int(self.test_data[0][0].shape[0]/batch_size)
        
        feed={self.is_training:False,self.keep_prob:0}
        output_data=[]
        output_test_data=[]
        for v in self.kesik_kisim:
              output_data.append([])
              output_test_data.append([])
        for batch_index in range(step_size):
                for place_index in range(len(self.data_place_list)):
                             feed[self.data_place_list[place_index]]=self.train_data[place_index][batch_size*batch_index:batch_size*(batch_index+1)]
                batch_output=sess.run(self.kesik_kisim,feed_dict=feed)
                for z in range(len(batch_output)):
                        output_data[z].append(batch_output[z])
        for batch_index in range(step_size_test):
                for place_index in range(len(self.data_place_list)):
                             feed[self.data_place_list[place_index]]=self.test_data[0][place_index][batch_size*batch_index:batch_size*(batch_index+1)]
                batch_output=sess.run(self.kesik_kisim,feed_dict=feed)
                for z in range(len(batch_output)):
                        output_test_data[z].append(batch_output[z])
        return output_data,output_test_data
    #@profile
    def train_nn(self,sess,epoch,opt,loss,acc,throw_info=True):
        place_list=[]
        if self.pre_trained==True:
                correct_place=self.merged_correct_place
                batch_size=96
                keep_prob=0.2
                initial_learning_rate=0.0001
                test_batch_size=batch_size
                self.merge_datasets()
                output_data,output_test_data=self._middleware_output(sess,batch_size)
                step_size_train=len(output_data[0])
                step_size_test=len(output_test_data[0])
                place_list=self.ara_place
        if self.pre_trained==False:
                correct_place=self.y
                keep_prob=0.3
                initial_learning_rate=0.0003
                batch_size=96
                test_batch_size=10
                place_list=self.data_place_list
                
                reshaped_test_data_list=self.test_data
                reshaped_data_list=self.train_data
                step_size_train=int(self.train_data[0].shape[0]/batch_size)
                step_size_test=int(self.test_data[0][0].shape[0]/test_batch_size)
        print(self.train_data[0].shape)
        print(self.test_data[0][0].shape)
        #self.test_data=[]
        #self.train_data=[]        
        train_accuracy_list=[]
        test_accuracy_list=[]
        cost_list=[]
        i=0
        f=True
        f2=True
        test_cost_list=[]
        sorunlular=[[]]
        self.show_info=False
        if self.show_info==True:
                plt.ion()
                fig, ax1 = plt.subplots()
                ax2=ax1.twinx()
                ax1.set_xlabel("epoch")
                ax1.set_ylabel("accuracy")
                ax2.set_ylabel("loss")
        self.ort_list=[]
        self.correct_ort_list=[]
        self.wrong_ort_list=[]
        while i<epoch:
            f3=True
            learning_rate = initial_learning_rate * 1/(1 + 0.03 * i)
            c=0
            accuracy_train=0
            cost=0
            
            for batch_index in range(step_size_train):
                #if batch_index%20==0:
                #        time.sleep(15)
                feed={self.learning_rate:learning_rate,correct_place:self.correct_label[batch_index*batch_size:(1+batch_index)*batch_size],self.is_training:False,self.keep_prob:keep_prob}
                for place_index in range(len(place_list)):
                        if self.pre_trained==False:
                                feed[place_list[place_index]]=reshaped_data_list[place_index][batch_index*batch_size:(1+batch_index)*batch_size]
                        else:
                                feed[place_list[place_index]]=output_data[place_index][batch_index]
                cost,_o1=sess.run([loss,opt],feed_dict=feed)
                
                c+=cost/step_size_train
                #c_sig+=cost_sig/step_size_train
                feed[self.is_training]=False
                feed[self.keep_prob]=0
                if i%1==0:
                        accuracy_train+=(sess.run(acc,feed_dict=feed))/step_size_train
            accuracy_test=0
            test_loss=0
            total_limit=0
            total_prob_diff=0
            index_list=[]
            ort=0
            correct_ort=0
            wrong_ort=0
            re_list=[]
            for batch_index in range(step_size_test):
                    feed_test={self.learning_rate:learning_rate,correct_place:self.correct_label_test[0][batch_index*test_batch_size:(1+batch_index)*test_batch_size],self.is_training:False,self.keep_prob:0}
                    for place_index in range(len(place_list)):
                        if self.pre_trained==False:
                                feed_test[place_list[place_index]]=reshaped_test_data_list[0][place_index][test_batch_size*batch_index:test_batch_size*(batch_index+1)]
                        else:
                                feed_test[place_list[place_index]]=output_test_data[place_index][batch_index]
                    ac_test=sess.run([acc,loss],feed_dict=feed_test)
                    #print("ort:",ac_test[3])

                    if math.isnan(ac_test[0]):
                            print(output_test_data[place_index][batch_index])
                            print(batch_index)
                            input()

                    accuracy_test+=ac_test[0]/step_size_test
                    #index_list.append(ac_test[2])
                    test_loss+=ac_test[1]/step_size_test

            train_accuracy_list.append(accuracy_train)
            test_accuracy_list.append(accuracy_test)
            cost_list.append(c)
            test_cost_list.append(test_loss)
            self.epoch_count=i
            if self.show_info==True:
                    ax1.clear()
                    ax2.clear()
                    ax1.plot(test_accuracy_list,label="test acc",color="red")
                    ax1.plot(train_accuracy_list,label="train acc",color="blue")
                    ax2.plot(cost_list,label="train loss",color="cyan")
                    ax2.plot(test_cost_list,label="test loss",color="magenta")
                    ax1.set_xlabel("epoch")
                    ax1.set_ylabel("accuracy")
                    ax2.set_ylabel("loss")
                    plt.draw()
                    handles, labels = [(a + b) for a, b in zip(ax1.get_legend_handles_labels(), ax2.get_legend_handles_labels())]
                    fig.legend(handles, labels,loc="lower left")
                    plt.pause(0.0001)
            if throw_info==True:
                print("epoch:",i,"cost:",c,"train_acc:",accuracy_train,"test_acc:",accuracy_test,"test_loss:",test_loss)
                if self.cond_func(accuracy_test)==True and False:
                        index_wr=0
                        sorunlular.append([])
                        for z in index_list:
                                for u in z:
                                        #print(index_wr*test_batch_size+u[0])
                                        sorunlular[-1].append(index_wr*test_batch_size+u[0])
                                index_wr+=1
                        eşleşen_sayisi=0
                        print("önceki hatalı sayisi:",len(sorunlular[-2]))
                        print("şimdiki hatalı sayisi:",len(sorunlular[-1]))
                        for t in range(len(sorunlular[-1])):
                                if t in sorunlular[-2]:
                                        eşleşen_sayisi+=1
                        print("eşleşen sayisi:",eşleşen_sayisi)
                        #input()
            if accuracy_test>0.96 and f==True:
                    initial_learning_rate=initial_learning_rate/10
                    keep_prob=0.2
                    f=False
            if accuracy_test>0.97 and f2==True:
                    initial_learning_rate=initial_learning_rate/10
                    keep_prob=0
                    f2=False
            if (self.cond_func(accuracy_test)==True):
                    
                    saver = tf.compat.v1.train.Saver(self.kernels)
                    save_path = saver.save(sess,self.path+"model.ckpt",global_step=1)
                    if self.pre_trained==True:
                            self.original_task_saver("new",self.train_data,self.correct_label,self.test_data,self.correct_label_test)
                    break
            i+=1



        print(test_accuracy_list)

        np.save(self.path+"test accuracy.npy",test_accuracy_list)
        if self.pre_trained!=True:
                            saver = tf.compat.v1.train.Saver()
                            save_path = saver.save(sess, self.path+"model.ckpt",global_step=1)
        plt.plot(range(len(train_accuracy_list)), train_accuracy_list, label = "training")
        plt.plot(range(len(test_accuracy_list)), test_accuracy_list, label = "test")
        #plt.plot(range(len(cost_list)), cost_list, label = "cost")
        plt.xlabel('x - axis')
        plt.legend()
        #plt.show() 
        plt.savefig(self.save_name)
        plt.cla()   # Clear axis
        plt.clf()
        self.test_accuracy=test_accuracy_list
    def shuffling(self,data,label):
        seed = np.random.randint(0, 100000)  
        for i in data:
                np.random.seed(seed)  
                np.random.shuffle(i)  
        np.random.seed(seed)  
        np.random.shuffle(label)
        return data,label
    def create_model(self,num_classes):
            self.data_place_list=[]
            normalized_img_height1=48
            normalized_img_width1=360
            normalized_img_height2=32
            normalized_img_width2=270
            cnn.set_model_name("1")
            arc1=self.cnn_arc([normalized_img_height1,normalized_img_width1],num_class)
            self.set_model_name("2")
            arc2=self.cnn_arc([normalized_img_height2,normalized_img_width2],num_class)
            merged=self.merge_layer([arc1,arc2],layer=15)
            cnn.set_model_name("3")
            output,correct_plc=cnn.fcn_arc(merged,num_classes)
            return output,correct_plc
    #@profile
    def prepare_data(self,sess,train_data):
            #print("dim:",self.train_data[0].ndim)
            #print("dim_Test:",self.test_data[0][0].ndim)
            reshape_place=tf.compat.v1.placeholder(tf.float32,[None,None,None])
            expand_data=tf.expand_dims(reshape_place,axis=3)
            if self.train_data[0].ndim==3:
                    
                    reshaped_data_list=[]
                    reshaped_test_data_list=[]
                    for i in train_data:
                            expanded=sess.run(expand_data,feed_dict={reshape_place:i})
                            reshaped_data_list.append(expanded)
                    train_data=reshaped_data_list
            if self.test_data[0][0].ndim==3:
                    reshaped_data_list=[]

                    new_test_data=[]
                    for test_data in self.test_data:
                        reshaped_test_data_list=[]
                        for i in test_data:
                            expanded=sess.run(expand_data,feed_dict={reshape_place:i})
                            reshaped_test_data_list.append(expanded)
                        test_data=reshaped_test_data_list
                        new_test_data.append(test_data)
                    self.test_data=new_test_data
                    reshaped_test_data_list=[]

            return train_data
    def pre_processing(self,konumlar,correct_label=[],data_type="valid"):
        if self.load_pre==False:
                normalized_img_height1=48
                normalized_img_width1=360
                normalized_img_height2=32
                normalized_img_width2=270
                daugman_sizes=[[normalized_img_height1,normalized_img_width1],[normalized_img_height2,normalized_img_width2]]
                if self.mean_and_var==[]:
                        if (self.pre_trained==False and data_type=="train"):
                                        dosya=open(self.path+"mean and variance.txt","w")
                        else:
                                        dosya=open(self.path+"mean and variance.txt","r")
                                        dosya_read=dosya.readlines()
                                        for i in dosya_read:
                                                try:
                                                    if i[-1]=="\n":
                                                            self.mean_and_var.append(float(i[:-1]))
                                                    else:
                                                            self.mean_and_var.append(float(i))
                                                except:
                                                        raise Exception("test ön işleminin gerçekleşebilmesi için eğitilmiş bir data bulunması gerekmektedir!")
                                        if self.mean_and_var==[]:
                                                raise Exception ("test ön işleminin gerçekleşebilmesi için eğitilmiş bir data bulunması gerekmektedir!")
                #dosya=open("C:/Users/ibrahim eyyüp inan/Desktop/konumlar.txt","w")
                #sayac=0
                #for i in konumlar:
                #       dosya.write(str(sayac)+" "+i+"\n")
                #       sayac+=1
                self.daugman.set_img(konumlar)
                data,error_list=self.daugman.normalization(daugman_sizes)
                data_count=range(len(daugman_sizes))
                for data_index in data_count:
                        data[data_index]=data[data_index]/255
                if data_type=="train":

                        for data_index in  data_count:
                                if self.pre_trained==False:
                                        self.mean_and_var.append(np.mean(data[data_index]))
                                        self.mean_and_var.append(np.var(data[data_index]))
                                        dosya.write(str(np.mean(data[data_index]))+"\n")
                                        dosya.write(str(np.var(data[data_index]))+"\n")
                                        
                                        data[data_index]=(data[data_index]-self.mean_and_var[-2])/(np.sqrt(self.mean_and_var[-1])+1e-8)
                                if self.pre_trained==True:
                                        data[data_index]=(data[data_index]-self.mean_and_var[data_index*2])/(np.sqrt(self.mean_and_var[data_index*2+1])+1e-8)

                        dosya.close()
                        correct_label=np.delete(correct_label,error_list,axis=0)
                        self.train_data=data
                        self.correct_label=correct_label
                        self.train_data,self.correct_label=self.shuffling(self.train_data,self.correct_label)
                elif data_type=="valid":
                        
                        if self.mean_and_var==[]:
                                raise Exception("Baştan eğitilecek modellerde eğitim datasının test datasından önce tanımlanması gerekmektedir!")
                        for data_index in  data_count:
                                data[data_index]=(data[data_index]-self.mean_and_var[data_index*2])/(np.sqrt(self.mean_and_var[data_index*2+1])+1e-8)
                        self.test_data.append(data)
                        self.correct_label_test.append(np.delete(correct_label,error_list,axis=0))
                        self.test_konumlar=list(np.delete(konumlar,error_list))
                elif data_type=="test":
                        if self.mean_and_var==[]:
                                raise Exception("Baştan eğitilecek modellerde eğitim datasının test datasından önce tanımlanması gerekmektedir!")
                        for data_index in  data_count:
                                data[data_index]=(data[data_index]-self.mean_and_var[data_index*2])/(np.sqrt(self.mean_and_var[data_index*2+1])+1e-8)
                                data[data_index]=np.expand_dims(data[data_index],axis=3)
                        
                        self.test_data.append(data)
                        self.test_konumlar=list(np.delete(konumlar,error_list))
                
    def original_task_saver(self,state=None,yeni_data=[],yeni_correct=[],yeni_test_data=[],yeni_test_correct=[]):
        
        
        if self.pre_trained==False and self.load_pre==False:
            print("saver shapes:")
            print(self.train_data[0].shape)
            print(self.test_data[0][0].shape)
            np.save(self.path+"train_data-360x48.npy",self.train_data[0])
            np.save(self.path+"train_data-270x32.npy",self.train_data[1])
            np.save(self.path+"correct.npy",self.correct_label)
            np.save(self.path+"test_data-360x48.npy",self.test_data[0][0])
            np.save(self.path+"test_data-270x32.npy",self.test_data[0][1])
            np.save(self.path+"correct_test.npy",self.correct_label_test[0])
        if self.load_pre==True and self.pre_trained==False:
            self.train_data=[[],[]]
            self.test_data=[[[],[]]]
            self.correct_label_test=[[]]
            self.train_data[0]=np.load(self.path+"train_data-360x48.npy")
            self.train_data[1]=np.load(self.path+"train_data-270x32.npy")
            self.correct_label=np.load(self.path+"correct.npy")
            self.test_data[0][0]=np.load(self.path+"test_data-360x48.npy")
            self.test_data[0][1]=np.load(self.path+"test_data-270x32.npy")
            self.correct_label_test[0]=np.load(self.path+"correct_test.npy")
            dosya=open(self.path+"mean and variance.txt","r")
            print("test shape:",self.test_data[0][0].shape)
            dosya_read=dosya.readlines()
            for i in dosya_read:
                try:
                        if i[-1]=="\n":
                                self.mean_and_var.append(float(i[:-1]))
                        else:
                                self.mean_and_var.append(float(i))
                except:
                        raise Exception("test ön işleminin gerçekleşebilmesi için eğitilmiş bir data bulunması gerekmektedir!")
            #for data_index in  range(2):
            #            self.unknown_data[data_index]=(self.unknown_data[data_index]-self.mean_and_var[data_index*2])/(np.sqrt(self.mean_and_var[data_index*2+1])+1e-8)
            #            self.unknown_data[data_index]=np.expand_dims(self.unknown_data[data_index],axis=3)
        if self.pre_trained==True and state==None:
            test_data_original=[]
            test_data_original.append(np.load(self.path+"test_data-360x48.npy"))
            test_data_original.append(np.load(self.path+"test_data-270x32.npy"))
            self.original_training_dataset.append(np.load(self.path+"train_data-360x48.npy"))
            self.original_training_dataset.append(np.load(self.path+"train_data-270x32.npy"))
            self.original_correct_label=np.load(self.path+"correct.npy")
            correct_label_test_original=np.load(self.path+"correct_test.npy")
            #for i in range(len(test_data_original)):                    
            #        test_data_original[i]=(test_data_original[i]-self.mean_and_var[i])/(np.sqrt(self.mean_and_var[i+1])+1e-8)
            self.test_data.append(test_data_original)
            self.correct_label_test.append(correct_label_test_original)
            
        if self.pre_trained==True and state!=None:
            np.save(self.path+"train_data-360x48.npy",yeni_data[0])
            np.save(self.path+"train_data-270x32.npy",yeni_data[1])
            np.save(self.path+"correct.npy",yeni_correct)
            np.save(self.path+"test_data-360x48.npy",yeni_test_data[0][0])
            np.save(self.path+"test_data-270x32.npy",yeni_test_data[0][1])
            np.save(self.path+"correct_test.npy",yeni_test_correct[0])
    def one_hot(self,indices,depth):
            correct_label=tf.compat.v1.Session().run(tf.one_hot(indices,depth))
            return correct_label
    #@profile
    def train(self,epoch=20):
        #self.temp=1
        self.temp=tf.Variable(np.array([1.],np.float32))
        self.original_task_saver()
        print(self.train_data[0].shape)
        sample_count=self.correct_label.shape[0]
        num_class=self.correct_label.shape[1]
        #self.pre_trained=True
        if self.pre_trained==False:
                    arc,_=self.create_model(num_class)
                    self.ar=arc
                    self.re=tf.reduce_max(self.ar,axis=1)
                    self.ort=tf.reduce_mean(self.re)
                    loss,opt,acc=self.optimizer(arc)
                    self.optimizer_temp=tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss,var_list=[self.temp])
        if self.pre_trained==True:
                    
                    
                    old_num_classes=self.original_correct_label.shape[1]                   
                    self.middle_layer=12
                    output,correct_place=self.create_model(old_num_classes+num_class)
                    self.merged_output=output[0]
                    self.merged_correct_place=correct_place
                    self.alpha=0.000001
                    loss,opt,acc=self.optimizer(self.merged_output,trainable_list=self.new_list,correct_place=self.merged_correct_place)

        init = tf.compat.v1.global_variables_initializer()
        with tf.compat.v1.Session() as sess:
            sess.run(init)                          
            self.train_data=self.prepare_data(sess,self.train_data)
            if self.pre_trained==False:
                    np.save(self.path+"train_data-360x48.npy",self.train_data[0])
                    np.save(self.path+"train_data-270x32.npy",self.train_data[1])
                    np.save(self.path+"correct.npy",self.correct_label)
            if os.path.exists(self.path+"model.ckpt-1.data-00000-of-00001")==True and self.pre_trained==True:
                    saver = tf.compat.v1.train.Saver(self.restore_list)
                    saver.restore(sess,tf.train.latest_checkpoint(self.path))
                    None
            #self.pre_trained=False
            self.test_accuracy_list=[]
            self.train_nn(sess,epoch,opt,loss,acc)
    def test(self,positions):
        changed=False
        if self.pre_trained==True:
                self.pre_trained=False
                changed=True
        self.alt_limit=0.95
        self.temp=tf.Variable(np.array([1.],np.float32))
        self.test_data=[]
        self.pre_processing(positions,data_type="test")
        initial_test_batch_size=300
        if self.restored==False:
                correct_label=np.load(self.path+"correct.npy")
                num_class=correct_label.shape[1]
                self.model_output,_=self.create_model(num_class)
                #self.sigmoid_output=self.hidden_layer(self.last_hidden,"sigmoid",0,num_class+1,state="last_hidden",restore=False,layer=-1)
                #self.loss_sig,self.opt_sig,self.acc_sig=self.optimizer(self.sigmoid_output,trainable_list=self.last_kernels,loss_type="binary")
                self.predict=self.argmax(self.model_output)
                self.red_max=self.reduce_max(self.model_output)
                self.test_accuracy=self.accuracy(self.model_output)
                self.unknown_indices=self.probability_limit(self.model_output)
                #self.predict2=self.argmax(self.sigmoid_output)
                #self.test_accuracy2=self.accuracy(self.sigmoid_output)
                #self.unknown_indices2=self.probability_limit(self.sigmoid_output)
                #self.restoration()
                #self.restored=True
                
                
        output_liste=[]
        output_liste2=[]
        test_outputs=[]
        init = tf.compat.v1.global_variables_initializer()
        if self.sess==None:
                self.sess=tf.compat.v1.Session()
                self.sess.run(init)
                #for i in range(len(self.kernels)):
                #        print("geldi assign")
                #        sess.run(self.kernels[i].assign(self.kernel_numpy_arrays[i]))
                if os.path.exists(self.path+"model.ckpt-1.data-00000-of-00001")==True and self.restored==False:
                    
                    saver = tf.compat.v1.train.Saver(self.kernels)
                    saver.restore(self.sess,tf.train.latest_checkpoint(self.path))
                    self.restored=True
        
        for test_data in self.test_data:
                        output_liste=[]
                        step_size=int(test_data[0].shape[0]/initial_test_batch_size)
                        if step_size!=0:
                                artan=int(test_data[0].shape[0]%step_size)
                        else:
                                artan=test_data[0].shape[0]
                        feed={self.keep_prob:0}
                        for batch_index in range(step_size+1):
                                test_batch_size=initial_test_batch_size
                                for place_index in range(len(self.data_place_list)):
                                        try:
                                                feed[self.data_place_list[place_index]]=test_data[place_index][test_batch_size*batch_index:test_batch_size*(batch_index+1)]
                                        except:
                                                print(place_index,test_batch_size,batch_index)
                                                input()
                                        if batch_index==step_size:
                                                feed[self.data_place_list[place_index]]=test_data[place_index][test_batch_size*batch_index:]
                                raw_out,output,unknowns=self.sess.run([self.red_max,self.predict,self.unknown_indices],feed_dict=feed)
                                output=list(output)
                                for i in unknowns:
                                        output[int(i)]=None
                                output_liste.append(output)
                        final_output=np.concatenate(output_liste,axis=0)
                        test_outputs.append(final_output)
        self.test_data=[]
        if changed==True:
                self.pre_trained=True
        return test_outputs,raw_out
                        
                        
                
if __name__=="__main__":

    konumlar_train=[]
    sag_sol={0:"L",1:"R"}
    num_person=10
    baş=61
    indices=[]
    konumlar_test=[]
    test_indices=[]
    taken_train=[]
    index=0
    
    test_indexes=[4,8,11,16,18,20]
    
    for t in range(1,21):#20 adet resim
      if t not in test_indexes:
        for q in range(0,2):# sağ sol  
            index=0
            for i in range(baş+1,baş+num_person+1):#kişi dosyaları
                    if os.path.exists("casia_dataset_examples/"+dg.file_zero(i)+"/"+str(sag_sol[q])+"/S2"
                                          +dg.file_zero(i)+str(sag_sol[q])+dg.file_zero2(t)+".jpg")==True:
                        if q==0:
                            indices.append(index*2)
                        else:
                            indices.append(index*2+1)
                        konumlar_train.append("casia_dataset_examples/"+dg.file_zero(i)+"/"+str(sag_sol[q])+"/S2"
                                              +dg.file_zero(i)+str(sag_sol[q])+dg.file_zero2(t)+".jpg")
                    index+=1
    for t in range(1,21):#20 adet resim
      if t in test_indexes:
        for q in range(0,2):# sağ sol  
            index=0
            for i in range(baş+1,baş+num_person+1):#kişi dosyaları
                    if os.path.exists("casia_dataset_examples/"+dg.file_zero(i)+"/"+str(sag_sol[q])+"/S2"
                                          +dg.file_zero(i)+str(sag_sol[q])+dg.file_zero2(t)+".jpg")==True:
                        if q==0:
                            test_indices.append(index*2)
                        else:
                            test_indices.append(index*2+1)
                        
                        konumlar_test.append("casia_dataset_examples/"+dg.file_zero(i)+"/"+str(sag_sol[q])+"/S2"
                                              +dg.file_zero(i)+str(sag_sol[q])+dg.file_zero2(t)+".jpg")
                    index+=1
        
    sayac=0
    cnn=cnn_architecture()
    cnn.load_pre=False # load_pre degiskeni True durumunda iken daha önceden hazirlanmis olan dataset'i kullanir. Tekrar tekrar fcn ve daugman isleminin gerçeklestirirlmemesi için kolaylik saglar.
    cnn.pre_trained=False # eger model daha önceden egitilmis bir modelin uzerine gelistirilecek ise "True" , eger tamamen yeni bir model olusturulacak ise "False" durumunda olmalidir.
    # True durumunda iken yeni olarak verilen veriler önceden egitilmis modelin üzerine egitilir. en son eklenen kisilerin index degerleri onceden egitilmis olan kisilerin index degerlerinin
    #devami seklindedir. ( Ayni kisilerin tekrardan farkli index degerlerine sahip bir sekilde egitilmediginden emin olunmalidir. Bu durum ayni kisi için birden fazla index degeri olmasina
    #sebep olacagindan beklenmedik sonuçlar olusabilir. )
    #cnn.set_save_path("C:/Users/ibrahim eyyüp inan/Desktop/trained_model/100/")
    num_class=num_person*2
    depth=num_class
    correct_label=cnn.one_hot(indices,depth)
    correct_label_test=cnn.one_hot(test_indices,depth)
    correct_kişiler=[]
    for r in test_indices:
        if r%2==1:
                correct_kişiler.append(baş+(r-1)/2+1)
        else:
                correct_kişiler.append(baş+(r)/2+1)


    def verify(test_accuracy):# verify fonksiyonu verilen degerin uzerinde bir basari elde edilirse islem sonlandirilir.
            if test_accuracy>0.96:
                    return True
            else:
                    return False
    
    
    cnn.set_optimizer("adam")# default olarak tanimlanmis bazi ozellikler degisitirilebilir.
    cnn.regularizator("l2")
    cnn.set_initializer("xavier_normal")
    cnn.save_condition_function(verify)
    cnn.set_konumlar(konumlar_test)
    cnn.save_name("trained_model/save.png")# grafik kaydedilecek konum
    training=False
    testing=True
    if training==True:
            #pre processing methodu verilen konum listesi ve correct label  üzerinde gerekli iþlemleri gerçeklestirir. verilen liste egitim dataset'i ise üçüncü parametre "train"
            # validation set ise "valid" seklinde verilir ( ilk olarak egitim dataset'i verilmelidir ) .
            cnn.pre_processing(konumlar_train,correct_label,"train")
            cnn.pre_processing(konumlar_test,correct_label_test,"valid")
            cnn.train(epoch=40)# bütün verilerin kurulmasinin ardindan train fonksiyonu cagirilarak egitim baslatilir.
    if testing==True:
            print(correct_kişiler)
            mean_list=[]
            for i in range(5):   
                    sonuçlar,raw_out=cnn.test(konumlar_test)
                    #geri dönen ilk parametre yüksek basari degerine sahip olmayan cevaplarin None durumuna getirilmis oldugu parametredir. Ikinci parametrede ise
                    # tamamen degistirilmemis bir sonuc listesi döndürülür.

                    # sisteme vermis egitmis oldugunuz kisilerin index degerlerini asla unutmayin !!!
                    kişiler=[]
                    kişiler2=[]
                    for i in sonuçlar[0]:
                                    if i!=None:
                                            if i%2==1:
                                                  kişiler.append((i-1)/2+1)
                                            else:
                                                  kişiler.append(i/2+1)
                                    else:
                                            kişiler.append(None)
                    print(kişiler)
                    print(len(kişiler))
                    dogru=0
                    unknown=0
                    for r in range(len(correct_kişiler)):
                                    if kişiler[r]==None:
                                            unknown+=1
                                    elif correct_kişiler[r]==kişiler[r]:
                                            dogru+=1
                    print("doğru cevaplar:",dogru)
                    print("bilinmeyen cevaplar:",unknown)
                    print("yanlış cevaplar:",len(correct_kişiler)-dogru-unknown)
            
    
            
            
        
        
            
            
        
        
        




                                             
