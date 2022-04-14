import numpy as np
from numpy import load
import math
import PIL
from PIL import Image,ImageOps
import fcn_save
import time
import tensorflow as tf
import random
import sys
import gc
from multiprocessing import Process,Queue
import multiprocessing
import os, signal
def draw_points(img,points,name,colored=False,pos=0):
        print("img shape:",img.shape)
        img_arr=np.expand_dims(img,axis=2)
        img_arr=np.concatenate((img_arr,img_arr,img_arr),axis=2)
        color1=255
        color2=0
        color3=0
        last=None
        print("reshaped:",img_arr.shape)
        for i in points:
            #print(i)
        
            for q in range(int(i[0])-1,int(i[0])+2):
                for t in range(int(i[1])-1,int(i[1])+2):    
                    img_arr[q][t][0]=color1
                    img_arr[q][t][1]=color2
                    img_arr[q][t][2]=color3
                    if pos==0 and colored==True and last!=t:
                        color1+=5
                        color2-=5
                        color3-=5
                        last=t
                if pos==1 and colored==True and last!=q:
                      color1+=5
                      color2-=5
                      color3-=5
                      last=q
        
                    
        img_yeni=Image.fromarray(img_arr)
        img_yeni.save(name)
        "C:/Users/ibrahim eyyüp inan/Desktop/pupil deneme.jpg"
def best_points(liste,std_limit=3):
        # bu fonksiyon verilmiş olan liste içerisindeki rakamların standart sapması std_limit parametresi ile alınan değerin altına inmesi için standart
        # sapmayı kötü etkileyen noktaları bulur ve listeden çıkarır
        std=np.std(liste)
        if std>std_limit:
                mean=np.mean(liste)
                minimum=np.amin(liste)
                maximum=np.amax(liste)
                min_index = np.where(liste == minimum)
                max_index = np.where(liste == maximum)
                min_diff=abs(mean-minimum)
                max_diff=abs(mean-maximum)
                if max_diff>min_diff:
                        yeni_liste=np.delete(liste,max_index)
                else:
                        yeni_liste=np.delete(liste,min_index)
                sonuc=best_points(yeni_liste)
                return sonuc
        else:
                return liste
def equation(point1,point2,point3,point4):
                #bu fonksiyon verilen dört noktadan 2 doğru üretir bu doğruların orta noktalarından bir merkez noktasi bulur
                point1=point1.tolist()
                point2=point2.tolist()
                point3=point3.tolist()
                point4=point4.tolist()
                line1_orta=[(point2[0]+point1[0])/2,(point2[1]+point1[1])/2]
                line2_orta=[(point4[0]+point3[0])/2,(point4[1]+point3[1])/2]
                shifted_point1=[point1[0]-line1_orta[0],point1[1]-line1_orta[1]]
                shifted_point2=[point2[0]-line1_orta[0],point2[1]-line1_orta[1]]
                shifted_point3=[point3[0]-line2_orta[0],point3[1]-line2_orta[1]]
                shifted_point4=[point4[0]-line2_orta[0],point4[1]-line2_orta[1]]
                rotated_point1=[shifted_point1[1]+line1_orta[0],-1*shifted_point1[0]+line1_orta[1]]
                rotated_point2=[shifted_point2[1]+line1_orta[0],-1*shifted_point2[0]+line1_orta[1]]
                rotated_point3=[shifted_point3[1]+line2_orta[0],-1*shifted_point3[0]+line2_orta[1]]
                rotated_point4=[shifted_point4[1]+line2_orta[0],-1*shifted_point4[0]+line2_orta[1]]
                ## döndürülmüş noktalar bulundu
                if (rotated_point2[1]-rotated_point1[1])!=0 and (rotated_point4[1]-rotated_point3[1])!=0:
                        line1_katsayi1=(rotated_point2[0]-rotated_point1[0])/(rotated_point2[1]-rotated_point1[1])
                        line2_katsayi1=(rotated_point4[0]-rotated_point3[0])/(rotated_point4[1]-rotated_point3[1])
                        line1_katsayi2=-1*line1_katsayi1*rotated_point2[1]+rotated_point2[0]
                        line2_katsayi2=-1*line2_katsayi1*rotated_point4[1]+rotated_point4[0]
                        
                        kesişim_katsayi1=line1_katsayi1-line2_katsayi1
                        kesişim_katsayi2=line2_katsayi2-line1_katsayi2
                        if kesişim_katsayi1!=0:
                                x=kesişim_katsayi2/kesişim_katsayi1
                                y=x*line1_katsayi1+line1_katsayi2
                                return [y,x]
                        else:
                                return [-1,-1]
                else:
                        return [-1,-1]
        
        
def center_position_finder(point_list):
    ## bu fonksiyon verilmiş olan noktalara göre bir merkez noktası bulur
    # verilen noktaların her ikisi arasında bir doğru oluşturur ve bu doğruyu tam ortasından dik kesen yeni bir doğru bulur.
    # bu şekilde bulunan doğruların kesişim noktası göz bebeğinin merkezidir. Fcn aşamasının hata yapması durumuna karşılık ikiden fazla doğrunun
    # kesişim noktası bulunur. Eğer bulunan noktaların standart sapması yüksek ise standart sapmanın yüksek olmasına sebep olan noktalar best_points fonksiyonu
    # ile bulunur ve devre dışı bırakılır
    center_list=[]
    for i in range(int(len(point_list)/2)):
            if i!=int(len(point_list)/2-1):
                    cent=equation(point_list[i],point_list[int(len(point_list)/2)+i],point_list[i+1],point_list[int(len(point_list)/2)+i+1])
                    if cent!=[-1,-1]:
                            center_list.append(cent)
            else:
                    cent=equation(point_list[i],point_list[int(len(point_list)/2)+i],point_list[0],point_list[int(len(point_list)/2)])
                    if cent!=[-1,-1]:
                            center_list.append(cent)
    center=[-1,-1]
    toplam_x=0
    toplam_y=0
    x=[]
    y=[]
    for i in center_list:
           try:
                   x.append(i[1])
                   y.append(i[0])
           except:
                   print("error point:")
                   print(i)
                   #print(point_list)
                   #input()
    best_x=best_points(x)
    best_y=best_points(y)
    center=[np.mean(best_y),np.mean(best_x)]
    #print(center)
    return center
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
def range_creator(boyut,aralık):
        liste=[]
        for i in range(boyut-1):
                ran=[int(aralık/boyut*(i+1)),False]
                liste.append(ran)
        return liste
class daugman(object):
    def __init__(self):
        self.error_list=[]
        self.pupil_center=[0,0]
        self.degree=0
        tf.compat.v1.disable_eager_execution()
        #tf.debugging.set_log_device_placement(True)
        self.graph_creator()
        self.fcn_class=fcn_save.fcn_segmentation()
        #self.set_img(konumlar)
        #print("fcn_class oluşturuldu")
        #input()
    def clear(self):
        self.fcn_class.clear()
    def rounder(self,sayi):
        if (sayi-float(int(sayi)))>(float(int(sayi+1))-sayi):
            return int(sayi)+1
        else:
            if int(sayi)!=0:
                return int(sayi)
            else:
                return 1
    def set_img(self,konum_img):
        tf.compat.v1.disable_eager_execution()
        tf.debugging.set_log_device_placement(True)
        self.img=[]
        kontrol_img=np.asarray(PIL.Image.open(konum_img[0]).resize((480,360)).convert("L"))
        self.img=np.zeros((len(konum_img),kontrol_img.shape[0],kontrol_img.shape[1]),np.uint8)
        self.images=[]
        
        for i in range(len(konum_img)):
            image=PIL.Image.open(konum_img[i]).resize((480,360)).convert("L")
            self.images.append(image)
            self.img[i]=np.asarray(image)

        self.segmented_irises,self.segmented_pupils=self.fcn_class.run(self.images)
        self.images=[]

    def pupil_center_detector(self,images):
        
        bulunacak_nokta=24
        center_list=np.zeros((images.shape[0],2),np.float32)
        radius_list=np.zeros((images.shape[0]),np.float32)
        positions=np.zeros((images.shape[0],bulunacak_nokta,2),np.float32)
        index_pupil=0
        shapes=images[0].shape
        index_img=0
        giris_sayisi=0
        first=True
        for self.segmented_pupil in images:
                ## bu döngünün içerisinde resim belirli rakamlarda parçalara bölünür. Bu sayede resmin tamamı irdelenmeden göz bebeğinin konumu hakkında tahmini bir pozisyon elde edilir.
                ## Bu işlemin amacı her bir göz bebeğinin yerinin tam olarak tespit edilmesinin gereksiz yorucu olmasıdır.
                ## standart bir for öngüsü ile 640x480 boyutlarında 20000 resim için tahmini olarak 80 milyon kez döngüye girilmesi gerekirken
                ## bu işlem yaklaşık 12*20*20000/100*80= 4 milyon kez çalışır ( 12 ve 20 parçalama sayısı ve yüzde 80 iç kesimlerde bulunma miktarı (tahmini değerler))
                center=[-1,-1]
                
                yükseklik=shapes[0]
                genislik=shapes[1]
                parçalama_hei=24
                parçalama_wid=30
                çizim_list=[]
                range_list_hei=range_creator(parçalama_hei,yükseklik)
                range_list_wid=range_creator(parçalama_wid,genislik)
                for i in range_list_hei:
                        for q in range_list_wid:
                                çizim_list.append([i[0],q[0]])
                                giris_sayisi+=1
                                if self.segmented_pupil[i[0]][q[0]]>128:
                                        i[1]=True
                                        q[1]=True
                aralik_liste_hei=[]
                aralik_liste_wid=[]
                try:
                        for i in range_list_hei:
                                if i[1]==True:
                                        aralik_liste_hei.append(i[0])

                        if len(aralik_liste_hei)==1:
                                aralik_liste_hei=[aralik_liste_hei[0]-int(yükseklik/parçalama_hei),aralik_liste_hei[0]+int(yükseklik/parçalama_hei)]
                        else:
                                aralik_liste_hei=[aralik_liste_hei[0],aralik_liste_hei[-1]]
                        for i in range_list_wid:
                                if i[1]==True:
                                        aralik_liste_wid.append(i[0])

                        if len(aralik_liste_wid)==1:
                                aralik_liste_wid=[aralik_liste_wid[0]-int(genislik/parçalama_wid),aralik_liste_wid[0]+int(genislik/parçalama_wid)]
                        else:
                                aralik_liste_wid=[aralik_liste_wid[0]-int(genislik/parçalama_wid),aralik_liste_wid[-1]+int(genislik/parçalama_wid)]
                except:
                        print("hata oluştu!")
                        print(range_list_hei)
                        print(range_list_wid)
                        print(index_img)
                        self.error_list.append(index_img)
                        #programın hata vermemesi için default değerler atanır
                        shapes=self.segmented_pupil.shape
                        center_list[index_img]=[int(shapes[0]/2),int(shapes[1]/2)]
                        radius_list[index_img]=10
                        
                        index_img+=1
                        continue
                bulunan_nokta=0      
                while bulunan_nokta<int(bulunacak_nokta/2):
                       tahmini_hei=random.randint(aralik_liste_hei[0],aralik_liste_hei[1])
                       for wid in range(aralik_liste_wid[0],aralik_liste_wid[1]):
                               giris_sayisi+=1
                               if self.segmented_pupil[tahmini_hei][wid]>128:
                                       positions[index_img][bulunan_nokta]=[tahmini_hei,wid]
                                       bulunan_nokta+=1
                                       break
                while bulunan_nokta<bulunacak_nokta:
                       tahmini_hei=random.randint(aralik_liste_hei[0],aralik_liste_hei[1])
                       for wid in range(aralik_liste_wid[1],aralik_liste_wid[0],-1):
                               giris_sayisi+=1
                               if self.segmented_pupil[tahmini_hei][wid]>128:
                                       positions[index_img][bulunan_nokta]=[tahmini_hei,wid]
                                       bulunan_nokta+=1
                                       break
                
                #if first==True:
                #       draw_points(self.segmented_pupil,positions[index_img],"C:/Users/ibrahim eyyüp inan/Desktop/pupil position.jpg")
                #        first=False
                index_img+=1
        #print("noktalar bitiş:",time.localtime())
        
        for index_img in range(len(images)):
                center=center_position_finder(positions[index_img])
                yari_cap=[]
                for i in range(len(positions[index_img])):
                        veri=math.sqrt(pow(abs(center[0]-positions[index_img][i][0]),2)+pow(abs(center[1]-positions[index_img][i][1]),2))
                        yari_cap.append(veri)
                best_yari_cap=best_points(yari_cap)
                pupil_radius=np.mean(best_yari_cap)
                radius_list[index_img]=pupil_radius
                center_list[index_img]=center[:]

        return center_list,radius_list
    def iris_range(self,degree=0.1):
        sample_count=self.img.shape[0]
        beginning_iris=np.zeros((sample_count,2),np.float32)
        end_iris=np.zeros((sample_count,2),np.float32)
        range_length=np.zeros((sample_count,1),np.float32)
        if degree!=0.1:
            self.degree=degree
        in_pupil=True
        for q in range(sample_count):
                try:
                        if math.isnan(self.pupil_center[q][0]) or math.isnan(self.pupil_center[q][1]):
                                self.pupil_center[q]=[100,100]
                        if q in self.error_list:
                                raise Exception("center noktası hesaplaması esnasında hata!")
                        position=np.zeros((sample_count,2),np.float32)
                        in_pupil=True
                        while True:
                                if in_pupil==True:
                                        if self.segmented_irises[q][int(self.pupil_center[q][0]-position[q][0])][int(self.pupil_center[q][1]+position[q][1])]<128:# position 0 y eksenini ifade eder. Bu sebeple negatif bir şekilde eklenir
                                            None
                                        if self.segmented_irises[q][int(self.pupil_center[q][0]-position[q][0])][int(self.pupil_center[q][1]+position[q][1])]>128:
                                            beg=[int(self.pupil_center[q][0]-position[q][0]),int(self.pupil_center[q][1]+position[q][1])]
                                            beginning_iris[q]=beg
                                            in_pupil=False
                                else:
                                        if self.segmented_irises[q][int(self.pupil_center[q][0]-position[q][0])][int(self.pupil_center[q][1]+position[q][1])]>128:
                                            None
                                        if self.segmented_irises[q][int(self.pupil_center[q][0]-position[q][0])][int(self.pupil_center[q][1]+position[q][1])]<128:
                                            end=[int(self.pupil_center[q][0]-position[q][0]),int(self.pupil_center[q][1]+position[q][1])]
                                            end_iris[q]=end
                                            break
                                    
                                position[q][0]=position[q][0]+math.sin(math.radians(self.degree))
                                position[q][1]=position[q][1]+math.cos(math.radians(self.degree))
                        range_length[q]=math.sqrt(pow(abs(beginning_iris[q][0]-end_iris[q][0]),2)+pow(abs(beginning_iris[q][1]-end_iris[q][1]),2))
                except Exception as e:
                        print("iris range hata!")
                        print(e)
                        print(q)
                        self.error_list.append(q)
                        print(self.pupil_center[q])
                        print(position[q])
                        shapes=self.segmented_irises[q].shape
                        # görüntüde sorun oluşması durumunda hatalılar bir listeye alınır ve değerleri programın hata vermesine engel olacak değerler atanır
                        if shapes[0]>shapes[1]:
                                range_length[q]=int(shapes[1]/5)
                        elif shapes[0]<shapes[1]:
                                range_length[q]=int(shapes[0]/5)
                

        return beginning_iris,end_iris,range_length
    def graph_creator(self):
            
            for i in range(1):
                    self.graph1=tf.Graph()
                    with self.graph1.as_default() as g:            
                        self.radius_place=tf.compat.v1.placeholder(tf.float32,[None],name="radius_place")
                        self.center_place=tf.compat.v1.placeholder(tf.float32,[None,2],name="center_place")
                        self.img_place=tf.compat.v1.placeholder(tf.float32,[None,None,None],name="img_place")
                        prepare_center=tf.reshape(self.center_place,[-1,2,1],name="prepare_center")
                        self.step_data=tf.compat.v1.placeholder(tf.float32,[None,1,None],name="step_data")
                        self.radians_place=tf.compat.v1.placeholder(tf.float32,[None,1],name="radians_place")
                        self.width_place=tf.compat.v1.placeholder(tf.int32,[],name="width_place")
                        self.height_place=tf.compat.v1.placeholder(tf.int32,[],name="height_place")
                        sinus=-tf.math.sin(self.radians_place,name="sinus")
                        cosinus=tf.math.cos(self.radians_place,name="cosinus")
                        repeat_sin=tf.repeat(sinus,repeats=self.height_place,axis=1,name="repeat_sin")## radians değerlerini üretir [None,1,normalized_img_height] data ile çarpılabilir hale getirir.
                        repeat_cos=tf.repeat(cosinus,repeats=self.height_place,axis=1,name="repeat_cos")
                        sin_mul=tf.expand_dims(tf.math.multiply(repeat_sin,self.step_data),1,name="sin_mul")
                        cos_mul=tf.expand_dims(tf.math.multiply(repeat_cos,self.step_data),1,name="cos_mul")
                        merged=tf.concat([sin_mul,cos_mul],1,name="merged")
                        center_reshape=tf.reshape(self.center_place,[-1,2,1,1],name="center_reshape")
                        center_repeat=tf.repeat(center_reshape, repeats=self.width_place, axis=2,name="center_repeat")
                        center_repeat2=tf.repeat(center_repeat,repeats=self.height_place,axis=3,name="center_repeat2")
                        adder=tf.math.add(center_repeat2,merged,name="adder")
                        transpose=tf.transpose(adder,perm=[0,2,3,1],name="transpose")
                        self.casted=tf.cast(transpose,dtype=tf.int32,name="casted")
                        gather=tf.gather_nd(self.img_place,self.casted,1,name="gather")
                        self.transpose2=tf.transpose(gather,perm=[0,2,1],name="transpose2")
            self.graph1.finalize()
    #@profile
    def normalization(self,boyut_listesi):
        from tensorflow.python.client import timeline
        self.error_list=[]
        sample_count=self.img.shape[0]
        self.pupil_center,self.pupil_radius=self.pupil_center_detector(self.segmented_pupils)
        boundaries_right=self.iris_range(0)
        boundaries_left=self.iris_range(180)
        range_length_right=boundaries_right[2]
        range_length_left=boundaries_left[2]
        self.segmented_irises=[]
        self.segmented_pupils=[]
        range_length=np.multiply(np.add(range_length_right,range_length_left),0.5)
        # bu aşamaya kadar irisin fiziksel konumu hakkında bilgi toplanıyor
        # bu aşamadan sonra fonksiyona parametre olarak verilmiş olan boyutlarda işlemler gerçekleştiriliyor


        
        
        #gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.5)
        #config = tf.compat.v1.ConfigProto(allow_soft_placement=True,log_device_placement=True,gpu_options=gpu_options)
        #config.gpu_options.allow_growth=True
        with tf.compat.v1.Session(graph=self.graph1) as sess:
                veri_listesi=[]
                boyut_index=0
                for boyut in boyut_listesi:
                        normalized_img_height=boyut[0]
                        normalized_img_width=boyut[1]
                        radian_step=360/normalized_img_width
                        radians=np.radians(np.arange(0,float(360)-radian_step/2,radian_step)).astype(np.float32).reshape((normalized_img_width,1))
                        step=(range_length.reshape((range_length.shape[0],)))/normalized_img_height
                        all_steps=np.zeros((sample_count,normalized_img_height),np.float32)
                        for index in range(self.pupil_radius.shape[0]):
                                all_steps[index]=np.arange(self.pupil_radius[index],range_length[index][0]+self.pupil_radius[index]-(step[index]/2),step[index])
                        all_steps=all_steps.reshape((sample_count,1,normalized_img_height))
                        self.pupil_center=self.pupil_center.astype(np.float32)
                        self.pupil_radius=self.pupil_radius.astype(np.float32)
                        batch_size=100
                        if batch_size!=0:
                                artan=sample_count%batch_size
                        else:
                                artan=sample_count
                        adim_sayisi=int(sample_count/batch_size)
                        #print("artan:",artan)
                        sonuc_np=np.zeros((sample_count,normalized_img_height,normalized_img_width),np.uint8)
                        data_index=0
                        for i in range(adim_sayisi):
                                sonuc=sess.run(self.transpose2,feed_dict={self.center_place:self.pupil_center[i*batch_size:(i+1)*batch_size],self.step_data:all_steps[i*batch_size:(i+1)*batch_size],
                                                             self.radians_place:radians,self.img_place:self.img[i*batch_size:(i+1)*batch_size],self.height_place:normalized_img_height,
                                                                          self.width_place:normalized_img_width})
                                sonuc=sonuc.astype(np.uint8)
                                for i in range(sonuc.shape[0]):
                                        sonuc_np[data_index]=sonuc[i]
                                        #image=Image.fromarray(sonuc[i])
                                        #image.save("C:/Users/ibrahim eyyüp inan/Desktop/segmented/normalized2/"+str(boyut_index)+"/normalized_img"+str(data_index)+".jpg")
                                        data_index+=1
                        if artan>0:
                                sonuc=sess.run(self.transpose2,feed_dict={self.center_place:self.pupil_center[adim_sayisi*batch_size:],self.step_data:all_steps[adim_sayisi*batch_size:],
                                                             self.radians_place:radians,self.img_place:self.img[adim_sayisi*batch_size:],self.height_place:normalized_img_height,
                                                                          self.width_place:normalized_img_width})
                                sonuc=sonuc.astype(np.uint8)
                                for i in range(sonuc.shape[0]):
                                        sonuc_np[data_index]=sonuc[i]
                                        #image=Image.fromarray(sonuc[i])
                                        #image.save("C:/Users/ibrahim eyyüp inan/Desktop/segmented/normalized2/"+str(boyut_index)+"/normalized_img"+str(data_index)+".jpg")
                                        data_index+=1
                        boyut_index+=1
                        # hatalı değerler sonuctan çıkartılıyor
                        sonuc_np=np.delete(sonuc_np,self.error_list,axis=0)
                        veri_listesi.append(sonuc_np)
                sess.close()

        self.img=[]
        return veri_listesi,self.error_list
    

if __name__=="__main__":
        obj=daugman()
        obj.set_img(["casia_dataset_examples/001/R/S2001R18.jpg"])
        data,err=obj.normalization([[64,360]])
        img=Image.fromarray(data[0][0])
        img.save("daugman.jpg")
        
