
-programin kullanilabilir hale getirilebilmesi için ilk olarak fcn_segmentation.py kodunun çalistirilmasi gerekir.
Bu kod ground_truth klasörü içerisindeki dataya göre bir fcn modeli olusturup fcn_model klasörünün içerisine kaydeder.
Fcn modelinin olusturulmasinin ardindan multi_cnn çalistirilarak model model egitilebilir hale gelir.
Egitilen model trained_model klasörü içerisine kaydedilir.

-Program tensorflow kütüphanesinin 2.1 versiyonunu kullanmaktadir. Programin efektif bir sekilde kullanilabilmesi için 
ekran karti kulanim desteginin saglanmasi gerekmektedir. Ekran karti desteginin saglanabilmesi için NVIDIA ekran karti için 
Cuda 10.1 versiyonunun ve cudNN modüllerinin yüklü olmasi gerekmektedir. GPU kullanimi desteginin saglanmamasi durumunda 
program yavas çalisacaktir.

-KULLANIM

Fcn modelinin egitiminden önce datanin hazirlanma asamasinin uzun sürmesinden dolayi egitim asamasinin baslamasi zaman alabilir.
Fcn modelinin gerçekçi bir model olmasi için kayip degerinin sifira çok yakin bir deger olmasi gerekir. Baslangiçta default olarak 
100 epoch ilerleme kaydedilir. Islem kullanicinin istegine göre devam ettirilebilir. Eger son verilmek isteniyor ise "devam edilsin mi" sorusu soruldugunda 
bir epoch sayisi girilir ve isleme son verilmek isteniyor ise "e" karakteri girilir. "e" karakterinin girilmesinin ardindan bir fcn modeli olusur.
Fcn modelinin örnek kullanimi fcn_save.py klasörü içerisinde örnek kod olarak gösterilmistir.
Egitilmis bir Fcn modeli fcn_model klasörü içerisinde bulunmaktadir.

Daugman's_rubber.py dosyasi kendisine verilen adresteki resmi fcn modelini kullanarak normalize eder. Örnek kod Daugman's_rubber.py dosyasinin
alt kisminda verilmistir.

iris recognization kendisine verilen correct label ve dataset için konumlar listesini kullanarak bir iris tanima modeli olusturur.
daha önceden egitilmis bir modelin üzerine yeni bir egitim yapilabilmesi için pre_trained degiskeninin "True" durumuna getirilmesi gerekir.
Önceden egitilmis olan datasetin hazirlanmis verileri trained_model klasörü içerisinde saklanir. Ayrintili bilgi iris_recognization.py dosyasi
içerisinde bulunan örnek kod üzerinde verilmistir.
- GEREKLI VERILER
 * /ground_truth klasörü içerisinde Fcn modelinin egitimi için gerekli olan dataset bulunmasi gerekmektedir. Bu dataset içerisinde herbir
resim için bir "image.jpg" adi ile örnek, sadece iris kisminin beyaz geriye kalan kisimlarin siyah oldugu "iris.jpg" ve ayni sekilde
sadece göz bebeginin beyaz geriye kalan kisimlarin siyah oldugu "pupil.jpg" resimlerinin bulunmasi gerekmektedir.
 * /casia_dataset_examples klasörü içinde sag ve sol göze ait resimler olmak üzere dataset içerigi bulunmasi gerekmektedir.
 * Projenin kullanimindan önce ilk olarak FCN model üretilmesi gerekmektedir. Bunun için fcn_segmentation.py çalistirilmalidir.
 * Fcn modeli olusturulduktan sonra Iris_recognization.py dosyasinin çalistirilmasi gerekmektedir.




