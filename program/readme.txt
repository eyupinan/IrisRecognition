
-programin kullanilabilir hale getirilebilmesi i�in ilk olarak fcn_segmentation.py kodunun �alistirilmasi gerekir.
Bu kod ground_truth klas�r� i�erisindeki dataya g�re bir fcn modeli olusturup fcn_model klas�r�n�n i�erisine kaydeder.
Fcn modelinin olusturulmasinin ardindan multi_cnn �alistirilarak model model egitilebilir hale gelir.
Egitilen model trained_model klas�r� i�erisine kaydedilir.

-Program tensorflow k�t�phanesinin 2.1 versiyonunu kullanmaktadir. Programin efektif bir sekilde kullanilabilmesi i�in 
ekran karti kulanim desteginin saglanmasi gerekmektedir. Ekran karti desteginin saglanabilmesi i�in NVIDIA ekran karti i�in 
Cuda 10.1 versiyonunun ve cudNN mod�llerinin y�kl� olmasi gerekmektedir. GPU kullanimi desteginin saglanmamasi durumunda 
program yavas �alisacaktir.

-KULLANIM

Fcn modelinin egitiminden �nce datanin hazirlanma asamasinin uzun s�rmesinden dolayi egitim asamasinin baslamasi zaman alabilir.
Fcn modelinin ger�ek�i bir model olmasi i�in kayip degerinin sifira �ok yakin bir deger olmasi gerekir. Baslangi�ta default olarak 
100 epoch ilerleme kaydedilir. Islem kullanicinin istegine g�re devam ettirilebilir. Eger son verilmek isteniyor ise "devam edilsin mi" sorusu soruldugunda 
bir epoch sayisi girilir ve isleme son verilmek isteniyor ise "e" karakteri girilir. "e" karakterinin girilmesinin ardindan bir fcn modeli olusur.
Fcn modelinin �rnek kullanimi fcn_save.py klas�r� i�erisinde �rnek kod olarak g�sterilmistir.
Egitilmis bir Fcn modeli fcn_model klas�r� i�erisinde bulunmaktadir.

Daugman's_rubber.py dosyasi kendisine verilen adresteki resmi fcn modelini kullanarak normalize eder. �rnek kod Daugman's_rubber.py dosyasinin
alt kisminda verilmistir.

iris recognization kendisine verilen correct label ve dataset i�in konumlar listesini kullanarak bir iris tanima modeli olusturur.
daha �nceden egitilmis bir modelin �zerine yeni bir egitim yapilabilmesi i�in pre_trained degiskeninin "True" durumuna getirilmesi gerekir.
�nceden egitilmis olan datasetin hazirlanmis verileri trained_model klas�r� i�erisinde saklanir. Ayrintili bilgi iris_recognization.py dosyasi
i�erisinde bulunan �rnek kod �zerinde verilmistir.
- GEREKLI VERILER
 * /ground_truth klas�r� i�erisinde Fcn modelinin egitimi i�in gerekli olan dataset bulunmasi gerekmektedir. Bu dataset i�erisinde herbir
resim i�in bir "image.jpg" adi ile �rnek, sadece iris kisminin beyaz geriye kalan kisimlarin siyah oldugu "iris.jpg" ve ayni sekilde
sadece g�z bebeginin beyaz geriye kalan kisimlarin siyah oldugu "pupil.jpg" resimlerinin bulunmasi gerekmektedir.
 * /casia_dataset_examples klas�r� i�inde sag ve sol g�ze ait resimler olmak �zere dataset i�erigi bulunmasi gerekmektedir.
 * Projenin kullanimindan �nce ilk olarak FCN model �retilmesi gerekmektedir. Bunun i�in fcn_segmentation.py �alistirilmalidir.
 * Fcn modeli olusturulduktan sonra Iris_recognization.py dosyasinin �alistirilmasi gerekmektedir.




