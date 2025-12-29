1. Proje Konusu ve Seçilme Gerekçesi 

Bu projenin amacı, derin öğrenme tabanlı görüntü sınıflandırma problemini gerçek bir veri seti üzerinde uygulamalı olarak incelemektir. Günümüzde görüntü işleme ve bilgisayarla görme alanları; güvenlik sistemleri, e-ticaret, medikal görüntüleme ve otonom sistemler gibi birçok alanda yaygın olarak kullanılmaktadır. Bu nedenle görüntü sınıflandırma problemleri, derin öğrenmenin en temel ve önemli uygulama alanlarından biridir.

Bu projede, klasik MNIST veri setinin daha zor ve gerçekçi bir versiyonu olan FashionMNIST veri seti seçilmiştir. FashionMNIST; tişört, pantolon, ayakkabı, çanta gibi 10 farklı giyim ürününü içeren gri seviye görüntülerden oluşmaktadır. Bu veri seti, hem akademik çalışmalarda hem de derin öğrenme eğitimlerinde yaygın olarak kullanılmaktadır.

Daha önce bu alanda yapılan çalışmalarda, özellikle Evrişimsel Sinir Ağları (CNN) kullanılarak yüksek doğruluk oranlarına ulaşıldığı görülmektedir. Bu proje kapsamında da literatürde sıklıkla kullanılan CNN mimarileri incelenmiş ve uygulamalı bir model geliştirilmiştir.

Bu alanın önemi; derin öğrenme algoritmalarının gerçek dünya problemlerine nasıl uyarlanabileceğini göstermesi ve teorik bilgilerin pratikte test edilebilmesi açısından oldukça yüksektir.

2. Veri Setinin Belirlenmesi

Projede kullanılan veri seti FashionMNIST’tir. Bu veri seti, Zalando tarafından hazırlanmış olup, MNIST formatına uygun şekilde düzenlenmiştir.

Veri setinin özellikleri:

Toplam 70.000 görüntü

60.000 eğitim, 10.000 test örneği

Görüntü boyutu: 28×28

Görüntüler: gri seviye (1 kanal)

Toplam 10 sınıf

Sınıflar:

T-shirt/Top

Trouser

Pullover

Dress

Coat

Sandal

Shirt

Sneaker

Bag

Ankle Boot

FashionMNIST’in tercih edilme sebebi; klasik MNIST’e göre daha zor olması ve sınıflar arasında görsel benzerliklerin bulunmasıdır. Bu durum, modelin gerçekten anlamlı özellikler öğrenmesini zorunlu kılmaktadır.

3. Uygulanan Yöntem / Algoritma ve Seçim Gerekçesi 

Bu projede Evrişimsel Sinir Ağı (Convolutional Neural Network – CNN) kullanılmıştır. CNN’ler, özellikle görüntü verileri üzerinde başarılı sonuçlar veren derin öğrenme mimarileridir.

Literatürde:

Tam bağlantılı yapay sinir ağları (MLP),

Destek Vektör Makineleri (SVM),

K-En Yakın Komşu (KNN)

gibi yöntemler FashionMNIST üzerinde denenmiş olsa da, CNN mimarilerinin bu yöntemlere kıyasla çok daha yüksek doğruluk sağladığı bilinmektedir.

CNN’lerin tercih edilme sebepleri:

Uzamsal özellikleri otomatik öğrenebilmesi

Filtreler sayesinde kenar, şekil ve desenleri algılayabilmesi

Parametre sayısının daha kontrollü olması

Görüntü sınıflandırmada literatür standardı olması

Bu projede iki evrişim katmanı, batch normalization, ReLU aktivasyon fonksiyonları, max pooling ve tam bağlantılı katmanlardan oluşan bir CNN mimarisi kullanılmıştır.

4. Model Eğitimi ve Model Değerlendirilmesi 

Model, PyTorch kütüphanesi kullanılarak eğitilmiştir. Eğitim sırasında:

Kayıp fonksiyonu: Cross Entropy Loss

Optimizasyon algoritması: Adam Optimizer

Öğrenme oranı: 0.001

Epoch sayısı: 5

Batch size: 100

Model eğitimi sırasında kayıp değeri düzenli olarak azalmış, doğruluk oranı ise artış göstermiştir. Eğitim tamamlandıktan sonra model, test veri seti üzerinde değerlendirilmiştir.

Performans Sonuçları:

Accuracy: %91

Precision (ortalama): 0.91

Recall (ortalama): 0.91

F1-score (ortalama): 0.91

<img width="1600" height="1400" alt="confusion_matrix" src="https://github.com/user-attachments/assets/8cddc4d2-5839-4e8c-96f8-40dc244e1d61" />


Confusion matrix incelendiğinde özellikle Shirt, T-shirt ve Pullover sınıflarında karışıklık olduğu görülmektedir. Bunun sebebi bu sınıfların görsel olarak birbirine çok benzemesidir.

5. Karşılaşılan Problemler ve Çözüm Yöntemleri

Proje sırasında önemli bir problemle karşılaşılmıştır. Model, test veri setinde yüksek doğruluk sağlamasına rağmen, dışarıdan verilen MNIST benzeri görüntülerde tüm tahminleri “Bag” sınıfı olarak yapmıştır.

Yapılan inceleme sonucunda problemin sebebinin girdi görüntülerinin arka plan–ön plan dağılımının eğitim verisiyle ters olması olduğu anlaşılmıştır. FashionMNIST veri setinde görüntüler siyah arka plan üzerinde beyaz nesnelerden oluşurken, dışarıdan verilen görüntüler beyaz arka plan üzerinde siyah nesneler içermektedir.

Bu durum literatürde domain shift (dağılım uyuşmazlığı) olarak adlandırılmaktadır.

Çözüm olarak, tahmin aşamasında görüntüler gri seviye olarak ters çevrilmiş (invert edilmiştir). Bu düzeltmeden sonra model, dış görüntüler üzerinde de doğru tahminler yapmaya başlamıştır.

6. Proje Dokümantasyonu ve GitHub Yapısı

Proje, modüler ve düzenli bir şekilde yapılandırılmıştır. Tüm kodlar, grafikler ve rapor dosyaları GitHub üzerinde paylaşılmıştır.

Klasör yapısı:

fashion_project/
│
├── model.py
├── train.py
├── evaluate.py
├── serve.py
├── app.py (Gradio arayüzü)
│
├── models/
├── results/
├── data/
│
├── README.md
└── report.pdf


Bu yapı, kodun okunabilirliğini ve yeniden kullanılabilirliğini artırmaktadır.

7. Projenin Sunumu ve Arayüz 

Modelin kullanımını kolaylaştırmak için Gradio kütüphanesi kullanılarak web tabanlı bir arayüz geliştirilmiştir. Bu arayüz sayesinde kullanıcılar kendi görüntülerini yükleyerek modelin tahminlerini ve sınıf olasılıklarını görebilmektedir.

Arayüzde:

Görüntü yükleme

Tahmin edilen sınıf

Sınıf olasılıkları (confidence)

Kullanıcı dostu tasarım

özellikleri bulunmaktadır.

8. Sonuç ve Gelecek Çalışmalar

Bu projede, FashionMNIST veri seti üzerinde CNN tabanlı bir derin öğrenme modeli başarıyla geliştirilmiştir. Model, test verisi üzerinde %91 doğruluk oranına ulaşmış ve literatürdeki benzer çalışmalarla uyumlu sonuçlar üretmiştir.

Ayrıca, gerçek dünya görüntülerinde karşılaşılan dağılım uyuşmazlığı problemi tespit edilmiş ve uygun ön işleme adımları ile çözülmüştür.

Gelecek çalışmalarda:

Daha derin CNN mimarileri,

Transfer learning (ResNet, MobileNet),

Veri artırma (augmentation),

Gerçek renkli görüntüler

kullanılarak performans daha da artırılabilir.
