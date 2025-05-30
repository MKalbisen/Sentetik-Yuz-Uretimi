## Sentetik-Yuz-Uretimi

# GAN Yüz Üretici

Bu proje, CelebA veri seti kullanarak yüz görüntüleri üreten bir GAN (Generative Adversarial Network) modelini eğitmek için tasarlanmıştır.

## Özellikler

- **64x64 çözünürlükte yüz üretimi**
- **Checkpoint sistemi ile eğitime devam etme**
- **İnteraktif checkpoint seçimi**
- **Eğitim sürecinin görselleştirilmesi**
- **Otomatik veri seti çıkarma ve yükleme**
- **GPU desteği**

## Kurulum

### Gereksinimler

Python 3.7 veya üzeri gereklidir.

```bash
pip install -r requirements.txt
```

### Dosya Yapısı

```
proje/
├── continue_training.py      # Ana eğitim scripti
├── model_utils.py           # Model yardımcı fonksiyonları
├── requirements.txt         # Python bağımlılıkları
├── celeba.zip              # CelebA veri seti (kullanıcı tarafından sağlanmalı)
├── model_kayıt/       # Checkpoint'lerin saklandığı klasör
└── sonuc_kayıt/     # Üretilen görüntülerin saklandığı klasör
```

## Kullanım

### 1. Veri Setinin Hazırlanması

- [CelebA](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset) veri setini indirin ve `celeba.zip` olarak ana klasöre yerleştirin
- Script otomatik olarak zip dosyasını çıkaracak ve kullanıma hazırlayacaktır

### 2. Eğitime Devam Etme

#### İnteraktif Mod

```bash
python continue_training.py
```

Bu komut çalıştırıldığında:
- Mevcut checkpoint'ler listelenecek
- Hangi checkpoint'ten devam edeceğinizi seçebilirsiniz
- Varsayılan olarak 20 ek epoch eğitim yapılacak


### 3. Komut Satırı Parametreleri

- `--checkpoint`: Yüklenecek checkpoint dosyasının yolu
- `--epochs`: Ek eğitim epoch sayısı (varsayılan: 20)
- `--auto`: Otomatik mod (en son checkpoint'i seç)

## Model Mimarisi

### Generator (HighResGenerator_64)
- Giriş: 100 boyutlu rastgele gürültü vektörü
- Çıkış: 64x64x3 RGB görüntü
- Transpose konvolüsyon katmanları ile yukarı örnekleme
- LayerNorm ve ReLU aktivasyon fonksiyonları
- Son katmanda Tanh aktivasyonu

### Discriminator (HighResDiscriminator_64)
- Giriş: 64x64x3 RGB görüntü
- Çıkış: Gerçek/sahte olasılığı (0-1)
- Konvolüsyon katmanları ile aşağı örnekleme
- LayerNorm ve LeakyReLU aktivasyon fonksiyonları
- Son katmanda Sigmoid aktivasyonu

## Hiperparametreler

```python
latent_dim = 100           # Gürültü vektörü boyutu
img_size = 64             # Görüntü boyutu
batch_size = 16           # Batch boyutu
lr_g = 0.0005            # Generator öğrenme oranı
lr_d = 0.0002            # Discriminator öğrenme oranı
max_data_samples = 30000  # Maksimum veri örneği
```

## Çıktılar

### Kaydedilen Dosyalar

1. **Checkpoint'ler** (`model_kayıt/`):
   - `model_epoch_X_continued.pth`: Her 10 epoch'ta kaydedilen model durumu

2. **Görüntüler** (`sonuc_kayıt/`):
   - `epoch_X_continued.png`: Her epoch sonunda üretilen örnek görüntüler
   - `continued_training_losses.png`: Eğitim kayıp grafikleri
   - `final_generated_faces_continued.png`: Son üretilen yüz görüntüleri

### Checkpoint İçeriği

Her checkpoint şunları içerir:
- Generator ve Discriminator model ağırlıkları
- Optimizer durumları
- Tüm eğitim kayıpları
- Epoch bilgisi
- Zaman damgası
- Ek eğitim bilgileri

## Eğitim Süreci

1. **Veri Yükleme**: CelebA veri seti otomatik olarak çıkarılır ve yüklenir
2. **Model Başlatma**: Generator ve Discriminator modelleri oluşturulur
3. **Checkpoint Yükleme**: Seçilen checkpoint'ten model durumu yüklenir
4. **Eğitim Döngüsü**: 
   - Discriminator eğitimi (gerçek + sahte görüntüler)
   - Generator eğitimi (sahte görüntüleri gerçekmiş gibi gösterme)
5. **Kaydetme**: Her epoch sonunda görüntü üretimi ve periyodik checkpoint kaydetme

## Sorun Giderme

### Yaygın Hatalar

1. **CUDA bellek hatası**: Batch boyutunu küçültün
2. **Veri seti bulunamadı**: `celeba.zip` dosyasının doğru konumda olduğundan emin olun
3. **Checkpoint yüklenemedi**: Checkpoint dosyasının bozuk olmadığını kontrol edin

### Performans İpuçları

- GPU kullanımı için CUDA kurulu olmalı
- Yeterli RAM (8GB+) önerilir
- SSD kullanımı veri yükleme süresini hızlandırır

## Teknik Detaylar

### Kayıp Fonksiyonu
- Binary Cross Entropy (BCE) Loss kullanılır
- Label smoothing uygulanır (real=0.9, fake=0.1)

### Optimizasyon
- Adam optimizer kullanılır
- Generator: lr=0.0005, beta1=0.0, beta2=0.9
- Discriminator: lr=0.0002, beta1=0.5, beta2=0.999

### Düzenlileştirme
- LayerNorm kullanılır (BatchNorm yerine)
- Xavier normal ağırlık başlatma
