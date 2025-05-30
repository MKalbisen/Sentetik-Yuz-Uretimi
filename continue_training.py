import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import zipfile
from tqdm import tqdm
import argparse
from PIL import Image

# Model yardımcı fonksiyonlarını import et
from model_utils import (
    HighResGenerator_64, 
    HighResDiscriminator_64, 
    weights_init,
    save_checkpoint,
    load_checkpoint,
    list_checkpoints,
    get_latest_checkpoint,
    select_checkpoint_interactive,
    show_checkpoint_info
)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def extract_dataset():
    current_dir = os.getcwd()
    zip_path = os.path.join(current_dir, "celeba.zip")
    extract_path = os.path.join(current_dir, "celeba_extracted")
    
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"celeba.zip dosyası bulunamadı: {zip_path}")
    
    if not os.path.exists(extract_path):
        print("Veri seti çıkarılıyor...")
        os.makedirs(extract_path, exist_ok=True)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print("Veri seti başarıyla çıkarıldı.")
    else:
        print("Veri seti zaten çıkarılmış.")
    
    for root, dirs, files in os.walk(extract_path):
        if len(files) > 100:
            return root
        for dir_name in dirs:
            potential_data_dir = os.path.join(root, dir_name)
            if os.path.isdir(potential_data_dir):
                img_files = [f for f in os.listdir(potential_data_dir) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if len(img_files) > 100:
                    return potential_data_dir
    
    return extract_path

def get_dataloader(data_dir, max_data_samples=5000, batch_size=16, img_size=64):
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    try:
        dataset = ImageFolder(root=data_dir, transform=transform)
        print(f"ImageFolder ile başarıyla yüklendi.")
    except Exception as e:
        print(f"ImageFolder ile yükleme başarısız: {e}")
        print(f"Manuel yükleme deneniyor...")
        
        class CustomDataset(torch.utils.data.Dataset):
            def __init__(self, data_dir, transform=None):
                self.transform = transform
                self.image_paths = []
                
                extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
                for ext in extensions:
                    import glob
                    self.image_paths.extend(glob.glob(os.path.join(data_dir, '**', ext), recursive=True))
                
                print(f"Toplam {len(self.image_paths)} resim dosyası bulundu.")
            
            def __len__(self):
                return len(self.image_paths)
            
            def __getitem__(self, idx):
                img_path = self.image_paths[idx]
                try:
                    image = Image.open(img_path).convert('RGB')
                    if self.transform:
                        image = self.transform(image)
                    return image, 0
                except Exception as img_error:
                    print(f"Resim yüklenirken hata: {img_path} - {img_error}")
                    # Hatalı resim durumunda random tensor döndür
                    return torch.randn(3, img_size, img_size), 0
        
        dataset = CustomDataset(data_dir, transform=transform)
    
    valid_indices = []
    max_samples = min(max_data_samples, len(dataset))
    
    print(f"Toplam veri setinde {len(dataset)} görüntü var.")
    print(f"Kullanılacak maksimum görüntü sayısı: {max_samples}")
    
    # Geçerli görüntüleri kontrol et
    for idx in range(max_samples):
        try:
            img, _ = dataset[idx]
            
            # Torch Tensor kontrolü
            if isinstance(img, torch.Tensor):
                if img.dim() == 3 and list(img.size()) == [3, img_size, img_size]:
                    valid_indices.append(idx)
            # NumPy array kontrolü
            elif hasattr(img, 'shape') and hasattr(img, 'dtype'):
                import numpy as np
                if isinstance(img, np.ndarray) and img.shape == (3, img_size, img_size):
                    valid_indices.append(idx)
            # PIL Image kontrolü
            elif hasattr(img, 'size') and not isinstance(img, torch.Tensor):
                try:
                    width, height = img.size
                    if width == img_size and height == img_size:
                        valid_indices.append(idx)
                except (AttributeError, TypeError):
                    continue
        except Exception as e:
            print(f"İndeks {idx} kontrol edilirken hata: {e}")
            continue
    
    print(f"Geçerli görüntü sayısı: {len(valid_indices)}")
    
    if len(valid_indices) == 0:
        raise ValueError("Hiç geçerli görüntü bulunamadı!")
    
    subset_dataset = Subset(dataset, valid_indices)
    dataloader = DataLoader(subset_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    return dataloader

def continue_training(checkpoint_path=None, additional_epochs=20, interactive=True):
    """
    Eğitime devam et
    
    Args:
        checkpoint_path: Yüklenecek checkpoint dosyası (None ise interaktif seçim)
        additional_epochs: Kaç epoch daha eğitim yapılacak
        interactive: İnteraktif mod (kullanıcıdan girdi al)
    """
    
    # Seed ayarla
    set_seed(42)
    
    # Cihaz
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Kullanılan cihaz: {device}")
    
    # Hiperparametreler
    latent_dim = 100
    img_size = 64
    batch_size = 16
    lr_g = 0.0005
    lr_d = 0.0002
    beta1_g = 0.0  #mode collapse riskine karşı beta1'i düşük tutuyoruz. Adam optimizasyonu için momentum parametresi.Geçmiş gradyanların ne kadar etkili olacağını belirler
    beta2_g = 0.9  #beta2 genellikle 0.9 veya 0.999 olarak ayarlanır
    #Gradyanların karelerinin hareketli ortalamasını kontrol eder
    #Adaptive learning rate hesaplamasında kullanılır
    #Her parametrenin öğrenme hızını bireysel olarak ayarlar
    beta1_d = 0.5  #daha stabil öğrenme sağlar
    beta2_d = 0.999 #daha stabil öğrenme sağlar
    max_data_samples = 30000
    
    # Çıktı klasörü
    current_dir = os.getcwd()
    output_dir = os.path.join(current_dir, "sonuc_kayıt")
    models_dir = os.path.join(current_dir, 'model_kayıt')
    
    # Checkpoint seçimi
    if checkpoint_path is None:
        if interactive:
            checkpoint_path = select_checkpoint_interactive(models_dir)
            if checkpoint_path is None:
                print("Checkpoint seçilmedi. Program sonlandırılıyor.")
                return
        else:
            checkpoint_path = get_latest_checkpoint(models_dir)
            if checkpoint_path is None:
                print("Hiç checkpoint bulunamadı!")
                return
    
    # Modelleri oluştur
    netG = HighResGenerator_64(latent_dim).to(device)
    netD = HighResDiscriminator_64().to(device)
    
    # Optimizörler
    criterion = nn.BCELoss()
    optimizerG = optim.Adam(netG.parameters(), lr=lr_g, betas=(beta1_g, beta2_g))
    optimizerD = optim.Adam(netD.parameters(), lr=lr_d, betas=(beta1_d, beta2_d))
    
    # Checkpoint'i yükle
    try:
        start_epoch, G_losses, D_losses, additional_info = load_checkpoint(
            checkpoint_path, netG, netD, optimizerG, optimizerD, device
        )
        print(f"Eğitime {start_epoch + 1}. epoch'tan devam ediliyor.")
        print(f"Önceki eğitimde toplam {len(G_losses)} iterasyon yapılmış.")
    except Exception as e:
        print(f"Checkpoint yüklenirken hata oluştu: {e}")
        return
    
    # Veri setini yükle
    try:
        data_dir = extract_dataset()
        dataloader = get_dataloader(data_dir, max_data_samples, batch_size, img_size)
        print(f"Veri seti hazır. Batch sayısı: {len(dataloader)}")
    except Exception as e:
        print(f"Veri seti yüklenirken hata oluştu: {e}")
        return
    
    # Sabit gürültü
    fixed_noise = torch.randn(64, latent_dim, device=device)
    
    # Klasörleri oluştur
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    # Eğitime devam et
    print(f"\n{'='*50}")
    print(f"EĞİTİME DEVAM EDİLİYOR")
    print(f"Başlangıç epoch: {start_epoch + 1}")
    print(f"Hedef epoch: {start_epoch + additional_epochs + 1}")
    print(f"Ek eğitim: {additional_epochs} epoch")
    print(f"{'='*50}\n")
    
    real_label = 0.9
    fake_label = 0.1
    
    for epoch in range(start_epoch + 1, start_epoch + additional_epochs + 1):
        for i, data in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{start_epoch + additional_epochs + 1}")):
            # Discriminator eğitimi
            netD.zero_grad()  #discriminatörün gradyanlarını sıfırlar
            real_cpu = data[0].to(device)  #data[0] gerçek görüntüleri içerir. 
            batch_size_current = real_cpu.size(0)  #bu batch'teki gerçek görüntü sayısını alır
            label = torch.full((batch_size_current,), real_label, dtype=torch.float, device=device)  
            output = netD(real_cpu)  #discriminatörün gerçek görüntüler üzerindeki çıktısını alır(veri tahmini)
            errD_real = criterion(output, label)  #discriminatörün gerçek görüntüler üzerindeki kaybını hesaplar
            errD_real.backward()  #discriminatörün gradyanlarını hesaplar
            D_x = output.mean().item()  #discriminatörün gerçek görüntüler üzerindeki tahminlerinin ortalamasını alır(D(x) yani ne kadar gerçek olduğunu tahmin eder)

            noise = torch.randn(batch_size_current, latent_dim, device=device)
            fake = netG(noise)  #generator'ün gürültüden sahte görüntüler üretmesini sağlar
            label.fill_(fake_label)  #sahte görüntüler için etiketleri sahte olarak ayarlar genellikle 0.0 olarak kullanılır
            output = netD(fake.detach()) #generatörün grapinden ayırıyoruz. Discriminatör eğitilirken generatör değişmesin diye
            errD_fake = criterion(output, label)  #sahte görüntüler üzerindeki kaybı hesaplar
            errD_fake.backward()
            D_G_z1 = output.mean().item()  #discriminatörün sahte görüntüler üzerindeki tahminlerinin ortalamasını alır(D(G(z)) yani ne kadar sahte olduğunu tahmin eder)
            errD = errD_real + errD_fake  #toplam kaybı hesaplar
            optimizerD.step()  
            
            # Generator eğitimi
            netG.zero_grad()  #generator'ın gradyanlarını sıfırlar
            label.fill_(real_label)  #generator için etiketleri gerçek olarak ayarlar 1.0 olarak kullanılır
            output = netD(fake)  #detach kullanmadık çünkü generator de eğitilecek
            errG = criterion(output, label)  #generator'ün kaybını hesaplar
            errG.backward()  #generator'ın gradyanlarını hesaplar
            D_G_z2 = output.mean().item()  #oluşturulan sahte veriye verilen puan
            optimizerG.step() #parametreleri günceller
            
            # Kayıpları kaydet
            G_losses.append(errG.item())  #kayıpları zaman içinde izlemek için kaydeder
            D_losses.append(errD.item())
            
            """
            Gradyan, bir fonksiyonun belli bir noktadaki eğimini (yani yönünü ve ne kadar değiştiğini) gösterir.
            Yapay sinir ağlarında ağın tahmini ile gerçek değer arasındaki farkı (yani kayıp) azaltmaya çalışırız. Bunu başarmak için:

            Gradyanları hesaplarız – ağırlıklar nasıl değiştirilmeli ki kayıp azalsın?

            Ağırlıkları güncelleriz – gradyanın ters yönünde giderek.

            Bu yönteme geri yayılım (backpropagation) ve gradyan inişi (gradient descent) denir.
            """

            # İlerleme raporu
            if i % 10 == 0:
                print(f'[{epoch+1}/{start_epoch + additional_epochs + 1}][{i}/{len(dataloader)}] '
                      f'Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} '
                      f'D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f}/{D_G_z2:.4f}')
        
        # Epoch sonunda görüntü üret ve kaydet
        with torch.no_grad():
            fake = netG(fixed_noise).detach().cpu()
        
        plt.figure(figsize=(8, 8))
        plt.axis("off")
        plt.title(f"Epoch {epoch+1} (Devam)")
        plt.imshow(np.transpose(torchvision.utils.make_grid(fake, padding=2, normalize=True), (1, 2, 0)))
        plt.savefig(os.path.join(output_dir, f'epoch_{epoch+1}_continued.png'))
        plt.close()
        
        # Model kaydet
        if (epoch + 1) % 10 == 0 or epoch == start_epoch + additional_epochs:
            checkpoint_filename = f'model_epoch_{epoch+1}_continued.pth'
            checkpoint_filepath = os.path.join(models_dir, checkpoint_filename)
            
            additional_checkpoint_info = {
                'continued_from_epoch': start_epoch,
                'total_training_iterations': len(G_losses)
            }
            
            save_checkpoint(
                netG, netD, optimizerG, optimizerD, epoch, 
                G_losses, D_losses, checkpoint_filepath, additional_checkpoint_info
            )
    
    print(f"\n{'='*50}")
    print(f"EĞİTİM TAMAMLANDI!")
    print(f"Toplam epoch: {start_epoch + additional_epochs + 1}")
    print(f"Toplam iterasyon: {len(G_losses)}")
    print(f"{'='*50}")
    
    # Son kayıp grafiğini çiz
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.title("Tüm Eğitim Boyunca Kayıplar")
    plt.plot(G_losses, label="Generator", alpha=0.7)
    plt.plot(D_losses, label="Discriminator", alpha=0.7)
    plt.axvline(x=len(G_losses) - additional_epochs * len(dataloader), 
                color='red', linestyle='--', label='Devam Noktası')
    plt.xlabel("İterasyon")
    plt.ylabel("Kayıp")
    plt.legend()
    
    # Son kısmı yakınlaştır
    recent_start = max(0, len(G_losses) - additional_epochs * len(dataloader))
    plt.subplot(1, 2, 2)
    plt.title("Son Eğitim Kayıpları")
    plt.plot(G_losses[recent_start:], label="Generator")
    plt.plot(D_losses[recent_start:], label="Discriminator")
    plt.xlabel("İterasyon (Son Kısım)")
    plt.ylabel("Kayıp")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'continued_training_losses.png'))
    plt.show()
    
    # Son test görüntüleri üret
    netG.eval()
    noise = torch.randn(16, latent_dim, device=device)
    with torch.no_grad():
        fake_images = netG(noise).detach().cpu()
    fake_images = (fake_images + 1) / 2
    
    plt.figure(figsize=(10, 10))
    plt.axis("off")
    plt.title("Son Üretilen Yüzler (Devam Sonrası)")
    plt.imshow(np.transpose(torchvision.utils.make_grid(fake_images, padding=2), (1, 2, 0)))
    plt.savefig(os.path.join(output_dir, 'final_generated_faces_continued.png'))
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='GAN Eğitimine Devam Et')
    parser.add_argument('--checkpoint', type=str, help='Yüklenecek checkpoint dosyası')
    parser.add_argument('--epochs', type=int, default=20, help='Ek eğitim epoch sayısı')
    parser.add_argument('--auto', action='store_true', help='Otomatik mod (en son checkpoint)')
    
    args = parser.parse_args()
    
    if args.auto:
        # Otomatik mod: en son checkpoint'i kullan
        continue_training(
            checkpoint_path=None, 
            additional_epochs=args.epochs, 
            interactive=False
        )
    else:
        # İnteraktif mod
        continue_training(
            checkpoint_path=args.checkpoint, 
            additional_epochs=args.epochs, 
            interactive=True
        )

if __name__ == '__main__':
    main()



