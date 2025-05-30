import torch
import torch.nn as nn
import os
from datetime import datetime

class HighResGenerator_64(nn.Module):
    def __init__(self, latent_dim=100):
        super(HighResGenerator_64, self).__init__()
        self.latent_dim = latent_dim
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.LayerNorm([512, 4, 4]),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.LayerNorm([256, 8, 8]),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.LayerNorm([128, 16, 16]),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.LayerNorm([64, 32, 32]),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        input = input.view(-1, self.latent_dim, 1, 1)
        return self.main(input)

class HighResDiscriminator_64(nn.Module):
    def __init__(self):
        super(HighResDiscriminator_64, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.LayerNorm([128, 16, 16]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.LayerNorm([256, 8, 8]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.LayerNorm([512, 4, 4]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif classname.find('LayerNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def save_checkpoint(generator, discriminator, optimizer_g, optimizer_d, epoch, losses_g, losses_d, filepath, additional_info=None):
    """
    Model checkpoint'ini kaydet
    
    Args:
        generator: Generator modeli
        discriminator: Discriminator modeli
        optimizer_g: Generator optimizer'ı
        optimizer_d: Discriminator optimizer'ı
        epoch: Mevcut epoch
        losses_g: Generator kayıpları listesi
        losses_d: Discriminator kayıpları listesi
        filepath: Kayıt dosya yolu
        additional_info: Ek bilgiler (dict)
    """
    checkpoint = {
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(), 
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_g_state_dict': optimizer_g.state_dict(), 
        'optimizer_d_state_dict': optimizer_d.state_dict(),
        'losses_g': losses_g,
        'losses_d': losses_d,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    if additional_info:
        checkpoint.update(additional_info)  
    
    torch.save(checkpoint, filepath) 
    print(f"Checkpoint kaydedildi: {filepath}")

def load_checkpoint(filepath, generator, discriminator, optimizer_g, optimizer_d, device):
    """
    Model checkpoint'ini yükle
    
    Args:
        filepath: Checkpoint dosya yolu
        generator: Generator modeli
        discriminator: Discriminator modeli
        optimizer_g: Generator optimizer'ı
        optimizer_d: Discriminator optimizer'ı
        device: Cihaz (cuda/cpu)
    
    Returns:
        tuple: (epoch, losses_g, losses_d, additional_info)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint dosyası bulunamadı: {filepath}") 
    
    print(f"Checkpoint yükleniyor: {filepath}")
    checkpoint = torch.load(filepath, map_location=device)  
    
    generator.load_state_dict(checkpoint['generator_state_dict'])  
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])

    optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
    optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])

    losses_g = checkpoint.get('losses_g', [])
    losses_d = checkpoint.get('losses_d', [])

    epoch = checkpoint['epoch']
    
    additional_info = {key: value for key, value in checkpoint.items() 
                      if key not in ['epoch', 'generator_state_dict', 'discriminator_state_dict', 
                                   'optimizer_g_state_dict', 'optimizer_d_state_dict', 'losses_g', 'losses_d']}
    
    print(f"Checkpoint başarıyla yüklendi. Epoch: {epoch + 1}")
    if 'timestamp' in additional_info:
        print(f"Kayıt zamanı: {additional_info['timestamp']}")
    
    return epoch, losses_g, losses_d, additional_info

def list_checkpoints(models_dir="models_high_res_updated_64"):
    """
    Mevcut checkpoint dosyalarını listele
    
    Args:
        models_dir: Modellerin bulunduğu klasör
    
    Returns:
        list: Checkpoint dosyalarının listesi
    """
    if not os.path.exists(models_dir):
        print(f"Model klasörü bulunamadı: {models_dir}")
        return []
    
    checkpoint_files = []
    for filename in os.listdir(models_dir):
        if filename.endswith('.pth'):
            filepath = os.path.join(models_dir, filename)
            try:
                checkpoint = torch.load(filepath, map_location='cpu')
                epoch = checkpoint.get('epoch', -1)
                timestamp = checkpoint.get('timestamp', 'Bilinmiyor')
                
                checkpoint_files.append({
                    'filename': filename,
                    'filepath': filepath,
                    'epoch': epoch,
                    'timestamp': timestamp
                })
            except:
                print(f"Bozuk checkpoint dosyası: {filename}")
    
    checkpoint_files.sort(key=lambda x: x['epoch'])
    return checkpoint_files

def get_latest_checkpoint(models_dir="models_high_res_updated_64"):
    """
    En son kaydedilen checkpoint'i bul
    
    Args:
        models_dir: Modellerin bulunduğu klasör
    
    Returns:
        str: En son checkpoint'in dosya yolu
    """
    checkpoints = list_checkpoints(models_dir)
    if not checkpoints:
        return None
    
    return checkpoints[-1]['filepath']

def show_checkpoint_info(filepath):
    """
    Checkpoint hakkında detaylı bilgi göster
    
    Args:
        filepath: Checkpoint dosya yolu
    """
    try:
        checkpoint = torch.load(filepath, map_location='cpu')
        print(f"\n=== Checkpoint Bilgileri ===")
        print(f"Dosya: {os.path.basename(filepath)}")
        print(f"Epoch: {checkpoint.get('epoch', 'Bilinmiyor') + 1}")
        print(f"Kayıt Zamanı: {checkpoint.get('timestamp', 'Bilinmiyor')}")
        
        if 'losses_g' in checkpoint and 'losses_d' in checkpoint:
            print(f"Generator Kayıp Sayısı: {len(checkpoint['losses_g'])}")
            print(f"Discriminator Kayıp Sayısı: {len(checkpoint['losses_d'])}")
            if checkpoint['losses_g']:
                print(f"Son Generator Kaybı: {checkpoint['losses_g'][-1]:.4f}")
            if checkpoint['losses_d']:
                print(f"Son Discriminator Kaybı: {checkpoint['losses_d'][-1]:.4f}")
        
        print("="*30)
    except Exception as e:
        print(f"Checkpoint bilgileri okunamadı: {e}")

def select_checkpoint_interactive(models_dir="models_high_res_updated_64"):
    """
    Kullanıcıdan checkpoint seçmesini iste
    
    Args:
        models_dir: Modellerin bulunduğu klasör
    
    Returns:
        str: Seçilen checkpoint'in dosya yolu
    """
    checkpoints = list_checkpoints(models_dir)
    
    if not checkpoints:
        print("Hiç checkpoint bulunamadı!")
        return None
    
    print("\n=== Mevcut Checkpoint'ler ===")
    for i, cp in enumerate(checkpoints):
        print(f"{i+1}. {cp['filename']} (Epoch: {cp['epoch']+1}, Zaman: {cp['timestamp']})")
    
    print(f"{len(checkpoints)+1}. En son checkpoint'i otomatik seç")
    print("0. İptal")
    
    while True:
        try:
            choice = int(input(f"\nSeçiminizi yapın (0-{len(checkpoints)+1}): "))
            
            if choice == 0:
                return None
            elif choice == len(checkpoints) + 1:
                return get_latest_checkpoint(models_dir)
            elif 1 <= choice <= len(checkpoints):
                selected = checkpoints[choice-1]
                show_checkpoint_info(selected['filepath'])
                return selected['filepath']
            else:
                print("Geçersiz seçim!")
        except ValueError:
            print("Lütfen bir sayı girin!")
        except KeyboardInterrupt:
            print("\nİptal edildi.")
            return None