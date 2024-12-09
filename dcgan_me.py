import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader,Dataset
import torchvision.utils as utils
import os
from PIL import Image
from tqdm import tqdm

class Generator_me(nn.Module):
    """ 
        输入[batch_size, z_dim, 1, 1]        用于产生图像的噪声
        输出[batch_size, 3, 64, 64]          batch_size张生成的图像
    """
    def __init__(self, z_dim=100, img_channels=3, feature_map_dim=64):
        super(Generator_me, self).__init__()

        self.gen = nn.Sequential(
            # Input: (z_dim, 1, 1)
            nn.ConvTranspose2d(z_dim, feature_map_dim * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_map_dim * 8),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(feature_map_dim * 8, feature_map_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_dim * 4),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(feature_map_dim * 4, feature_map_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_dim * 2),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(feature_map_dim * 2, feature_map_dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_dim),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(feature_map_dim, img_channels, 4, 2, 1, bias=False),
            nn.Tanh()  # To ensure output values are between -1 and 1
        )

    def forward(self, z):
        return self.gen(z)

class Discriminator_me(nn.Module):
    """
    输入[batch_size, 3, 64, 64]     batch_size张RGB图像 
    输出[batch_size, 1]             batch_size张图像为真的概率
    """
    def __init__(self, img_channels=3, feature_map_dim=64):
        super(Discriminator_me, self).__init__()

        self.disc = nn.Sequential(
            nn.Conv2d(img_channels, feature_map_dim, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(feature_map_dim, feature_map_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(feature_map_dim * 2, feature_map_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(feature_map_dim * 4, feature_map_dim * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_dim * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(feature_map_dim * 8, 1, 4, 1, 0, bias=False)
        )

    def forward(self, x):
        return self.disc(x).view(-1, 1)

class datasets_me(Dataset):
    def __init__(self, root, transforms=None):
        imgs = []
        labels = []  # Assuming you have labels
        for path in os.listdir(root):
            imgs.append(os.path.join(root, path))
            labels.append(0)  # Placeholder label, replace with actual labels if available
        self.imgs = imgs
        self.labels = labels  # Store labels
        self.transforms = transforms

    def __getitem__(self, index):
        img_path = self.imgs[index]
        img = Image.open(img_path)
        img = img.convert('RGB')
        img = self.transforms(img)
        label = self.labels[index]  # Get the corresponding label
        return img, label

    def __len__(self):
        return len(self.imgs)


def load_data(root, batch_size):
    ## 数据集加载进DataLoader
    transform = T.Compose([
        # T.Resize((64, 64)),  # 修改为64x64,事先预处理好resize操作，这样dataloader读取数据速度更快(大数据集推荐这样处理)
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize to [-1, 1] for DCGAN
    ])

    train_data = datasets_me(root, transforms=transform)
    dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True,num_workers=4, 
                                        pin_memory=True,prefetch_factor=2, persistent_workers=True)

    # #0 ------------------------- 随机显示一张图片 -------------------------
    # imgs, labels = next(iter(dataloader))
    # print(f"imgs.shape:",imgs.shape)        # imgs.shape: torch.Size([128, 3, 64, 64])

    # first_img = np.transpose(imgs[0],(1, 2, 0))  # 转为 (H, W, C)
    # first_img = (first_img * 0.5) + 0.5  # 反归一化到 [0, 1]

    # first_img = first_img.numpy()  # 转为 NumPy 格式
    # print(f"First image shape: {first_img.shape}")  # 确认形状是否为 (64, 64, 3)
    # plt.imshow(first_img)
    # plt.show()
    # #0 ------------------------- 随机显示一张图片 -------------------------

    return dataloader


def save_checkpoint(generator, discriminator, optimizer_G, optimizer_D, epoch, file_path):
    """保存"""
    torch.save({
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
        'optimizer_D_state_dict': optimizer_D.state_dict(),
    }, file_path)
    print(f"Checkpoint saved at epoch {epoch + 1}")

    torch.cuda.empty_cache()  # 释放不再需要的内存

# def load_checkpoint(file_path, generator, discriminator, optimizer_G, optimizer_D):
#     if os.path.exists(file_path):
#         checkpoint = torch.load(file_path)
#         generator.load_state_dict(checkpoint['generator_state_dict'])
#         discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
#         optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
#         optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
#         start_epoch = checkpoint['epoch'] + 1  # 下一次从断点开始
#         print(f"Checkpoint loaded. Resuming from epoch {start_epoch}")
#         return start_epoch
#     else:
#         print("No checkpoint found, starting training from scratch.")
#         return 0


def train():
    # 后期可以将超参数作为变量取出，这样增加可移植性
    latent_dim = 100  
    img_dim = 3 * 64 * 64
    batch_size = 128
    lr = 0.0002
    start_epoch = 0
    epochs = 100
    root = './imgs'
    #checkpoint_path = './model_path/checkpoint_30.pth'
    model_path = './model_dcgan_path'
    if not os.path.exists('./model_dcgan_path'):
        os.makedirs('./model_dcgan_path')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    generator_me = Generator_me(z_dim=latent_dim).to(device)
    discriminator_me = Discriminator_me().to(device)


    criterion = nn.BCEWithLogitsLoss()
    optim_G = optim.Adam(generator_me.parameters(), lr=lr,betas=(0.5, 0.999))
    optim_D = optim.Adam(discriminator_me.parameters(), lr=lr, betas=(0.5, 0.999))

    dataloader = load_data(root, batch_size)
    
     # 恢复 checkpoint（如果存在）
    #start_epoch = load_checkpoint(checkpoint_path, generator_me, discriminator_me, optim_G, optim_D)
    for epoch in range(start_epoch,epochs):
        # 使用 tqdm 包裹 dataloader 以显示进度条
        with tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{epochs}") as pbar:
            for i, (imgs, _) in pbar:
                real_imgs = imgs.to(device)      # 真实照片x，[batch_size, 3, 64, 64]
                batch_size = real_imgs.size(0)
                real_labels = torch.ones(batch_size, 1).to(device)    # 真实图片设置标签 1（表示真实）
                fake_labels = torch.zeros(batch_size, 1).to(device)   # 生成的假图片设置标签 0（表示假）

                # 生成假图片
                noise = torch.randn(batch_size, latent_dim, 1, 1).to(device)    
                fake_imgs = generator_me(noise)  # 生成G(z)


                # === 训练判别器 ===
                optim_D.zero_grad()
                real_loss = criterion(discriminator_me(real_imgs), real_labels)                  # 计算判别器在真实图片上的损失
                fake_loss = criterion(discriminator_me(fake_imgs.detach()), fake_labels)         # 计算判别器在假图片上的损失
                d_loss  = (real_loss + fake_loss) / 2     # 判别器的总损失
                d_loss.backward()
                optim_D.step()                    

                # === 训练生成器 ===
                optim_G.zero_grad()
                g_loss = criterion(discriminator_me(fake_imgs), real_labels)       
                g_loss.backward()
                optim_G.step()                  

                # 显示当前显存占用情况（以 MB 为单位）
                memory_allocated = torch.cuda.memory_allocated(device) / (1024 * 1024)


                # 进度条显示生成器和判别器的损失
                pbar.set_postfix(   D_loss=d_loss.item(),
                                    G_loss=g_loss.item()
                                    )                       
        if (epoch+1) % 10 == 0:
            save_checkpoint(generator_me, discriminator_me, optim_G, optim_D, epoch, f"./{model_path}/checkpoint_{epoch+1}.pth")
            print(f"Model saved at epoch {epoch + 1}")


def test():
    latent_dim = 100  # 潜在维度
    img_dim = 3 * 64 * 64  # 图片的维度 (3x64x64, RGB)
    model_path = './model_dcgan_path/checkpoint_100.pth'  # 已保存的模型路径
    output_dir = './generated_images'  # 保存生成图片的目录

    if not os.path.exists(output_dir): 
        os.makedirs(output_dir)         

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 创建生成器并加载训练好的模型
    generator = Generator_me(z_dim=latent_dim).to(device)
    checkpoint = torch.load(model_path)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    

    with torch.no_grad():
        noise = torch.randn(128, latent_dim, 1, 1).to(device)
        fake_imgs=generator(noise)
        fake_imgs = fake_imgs.view(-1,3,64,64)      # [batch_size, channel, height, width]
        fake_imgs = (fake_imgs * 0.5) + 0.5  # 反归一化到 [0, 1]


        # 显示1张
        img_0 = fake_imgs[0].cpu().detach().numpy()  # [channel, height, width]
        img_0 = np.transpose(img_0, (1, 2, 0))  # 转换为 (height, width, channel)


        # 显示8x8的拼接的图片
        if fake_imgs.size(0) >= 64:
            fake_imgs = fake_imgs[:64]

        grid = utils.make_grid(fake_imgs, padding=2, normalize=True, nrow=8)  # 制作一个网格图
        grid=np.transpose(grid.cpu(), (1, 2, 0))

        fig,ax=plt.subplots(1,2,figsize=(12, 6))

        ax[0].imshow(img_0)
        ax[0].set_title('Signle Image')
        ax[1].imshow(grid)
        ax[1].set_title('8x8 Images')
        plt.axis('off') 
        plt.savefig(os.path.join(output_dir, 'face_01.png'))                 ## 保存生成的图片
        plt.show()      


if __name__ == '__main__':
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    TRAIN = 0
    TEST = 1-TRAIN
    if TRAIN:
        train()
    elif TEST:
        test()
    else:
        print("Please set TRAIN or TEST to 1.")




