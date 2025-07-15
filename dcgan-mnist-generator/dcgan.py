import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os

# ハイパーパラメータ
latent_dim = 100
image_size = 28
channels = 1
batch_size = 128
epochs = 30
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 保存ディレクトリ
os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# データセット準備
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Generator
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, image_size * image_size),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.main(z)
        return img.view(z.size(0), 1, image_size, image_size)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(image_size * image_size, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.model(img)

# 初期化
G = Generator().to(device)
D = Discriminator().to(device)
loss_fn = nn.BCELoss()
optimizer_G = torch.optim.Adam(G.parameters(), lr=0.0002)
optimizer_D = torch.optim.Adam(D.parameters(), lr=0.0002)

# 学習ループ
for epoch in range(epochs):
    for i, (imgs, _) in enumerate(loader):
        real = imgs.to(device)
        batch_size = real.size(0)

        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # ----- Discriminator学習 -----
        z = torch.randn(batch_size, latent_dim).to(device)
        fake = G(z)

        real_loss = loss_fn(D(real), real_labels)
        fake_loss = loss_fn(D(fake.detach()), fake_labels)
        d_loss = real_loss + fake_loss

        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()

        # ----- Generator学習 -----
        z = torch.randn(batch_size, latent_dim).to(device)
        fake = G(z)
        g_loss = loss_fn(D(fake), real_labels)

        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()

    print(f"[{epoch+1}/{epochs}] D loss: {d_loss.item():.4f}, G loss: {g_loss.item():.4f}")
    save_image(fake[:25], f"outputs/generated_{epoch+1:03d}.png", nrow=5, normalize=True)

# モデル保存
torch.save(G.state_dict(), "models/generator.pth")
torch.save(D.state_dict(), "models/discriminator.pth")
