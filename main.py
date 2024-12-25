import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

class PaintingDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = glob.glob(os.path.join(root_dir, "*.jpg"))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image


class Generator(nn.Module):
    def __init__(self, latent_dim=100, feature_maps=64):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, feature_maps * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_maps * 2, feature_maps, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_maps, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)


class Discriminator(nn.Module):
    def __init__(self, feature_maps=64):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, feature_maps, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_maps * 4, feature_maps * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_maps * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).view(-1)

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Настройки
    image_size = 64
    batch_size = 64
    latent_dim = 100
    lr = 0.0002
    beta1 = 0.5
    num_epochs = 200


    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])

    dataset_root = "./russian_classic_paintings"
    dataset = PaintingDataset(root_dir=dataset_root, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)


    generator = Generator(latent_dim=latent_dim).to(device)
    discriminator = Discriminator().to(device)


    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    generator.apply(weights_init)
    discriminator.apply(weights_init)

    optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
    criterion = nn.BCELoss()

    fixed_noise = torch.randn(16, latent_dim, 1, 1, device=device)
    g_losses, d_losses = [], []

    for epoch in range(num_epochs):
        for i, real_images in enumerate(dataloader):
            batch_size_cur = real_images.size(0)
            real_images = real_images.to(device)
            real_labels = torch.ones(batch_size_cur, device=device)
            fake_labels = torch.zeros(batch_size_cur, device=device)

            discriminator.zero_grad()
            outputs_real = discriminator(real_images)
            d_loss_real = criterion(outputs_real, real_labels)

            noise = torch.randn(batch_size_cur, latent_dim, 1, 1, device=device)
            fake_images = generator(noise)
            outputs_fake = discriminator(fake_images.detach())
            d_loss_fake = criterion(outputs_fake, fake_labels)

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_d.step()

            generator.zero_grad()
            noise = torch.randn(batch_size_cur, latent_dim, 1, 1, device=device)
            fake_images = generator(noise)
            outputs = discriminator(fake_images)
            g_loss = criterion(outputs, real_labels)

            g_loss.backward()
            optimizer_g.step()

            if i % 100 == 0:
                print(f"[Epoch {epoch+1}/{num_epochs}] [Batch {i}/{len(dataloader)}] "
                      f"D loss: {d_loss.item():.4f}, G loss: {g_loss.item():.4f}")

        g_losses.append(g_loss.item())
        d_losses.append(d_loss.item())

        with torch.no_grad():
            fake_preview = generator(fixed_noise).detach().cpu()
        # Пример кода для визуализации:
        # fig, axes = plt.subplots(4, 4, figsize=(6, 6))
        # for idx, ax in enumerate(axes.flat):
        #     img = fake_preview[idx]
        #     img = (img * 0.5 + 0.5).clamp(0, 1)
        #     np_img = img.permute(1, 2, 0).numpy()
        #     ax.imshow(np_img)
        #     ax.axis("off")
        # plt.tight_layout()
        # plt.show()


    print("Обучение завершено. Генерируем новые изображения...")
    generator.eval()
    with torch.no_grad():
        sample_noise = torch.randn(16, latent_dim, 1, 1, device=device)
        generated_images = generator(sample_noise).cpu()


    fig, axes = plt.subplots(4, 4, figsize=(6, 6))
    for i, ax in enumerate(axes.flat):
        img = generated_images[i]
        img = (img * 0.5 + 0.5).clamp(0, 1)
        np_img = img.permute(1, 2, 0).numpy()
        ax.imshow(np_img)
        ax.axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
