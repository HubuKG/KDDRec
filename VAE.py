import torch
import torch.nn as nn
import torch.optim as optim
from Params import args

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):# 1000 128
        super(VAE, self).__init__()
        # self.LeakyReLURate=0.3

        # 编码器：提取噪声分布
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),# 128
            nn.ReLU(),
            # nn.LeakyReLU(self.LeakyReLURate),
            nn.Linear(128, args.vae_hidden_dims),# 128 64
            nn.ReLU()
            # nn.LeakyReLU(self.LeakyReLURate)
        )
        self.mu_layer = nn.Linear(args.vae_hidden_dims, latent_dim)# 64
        self.logvar_layer = nn.Linear(args.vae_hidden_dims, latent_dim)# 64

        # 解码器：生成噪声
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, args.vae_hidden_dims),# 64
            nn.ReLU(),
            # nn.LeakyReLU(self.LeakyReLURate),
            nn.Linear(args.vae_hidden_dims, 128),# 64 128
            nn.ReLU(),
            # nn.LeakyReLU(self.LeakyReLURate),
            nn.Linear(128, input_dim),# 128
            nn.Tanh()  # 输出范围 [-1,1]，适用于嵌入数据
        )

    def encode(self, x):#x:1024x49545
        h = self.encoder(x)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# 训练 VAE
# input_dim = 100  # 例如知识图谱实体/关系嵌入的维度
# latent_dim = 10  # 隐变量维度
# vae = VAE(input_dim, latent_dim)
# optimizer = optim.Adam(vae.parameters(), lr=0.001)
# criterion = nn.MSELoss()  # 适用于嵌入数据的重构


# def loss_function(recon_x, x, mu, logvar):
#     MSE = criterion(recon_x, x)  # 计算重构误差
#     KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())  # KL 散度
#     return MSE + 0.01 * KLD  # 0.01 控制 KL 散度权重
#
#
# # 训练循环
# for epoch in range(100):
#     noisy_embeddings = torch.randn((32, input_dim))  # 模拟 32 个带噪声的知识图谱嵌入
#     optimizer.zero_grad()
#     recon_x, mu, logvar = vae(noisy_embeddings)
#     loss = loss_function(recon_x, noisy_embeddings, mu, logvar)
#     loss.backward()
#     optimizer.step()
#     if epoch % 10 == 0:
#         print(f'Epoch {epoch}, Loss: {loss.item():.4f}')
