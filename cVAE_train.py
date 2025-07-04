import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
import numpy as np
import os
import matplotlib.pyplot as plt
import torchvision
from sklearn.model_selection import train_test_split
from conditioned_VAE_model import ConditionedVAE

# ========================
# 1. 超参数
# ========================
batch_size = 128
vae_epochs = 50
lr_vae = 1e-3
kld_weight = 1e-3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ========================
# 2. 数据加载
# ========================
X = np.load("X_scaled_zc.npy")      # 条件: (N,4)
y = np.load("target_scaled_zc.npy") # 图: (N,35*31)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

def to_tensor(X_arr, y_arr):
    X_t = torch.FloatTensor(X_arr)
    y_t = torch.FloatTensor(y_arr).view(-1,1,35,31)
    return X_t, y_t

X_train_t, y_train_t = to_tensor(X_train, y_train)
X_val_t,   y_val_t   = to_tensor(X_val, y_val)
print(y_train_t.shape)

train_loader_vae = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)

# ========================
# 3. 训练VAE
# ========================
vae = ConditionedVAE().to(device)
opt_vae = torch.optim.Adam(vae.parameters(), lr=lr_vae)
vae_loss_history = []

for epoch in range(vae_epochs):
    vae.train()
    total_loss = 0
    for cond_input, img in tqdm(train_loader_vae, desc=f"VAE Epoch {epoch}"):
        cond_input = cond_input.to(device)
        img = img.to(device)
        recon, mean, logvar = vae(img, cond_input)
        # 重建 + KL
        recon_loss = F.mse_loss(recon, img)
        kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
        loss = recon_loss + kld_weight * kl_loss
        opt_vae.zero_grad()
        loss.backward() 
        opt_vae.step()
        total_loss += loss.item()
    vae_loss = total_loss / len(train_loader_vae)
    vae_loss_history.append(vae_loss)
    print(f"VAE Epoch {epoch} Loss: {total_loss/len(train_loader_vae):.6f}")

# 保存VAE
torch.save(vae.state_dict(), "vae.pth")

# ========================
# 4. 绘制训练过程
# ========================
plt.figure(figsize=(5, 4))
plt.rcParams['axes.unicode_minus'] = False
plt.plot(vae_loss_history, label='vae_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('VAE Loss Curve')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig('vae_training_metrics.png')



