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
import seaborn as sns
from conditioned_VAE_model import ConditionedVAE

device = torch.device("cpu")

# ========================
# 1. 数据加载
# ========================
X = np.load("X_scaled_zc.npy")      
y = np.load("target_scaled_zc.npy") 

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

def to_tensor(X_arr, y_arr):
    X_t = torch.FloatTensor(X_arr)
    y_t = torch.FloatTensor(y_arr).view(-1,1,35,31)
    return X_t, y_t

X_train_t, y_train_t = to_tensor(X_train, y_train)
X_val_t,   y_val_t   = to_tensor(X_val, y_val)

# ========================
# 2. 采样VAE
# ========================
vae = ConditionedVAE().to(device)
vae.load_state_dict(torch.load('vae.pth'))

# ------------ pixel to pixel scatter --------------- #
vae.eval()
with torch.no_grad():
    recon, _, _ = vae(y_val_t.to(device), X_val_t.to(device))
    recon = recon.cpu()
    original = y_val_t.cpu()

x = original.view(-1).numpy() 
y = recon.view(-1).numpy()  
print(x.shape, y.shape)

# 可视化：散点图
plt.figure(figsize=(6, 6))
plt.scatter(x, y, s=1, alpha=0.5) 
plt.xlabel("Original Pixel Value")
plt.ylabel("Reconstructed Pixel Value")
plt.title("Pixel-wise Scatter: Original vs Reconstructed")
plt.grid(True)
plt.tight_layout()
plt.savefig('vae_pixel_wise_scatter.png')
# -------------------------------------- #

# ----- plot comparison heatmap ------- #
cond_input = X_val_t.to(device)
img = y_val_t.to(device)
num = 100
vae.eval()


with torch.no_grad():
    recon, _, _ = vae(img, cond_input)
    
    # Get the min and max values for consistent scaling
    vmin = min(img[num,0].min(), recon[num,0].min())
    vmax = max(img[num,0].max(), recon[num,0].max())
    
    # Create a figure with 2 subplots and a shared colorbar
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot original
    sns.heatmap(img[num,0].cpu(), ax=ax1, cmap='viridis', 
                vmin=vmin, vmax=vmax, cbar=False)
    ax1.set_title("Original")
    
    # Plot reconstruction
    im = sns.heatmap(recon[num,0].cpu(), ax=ax2, cmap='viridis',
                    vmin=vmin, vmax=vmax, cbar=False)
    ax2.set_title("Reconstructed")
    
    # Add a single colorbar for both plots
    fig.colorbar(im.get_children()[0], ax=[ax1, ax2], 
                 orientation='vertical', fraction=0.05, pad=0.02)
    
    plt.savefig('vae_heat_map.png')
    plt.show()
# ---------------------- #