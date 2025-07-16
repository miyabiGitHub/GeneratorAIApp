import streamlit as st
import os
import gdown
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# --- ハイパーパラメータ ---
latent_dim = 100
image_size = 28
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ensure_generator_exists():
    model_path = "models/generator.pth"
    if not os.path.exists(model_path):
        st.warning("モデルが見つかりません。Google Drive からダウンロード中…")

        # ここにDriveのファイルIDを挿入
        file_id = "14ZDV5B0J_K4y_B1YDucH6Nle2HXty_V-"
        url = f"https://drive.google.com/uc?id={file_id}"
        os.makedirs("models", exist_ok=True)
        gdown.download(url, model_path, quiet=False)

        st.success("✅ モデルのダウンロードが完了しました！")

ensure_generator_exists()

# --- Generator定義（dcgan.pyと同じ） ---
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

# --- モデル読み込み ---
@st.cache_resource
def load_generator():
    model = Generator().to(device)
    model.load_state_dict(torch.load("models/generator.pth", map_location=device))
    model.eval()
    return model

G = load_generator()

# --- Streamlit UI ---
st.title("🧠 数字生成AI（DCGAN）")
st.markdown("ノイズを入力して、手書き数字っぽい画像を生成しよう。")

col1, col2 = st.columns([2, 1])
with col2:
    seed = st.slider("乱数シード", 0, 1000, 42)
    num_images = st.slider("生成枚数", 1, 10, 5)
    generate_btn = st.button("画像を生成")

if generate_btn:
    torch.manual_seed(seed)
    z = torch.randn(num_images, latent_dim).to(device)
    with torch.no_grad():
        fake_imgs = G(z).cpu().numpy()

    # 画像表示
    fig, axs = plt.subplots(1, num_images, figsize=(num_images * 2, 2))
    if num_images == 1:
        axs = [axs]

    for i in range(num_images):
        axs[i].imshow(fake_imgs[i][0], cmap='gray')
        axs[i].axis('off')
    st.pyplot(fig)
