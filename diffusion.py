import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageNet
from transformers import CLIPModel, CLIPProcessor, BlipForConditionalGeneration, BlipProcessor
import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from einops import rearrange


class Config:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_size = 128
    latent_dim = 128
    batch_size = 32
    num_epochs = 50
    lr = 1e-4
    timesteps = 1000
    checkpoint_path = "ldm_checkpoint.pth"


class StableVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(128, 256, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(256, 512, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(512, Config.latent_dim*2, 4, 2, 1),
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(Config.latent_dim, 512, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1), nn.Tanh(),
        )

    def encode(self, x):
        mu_logvar = self.encoder(x)
        return mu_logvar.chunk(2, dim=1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        half_dim = dim // 2
        emb = torch.log(torch.tensor(10000)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=Config.device) * -emb)
        self.register_buffer('emb', emb)

    def forward(self, t):
        t = t[:, None] * self.emb[None, :]
        return torch.cat((t.sin(), t.cos()), dim=-1)


class Attention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.GroupNorm(32, dim)
        self.qkv = nn.Conv2d(dim, dim*3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=1)
        attn = (q.reshape(B, C, -1).transpose(1, 2) @ k.reshape(B, C, -1)) * (C ** -0.5)
        attn = attn.softmax(dim=-1)
        x = (attn @ v.reshape(B, C, -1)).transpose(1, 2).reshape(B, C, H, W)
        return self.proj(x) + x


class ResBlock(nn.Module):
    def __init__(self, dim, time_dim, text_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_dim, dim)
        self.text_mlp = nn.Linear(text_dim, dim)
        self.block = nn.Sequential(
            nn.GroupNorm(32, dim),
            nn.SiLU(),
            nn.Conv2d(dim, dim, 3, padding=1),
        )

    def forward(self, x, t, text_emb):
        h = self.block(x + self.time_mlp(t)[:,:,None,None] + self.text_mlp(text_emb)[:,:,None,None])
        return h + x


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        ch = 128
        self.time_emb = TimeEmbedding(ch*4)
        self.text_emb = nn.Linear(512, ch*4)  # CLIP text features
        
        self.down = nn.ModuleList([
            nn.Conv2d(Config.latent_dim, ch, 3, padding=1),
            ResBlock(ch, ch*4, ch*4),
            Attention(ch),
            nn.Conv2d(ch, ch*2, 3, stride=2, padding=1),
            ResBlock(ch*2, ch*4, ch*4),
            Attention(ch*2),
            nn.Conv2d(ch*2, ch*4, 3, stride=2, padding=1),
            ResBlock(ch*4, ch*4, ch*4),
            Attention(ch*4),
        ])
        
        self.mid = nn.Sequential(
            ResBlock(ch*4, ch*4, ch*4),
            Attention(ch*4),
            ResBlock(ch*4, ch*4, ch*4),
        )
        
        self.up = nn.ModuleList([
            nn.ConvTranspose2d(ch*4, ch*2, 3, 2, 1, output_padding=1),
            ResBlock(ch*2, ch*4, ch*4),
            Attention(ch*2),
            nn.ConvTranspose2d(ch*2, ch, 3, 2, 1, output_padding=1),
            ResBlock(ch, ch*4, ch*4),
            Attention(ch),
            nn.Conv2d(ch, Config.latent_dim, 3, padding=1),
        ])

    def forward(self, x, t, text_emb):
        t = self.time_emb(t)
        text_emb = self.text_emb(text_emb)
        
        # Downsample
        skips = []
        for layer in self.down:
            if isinstance(layer, ResBlock):
                x = layer(x, t, text_emb)
            elif isinstance(layer, Attention):
                x = layer(x)
            else:
                x = layer(x)
            skips.append(x)
        
        # Middle
        x = self.mid(x)
        
        # Upsample
        for layer in self.up:
            if isinstance(layer, nn.ConvTranspose2d):
                x = layer(x)
                x = torch.cat([x, skips.pop()], dim=1)
            elif isinstance(layer, ResBlock):
                x = layer(x, t, text_emb)
            elif isinstance(layer, Attention):
                x = layer(x)
            else:
                x = layer(x)
        
        return x


class LabelGenerator:
    def __init__(self):
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-large").to(Config.device)
        
    def generate_label(self, image_tensor):
        image = transforms.ToPILImage()(image_tensor.cpu().add(1).div(2).clamp(0,1))
        inputs = self.processor(images=image, return_tensors="pt").to(Config.device)
        outputs = self.model.generate(**inputs, max_length=20)
        return self.processor.decode(outputs[0], skip_special_tokens=True)


class Diffusion:
    def __init__(self):
        self.timesteps = Config.timesteps
        self.betas = self._cosine_schedule()
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        
    def _cosine_schedule(self):
        s = 0.008
        steps = self.timesteps + 1
        x = torch.linspace(0, self.timesteps, steps)
        alphas = torch.cos(((x / self.timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas = alphas / alphas[0]
        return torch.clip(1 - alphas[1:] / alphas[:-1], 0, 0.999)
    
    def sample_timesteps(self, n):
        return torch.randint(1, self.timesteps, (n,)).to(Config.device)
    
    def noise_latents(self, latents, t):
        sqrt_alpha_bar = torch.sqrt(self.alpha_bars[t])[:,None,None,None]
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bars[t])[:,None,None,None]
        noise = torch.randn_like(latents)
        return sqrt_alpha_bar * latents + sqrt_one_minus_alpha_bar * noise, noise


class LatentDiffusion:
    def __init__(self):
        self.vae = StableVAE().to(Config.device)
        self.unet = UNet().to(Config.device)
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(Config.device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.label_gen = LabelGenerator()
        self.diffusion = Diffusion()
        
        self.optimizer = torch.optim.AdamW([
            {'params': self.vae.parameters()},
            {'params': self.unet.parameters()}
        ], lr=Config.lr)
        
    def train_step(self, images):
        # Generate labels and encode text
        texts = [self.label_gen.generate_label(img) for img in images]
        text_inputs = self.clip_processor(text=texts, return_tensors="pt", padding=True).to(Config.device)
        text_emb = self.clip.get_text_features(**text_inputs)
        
        # Encode images to latent space
        with torch.no_grad():
            mu, logvar = self.vae.encode(images)
            latents = self.vae.reparameterize(mu, logvar)
        
        # Diffusion process
        t = self.diffusion.sample_timesteps(images.size(0))
        noisy_latents, noise = self.diffusion.noise_latents(latents, t)
        predicted_noise = self.unet(noisy_latents, t, text_emb)
        
        # Loss calculation
        vae_loss = F.mse_loss(self.vae.decode(latents), images)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / images.size(0)
        diffusion_loss = F.mse_loss(predicted_noise, noise)
        total_loss = vae_loss + 0.001*kl_loss + diffusion_loss
        
        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.unet.parameters(), 1.0)
        self.optimizer.step()
        
        return total_loss.item()

    def save_checkpoint(self):
        torch.save({
            'vae': self.vae.state_dict(),
            'unet': self.unet.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, Config.checkpoint_path)

    def load_checkpoint(self):
        if os.path.exists(Config.checkpoint_path):
            state = torch.load(Config.checkpoint_path)
            self.vae.load_state_dict(state['vae'])
            self.unet.load_state_dict(state['unet'])
            self.optimizer.load_state_dict(state['optimizer'])

    def generate(self, prompt, steps=100):
        self.vae.eval()
        self.unet.eval()
        
        # Encode text
        text_inputs = self.clip_processor(text=[prompt], return_tensors="pt", padding=True).to(Config.device)
        text_emb = self.clip.get_text_features(**text_inputs)
        
        # Initialize latent
        z = torch.randn(1, Config.latent_dim, 16, 16).to(Config.device)
        
        # Diffusion sampling
        for t in tqdm(reversed(range(0, Config.timesteps)), desc="Generating"):
            ts = torch.full((1,), t, device=Config.device).long()
            pred_noise = self.unet(z, ts, text_emb)
            alpha = self.diffusion.alphas[t]
            alpha_bar = self.diffusion.alpha_bars[t]
            beta = self.diffusion.betas[t]
            
            if t > 0:
                noise = torch.randn_like(z)
            else:
                noise = 0
                
            z = (z - (beta / torch.sqrt(1 - alpha_bar)) * pred_noise) / torch.sqrt(alpha)
            z += torch.sqrt(beta) * noise
        
        # Decode latent
        with torch.no_grad():
            image = self.vae.decode(z).clamp(-1, 1)
        return image[0].permute(1, 2, 0).cpu().detach().numpy()


if __name__ == "__main__":
    # Initialize system
    ldm = LatentDiffusion()
    ldm.load_checkpoint()
    
    # Dataset preparation
    transform = transforms.Compose([
        transforms.Resize(Config.image_size),
        transforms.CenterCrop(Config.image_size),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])
    dataset = ImageNet(root='./data', split='train', transform=transform)
    loader = DataLoader(dataset, batch_size=Config.batch_size, shuffle=True, num_workers=4)
    
    # Training loop
    for epoch in range(Config.num_epochs):
        total_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{Config.num_epochs}")
        for images, _ in pbar:
            loss = ldm.train_step(images.to(Config.device))
            total_loss += loss
            pbar.set_postfix(loss=loss)
        
        print(f"Epoch {epoch+1} - Average Loss: {total_loss/len(loader):.4f}")
        ldm.save_checkpoint()
    
    # Interactive generation
    while True:
        prompt = input("Enter text prompt (or 'quit' to exit): ")
        if prompt.lower() == 'quit':
            break
            
        generated = ldm.generate(prompt)
        plt.imshow((generated + 1)/2)
        plt.axis('off')
        plt.show()
