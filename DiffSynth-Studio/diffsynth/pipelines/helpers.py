# mainly from lucidrains's perceiver repo
import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch import einsum
import torch.nn.functional as F

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)

class PerceiverAttention(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents):
        """
        x:       [Batch, N_tokens, Dim]  (Video Input)
        latents: [Batch, M_queries, Dim] (Learnable Latents)
        """
        
        x = self.norm_media(x)
        latents = self.norm_latents(latents)

        b, m, h = latents.shape[0], latents.shape[1], self.heads

        q = self.to_q(latents)

        kv_input = torch.cat((x, latents), dim=1)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        q = rearrange(q, 'b n (h d) -> b h n d', h=h)
        k = rearrange(k, 'b n (h d) -> b h n d', h=h)
        v = rearrange(v, 'b n (h d) -> b h n d', h=h)

        # q: [b, h, m, d], k: [b, h, n+m, d]
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        dots = dots - dots.amax(dim=-1, keepdim=True).detach()
        
        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        return self.to_out(out)

class PerceiverResampler(nn.Module):
    def __init__(
        self,
        dim=1024,              
        depth=2,               
        dim_head=64,
        heads=8,
        num_latents=32,        
        max_num_media_tokens=4096 
    ):
        super().__init__()
        
        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        nn.init.normal_(self.latents, std=0.02)

        self.pos_emb = nn.Embedding(max_num_media_tokens, dim)

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                FeedForward(dim=dim)
            ]))

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        """
        Input: [Batch, Sequence_Length, Dim]
        Output: [Batch, Num_Latents, Dim]
        """
        b, n, device = *x.shape[:2], x.device
        
        latents = repeat(self.latents, 'n d -> b n d', b=b)

        pos_indices = torch.arange(n, device=device)

        x = x + self.pos_emb(pos_indices)

        for attn, ff in self.layers:
            latents = latents + attn(x, latents)
            latents = latents + ff(latents)
            
        return self.norm(latents)

class VisualThinkingAdapter(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim*3),
            nn.GELU(),
            nn.Linear(out_dim*3, out_dim)
        )
    def forward(self, x):
        return self.net(x)

class VisualThinkingDualAdapter(nn.Module):
    def __init__(self, in_dim, out_dim, t_min, t_max):
        super().__init__()
        
        self.head_dino = nn.Sequential(
            nn.Linear(in_dim, out_dim*3),
            nn.GELU(),
            nn.Linear(out_dim*3, out_dim)
        )
        
        self.head_vae = nn.Sequential(
            nn.Linear(in_dim, out_dim*3),
            nn.GELU(),
            nn.Linear(out_dim*3, out_dim)
        )

        self.t_min = t_min
        self.t_max = t_max

    def _get_alpha(self, timestep, device):
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], device=device, dtype=torch.float32)
        
        alpha = (timestep - self.t_min) / (self.t_max - self.t_min + 1e-6)
        
        alpha = alpha.clamp(0.0, 1.0)
        
        return alpha.view(-1, 1, 1)

    def forward(self, x, timestep):
        """
        x: [B, Seq_Len, Dim]
        timestep: [B] or Scalar
        """
        pred_dino = self.head_dino(x) 
        pred_vae  = self.head_vae(x) 
        
        alpha_view = self._get_alpha(timestep, x.device).type_as(pred_dino)
        
        mixed_out = alpha_view * pred_dino + (1 - alpha_view) * pred_vae
        
        return mixed_out, pred_dino, pred_vae

    def get_loss(self, pred_dino, pred_vae, gt_dino, gt_vae, timestep, epsilon=0.1):
        alpha = self._get_alpha(timestep, pred_dino.device).type_as(pred_dino)
        
        loss_dino = F.mse_loss(pred_dino, gt_dino, reduction='none').mean(dim=[1, 2])
        loss_vae  = F.mse_loss(pred_vae, gt_vae, reduction='none').mean(dim=[1, 2])
        
        w = alpha.squeeze()
        
        weight_dino = w + epsilon
        weight_vae  = (1 - w) + epsilon
        
        total_weight = weight_dino + weight_vae
        weight_dino = weight_dino / total_weight
        weight_vae  = weight_vae  / total_weight
        
        weighted_loss = weight_dino * loss_dino + weight_vae * loss_vae
        
        return weighted_loss.mean()

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class VisualThinkingAdaLNAdapter(nn.Module):
    def __init__(self, in_dim, out_dim, t_min, t_max):
        super().__init__()
        
        self.time_embed_dim = 256
        
        self.input_norm = nn.LayerNorm(in_dim, elementwise_affine=False, eps=1e-6)
        
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_embed_dim, in_dim),
            nn.SiLU(),
            nn.Linear(in_dim, 2 * in_dim) 
        )
        
        self.head_dino = nn.Sequential(
            nn.Linear(in_dim, out_dim*3),
            nn.GELU(),
            nn.Linear(out_dim*3, out_dim)
        )
        
        self.head_vae = nn.Sequential(
            nn.Linear(in_dim, out_dim*3),
            nn.GELU(),
            nn.Linear(out_dim*3, out_dim)
        )

        self.t_min = t_min
        self.t_max = t_max

    def _get_sinusoidal_emb(self, timesteps, embedding_dim):
        assert len(timesteps.shape) == 1
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
        return emb

    def _get_alpha(self, timestep, device):
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], device=device, dtype=torch.float32)
        alpha = (timestep - self.t_min) / (self.t_max - self.t_min + 1e-6)
        alpha = alpha.clamp(0.0, 1.0)
        return alpha.view(-1, 1, 1)

    def forward(self, x, timestep):
        """
        x: [B, Seq_Len, Dim] (Static Special Token)
        timestep: [B] or Scalar
        """
        B, L, D = x.shape
        
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], device=x.device, dtype=torch.float32)
        if timestep.dim() == 0:
            timestep = timestep.repeat(B)
            
        t_emb = self._get_sinusoidal_emb(timestep, self.time_embed_dim)
        t_emb = t_emb.to(dtype=x.dtype)
        
        time_params = self.time_mlp(t_emb)
        
        shift, scale = time_params.chunk(2, dim=1)
        shift = shift.unsqueeze(1)
        scale = scale.unsqueeze(1)
        
        x_modulated = self.input_norm(x) * (1 + scale) + shift
        
        pred_dino = self.head_dino(x_modulated) 
        pred_vae  = self.head_vae(x_modulated) 
        
        alpha_view = self._get_alpha(timestep, x.device).type_as(pred_dino)
        mixed_out = alpha_view * pred_dino + (1 - alpha_view) * pred_vae
        
        return mixed_out, pred_dino, pred_vae

    def get_loss(self, pred_dino, pred_vae, gt_dino, gt_vae, timestep, epsilon=0.1):
        alpha = self._get_alpha(timestep, pred_dino.device).type_as(pred_dino)
        
        loss_dino = F.mse_loss(pred_dino, gt_dino, reduction='none').mean(dim=[1, 2])
        loss_vae  = F.mse_loss(pred_vae, gt_vae, reduction='none').mean(dim=[1, 2])
        
        w = alpha.squeeze()
        weight_dino = w + epsilon
        weight_vae  = (1 - w) + epsilon
        
        total_weight = weight_dino + weight_vae
        weight_dino = weight_dino / total_weight
        weight_vae  = weight_vae  / total_weight
        
        weighted_loss = weight_dino * loss_dino + weight_vae * loss_vae
        
        return weighted_loss.mean()

def vt_get_alpha(timestep, t_min, t_max, device):
    if not torch.is_tensor(timestep):
        timestep = torch.tensor([timestep], device=device, dtype=torch.float32)


    alpha = (timestep - t_min) / (t_max - t_min + 1e-6)
    alpha = alpha.clamp(0.0, 1.0)
    return alpha.view(-1, 1, 1)

class PhysicalTransitionAdapter(nn.Module):
    def __init__(
        self, 
        input_dim=3584,       
        hidden_dim=1024,     
        output_dim=3584,     
        num_classes=47        
    ):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim)
        )
        
        self.cls_head = nn.Linear(hidden_dim, num_classes)
        
        self.gate_head = nn.Linear(hidden_dim, 1)

        self.context_proj = VisualThinkingAdapter(in_dim=hidden_dim, out_dim=output_dim)

    def _init_weights(self):
        nn.init.zeros_(self.context_proj.weight)
        nn.init.zeros_(self.context_proj.bias)

        nn.init.constant_(self.gate_head.bias, -3.0)
        nn.init.zeros_(self.gate_head.weight)

    def forward(self, phy_token_embed):
        feat = self.backbone(phy_token_embed)
        
        logits = self.cls_head(feat)
        
        gate = torch.sigmoid(self.gate_head(feat))
        
        raw_context = self.context_proj(feat)
        
        gated_context = raw_context * gate
        
        gated_context = gated_context.unsqueeze(1)
        
        return gated_context, logits, gate

if __name__ == "__main__":
    model = PerceiverResampler(dim=3584, num_latents=32, depth=2)
        
    video_features_flat = torch.randn(2, 2048, 3584)
    
    output = model(video_features_flat)
    
    print("Input shape:", video_features_flat.shape)
    print("Output shape:", output.shape)