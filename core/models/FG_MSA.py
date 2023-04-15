import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

from core.utils.occu_metric import sample


class FGMSA(nn.Module):
    def __init__(
        self, q_size, kv_size, n_heads, n_head_channels, n_groups=6,
        attn_drop=0., proj_drop=0., stride=1, 
        offset_range_factor=2, use_pe=True, dwc_pe=False,
        no_off=False, fixed_pe=False, stage_idx=3,use_last_ref=False,
        out_dim=384,fg=False,in_dim=384
    ):
        super().__init__()
        self.dwc_pe = dwc_pe
        self.n_head_channels = n_head_channels
        self.scale = self.n_head_channels ** -0.5
        self.n_heads = n_heads
        self.q_h, self.q_w = q_size
        self.kv_h, self.kv_w = kv_size
        self.nc = n_head_channels * n_heads
        self.n_groups = n_groups
        self.n_group_channels = self.nc // self.n_groups
        self.n_group_heads = self.n_heads // self.n_groups
        self.use_pe = use_pe
        self.fixed_pe = fixed_pe
        self.no_off = no_off
        self.offset_range_factor = offset_range_factor
        self.use_last_ref = use_last_ref
        self.fg = fg

        self.ref_res = None
        
        ksizes = [9, 7, 5, 3]
        kk = ksizes[stage_idx]

        self.conv_offset_0 = nn.Conv2d(self.nc, self.nc, kernel_size=kk, stride=stride, padding="same", groups=self.n_groups)
        self.conv_norm = nn.LayerNorm(eps=1e-3, normalized_shape=self.nc)
        # self.out_norm = layers.LayerNormalization(1e-5)
        self.conv_offset_proj = nn.Conv2d(self.n_group_channels, 2, kernel_size=1,stride=1, bias=False)
        if self.fg:
            self.conv_offset_proj2 = nn.Conv2d(2, out_dim, kernel_size=1,stride=1)

        self.proj_q = nn.Conv2d(self.nc, self.nc, kernel_size=1, stride=1)

        self.proj_k = nn.Conv2d(self.nc, self.nc,kernel_size=1, stride=1)

        self.proj_v = nn.Conv2d(self.nc, self.nc,kernel_size=1, stride=1)

        self.proj_out = nn.Conv2d(self.nc, out_dim,kernel_size=1, stride=1)

        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)

        if self.use_pe:
            self.rpe_table = nn.Parameter(torch.fmod(torch.randn((self.kv_h * 2 - 1, self.kv_w * 2 - 1,self.n_heads)), 2) * 0.01, requires_grad=True)
        else:
            self.rpe_table = None
        
        dummy_x = torch.zeros((1,self.q_h,self.q_w,in_dim))
        self.in_dim = in_dim
        self.out_dim = out_dim
        ref = torch.zeros((1*self.n_groups,self.q_h,self.q_w,2))
        self(dummy_x,last_reference=ref)
        summary(self)
    
    def _get_offset(self,x):
        # x [B, C, H, W]
        x = self.conv_offset_0(x) # [B, C, H, W]
        x = x.permute(0,2,3,1) # [B, H, W, C]
        x = torch.reshape(x,[-1, self.q_h*self.q_w, self.nc])
        x = self.conv_norm(x)
        x = torch.reshape(x,[-1, self.q_h,self.q_w, self.nc]) # [B, H, W, C]
        x = F.gelu(x)
        x = torch.reshape(torch.reshape(x,[-1,self.q_h,self.q_w,self.n_groups,self.n_group_channels]).permute([0,3,1,2,4]),[-1,self.q_h,self.q_w,self.n_group_channels])
        x = x.permute(0,3,1,2) # channels as 2nd dim
        x = self.conv_offset_proj(x)
        x = x.permute(0,2,3,1) # channels as last dim
        return x
        
    
    def _get_ref_points(self, H_key, W_key, B):
        ref_y, ref_x = torch.meshgrid(
            torch.arange(0, H_key), 
            torch.arange(0, W_key),
            indexing="xy"
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref = ref.to(torch.float32)
        ref = torch.repeat_interleave(ref[np.newaxis,...],B*self.n_groups,dim=0)
        ref = ref.detach()
        return ref

    def forward(self, x,training=True,last_reference=None):
        B, H, W,C = x.size()
        x = x.permute(0,3,1,2) # channels as 2nd dim
        q = self.proj_q(x)
        offset = self._get_offset(q) # [B, H, W, 2]
        _,Hk,Wk,_ = offset.size()
        n_sample = Hk * Wk
        
        if self.offset_range_factor > 0:
            offset_range = torch.reshape(torch.tensor([Hk/2,Wk/2], device=x.device).detach(),(1, 1, 1, 2))
            offset = torch.tanh(offset)
            offset = torch.multiply(offset, offset_range)
            self.ref_res = torch.reshape(offset,(B,self.n_groups, Hk, Wk, 2))
        
        if self.fg:
            time_offset = torch.reshape(offset,(B * self.n_groups, Hk, Wk, 2))
            time_offset = time_offset.permute(0,3,1,2) # channels as 2nd dim
            flow_hidden = self.conv_offset_proj2(time_offset)
            flow_hidden = flow_hidden.permute(0,2,3,1) # channels last
            flow_hidden = torch.reshape(flow_hidden,(B, self.n_groups, Hk,Wk, self.out_dim))
        
        if self.use_last_ref:
            reference = torch.reshape(last_reference,(B*self.n_groups, Hk, Wk, 2))
        else:
            reference = self._get_ref_points(Hk, Wk, B).to(x.device)
            
        if self.no_off:
            offset = torch.zeros_like(offset, device=x.device)
            
        if self.offset_range_factor >= 0:
            pos = offset + reference
        else:
            pos = torch.tanh(offset + reference)
        
        x = x.permute(0,2,3,1) # channels last
        x = torch.reshape(torch.reshape(x,[B, H, W,self.n_groups,self.n_group_channels]).permute([0,3,1,2,4]),[B*self.n_groups, H, W,self.n_group_channels])
        
        warp = torch.concat([pos[...,1][...,np.newaxis],pos[...,0][...,np.newaxis]],dim=-1)
        x_sampled = sample(image=x, warp=warp,pixel_type=0)
        x_sampled = torch.reshape(torch.reshape(x, [B,self.n_groups, H, W,self.n_group_channels]).permute([0,2,3,1,4]),[B,n_sample,1,C])
        x_sampled = x_sampled.permute([0, 3, 1, 2]) # channels as 2nd dim
            
        q = torch.reshape(torch.reshape(q,(B, H * W,self.n_heads, self.n_head_channels)).permute([0,2,1,3]),[B*self.n_heads,H * W,self.n_head_channels])
        k = torch.reshape(torch.reshape(self.proj_k(x_sampled),(B, n_sample,self.n_heads, self.n_head_channels)).permute([0,2,1,3]),[B*self.n_heads,n_sample,self.n_head_channels])
        v = torch.reshape(torch.reshape(self.proj_v(x_sampled),(B, n_sample,self.n_heads, self.n_head_channels)).permute([0,2,1,3]),[B*self.n_heads,n_sample,self.n_head_channels])
        attn = torch.einsum('bqc, bkc-> bqk', q, k)
        attn = attn * self.scale
        
        if self.use_pe:
            rpe_table = self.rpe_table
            # rpe_bias = rpe_table[None, ...].expand(B, -1, -1, -1)
            rpe_bias = torch.repeat_interleave(rpe_table[np.newaxis,...],repeats=B,dim=0)
            
            q_grid = self._get_ref_points(H, W, B).to(x.device)
            
            displacement = torch.unsqueeze(torch.reshape(q_grid,(B * self.n_groups, H * W, 2)),dim=2) - torch.unsqueeze(torch.reshape(pos,(B * self.n_groups, n_sample, 2)),dim=1)

            rpe_bias = torch.reshape(rpe_bias, (B,2 * H - 1, 2 * W - 1,self.n_groups, self.n_group_heads)).permute([0,3,1,2,4])
            displacement = torch.concat([displacement[...,1][...,np.newaxis],displacement[...,0][...,np.newaxis]],dim=-1)
            
            attn_bias = sample(
                image=torch.reshape(rpe_bias,[B * self.n_groups,2 * H - 1, 2 * W - 1,self.n_group_heads]),
                warp=displacement,
                pixel_type=0
            )

            attn_bias = torch.reshape(attn_bias, [B*self.n_groups,H*W,n_sample,self.n_group_heads]).permute([0,3,1,2])
            
            attn_bias = torch.reshape(attn_bias,[B*self.n_heads,H*W,n_sample] )
            
            attn = attn + attn_bias

        attn = F.softmax(attn, dim=2)
        attn = self.attn_drop(attn)
        out = torch.einsum('bkv, bvc -> bck', attn, v)
        out = torch.reshape(out,(B,C,H,W))
        y = self.proj_drop(self.proj_out(out))
        y = y.permute([0,2,3,1]) # channels as last dim

        if self.fg:
            return y,torch.reshape(pos,(B, self.n_groups, Hk, Wk, 2)),flow_hidden
        
        return y, torch.reshape(pos,(B, self.n_groups, Hk, Wk, 2)), torch.reshape(reference,(B, self.n_groups, Hk, Wk, 2))


if __name__=='__main__':
    FGMSA(q_size=(32,32), kv_size=(32,32), n_heads=8, n_head_channels=24,n_groups=8,in_dim=192,out_dim=192)