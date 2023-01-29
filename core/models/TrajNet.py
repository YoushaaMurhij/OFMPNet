import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from core.models.MultiHeadAttention import MultiHeadAttention

from torchinfo import summary



class TrajEncoder(nn.Module):
    def __init__(self,num_heads=4,out_dim=256):
        super(TrajEncoder, self).__init__()
        self.node_feature = nn.Sequential(nn.Conv1d(5, 64, kernel_size=1), nn.ELU())
        self.node_attention = MultiHeadAttention(input_channels=(64,64,64), num_heads=num_heads, head_size=64, dropout=0.1, output_size=64*5)
        self.vector_feature = nn.Linear(3, 64, bias=False)
        self.sublayer = nn.Sequential(nn.Linear(384, out_dim), nn.ELU())

    def forward(self, inputs, mask):
        mask = mask.to(torch.float32)
        mask = torch.matmul(mask[:, :, np.newaxis], mask[:, np.newaxis, :])
        nodes = self.node_feature(inputs[:, :, :5].permute(0,2,1))
        nodes = nodes.permute(0,2,1)
        nodes = self.node_attention(inputs=[nodes, nodes, nodes], mask=mask)
        nodes, _ = torch.max(nodes, 1)
        vector = self.vector_feature(inputs[:, 0, 5:])
        out = torch.concat([nodes, vector], dim=1)
        polyline_feature = self.sublayer(out)

        return polyline_feature


    
class Cross_Attention(nn.Module):
    def __init__(self, num_heads, key_dim, conv_attn=False):
        super(Cross_Attention, self).__init__()
        self.mha = MultiHeadAttention(input_channels=(key_dim, key_dim), num_heads=num_heads, head_size=key_dim//num_heads, output_size=key_dim, dropout=0.1)
        self.norm1 = nn.LayerNorm(eps=1e-3, normalized_shape=key_dim)
        self.norm2 = nn.LayerNorm(eps=1e-3, normalized_shape=key_dim)
        self.FFN1 = nn.Sequential(nn.Linear(key_dim, 4*key_dim),nn.ELU())
        self.dropout1 = nn.Dropout(0.1)
        self.FFN2 = nn.Sequential(nn.Linear(4*key_dim, key_dim),nn.ELU())
        self.dropout2 = nn.Dropout(0.1)

    def forward(self, query, key, mask=None, training=True):
        value = self.mha(inputs=[query, key], mask=mask)
        value = self.norm1(value)
        value = self.FFN1(value)
        value = self.dropout1(value)
        value = self.FFN2(value)
        value = self.dropout2(value)
        value = self.norm2(value)
        return value



class TrajNet(nn.Module):
    def __init__(self,cfg,past_to_current_steps=11,obs_actors=48,occ_actors=16,actor_only=True,no_attn=False,
        double_net=False):
        super(TrajNet, self).__init__()

        self.actor_only = actor_only
        self.obs_actors=obs_actors
        self.occ_actors=occ_actors
        self.double_net = double_net
        
        self.traj_encoder = TrajEncoder(num_heads=cfg['traj_heads'],out_dim=cfg['out_dim'])
        # self.traj_encoder  = TrajEncoderLSTM(cfg['out_dim'])
        self.no_attn = no_attn
        if not no_attn:
            if double_net:
                self.cross_attention = nn.Module([Cross_AttentionT(num_heads=cfg['att_heads'],key_dim=192,output_dim=cfg['out_dim']) for _ in range(2)])
            else:
                self.cross_attention = Cross_Attention(num_heads=cfg['att_heads'], key_dim=cfg['out_dim'])

        self.obs_norm = nn.LayerNorm(eps=1e-3, normalized_shape=cfg['out_dim'])
        self.occ_norm = nn.LayerNorm(eps=1e-3, normalized_shape=cfg['out_dim'])
        # self.obs_drop = tf.keras.layers.Dropout(0.1)
        # self.occ_drop = tf.keras.layers.Dropout(0.1)

        # dummy_obs_actors = torch.zeros([2,obs_actors,past_to_current_steps,8])
        # dummy_occ_actors = torch.zeros([2,occ_actors,past_to_current_steps,8])
        # dummy_ccl = tf.zeros([1,256,10,7])

        self.bi_embed = torch.tensor([[1,0],[0,1]], dtype=torch.float32).repeat_interleave(torch.tensor([obs_actors, occ_actors]), dim=0)
        self.seg_embed = nn.Linear(2, cfg['out_dim'], bias=False)

        # self(dummy_obs_actors,dummy_occ_actors)
        # summary(self)
    
    def forward(self,obs_traj,occ_traj,map_traj=None,training=True):

        obs_mask = torch.not_equal(obs_traj, 0)[:,:,:,0]
        obs = [self.traj_encoder(obs_traj[:, i],obs_mask[:,i]) for i in range(self.obs_actors)]
        obs = torch.stack(obs,dim=1)

        occ_mask = torch.not_equal(occ_traj, 0)[:,:,:,0]
        occ = [self.traj_encoder(occ_traj[:, i],occ_mask[:,i]) for i in range(self.occ_actors)]
        occ = torch.stack(occ,dim=1)

        embed = self.bi_embed[np.newaxis, :, :].repeat_interleave(occ.size()[0], dim=0).to(obs_traj.device)
        embed = self.seg_embed(embed)

        c_attn_mask = torch.not_equal(torch.max(torch.concat([obs_mask,occ_mask], dim=1).to(torch.int32),dim=-1)[0],0) #[batch,64] (last step denote the current)
        c_attn_mask = c_attn_mask.to(torch.float32)

        if self.no_attn:
            if self.double_net:
                concat_actors = torch.concat([obs,occ], dim=1)
                obs = self.obs_norm(concat_actors+embed)
                occ = self.occ_norm(concat_actors+embed)
                return obs,occ,c_attn_mask
            else:
                return self.obs_norm(obs + embed[:,:self.obs_actors,:]),self.occ_norm(occ + embed[:,self.obs_actors:,:]),c_attn_mask

        # interactions given seg_embedding
        concat_actors = torch.concat([obs,occ], dim=1)
        concat_actors = torch.multiply(c_attn_mask[:, :, np.newaxis].to(torch.float32), concat_actors)
        query = concat_actors + embed

        attn_mask = torch.matmul(c_attn_mask[:, :, np.newaxis], c_attn_mask[:, np.newaxis, :]) #[batch,64,64]

        if self.double_net:
            value = self.cross_attention[0](query=query, key=concat_actors, mask=attn_mask, training=training)
            val_obs,val_occ = value[:,:self.obs_actors,:] , value[:,self.obs_actors:,:]

            value_flow = self.cross_attention[1](query=query, key=concat_actors, mask=attn_mask, training=training)
            val_obs_f,val_occ_f = value_flow[:,:self.obs_actors,:] , value_flow[:,self.obs_actors:,:]

            obs = obs + val_obs
            occ = occ + val_occ

            ogm = torch.concat([obs,occ], dim=1) + embed

            obs_f = obs + val_obs_f
            occ_f = occ + val_occ_f

            flow = torch.concat([obs_f,occ_f], dim=1) + embed

            return self.obs_norm(ogm) , self.occ_norm(flow) , c_attn_mask
        
        value = self.cross_attention(query=query, key=concat_actors, mask=attn_mask, training=training)
        val_obs,val_occ = value[:,:self.obs_actors,:] , value[:,self.obs_actors:,:]

        obs = obs + val_obs
        occ = occ + val_occ

        concat_actors = torch.concat([obs,occ], dim=1)

        obs = self.obs_norm(obs + embed[:,:self.obs_actors,:])
        occ = self.occ_norm(occ + embed[:,self.obs_actors:,:])

        return obs,occ,c_attn_mask

class Cross_AttentionT(nn.Module):
    def __init__(self, num_heads, key_dim,output_dim,conv_attn=False,sep_actors=False):
        super(Cross_AttentionT, self).__init__()
        self.mha = MultiHeadAttention(input_channels=(key_dim * num_heads,key_dim * num_heads), num_heads=num_heads, head_size=key_dim//num_heads,output_size=key_dim,dropout=0.1)
        self.sep_actors = sep_actors
        
        self.norm1 = nn.LayerNorm(eps=1e3, normalized_shape=key_dim)
        self.norm2 = nn.LayerNorm(eps=1e3, normalized_shape=output_dim)
        self.FFN1 = nn.Sequential(nn.Linear(key_dim, 4*key_dim), nn.ELU())
        self.dropout1 = nn.Dropout(0.1)
        self.FFN2 = nn.Linear(4 * key_dim, output_dim)
        self.dropout2 = nn.Dropout(0.1)
        self.conv_attn = conv_attn

    def forward(self, query, key, mask, training=True,actor_mask=None):
        value = self.mha(inputs=[query, key], mask=mask)
        value = self.norm1(value)
        value = self.FFN1(value)
        value = self.dropout1(value)
        value = self.FFN2(value)
        value = self.dropout2(value)
        value = self.norm2(value)
        return value

class TrajNetCrossAttention(nn.Module):
    def __init__(self,traj_cfg,pic_size=(8,8),pic_dim=768,past_to_current_steps=11,obs_actors=48,occ_actors=16,actor_only=True,
        multi_modal=True,sep_actors=False):
        super(TrajNetCrossAttention, self).__init__()

        self.traj_net = TrajNet(traj_cfg,no_attn=traj_cfg['no_attn'],
            past_to_current_steps=past_to_current_steps,obs_actors=obs_actors,occ_actors=occ_actors,actor_only=actor_only,
            double_net=False)

        self.obs_actors = obs_actors
        self.H, self.W = pic_size
        self.pic_dim = pic_dim

        self.multi_modal = multi_modal
        self.actor_only = actor_only
        self.sep_actors = sep_actors
  
        self.cross_attn_obs = nn.ModuleList([Cross_AttentionT(num_heads=3, output_dim=pic_dim,key_dim=128,sep_actors=sep_actors) for _ in range(8)])

        # dummy_obs_actors = torch.zeros([2,obs_actors,past_to_current_steps,8])
        # dummy_occ_actors = torch.zeros([2,occ_actors,past_to_current_steps,8])
        # dummy_ccl = torch.zeros([2,256,10,7])
        # dummy_pic_encode = torch.zeros((2,) + pic_size + (pic_dim,))
        
        # flow_pic_encode = torch.zeros((2,) + pic_size + (pic_dim,))
        # if multi_modal:
        #     dummy_pic_encode = torch.zeros((2,8,) + pic_size + (pic_dim,))

        # self(dummy_pic_encode,dummy_obs_actors,dummy_occ_actors,dummy_ccl,flow_pic_encode=flow_pic_encode)
        # summary(self)
    
    def map_encode(self,map_traj,training):

        segs = map_traj.get_shape()[1]
        map_mask = torch.not_equal(map_traj[:,:,:,0], 0) #[batch,256,10]
        amap_mask = torch.reshape(map_mask,[-1,10])
        map_traj = torch.reshape(map_traj, [-1,10,7])
        map_enc = self.map_encoder(map_traj,amap_mask,training)
        map_enc = torch.reshape(map_enc,[-1,256,map_enc.get_shape()[-1]])

        map_mask = map_mask[:,:,0].to(torch.int32)#[batch,256]
        return map_enc,map_mask


    def forward(self,pic_encode,obs_traj,occ_traj,map_traj=None,training=True,flow_pic_encode=None):

        obs,occ,traj_mask = self.traj_net(obs_traj,occ_traj,map_traj,training)

        if self.sep_actors:
            actor_mask = torch.matmul(traj_mask[:, :, np.newaxis], traj_mask[:, np.newaxis, :])
        
        flat_encode = torch.reshape(pic_encode, shape=[-1,8,self.H*self.W,self.pic_dim])
        pic_mask = torch.ones_like(flat_encode[:,0,:,0],dtype=torch.float32)

        obs_attn_mask = torch.matmul(pic_mask[:, :, np.newaxis], traj_mask[:, np.newaxis, :])

        query = flat_encode
        key = torch.concat([obs,occ], dim=1)
        res_list = []
        for i in range(8):
            if self.sep_actors:
                o = self.cross_attn_obs[i](query[:,i],key,obs_attn_mask,training,actor_mask)
            else:
                o = self.cross_attn_obs[i](query[:,i],key,obs_attn_mask,training)
            v = o + query[:,i]
            res_list.append(v)
            
        obs_value = torch.stack(res_list,dim=1)
        obs_value = torch.reshape(obs_value, shape=[-1,8,self.H,self.W,self.pic_dim])

        return obs_value





if __name__=='__main__':
    cfg=dict(input_size=(512,512), window_size=8, embed_dim=96, depths=[2,2,2], num_heads=[3,6,12])
    sep_actors = False
    actor_only = True
    past_to_current_steps=11
    obs_actors=48
    occ_actors=16
    if sep_actors:
            traj_cfg = dict(traj_heads=4,att_heads=6,out_dim=384,no_attn=True)
    else:
        traj_cfg = dict(traj_heads=4,att_heads=6,out_dim=384,no_attn=False)
    

    # TrajNet(traj_cfg,no_attn=traj_cfg['no_attn'],
    #         past_to_current_steps=past_to_current_steps,obs_actors=obs_actors,occ_actors=occ_actors,actor_only=actor_only,
    #         double_net=False)
        
    resolution=[8,16,32]
    hw = resolution[4-len(cfg['depths'][:])]
    TrajNetCrossAttention(traj_cfg,actor_only=actor_only,pic_size=(hw,hw),pic_dim=768//(2**(4-len(cfg['depths'][:])))
        ,multi_modal=True,sep_actors=sep_actors)