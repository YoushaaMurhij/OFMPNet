import torch
from core.models.SwinTransformerEncoder import SwinTransformerEncoder
from core.models.TrajNet import TrajNetCrossAttention
from core.models.FG_MSA import FGMSA
from core.models.Pyramid3DDecoder import Pyramid3DDecoder

from torchinfo import summary



class OFMPNet(torch.nn.Module):
    def __init__(self,cfg,use_pyramid=True,actor_only=True,sep_actors=False,
        fg_msa=False,use_last_ref=False,fg=False,large_ogm=True):

        super(OFMPNet, self).__init__()

        self.encoder = SwinTransformerEncoder(include_top=True,img_size=cfg['input_size'], window_size=cfg[
            'window_size'], embed_dim=cfg['embed_dim'], depths=cfg['depths'], num_heads=cfg['num_heads'],
            sep_encode=True,flow_sep=True,use_flow=True,drop_rate=0.0, attn_drop_rate=0.0,drop_path_rate=0.1,
            large_input=large_ogm)

         
        if sep_actors:
            traj_cfg = dict(traj_heads=4,att_heads=6,out_dim=384,no_attn=True)
        else:
            traj_cfg = dict(traj_heads=4,att_heads=6,out_dim=384,no_attn=False)
        
        resolution=[8,16,32]
        hw = resolution[4-len(cfg['depths'][:])]
        self.trajnet_attn = TrajNetCrossAttention(traj_cfg,actor_only=actor_only,pic_size=(hw,hw),pic_dim=768//(2**(4-len(cfg['depths'][:])))
        ,multi_modal=True,sep_actors=sep_actors)
        self.fg_msa = fg_msa
        self.fg = fg
        if fg_msa:
            self.fg_msa_layer = FGMSA(q_size=(16,16), kv_size=(16,16),n_heads=8,n_head_channels=48,n_groups=8,out_dim=384,use_last_ref=False,fg=fg)
        self.decoder = Pyramid3DDecoder(config=None,img_size=cfg['input_size'],pic_dim=768//(2**(4-len(cfg['depths'][:]))),use_pyramid=use_pyramid,timestep_split=True,
        shallow_decode=(4-len(cfg['depths'][:])),flow_sep_decode=True,conv_cnn=False)

        # dummy_ogm =torch.zeros((1,)+cfg['input_size']+(11,2,))
        # dummy_map =torch.zeros((1,)+(256,256)+(3,))

        # dummy_obs_actors = torch.zeros([1,48,11,8])
        # dummy_occ_actors = torch.zeros([1,16,11,8])
        # dummy_ccl = torch.zeros([1,256,10,7])
        # dummy_flow =torch.zeros((1,)+cfg['input_size']+(2,))
        # self.ref_res = None

        # self(dummy_ogm,dummy_map,obs=dummy_obs_actors,occ=dummy_occ_actors,mapt=dummy_ccl,flow=dummy_flow)
        summary(self)
    
    def forward(self,ogm,map_img,training=True,obs=None,occ=None,mapt=None,flow=None,dense_vec=None,dense_map=None):

        #visual encoder:
        res_list = self.encoder(ogm,map_img,flow,training)
        q = res_list[-1]

        if self.fg_msa:
            q = torch.reshape(q,[-1,16,16,384])
            #fg-msa:
            res,pos,ref = self.fg_msa_layer(q,training=training)
            q = res + q
            q = torch.reshape(q,[-1,16*16,384])
        query = torch.repeat_interleave(torch.unsqueeze(q, dim=1),repeats=8,axis=1)
        if self.fg:
            # added Projected flow-features to each timestep
            ref = torch.reshape(ref,[-1,8,256,384])
            query = ref + query
        
        #time-sep-cross attention and vector encoders:
        obs_value = self.trajnet_attn(query,obs,occ,mapt,training)

        #fpn decoding:
        y = self.decoder(obs_value,training,res_list)
        y = torch.reshape(y.permute([0,2,3,1,4]),[-1,256,256,32])
        return y

if __name__=="__main__":
    cfg=dict(input_size=(512,512), window_size=8, embed_dim=96, depths=[2,2,2], num_heads=[3,6,12])
    model = OFMPNet(cfg,actor_only=True,sep_actors=False,fg_msa=True)