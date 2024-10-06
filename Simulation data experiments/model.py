# -*- coding: utf-8 -*-
"""NeRF2 NN model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# Misc
img2mse = lambda x, y : torch.mean((x - y) ** 2)
img2me = lambda x, y : torch.mean(abs(x - y))
sig2mse = lambda x, y : torch.mean((x - y) ** 2)
csi2snr = lambda x, y: -10 * torch.log10(
    torch.norm(x - y, dim=(1, 2)) ** 2 /
    torch.norm(y, dim=(1, 2)) ** 2
)




class Embedder():
    """positional encoding
    """
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']    # input dimension of gamma
        out_dim = 0

        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']    # L-1, 10-1 by default
        N_freqs = self.kwargs['num_freqs']         # L


        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)  #2^[0,1,...,L-1]
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        """return: gamma(input)
        """
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)




def get_embedder(multires, is_embeded=True, input_dims=3):
    """get positional encoding function

    Parameters
    ----------
    multires : log2 of max freq for positional encoding, i.e., (L-1)
    i : set 1 for default positional encoding, 0 for none
    input_dims : input dimension of gamma


    Returns
    -------
        embedding function; output_dims
    """
    if is_embeded == False:
        return nn.Identity(), input_dims

    embed_kwargs = {
                'include_input' : True,
                'input_dims' : input_dims,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim



class NeRF2(nn.Module):

    def __init__(self, D=8, W=256, skips=[4],
                 input_dims={'pts':3, 'view':3, 'tx':3},
                 multires = {'pts':10, 'view':10, 'tx':10},
                 is_embeded={'pts':True, 'view':True, 'tx':False},
                 attn_output_dims=2, sig_output_dims=2):
        """NeRF2 model

        Parameters
        ----------
        D : int, hidden layer number, default by 8
        W : int, Dimension per hidden layer, default by 256
        skip : list, skip layer index
        input_dims: dict, input dimensions
        multires: dict, log2 of max freq for position, view, and tx position positional encoding, i.e., (L-1)
        is_embeded : dict, whether to use positional encoding
        attn_output_dims : int, output dimension of attenuation
        sig_output_dims : int, output dimension of signal
        """

        super().__init__()
        self.skips = skips

        # set positional encoding function
        self.embed_pts_fn, input_pts_dim = get_embedder(multires['pts'], is_embeded['pts'], input_dims['pts'])
        self.embed_view_fn, input_view_dim = get_embedder(multires['view'], is_embeded['view'], input_dims['view'])
        self.embed_tx_fn, input_tx_dim = get_embedder(multires['tx'], is_embeded['tx'], input_dims['tx'])

        # attenuation network
        self.attenuation_linears = nn.ModuleList(
            [nn.Linear(input_pts_dim, W)] +
            [nn.Linear(W, W) if i not in skips else nn.Linear(W + input_pts_dim, W)
             for i in range(D - 1)]
        )

        self.attenuation_linears1 = nn.ModuleList(
            [nn.Linear(input_pts_dim, W)] +
            [nn.Linear(W, W) if i not in skips else nn.Linear(W + input_pts_dim, W)
             for i in range(D - 1)]
        )


        ## signal network
        self.signal_linears = nn.ModuleList(
            [nn.Linear(input_view_dim + input_tx_dim + W, W)] +
            [nn.Linear(W, W//2)]
        )

        self.signal_linears1 = nn.ModuleList(
            [nn.Linear(input_view_dim + input_tx_dim + W, W)] +
            [nn.Linear(W, W//2)]
        )

        ## output head, 2 for amplitude and phase
        self.attenuation_output1 = nn.Linear(W, attn_output_dims) # 第一根网络的幅度

        self.attenuation_output2 = nn.Linear(W, attn_output_dims) ## 第二根网络的幅度


        self.feature_layer = nn.Linear(W, W)

        self.feature_layer2 = nn.Linear(W, W)

        self.feature_layer1 = nn.Linear(W // 2, W // 4)  # 提取第一条网络特征--------------

        self.signal_output1 = nn.Linear(W//2, sig_output_dims) #第一根网络的相位

        self.signal_output2 = nn.Linear(W // 2, sig_output_dims) #第二根网络的相位




    def forward(self, pts, view, tx, pts1, view1, tx1):
    #def forward(self,pts1 , view1, tx1):
        """forward function of the model

        Parameters
        ----------
        pts: [batchsize, n_samples, 3], position of voxels
        view: [batchsize, n_samples, 3], view direction
        tx: [batchsize, n_samples, 3], position of transmitter

        Returns
        ----------
        outputs: [batchsize, n_samples, 4].   attn_amp, attn_phase, signal_amp, signal_phase
        """
        pts = self.embed_pts_fn(pts).contiguous() #tx ris

        pts1 = self.embed_pts_fn(pts1).contiguous() #rx ris

        view = self.embed_view_fn(view).contiguous() #tx ris
        view1 = self.embed_view_fn(view1).contiguous() #rx ris

        tx = self.embed_tx_fn(tx).contiguous() #tx ris
        tx1 = self.embed_tx_fn(tx1).contiguous() #rx ris

        shape = pts.shape  #tx ris

        pts = pts.view(-1, list(pts.shape)[-1])#tx ris


        view = view.view(-1, list(view.shape)[-1])#tx ris
        view1 = view1.view(-1, list(view1.shape)[-1])#rx ris

        tx = tx.view(-1, list(tx.shape)[-1]) #tx ris
        tx1 = tx1.view(-1, list(tx1.shape)[-1]) #rx ris

        x = pts #tx ris

        for i, layer in enumerate(self.attenuation_linears):
            x = F.relu(layer(x))
            if i in self.skips:
                x = torch.cat([pts, x], -1)


        attn1 = self.attenuation_output1(x)    # (batch_size, 2)第一个网络的幅度
        feature = self.feature_layer(x)

        x = torch.cat([feature, view, tx], -1)


        #提取第一条网络特征
        for i, layer in enumerate(self.signal_linears):
            x = F.relu(layer(x))

        sign1 = self.signal_output1(x)


        #feature_first = torch.cat([attn1,sign1], -1)

        outputs1 = torch.cat([attn1, sign1], -1).contiguous()  # [batchsize, n_samples, 4]tx rx

        outputs1 = outputs1.view(shape[:-1] + outputs1.shape[-1:])



        shape = pts1.shape

        pts1 = pts1.view(-1, list(pts1.shape)[-1])  # rx ris

        x = pts1  # rx ris


        for m, layer in enumerate(self.attenuation_linears1):

            x = F.relu(layer(x))
            if m in self.skips:
                x = torch.cat([pts1, x], -1)


        attn2 = self.attenuation_output2(x)

        feature2 = self.feature_layer2(x)

        x1 = torch.cat([feature2, view1, tx1], -1)

        for i, layer in enumerate(self.signal_linears1):
            x1 = F.relu(layer(x1))
        signal_phase2 = self.signal_output2(x1)    #[batchsize, n_samples, 2]

        outputs2 = torch.cat([attn2, signal_phase2], -1).contiguous()    # [batchsize, n_samples, 4]tx rx

        outputs2 = outputs2.view(shape[:-1] + outputs2.shape[-1:])

        return outputs1, outputs2
