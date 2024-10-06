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

        super().__init__()
        self.skips = skips

        # set positional encoding function
        self.embed_pts_fn, input_pts_dim = get_embedder(multires['pts'], is_embeded['pts'], input_dims['pts'])
        self.embed_view_fn, input_view_dim = get_embedder(multires['view'], is_embeded['view'], input_dims['view'])
        self.embed_tx_fn, input_tx_dim = get_embedder(multires['tx'], is_embeded['tx'], input_dims['tx'])

        # attenuation network
        self.attenuation_linears = nn.ModuleList(
            [nn.Linear(input_pts_dim+input_pts_dim, W)] +
            [nn.Linear(W, W) if i not in skips else nn.Linear( W, W)
             for i in range(D - 1)]
        )

        self.attenuation_linears1 = nn.ModuleList(
            [nn.Linear(input_pts_dim+4, W)] +
            [nn.Linear(W, W) if i not in skips else nn.Linear(W + input_pts_dim, W)
             for i in range(D - 1)]
        )


        self.signal_linears1 = nn.ModuleList(
            [nn.Linear(W, W)] +
            [nn.Linear(W, W//2)]
        )

        self.signal_linears = nn.ModuleList(
            [nn.Linear(input_tx_dim + W, W)] +
            [nn.Linear(W, W//2)]
        )

        self.attenuation_output = nn.Linear(W, 1)

        self.feature_layer = nn.Linear(W, W)

        self.signal_output = nn.Linear(W//2, 1)


    def forward(self,tx, ris, rx):
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


        tx = self.embed_tx_fn(tx).contiguous() #tx ris
        ris = self.embed_tx_fn(ris).contiguous() #rx ris
        rx = self.embed_tx_fn(rx).contiguous()

        tx = tx.view(-1, list(tx.shape)[-1]) #tx ris
        ris = ris.view(-1, list(ris.shape)[-1]) #tx ris
        rx = rx.view(-1, list(rx.shape)[-1]) #tx ris


        x = torch.cat([tx ,ris], -1) #tx ris


        for i, layer in enumerate(self.attenuation_linears):
            x = F.relu(layer(x))

            if i in self.skips:
                x = torch.cat([x], -1)


        feature = self.feature_layer(x)

        x = torch.cat([feature, rx], -1)

        #提取第一条网络特征
        for i, layer in enumerate(self.signal_linears):
            x = F.relu(layer(x))
        output = self.signal_output(x)

        return output
