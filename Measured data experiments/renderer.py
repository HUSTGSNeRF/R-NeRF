# -*- coding: utf-8 -*-
"""code for ray marching and signal rendering
"""
import torch
import numpy as np
import torch.nn.functional as F
import scipy.constants as sc
from einops import rearrange, repeat


class Renderer():

    def __init__(self, networks_fn, **kwargs) -> None:
        """
        Parameters
        -----------------
        near : float. The near bound of the rays
        far : float. The far bound of the rays
        n_samples: int. num of samples per ray
        """

        ## Rendering parameters
        self.network_fn = networks_fn
        self.n_samples = kwargs['n_samples']
        self.near = kwargs['near']
        self.far = kwargs['far']


    def sample_points(self, rays_o, rays_d):
        """sample points along rays

        Parameters
        ----------
        rays_o : tensor. [n_rays, 3]. The origin of rays
        rays_d : tensor. [n_rays, 3]. The direction of rays

        Returns
        -------
        pts : tensor. [n_rays, n_samples, 3]. The sampled points along rays
        t_vals : tensor. [n_rays, n_samples]. The distance from origin to each sampled point
        """
        shape = list(rays_o.shape)
        shape[-1] = 1
        near, far = torch.full(shape, self.near), torch.full(shape, self.far)
        t_vals = torch.linspace(0., 1., steps=self.n_samples) * (far - near) + near  # scale t with near and far
        t_vals = t_vals.to(rays_o.device)
        pts = rays_o[...,None,:] + rays_d[...,None,:] * t_vals[...,:,None] # p = o + td, [n_rays, n_samples, 3]

        return pts, t_vals

class Renderer_RSSI(Renderer):
    """Renderer for RSSI (integral from all directions)
    """

    def __init__(self, networks_fn, **kwargs) -> None:
        """
        Parameters
        -----------------
        near : float. The near bound of the rays
        far : float. The far bound of the rays
        n_samples: int. num of samples per ray
        """
        super().__init__(networks_fn, **kwargs)


    def render_rssi(self, tx, rays_o, rays_d, ris ,rays_o1,rays_d1):
        """render the RSSI for each gateway. To avoid OOM, we split the rays into chunks

        Parameters
        ----------
        tx: tensor. [batchsize, 3]. The position of the transmitter
        rays_o : tensor. [batchsize, 3]. The origin of rays
        rays_d : tensor. [batchsize, 9x36x3]. The direction of rays
        """

        batchsize, _ = tx.shape
        batchsize1, _ = ris.shape

        rays_d = torch.reshape(rays_d, (batchsize, -1, 3))    # [batchsize, 9x36, 3] tx ris
        rays_d1 = torch.reshape(rays_d1, (batchsize1, -1, 3))    # [batchsize, 9x36, 3] rx ris


        chunks = 36
        chunks_num = 36 // chunks

        rays_o_chunk = rays_o.expand(chunks, -1, -1).permute(1,0,2) #[bs, cks, 3] tx ris
        rays_o_chunk1 = rays_o1.expand(chunks, -1, -1).permute(1,0,2) #[bs, cks, 3] rx ris


        tags_chunk = tx.expand(chunks, -1, -1).permute(1,0,2)        #[bs, cks, 3]tx ris
        tags_chunk1 = ris.expand(chunks, -1, -1).permute(1,0,2)        #[bs, cks, 3]rx ris


        recv_signal = torch.zeros(batchsize1).cuda() # tx ris
        #recv_signal1 = torch.zeros(batchsize1).cuda() # rx ris

        for i in range(chunks_num):

            rays_d_chunk = rays_d[:,i*chunks:(i+1)*chunks, :]  # [bs, cks, 3] tx ris
            rays_d_chunk1 = rays_d1[:,i*chunks:(i+1)*chunks, :]  # [bs, cks, 3] rx ris


            pts, t_vals = self.sample_points(rays_o_chunk, rays_d_chunk) # [bs, cks, pts, 3] tx ris
            pts1, t_vals1 = self.sample_points(rays_o_chunk1, rays_d_chunk1) # [bs, cks, pts, 3] rx ris


            views_chunk = rays_d_chunk[..., None, :].expand(pts.shape)  # [bs, cks, pts, 3] tx ris
            views_chunk1 = rays_d_chunk1[..., None, :].expand(pts1.shape)  # [bs, cks, pts, 3]rx ris

            tx_chunk = tags_chunk[..., None, :].expand(pts.shape)  # [bs, cks, pts, 3]
            tx_chunk1 = tags_chunk1[..., None, :].expand(pts1.shape)  # [bs, cks, pts, 3]

            # Run network and compute outputs
            #print(self.network_fn)
            raw, raw1= self.network_fn(pts, views_chunk, tx_chunk, pts1, views_chunk1, tx_chunk1)    # [batchsize, chunks, n_samples, 4]

            # print("sssssssssssssssssssssssss",raw)
            recv_signal_chunks = self.raw2outputs_signal(raw, raw1, t_vals, t_vals1, rays_d_chunk, rays_d_chunk1)  # [bs]
            recv_signal += recv_signal_chunks

        #print(raw.shape)
        return recv_signal    # [batchsize,]


    def raw2outputs_signal(self, raw, raw1, r_vals,r_vals1, rays_d,rays_d1):
        """Transforms model's predictions to semantically meaningful values.

        Parameters
        ----------
        raw : [batchsize, chunks,n_samples,  4]. Prediction from model.
        r_vals : [batchsize, chunks, n_samples]. Integration distance.
        rays_d : [batchsize,chunks, 3]. Direction of each ray

        Return:
        ----------
        receive_signal : [batchsize]. abs(singal of each ray)
        """
        wavelength = sc.c / 5.8e9
        raw2phase = lambda raw, dists: raw + 2 * np.pi * dists / wavelength
        raw2phase1 = lambda raw1, dists1: raw1 + 2*np.pi*dists1/wavelength

        raw2amp = lambda raw, dists: -raw*dists
        raw2amp1 = lambda raw1, dists1: -raw1 * dists1

        dists = r_vals[...,1:] - r_vals[...,:-1]
        dists1 = r_vals1[..., 1:] - r_vals1[..., :-1]

        dists = torch.cat([dists, torch.Tensor([1e10]).cuda().expand(dists[...,:1].shape)], -1)  # [batchsize, chunks, n_samples]
        dists = dists * torch.norm(rays_d[...,None,:], dim=-1)  # [batchsize,chunks, n_samples, 3].

        dists1 = torch.cat([dists1, torch.Tensor([1e10]).cuda().expand(dists1[..., :1].shape)], -1)  # [batchsize, chunks, n_samples]
        dists1 = dists1 * torch.norm(rays_d1[..., None, :], dim=-1)  # [batchsize,chunks, n_samples, 3]

        att_a, att_p, s_a, s_p = raw[..., 0], raw[..., 1], raw[..., 2], raw[..., 3]
        att_a1, att_p1, s_a1, s_p1 = raw1[...,0], raw1[...,1], raw1[...,2], raw1[...,3]    # [batchsize,chunks, N_samples]

        att_p, s_p = torch.sigmoid(att_p)*np.pi*2-np.pi, torch.sigmoid(s_p)*np.pi*2-np.pi
        att_a, s_a = abs(F.leaky_relu(att_a)), abs(F.leaky_relu(s_a))

        att_p1, s_p1 = torch.sigmoid(att_p1)*np.pi*2-np.pi, torch.sigmoid(s_p1)*np.pi*2-np.pi
        att_a1, s_a1 = abs(F.leaky_relu(att_a1)), abs(F.leaky_relu(s_a1))

        #计算幅度和相位的累积值，以便进行积分。
        amp = raw2amp(att_a, dists)  # [batchsize,chunks, N_samples]
        phase = raw2phase(att_p, dists)

        amp1 = raw2amp1(att_a1, dists1)  # [batchsize,chunks, N_samples]
        phase1 = raw2phase1(att_p1, dists1)

        #将每个射线的幅度和相位值进行累积，并根据输入参数计算接收信号的强度。       
        amp_i = torch.exp(torch.cumsum(amp, -1))            # [batchsize,chunks, N_samples]
        phase_i = torch.exp(1j*torch.cumsum(phase, -1))                # [batchsize,chunks, N_samples]

        amp_i1 = torch.exp(torch.cumsum(amp1, -1))            # [batchsize,chunks, N_samples]
        phase_i1 = torch.exp(1j*torch.cumsum(phase1, -1))                # [batchsize,chunks, N_samples]

        #最后，对每个样本的所有射线的接收信号强度进行求和，得到最终的接收信号强度。
        recv_signal = torch.sum(s_a*torch.exp(1j*s_p)*s_a1*torch.exp(1j*s_p1)*amp_i*phase_i*amp_i1*phase_i1, -1)  # integral along line [batchsize,chunks]
        recv_signal = torch.sum(recv_signal, -1)   # integral along direction [batchsize,]

        return abs(recv_signal)

renderer_dict = {"spectrum": Renderer_spectrum, "rssi": Renderer_RSSI, "csi": Renderer_CSI}