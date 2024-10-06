# -*- coding: utf-8 -*-
"""dataset processing and loading
"""
import os
import random

import imageio
import numpy as np
import pandas as pd
import torch
import yaml
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
from tqdm import tqdm
from einops import rearrange


def rssi2amplitude(rssi):
    """convert rssi to amplitude
    """
    return (rssi-0.0298)/(123-0.0298)


def amplitude2rssi(amplitude):
    """convert amplitude to rssi
    """
    return amplitude*(123-0.0298)+0.0298


def split_dataset(datadir, ratio=0.8, dataset_type='rfid'):
    """random shuffle train/test set
    """
    if dataset_type == "rfid":
        spectrum_dir = os.path.join(datadir, 'spectrum')
        spt_names = sorted([f for f in os.listdir(spectrum_dir) if f.endswith('.png')])
        index = [x.split('.')[0] for x in spt_names]
        random.shuffle(index)
    elif dataset_type == "ble":
        rssi_dir = os.path.join(datadir, 'gateway_rssi.csv')
        index = pd.read_csv(rssi_dir).index.values
        random.shuffle(index)
    elif dataset_type == "mimo":
        csi_dir = os.path.join(datadir, 'csidata.pt')
        index = [i for i in range(torch.load(csi_dir).shape[0])]
        random.shuffle(index)

    train_len = int(len(index) * ratio)
    train_index = np.array(index[:train_len])
    test_index = np.array(index[train_len:])

    np.savetxt(os.path.join(datadir, "train_index.txt"), train_index, fmt='%s')
    np.savetxt(os.path.join(datadir, "test_index.txt"), test_index, fmt='%s')


class BLE_dataset(Dataset):
    """ble dataset class
    """
    def __init__(self, datadir, indexdir, scale_worldsize=1) -> None:
        super().__init__()
        self.datadir = datadir
        tx_pos_dir = os.path.join(datadir, 'tx_pos.csv')
        ris_pos_dir = os.path.join(datadir, 'ris_pos.csv')
        rx_pos_dir = os.path.join(datadir, 'rx_pos.csv')

        #修改--------------------------------------------------------------------------------
        self.gateway_pos_dir_ris = os.path.join(datadir, 'gateway_positionris.yml')
        self.gateway_pos_dir_rx = os.path.join(datadir, 'gateway_positionrx.yml')
        self.gateway_pos_dir_tx = os.path.join(datadir, 'gateway_positiontx.yml')

        self.rssi_dir = os.path.join(datadir, 'gateway_rssi.csv')
        self.rssi_dir_ris = os.path.join(datadir, 'gateway_rssi_ris.csv')
        self.dataset_index = np.loadtxt(indexdir, dtype=int)
        self.beta_res, self.alpha_res = 9, 36  # resulution of rays

        # load gateway position
        with open(os.path.join(self.gateway_pos_dir_ris)) as f:
            gateway_pos_dict1 = yaml.safe_load(f)
            self.gateway_pos1 = torch.tensor([pos for pos in gateway_pos_dict1.values()], dtype=torch.float32)
            self.gateway_pos1 = self.gateway_pos1 / scale_worldsize
            self.n_gateways_ris = len(self.gateway_pos1)

        with open(os.path.join(self.gateway_pos_dir_rx)) as f:
            gateway_pos_dict2 = yaml.safe_load(f)
            self.gateway_pos2 = torch.tensor([pos for pos in gateway_pos_dict2.values()], dtype=torch.float32)
            self.gateway_pos2 = self.gateway_pos2 / scale_worldsize

            self.n_gateways_rx = len(self.gateway_pos2)

        with open(os.path.join(self.gateway_pos_dir_tx)) as f:
            gateway_pos_dict3 = yaml.safe_load(f)
            self.gateway_pos3 = torch.tensor([pos for pos in gateway_pos_dict3.values()], dtype=torch.float32)
            self.gateway_pos3 = self.gateway_pos3 / scale_worldsize
            self.n_gateways_tx = len(self.gateway_pos3)

        # Load transmitter position
        self.tx_poses = torch.tensor(pd.read_csv(tx_pos_dir).values, dtype=torch.float32)
        self.tx_poses = self.tx_poses / scale_worldsize

        # Load report position
        self.rx_poses = torch.tensor(pd.read_csv(rx_pos_dir).values, dtype=torch.float32)
        self.rx_poses = self.rx_poses / scale_worldsize

        ##处理ris坐标--------------------------------------------------------------
        self.ris_poses = torch.tensor(pd.read_csv(ris_pos_dir).values, dtype=torch.float32)
        self.ris_poses = self.ris_poses / scale_worldsize


        # Load gateway received RSSI
        self.rssis = torch.tensor(pd.read_csv(self.rssi_dir).values, dtype=torch.float32)
        #self.rssis_ris = torch.tensor(pd.read_csv(self.rssi_dir_ris).values, dtype=torch.float32)

        self.nn_inputs,self.nn_labels = self.load_data()


    def load_data(self):

        nn_inputs = torch.tensor(np.zeros((len(self), 3+3+3)), dtype=torch.float32)

        nn_labels = torch.tensor(np.zeros((len(self), 1)), dtype=torch.float32)

        #ris信息-----------------------------------------------------------------
        gateways_ray_o1, gateways_rays_d1 = self.gen_rays_gateways_ris()

        # rx信息-----------------------------------------------------------------
        gateways_ray_o2, gateways_rays_d2 = self.gen_rays_gateways_rx()

        ## Load data
        data_counter = 0
        data_counter1 = 0
        for idx in tqdm(self.dataset_index, total=len(self.dataset_index)):
            rssis = self.rssis[idx]

            tx_pos = self.tx_poses[idx].view(-1)  # [3]

            ris_pos = self.ris_poses[idx].view(-1)  # [3]
            gateway_ray_o1 = gateways_ray_o1[idx].view(-1)  # [3]

            for i_gateway, rssi in enumerate(rssis):
                if rssi != -100:

                    rx_pos = self.rx_poses[i_gateway].view(-1)

                    nn_inputs[data_counter1] = torch.cat([tx_pos, ris_pos, rx_pos], dim=-1)
                    

                    nn_labels[data_counter1] = rssi
                    data_counter1 += 1



        nn_labels = rssi2amplitude(nn_labels)

        return nn_inputs, nn_labels

    def gen_rays_gateways_tx(self):
        """generate sample rays origin at gateways, for each gateways, we sample 36x9 rays

        Returns
        -------
        r_o : tensor. [n_gateways, 1, 3]. The origin of rays
        r_d : tensor. [n_gateways, n_rays, 3]. The direction of rays, unit vector
        """


        alphas = torch.linspace(0, 350, self.alpha_res) / 180 * np.pi
        betas = torch.linspace(10, 90, self.beta_res) / 180 * np.pi
        alphas = alphas.repeat(self.beta_res)    # [0,1,2,3,....]
        betas = betas.repeat_interleave(self.alpha_res)    # [0,0,0,0,...]

        radius = 1
        x = radius * torch.cos(alphas) * torch.cos(betas)  # (1*360)
        y = radius * torch.sin(alphas) * torch.cos(betas)
        z = radius * torch.sin(betas)

        r_d = torch.stack([x, y, z], axis=0).T  # [9*36, 3]

        #计算ris的w------------------------------------------------------------------
        r_d3 = r_d.expand([self.n_gateways_tx, self.beta_res * self.alpha_res, 3])  # [n_gateways, 9*36, 3]
        r_o3 = self.gateway_pos3.unsqueeze(1) # [21, 1, 3]
        r_o3, r_d3 = r_o3.contiguous(), r_d3.contiguous()
        print(r_d3,r_o3)

        return r_o3, r_d3



    def gen_rays_gateways_ris(self):
        """generate sample rays origin at gateways, for each gateways, we sample 36x9 rays

        Returns
        -------
        r_o : tensor. [n_gateways, 1, 3]. The origin of rays
        r_d : tensor. [n_gateways, n_rays, 3]. The direction of rays, unit vector
        """


        alphas = torch.linspace(0, 350, self.alpha_res) / 180 * np.pi
        betas = torch.linspace(10, 90, self.beta_res) / 180 * np.pi
        alphas = alphas.repeat(self.beta_res)    # [0,1,2,3,....]
        betas = betas.repeat_interleave(self.alpha_res)    # [0,0,0,0,...]

        radius = 1
        x = radius * torch.cos(alphas) * torch.cos(betas)  # (1*360)
        y = radius * torch.sin(alphas) * torch.cos(betas)
        z = radius * torch.sin(betas)

        r_d = torch.stack([x, y, z], axis=0).T  # [9*36, 3]

        #计算ris的w------------------------------------------------------------------
        r_d1 = r_d.expand([self.n_gateways_ris, self.beta_res * self.alpha_res, 3])  # [n_gateways, 9*36, 3]
        r_o1 = self.gateway_pos1.unsqueeze(1) # [21, 1, 3]
        r_o1, r_d1 = r_o1.contiguous(), r_d1.contiguous()
        print(r_d1,r_o1)
        return r_o1, r_d1

    def gen_rays_gateways_rx(self):
        """generate sample rays origin at gateways, for each gateways, we sample 36x9 rays

        Returns
        -------
        r_o : tensor. [n_gateways, 1, 3]. The origin of rays
        r_d : tensor. [n_gateways, n_rays, 3]. The direction of rays, unit vector
        """

        alphas = torch.linspace(0, 350, self.alpha_res) / 180 * np.pi
        betas = torch.linspace(10, 90, self.beta_res) / 180 * np.pi
        alphas = alphas.repeat(self.beta_res)  # [0,1,2,3,....]
        betas = betas.repeat_interleave(self.alpha_res)  # [0,0,0,0,...]

        radius = 1
        x = radius * torch.cos(alphas) * torch.cos(betas)  # (1*360)
        y = radius * torch.sin(alphas) * torch.cos(betas)
        z = radius * torch.sin(betas)

        r_d = torch.stack([x, y, z], axis=0).T  # [9*36, 3]



        # 计算rx的w------------------------------------------------------------------
        r_d2 = r_d.expand([self.n_gateways_rx, self.beta_res * self.alpha_res, 3])  # [n_gateways, 9*36, 3]
        r_o2 = self.gateway_pos2.unsqueeze(1)  # [21, 1, 3]
        r_o2, r_d2 = r_o2.contiguous(), r_d2.contiguous()
        #print(r_d2, r_o2)
        return r_o2, r_d2


    def __len__(self):
        rssis = self.rssis[self.dataset_index]
        return torch.sum(rssis != -100)

    def __getitem__(self, index):
        return self.nn_inputs[index], self.nn_labels[index]


dataset_dict = {"ble": BLE_dataset}
