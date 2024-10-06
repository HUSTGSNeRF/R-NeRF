import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import argparse
from shutil import copyfile

import numpy as np
import torch
import torch.optim as optim
import yaml
from skimage.metrics import structural_similarity as ssim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
import scipy.io as scio

from dataloader import *
from model import *
from renderer import renderer_dict
from utils.data_painter import paint_spectrum_compare
from utils.logger import logger_config


class RNeRF_Runner():

    def __init__(self, mode, dataset_type, **kwargs) -> None:

        kwargs_path = kwargs['path']
        kwargs_render = kwargs['render']
        kwargs_network = kwargs['networks']
        kwargs_train = kwargs['train']
        self.dataset_type = dataset_type

        ## Path settings
        self.expname = kwargs_path['expname']
        self.datadir = kwargs_path['datadir']
        self.logdir = kwargs_path['logdir']
        self.devices = torch.device('cuda')

        ## Logger
        log_filename = "logger.log"
        log_savepath = os.path.join(self.logdir, self.expname, log_filename)
        self.logger = logger_config(log_savepath=log_savepath, logging_name='nerf2')
        self.logger.info("expname:%s, datadir:%s, logdir:%s", self.expname, self.datadir, self.logdir)
        self.writer = SummaryWriter(os.path.join(self.logdir, self.expname, 'tensorboard'))


        ## Networks
        self.nerf2_network = RNeRF(**kwargs_network).to(self.devices)
        params = list(self.nerf2_network.parameters())
        self.optimizer = torch.optim.Adam(params, lr=float(kwargs_train['lr']),
                                          weight_decay=float(kwargs_train['weight_decay']),
                                          betas=(0.9, 0.999))
        self.cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer,
                                                                        T_max=float(kwargs_train['T_max']), eta_min=float(kwargs_train['eta_min']),
                                                                        last_epoch=-1)

        ## Renderer
        renderer = renderer_dict[kwargs_render['mode']]
        self.renderer = renderer(networks_fn=self.nerf2_network, **kwargs_render)
        self.scale_worldsize = kwargs_render['scale_worldsize']

        ## Print total number of parameters
        total_params = sum(p.numel() for p in params if p.requires_grad)
        self.logger.info("Total number of parameters: %s", total_params)

        ## Train Settings
        self.current_iteration = 1
        if kwargs_train['load_ckpt'] or mode == 'test':
            self.load_checkpoints()
        self.batch_size = kwargs_train['batch_size']
        self.total_iterations = kwargs_train['total_iterations']
        self.save_freq = kwargs_train['save_freq']

        ## Dataset
        dataset = dataset_dict[dataset_type]
        train_index = os.path.join(self.datadir, "train_index.txt")
        test_index = os.path.join(self.datadir, "test_index.txt")
        if not os.path.exists(train_index) or not os.path.exists(test_index):
            split_dataset(self.datadir, ratio=0.9, dataset_type=dataset_type)
        self.logger.info("Loading training set...")
        self.train_set = dataset(self.datadir, train_index, self.scale_worldsize)
        self.logger.info("Loading test set...")
        self.test_set = dataset(self.datadir, test_index, self.scale_worldsize)

        self.train_iter = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=0)
        self.test_iter = DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=0)
        self.logger.info("Train set size:%d, Test set size:%d", len(self.train_set), len(self.test_set))


    def load_checkpoints(self):
        ckptsdir = os.path.join(self.logdir, self.expname, 'ckpts')
        if not os.path.exists(ckptsdir):
            os.makedirs(ckptsdir)
        ckpts = [os.path.join(ckptsdir, f) for f in sorted(os.listdir(ckptsdir)) if 'tar' in f]
        self.logger.info('Found ckpts %s', ckpts)

        if len(ckpts) > 0:
            ckpt_path = ckpts[-1]
            self.logger.info('Loading ckpt %s', ckpt_path)
            ckpt = torch.load(ckpt_path, map_location=self.devices)

            self.nerf2_network.load_state_dict(ckpt['nerf2_network_state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            self.cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer,T_max=20,eta_min=1e-5)
            self.cosine_scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            self.current_iteration = ckpt['current_iteration']



    def save_checkpoint(self):
        ckptsdir = os.path.join(self.logdir, self.expname, 'ckpts')
        model_lst = [x for x in sorted(os.listdir(ckptsdir)) if x.endswith('.tar')]
        if len(model_lst) > 2:
            os.remove(ckptsdir + '/%s' % model_lst[0])

        ckptname = os.path.join(ckptsdir, '{:06d}.tar'.format(self.current_iteration))
        torch.save({
            'current_iteration': self.current_iteration,
            'nerf2_network_state_dict': self.nerf2_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.cosine_scheduler.state_dict()
        }, ckptname)
        return ckptname



    def train(self):
        """train the model
        """
        self.logger.info("Start training. Current Iteration:%d", self.current_iteration)
        while self.current_iteration <= self.total_iterations:
            with tqdm(total=len(self.train_iter), desc=f"Iteration {self.current_iteration}/{self.total_iterations}") as pbar:
                for train_input, train_label in self.train_iter:
                    if self.current_iteration > self.total_iterations:
                        break

                    train_input, train_label = train_input.to(self.devices), train_label.to(self.devices)
                    if self.dataset_type == "ble":
                        tx_o, rays_o, rays_d, ris_o, rays_o1, rays_d1= train_input[:, :3], train_input[:, 3:6], train_input[:, 6:978],  train_input[:, 978:981], train_input[:, 981:984], train_input[:, 984:1956]
                        
                        predict_rssi = self.renderer.render_rssi(tx_o, rays_o, rays_d, ris_o, rays_o1, rays_d1)

                        loss = sig2mse(predict_rssi, train_label.view(-1))
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    self.cosine_scheduler.step()
                    self.current_iteration += 1

                    self.writer.add_scalar('Loss/loss', loss, self.current_iteration)
                    pbar.update(1)
                    pbar.set_description(f"Iteration {self.current_iteration}/{self.total_iterations}")
                    pbar.set_postfix_str('loss = {:.6f}, lr = {:.6f}'.format(loss.item(), self.optimizer.param_groups[0]['lr']))

                    if self.current_iteration % self.save_freq == 0:
                        ckptname = self.save_checkpoint()
                        pbar.write('Saved checkpoints at {}'.format(ckptname))

    def eval_network_rssi(self):
        """test the model and save predicted RSSI values to a file
        """
        self.logger.info("Start evaluation")
        self.nerf2_network.eval()

        with torch.no_grad():
            with open(os.path.join(self.logdir, self.expname, "result.txt"), 'w') as f:
                for test_input, test_label in self.test_iter:
                    test_input, test_label = test_input.to(self.devices), test_label.to(self.devices) 

                    tx_o, rays_o, rays_d, ris_o, rays_o1, rays_d1= test_input[:, :3], test_input[:, 3:6], test_input[:, 6:978],  test_input[:, 978:981], test_input[:, 981:984], test_input[:, 984:1956]
                    predict_rssi = self.renderer.render_rssi(tx_o, rays_o, rays_d, ris_o, rays_o1, rays_d1)

                    ## save predicted spectrum
                    predict_rssi = amplitude2rssi(predict_rssi.detach().cpu())
                    gt_rssi = amplitude2rssi(test_label.detach().cpu())
                    error = abs(predict_rssi - gt_rssi.reshape(-1))

                    for i, rssi in enumerate(predict_rssi):
                        f.write("{:.2f}, {:.2f}".format(gt_rssi[i].item(), rssi.item()) + '\n')

        result = np.loadtxt(os.path.join(self.logdir,self.expname, "result.txt"), delimiter=",")
        self.logger.info("Total Median error:%.2f", np.median(abs(result[:,0] - result[:,1])))
        self.logger.info("Total MAE error:%.2f", np.mean(abs(result[:,0] - result[:,1])))       

        gt_rssi1 = result[:, 0]  # 真实的RSSI值
        predict_rssi1 = result[:, 1]  # 预测的RSSI值

        # 计算MSE
        mse = np.mean((gt_rssi1 - predict_rssi1) ** 2)

        # 记录MSE到日志
        self.logger.info("Total MSE error:%.2f", mse)
        rmse = np.sqrt(mse)
        self.logger.info("Total RMSE error:%.2f", rmse)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/mimo-csi.yml', help='config file path')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--dataset_type', type=str, default='mimo')
    args = parser.parse_args()
    print(args)
    torch.cuda.set_device(args.gpu)

    with open(args.config) as f:
        kwargs = yaml.safe_load(f)
        f.close()

    ## backup config file
    if args.mode == 'train':
        logdir = os.path.join(kwargs['path']['logdir'], kwargs['path']['expname'])
        os.makedirs(logdir, exist_ok=True)
        copyfile(args.config, os.path.join(logdir,'config.yml'))

    worker = RNeRF_Runner(mode=args.mode, dataset_type=args.dataset_type, **kwargs)
    if args.mode == 'train':
        worker.train()
    elif args.mode == 'test':
        if args.dataset_type == 'ble':
            worker.eval_network_rssi()
