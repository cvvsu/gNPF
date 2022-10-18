import os, time
import numpy as np 
from datetime import datetime

import torch 
from torch import nn 

from network import GANLoss, get_networks, get_gradient_penalty
from utils import mkdirs


class NPFNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.isTrain = not args.isTest
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.savedir = os.path.join(args.ckptdir, args.exp_name)
        os.makedirs(self.savedir, exist_ok=True)
        self.G = get_networks('G', args.dim, args.nf, self.device)
       
        if self.isTrain:
            self.D = get_networks('D', args.dim, args.nf, self.device)            
            self.criterion = GANLoss().to(self.device)
            # self.criterionL1 = nn.L1Loss()

            # wgan-gp can use Adam, while wgan we should use RMSprop or SGD
            self.optimizer_G = torch.optim.Adam(self.G.parameters(), lr=self.args.lr, betas=(args.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.D.parameters(), lr=self.args.lr, betas=(args.beta1, 0.999))
            # self.optimizer_G = torch.optim.RMSprop(self.G.parameters(), lr=self.args.lr)
            # self.optimizer_D = torch.optim.RMSprop(self.D.parameters(), lr=self.args.lr)

            self.scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_G, T_max=self.epochs, eta_min=0)
            self.scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_D, T_max=self.epochs, eta_min=0)
           
            # self.set_random_state(args.seed)

            self.fix_z = torch.randn(8, args.dim, 1, 1, device=self.device, requires_grad=False)
            torch.save(self.fix_z, os.path.join(self.savedir, 'fix_z.pt'))

    # def set_random_state(self, seed):
    #     np.random.seed(seed)
    #     torch.manual_seed(seed)
    #     if torch.cuda.is_available():
    #         torch.cuda.manual_seed(seed)
    #         torch.cuda.manual_seed_all(seed)
    #     torch.backends.cudnn.deterministic = True
    #     torch.backends.cudnn.benchmark = False

    def set_input(self, input):
        self.real = input.to(self.device)
        self.z = torch.randn(self.args.batch_size, self.args.dim, 1, 1, device=self.device)

    def forward(self):        
        self.fake = self.G(self.z)
    
    def backward_D(self):
        pred_real = self.D(self.real)
        loss_D_real = self.criterion(pred_real, True)

        pred_fake = self.D(self.fake.detach())
        loss_D_fake = self.criterion(pred_fake, False)

        self.loss_D = (loss_D_real + loss_D_fake) * 0.5
        self.loss_D.backward()
        
    def backward_G(self):
        pred = self.D(self.fake)
        self.loss_G = self.criterion(pred, True)
        self.loss_G.backward()

    def set_requires_grad(self, nets, requires_grad=False):
        for net in nets:
            for param in net.parameters():
                param.requires_grad=requires_grad

    def optimize_parameters(self):
        # update D more
        # for _ in range(3):
        self.set_requires_grad([self.D], True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        # update G
        self.set_requires_grad([self.D], False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

        return self.loss_G, self.loss_D

    def train(self, loader):
        for epoch in range(1, self.args.epochs+1):
            tic_epoch = time.time()
            Gloss, Dloss = 0.0, 0.0
            for idx, input in enumerate(loader):
                if self.args.gan_type == 'wgan-gp':
                    input = input.to(self.device)
                    # update D
                    for _ in range(5):
                        noise = torch.randn(self.args.batch_size, self.args.dim, 1, 1, device=self.device)
                        fake = self.G(noise)
                        D_real = self.D(input).reshape(-1)
                        D_fake = self.D(fake).reshape(-1)
                        gp = get_gradient_penalty(self.D, input, fake, device=self.device)
                        loss_D = (-(torch.mean(D_real) - torch.mean(D_fake)) + self.args.lambda_gp * gp)
                        self.D.zero_grad()
                        loss_D.backward(retain_graph=True)
                        self.optimizer_D.step()
                        Dloss += loss_D.item()
                    
                    # update G
                    g_fake = self.D(fake).reshape(-1)
                    loss_G = - torch.mean(g_fake)
                    self.G.zero_grad()
                    loss_G.backward()
                    self.optimizer_G.step()
                    Gloss += loss_G.item()

                else:
                    self.set_input(input)
                    self.forward()
                    loss_G, loss_D = self.optimize_parameters()   
                    Gloss += loss_G.item()
                    Dloss += loss_D.item()

            Gloss, Dloss = Gloss / (idx + 1), Dloss / (idx + 1)        
            self.save_networks(epoch) 
            self.save_image(epoch)
            old_lr, lr = self.update_lr()
            print(f"[{datetime.now(). strftime('%Y-%m-%d %H:%M:%S')}]",
                        f"[{epoch}/{self.args.epochs}: {time.time()-tic_epoch:.3f} s]",
                        f"[loss_G: {Gloss:.6f} | loss_D: {Dloss:.6f}], [lr: {old_lr:.6f}->{lr:.6f}]")

    def update_lr(self):
        old_lr = self.optimizer_G.param_groups[0]['lr']
        for scheduler in [self.scheduler_G, self.scheduler_D]:
            scheduler.step()
        lr = self.optimizer_G.param_groups[0]['lr']
        return old_lr, lr

    def save_networks(self, epoch):
        torch.save(self.G.state_dict(), os.path.join(self.savedir, f'G_{epoch}.pth'))
        torch.save(self.D.state_dict(), os.path.join(self.savedir, f'D_{epoch}.pth'))
        # self.G.to(self.device)
        # self.D.to(self.device)
    
    def save_image(self, epoch):
        fake = self.G(self.fix_z).detach().cpu().squeeze().numpy()
        np.save(os.path.join(self.savedir, f'{epoch}.npy'), fake)


def build_model(args):
    model = NPFNet(args)
    return model
