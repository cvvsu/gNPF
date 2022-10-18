import torch 
from torch import nn 
from torch.nn import functional as F


class Reshape(nn.Module):
    def forward(self, z):
        return torch.reshape(z, (z.shape[0], 1, 9, 3))


class GNPF(nn.Module):
    """Generate NPF datasets through a Gaussian distribution.
    nn.ConvTranspose2d will introduce the checkboard effect: 
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/issues/78
    """
    def __init__(self, dim=100, ngf=64):
        super().__init__()

        # [N, 100, 1, 1] -> [N, 512, 4, 4]
        model = [
            nn.Upsample(scale_factor=4),
            nn.Conv2d(dim, ngf*8, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(ngf*8),
            nn.ReLU(),
        ]

        # [N, 512, 4, 4] -> [N, 256, 9, 3]
        model += [
            nn.Upsample(scale_factor=(2, 1)),  # (8, 4)
            nn.Conv2d(ngf*8, ngf*4, kernel_size=4, stride=1, padding=(2, 1), bias=False),
            nn.InstanceNorm2d(ngf*4),
            nn.ReLU(),
        ]


        # [N, 256, 9, 3] -> [N, 128, 18, 6]
        model += [
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ngf*4, ngf*2, 3, 1, 1, bias=False),
            # nn.ConvTranspose2d(16, 8, 4, 2, 1),
            nn.InstanceNorm2d(ngf*2),
            nn.ReLU(),
        ]
        
        # [N, 128, 18, 6] -> [N, 64, 36, 13]
        model += [
            nn.Upsample(scale_factor=2),  # (36, 12)
            nn.Conv2d(ngf*2, ngf, (3, 2), 1, 1, bias=False),
            # nn.ConvTranspose2d(8, 4, 4, 2, 1, output_padding=(0, 1)),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(),            
        ]

        # [N, 64, 36, 13] -> [N, 32, 72, 26]
        model += [
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ngf, ngf//2, 3, 1, 1, bias=False),
            # nn.ConvTranspose2d(4, 2, 4, 2, 1),
            nn.InstanceNorm2d(ngf//2),
            nn.ReLU(),
        ]

        # [N, 32, 72, 26] -> [N, 1, 144, 52]
        model += [
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ngf//2, 1, 3, 1, 1, bias=False),
            # nn.ConvTranspose2d(2, 1, 4, 2, 1),
            # nn.ReLU(),
            # nn.Tanh()  # output in [-1, 1]
            nn.Sigmoid()
        ]

        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        return self.model(x)

    
class DNPF(nn.Module):
    def __init__(self, ndf=64):
        super().__init__()
        model = [
            nn.Conv2d(1, ndf, 4, 2, 1, bias=False),  # (72, 26)
            nn.LeakyReLU(0.2, True),
        ]

        ndf_pre = ndf
        for i in range(3):
            ndf = ndf_pre 
            model += [
                nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),  # (36, 13) -> (18, 6) -> (9, 3)
                nn.InstanceNorm2d(ndf),
                nn.LeakyReLU(0.2, True),
            ]
            ndf_pre = ndf * 2

        model += [
            nn.Conv2d(ndf_pre, 1, 4, (2, 1), (1, 2), bias=False)
        ]       

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x) 


class GANLoss(nn.Module):
    """
    Reference: CycleGAN code.
    """
    def __init__(self):
        super().__init__()
        self.register_buffer('real_label', torch.tensor(1.0))
        self.register_buffer('fake_label', torch.tensor(0.0))
        self.criterion = nn.MSELoss()
    
    def get_target_tensor(self, pred, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(pred)
    
    def __call__(self, pred, target_is_real):
        target_tensor = self.get_target_tensor(pred, target_is_real)
        return self.criterion(pred, target_tensor)


def get_gradient_penalty(critic, real, fake, device="cpu"):
    """
    https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/GANs/4.%20WGAN-GP/utils.py
    """
    BATCH_SIZE, C, H, W = real.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * alpha + fake * (1 - alpha)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty


def init_weights(net):        
    classname = net.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(net.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(net.weight.data, 1.0, 0.02)
        nn.init.constant_(net.bias.data, 0)


def get_networks(net_name, dim, nf, device):

    if net_name == 'G':
        net = GNPF(dim, nf)
        net.to(device)
        net.apply(init_weights)
        print(f'Parameters have been normalized for Generator.')
        return net
    elif net_name == 'D':
        net = DNPF(nf)
        net.to(device)    
        net.apply(init_weights)
        print(f'Parameters have been normalized for Discriminator.')
        return net
    else:
        raise ValueError('Unknown network name {net_name}.')


def get_num_parameters(net):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


# test the shapes
if __name__=='__main__':
    
    x = torch.randn(64, 100, 1, 1)
    G = GNPF()
    print(f'G parameters: {get_num_parameters(G)}')
    out_g = G(x)
    print(out_g.shape)
    
    D = DNPF()
    print(f'D parameters: {get_num_parameters(D)}')
    out_d = D(out_g)
    print(out_d.shape)

    # criterion = GANLoss()
    # loss = criterion(out_d, True)
    # print(out_g.shape, out_d.shape, loss)