import argparse

def get_message(parser, args):
    r"""
    References:
        1.https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/options/base_options.py
    """
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(args).items()):
        comment = ''
        default = parser.get_default(k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    return message


def get_args():
    parser = argparse.ArgumentParser('gNPF')

    # basic
    parser.add_argument('--ckptdir', default='checkpoints', type=str, help='folder stores the saved models and results')
    parser.add_argument('--exp_name', type=str, default='experiment', help='name for current experiment')
    parser.add_argument('--seed', type=int, default=233, help='random seed for re-producing')

    # datasets
    parser.add_argument('--dataroot', type=str, default='datasets', help='folder that stores the datasets')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers')
    parser.add_argument('--batch_size', type=int, default=32, help='mini batch size')
    parser.add_argument('--vmax', type=float, default=1e4, help='maximum value for normalization')
    parser.add_argument('--station', type=str, default='hyy', help='station [hyy | kum | var | all]')

    # training
    parser.add_argument('--dim', type=int, default=100, help='hidden dim')
    parser.add_argument('--nf', type=int, default=64, help='number of filters')
    parser.add_argument('--epochs', type=int, default=20, help='training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam')
    parser.add_argument('--gan_type', type=str, default='wgan-gp', help='wgan-gp can alleviate the mode collapse problem')
    parser.add_argument('--lambda_gp', type=float, default=10, help='lambda for gp')    

    # test
    parser.add_argument('--isTest', action= 'store_true', help='test phase if true')

    args = parser.parse_args()
    return args, get_message(parser, args)