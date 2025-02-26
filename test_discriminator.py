# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for Denoising Diffusion GAN. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------


import argparse
import torch
import numpy as np
from tqdm.auto import trange, tqdm

import os

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from datasets_prep.lsun import LSUN
from datasets_prep.stackmnist_data import StackedMNIST, _data_transforms_stacked_mnist
from datasets_prep.lmdb_datasets import LMDBDataset


from torch.multiprocessing import Process
import torch.distributed as dist
import shutil

from torch.utils.tensorboard import SummaryWriter

from sampler.utils import q_sample_pairs, sample_posterior, Diffusion_Coefficients, Posterior_Coefficients


def copy_source(file, output_dir):
    shutil.copyfile(file, os.path.join(output_dir, os.path.basename(file)))


def load_checkpoint(net, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    for key in list(ckpt.keys()):
        ckpt[key[7:]] = ckpt.pop(key)
    net.load_state_dict(ckpt)


# %%
def train(args, rank=0, gpu=0):
    from score_sde.models.discriminator import Discriminator_small, Discriminator_large
    from score_sde.models.ncsnpp_generator_adagn import NCSNpp
    from EMA import EMA

    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)
    device = torch.device("cuda:{}".format(gpu))

    batch_size = args.batch_size

    nz = args.nz  # latent dimension

    writer = SummaryWriter("logs")

    if args.dataset == "cifar10":
        dataset = CIFAR10(
            "./data",
            train=True,
            transform=transforms.Compose(
                [
                    transforms.Resize(32),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            ),
            download=True,
        )

    elif args.dataset == "stackmnist":
        train_transform, valid_transform = _data_transforms_stacked_mnist()
        dataset = StackedMNIST(
            root="./data", train=True, download=True, transform=train_transform
        )

    elif args.dataset == "lsun":

        train_transform = transforms.Compose(
            [
                transforms.Resize(args.image_size),
                transforms.CenterCrop(args.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        train_data = LSUN(
            root="/datasets/LSUN/",
            classes=["church_outdoor_train"],
            transform=train_transform,
        )
        subset = list(range(0, 120000))
        dataset = torch.utils.data.Subset(train_data, subset)

    elif args.dataset == "celeba_256":
        train_transform = transforms.Compose(
            [
                transforms.Resize(args.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        dataset = LMDBDataset(
            root="/datasets/celeba-lmdb/",
            name="celeba",
            train=True,
            transform=train_transform,
        )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    netG = NCSNpp(args).to(device)

    load_checkpoint(
        netG,
        "./saved_info/dd_gan/{}/{}/netG_1200.pth".format(
            args.dataset, args.exp
        ),
        device,
    )
    netG.eval()

    netD = Discriminator_small(
        nc=2 * args.num_channels,
        ngf=args.ngf,
        t_emb_dim=args.t_emb_dim,
        act=nn.LeakyReLU(0.2),
    ).to(device)
    netD.load_state_dict(torch.load("./saved_info/dd_gan/{}/{}/netD_{}.pth".format(args.dataset, args.exp, args.epoch)))
    netD.eval()
    print(f"Loaded discriminator trained within {args.epoch} epochs")

    coeff = Diffusion_Coefficients(args, device)
    pos_coeff = Posterior_Coefficients(args, device)

    metric_per_t = {t: {
        "neg_likelihood": 0,
        "num_samples": 0,
        "neg_likelihood_fake": 0,
    } for t in range(args.num_timesteps)}
    
    with torch.no_grad():
        for iteration, (x, y) in enumerate(tqdm(data_loader)):
            # sample from p(x_0)
            real_data = x.to(device, non_blocking=True)

            # sample t
            t = torch.randint(
                0, args.num_timesteps, (real_data.size(0),), device=device
            )

            x_t, x_tp1 = q_sample_pairs(coeff, real_data, t)

            # train with real
            D_real = netD(x_t, t, x_tp1.detach()).view(-1)
            errD_real = F.softplus(-D_real)

            # train with fake
            latent_z = torch.randn(batch_size, nz, device=device)

            x_0_predict = netG(x_tp1.detach(), t, latent_z)
            x_pos_sample = sample_posterior(pos_coeff, x_0_predict, x_tp1, t)

            D_fake = netD(x_pos_sample, t, x_tp1.detach()).view(-1)
            errD_fake = F.softplus(D_fake)


            for i, t_val in enumerate(t):
                dict = metric_per_t[t_val.item()]
                dict["neg_likelihood"] += errD_real[i] + errD_fake[i]
                dict["neg_likelihood_fake"] += errD_fake[i]
                dict["num_samples"] += 1
    
    for t in metric_per_t.keys():
        dict = metric_per_t[t_val.item()]
        dict["neg_likelihood"] += errD_real[i] + errD_fake[i]
        dict["neg_likelihood_fake"] += errD_fake[i]
        dict["num_samples"] += 1
        neg_likehood = 0.5 * dict["neg_likelihood"] / dict["num_samples"]
        neg_likehood_fake = dict["neg_likelihood_fake"] / dict["num_samples"]
        print(f"t={t} neg_likelihood={neg_likehood:.2f} neg_likelihood_fake={neg_likehood_fake:.2f}")


# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser("ddgan parameters")
    parser.add_argument(
        "--seed", type=int, default=1024, help="seed used for initialization"
    )

    parser.add_argument("--resume", action="store_true", default=False)

    parser.add_argument("--image_size", type=int, default=32, help="size of image")
    parser.add_argument("--num_channels", type=int, default=3, help="channel of image")
    parser.add_argument(
        "--centered", action="store_false", default=True, help="-1,1 scale"
    )
    parser.add_argument("--use_geometric", action="store_true", default=False)
    parser.add_argument(
        "--beta_min", type=float, default=0.1, help="beta_min for diffusion"
    )
    parser.add_argument(
        "--beta_max", type=float, default=20.0, help="beta_max for diffusion"
    )

    parser.add_argument(
        "--num_channels_dae",
        type=int,
        default=128,
        help="number of initial channels in denosing model",
    )
    parser.add_argument(
        "--n_mlp", type=int, default=3, help="number of mlp layers for z"
    )
    parser.add_argument("--ch_mult", nargs="+", type=int, help="channel multiplier")
    parser.add_argument(
        "--num_res_blocks",
        type=int,
        default=2,
        help="number of resnet blocks per scale",
    )
    parser.add_argument(
        "--attn_resolutions", default=(16,), help="resolution of applying attention"
    )
    parser.add_argument("--dropout", type=float, default=0.0, help="drop-out rate")
    parser.add_argument(
        "--resamp_with_conv",
        action="store_false",
        default=True,
        help="always up/down sampling with conv",
    )
    parser.add_argument(
        "--conditional", action="store_false", default=True, help="noise conditional"
    )
    parser.add_argument("--fir", action="store_false", default=True, help="FIR")
    parser.add_argument("--fir_kernel", default=[1, 3, 3, 1], help="FIR kernel")
    parser.add_argument(
        "--skip_rescale", action="store_false", default=True, help="skip rescale"
    )
    parser.add_argument(
        "--resblock_type",
        default="biggan",
        help="tyle of resnet block, choice in biggan and ddpm",
    )
    parser.add_argument(
        "--progressive",
        type=str,
        default="none",
        choices=["none", "output_skip", "residual"],
        help="progressive type for output",
    )
    parser.add_argument(
        "--progressive_input",
        type=str,
        default="residual",
        choices=["none", "input_skip", "residual"],
        help="progressive type for input",
    )
    parser.add_argument(
        "--progressive_combine",
        type=str,
        default="sum",
        choices=["sum", "cat"],
        help="progressive combine method.",
    )

    parser.add_argument(
        "--embedding_type",
        type=str,
        default="positional",
        choices=["positional", "fourier"],
        help="type of time embedding",
    )
    parser.add_argument(
        "--fourier_scale", type=float, default=16.0, help="scale of fourier transform"
    )
    parser.add_argument("--not_use_tanh", action="store_true", default=False)

    # geenrator and training
    parser.add_argument(
        "--exp", default="experiment_cifar_default", help="name of experiment"
    )
    parser.add_argument("--dataset", default="cifar10", help="name of dataset")
    parser.add_argument("--nz", type=int, default=100)
    parser.add_argument("--num_timesteps", type=int, default=4)

    parser.add_argument("--z_emb_dim", type=int, default=256)
    parser.add_argument("--t_emb_dim", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=128, help="input batch size")
    parser.add_argument("--epoch", type=int, default=1200)
    parser.add_argument("--ngf", type=int, default=64)

    parser.add_argument("--lr_g", type=float, default=1.5e-4, help="learning rate g")
    parser.add_argument("--lr_d", type=float, default=1e-4, help="learning rate d")
    parser.add_argument("--beta1", type=float, default=0.5, help="beta1 for adam")
    parser.add_argument("--beta2", type=float, default=0.9, help="beta2 for adam")
    parser.add_argument("--no_lr_decay", action="store_true", default=False)

    parser.add_argument("--r1_gamma", type=float, default=0.05, help="coef for r1 reg")
    parser.add_argument(
        "--lazy_reg", type=int, default=None, help="lazy regulariation."
    )

    args = parser.parse_args()
    train(args)