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
from sampler.base import ddpm_sample

def copy_source(file, output_dir):
    shutil.copyfile(file, os.path.join(output_dir, os.path.basename(file)))


def broadcast_params(params):
    for param in params:
        dist.broadcast(param.data, src=0)


# %%
def train(rank, gpu, args):
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

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=args.world_size, rank=rank
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
    )

    netG = NCSNpp(args).to(device)

    if args.dataset == "cifar10" or args.dataset == "stackmnist":
        netD = Discriminator_small(
            nc=2 * args.num_channels,
            ngf=args.ngf,
            t_emb_dim=args.t_emb_dim,
            act=nn.LeakyReLU(0.2),
        ).to(device)
    else:
        netD = Discriminator_large(
            nc=2 * args.num_channels,
            ngf=args.ngf,
            t_emb_dim=args.t_emb_dim,
            act=nn.LeakyReLU(0.2),
        ).to(device)

    broadcast_params(netG.parameters())
    broadcast_params(netD.parameters())

    optimizerD = optim.Adam(
        netD.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2)
    )

    optimizerG = optim.Adam(
        netG.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2)
    )

    if args.use_ema:
        optimizerG = EMA(optimizerG, ema_decay=args.ema_decay)

    schedulerG = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizerG, args.num_epoch, eta_min=1e-5
    )
    schedulerD = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizerD, args.num_epoch, eta_min=1e-5
    )

    # ddp
    netG = nn.parallel.DistributedDataParallel(netG, device_ids=[gpu])
    netD = nn.parallel.DistributedDataParallel(netD, device_ids=[gpu])

    exp = args.exp
    parent_dir = "./saved_info/dd_gan/{}".format(args.dataset)

    exp_path = os.path.join(parent_dir, exp)
    if rank == 0:
        if not os.path.exists(exp_path):
            os.makedirs(exp_path)
            copy_source(__file__, exp_path)
            shutil.copytree(
                "score_sde/models", os.path.join(exp_path, "score_sde/models")
            )

    coeff = Diffusion_Coefficients(args, device)
    pos_coeff = Posterior_Coefficients(args, device)

    if args.resume:
        checkpoint_file = os.path.join(exp_path, "content.pth")
        checkpoint = torch.load(checkpoint_file, map_location=device)
        init_epoch = checkpoint["epoch"]
        epoch = init_epoch
        netG.load_state_dict(checkpoint["netG_dict"])
        # load G

        optimizerG.load_state_dict(checkpoint["optimizerG"])
        schedulerG.load_state_dict(checkpoint["schedulerG"])
        # load D
        netD.load_state_dict(checkpoint["netD_dict"])
        optimizerD.load_state_dict(checkpoint["optimizerD"])
        schedulerD.load_state_dict(checkpoint["schedulerD"])
        global_step = checkpoint["global_step"]
        print("=> loaded checkpoint (epoch {})".format(checkpoint["epoch"]))
    else:
        global_step, epoch, init_epoch = 0, 0, 0

    for epoch in range(init_epoch, args.num_epoch + 1):
        train_sampler.set_epoch(epoch)

        for iteration, (x, y) in enumerate(tqdm(data_loader)):
            for p in netD.parameters():
                p.requires_grad = True

            netD.zero_grad()

            # sample from p(x_0)
            real_data = x.to(device, non_blocking=True)

            # sample t
            t = torch.randint(
                0, args.num_timesteps, (real_data.size(0),), device=device
            )

            x_t, x_tp1 = q_sample_pairs(coeff, real_data, t)
            x_t.requires_grad = True

            # train with real
            D_real = netD(x_t, t, x_tp1.detach()).view(-1)

            errD_real = F.softplus(-D_real)
            errD_real = errD_real.mean()

            errD_real.backward(retain_graph=True)

            if args.lazy_reg is None:
                grad_real = torch.autograd.grad(
                    outputs=D_real.sum(), inputs=x_t, create_graph=True
                )[0]
                grad_penalty = (
                    grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2
                ).mean()

                grad_penalty = args.r1_gamma / 2 * grad_penalty
                grad_penalty.backward()
            else:
                if global_step % args.lazy_reg == 0:
                    grad_real = torch.autograd.grad(
                        outputs=D_real.sum(), inputs=x_t, create_graph=True
                    )[0]
                    grad_penalty = (
                        grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2
                    ).mean()

                    grad_penalty = args.r1_gamma / 2 * grad_penalty
                    grad_penalty.backward()

            # train with fake
            latent_z = torch.randn(batch_size, nz, device=device)

            x_0_predict = netG(x_tp1.detach(), t, latent_z)
            x_pos_sample = sample_posterior(pos_coeff, x_0_predict, x_tp1, t)

            output = netD(x_pos_sample, t, x_tp1.detach()).view(-1)

            errD_fake = F.softplus(output)
            errD_fake = errD_fake.mean()
            errD_fake.backward()

            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            # update G
            for p in netD.parameters():
                p.requires_grad = False
            netG.zero_grad()

            t = torch.randint(
                0, args.num_timesteps, (real_data.size(0),), device=device
            )

            x_t, x_tp1 = q_sample_pairs(coeff, real_data, t)

            latent_z = torch.randn(batch_size, nz, device=device)

            x_0_predict = netG(x_tp1.detach(), t, latent_z)
            x_pos_sample = sample_posterior(pos_coeff, x_0_predict, x_tp1, t)

            output = netD(x_pos_sample, t, x_tp1.detach()).view(-1)

            errG = F.softplus(-output)
            errG = errG.mean()

            errG.backward()
            optimizerG.step()

            global_step += 1
            if iteration % 100 == 0:
                if rank == 0:
                    print(
                        "epoch {} iteration{}, G Loss: {}, D Loss: {}".format(
                            epoch, iteration, errG.item(), errD.item()
                        )
                    )
                    writer.add_scalar("Loss/G", errG.item(), iteration)
                    writer.add_scalar("Loss/D", errD.item(), iteration)

        if not args.no_lr_decay:

            schedulerG.step()
            schedulerD.step()

        if rank == 0:
            if epoch % 10 == 0:
                torchvision.utils.save_image(
                    x_pos_sample,
                    os.path.join(exp_path, "xpos_epoch_{}.png".format(epoch)),
                    normalize=True,
                )

            x_t_1 = torch.randn_like(real_data)
            fake_sample = ddpm_sample(
                pos_coeff, netG, args.num_timesteps, x_t_1, T, args
            )
            torchvision.utils.save_image(
                fake_sample,
                os.path.join(exp_path, "sample_discrete_epoch_{}.png".format(epoch)),
                normalize=True,
            )

            if args.save_content:
                if epoch % args.save_content_every == 0:
                    print("Saving content.")
                    content = {
                        "epoch": epoch + 1,
                        "global_step": global_step,
                        "args": args,
                        "netG_dict": netG.state_dict(),
                        "optimizerG": optimizerG.state_dict(),
                        "schedulerG": schedulerG.state_dict(),
                        "netD_dict": netD.state_dict(),
                        "optimizerD": optimizerD.state_dict(),
                        "schedulerD": schedulerD.state_dict(),
                    }

                    torch.save(content, os.path.join(exp_path, "content.pth"))

            if epoch % args.save_ckpt_every == 0:
                if args.use_ema:
                    optimizerG.swap_parameters_with_ema(store_params_in_ema=True)

                torch.save(
                    netG.state_dict(),
                    os.path.join(exp_path, "netG_{}.pth".format(epoch)),
                )
                torch.save(
                    netD.state_dict(),
                    os.path.join(exp_path, "netD_{}.pth".format(epoch)),
                )
                if args.use_ema:
                    optimizerG.swap_parameters_with_ema(store_params_in_ema=True)


def init_processes(rank, size, fn, args):
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = args.master_address
    os.environ["MASTER_PORT"] = "6020"
    torch.cuda.set_device(args.local_rank)
    gpu = args.local_rank
    dist.init_process_group(
        backend="nccl", init_method="env://", rank=rank, world_size=size
    )
    fn(rank, gpu, args)
    dist.barrier()
    cleanup()


def cleanup():
    dist.destroy_process_group()


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
    parser.add_argument("--num_epoch", type=int, default=1200)
    parser.add_argument("--ngf", type=int, default=64)

    parser.add_argument("--lr_g", type=float, default=1.5e-4, help="learning rate g")
    parser.add_argument("--lr_d", type=float, default=1e-4, help="learning rate d")
    parser.add_argument("--beta1", type=float, default=0.5, help="beta1 for adam")
    parser.add_argument("--beta2", type=float, default=0.9, help="beta2 for adam")
    parser.add_argument("--no_lr_decay", action="store_true", default=False)

    parser.add_argument(
        "--use_ema", action="store_true", default=False, help="use EMA or not"
    )
    parser.add_argument(
        "--ema_decay", type=float, default=0.9999, help="decay rate for EMA"
    )

    parser.add_argument("--r1_gamma", type=float, default=0.05, help="coef for r1 reg")
    parser.add_argument(
        "--lazy_reg", type=int, default=None, help="lazy regulariation."
    )

    parser.add_argument("--save_content", action="store_true", default=False)
    parser.add_argument(
        "--save_content_every",
        type=int,
        default=50,
        help="save content for resuming every x epochs",
    )
    parser.add_argument(
        "--save_ckpt_every", type=int, default=25, help="save ckpt every x epochs"
    )

    ###ddp
    parser.add_argument(
        "--num_proc_node",
        type=int,
        default=1,
        help="The number of nodes in multi node env.",
    )
    parser.add_argument(
        "--num_process_per_node", type=int, default=1, help="number of gpus"
    )
    parser.add_argument("--node_rank", type=int, default=0, help="The index of node.")
    parser.add_argument(
        "--local_rank", type=int, default=0, help="rank of process in the node"
    )
    parser.add_argument(
        "--master_address", type=str, default="127.0.0.1", help="address for master"
    )

    args = parser.parse_args()
    print(args)
    args.world_size = args.num_proc_node * args.num_process_per_node
    size = args.num_process_per_node

    if size > 1:
        processes = []
        for rank in range(size):
            args.local_rank = rank
            global_rank = rank + args.node_rank * args.num_process_per_node
            global_size = args.num_proc_node * args.num_process_per_node
            args.global_rank = global_rank
            print(
                "Node rank %d, local proc %d, global proc %d"
                % (args.node_rank, rank, global_rank)
            )
            p = Process(
                target=init_processes, args=(global_rank, global_size, train, args)
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        print("starting in debug mode")

        init_processes(0, size, train, args)
