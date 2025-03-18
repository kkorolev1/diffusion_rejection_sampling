# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for Denoising Diffusion GAN. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------
import argparse
import torch
import torch.nn as nn
import numpy as np
import pickle
from tqdm import trange

import os

import torchvision
from score_sde.models.ncsnpp_generator_adagn import NCSNpp
from score_sde.models.discriminator import Discriminator_small
from pytorch_fid.fid_score import calculate_fid_given_paths

from sampler.utils import Posterior_Coefficients
from sampler.base import ddpm_sample 
from sampler.rejection_sampling import rejection_sample, rejection_sample_reinit, RejectionSamplingProcessor


def load_checkpoint(net, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    for key in list(ckpt.keys()):
        ckpt[key[7:]] = ckpt.pop(key)
    net.load_state_dict(ckpt)


# %%
def sample_and_test(args):
    torch.manual_seed(42)
    device = "cuda:0"

    if args.dataset == "cifar10":
        real_img_dir = "pytorch_fid/cifar10_train_stat.npy"
    elif args.dataset == "celeba_256":
        real_img_dir = "pytorch_fid/celeba_256_stat.npy"
    elif args.dataset == "lsun":
        real_img_dir = "pytorch_fid/lsun_church_stat.npy"
    else:
        real_img_dir = args.real_img_dir

    to_range_0_1 = lambda x: (x + 1.0) / 2.0

    netG = NCSNpp(args).to(device)

    # loading weights from ddp in single gpu
    load_checkpoint(
        netG,
        "./saved_info/dd_gan/{}/{}/netG_{}.pth".format(
            args.dataset, "ddgan_cifar10_exp1", args.g_epoch
        ),
        device,
    )
    netG.eval()
    print(f"Loading Generator from {args.g_epoch} epoch")

    if args.use_rejection_sampling:
        netD = Discriminator_small(
            nc=2 * args.num_channels,
            ngf=args.ngf,
            t_emb_dim=args.t_emb_dim,
            act=nn.LeakyReLU(0.2),
        ).to(device)
        netD.load_state_dict(torch.load("./saved_info/dd_gan/{}/{}/netD_{}.pth".format(args.dataset, "ddgan_cifar10_exp1", args.d_epoch)))
        netD.eval()
        print(f"Loading Discriminator from {args.d_epoch} epoch")
        print(f"Rejection Sampling is enabled: {args.enable_reinit=}, {args.constant_quantile=}, {args.reject_full_trajectory=}, {args.reinit_steps=}")

    pos_coeff = Posterior_Coefficients(args, device)
    iters_needed = args.num_samples // args.batch_size

    save_dir = "./generated_samples/{}".format(args.exp)
    # os.rmdir(save_dir)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if args.use_rejection_sampling:
        with open("log_ratio_dict.pickle", "rb") as f:
            log_ratio_dict = pickle.load(f)
        rs_processor = RejectionSamplingProcessor(netD, log_ratio_dict, device, constant_quantile=args.constant_quantile)

    if args.compute_fid:
        print(f"Images are {args.precomputed=}")
        if not args.precomputed:
            for i in trange(iters_needed):
                with torch.no_grad():
                    x_t_1 = torch.randn(
                        args.batch_size, args.num_channels, args.image_size, args.image_size
                    ).to(device)
                    if args.use_rejection_sampling:
                        fn_sample = rejection_sample_reinit if args.enable_reinit else rejection_sample
                        fake_sample = fn_sample(
                            pos_coeff,
                            netG,
                            rs_processor,
                            args.num_timesteps,
                            x_t_1,
                            args,
                        )
                    else:
                        fake_sample = ddpm_sample(
                            pos_coeff, netG, args.num_timesteps, x_t_1, args
                        )

                    fake_sample = to_range_0_1(fake_sample)
                    for j, x in enumerate(fake_sample):
                        index = i * args.batch_size + j
                        torchvision.utils.save_image(
                            x,
                            "{}/{}.jpg".format(save_dir, index),
                        )

        paths = [save_dir, real_img_dir]

        kwargs = {"batch_size": 100, "device": device, "dims": 2048}
        fid = calculate_fid_given_paths(paths=paths, **kwargs)
        print("FID = {}".format(fid))
    else:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser("ddgan parameters")
    parser.add_argument(
        "--seed", type=int, default=1024, help="seed used for initialization"
    )
    parser.add_argument(
        "--compute_fid",
        action="store_true",
        default=False,
        help="whether or not compute FID",
    )
    parser.add_argument(
        "--use_rejection_sampling",
        action="store_true",
        default=False,
        help="whether or not to use rejection sampling",
    )
    parser.add_argument(
        "--enable_reinit",
        action="store_true",
        default=False,
        help="whether or not to use reinitialization in rejection sampling",
    )
    parser.add_argument("--constant_quantile", type=float, default=0.7, help="use quantile of log ratio samples")
    parser.add_argument("--reject_full_trajectory", action="store_true", default=False, help="whether or not to reject full trajectory")
    parser.add_argument("--reinit_steps", type=int, default=1, help="how many steps forward during reinitialization")
    parser.add_argument("--num_samples", type=int, default=50_000)
    parser.add_argument("--g_epoch", type=int, default=1200)
    parser.add_argument("--d_epoch", type=int, default=1400)
    parser.add_argument("--precomputed", action="store_true", default=False, help="whether fake images are precomputed")
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
    parser.add_argument(
        "--real_img_dir",
        default="./pytorch_fid/cifar10_train_stat.npy",
        help="directory to real images for FID computation",
    )

    parser.add_argument("--dataset", default="cifar10", help="name of dataset")
    parser.add_argument("--image_size", type=int, default=32, help="size of image")

    parser.add_argument("--nz", type=int, default=100)
    parser.add_argument("--num_timesteps", type=int, default=4)

    parser.add_argument("--z_emb_dim", type=int, default=256)
    parser.add_argument("--t_emb_dim", type=int, default=256)
    parser.add_argument(
        "--batch_size", type=int, default=200, help="sample generating batch size"
    )
    parser.add_argument("--ngf", type=int, default=64)

    args = parser.parse_args()

    sample_and_test(args)
