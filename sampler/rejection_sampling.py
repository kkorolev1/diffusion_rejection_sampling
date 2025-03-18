from collections import defaultdict
import torch
import numpy as np
from tqdm.auto import trange

from sampler.utils import sample_posterior, q_sample_next

# netD(x_t, t, x_tp1.detach())
# x_0_predict = netG(x_tp1.detach(), t, latent_z)
# x_pos_sample = sample_posterior(pos_coeff, x_0_predict, x_tp1, t) 


def repeat_tensor(x, new_batch_size):
    repeat_factor = (new_batch_size + x.size(0) - 1) // x.size(0)
    repeated_tensor = x.repeat(repeat_factor, *[1] * (x.dim() - 1))
    return repeated_tensor[:new_batch_size]

class RejectionSamplingProcessor:
    def __init__(self, discriminator, log_ratio_dict, device, constant_quantile=1.0):
        self.discriminator = discriminator.to(device)
        self.log_ratio_dict = {k: v.to(device) for k, v in log_ratio_dict.items()}
        self.log_constant_dict = {k: max(torch.quantile(v, constant_quantile).item(), 0) for k, v in log_ratio_dict.items()}
        self.log_min_dict = {k: torch.quantile(v, 0.05).item() for k, v in log_ratio_dict.items()}
        self.log_max_dict = {k: torch.quantile(v, 0.95).item() for k, v in log_ratio_dict.items()}

    @torch.no_grad()
    def get_log_ratio(self, x_t, x_tp1, t, use_clipping=True):
        batch_size = x_t.shape[0]
        new_batch_size = (batch_size + 3) // 4 * 4
        log_ratio = self.discriminator(
            repeat_tensor(x_t, new_batch_size),
            repeat_tensor(t, new_batch_size), 
            repeat_tensor(x_tp1, new_batch_size)
        ).flatten()
        log_ratio = log_ratio[:batch_size]
        if use_clipping:
            mins = RejectionSamplingProcessor.collect_from_dict(self.log_min_dict, t)
            maxs = RejectionSamplingProcessor.collect_from_dict(self.log_max_dict, t)
            return torch.clamp(log_ratio, min=mins, max=maxs)
        return log_ratio

    @torch.no_grad()
    def get_accept_mask(self, x_t, x_tp1, t):
        log_ratio = self.get_log_ratio(x_t, x_tp1, t)
        log_mt = RejectionSamplingProcessor.collect_from_dict(self.log_constant_dict, t)
        accept_mask = torch.log(torch.rand_like(log_ratio) + 1e-7) < log_ratio - log_mt
        return accept_mask

    @staticmethod
    def collect_from_dict(dict, time_tensor):
        return torch.tensor([dict[t.item()] for t in time_tensor], dtype=float, device=time_tensor.device)

    @classmethod
    @torch.no_grad()
    def recalculate_log_ratio(cls, coefficients, generator, discriminator, n_time, opt, device, n_runs=1):
        log_ratio_dict = defaultdict(list)

        def sample_loop(x_init):
            x = x_init
            for i in reversed(range(n_time)):
                t = torch.full((x.size(0),), i, dtype=torch.int64).to(x.device)
                latent_z = torch.randn(
                    x.size(0), opt.nz, device=x.device
                )
                x_0 = generator(x, t, latent_z)
                x_new = sample_posterior(coefficients, x_0, x, t)
                log_ratio = discriminator(x_new, t, x).flatten()
                log_ratio_dict[i].append(log_ratio)
                x = x_new.detach()
            return x

        for i in trange(n_runs):
            if i % 50 == 0:
                print(i)
            x_T = torch.randn(
                opt.batch_size, opt.num_channels, opt.image_size, opt.image_size
            ).to(device)
            _ = sample_loop(x_T)

        for t, lst in log_ratio_dict.items():
            log_ratio_dict[t] = torch.cat(lst).cpu()
        
        return cls(discriminator, log_ratio_dict, device)


@torch.no_grad
def rejection_sample(
    coefficients,
    generator,
    rs_processor,
    n_time,
    x_init,
    opt,
):
    def sample_loop(x, t):
        latent_z = torch.randn(x.shape[0], opt.nz, device=x.device)
        x_0 = generator(x, t, latent_z)
        x_new = sample_posterior(coefficients, x_0, x, t)
        accept_mask = rs_processor.get_accept_mask(x_new, x, t)
        x[accept_mask] = x_new[accept_mask]
        t[accept_mask] = t[accept_mask] - 1
        if opt.reject_full_trajectory:
            x[~accept_mask] = torch.randn(((~accept_mask).sum(), *x_init.shape[1:]), dtype=x_init.dtype, device=x_init.device)
            t[~accept_mask] = torch.full(((~accept_mask).sum(),), n_time - 1, dtype=torch.int64, device=x.device)
        return x, t

    x = x_init
    t = torch.full((x.shape[0],), n_time - 1, dtype=torch.int64, device=x.device)

    finished_samples = 0
    total_samples = x_init.shape[0]
    x_res = []
    while finished_samples < total_samples:
        x, t = sample_loop(x, t)
        finished_mask = t < 0
        if finished_mask.sum() > 0:
            x_res.append(x[finished_mask])
            x = x[~finished_mask]
            t = t[~finished_mask]
            finished_samples += finished_mask.sum()
    return torch.cat(x_res)


@torch.no_grad()
def rejection_sample_reinit(
    coefficients,
    generator,
    rs_processor,
    n_time,
    x_init,
    opt,
):
    def reinit(x, t, steps=1):
        s = torch.clamp(t + steps, max=n_time)
        x = q_sample_next(coefficients, x, t, s)
        return x, s-1

    def sample_loop(x, t):
        latent_z = torch.randn(x.shape[0], opt.nz, device=x.device)
        x_0 = generator(x, t, latent_z)
        x_new = sample_posterior(coefficients, x_0, x, t)
        accept_mask = rs_processor.get_accept_mask(x_new, x, t)
        x = x_new
        t = t - 1
        # t[accept_mask] = t[accept_mask] - 1
        if (~accept_mask).sum() > 0:
            x[~accept_mask], t[~accept_mask] = reinit(x[~accept_mask], t[~accept_mask], opt.reinit_steps)
        return x, t

    x = x_init
    t = torch.full((x.shape[0],), n_time - 1, dtype=torch.int64, device=x.device)

    finished_samples = 0
    total_samples = x_init.shape[0]
    x_res = []
    while finished_samples < total_samples:
        x, t = sample_loop(x, t)
        finished_mask = t < 0
        if finished_mask.sum() > 0:
            x_res.append(x[finished_mask])
            x = x[~finished_mask]
            t = t[~finished_mask]
            finished_samples += finished_mask.sum()
    return torch.cat(x_res)
