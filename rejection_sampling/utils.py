from collections import defaultdict
import torch
from tqdm.auto import trange, tqdm


def extract(input, t, shape):
    out = torch.gather(input, 0, t)
    reshape = [shape[0]] + [1] * (len(shape) - 1)
    out = out.reshape(*reshape)

    return out


def q_sample_tp1(pos_coeff, x_t, t):
    sigmas = pos_coeff.betas**0.5
    a_s = torch.sqrt(1 - pos_coeff.betas)

    noise = torch.randn_like(x_t)
    x_t_plus_one = (
        extract(a_s, t, x_t.shape) * x_t + extract(sigmas, t, x_t.shape) * noise
    )
    return x_t_plus_one


def sample_posterior(coefficients, x_0, x_t, t):

    def q_posterior(x_0, x_t, t):
        mean = (
            extract(coefficients.posterior_mean_coef1, t, x_t.shape) * x_0
            + extract(coefficients.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        var = extract(coefficients.posterior_variance, t, x_t.shape)
        log_var_clipped = extract(
            coefficients.posterior_log_variance_clipped, t, x_t.shape
        )
        return mean, var, log_var_clipped

    def p_sample(x_0, x_t, t):
        mean, _, log_var = q_posterior(x_0, x_t, t)

        noise = torch.randn_like(x_t)

        nonzero_mask = 1 - (t == 0).type(torch.float32)

        return (
            mean + nonzero_mask[:, None, None, None] * torch.exp(0.5 * log_var) * noise
        )

    sample_x_pos = p_sample(x_0, x_t, t)

    return sample_x_pos

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

    @torch.no_grad()
    @classmethod
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

        for _ in trange(n_runs):
            x_1 = torch.randn(
                opt.batch_size, opt.num_channels, opt.image_size, opt.image_size
            ).to(device)
            _ = sample_loop(x_1)

        for t, lst in log_ratio_dict.items():
            log_ratio_dict[t] = torch.cat(lst).cpu()
        
        return cls(discriminator, log_ratio_dict)


@torch.no_grad
def rejection_sample(
    coefficients,
    generator,
    rs_processor,
    n_time,
    x_init,
    opt,
    reject_full_trajectory=False,
):
    x = x_init
    t = torch.full((x.shape[0],), n_time - 1, dtype=torch.int64, device=x.device)

    finished_samples = 0
    total_samples = x_init.shape[0]

    def sample_loop(x, t):
        latent_z = torch.randn(x.shape[0], opt.nz, device=x.device)
        x_0 = generator(x, t, latent_z)
        x_new = sample_posterior(coefficients, x_0, x, t)
        accept_mask = rs_processor.get_accept_mask(x_new, x, t)
        x[accept_mask] = x_new[accept_mask]
        t[accept_mask] = t[accept_mask] - 1
        if reject_full_trajectory:
            x[~accept_mask] = torch.randn(((~accept_mask).sum(), *x_init.shape[1:]), dtype=x_init.dtype, device=x_init.device)
            t[~accept_mask] = torch.full(((~accept_mask).sum(),), n_time - 1, dtype=torch.int64, device=x.device)
        return x, t

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
    # Iterative version    
    # def reinit_loop(x, t):
    #     cur_mask = torch.ones((t.shape[0],), dtype=bool, device=x.device)
    #     while cur_mask.sum() > 0:
    #         x_hat = q_sample_tp1(coefficients, x, t)
    #         accept_mask = rs_processor.get_accept_mask(x, x_hat, t) | (t == n_time - 1)
    #         x[cur_mask] = x_hat[cur_mask] # copy only x's that are in reinit regime
    #         cur_mask = cur_mask & (~accept_mask)
    #         t[cur_mask] = t[cur_mask] + 1
    #     return x, t

    # Recursive version
    def reinit_loop(x, t):
        x_hat = q_sample_tp1(coefficients, x, t)
        accept_mask = rs_processor.get_accept_mask(x, x_hat, t) | (t == n_time - 1)
        x[accept_mask] = x_hat[accept_mask]
        if (~accept_mask).sum() > 0:
            t[~accept_mask] = t[~accept_mask] + 1
            x[~accept_mask], t[~accept_mask] = reinit_loop(x[~accept_mask], t[~accept_mask])
        return x, t

    def sample_loop(x, t):
        latent_z = torch.randn(x.shape[0], opt.nz, device=x.device)
        x_0 = generator(x, t, latent_z)
        x_new = sample_posterior(coefficients, x_0, x, t)
        accept_mask = rs_processor.get_accept_mask(x_new, x, t)
        x = x_new
        t[accept_mask] = t[accept_mask] - 1
        if (~accept_mask).sum() > 0:
            x[~accept_mask], t[~accept_mask] = reinit_loop(x[~accept_mask], t[~accept_mask])
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
            x_res.append(x[finished_mask].cpu())
            x = x[~finished_mask]
            t = t[~finished_mask]
            finished_samples += finished_mask.sum().item()
    return torch.cat(x_res).to(x_init.device)


# def reinit(coefficients, generator, rs_processor, x_t, t, n_time, opt):
#     x_tp1 = q_sample_tp1(coefficients, x_t, t) # don't need to pass t+1, because indices are shifted for reverse process
#     accept_mask = rs_processor.get_accept_mask(x_t, x_tp1, t + 1) | (t == n_time - 1)
#     if accept_mask.all():
#         return x_tp1
#     x_tp2 = reinit(coefficients, rs_processor, x_tp1, t + 1, n_time)
#     return one_step_diffrs(coefficients, generator, rs_processor, x_tp2, t + 1, n_time, opt)

# def one_step_diffrs(coefficients, generator, rs_processor, x_tp1, t, n_time, opt):
#     x_new = None
#     while x_new is None:
#         latent_z = torch.randn(x_tp1.size(0), opt.nz, device=x_tp1.device)
#         x_0 = generator(x_tp1, t, latent_z)
#         x_t = sample_posterior(coefficients, x_0, x_tp1, t)
#         accept_mask = rs_processor.get_accept_mask(x_t, x_tp1, t)
#         if accept_mask[0]:
#             x_new = x_t
#         else:
#             x_tp1 = reinit(coefficients, generator, rs_processor, x_t, t, n_time, opt)
#     return x_new

# def rejection_sample_reinit(
#     coefficients,
#     generator,
#     rs_processor,
#     n_time,
#     x_init,
#     opt,
# ):
#     x = x_init
#     with torch.no_grad():
#         for t_val in range(n_time - 1, -1, -1):
#             t = torch.full((x.size(0),), t_val, dtype=torch.int64, device=x.device)
#             x = one_step_diffrs(coefficients, generator, rs_processor, x, t, n_time, opt)
#     return x

