import torch
import torch.nn.functional as F


class PositionalEncoding:
    def __init__(self, multiRes, includeInput=True, dim=3):
        self.embed_fns = []
        self.totalDims = 0
        encode_fn = [torch.sin, torch.cos]
        if includeInput:
            self.embed_fns.append(lambda x: x)
            self.totalDims += dim
        for res in range(multiRes):
            res = 2 ** res
            for fn in encode_fn:
                self.embed_fns.append(lambda x, fn_=fn, res_=res: fn_(res_ * x))
                self.totalDims += dim

    def __call__(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], dim=-1)


def get_rays(H, W, K, c2w, device):
    """
    output: (rays_o,rays_d)
    rays_o shape: (H,W,3)
    rays_d shape: (H,W,3)

    K is just a matrix
    [focal  0    Width/2]
    [0    focal Height/2]
    [0      0       1   ]
    for a blender data
    """
    (x, y) = torch.meshgrid(torch.arange(W, dtype=torch.float32), torch.arange(H, dtype=torch.float32), indexing='xy')
    dirs = torch.stack([(x - K[0][2]) / K[0][0], -(y - K[1][2]) / K[1][1], -torch.ones_like(x)], dim=-1)  # (H,W,3)
    dirs = dirs.to(device)
    rays_d = dirs @ (c2w[:3, :3].t())
    # Same as: rays_d=torch.sum(dirs[...,None,:]*c2w[:3,:3],dim=-1)
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d


def sample_pdf(bins, weight, N_sample, device, perturb=True):
    """
        -input
        bins=(Batch,Nc-1)
        weight=(Batch,Nc-2)
        N_sample is Nf in Original paper

        -output
        (Batch,Nf)
        *Batch can be just (Batch size) or also (Batch size, *, H, W)
    """
    weight = weight + 1e-5  # prevent NAN
    pdf = weight / torch.sum(weight, dim=-1, keepdim=True)
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat([torch.zeros_like(weight[..., :1]), cdf], dim=-1)  # (Batch, Nc-1)
    NcMinusOne = cdf.shape[-1]

    if perturb:
        u = torch.rand(list(weight.shape[:-1]) + [N_sample])
    else:
        u = torch.linspace(0., 1., steps=N_sample)
        u = u.expand(list(weight.shape[:-1]) + [N_sample])

    u = u.contiguous()  # (Batch, Nf)
    u = u.to(device)
    idxs = torch.searchsorted(cdf, u, right=True)  # (Batch, Nf)
    below = torch.max(torch.zeros_like(idxs), idxs - 1)
    above = torch.min(torch.ones_like(idxs) * (NcMinusOne - 1), idxs)
    inds_g = torch.stack([below, above], dim=-1)  # (Batch, Nf, 2)

    matched_shape = list(inds_g.shape[:-1]) + [NcMinusOne]  # (Batch, Nf, Nc-1)
    cdf_g = torch.gather(cdf[..., None, :].expand(matched_shape), dim=-1, index=inds_g)
    bins_g = torch.gather(bins[..., None, :].expand(matched_shape), dim=-1, index=inds_g)

    denom = cdf_g[..., 1] - cdf_g[..., 0]  # (Batch, Nf)
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)

    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])
    return samples


def get_tVals(batch_size, sample_size, near=2., far=6., lindisp=True, perturb=True):
    """
        returns t values of shape [batch_size,sample_size]
        with each raw values [near,far)
        for example, for batch_size=3, sample_size=5, near=2., far=6.
        [
        [2.1, 2.2, 2.4, 3.1, 5.4],
        [2.2, 2.5, 3.7, 4.5, 5.9],
        [2.1, 2.3, 4.6, 4.9, 5.8]
        ]
    """
    near = torch.tensor(near, dtype=torch.float32).expand((batch_size, sample_size))
    far = torch.tensor(far, dtype=torch.float32).expand((batch_size, sample_size))

    tVals = torch.linspace(0., 1., steps=sample_size)
    if lindisp:
        tVals = 1. / (1. / near * (1 - tVals) + 1. / far * tVals)
    else:
        tVals = near + (far - near) * tVals
    # tVals shape: (Batch,Nc)

    if perturb:
        mid = (tVals[..., 1:] + tVals[..., :-1]) * 0.5
        above = torch.cat([mid, tVals[..., -1:]], dim=-1)
        below = torch.cat([tVals[..., :1], mid], dim=-1)
        tRand = torch.rand((batch_size, sample_size))
        tVals = below + tRand * (above - below)
    return tVals


def VolumeRender(raw_input, t_vals, rays_d, device, raw_noise_std=0., white_backGround=False):
    """
        raw_input shape: (ray_size, sample_size, 4)
        t_vals_shape: (ray_size, sample_size)
        rays_d shape: (ray_size, 3)
    """

    raw_RGB = raw_input[..., :3]  # (ray_size,sample_size,3)
    raw_sigma = raw_input[..., 3]  # (ray_size,sample_size)

    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw_sigma.shape) * raw_noise_std
    sigma = F.relu(raw_sigma + noise)
    RGB = torch.sigmoid(raw_RGB)

    delta = t_vals[..., 1:] - t_vals[..., :-1]
    delta = torch.cat([delta, torch.tensor(1e10, dtype=torch.float32, device=device).expand(delta[..., :1].shape)],
                      dim=-1)  # (ray_size, sample_size)
    delta = delta * torch.norm(rays_d, dim=-1, keepdim=True)

    exponentialTerm = torch.exp(-sigma * delta)
    alpha = 1 - exponentialTerm

    Transmittance = torch.cat(
        [torch.ones_like(exponentialTerm[..., :1]), torch.cumprod(exponentialTerm + 1e-10, dim=-1)],
        dim=-1)[..., :-1]  # (ray_size, sample_size)
    weight = Transmittance * alpha

    RGB_map = torch.sum(weight[..., None] * RGB, dim=-2)  # (ray_size,3)
    depth_map = torch.sum(weight * t_vals, dim=-1)
    disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weight, -1))
    acc_map = torch.sum(weight, -1)

    if white_backGround:
        RGB_map = RGB_map + (1. - acc_map[..., None])

    return RGB_map, disp_map, acc_map, weight, depth_map
