import torch
import os
import imageio
from tqdm import tqdm
from torchmetrics import PeakSignalNoiseRatio
from torch.utils.tensorboard import SummaryWriter
from load_blender import *
from Utils import *
from Model import *
from Config import *


def batchify(fn, netChunk=None):
    if netChunk is None:
        return fn
    else:
        def fn_(x):
            return torch.cat([fn(x[i:i + netChunk]) for i in range(0, x.shape[0], netChunk)], dim=0)

        return fn_


def render(rays, Coarse, Fine, posENC, dirENC, perturb_, DEVICE):
    """
    Important:----------------------------------------------------------------------
    rays's shape must be (ray_size,6 or 9)
    if you are using this function like rendering full size image with (H,W, 6 or 9)
    you have to batchify the input rays.
    --------------------------------------------------------------------------------
    """

    ray_size = rays.shape[0]
    rays_o, rays_d = rays[..., :3], rays[..., 3:6]  # (ray_size,3), (ray_size,3)
    viewDir = rays[..., 6:] if use_viewDirection else None  # (ray_size,3)

    tVals = get_tVals(batch_size=ray_size, sample_size=Nc, near=2., far=6., lindisp=lindisp,
                      perturb=perturb_)  # (ray_size,Nc)
    tVals = tVals.to(DEVICE)
    points = rays_o[..., None, :] + rays_d[..., None, :] * tVals[..., None]  # (ray_size,Nc,3)

    # Run Coarse Network
    points_coarse_shape = points.shape
    points = torch.reshape(points, [-1, 3])
    embedded = posENC(points)  # (ray_size*Nc, 2*lPosition*3+3)
    if use_viewDirection:
        viewDir_ = viewDir[..., None, :].expand(points_coarse_shape)
        viewDir_ = torch.reshape(viewDir_, [-1, 3])
        embedded = torch.cat([embedded, dirENC(viewDir_)], dim=-1)  # (ray_size*Nc, 2*lPosition*3+3 + 2*lDirection*3+3)

    outputs = batchify(Coarse, networkChunk)(embedded)  # (ray_size*Nc, 4) first 3 for RGB last for sigma
    outputs = torch.reshape(outputs, shape=list(points_coarse_shape[:-1]) + [outputs.shape[-1]])  # (ray_size, Nc, 4)

    RGB_coarse, disp_coarse, acc_coarse, weights, depth_coarse = \
        VolumeRender(outputs, tVals, rays_d, DEVICE, raw_noise_std, white_background)

    # Run Fine Network
    if use_FineModel:
        tValsMid = (tVals[..., 1:] + tVals[..., :-1]) * 0.5
        tValsFine = sample_pdf(tValsMid, weights[..., 1:-1], Nf, DEVICE, perturb)
        tValsFine = tValsFine.detach()

        tValsFine, _ = torch.sort(torch.cat([tVals, tValsFine], dim=-1), dim=-1)  # (ray_size,Nc+Nf)

        points = rays_o[..., None, :] + rays_d[..., None, :] * tValsFine[..., None]  # (ray_size,Nc+Nf,3)

        points_fine_shape = points.shape
        points = torch.reshape(points, [-1, 3])
        embedded = posENC(points)  # (ray_size*(Nc+Nf), 2*lPosition*3+3)
        if use_viewDirection:
            viewDir_ = viewDir[..., None, :].expand(points_fine_shape)
            viewDir_ = torch.reshape(viewDir_, [-1, 3])
            embedded = torch.cat([embedded, dirENC(viewDir_)], dim=-1)
        outputs = batchify(Fine, networkChunk)(embedded)
        outputs = torch.reshape(outputs, shape=list(points_fine_shape[:-1]) + [outputs.shape[-1]])

        RGB_fine, disp_fine, acc_fine, weights, depth_fine = \
            VolumeRender(outputs, tValsFine, rays_d, DEVICE, raw_noise_std, white_background)
        ret = {'rgb_map': RGB_fine, 'disp_map': disp_fine, 'acc_map': acc_fine, 'depth_map': depth_fine,
               'rgb_coarse': RGB_coarse, 'disp_coarse': disp_coarse, 'acc_coarse': acc_coarse,
               'depth_coarse': depth_coarse}
    else:
        ret = {'rgb_map': RGB_coarse, 'disp_map': disp_coarse, 'acc_map': acc_coarse, 'depth_map': depth_coarse}

    return ret


def render_full_image(render_pose, hw, K, Coarse, Fine, posENC, dirENC, DEVICE):
    H, W = hw
    rays_o, rays_d = get_rays(H, W, K, render_pose, DEVICE)  # (H,W,3)
    rays_o = torch.reshape(rays_o, [-1, 3])
    rays_d = torch.reshape(rays_d, [-1, 3])
    rays = torch.cat([rays_o, rays_d], dim=-1)
    if use_viewDirection:
        viewDir = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
        rays = torch.cat([rays, viewDir], dim=-1)

    all_ret = {}
    for i in tqdm(range(0, rays.shape[0], chunk), desc='Rendering Image', leave=False):
        ret = render(rays[i:i + chunk], Coarse, Fine, posENC, dirENC, perturb_=False, DEVICE=DEVICE)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])
    all_ret = {k: torch.cat(all_ret[k], dim=0) for k in all_ret}
    return all_ret


def Train(dataSetPath, exp_name, test_img_idx):
    print('Loading Data and Preprocessing...')
    image, poses, renderPoses, hwf, idx_split = load_blender_data(dataSetPath, half_res=half_res, renderSize=40,
                                                                  renderAngle=30.0)
    '''
        image: (400,H,W,4)
        poses: (400,4,4)
        renderPoses: (renderSize,4,4)
        hwf=[Height,Width,focal]
        idx_split=[idx_train,idx_val,idx_test]
    '''
    idx_train, idx_val, idx_test = idx_split
    H, W, focal = hwf
    K = np.array([
        [focal, 0, 0.5 * W],
        [0, focal, 0.5 * H],
        [0, 0, 1]
    ])

    if white_background:
        image = image[..., :3] * image[..., -1:] + (1. - image[..., -1:])  # image: (400,H,W,3)
    else:
        image = image[..., :3]

    if rendering_during_train:
        if render_testSet:
            renderPoses = poses[idx_test]
        renderPoses = renderPoses.to(DEVICE)
    poses = torch.tensor(poses).to(DEVICE)
    print('Finished!')
    print('Loading Encoder and Model...')
    posENC = PositionalEncoding(lPosition)
    dirENC = PositionalEncoding(lDirection) if use_viewDirection else None
    direction_ch = dirENC.totalDims if use_viewDirection else 0
    # output_ch=5 if Nf>0 else 4 -> ??

    Coarse = NeRF(depth=8, hidden_units=256, position_ch=posENC.totalDims,
                  direction_ch=direction_ch, output_ch=4, use_viewdirs=use_viewDirection).to(DEVICE)
    grad_vars = list(Coarse.parameters())

    if use_FineModel:
        Fine = NeRF(depth=8, hidden_units=256, position_ch=posENC.totalDims,
                    direction_ch=direction_ch, output_ch=4, use_viewdirs=use_viewDirection).to(DEVICE)
        grad_vars += list(Fine.parameters())
    else:
        Fine = False
    optimizer = torch.optim.Adam(params=grad_vars, lr=learning_rate, betas=(0.9, 0.999))
    decay_rate = 0.1
    decay_steps = lr_decay * 1000
    mseLoss = torch.nn.MSELoss()
    PSNR = PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)
    testPSNR = PeakSignalNoiseRatio(data_range=1.0)
    print('Finished!')
    print('Main Train Start!')

    epochTQDM = tqdm(range(1, totalSteps + 1))
    writer = SummaryWriter('runs/' + exp_name)
    totalLoss = 0.
    totalPSNR = 0.
    bestPSNR = 0.
    for step in epochTQDM:
        random_idx = np.random.choice(idx_train)
        target = image[random_idx]
        target = torch.tensor(target).to(DEVICE)
        pose = poses[random_idx]

        rays_o, rays_d = get_rays(H, W, K, pose, DEVICE)  # (H,W,3), (H,W,3)
        if step < preCrop_iter:
            dH = int(0.5 * H * preCrop_fraction)
            dW = int(0.5 * W * preCrop_fraction)
            coords = torch.stack(torch.meshgrid(torch.arange(H // 2 - dH, H // 2 + dH),
                                                torch.arange(W // 2 - dW, W // 2 + dW), indexing='ij'), dim=-1)
        else:
            coords = torch.stack(torch.meshgrid(torch.arange(0, H), torch.arange(0, W), indexing='ij'),
                                 dim=-1)  # (H,W,2)

        coords = torch.reshape(coords, shape=[-1, 2])
        batch_ray_idxs = np.random.choice(coords.shape[0], size=N_rand, replace=False)
        selected_coords = coords[batch_ray_idxs].long()  # (N_rand,2)
        rays_o = rays_o[selected_coords[:, 0], selected_coords[:, 1]]  # (N_rand,3)
        rays_d = rays_d[selected_coords[:, 0], selected_coords[:, 1]]  # (N_rand,3)
        target = target[selected_coords[:, 0], selected_coords[:, 1]]  # (N_rand,3)

        # main train
        rays = torch.cat([rays_o, rays_d], dim=-1)
        if use_viewDirection:
            viewDir = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
            rays = torch.cat([rays, viewDir], dim=-1)
        '''
            rays's shape= (N_rand,9) if use_viewDirection else (N_rand,6)
        '''
        render_return = render(rays, Coarse, Fine, posENC, dirENC, perturb_=perturb, DEVICE=DEVICE)
        pred = render_return['rgb_map']

        optimizer.zero_grad()
        loss = mseLoss(pred, target)

        if use_FineModel:
            loss_coarse = mseLoss(render_return['rgb_coarse'], target)
            loss += loss_coarse
        psnr = PSNR(pred, target)
        loss.backward()
        optimizer.step()
        totalPSNR += psnr.item()
        totalLoss += loss.item()

        new_lr = learning_rate * (decay_rate ** (step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

        # render test set
        if step % (render_one_test_image_epoch * 100) == 0:
            with torch.no_grad():
                Coarse.eval()
                Fine.eval()
                render_return = render_full_image(poses[idx_test[test_img_idx]], [H, W], K, Coarse, Fine, posENC,
                                                  dirENC, DEVICE)
                pred_image = torch.reshape(render_return['rgb_map'].cpu(), [H, W, 3])
                target_image = torch.tensor(image[idx_test[test_img_idx]])
                psnr = testPSNR(pred_image, target_image)
                pred_image = (255. * pred_image).to(torch.uint8).numpy()
                imageio.imsave(testImg_save_pth + '/{:05d}_{:.2f}.png'.format(int(step // 100), psnr), pred_image)
            writer.add_scalar('PSNR_test', psnr, int(step // 100))
            # model save
            if bestPSNR < psnr:
                bestPSNR = psnr
                save_path = model_save_pth + '/Epoch_{}.tar'.format(int(step // 100))
                torch.save({
                    'step': step,
                    'Coarse': Coarse.state_dict(),
                    'Fine': Fine.state_dict(),
                    'optimizer': optimizer.state_dict()
                }, save_path)
            Coarse.train()
            Fine.train()

        # Tqdm setting, every Epoch
        if (step % 100) == 0:
            avgPSNR = totalPSNR / 100.
            avgLoss = totalLoss / 100.
            epoch = int(step // 100)
            epochTQDM.set_postfix({
                'epoch': epoch,
                'loss': '{:.04f}'.format(avgLoss),
                'psnr': '{:.02f}'.format(avgPSNR)
            })
            totalLoss = 0.
            totalPSNR = 0.
            writer.add_scalar('Loss', avgLoss, epoch)
            writer.add_scalar('PSNR_train', avgPSNR, epoch)


if __name__ == '__main__':
    print('--NeRF Program--')
    if torch.cuda.is_available():
        print('GPU is available!')
        num = '0'
        if torch.cuda.device_count() > 1:
            print('There are {} possible GPUs'.format(torch.cuda.device_count()))
            num = input('please enter number (ex) 0 for cuda:0, 1 for cuda:1): ')
        DEVICE = torch.device('cuda:' + num)
    else:
        print('CPU is available!')
        DEVICE = torch.device('cpu')
    print(f'Training with Device: {DEVICE}')
    while True:
        data_name = input('Please enter data name: ')
        if data_name not in available_datas:
            print('--Available data sets--')
            for names in available_datas:
                print(names)
        else:
            break
    data_path = './nerf_synthetic/' + data_name
    if not os.path.exists(data_path):
        print('There is no such data set. Please re-download data sets')
        raise NotImplementedError()

    expName = input('Please enter the experiment name: ')
    testImg_save_pth = testImg_save_pth + expName
    model_save_pth = model_save_pth + expName
    os.makedirs(testImg_save_pth, exist_ok=True)
    os.makedirs(model_save_pth, exist_ok=True)
    print('Enter the test image index')
    test_image_index = int(input('(Normally 0 will be fine, but for chair and lego dataset I recommend 55): '))
    Train(data_path, expName, test_image_index)
