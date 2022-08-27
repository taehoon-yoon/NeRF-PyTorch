import imageio
import glob
import os
import numpy as np
import torch
from tqdm import tqdm
from Train import render_full_image
from Config import *
from Utils import *
from load_blender import load_blender_data
from Model import NeRF


def Infer(renderSize, renderAngle, dataSetPath, modelPath, DEVICE):
    print('Loading Data...')
    _, _, render_poses, hwf, _ = load_blender_data(dataSetPath, half_res=half_res, renderSize=renderSize,
                                                   renderAngle=renderAngle)
    render_poses = render_poses.to(DEVICE)
    H, W, focal = hwf
    K = np.array([
        [focal, 0, 0.5 * W],
        [0, focal, 0.5 * H],
        [0, 0, 1]
    ])

    # Load Saved Model
    models = glob.glob(os.path.join(modelPath, '*'))
    while True:
        modelPrompt = input('Enter model name to render image, (to use '
                            'best model enter 0 else enter model name(ex: Epoch_5000)): ')
        if modelPrompt == '0':
            best_epoch = 0
            model_path = None
            for path in models:
                fileName = os.path.basename(path)
                epoch = int(fileName.split('_')[-1].split('.')[0])
                if epoch > best_epoch:
                    best_epoch = epoch
                    model_path = path
            break
        else:
            model_path = os.path.join(modelPath, modelPrompt, '.tar')
            if not os.path.exists(model_path):
                continue
            else:
                break
    model_name = os.path.basename(model_path).split('.')[0]
    print('Loading model -{}-...'.format(model_name))
    ckpt = torch.load(model_path)
    posENC = PositionalEncoding(lPosition)
    dirENC = PositionalEncoding(lDirection) if use_viewDirection else None
    direction_ch = dirENC.totalDims if use_viewDirection else 0

    Coarse = NeRF(depth=8, hidden_units=256, position_ch=posENC.totalDims,
                  direction_ch=direction_ch, output_ch=4, use_viewdirs=use_viewDirection).to(DEVICE)
    Coarse.eval()
    print('Coarse Model load: {}'.format(Coarse.load_state_dict(ckpt['Coarse'])))
    if use_FineModel:
        Fine = NeRF(depth=8, hidden_units=256, position_ch=posENC.totalDims,
                    direction_ch=direction_ch, output_ch=4, use_viewdirs=use_viewDirection).to(DEVICE)
        print('Fine Model load: {}'.format(Fine.load_state_dict(ckpt['Fine'])))
        Fine.eval()
    else:
        print('Warning: Not using Fine model!')
        Fine = False

    # Main Rendering part
    rendered_imgs = []
    with torch.no_grad():
        idx = 0
        for theta in tqdm(np.linspace(-180, 180, renderSize + 1)[:-1], desc='Progress', leave=False):
            render_return = render_full_image(render_poses[idx], [H, W], K, Coarse, Fine, posENC, dirENC, DEVICE)
            pred_image = torch.reshape(render_return['rgb_map'].cpu(), [H, W, 3])
            pred_image = (255. * pred_image).to(torch.uint8).numpy()
            rendered_imgs.append(pred_image)
            imageio.imsave(renderingImg_save_path +
                           'Theta{:.2f}_RenderAngle{:.1f}.png'.format(theta + 180.0, renderAngle), pred_image)
            idx += 1
    imageio.mimsave(renderingImg_save_path + 'final.gif', rendered_imgs,fps=min(renderSize//5,60))


if __name__ == '__main__':
    print('--NeRF Rendering Program--')
    if torch.cuda.is_available():
        print('GPU is available!')
        num = '0'
        if torch.cuda.device_count() > 1:
            print('There are {} possible GPUs'.format(torch.cuda.device_count()))
            num = input('please enter number (ex) 0 for gpu:0, 1 for gpu:1): ')
        DEVICE = torch.device('cuda:' + num)
    else:
        print('CPU is available!')
        DEVICE = torch.device('cpu')
    print(f'Rendering with Device: {DEVICE}')
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

    while True:
        model_dir = input('Please enter saved model directory located under models: ')
        model_paths = './models/' + model_dir
        if not os.path.exists(model_paths):
            print('Invalid path')
            continue
        break
    render_angle = float(input('Enter Rendering angle in degree: '))
    render_size = int(input('Enter Rendering size: '))
    save_dir = input('Please enter directory name to save rendered image: ')
    renderingImg_save_path = renderingImg_save_path + save_dir + '/'
    os.makedirs(renderingImg_save_path, exist_ok=True)
    Infer(render_size, render_angle, data_path, model_paths, DEVICE)
    print('--Program End--')
