import os
import torch
import numpy as np
import json
from PIL import Image

translate_positive_z = lambda z: torch.tensor([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, z],
    [0, 0, 0, 1]
], dtype=torch.float32)

rotate_worldCoordinate_x_axis_CCW = lambda phi: torch.tensor([
    [1, 0, 0, 0],
    [0, np.cos(phi), -np.sin(phi), 0],
    [0, np.sin(phi), np.cos(phi), 0],
    [0, 0, 0, 1]
], dtype=torch.float32)

rotate_worldCoordinate_y_axis_CCW = lambda theta: torch.tensor([
    [np.cos(theta), 0, -np.sin(theta), 0],
    [0, 1, 0, 0],
    [np.sin(theta), 0, np.cos(theta), 0],
    [0, 0, 0, 1]
], dtype=torch.float32)

change_worldCoordinate_yz_axis = torch.tensor([
    [-1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1]
], dtype=torch.float32)


def pose_spherical(theta, phi, radius):
    c2w = translate_positive_z(radius)
    c2w = rotate_worldCoordinate_x_axis_CCW(phi / 180. * np.pi) @ c2w
    c2w = rotate_worldCoordinate_y_axis_CCW(theta / 180. * np.pi) @ c2w
    c2w = change_worldCoordinate_yz_axis @ c2w
    return c2w


def load_blender_data(dirpath, half_res=False, testSkip=1, renderSize=40, renderAngle=30.0):
    """
    output: Img,Pose,RenderPose,[H,W,focal],index_split
    Img=(400,H,W,4) where 4 is RGBA
    Pose=(400,4,4)
    RenderPose=(renderSize,4,4)
    index_split= 3 numpy array with 0~99,100~199,200~399
    each correspond to train,val,test idx
    """
    splits = ['train', 'val', 'test']
    jsons = {}
    for s in splits:
        with open(os.path.join(dirpath, 'transforms_{}.json'.format(s)), 'r') as f:
            jsons[s] = json.load(f)
    allImg = []
    allPose = []
    counts = [0]
    for s in splits:
        if s == 'train' or testSkip == 0:
            skip = 1
        else:
            skip = testSkip

        jsonData = jsons[s]
        Imgs = []
        Poses = []
        for frame in jsonData['frames'][::skip]:
            file_path = frame['file_path'].replace('./', '')
            matrix = np.array(frame['transform_matrix'], dtype=np.float32)
            img = Image.open(os.path.join(dirpath, file_path + '.png'))
            if half_res:
                H, W = img.height,img.width
                H = H // 2
                W = W // 2
                img = img.resize((H, W), resample=Image.LANCZOS)
            img = np.array(img, dtype=np.float32) / 255.
            Imgs.append(img)
            Poses.append(matrix)
        counts.append(counts[-1] + len(Imgs))
        allImg.append(Imgs)
        allPose.append(Poses)
    i_split = [np.arange(counts[i], counts[i + 1]) for i in range(3)]
    allImg = np.concatenate(allImg, axis=0)
    allPose = np.concatenate(allPose, axis=0)

    H, W = allImg[0].shape[:2]
    camera_angle_x = jsons['train']['camera_angle_x']
    focal = 0.5 * W / np.tan(0.5 * camera_angle_x)

    render_poses = torch.stack([pose_spherical(theta, -renderAngle, 4.0)
                                for theta in np.linspace(-180, 180, renderSize + 1)[:-1]], dim=0)

    return allImg, allPose, render_poses, [H, W, focal], i_split
