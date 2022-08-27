"""
Important parameters
"""
half_res = False
chunk = 1024 * 32
networkChunk = 1024 * 64
totalSteps = 500000  # Equivalent to 5000 epochs, **1 epoch= 100 steps**
render_one_test_image_epoch = 50  # Equivalent to every 5000 steps
"""
#######################################################################################################################
"""
"""
Parameters you might have to change
"""
lPosition = 10
lDirection = 4
Nc = 64
Nf = 128
N_rand = 1024
preCrop_iter = 5000  # 5000steps, equivalent to 50 epochs
preCrop_fraction = 0.5
learning_rate = 5e-4
lr_decay = 500
'''
lr_decay: exponential learning rate decay (in 1000 steps), which is 5000 epochs
I know this may cause confusion. But I wanted to closely follow original implementation.
'''
"""
#######################################################################################################################
"""
"""
Recommend not to change
"""
white_background = True
perturb = True
use_viewDirection = True
use_FineModel = True
lindisp = False
'''
lindisp: bool. If True, sample linearly in inverse depth rather than in depth. See the README in repo
'''
raw_noise_std = 0.
testImg_save_pth = './fine_Image/'
model_save_pth = './models/'
renderingImg_save_path = './rendered_Image/'
"""
#######################################################################################################################
"""
"""
Do not change
"""
available_datas = ['chair', 'drums', 'ficus', 'hotdog', 'lego',
                   'materials', 'mic', 'ship']
rendering_during_train = False
render_testSet = True
'''
render_testSet: only meaningful when rendering_during_train is True, if render_testSet=True it will render video
with test set poses, if set to False, it will render video according to renderAngle, renderSize
which is argument of load_blender_data

*****Modified*****
IMPORTANT
Since inference will be executed in Inference.py, Do not change above two parameter.
But I still leave it as parameter, for those of you who want to play with my code.
'''
