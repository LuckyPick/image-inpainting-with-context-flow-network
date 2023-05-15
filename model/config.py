import os
import yaml


class Config(dict):
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self._yaml = f.read()
            self._dict = yaml.load(self._yaml)
            self._dict['PATH'] = os.path.dirname(config_path)

    def __getattr__(self, name):
        if self._dict.get(name) is not None:
            return self._dict[name]

        if DEFAULT_CONFIG.get(name) is not None:
            return DEFAULT_CONFIG[name]

        return None

    def print(self):
        print('Model configurations:')
        print('---------------------------------')
        print(self._yaml)
        print('')
        print('---------------------------------')
        print('')


DEFAULT_CONFIG = {
    'use_cuda': True,
    'GPU': [0, 1, 2, 3, 4, 5],         # list of gpu ids

    'data_path': "",
    'mask_path': "",
    'image_shape': [256, 256, 3],
    'mask_shape':  [256, 256, 1],
    'with_subfolder': False,
    'random_crop': False,
    'return_name': False,
    'batch_size': 24,
    'num_workers': 6,

    'VERBOSE': 0,
    'LR': 0.0001,               # learning rate
    'D2G_LR': 0.1,                 # discriminator/generator learning rate ratio
    'BETA1': 0.0,                # adam optimizer beta1
    'BETA2': 0.9,                 # adam optimizer beta2
    'MAX_ITERS': 2e6,               # maximum number of iterations to train the model

    'L1_LOSS_WEIGHT': 1,           # l1 loss weight
    'STYLE_LOSS_WEIGHT': 250,     # style loss weight
    'CONTENT_LOSS_WEIGHT': 0.1,    # perceptual loss weight
    'INPAINT_ADV_LOSS_WEIGHT': 0.1,  # adversarial loss weight

    'GAN_LOSS': 'nsgan',            # nsgan | lsgan | hinge

    # how many iterations to wait before saving model (0: never)
    'SAVE_INTERVAL': 1000,
    # how many iterations to wait before sampling (0: never)
    'SAMPLE_INTERVAL': 1000,
    'SAMPLE_SIZE': 12,          # number of images to sample
    # how many iterations to wait before model evaluation (0: never)
    'EVAL_INTERVAL': 0,
    # how many iterations to wait before logging training status (0: never)
    'LOG_INTERVAL': 10,
}
