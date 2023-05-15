import torch
from torch import nn
import os
import torch.optim as optim
from model import networks
from torch.utils.data import DataLoader
from utils.dataset import MyDataset
from utils.utils import Progbar, create_dir, stitch_images
from model.config import Config
'''
    only for Coarse Networks Training
'''

class OnlyCoarse(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        coarse_gen = networks.CoarseGenerator(
            input_dim=3, cnum=64, use_cuda=config.use_cuda, )

        if len(config.GPU) > 1:
            coarse_gen = nn.DataParallel(coarse_gen, config.GPU)

        l1_loss = nn.L1Loss()
        self.name = 'coarse_gen'
        self.coarse_weights_path = os.path.join(
            config.PATH, 'inpainting' + '_coarse.pth')
        self.iteration = 0
        self.add_module('coarse_gen', coarse_gen)
        self.add_module('l1_loss', l1_loss)

        self.coarse_optimizer = optim.Adam(
            params=coarse_gen.parameters(),
            lr=float(config.LR),
            betas=(config.BETA1, config.BETA2)
        )

    def process(self, images, masks):
        self.iteration += 1
        self.coarse_optimizer.zero_grad()
        stage1 = self(images, masks)
        coarse_l1_loss = 0
        coarse_l1_loss += self.l1_loss(stage1, images) * \
            self.config.COARSE_L1_LOSS_WEIGHT / torch.mean(masks)
        # coarse_l1_loss += self.l1_loss(stage1, images) * \
        #     self.config.COARSE_L1_LOSS_WEIGHT

        return stage1, coarse_l1_loss

    def forward(self, images, masks):
        return self.coarse_gen(images, masks)

    def backward(self, coarse_loss=None):
        coarse_loss.backward()
        self.coarse_optimizer.step()

    def save(self):
        print('\nsaving %s...\n' % self.name)
        torch.save({
            'coarse_gen': self.coarse_gen.state_dict()
        }, self.coarse_weights_path)

    def load(self):
        if os.path.exists(self.coarse_weights_path):
            print('Loading %s coarse generator...' % self.name)

            if torch.cuda.is_available():
                data = torch.load(self.coarse_weights_path)
            else:
                data = torch.load(self.coarse_weights_path,
                                  map_location=lambda storage, loc: storage)

            self.coarse_gen.load_state_dict(data['coarse_gen'])


class CoarseModel():
    def __init__(self, config):
        self.config = config
        self.model = OnlyCoarse(config).to(config.DEVICE)
        self.train_dataset = MyDataset(config)
        self.sample_iterator = self.train_dataset.create_iterator(
            config.SAMPLE_SIZE)
        self.sample_path = os.path.join(config.PATH, 'coarse_samples')

    def load(self):
        self.model.load()

    def save(self):
        self.model.save()

    def train(self):
        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            drop_last=True,
            shuffle=True
        )

        epoch = 0
        keep_training = True
        max_iteration = int(float((self.config.MAX_ITERS)))
        total = len(self.train_dataset)

        if total == 0:
            print('No training data was provided!')
            return

        while(keep_training):
            epoch += 1
            print('\n\nTraining epoch: %d' % epoch)

            progbar = Progbar(
                total, width=20, stateful_metrics=['epoch', 'iter'])

            for items in train_loader:
                self.model.train()
                images, masks = self.cuda(*items)
                stage1, coarse_l1_loss = self.model.process(images, masks)
                self.model.backward(coarse_l1_loss)
                iteration = self.model.iteration

                if iteration >= max_iteration:
                    keep_training = False
                    break

                logs = [
                    ('epoch', epoch),
                    ('iter', iteration),
                    ('l_coarse', coarse_l1_loss.item())
                ]
                progbar.add(len(images), values=logs if self.config.VERBOSE else [
                            x for x in logs if not x[0].startswith('l_')])

                if self.config.SAMPLE_INTERVAL and iteration % self.config.SAMPLE_INTERVAL == 0:
                    self.sample()
                if self.config.SAVE_INTERVAL and iteration % self.config.SAVE_INTERVAL == 0:
                    self.save()
        print('\nEnd training....')

    def sample(self, it=None):
        self.model.eval()

        items = next(self.sample_iterator)
        images, masks = self.cuda(*items)
        iteration = self.model.iteration
        masked_images = (images * (1-masks).float()) + masks
        stages1 = self.model(images, masks)
        outputs_merged = stages1 * masks + images * (1. - masks)

        image_per_row = 2
        images = stitch_images(
            self.postprocess(images),
            self.postprocess(masked_images),
            self.postprocess(stages1),
            # self.postprocess(outputs),
            self.postprocess(outputs_merged),
            img_per_row=image_per_row
        )
        path = os.path.join(self.sample_path, 'coarseinpainting')
        name = os.path.join(path, str(iteration).zfill(6)+".png")
        create_dir(path)
        print('\n saving sample' + name)
        images.save(name)

    def postprocess(self, img):
        img = (img + 1.) * 127.5
        img = img.permute(0, 2, 3, 1)
        return img.int()

    def cuda(self, *args):
        return (item.to(self.config.DEVICE) for item in args)


if __name__ == '__main__':
    config = Config("./configs/config.yaml")
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in config.GPU)
    config.DEVICE = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    model = CoarseModel(config)

    model.load()
    config.print()
    print('\nstart training...\n')
    model.train()
