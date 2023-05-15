from torch.utils.data import DataLoader
from utils.dataset import MyDataset
from model.model import InpaintingModel
import os
from utils.utils import Progbar, create_dir, stitch_images
from metrics import PSNR
import torch


class Cfmodel():
    def __init__(self, config):
        self.config = config
        self.inpainting_model = InpaintingModel(config).to(config.DEVICE)
        self.psnr = PSNR(255.0).to(config.DEVICE)
        self.train_dataset = MyDataset(config)
        self.sample_iterator = self.train_dataset.create_iterator(
            config.SAMPLE_SIZE)
        self.log_file = os.path.join(
            config.PATH, 'log_' + 'inpainting' + '.dat')
        self.sample_path = os.path.join(config.PATH, 'samples')

    def load(self):
        self.inpainting_model.load()

    def save(self):
        self.inpainting_model.save()

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
                self.inpainting_model.train()

                images, masks = self.cuda(*items)
                outputs, stage1, coarse_l1_loss, fine_loss, dis_loss, logs = self.inpainting_model.process(
                    images, masks)
                outputs_merged = (outputs*masks) + images*(1-masks)

                psnr = self.psnr(self.postprocess(images),
                                 self.postprocess(outputs_merged))
                mae = (torch.sum(torch.abs(images - outputs_merged)) /
                       torch.sum(images)).float()
                logs.append(('psnr', psnr.item()))
                logs.append(('mae', mae.item()))

                self.inpainting_model.backward(
                    coarse_l1_loss, fine_loss, dis_loss)
                iteration = self.inpainting_model.iteration

                if iteration >= max_iteration:
                    keep_training = False
                    break
                logs = [
                    ("epoch", epoch),
                    ("iter", iteration),
                ] + logs

                progbar.add(len(images), values=logs if self.config.VERBOSE else [
                            x for x in logs if not x[0].startswith('l_')])

                # log model at checkpoints
                if self.config.LOG_INTERVAL and iteration % self.config.LOG_INTERVAL == 0:
                    self.log(logs)

                # sample model at checkpoints
                if self.config.SAMPLE_INTERVAL and iteration % self.config.SAMPLE_INTERVAL == 0:
                    self.sample()

                # evaluate model at checkpoints
                # if self.config.EVAL_INTERVAL and iteration % self.config.EVAL_INTERVAL == 0:
                #     print('\nstart eval...\n')
                #     self.eval()

                # save model at checkpoints
                if self.config.SAVE_INTERVAL and iteration % self.config.SAVE_INTERVAL == 0:
                    self.save()
        print('\nEnd training....')

    def sample(self, it=None):
        self.inpainting_model.eval()

        items = next(self.sample_iterator)
        images, masks = self.cuda(*items)
        iteration = self.inpainting_model.iteration
        masked_images = (images * (1-masks).float()) + masks
        stages1, outputs = self.inpainting_model(images, masks)
        outputs_merged = outputs * masks + images * (1. - masks)

        image_per_row = 2
        images = stitch_images(
            self.postprocess(images),
            self.postprocess(masked_images),
            self.postprocess(stages1),
            self.postprocess(outputs),
            self.postprocess(outputs_merged),
            img_per_row=image_per_row
        )
        path = os.path.join(self.sample_path, 'inpainting')
        name = os.path.join(path, str(iteration).zfill(6)+".png")
        create_dir(path)
        print('\n saving sample' + name)
        images.save(name)

    def log(self, logs):
        with open(self.log_file, 'a') as f:
            f.write('%s\n' % ' '.join([str(item[1]) for item in logs]))

    def postprocess(self, img):
        # [-1,1] =>[0,255]
        # mean = [0.485, 0.456, 0.406]
        # std = [0.229, 0.224, 0.225]
        # img = self.unnorm(img, mean, std)
        # img *= 255.0
        # img = img.permute(0, 2, 3, 1)
        # return img.int()
        img = (img + 1.) * 127.5
        img = img.permute(0, 2, 3, 1)
        return img.int()

    def cuda(self, *args):
        return (item.to(self.config.DEVICE) for item in args)

    # def unnorm(self, tensor, mean, std):
    #     tensor = torch.split(tensor, 1, dim=0)
    #     temp = []
    #     for t, m, s in zip(tensor, mean, std):
    #         t = t * s + m
    #         temp.append(t)
    #     temp = torch.cat(temp, 0)
    #     return temp
