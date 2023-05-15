import torch
from torch import nn
import os
import torch.optim as optim
from . import networks
from . import loss


class BaseModel(nn.Module):
    def __init__(self, name, config):
        super(BaseModel, self).__init__()

        self.name = name
        self.config = config
        self.iteration = 0

        self.coarse_weights_path = os.path.join(
            config.PATH, name + '_coarse.pth')
        self.fine_weights_path = os.path.join(config.PATH, name + '_fine.pth')
        self.dis_weights_path = os.path.join(config.PATH, name + '_dis.pth')

    def load(self):
        if os.path.exists(self.coarse_weights_path):
            print('Loading %s coarse generator...' % self.name)

            if torch.cuda.is_available():
                data = torch.load(self.coarse_weights_path)
            else:
                data = torch.load(self.coarse_weights_path,
                                  map_location=lambda storage, loc: storage)

            self.coarse_gen.load_state_dict(data['coarse_gen'])
            # self.iteration = data['iteration']

        if os.path.exists(self.fine_weights_path):
            print('Loading %s fine generator...' % self.name)

            if torch.cuda.is_available():
                data = torch.load(self.fine_weights_path)
            else:
                data = torch.load(self.fine_weights_path,
                                  map_location=lambda storage, loc: storage)

            self.fine_gen.load_state_dict(data['fine_gen'])
            self.iteration = data['iteration']

        if os.path.exists(self.dis_weights_path):
            print('Loading %s discriminator...' % self.name)

            if torch.cuda.is_available():
                data = torch.load(self.dis_weights_path)
            else:
                data = torch.load(self.dis_weights_path,
                                  map_location=lambda storage, loc: storage)

            self.discriminator.load_state_dict(data['discriminator'])

    def save(self):
        print('\nsaving %s...\n' % self.name)
        torch.save({
            # 'iteration': self.iteration,
            'coarse_gen': self.coarse_gen.state_dict()
        }, self.coarse_weights_path)
        torch.save({
            'iteration': self.iteration,
            'fine_gen': self.fine_gen.state_dict()
        }, self.fine_weights_path)
        torch.save({
            'discriminator': self.discriminator.state_dict()
        }, self.dis_weights_path)


class InpaintingModel(BaseModel):
    def __init__(self, config):
        super().__init__('inpainting', config)
        coarse_gen = networks.CoarseGenerator(
            input_dim=3, cnum=64, use_cuda=config.use_cuda, )
        fine_gen = networks.FineGenerator(
            input_dim=3, cnum=64, use_cuda=config.use_cuda, )
        # generator = networks.Generator(config)
        discriminator = networks.Discriminator(
            in_channels=3, use_sigmoid=config.GAN_LOSS != 'hinge')

        if len(config.GPU) > 1:
            coarse_gen = nn.DataParallel(coarse_gen, config.GPU)
            fine_gen = nn.DataParallel(fine_gen, config.GPU)
            # generator = nn.DataParallel(generator, )
            discriminator = nn.DataParallel(discriminator, )

        l1_loss = nn.L1Loss()
        perceptual_loss = loss.PerceptualLoss()
        adversarial_loss = loss.AdversarialLoss(type=config.GAN_LOSS)
        style_loss = loss.StyleLoss()

        self.add_module('coarse_gen', coarse_gen)
        self.add_module('fine_gen', fine_gen)
        # self.add_module('generator', generator)
        self.add_module('discriminator', discriminator)
        self.add_module('l1_loss', l1_loss)
        self.add_module('perceptual_loss', perceptual_loss)
        self.add_module('style_loss', style_loss)
        self.add_module('adversarial_loss', adversarial_loss)

        self.coarse_optimizer = optim.Adam(
            params=coarse_gen.parameters(),
            lr=float(config.LR),
            betas=(config.BETA1, config.BETA2)
        )
        self.fine_optimizer = optim.Adam(
            params=fine_gen.parameters(),
            lr=float(config.LR),
            betas=(config.BETA1, config.BETA2)
        )
        self.dis_optimizer = optim.Adam(
            params=discriminator.parameters(),
            lr=float(config.LR) * float(config.D2G_LR),
            betas=(config.BETA1, config.BETA2)
        )

    def process(self, images, masks):
        self.iteration += 1

        # zero optimizers
        self.coarse_optimizer.zero_grad()
        self.fine_optimizer.zero_grad()
        self.dis_optimizer.zero_grad()

        # process
        stage1, outputs = self(images, masks)
        coarse_l1_loss = 0
        fine_loss = 0
        dis_loss = 0

        # coarse_gen loss
        coarse_l1_loss += self.l1_loss(stage1*masks, images*masks) * self.config.L1_LAMBDA + \
            self.l1_loss(stage1*(1-masks), images*(1-masks)) * \
            self.config.COARSE_L1_LOSS_WEIGHT
        # gen_loss += gen_coarse_l1_loss
        # discriminator loss
        dis_input_real = images
        dis_input_fake = outputs.detach()
        dis_real = self.discriminator(
            dis_input_real)                    # in: [rgb(3)]
        dis_fake = self.discriminator(
            dis_input_fake)                    # in: [rgb(3)]
        dis_real_loss = self.adversarial_loss(dis_real, True, True)
        dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
        dis_loss += (dis_real_loss + dis_fake_loss) / 2

        # fine_gen loss

        # adversarial loss
        gen_input_fake = outputs
        gen_fake = self.discriminator(
            gen_input_fake)                    # in: [rgb(3)]
        gen_gan_loss = self.adversarial_loss(
            gen_fake, True, False) * self.config.INPAINT_ADV_LOSS_WEIGHT
        fine_loss += gen_gan_loss

        # l1 loss
        gen_fine_l1_loss = self.l1_loss(outputs*masks, images*masks) * self.config.L1_LAMBDA + \
            self.l1_loss(outputs*(1-masks), images*(1-masks)) * \
            self.config.COARSE_L1_LOSS_WEIGHT
        fine_loss += gen_fine_l1_loss

        # perceptual loss
        fine_content_loss = self.perceptual_loss(
            outputs, images)*self.config.CONTENT_LOSS_WEIGHT
        fine_loss += fine_content_loss

        # style loss
        fine_style_loss = self.style_loss(
            outputs*masks, images*masks)*self.config.STYLE_LOSS_WEIGHT
        fine_loss += fine_style_loss

        # create logs
        logs = [
            ('l_gen_c_l1', coarse_l1_loss.item()),
            ('l_gen_f_l1', gen_fine_l1_loss.item()),
            ('l_gen_gan_l1', gen_gan_loss.item()),
            ('l_gen_con_l1', fine_content_loss.item()),
            ('l_gen_sty_l1', fine_style_loss.item()),
            ('l_dis', dis_loss.item())
        ]

        return outputs, stage1, coarse_l1_loss, fine_loss, dis_loss, logs

    def forward(self, images, masks):
        stage1 = self.coarse_gen(images, masks)
        outputs = self.fine_gen(images, stage1.detach(), masks)
        return stage1, outputs

    def backward(self, coarse_loss=None, fine_loss=None, dis_loss=None):
        coarse_loss.backward()
        self.coarse_optimizer.step()

        fine_loss.backward()
        self.fine_optimizer.step()

        dis_loss.backward()
        self.dis_optimizer.step()
