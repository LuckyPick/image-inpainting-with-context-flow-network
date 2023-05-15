from model.config import Config
from cfmodel import Cfmodel
import os
import torch


def main():
    config = Config("./configs/config.yaml")
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in config.GPU)
    config.DEVICE = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    model = Cfmodel(config)

    model.load()
    config.print()
    print('\nstart training...\n')
    model.train()


if __name__ == "__main__":
    main()
