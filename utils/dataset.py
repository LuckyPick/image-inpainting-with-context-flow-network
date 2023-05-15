import sys
import torch.utils.data as data
from os import listdir


from torch.utils.data.dataloader import DataLoader
from utils.tools import default_loader, is_image_file, normalize
import os
import torchvision.transforms as transforms


class MyDataset(data.Dataset):
    def __init__(self, config):
        # def __init__(self, data_path, mask_path, image_shape, mask_shape, with_subfolder=False, random_crop=True, return_name=False):
        super(MyDataset, self).__init__()
        data_path = config.data_path
        mask_path = config.mask_path
        image_shape = config.image_shape
        mask_shape = config.mask_shape
        with_subfolder = config.with_subfolder
        random_crop = config.random_crop
        return_name = config.return_name

        if with_subfolder:
            self.img = self._find_samples_in_subfolders(data_path)
            # self.mask = self._find_samples_in_subfolders(mask_path)
            self.mask = [x for x in listdir(mask_path) if is_image_file(x)]
        else:
            self.img = [x for x in listdir(data_path) if is_image_file(x)]
            self.mask = [x for x in listdir(mask_path) if is_image_file(x)]
        self.mask_path = mask_path
        self.data_path = data_path
        self.image_shape = image_shape[:-1]
        self.mask_shape = mask_shape[:-1]
        self.random_crop = random_crop
        self.return_name = return_name

    def __getitem__(self, index):
        path = os.path.join(self.data_path, self.img[index])
        img = default_loader(path)
        if index >= len(self.mask):
            mask_index = index % len(self.mask)
        else:
            mask_index = index
        path2 = os.path.join(self.mask_path, self.mask[mask_index])
        mask = default_loader(path2)

        if self.random_crop:
            imgw, imgh = img.size
            if imgh < self.image_shape[0] or imgw < self.image_shape[1]:
                img = transforms.Resize(min(self.image_shape))(img)
            img = transforms.RandomCrop(self.image_shape)(img)
        else:
            img = transforms.Resize(self.image_shape)(img)
            img = transforms.RandomCrop(self.image_shape)(img)

        img = transforms.ToTensor()(img)  # turn the image to a tensor
        img = normalize(img)

        if self.random_crop:
            maskw, maskh = mask.size
            if maskh < self.mask_shape[0] or maskw < self.mask_shape[1]:
                mask = transforms.Resize(min(self.mask_shape))(mask)
            mask = transforms.RandomCrop(self.mask_shape)(mask)
        else:
            mask = transforms.Resize(self.mask_shape)(mask)
            mask = transforms.RandomCrop(self.mask_shape)(mask)
        mask = transforms.ToTensor()(mask)
        mask = 1.0 - \
            mask[0].reshape([1, self.mask_shape[0], self.mask_shape[1]])
        if self.return_name:
            return self.img[index], img, self.mask[mask_index], mask
        else:
            return img, mask

    def _find_samples_in_subfolders(self, dir):
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(
                dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        samples = []
        for target in sorted(class_to_idx.keys()):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue
            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    if is_image_file(fname):
                        path = os.path.join(root, fname)
                        # item = (path, class_to_idx[target])
                        # samples.append(item)
                        samples.append(path)
        return samples

    def __len__(self):
        # print(f"mask_len{len(self.mask)}")
        return len(self.img)

    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True,
                shuffle=True
            )
            for item in sample_loader:
                yield item
