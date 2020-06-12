from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torchvision import datasets, transforms
from PIL import Image

def image_load(path, size=512):
    image = Image.open(path)
    image = transforms.Compose([
        transforms.CenterCrop(min(image.size)),
        transforms.Resize((size, size)),
        transforms.ToTensor()
    ])(image).cuda()
    return image

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num-workers', type=int, default = 4)
    parser.add_argument('-e', '--epoch', type=int, default=40)
    parser.add_argument('-d', '--display-step', type=int, default = 10)
    parser.add_argument('--content', type=str, default='data/content.jpg')
    parser.add_argument('--style', type=str, default='data/style.jpg')
    opt = parser.parse_args()

    content_image = image_load(opt.content)
    style_image = image_load(opt.style)

    print(content_image.shape, style_image.shape)