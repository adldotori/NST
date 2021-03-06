import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import os.path as osp
import argparse
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from loss import *
from dataloader import *

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num-workers', type=int, default = 4)
    parser.add_argument('-e', '--epoch', type=int, default=20000)
    parser.add_argument('-d', '--display-epoch', type=int, default = 400)
    parser.add_argument('--content', type=str, default='data/content.jpg')
    parser.add_argument('--style', type=str, default='data/style.jpg')
    parser.add_argument('--result', type=str, default='sample/')
    opt = parser.parse_args()
    return opt

def imsave(result, path):
    image = result[0].to("cpu").clone().detach().numpy()
    image = np.transpose(image, (1, 2, 0))
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = (image * 255).astype('uint8')
    Image.fromarray(image).save(path)
    
def train(opt):
    # Load Dataset
    content_image = image_load(opt.content)
    style_image = image_load(opt.style)

    generate_image = torch.randn_like(content_image).requires_grad_(True)

    # Set Optimizer
    optim = torch.optim.Adam([generate_image], lr=0.01)

    # Set Loss
    loss = Loss(alpha=1, beta=1000)

    writer = SummaryWriter()


    if not osp.isdir(opt.result):
        os.makedirs(opt.result)

    for epoch in range(opt.epoch):
        optim.zero_grad()
        total_loss, c_loss, s_loss = loss(generate_image, content_image, style_image)
        total_loss.backward()
        optim.step()
        
        writer.add_scalar('loss/total', total_loss, epoch)
        writer.add_scalar('loss/content', c_loss, epoch)
        writer.add_scalar('loss/style', s_loss, epoch)
        
        if (epoch + 1) % opt.display_epoch == 0:
            writer.add_images('image', generate_image, epoch, dataformats="NCHW")
            print('[Epoch {}] Total : {:.2} | C_loss : {:.2} | S_loss : {:.2}'.format(epoch + 1, total_loss, c_loss, s_loss))
            imsave(generate_image, osp.join(opt.result, '{}.png'.format(epoch + 1)))
            imsave(content_image, 'content.png')
            imsave(style_image, 'style.png')

if __name__ == '__main__':
    opt = get_opt()
    train(opt)