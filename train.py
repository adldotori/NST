import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import argparse
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from loss import *
from dataloader import *

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num-workers', type=int, default = 4)
    parser.add_argument('-e', '--epoch', type=int, default=40)
    parser.add_argument('-d', '--display-step', type=int, default = 10)
    parser.add_argument('--content', type=str, default='data/content.jpg')
    parser.add_argument('--style', type=str, default='data/style.jpg')
    opt = parser.parse_args()
    return opt

def train(opt):
    # Init Model

    # Load Dataset
    content_image = image_load(opt.content)
    style_image = image_load(opt.style)

    # Set Optimizer
    optim = torch.optim.Adam(generator.parameters(), lr=0.0001)

    # Set Loss
    loss = Loss()

    writer = SummaryWriter()

    for epoch in range(opt.epoch):
        # load dataset only batch_size
        content_image = content_image.cuda()

        optim_gen.zero_grad()

        loss, c_loss, s_loss = loss(content_image, style_image, )
        loss.backward()
        optim.step()
        

        writer.add_scalar('loss/total', loss, step)
        writer.add_scalar('loss/content', c_loss, step)
        writer.add_scalar('loss/style', s_loss, step)
        
        if step % opt.display_step == 0:
            writer.add_images('image', image[0][0], step, dataformats="HW")
            writer.add_images('result', gen[0][0], step, dataformats="HW")

            print('[Epoch {}] Total : {:.2} | G_loss : {:.2} | D_loss : {:.2}'.format(epoch + 1, loss_gen+loss_dis, loss_gen, loss_dis))
            
            generator.eval()
            z = Variable(torch.randn(10, 100)).cuda()
            label = Variable(torch.arange(0,10)).cuda()
            label = make_one_hot(label, 10)
            sample_images = generator(z, label)
            grid = make_grid(sample_images, nrow=5, normalize=True)
            writer.add_image('sample_image', grid, step)

            torch.save(generator.state_dict(), 'checkpoint.pt')

if __name__ == '__main__':
    opt = get_opt()
    train(opt)