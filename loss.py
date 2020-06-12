import torch
import torch.nn as nn
from torchvision import models

def gram_marix(x):
    N, C, H, W = x.size() 
    features = x.view(N, C, H * W)
    G = torch.bmm(features, features.transpose(1,2))
    return G

class VGG19(nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False 

    def forward(self, X):
        out1 = self.slice1(X)
        out2 = self.slice2(out1)
        out3 = self.slice3(out2)
        out4 = self.slice4(out3)
        out5 = self.slice5(out4)
        out = [out1, out2, out3, out4, out5]
        return out

class ContentLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        loss = (x - y) ** 2
        loss = torch.mean(loss) / 2
        return loss

class StyleLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        size = x.view(-1).shape[0]
        x = gram_marix(x)
        y = gram_marix(y)
        loss = (x - y) ** 2
        loss = torch.mean(loss)
        loss = loss / (4 * size ** 2)
        return loss

class Loss(nn.Module):
    def __init__(self, content_params={4:1}, style_params={1:0.2,2:0.2,3:0.2,4:0.2,5:0.2}, alpha=10, beta=40):
        super().__init__()
        self.vgg19 = VGG19()
        self.content_loss = ContentLoss()
        self.style_loss = StyleLoss()
        self.alpha = alpha
        self.beta = beta
        self.content_params = content_params
        self.style_params = style_params

    def forward(self, x, content, style):
        x = self.vgg19(x)
        content = self.vgg19(content)
        style = self.vgg19(style)
        
        content_loss = 0
        for key, item in self.content_params.items():
            content_loss += item * self.content_loss(x[key-1], content[key-1])
        
        style_loss = 0
        for key, item in self.style_params.items():
            style_loss += item * self.style_loss(x[key-1], style[key-1])

        total_loss = content_loss * self.alpha + style_loss * self.beta

        return total_loss, content_loss, style_loss 

if __name__ == '__main__':
    x = torch.randn(1, 3, 256, 256).cuda()
    y = torch.randn(1, 3, 256, 256).cuda()
    z = torch.randn(1, 3, 256, 256).cuda()

    loss = Loss().cuda()
    loss = loss(x,y,z)
    print(loss)
