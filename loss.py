import torch
import torch.nn as nn
from torchvision import models

def gram_marix(x):
    N, C, H, W = x.size() 
    features = x.view(N * C, H * W)
    G = torch.mm(features, features.t())
    return G.div_(N * C * H * W)

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
        return loss

class Loss(nn.Module):
    def __init__(self, content_layers={13:1}, style_layers={1:0.2,6:0.2,11:0.2,20:0.2,29:0.2}, alpha=1, beta=100):
        super().__init__()
        self.vgg19 = models.vgg19(pretrained=True).cuda()
        for name, child in self.vgg19.named_children():
            print(name, child)
            if isinstance(child, nn.MaxPool2d):
                self.vgg19[int(name)] = nn.AvgPool2d(kernel_size=2, stride=2)
        # 0: 'conv1_1',
        # 5: 'conv2_1', 
        # 6: 'conv2_2', - content layer
        # 10: 'conv3_1',
        # 19: 'conv4_1',
        # 28: 'conv5_1'

        self.content_loss = ContentLoss()
        self.style_loss = StyleLoss()
        self.alpha = alpha
        self.beta = beta
        self.content_layers = content_layers
        self.style_layers = style_layers

        for param in self.vgg19.parameters():
            param.requires_grad = False

    def get_features(self, x, layers):
        features = {}
        for cnt, layer in enumerate(self.vgg19.features):
            x = layer(x)
            if cnt in layers:
                features[cnt] = x
        return features

    def forward(self, x, content, style):
        content = content.detach()
        style = style.detach()
        x_content = self.get_features(x, self.content_layers)
        x_style = self.get_features(x, self.style_layers)
        content = self.get_features(content, self.content_layers)
        style = self.get_features(style, self.style_layers)
        
        content_loss = 0
        for key, item in self.content_layers.items():
            content_loss += item * self.content_loss(x_content[key], content[key])
        
        style_loss = 0
        for key, item in self.style_layers.items():
            style_loss += item * self.style_loss(x_style[key], style[key])

        total_loss = content_loss * self.alpha + style_loss * self.beta

        return total_loss, content_loss, style_loss 

if __name__ == '__main__':
    x = torch.randn(1, 3, 256, 256).cuda()
    y = torch.randn(1, 3, 256, 256).cuda()
    z = torch.randn(1, 3, 256, 256).cuda()

    loss = Loss().cuda()
    loss = loss(x,y,z)
    print(loss)
