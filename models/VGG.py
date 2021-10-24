import torch
import torch.nn as nn

# class VGG13(nn.Module):
#     def __init__(self):
#         super(VGG13, self).__init__()
#         conv_blks = []
#         in_channels = 3

#         conv_arch = ((2, 64), (2, 128), (2, 256), (2, 512), (2, 512))

#         for (num_convs, out_channels) in conv_arch:
#             conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
#             in_channels = out_channels

#         self.net = nn.Sequential(
#                 *conv_blks, nn.Flatten(),
#                 # The fully-connected part
#                 nn.Linear(out_channels * 1 * 1, 256), nn.ReLU(), nn.Dropout(0.5),
#                 nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.5),
#                 nn.Linear(128, 100))


#     def vgg_block(num_convs, in_channels, out_channels):
#         layers = []
#         for _ in range(num_convs):
#             layers.append(
#                 nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
#             layers.append(nn.ReLU())
#             in_channels = out_channels
#         layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
#         return nn.Sequential(*layers)

#     def forward(self, X):
#         return self.net(X)
    
#     def display(self, X):
#         for blk in self.net:
#             X = blk(X)
#             print(blk.__class__.__name__, 'output shape:\t', X.shape)



import torch
import torch.nn as nn

cfg = {
    'A' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

class VGG(nn.Module):

    def __init__(self, features, num_class=100):
        super().__init__()
        self.features = features

        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_class)
        )

    def forward(self, x):
        output = self.features(x)
        output = output.view(output.size()[0], -1)
        output = self.classifier(output)
    
        return output

def make_layers(cfg, batch_norm=False):
    layers = []

    input_channel = 3
    for l in cfg:
        if l == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue

        layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]

        if batch_norm:
            layers += [nn.BatchNorm2d(l)]
        
        layers += [nn.ReLU(inplace=True)]
        input_channel = l
    
    return nn.Sequential(*layers)

def VGG11(num_class=100, bn=True):
    return VGG(make_layers(cfg['A'], batch_norm=bn), num_class=num_class)

def VGG13(num_class=100, bn=True):
    return VGG(make_layers(cfg['B'], batch_norm=bn), num_class=num_class)

def VGG16(num_class=100, bn=True):
    return VGG(make_layers(cfg['D'], batch_norm=bn), num_class=num_class)

def VGG19(num_class=100, bn=True):
    return VGG(make_layers(cfg['E'], batch_norm=bn), num_class=num_class)




# if __name__ == '__main__':
#     X = torch.randn(size=(1, 3, 32, 32))
#     vgg13 = VGG13()
#     vgg13.display(X)