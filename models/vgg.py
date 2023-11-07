from pyexpat import model
import torch.nn as nn
import torch
import torchvision
import copy

import sys
sys.path.append("..")
from VCUG.models.cbam import CBAM, cbam_block
from VCUG.models.attention_block import se_block, eca_block

# official pretrain weights
model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'
}


class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=False):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        # N x 3 x 224 x 224
        x = self.features(x)
        # N x 512 x 7 x 7
        x = torch.flatten(x, start_dim=1)
        # N x 512*7*7
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_features(cfg: list):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg(model_name="vgg16", **kwargs):
    assert model_name in cfgs, "Warning: model number {} not in cfgs dict!".format(model_name)
    cfg = cfgs[model_name]

    model = VGG(make_features(cfg), **kwargs)
    return model


def get_torchvision_vgg16(pretrained, num_classes):
    model_vgg = torchvision.models.vgg16(pretrained=pretrained)
    # change fc layer structure
    model_vgg.classifier = nn.Sequential(*list(model_vgg.classifier.children())[:-1])
    model_vgg.classifier.add_module('final_linear', nn.Linear(in_features=4096,out_features=num_classes))
    return model_vgg


class torchvision_vgg16_attention(nn.Module):
    def __init__(self, pretrained, num_classes):
        super(torchvision_vgg16_attention, self).__init__()
        vgg = torchvision.models.vgg16(pretrained=pretrained)
        # self.attention_block = CBAM(in_channel=3)
        self.features = vgg.features
        self.attention_block = CBAM(in_channel=512)
        # self.attention_block = se_block(channel=512)
        # self.attention_block = cbam_block(channel=512)
        # self.attention_block = eca_block(channel=512)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(*list(vgg.classifier.children())[:-1])
        self.classifier.add_module('final_linear', nn.Linear(in_features=4096,out_features=num_classes))

    def forward(self, x):
        # x = self.attention_block(x)
        fea = self.features(x)
        fea = self.attention_block(fea)
        fea = self.avgpool(fea)
        fea = torch.flatten(fea, 1)
        cls = self.classifier(fea)
        return cls



class torchvision_vgg16_2branches(nn.Module):
    def __init__(self, num_classes_l, num_classes_r):
        super(torchvision_vgg16_2branches, self).__init__()
        vgg = torchvision.models.vgg16()
        self.features = copy.deepcopy(vgg.features)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier_l = copy.deepcopy(vgg.classifier)
        self.classifier_l = nn.Sequential(*list(self.classifier_l.children())[:-1])
        self.classifier_l.add_module('final_linear', nn.Linear(in_features=4096,out_features=num_classes_l))
        self.classifier_r = copy.deepcopy(vgg.classifier)
        self.classifier_r = nn.Sequential(*list(self.classifier_r.children())[:-1])
        self.classifier_r.add_module('final_linear', nn.Linear(in_features=4096, out_features=num_classes_r))

    def forward(self, x):
        fea = self.features(x)
        fea = self.avgpool(fea)
        fea = torch.flatten(fea, 1)
        cls_l = self.classifier_l(fea)
        cls_r = self.classifier_r(fea)
        return cls_l, cls_r