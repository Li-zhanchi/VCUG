
from turtle import forward
import torch
from torch import device, nn
import torch.nn.functional as F

from models.resnet import resnet101, resnet34
from models.getmodels import get_model, get_model_trained


class t_hognet(nn.Module):
    def __init__(self, fc1_channel, fc2_channel):
        super(t_hognet, self).__init__()
        self.fc1 = nn.Linear(56*56, fc1_channel)
        self.fc2 = nn.Linear(fc1_channel, fc2_channel)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.fc2(x)
        x = self.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)

        return x




# def get_hognet(num_classes, k):
#     # 用resnet作为
#     hognet, _, _, _  = get_model('resnet_singlechannel', num_classes, k)
#     hognet = nn.Sequential(*list(hognet.children())[:-1])


#     return hognet


class hcnet(nn.Module):
    def __init__(self, num_classes, k):
        super(hcnet, self).__init__()
        expansion = 4
        # self.cnnnet = get_model_trained(model_name='resnet101', num_classes=num_classes, k=k)
        self.cnnnet, _, _, _  = get_model('resnet101', num_classes, k)
        self.cnnnet = nn.Sequential(*list(self.cnnnet.children())[:-1])

        # # resnet版本的hog支线
        # self.hognet, _, _, _  = get_model('resnet_singlechannel', num_classes, k)
        # self.hognet = nn.Sequential(*list(self.hognet.children())[:-1])
        # self.fc = nn.Linear(512 * expansion *2, num_classes)
        
        # 自己设计的hog支线
        fc2_channel = 2048
        self.hognet = t_hognet(fc1_channel=2048, fc2_channel=fc2_channel)

        
        t_channel=2048
        self.fc1 = nn.Linear(512 * expansion+fc2_channel, t_channel)
        self.fc2 = nn.Linear(t_channel, num_classes)


    def forward(self, img, hog):
        feature_cnn = torch.flatten(self.cnnnet(img), 1)
        # # resnet_hog版本
        # feature_hog = torch.flatten(self.hognet(hog), 1)
        # feature = torch.cat((feature_cnn,feature_hog),1)
        # feature = self.fc(feature)

        # 自己设计的分支版本
        feature_hog = self.hognet(hog)
        feature = torch.cat((feature_cnn,feature_hog),1)
        feature = self.fc1(feature)
        feature = F.dropout(feature, p=0.2, training=self.training)
        feature = self.fc2(feature)

        return feature


def get_hcnet(num_classes, k, epochs=200, model_name_assign=None):
    net = hcnet(num_classes=num_classes, k=k)
    lr = 1e-5
    wd = 1e-3
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(epochs//3,40), gamma=0.2)
    if model_name_assign!=None:
        model_name_save=model_name_assign
    else:
        model_name_save = 'hcnet_{}'.format(k)
    
    return net, optimizer, scheduler, model_name_save