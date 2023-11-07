from pyexpat import model
import torch
import torch.nn as nn


from models.vgg import torchvision_vgg16_2branches, get_torchvision_vgg16
from models.resnet import resnet101, resnet34, resnet101_2branch, resnet_singlechannel
from models.mobilenet_v2 import MobileNetV2, MobileNetV2_2branch
from models.shufflenet import shufflenet_v2_x1_0, shufflenet_v2_x1_0_2branch
from models.densenet import densenet161, densenet161_twoBranch, load_state_dict_densenet
from models.efficientnet import efficientnet_b0, efficientnet_b0_2branch
from models.regnet import create_regnet, create_regnet_2branch
from models.efficientnetv2 import efficientnetv2_l_2branch, efficientnetv2_s, efficientnetv2_l
from models.googlenet import GoogLeNet_2branch,  GoogLeNet

import math


def load_weight(net, model_weight_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # weights_dict = torch.load(model_weight_path, map_location=device)
    weights_dict = torch.load(model_weight_path)
    load_weights_dict = {}
    for k, v in weights_dict.items():
        if k in  net.state_dict().keys():
            if net.state_dict()[k].numel() == v.numel():
                load_weights_dict[k]=v
    net.load_state_dict(load_weights_dict, strict=False)

    return net


def get_model(model_name, num_classes, k, epochs, step_szie, model_name_assign):
    net, optimizer, scheduler, model_name_save = None, None, None, None

    if model_name=="vgg16":
        pretrained = True
        net = get_torchvision_vgg16(pretrained=pretrained, num_classes=num_classes)
        lr = 1e-5
        wd = 1e-3
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_szie, gamma=0.1)
        model_name_save = 'vgg16_{}'.format(k)

    if model_name=="densenet161":
        net = densenet161(num_classes=num_classes)
        model_weight_path = "./pretrained/densenet161-8d451a50.pth"
        load_state_dict_densenet(net, model_weight_path)

        # net = densenet201(num_classes=num_classes).to(device)
        # model_weight_path = "./pretrained/densenet201-c1103571.pth"
        # model_name = 'densenet201'                                                   
        lr = 1e-5
        pg = [p for p in net.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(pg, lr=lr, weight_decay=1E-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_szie, gamma=0.1)
        model_name_save = 'densenet161_{}'.format(k)

    if model_name=="resnet101":
        net = resnet101()
        in_channel = net.fc.in_features
        net.fc = nn.Linear(in_channel, num_classes)
        model_weight_path = "./pretrained/resnet101-5d3b4d8f.pth"
        net = load_weight(net=net, model_weight_path=model_weight_path)

        # net = resnet34()
        # model_weight_path = "./pretrained/resnet34-pre.pth"
        # model_name = 'resnet34_{}'.format(k)

        lr = 1e-5
        wd = 1e-3
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_szie, gamma=0.1)
        model_name_save = 'resnet101_{}'.format(k)

    if model_name=="mobildenet_v2":
        net = MobileNetV2(num_classes=num_classes)
        model_weight_path = "./pretrained/mobilenet_v2-b0353104.pth"
        net = load_weight(net=net, model_weight_path=model_weight_path)
        
        lr = 1e-4
        wd = 1e-3
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_szie, gamma=0.1)
        model_name_save = 'mobildenet_v_{}'.format(k)

    if model_name=="shufflenet_v2_x1_0":
        net = shufflenet_v2_x1_0(num_classes=num_classes)
        model_weight_path = "./pretrained/shufflenetv2_x1-5666bf0f80.pth"
        net = load_weight(net=net, model_weight_path=model_weight_path)

        lrf = 0.1
        lr = 1e-3
        pg = [p for p in net.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(pg, lr=lr, weight_decay=1E-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_szie, gamma=0.1)
        model_name_save = 'shufflenet_v2_x1_0_{}'.format(k)

    # here
    if model_name=="efficientnet_b0":
        net = efficientnet_b0(num_classes=num_classes)
        model_weight_path = "./pretrained/efficientnetb0.pth"
        net = load_weight(net=net, model_weight_path=model_weight_path)

        lrf = 0.1
        lr = 1e-3
        pg = [p for p in net.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(pg, lr=lr, weight_decay=1E-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_szie, gamma=0.1)
        model_name_save = 'efficientnet_b0_{}'.format(k)

    if model_name=="RegNetY_400MF":
        net = create_regnet(model_name='RegNetY_400MF',
                            num_classes=num_classes)
        model_weight_path = "./pretrained/regnety_400mf.pth"
        net = load_weight(net=net, model_weight_path=model_weight_path)

        lrf = 0.01                                                      
        lr = 1e-3
        pg = [p for p in net.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(pg, lr=lr, weight_decay=1E-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_szie, gamma=0.1)
        model_name_save = 'RegNetY_400MF_{}'.format(k)

    if model_name=="efficientnetv2_l":
        net = efficientnetv2_l(num_classes=num_classes)
        model_weight_path = "./pretrained/pre_efficientnetv2-l.pth"
        net = load_weight(net=net, model_weight_path=model_weight_path)
        # net = efficientnetv2_s(num_classes=num_classes).to(device)
        # model_weight_path = "./pretrained/pre_efficientnetv2-s.pth"
        # model_name = 'efficientnetv2_s'

        lrf = 0.01                                                      
        lr = 1e-4
        pg = [p for p in net.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(pg, lr=lr, weight_decay=1E-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_szie, gamma=0.1)
        model_name_save = 'efficientnetv2_l_{}'.format(k)

    if model_name=="googlenet":
        net = GoogLeNet(num_classes=num_classes)
        model_weight_path = "./pretrained/googlenet-1378be20.pth"
        net = load_weight(net=net, model_weight_path=model_weight_path)
        lr = 1e-4
        wd = 1e-3
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_szie, gamma=0.2)
        model_name_save = 'googlenet_{}'.format(k)

    if model_name_assign!=None:
        model_name_save = model_name_assign

    return net, optimizer, scheduler, model_name_save



def get_model_without_pretrained(model_name, num_classes, k, epochs=200, step_szie=None, model_name_assign=None):
    net, optimizer, scheduler, model_name_save = None, None, None, None

    if model_name=="vgg16":
        pretrained = False
        net = get_torchvision_vgg16(pretrained=pretrained, num_classes=num_classes)
        lr = 1e-5
        wd = 1e-3
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_szie, gamma=0.2)
        model_name_save = 'vgg16_{}'.format(k)

    if model_name=="densenet161":
        net = densenet161(num_classes=num_classes)
        lrf = 0.1                                                      
        lr = 1e-3
        pg = [p for p in net.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(pg, lr=lr, momentum=0.9, weight_decay=1E-4, nesterov=True)
        lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - lrf) + lrf  # cosine
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
        model_name_save = 'densenet161_{}'.format(k)

    if model_name=="resnet101":
        net = resnet101()
        in_channel = net.fc.in_features
        net.fc = nn.Linear(in_channel, num_classes)
        lr = 1e-5
        wd = 1e-3
        if step_szie==None: step_szie=max(epochs//3,30)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_szie, gamma=0.2)
        model_name_save = 'resnet101_{}'.format(k)

    if model_name=="mobildenet_v2":
        net = MobileNetV2(num_classes=num_classes)
        lr = 1e-5
        wd = 1e-3
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_szie, gamma=0.2)
        model_name_save = 'mobildenet_v_{}'.format(k)

    if model_name=="shufflenet_v2_x1_0":
        net = shufflenet_v2_x1_0(num_classes=num_classes)
        lrf = 0.1
        lr = 1e-2
        pg = [p for p in net.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(pg, lr=lr, momentum=0.9, weight_decay=4E-5)
        lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - lrf) + lrf  # cosine
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
        model_name_save = 'shufflenet_v2_x1_0_{}'.format(k)

    # here
    if model_name=="efficientnet_b0":
        net = efficientnet_b0(num_classes=num_classes)
        lrf = 0.01                                                      
        lr = 1e-2
        pg = [p for p in net.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(pg, lr=lr, momentum=0.9, weight_decay=1E-4)
        lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - lrf) + lrf  # cosine
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
        model_name_save = 'efficientnet_b0_{}'.format(k)

    if model_name=="RegNetY_400MF":
        net = create_regnet(model_name='RegNetY_400MF',
                            num_classes=num_classes)
        lrf = 0.01                                                      
        lr = 1e-3
        pg = [p for p in net.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(pg, lr=lr, momentum=0.9, weight_decay=5E-5)
        lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - lrf) + lrf  # cosine
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
        model_name_save = 'RegNetY_400MF_{}'.format(k)

    if model_name=="efficientnetv2_l":
        net = efficientnetv2_l(num_classes=num_classes)
        lrf = 0.01                                                      
        lr = 1e-2
        pg = [p for p in net.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(pg, lr=lr, momentum=0.9, weight_decay=1E-4)
        lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - lrf) + lrf  # cosine
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
        model_name_save = 'efficientnetv2_l_{}'.format(k)

    if model_name=="googlenet":
        net = GoogLeNet(num_classes=num_classes)
        lr = 1e-5
        wd = 1e-3
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_szie, gamma=0.2)
        model_name_save = 'googlenet_{}'.format(k)

    if model_name_assign!=None:
        model_name_save = model_name_assign

    return net, optimizer, scheduler, model_name_save



def get_only_model(model_name, num_classes, model_weight_path=None):
    if model_name=="resnet101":
        net = resnet101()
    in_channel = net.fc.in_features
    net.fc = nn.Linear(in_channel, num_classes)
    if model_weight_path==None:
        model_weight_path = "./pretrained/resnet101-5d3b4d8f.pth"
    net = load_weight(net=net, model_weight_path=model_weight_path)

    return net



def get_model_trained_631(model_name, num_classes, model_weight_path_root="/home/tanzl/code/VCUG_retrain/result0720_631"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    if model_name == 'vgg16':
        net = get_torchvision_vgg16(pretrained=True, num_classes=num_classes)
        model_weight_path = "{}vgg16_last.pth".format(model_weight_path_root)
        net.load_state_dict(torch.load(model_weight_path, map_location=device))

    if model_name == "resnet101":
        net = resnet101()
        in_channel = net.fc.in_features
        net.fc = nn.Linear(in_channel, num_classes)
        model_weight_path = "{}resnet101_last.pth".format(model_weight_path_root)
        net.load_state_dict(torch.load(model_weight_path, map_location=device))

    if model_name == "resnet34":
        net = resnet34()
        in_channel = net.fc.in_features
        net.fc = nn.Linear(in_channel, num_classes)
        model_weight_path = "{}resnet34_last.pth".format(model_weight_path_root)
        net.load_state_dict(torch.load(model_weight_path, map_location=device))
    
    if model_name == "mobildenet_v2":
        net = MobileNetV2(num_classes=num_classes)
        model_weight_path = "{}mobildenet_v2_last.pth".format(model_weight_path_root)
        net.load_state_dict(torch.load(model_weight_path, map_location=device))

    if model_name == "shufflenet_v2_x1_0":
        net = shufflenet_v2_x1_0(num_classes=num_classes)
        model_weight_path = "{}shufflenet_v2_x1_0_last.pth".format(model_weight_path_root)
        net.load_state_dict(torch.load(model_weight_path, map_location=device))

    if model_name == "densenet161":
        net = densenet161(num_classes=num_classes)
        model_weight_path = "{}densenet161_last.pth".format(model_weight_path_root)
        net.load_state_dict(torch.load(model_weight_path, map_location=device))
    
    if model_name == "efficientnet_b0":
        net = efficientnet_b0(num_classes=num_classes)
        model_weight_path = "{}efficientnet_b0_last.pth".format(model_weight_path_root)
        net.load_state_dict(torch.load(model_weight_path, map_location=device))

    if model_name == "RegNetY_400MF":
        net = create_regnet(model_name='RegNetY_400MF',
                            num_classes=num_classes)
        model_weight_path = "{}RegNetY_400MF_last.pth".format(model_weight_path_root)
        net.load_state_dict(torch.load(model_weight_path, map_location=device))

    if model_name == "efficientnetv2_l":
        net = efficientnetv2_l(num_classes=num_classes)
        model_weight_path = "{}efficientnetv2_l_last.pth".format(model_weight_path_root)
        net.load_state_dict(torch.load(model_weight_path, map_location=device))

    if model_name == "googlenet":
            model_weight_path = "{}googlenet_last.pth".format(model_weight_path_root)
            net = GoogLeNet(num_classes=num_classes)
            net.load_state_dict(torch.load(model_weight_path, map_location=device))

    return net



def get_model_trained(model_name, num_classes, k, model_weight_path_root="/home/tanzl/code/VCUG/result0624_singelBranch/"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_name == 'vgg16':
        net = get_torchvision_vgg16(pretrained=True, num_classes=num_classes)
        model_weight_path = "{}vgg16_{}_last.pth".format(model_weight_path_root, k)
        net.load_state_dict(torch.load(model_weight_path, map_location=device))

    if model_name == "resnet101":
        net = resnet101()
        in_channel = net.fc.in_features
        net.fc = nn.Linear(in_channel, num_classes)
        model_weight_path = "{}resnet101_{}_last.pth".format(model_weight_path_root, k)
        net.load_state_dict(torch.load(model_weight_path, map_location=device))

    if model_name == "resnet34":
        net = resnet34()
        in_channel = net.fc.in_features
        net.fc = nn.Linear(in_channel, num_classes)
        model_weight_path = "{}resnet34_{}_last.pth".format(model_weight_path_root, k)
        net.load_state_dict(torch.load(model_weight_path, map_location=device))
    
    if model_name == "mobildenet_v2":
        net = MobileNetV2(num_classes=num_classes)
        model_weight_path = "{}mobildenet_v2_{}_last.pth".format(model_weight_path_root, k)
        net.load_state_dict(torch.load(model_weight_path, map_location=device))

    if model_name == "shufflenet_v2_x1_0":
        net = shufflenet_v2_x1_0(num_classes=num_classes)
        model_weight_path = "{}shufflenet_v2_x1_{}_last.pth".format(model_weight_path_root, k)
        net.load_state_dict(torch.load(model_weight_path, map_location=device))

    if model_name == "densenet161":
        net = densenet161(num_classes=num_classes)
        model_weight_path = "{}densenet161_{}_last.pth".format(model_weight_path_root, k)
        net.load_state_dict(torch.load(model_weight_path, map_location=device))
    
    if model_name == "efficientnet_b0":
        net = efficientnet_b0(num_classes=num_classes)
        model_weight_path = "{}efficientnet_b0_{}_last.pth".format(model_weight_path_root, k)
        net.load_state_dict(torch.load(model_weight_path, map_location=device))

    if model_name == "RegNetY_400MF":
        net = create_regnet(model_name='RegNetY_400MF',
                            num_classes=num_classes)
        model_weight_path = "{}RegNetY_400MF_{}_last.pth".format(model_weight_path_root, k)
        net.load_state_dict(torch.load(model_weight_path, map_location=device))

    if model_name == "efficientnetv2_l":
        net = efficientnetv2_l(num_classes=num_classes)
        model_weight_path = "{}efficientnetv2_l_{}_last.pth".format(model_weight_path_root, k)
        net.load_state_dict(torch.load(model_weight_path, map_location=device))

    if model_name == "googlenet":
            model_weight_path = "{}googlenet_{}_last.pth".format(model_weight_path_root, k)
            net = GoogLeNet(num_classes=num_classes)
            net.load_state_dict(torch.load(model_weight_path, map_location=device))

    if model_name == "resnet_svm":
        net = resnet101()
        in_channel = net.fc.in_features
        net.fc = nn.Linear(in_channel, num_classes)
        model_weight_path = "{}resnet101_svm_{}_last.pth".format(model_weight_path_root, k)
        net.load_state_dict(torch.load(model_weight_path, map_location=device))
    return net

def get_model_2branch(model_name, k, step_szie):
    num_classes_l = 6
    num_classes_r = 6

    if model_name=="vgg16_2branch":
        net = torchvision_vgg16_2branches(num_classes_l=num_classes_l, num_classes_r=num_classes_r)
        model_weight_path = "/home/tanzl/code/VCUG/result0624_singelBranch/vgg16_{}_last.pth".format(k)
        net = load_weight(net=net, model_weight_path=model_weight_path)
        lr = 1e-5
        wd = 1e-3
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_szie, gamma=0.2)
        model_name = 'gg16_2branch_{}'.format(k)


    if model_name=="densenet161_2branch":
        net = densenet161_twoBranch(num_classes_l=num_classes_l, num_classes_r=num_classes_r)
        model_weight_path = "/home/tanzl/code/VCUG/result0624_singelBranch/densenet161_{}_last.pth".format(k)
        net = load_weight(net=net, model_weight_path=model_weight_path)                                                  
        lr = 1e-5
        pg = [p for p in net.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(pg, lr=lr, weight_decay=1E-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_szie, gamma=0.1)
        model_name = 'densenet161_2branch_{}'.format(k)

    if model_name=="resnet101_2branch":
        net = resnet101_2branch(num_classes_l=num_classes_l, num_classes_r=num_classes_r)
        model_weight_path = "/home/tanzl/code/VCUG/result0624_singelBranch/resnet101_{}_last.pth".format(k)
        net = load_weight(net=net, model_weight_path=model_weight_path)
        lr = 1e-5
        wd = 1e-3
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_szie, gamma=0.2)
        model_name = 'resnet101_2branch_{}'.format(k)

    if model_name=="mobildenet_v2_2branch":
        net = MobileNetV2_2branch(num_classes_l=num_classes_l, num_classes_r=num_classes_r)
        model_weight_path = "/home/tanzl/code/VCUG/result0624_singelBranch/mobildenet_v2_{}_last.pth".format(k)
        net = load_weight(net=net, model_weight_path=model_weight_path)
        lr = 1e-4
        wd = 1e-3
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_szie, gamma=0.2)
        model_name = 'mobildenet_v2_2branch_{}'.format(k)

    if model_name=="shufflenet_v2_x1_2branch":
        net = shufflenet_v2_x1_0_2branch(num_classes_l=num_classes_l, num_classes_r=num_classes_r)
        model_weight_path = "/home/tanzl/code/VCUG/result0624_singelBranch/shufflenet_v2_x1_{}_last.pth".format(k)
        net = load_weight(net=net, model_weight_path=model_weight_path)

        lrf = 0.1
        lr = 1e-3
        pg = [p for p in net.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(pg, lr=lr, weight_decay=1E-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_szie, gamma=0.1)
        model_name = 'shufflenet_v2_x1_0_2branch_{}'.format(k)

    if model_name=="efficientnet_b0_2branch":
        net = efficientnet_b0_2branch(num_classes_l=num_classes_l, num_classes_r=num_classes_r)
        model_weight_path = "/home/tanzl/code/VCUG/result0624_singelBranch/efficientnet_b0_{}_last.pth".format(k)
        net = load_weight(net=net, model_weight_path=model_weight_path)

        lrf = 0.1
        lr = 1e-3
        pg = [p for p in net.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(pg, lr=lr, weight_decay=1E-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_szie, gamma=0.1)
        model_name = 'efficientnet_b0_2branch_{}'.format(k)

    if model_name=="RegNetY_400MF_2branch":
        net = create_regnet_2branch(model_name='RegNetY_400MF',
                            num_classes_l=num_classes_l, num_classes_r=num_classes_r)
        model_weight_path = "/home/tanzl/code/VCUG/result0624_singelBranch/RegNetY_400MF_{}_last.pth".format(k)
        net = load_weight(net=net, model_weight_path=model_weight_path)                                                     
        lr = 1e-3
        pg = [p for p in net.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(pg, lr=lr, weight_decay=1E-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_szie, gamma=0.1)
        model_name = 'RegNetY_400MF_2branch_{}'.format(k)

    if model_name=="efficientnetv2_l_2branch":
        net = efficientnetv2_l_2branch(num_classes_l=num_classes_l, num_classes_r=num_classes_r)
        model_weight_path = "/home/tanzl/code/VCUG/result0624_singelBranch/efficientnetv2_l_{}_last.pth".format(k)
        net = load_weight(net=net, model_weight_path=model_weight_path)

        lrf = 0.01                                                      
        lr = 1e-4
        pg = [p for p in net.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(pg, lr=lr, weight_decay=1E-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_szie, gamma=0.1)
        model_name = 'efficientnetv2_l_2branch_{}'.format(k)

    if model_name=="googlenet_2branch":
        net = GoogLeNet_2branch(num_classes_l=num_classes_l, num_classes_r=num_classes_r)
        model_weight_path = "/home/tanzl/code/VCUG/result0624_singelBranch/googlenet_{}_last.pth".format(k)
        net = load_weight(net=net, model_weight_path=model_weight_path)
        lr = 1e-5
        wd = 1e-3
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_szie, gamma=0.2)
        model_name = 'googlenet_2branch_{}'.format(k)

    return net, optimizer, scheduler, model_name


def get_model_2branch_631(model_name, step_szie):
    num_classes_l = 6
    num_classes_r = 6

    if model_name=="vgg16_2branch":
        net = torchvision_vgg16_2branches(num_classes_l=num_classes_l, num_classes_r=num_classes_r)
        model_weight_path = "/home/tanzl/code/VCUG_retrain/result0720_631/vgg16_last.pth"
        net = load_weight(net=net, model_weight_path=model_weight_path)
        lr = 1e-5
        wd = 1e-3
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_szie, gamma=0.2)
        model_name = 'vgg16_2branch'


    if model_name=="densenet161_2branch":
        net = densenet161_twoBranch(num_classes_l=num_classes_l, num_classes_r=num_classes_r)
        model_weight_path = "/home/tanzl/code/VCUG_retrain/result0720_631/densenet161_last.pth"
        net = load_weight(net=net, model_weight_path=model_weight_path)                                                  
        lr = 1e-5
        pg = [p for p in net.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(pg, lr=lr, weight_decay=1E-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_szie, gamma=0.1)
        model_name = 'densenet161_2branch'

    if model_name=="resnet101_2branch":
        net = resnet101_2branch(num_classes_l=num_classes_l, num_classes_r=num_classes_r)
        model_weight_path = "/home/tanzl/code/VCUG_retrain/result0720_631/resnet101_last.pth"
        net = load_weight(net=net, model_weight_path=model_weight_path)
        lr = 1e-5
        wd = 1e-3
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_szie, gamma=0.2)
        model_name = 'resnet101_2branch'

    if model_name=="mobildenet_v2_2branch":
        net = MobileNetV2_2branch(num_classes_l=num_classes_l, num_classes_r=num_classes_r)
        model_weight_path = "/home/tanzl/code/VCUG_retrain/result0720_631/mobildenet_v2_last.pth"
        net = load_weight(net=net, model_weight_path=model_weight_path)
        lr = 1e-4
        wd = 1e-3
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_szie, gamma=0.2)
        model_name = 'mobildenet_v2_2branch'

    if model_name=="shufflenet_v2_x1_2branch":
        net = shufflenet_v2_x1_0_2branch(num_classes_l=num_classes_l, num_classes_r=num_classes_r)
        model_weight_path = "/home/tanzl/code/VCUG_retrain/result0720_631/shufflenet_v2_x1_0_last.pth"
        net = load_weight(net=net, model_weight_path=model_weight_path)

        lrf = 0.1
        lr = 1e-3
        pg = [p for p in net.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(pg, lr=lr, weight_decay=1E-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_szie, gamma=0.1)
        model_name = 'shufflenet_v2_x1_0_2branch'

    if model_name=="efficientnet_b0_2branch":
        net = efficientnet_b0_2branch(num_classes_l=num_classes_l, num_classes_r=num_classes_r)
        model_weight_path = "/home/tanzl/code/VCUG_retrain/result0720_631/efficientnet_b0_last.pth"
        net = load_weight(net=net, model_weight_path=model_weight_path)

        lrf = 0.1
        lr = 1e-3
        pg = [p for p in net.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(pg, lr=lr, weight_decay=1E-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_szie, gamma=0.1)
        model_name = 'efficientnet_b0_2branch'

    if model_name=="RegNetY_400MF_2branch":
        net = create_regnet_2branch(model_name='RegNetY_400MF',
                            num_classes_l=num_classes_l, num_classes_r=num_classes_r)
        model_weight_path = "/home/tanzl/code/VCUG_retrain/result0720_631/RegNetY_400MF_last.pth"
        net = load_weight(net=net, model_weight_path=model_weight_path)                                                     
        lr = 1e-3
        pg = [p for p in net.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(pg, lr=lr, weight_decay=1E-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_szie, gamma=0.1)
        model_name = 'RegNetY_400MF_2branch'

    if model_name=="efficientnetv2_l_2branch":
        net = efficientnetv2_l_2branch(num_classes_l=num_classes_l, num_classes_r=num_classes_r)
        model_weight_path = "/home/tanzl/code/VCUG_retrain/result0720_631/efficientnetv2_l_last.pth"
        net = load_weight(net=net, model_weight_path=model_weight_path)

        lrf = 0.01                                                      
        lr = 1e-4
        pg = [p for p in net.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(pg, lr=lr, weight_decay=1E-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_szie, gamma=0.1)
        model_name = 'efficientnetv2_l_2branch'

    if model_name=="googlenet_2branch":
        net = GoogLeNet_2branch(num_classes_l=num_classes_l, num_classes_r=num_classes_r)
        model_weight_path = "/home/tanzl/code/VCUG_retrain/result0720_631/googlenet_last.pth"
        net = load_weight(net=net, model_weight_path=model_weight_path)
        lr = 1e-5
        wd = 1e-3
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_szie, gamma=0.2)
        model_name = 'googlenet_2branch'

    return net, optimizer, scheduler, model_name



def get_model_2branch_trained(model_name, k):
    num_classes_l = 6
    num_classes_r = 6
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_name=="vgg16_2branch":
        net = torchvision_vgg16_2branches(num_classes_l=num_classes_l, num_classes_r=num_classes_r)
        model_weight_path = "/home/tanzl/code/VCUG/result0629_doubleBranch/vgg16_2branch_{}_last.pth".format(k)
        net.load_state_dict(torch.load(model_weight_path, map_location=device))

    if model_name=="densenet161_2branch":
        net = densenet161_twoBranch(num_classes_l=num_classes_l, num_classes_r=num_classes_r)
        model_weight_path = "/home/tanzl/code/VCUG/result0629_doubleBranch/densenet161_2branch_{}_last.pth".format(k)
        net.load_state_dict(torch.load(model_weight_path, map_location=device))

    if model_name=="resnet101_2branch":
        net = resnet101_2branch(num_classes_l=num_classes_l, num_classes_r=num_classes_r)
        model_weight_path = "/home/tanzl/code/VCUG/result0629_doubleBranch/resnet101_2branch_{}_last.pth".format(k)
        net.load_state_dict(torch.load(model_weight_path, map_location=device))

    if model_name=="mobildenet_v2_2branch":
        net = MobileNetV2_2branch(num_classes_l=num_classes_l, num_classes_r=num_classes_r)
        model_weight_path = "/home/tanzl/code/VCUG/result0629_doubleBranch/mobildenet_v2_2branch_{}_last.pth".format(k)
        net.load_state_dict(torch.load(model_weight_path, map_location=device))

    if model_name=="shufflenet_v2_x1_0_2branch":
        net = shufflenet_v2_x1_0_2branch(num_classes_l=num_classes_l, num_classes_r=num_classes_r)
        model_weight_path = "/home/tanzl/code/VCUG/result0629_doubleBranch/shufflenet_v2_x1_0_2branch_{}_last.pth".format(k)
        net.load_state_dict(torch.load(model_weight_path, map_location=device))

    if model_name=="efficientnet_b0_2branch":
        net = efficientnet_b0_2branch(num_classes_l=num_classes_l, num_classes_r=num_classes_r)
        model_weight_path = "/home/tanzl/code/VCUG/result0629_doubleBranch/efficientnet_b0_2branch_{}_last.pth".format(k)
        net.load_state_dict(torch.load(model_weight_path, map_location=device))

    if model_name=="RegNetY_400MF_2branch":
        net = create_regnet_2branch(model_name='RegNetY_400MF',
                            num_classes_l=num_classes_l, num_classes_r=num_classes_r)
        model_weight_path = "/home/tanzl/code/VCUG/result0629_doubleBranch/RegNetY_400MF_2branch_{}_last.pth".format(k)
        net.load_state_dict(torch.load(model_weight_path, map_location=device))

    if model_name=="efficientnetv2_l_2branch":
        net = efficientnetv2_l_2branch(num_classes_l=num_classes_l, num_classes_r=num_classes_r)
        model_weight_path = "/home/tanzl/code/VCUG/result0629_doubleBranch/efficientnetv2_l_2branch_{}_last.pth".format(k)
        net.load_state_dict(torch.load(model_weight_path, map_location=device))

    if model_name=="googlenet_2branch":
        net = GoogLeNet_2branch(num_classes_l=num_classes_l, num_classes_r=num_classes_r)
        model_weight_path = "/home/tanzl/code/VCUG/result0629_doubleBranch/googlenet_2branch_{}_last.pth".format(k)
        net.load_state_dict(torch.load(model_weight_path, map_location=device))


    return net


def get_model_2branch_trained_631(model_name, model_weight_path_root):
    num_classes_l = 6
    num_classes_r = 6
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    if model_name=="vgg16_2branch":
        net = torchvision_vgg16_2branches(num_classes_l=num_classes_l, num_classes_r=num_classes_r)
        model_weight_path = model_weight_path_root + "/vgg16_2branch_last.pth"
        net.load_state_dict(torch.load(model_weight_path, map_location=device))

    if model_name=="densenet161_2branch":
        net = densenet161_twoBranch(num_classes_l=num_classes_l, num_classes_r=num_classes_r)
        model_weight_path = model_weight_path_root + "/densenet161_2branch_last.pth"
        net.load_state_dict(torch.load(model_weight_path, map_location=device))

    if model_name=="resnet101_2branch":
        net = resnet101_2branch(num_classes_l=num_classes_l, num_classes_r=num_classes_r)
        model_weight_path = model_weight_path_root + "/resnet101_2branch_last.pth"
        net.load_state_dict(torch.load(model_weight_path, map_location=device))

    if model_name=="mobildenet_v2_2branch":
        net = MobileNetV2_2branch(num_classes_l=num_classes_l, num_classes_r=num_classes_r)
        model_weight_path = model_weight_path_root + "/mobildenet_v2_2branch_last.pth"
        net.load_state_dict(torch.load(model_weight_path, map_location=device))

    if model_name=="shufflenet_v2_x1_0_2branch":
        net = shufflenet_v2_x1_0_2branch(num_classes_l=num_classes_l, num_classes_r=num_classes_r)
        model_weight_path = model_weight_path_root + "/shufflenet_v2_x1_0_2branch_last.pth"
        net.load_state_dict(torch.load(model_weight_path, map_location=device))

    if model_name=="efficientnet_b0_2branch":
        net = efficientnet_b0_2branch(num_classes_l=num_classes_l, num_classes_r=num_classes_r)
        model_weight_path = model_weight_path_root + "/efficientnet_b0_2branch_last.pth"
        net.load_state_dict(torch.load(model_weight_path, map_location=device))

    if model_name=="RegNetY_400MF_2branch":
        net = create_regnet_2branch(model_name='RegNetY_400MF',
                            num_classes_l=num_classes_l, num_classes_r=num_classes_r)
        model_weight_path = model_weight_path_root + "/RegNetY_400MF_2branch_last.pth"
        net.load_state_dict(torch.load(model_weight_path, map_location=device))

    if model_name=="efficientnetv2_l_2branch":
        net = efficientnetv2_l_2branch(num_classes_l=num_classes_l, num_classes_r=num_classes_r)
        model_weight_path = model_weight_path_root + "/efficientnetv2_l_2branch_last.pth"
        net.load_state_dict(torch.load(model_weight_path, map_location=device))

    if model_name=="googlenet_2branch":
        net = GoogLeNet_2branch(num_classes_l=num_classes_l, num_classes_r=num_classes_r)
        model_weight_path = model_weight_path_root + "/googlenet_2branch_last.pth"
        net.load_state_dict(torch.load(model_weight_path, map_location=device))
    return net




