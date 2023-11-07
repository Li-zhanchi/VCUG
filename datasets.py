import torch
from PIL import Image
import numpy as np
np.random.seed(2022)
import os
from torchvision import transforms
import random
random.seed(2022)
import cv2


def load_data_vcug_kSplit(vcug_dir, batch_size, k, label_model='label.txt'):
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    dataset_train = VCUGDataset(vcug_dir=vcug_dir,  txt='train_'+str(k)+'.txt', label_model=label_model)
    dataset_test = VCUGDataset(vcug_dir=vcug_dir,  txt='test_'+str(k)+'.txt', label_model=label_model)

    train_iter = torch.utils.data.DataLoader(
        dataset_train, batch_size,
        shuffle=True,
        num_workers=nw)
    test_iter = torch.utils.data.DataLoader(
        dataset_test, batch_size,
        shuffle=False,
        num_workers=nw)
    return train_iter, test_iter

class VCUGDataset(torch.utils.data.Dataset):
    def __init__(self, vcug_dir, txt, label_model='label.txt'):
        self.transform = transforms.Compose([transforms.Resize(512),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.451, 0.451, 0.451), (0.230, 0.230, 0.230)),
                                    ])
        self.vcug_dir = vcug_dir
        self.vcug_img = []

        with open(os.path.join(vcug_dir, txt), 'r') as f:
            for line in f.readlines():
                self.vcug_img.append(line.strip())
        self.label_model = label_model
        self.txt = txt

    def __getitem__(self, idx):
        img_path = os.path.join(self.vcug_dir, self.vcug_img[idx], self.vcug_img[idx] + '.png')
        img = Image.open(img_path).convert("RGB")
        img = add_square(img)
        img = self.transform(img)

        cls_path = os.path.join(self.vcug_dir, self.vcug_img[idx], self.label_model) 
        cls = []
        with open(cls_path, 'r') as f:
            for line in f.readlines():
                cls.append(line.strip())
        cls = int(cls[0])
        return img, cls

    def __len__(self):
        return len(self.vcug_img)

def load_data_vcug_1or2(vcug_dir, batch_size, label_model='label.txt', k=0):
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    dataset_train = VCUGDataset_1or2branch(vcug_dir=vcug_dir,  txt='train_'+str(k)+'.txt', label_model=label_model)
    dataset_test = VCUGDataset_1or2branch(vcug_dir=vcug_dir,  txt='test_'+str(k)+'.txt', label_model=label_model)

    train_iter = torch.utils.data.DataLoader(
        dataset_train, batch_size,
        shuffle=True,
        num_workers=nw)
    test_iter = torch.utils.data.DataLoader(
        dataset_test, batch_size,
        shuffle=False,
        num_workers=nw)
    return train_iter, test_iter

class VCUGDataset_1or2branch(torch.utils.data.Dataset):
    def __init__(self, vcug_dir, txt, label_model='label.txt'):
        # 给这样的模型提供数据集：对图像进行分类，判断是单分支还是双分支
        self.transform = transforms.Compose([transforms.Resize(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.451, 0.451, 0.451), (0.230, 0.230, 0.230))
                                    ])
        self.vcug_dir = vcug_dir
        self.vcug_img = []
        self.png_root = '/home/tanzl/data/VCUG/trainTest/'

        with open(os.path.join(vcug_dir, txt), 'r') as f:
            for line in f.readlines():
                self.vcug_img.append(line.strip())

        if 'train' in txt:
            self.balance_12()
        self.label_model = label_model
        self.txt = txt

    def __getitem__(self, idx):
        img_path = os.path.join(self.png_root, self.vcug_img[idx], self.vcug_img[idx].split('/')[-1] + '.png')
        img = Image.open(img_path).convert("RGB")
        img = add_square(img)
        img = self.transform(img)
        if 'single' in img_path:
            cls = 0
        else:
            cls = 1
        return img, cls

    def balance_12(self):
        vcug_img_1 = [img for img in self.vcug_img if 'single' in img]
        vcug_img_2 = [img for img in self.vcug_img if 'double' in img]
        print("处理前，1:{},2:{}".format(len(vcug_img_1),len(vcug_img_2)))


        # 将类别2上采样到类别1的方法
        num_samples = len(vcug_img_2)
        num_samples_diff = len(vcug_img_1) - num_samples
        up_sampled_vcug_img_2 = np.random.choice(vcug_img_2, size=num_samples_diff, replace=True)
        self.vcug_img.extend(up_sampled_vcug_img_2)


        # 将类别1下采样到类别2的方法
        # vcug_img_1 = np.random.choice(vcug_img_1, size=len(vcug_img_2), replace=False).tolist()
        # self.vcug_img = vcug_img_1 + vcug_img_2


        vcug_img_1 = [img for img in self.vcug_img if 'single' in img]
        vcug_img_2 = [img for img in self.vcug_img if 'double' in img]
        print("处理前，1:{},2:{}".format(len(vcug_img_1),len(vcug_img_2)))

    def __len__(self):
        return len(self.vcug_img)


def load_data_vcug_1or2_631(vcug_dir, batch_size, label_model='label.txt'):
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    dataset_train = VCUGDataset_1or2branch_631(vcug_dir=vcug_dir,  txt='split_6.txt', label_model=label_model)
    dataset_val = VCUGDataset_1or2branch_631(vcug_dir=vcug_dir,  txt='split_3.txt', label_model=label_model)
    dataset_test = VCUGDataset_1or2branch_631(vcug_dir=vcug_dir,  txt='split_1.txt', label_model=label_model)


    train_iter = torch.utils.data.DataLoader(
        dataset_train, batch_size,
        shuffle=True,
        num_workers=nw)
    val_iter = torch.utils.data.DataLoader(
        dataset_val, batch_size,
        shuffle=False,
        num_workers=nw)
    test_iter = torch.utils.data.DataLoader(
        dataset_test, batch_size,
        shuffle=False,
        num_workers=nw)
    return train_iter, val_iter, test_iter

class VCUGDataset_1or2branch_631(torch.utils.data.Dataset):
    def __init__(self, vcug_dir, txt, label_model='label.txt'):
        # 给这样的模型提供数据集：对图像进行分类，判断是单分支还是双分支
        self.transform = transforms.Compose([transforms.Resize(512),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.451, 0.451, 0.451), (0.230, 0.230, 0.230))
                                    ])
        self.vcug_dir = vcug_dir
        self.vcug_img = []
        self.png_root = '/home/tanzl/data/VCUG/trainTest/'

        with open(os.path.join(vcug_dir, txt), 'r') as f:
            for line in f.readlines():
                self.vcug_img.append(line.strip())

        if txt=='split_6.txt':
            self.balance_12()
        self.label_model = label_model
        self.txt = txt



        # augment操作
        degrees = 20
        translate=(0, 0.1)
        scale=(0.8, 1)
        fillcolor = (0, 0, 0)
        self.trans_RandomAffine =  transforms.RandomAffine(degrees=degrees, translate=translate, scale=scale, fill=fillcolor)

        distortion_scale = 0.4
        p = 1
        fillcolor = (0, 0, 0)
        self.RandomPerspective = transforms.RandomPerspective(distortion_scale=distortion_scale, p=p, fill=fillcolor)

        self.RandomAdjustSharpness = transforms.RandomAdjustSharpness(sharpness_factor=8,p=1)  

        self.trans_flip = transforms.RandomHorizontalFlip(p=1)


    def __getitem__(self, idx):
        img_path = os.path.join(self.png_root, self.vcug_img[idx], self.vcug_img[idx].split('/')[-1] + '.png')
        img = Image.open(img_path).convert("RGB")
        img = remove_black_border(img)  # 去除黑边
        img = add_circle_mask(img)   # 进行圆形的掩膜
        if (self.txt=='split_6.txt') or (self.txt=='split_all.txt'):
            img = random_aspect_ratio_crop(img, 0.75, 1.3333)  # 进行随机长宽比的裁剪

        img = add_square(img)
        img = self.transform(img)

        if (self.txt=='split_6.txt') or (self.txt=='split_all.txt'):
            thred = 0.5
            if random.random() > thred:
                img = self.trans_RandomAffine(img)
            if random.random() > thred:
                img = self.RandomPerspective(img)
            if random.random() > 0.7:
                img = self.RandomAdjustSharpness(img)
            if random.random() > thred:
                img = self.trans_flip(img)


        if 'single' in img_path:
            cls = 0
        else:
            cls = 1
        return img, cls

    def balance_12(self):
        vcug_img_1 = [img for img in self.vcug_img if 'single' in img]
        vcug_img_2 = [img for img in self.vcug_img if 'double' in img]
        print("处理前，1:{},2:{}".format(len(vcug_img_1),len(vcug_img_2)))


        # 将类别2上采样到类别1的方法
        num_samples = len(vcug_img_2)
        num_samples_diff = len(vcug_img_1) - num_samples
        up_sampled_vcug_img_2 = np.random.choice(vcug_img_2, size=num_samples_diff, replace=True)
        self.vcug_img.extend(up_sampled_vcug_img_2)


        # 将类别1下采样到类别2的方法
        # vcug_img_1 = np.random.choice(vcug_img_1, size=len(vcug_img_2), replace=False).tolist()
        # self.vcug_img = vcug_img_1 + vcug_img_2

        vcug_img_1 = [img for img in self.vcug_img if 'single' in img]
        vcug_img_2 = [img for img in self.vcug_img if 'double' in img]
        print("处理后，1:{},2:{}".format(len(vcug_img_1),len(vcug_img_2)))

    def __len__(self):
        return len(self.vcug_img)



def load_data_vcug2branch_kSplit(vcug_dir, batch_size, k, label_model='label.txt'):
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    dataset_train = VCUGDataset_2branch(vcug_dir=vcug_dir,  txt='train_'+str(k)+'.txt', label_model=label_model)
    dataset_test = VCUGDataset_2branch(vcug_dir=vcug_dir,  txt='test_'+str(k)+'.txt', label_model=label_model)

    train_iter = torch.utils.data.DataLoader(
        dataset_train, batch_size,
        shuffle=True,
        num_workers=nw)
    test_iter = torch.utils.data.DataLoader(
        dataset_test, batch_size,
        shuffle=False,
        num_workers=nw)
    return train_iter, test_iter

class VCUGDataset_2branch(torch.utils.data.Dataset):
    def __init__(self, vcug_dir, txt, label_model='label.txt'):
        self.transform = transforms.Compose([transforms.Resize(512),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.451, 0.451, 0.451), (0.230, 0.230, 0.230)),
                                    ])
        self.vcug_dir = vcug_dir
        self.vcug_img = []

        with open(os.path.join(vcug_dir, txt), 'r') as f:
            for line in f.readlines():
                self.vcug_img.append(line.strip())
        self.label_model = label_model
        self.txt = txt

    def __getitem__(self, idx):
        img_path = os.path.join(self.vcug_dir, self.vcug_img[idx], self.vcug_img[idx] + '.png')
        img = Image.open(img_path).convert("RGB")
        img = add_square(img)
        img = self.transform(img)

        cls_path = os.path.join(self.vcug_dir, self.vcug_img[idx], self.label_model) 
        cls = []
        with open(cls_path, 'r') as f:
            for line in f.readlines():
                cls.append(line.strip())
        cls_l = int(cls[0])
        cls_r = int(cls[1])
        return img, cls_l, cls_r

    def __len__(self):
        return len(self.vcug_img)

def load_data_vcug_out(vcug_dir_single, vcug_dir_double, batch_size, label_model='label.txt'):
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    dataset_single = VCUGDataset_out_single(vcug_dir=vcug_dir_single, label_model=label_model)
    dataset_double = VCUGDataset_out_double(vcug_dir=vcug_dir_double, label_model=label_model)
    single_iter = torch.utils.data.DataLoader(
        dataset_single, batch_size,
        shuffle=False,
        num_workers=nw)
    double_iter = torch.utils.data.DataLoader(
        dataset_double, batch_size,
        shuffle=False,
        num_workers=nw)
    return single_iter, double_iter


def load_data_vcug_631(vcug_dir, batch_size, label_model='label.txt'):
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    dataset_train = VCUGDataset_631(vcug_dir=vcug_dir,  txt='split_6.txt', label_model=label_model)
    dataset_val = VCUGDataset_631(vcug_dir=vcug_dir,  txt='split_3.txt', label_model=label_model)
    dataset_test = VCUGDataset_631(vcug_dir=vcug_dir,  txt='split_1.txt', label_model=label_model)

    # train_iter = torch.utils.data.DataLoader(
    #     dataset_train, batch_size,
    #     shuffle=True,
    #     num_workers=nw)

    train_iter = torch.utils.data.DataLoader(
        dataset_train, batch_size,
        shuffle=False,
        num_workers=nw)


    val_iter = torch.utils.data.DataLoader(
        dataset_val, batch_size,
        shuffle=False,
        num_workers=nw)
    test_iter = torch.utils.data.DataLoader(
        dataset_test, batch_size,
        shuffle=False,
        num_workers=nw)
    return train_iter, val_iter, test_iter


def load_data_vcug_631_all(vcug_dir, batch_size, label_model='label.txt'):
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    dataset_all = VCUGDataset_631(vcug_dir=vcug_dir,  txt='split_all.txt', label_model=label_model)

    all_iter = torch.utils.data.DataLoader(
        dataset_all, batch_size,
        shuffle=True,
        num_workers=nw)
    return all_iter


class VCUGDataset_631(torch.utils.data.Dataset):
    def __init__(self, vcug_dir, txt, label_model='label.txt'):
        self.transform = transforms.Compose([transforms.Resize(512),  # 注意
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.451, 0.451, 0.451), (0.230, 0.230, 0.230)),
                                    ])
        self.vcug_dir = vcug_dir
        self.vcug_img = []

        with open(os.path.join(vcug_dir, txt), 'r') as f:
            for line in f.readlines():
                self.vcug_img.append(line.strip())
        self.label_model = label_model
        self.txt = txt

        # augment操作
        degrees = 20
        translate=(0, 0.1)
        scale=(0.8, 1)
        fillcolor = (0, 0, 0)
        self.trans_RandomAffine =  transforms.RandomAffine(degrees=degrees, translate=translate, scale=scale, fill=fillcolor)

        distortion_scale = 0.4
        p = 1
        fillcolor = (0, 0, 0)
        self.RandomPerspective = transforms.RandomPerspective(distortion_scale=distortion_scale, p=p, fill=fillcolor)

        self.RandomAdjustSharpness = transforms.RandomAdjustSharpness(sharpness_factor=8,p=1)  

        self.trans_flip = transforms.RandomHorizontalFlip(p=1)



    def __getitem__(self, idx):
        img_path = os.path.join(self.vcug_dir, self.vcug_img[idx], self.vcug_img[idx] + '.png')
        img = Image.open(img_path).convert("RGB")
        img = remove_black_border(img)  # 去除黑边
        img = add_circle_mask(img)   # 进行圆形的掩膜
        if (self.txt=='split_6.txt') or (self.txt=='split_all.txt'):
            img = random_aspect_ratio_crop(img, 0.75, 1.3333)  # 进行随机长宽比的裁剪
        
        img = add_square(img)   # 添加正方形的外界框
        img = self.transform(img)

        if (self.txt=='split_6.txt') or (self.txt=='split_all.txt'):
            thred = 0.5
            if random.random() > thred:
                img = self.trans_RandomAffine(img)
            if random.random() > thred:
                img = self.RandomPerspective(img)
            if random.random() > 0.7:
                img = self.RandomAdjustSharpness(img)
            if random.random() > thred:
                img = self.trans_flip(img)


        cls_path = os.path.join(self.vcug_dir, self.vcug_img[idx], self.label_model) 
        cls = []
        with open(cls_path, 'r') as f:
            for line in f.readlines():
                cls.append(line.strip())
        cls = int(cls[0])
        return img, cls

    def __len__(self):
        return len(self.vcug_img)


def load_data_vcug_2branch_631(vcug_dir, batch_size, label_model='label.txt'):
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    dataset_train = VCUGDataset_2branch_631(vcug_dir=vcug_dir,  txt='split_6.txt', label_model=label_model)
    dataset_val = VCUGDataset_2branch_631(vcug_dir=vcug_dir,  txt='split_3.txt', label_model=label_model)
    dataset_test = VCUGDataset_2branch_631(vcug_dir=vcug_dir,  txt='split_1.txt', label_model=label_model)

    # train_iter = torch.utils.data.DataLoader(
    #     dataset_train, batch_size,
    #     shuffle=True,
    #     num_workers=nw)

    train_iter = torch.utils.data.DataLoader(
        dataset_train, batch_size,
        shuffle=False,
        num_workers=nw)

    val_iter = torch.utils.data.DataLoader(
        dataset_val, batch_size,
        shuffle=False,
        num_workers=nw)
    test_iter = torch.utils.data.DataLoader(
        dataset_test, batch_size,
        shuffle=False,
        num_workers=nw)
    return train_iter, val_iter, test_iter


def load_data_vcug_2branch_631_all(vcug_dir, batch_size, label_model='label.txt'):
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    dataset_all = VCUGDataset_2branch_631(vcug_dir=vcug_dir,  txt='split_all.txt', label_model=label_model)

    all_iter = torch.utils.data.DataLoader(
        dataset_all, batch_size,
        shuffle=True,
        num_workers=nw)
    return all_iter



class VCUGDataset_2branch_631(torch.utils.data.Dataset):
    def __init__(self, vcug_dir, txt, label_model='label.txt'):
        self.transform = transforms.Compose([transforms.Resize(512),  # 注意
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.451, 0.451, 0.451), (0.230, 0.230, 0.230)),
                                    ])
        self.vcug_dir = vcug_dir
        self.vcug_img = []

        with open(os.path.join(vcug_dir, txt), 'r') as f:
            for line in f.readlines():
                self.vcug_img.append(line.strip())
        self.label_model = label_model
        self.txt = txt

        # augment操作
        degrees = 20
        translate=(0, 0.1)
        scale=(0.8, 1)
        fillcolor = (0, 0, 0)
        self.trans_RandomAffine =  transforms.RandomAffine(degrees=degrees, translate=translate, scale=scale, fill=fillcolor)

        distortion_scale = 0.4
        p = 1
        fillcolor = (0, 0, 0)
        self.RandomPerspective = transforms.RandomPerspective(distortion_scale=distortion_scale, p=p, fill=fillcolor)
        self.RandomAdjustSharpness = transforms.RandomAdjustSharpness(sharpness_factor=8,p=1)  
        self.trans_flip = transforms.RandomHorizontalFlip(p=1)


    def __getitem__(self, idx):
        img_path = os.path.join(self.vcug_dir, self.vcug_img[idx], self.vcug_img[idx] + '.png')
        img = Image.open(img_path).convert("RGB")
        img = remove_black_border(img)  # 去除黑边
        img = add_circle_mask(img)   # 进行圆形的掩膜
        if (self.txt=='split_6.txt') or (self.txt=='split_all.txt'):
            img = random_aspect_ratio_crop(img, 0.75, 1.3333)  # 进行随机长宽比的裁剪
        
        img = add_square(img)   # 添加正方形的外界框
        img = self.transform(img)

        if (self.txt=='split_6.txt') or (self.txt=='split_all.txt'):
            thred = 0.5
            if random.random() > thred:
                img = self.trans_RandomAffine(img)
            if random.random() > thred:
                img = self.RandomPerspective(img)
            if random.random() > 0.7:
                img = self.RandomAdjustSharpness(img)



        cls_path = os.path.join(self.vcug_dir, self.vcug_img[idx], self.label_model) 
        cls = []
        with open(cls_path, 'r') as f:
            for line in f.readlines():
                cls.append(line.strip())
        cls_l = int(cls[0])
        cls_r = int(cls[1])

        if (self.txt=='split_6.txt') or (self.txt=='split_all.txt'):  # 如果进行翻转的话，坐标也要翻转
            if random.random() > thred:
                img = self.trans_flip(img)
                temp = cls_l
                cls_l = cls_r
                cls_r = temp
        return img, cls_l, cls_r

    def __len__(self):
        return len(self.vcug_img)



class VCUGDataset_out_single(torch.utils.data.Dataset):
    def __init__(self, vcug_dir, label_model='label.txt'):
        self.transform = transforms.Compose([transforms.Resize(512),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.451, 0.451, 0.451), (0.230, 0.230, 0.230)),
                                    ])
        self.vcug_dir = vcug_dir
        self.vcug_img = []
        for idx in os.listdir(self.vcug_dir):
            self.vcug_img.append(idx)
        self.label_model = label_model

    def __getitem__(self, idx):
        img_path = os.path.join(self.vcug_dir, self.vcug_img[idx], self.vcug_img[idx] + '.png')
        img = Image.open(img_path).convert("RGB")
        img = add_square(img)
        img = self.transform(img)

        cls_path = os.path.join(self.vcug_dir, self.vcug_img[idx], self.label_model) 
        cls = []
        with open(cls_path, 'r') as f:
            for line in f.readlines():
                cls.append(line.strip())
        cls = int(cls[0])
        return img, cls, os.path.join(self.vcug_dir, self.vcug_img[idx])

    def __len__(self):
        return len(self.vcug_img)

class VCUGDataset_out_double(torch.utils.data.Dataset):
    def __init__(self, vcug_dir, label_model='label.txt'):
        self.transform = transforms.Compose([transforms.Resize(512),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.451, 0.451, 0.451), (0.230, 0.230, 0.230)),
                                    ])
        self.vcug_dir = vcug_dir
        self.vcug_img = []
        for idx in os.listdir(self.vcug_dir):
            self.vcug_img.append(idx)
        self.label_model = label_model

    def __getitem__(self, idx):
        img_path = os.path.join(self.vcug_dir, self.vcug_img[idx], self.vcug_img[idx] + '.png')
        img = Image.open(img_path).convert("RGB")
        img = add_square(img)
        img = self.transform(img)

        cls_path = os.path.join(self.vcug_dir, self.vcug_img[idx], self.label_model) 
        cls = []
        with open(cls_path, 'r') as f:
            for line in f.readlines():
                cls.append(line.strip())
        cls_l = int(cls[0])
        cls_r = int(cls[1])
        return img, cls_l, cls_r, os.path.join(self.vcug_dir, self.vcug_img[idx])

    def __len__(self):
        return len(self.vcug_img)

class VCUGDataset_randomTrans(torch.utils.data.Dataset):
    def __init__(self, vcug_dir, txt='train.txt', label_model='label_oneLabel.txt'):
        # 对数据进行随机的transform的数据集
        if 'train' in txt:
            transform_list = [transforms.Resize(224)]
            transform_list.append(transforms.ToTensor())
            transform_list.append(transforms.Normalize((0.451, 0.451, 0.451), (0.230, 0.230, 0.230)))
            

            self.transform = transforms.Compose(transform_list)
        else:
            self.transform = transforms.Compose([transforms.Resize(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.451, 0.451, 0.451), (0.230, 0.230, 0.230))
                                        ])
        self.vcug_dir = vcug_dir
        self.vcug_img = []
        with open(os.path.join(vcug_dir, txt), 'r') as f:
            for line in f.readlines():
                self.vcug_img.append(line.strip())
        self.label_model = label_model

        # transform操作
        degrees = 20
        translate=(0, 0.1)
        scale=(0.8, 1)
        fillcolor = (0, 0, 0)
        self.trans_RandomAffine =  transforms.RandomAffine(degrees=degrees, translate=translate, scale=scale, fill=fillcolor)
        
        distortion_scale = 0.4
        p = 1
        fillcolor = (0, 0, 0)
        self.RandomPerspective = transforms.RandomPerspective(distortion_scale=distortion_scale, p=p, fill=fillcolor)
        
        # transform_list.append(transforms.RandomInvert(p=1)) #奇怪的画风    
        self.RandomAdjustSharpness = transforms.RandomAdjustSharpness(sharpness_factor=20,p=1)  #这个可以有，可以训练集和测试集都用

        self.txt = txt

    def __getitem__(self, idx):
        idx = idx % (len(self.vcug_img)-1)
        img_path = os.path.join(self.vcug_dir, self.vcug_img[idx], self.vcug_img[idx] + '_img.png')
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        # 进行随机的transform
        if 'train' in self.txt:
            thred = 0.5
            if random.random() > thred:
                img = self.trans_RandomAffine(img)
            if random.random() > thred:
                img = self.RandomPerspective(img)
            if random.random() > thred:
                img = self.RandomAdjustSharpness(img)

        cls_path = os.path.join(self.vcug_dir, self.vcug_img[idx], self.label_model)
        cls = []
        with open(cls_path, 'r') as f:
            for line in f.readlines():
                cls.append(line.strip())
        cls_l = int(cls[0])
        cls_r = int(cls[1]) if len(cls) == 2 else 404
        
        return img, cls_l, cls_r

    def __len__(self):
        return len(self.vcug_img)*9

def collate_fn(batch):
    return tuple(zip(*batch))

def add_square(pil_file):
    # 将原图用一个最小的正方形框柱，不够的用黑色填充
    w, h = pil_file.size
    target = max(w,h)
    image1 = Image.new("RGB", (target, target))
    if h >= w:
        pad_w = int((target - w) / 2)
        image1.paste(pil_file,(pad_w, 0))
    else:
        pad_h = int((target - h) / 2)
        image1.paste(pil_file, (0, pad_h))
    return image1

def add_square_mask(pil_file, box, ratio = None):

    if ratio!=None:
        # # 对box进行扩大一点范围
        center_x = (box[0]+box[2])/2
        center_y = (box[1]+box[3])/2
        
        max_w, max_h = pil_file.size
        box[0] = max(int(center_x + (box[0]-center_x)*(1+ratio)), 0)
        box[1] = max(int(center_y + (box[1]-center_y)*(1+ratio)), 0)
        box[2] = min(int(center_x + (box[2]-center_x)*(1+ratio)), max_w)
        box[3] = min(int(center_y + (box[3]-center_y)*(1+ratio)), max_h)

    # 将box作为一个掩膜将原图中的抠出来
    w, h = pil_file.size
    image_background = Image.new("RGB", (w, h))
    
    pil_file = pil_file.crop(box=box)

    image_background.paste(pil_file,(box[0], box[1]))

    return image_background

def analyse_data(vcug_dir):
    cls = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0}
    vcug_img = []
    with open(os.path.join(vcug_dir, txt), 'r') as f:
        for line in f.readlines():
            vcug_img.append(line.strip())
    for idx in range(len(vcug_img)):
        cls_path = os.path.join(vcug_dir, vcug_img[idx], 'label.txt')
        single_cls = []
        with open(cls_path, 'r') as f:
            for line in f.readlines():
                single_cls.append(line.strip())
        single_cls = int(single_cls[0])
        cls[single_cls] = cls[single_cls] + 1
    print(cls)

def read_img_to_tensor_RGB_resize(path, crop_size=None):
    if crop_size==None:
        img_resize = Image.open(path)
        return torch.from_numpy(np.transpose(np.array(img_resize.convert("RGB")), (2, 0, 1)))
    else:
        img_resize = Image.open(path).resize((crop_size[0], crop_size[1]), 1)
        return torch.from_numpy(np.transpose(np.array(img_resize.convert("RGB")), (2, 0, 1)))

def vcug_label_indices(colormap, colormap2label):
    """将VOC标签中的RGB值映射到它们的类别索引`"""
    colormap = colormap.permute(1, 2, 0).numpy().astype('int32')
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256
           + colormap[:, :, 2])
    return colormap2label[idx]



def remove_black_border(image_pil, tolerance=10):
    # 除去图像的黑边
    image = np.array(image_pil)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    top, bottom, left, right = 0, height, 0, width
    for i in range(height):
        if np.sum(gray[i, :]) > tolerance:
            top = i
            break
    for i in range(height - 1, -1, -1):
        if np.sum(gray[i, :]) > tolerance:
            bottom = i
            break
    for j in range(width):
        if np.sum(gray[:, j]) > tolerance:
            left = j
            break
    for j in range(width - 1, -1, -1):
        if np.sum(gray[:, j]) > tolerance:
            right = j
            break
    cropped_image = image[top:bottom, left:right]
    result_pil = Image.fromarray(cropped_image)
    return result_pil


def add_circle_mask(image_pil):
    image = np.array(image_pil)
    mask = np.zeros_like(image)
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    radius = max(height, width) // 2
    cv2.circle(mask, center, radius, (255, 255, 255), -1)
    result = cv2.bitwise_and(image, mask)

    result_pil = Image.fromarray(result)
    return result_pil




def random_aspect_ratio_crop(image, min_ratio, max_ratio):
    # 加载图片
    width, height = image.size
    # 如果长宽比本身比较大，则将原图返回，不进行裁剪。
    if max(height/width, width/height)>1.2:
        return image
    # 随机生成裁剪长宽比例  aspect_ratio = height/width
    aspect_ratio = random.uniform(min_ratio, max_ratio)

    if height/width>aspect_ratio:
        new_width = width
        new_height = int(width*aspect_ratio)

        left = 0
        upper = random.randint(0, height - new_height)
        right = left + new_width
        lower = upper + new_height

    else:
        new_height = height
        new_width = int(height/aspect_ratio)

        # 随机裁剪
        left = random.randint(0, width - new_width)
        upper = 0
        right = left + new_width
        lower = upper + new_height

    # 进行裁剪
    cropped_image = image.crop((left, upper, right, lower))

    return cropped_image




