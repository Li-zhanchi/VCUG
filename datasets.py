import torch
from torchvision import transforms
import numpy as np
import os
import random
from PIL import Image
import cv2

def random_aspect_ratio_crop(image, min_ratio, max_ratio):
    width, height = image.size
    if max(height/width, width/height)>1.2:
        return image
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
        left = random.randint(0, width - new_width)
        upper = 0
        right = left + new_width
        lower = upper + new_height
    cropped_image = image.crop((left, upper, right, lower))
    return cropped_image

def remove_black_border(image_pil, tolerance=10):
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
        self.transform = transforms.Compose([transforms.Resize(512),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.451, 0.451, 0.451), (0.230, 0.230, 0.230))
                                    ])
        self.vcug_dir = vcug_dir
        self.vcug_img = []
        self.png_root = '..'

        with open(os.path.join(vcug_dir, txt), 'r') as f:
            for line in f.readlines():
                self.vcug_img.append(line.strip())

        if txt=='split_6.txt':
            self.balance_12()
        self.label_model = label_model
        self.txt = txt

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
        img = remove_black_border(img)
        img = add_circle_mask(img)
        if (self.txt=='split_6.txt') or (self.txt=='split_all.txt'):
            img = random_aspect_ratio_crop(img, 0.75, 1.3333)

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

        num_samples = len(vcug_img_2)
        num_samples_diff = len(vcug_img_1) - num_samples
        up_sampled_vcug_img_2 = np.random.choice(vcug_img_2, size=num_samples_diff, replace=True)
        self.vcug_img.extend(up_sampled_vcug_img_2)

        vcug_img_1 = [img for img in self.vcug_img if 'single' in img]
        vcug_img_2 = [img for img in self.vcug_img if 'double' in img]


    def __len__(self):
        return len(self.vcug_img)

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
        img = remove_black_border(img)
        img = add_circle_mask(img) 
        if (self.txt=='split_6.txt') or (self.txt=='split_all.txt'):
            img = random_aspect_ratio_crop(img, 0.75, 1.3333)
        
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



        cls_path = os.path.join(self.vcug_dir, self.vcug_img[idx], self.label_model) 
        cls = []
        with open(cls_path, 'r') as f:
            for line in f.readlines():
                cls.append(line.strip())
        cls_l = int(cls[0])
        cls_r = int(cls[1])

        if (self.txt=='split_6.txt') or (self.txt=='split_all.txt'):
            if random.random() > thred:
                img = self.trans_flip(img)
                temp = cls_l
                cls_l = cls_r
                cls_r = temp
        return img, cls_l, cls_r

    def __len__(self):
        return len(self.vcug_img)

def load_data_vcug_2branch_631(vcug_dir, batch_size, label_model='label.txt'):
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    dataset_train = VCUGDataset_2branch_631(vcug_dir=vcug_dir,  txt='split_6.txt', label_model=label_model)
    dataset_val = VCUGDataset_2branch_631(vcug_dir=vcug_dir,  txt='split_3.txt', label_model=label_model)
    dataset_test = VCUGDataset_2branch_631(vcug_dir=vcug_dir,  txt='split_1.txt', label_model=label_model)
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
        img = remove_black_border(img) 
        img = add_circle_mask(img) 
        if (self.txt=='split_6.txt') or (self.txt=='split_all.txt'):
            img = random_aspect_ratio_crop(img, 0.75, 1.3333)
        
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


        cls_path = os.path.join(self.vcug_dir, self.vcug_img[idx], self.label_model) 
        cls = []
        with open(cls_path, 'r') as f:
            for line in f.readlines():
                cls.append(line.strip())
        cls = int(cls[0])
        return img, cls

    def __len__(self):
        return len(self.vcug_img)

def load_data_vcug_631(vcug_dir, batch_size, label_model='label.txt'):
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    dataset_train = VCUGDataset_631(vcug_dir=vcug_dir,  txt='split_6.txt', label_model=label_model)
    dataset_val = VCUGDataset_631(vcug_dir=vcug_dir,  txt='split_3.txt', label_model=label_model)
    dataset_test = VCUGDataset_631(vcug_dir=vcug_dir,  txt='split_1.txt', label_model=label_model)

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

def collate_fn(batch):
    return tuple(zip(*batch))

def add_square(pil_file):
    w, h = pil_file.size
    target = max(w, h)
    image1 = Image.new("RGB", (target, target))
    pad_w = int((target - w) / 2)
    pad_h = int((target - h) / 2)
    image1.paste(pil_file, (pad_w, pad_h))
    return image1

def add_circle_mask(image_pil):
    image = np.array(image_pil)
    mask = np.zeros_like(image)
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    radius = max(height, width) // 2
    cv2.circle(mask, center, radius, (255, 255, 255), -1)
    result = cv2.bitwise_and(image, mask)
    return Image.fromarray(result)





