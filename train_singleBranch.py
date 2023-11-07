import torch
import torch.nn as nn
import numpy as np
np.random.seed(2022)
import random
random.seed(2022)



# 导入模型
from models.getmodels import get_model

# 导入数据集
from datasets import load_data_vcug_631

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 300 #300
    batch_size = 4
    k = 0


    vcug_dir = '/home/tanzl/data/VCUG/trainTest/singleBranch/'
    train_loader, validate_loader, test_loader = load_data_vcug_631(vcug_dir=vcug_dir, batch_size=batch_size)

    val_num = validate_loader.dataset.__len__()

    # --------------------定义模型------------------------------------------
    # model_name = "densenet161"  
    # model_name = "vgg16"
    # model_name = "resnet101"
    # model_name = "mobildenet_v2"
    # model_name = "shufflenet_v2_x1_0"
    # model_name = "efficientnet_b0"
    # model_name = "RegNetY_400MF"
    model_name = "efficientnetv2_l"
    # model_name = "googlenet"

    print(model_name)

    num_classes = 6   # 不区分左右

    # 预训练的模型
    net, optimizer, scheduler, model_name_k = get_model(model_name=model_name, 
                                                    num_classes=num_classes, 
                                                    k=k, epochs=epochs, step_szie=90, model_name_assign=None)
    
    # ----------------------训练------------------------------
    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    
    best_acc = 0.0
    best_epoch = 0
    
    
    num_batches = len(train_loader)

    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        for step, data in enumerate(train_loader):
            images, labels = data #应该是label_l, label_r
            optimizer.zero_grad()
            if 'googlenet' in  net.__module__:
                logits, aux_logits2, aux_logits1 = net(images.to(device))
                loss0 = loss_function(logits, labels.to(device))
                loss1 = loss_function(aux_logits1, labels.to(device))
                loss2 = loss_function(aux_logits2, labels.to(device))
                loss = loss0 + loss1 * 0.3 + loss2 * 0.3
                outputs = logits
            else:
                outputs = net(images.to(device))
                loss = loss_function(outputs, labels.to(device))  
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if ((step + 1) % (num_batches // 2) == 0 or step == num_batches - 1):
                predict_y = torch.max(outputs, dim=1)[1]
                acc = torch.eq(predict_y, labels.to(device)).sum().item()
                num = labels.__len__()
                train_accurate = acc / num
                print("train epoch[{}/{}], loss:{:.3f}, train_acc:{:.3f}".format(epoch + 1, epochs, loss, train_accurate))

        if scheduler != None:
            scheduler.step()

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            for val_data in validate_loader:
                val_images, val_label  = val_data  # 左右分支
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_label.to(device)).sum().item()

        val_accurate = acc / val_num
        print("val_accurate:", val_accurate)


        
        if val_accurate > best_acc:
            best_acc = val_accurate
            best_epoch = epoch
            save_path = '/home/tanzl/code/VCUG_retrain/result0720_631/{}_best.pth'.format(model_name)
            torch.save(net.state_dict(), save_path) 
        save_path = '/home/tanzl/code/VCUG_retrain/result0720_631/{}_last.pth'.format(model_name)
        torch.save(net.state_dict(), save_path) 

    print(f'{model_name}: best_acc: {best_acc:.3f}, best_epoch: {best_epoch}/{epochs},  last_acc: {val_accurate:.3f}')

    return val_accurate
    


if __name__ == '__main__':
    acc = train()
    print("acc", acc)

