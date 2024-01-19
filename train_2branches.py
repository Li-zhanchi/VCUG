import torch
import torch.nn as nn

from models.getmodels import get_model_2branch_631
from datasets import load_data_vcug_2branch_631_all, load_data_vcug_out

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 1
    vcug_dir_single = '..'
    vcug_dir_double = '..'

    validate_loader_single, validate_loader_double = load_data_vcug_out(vcug_dir_single, vcug_dir_double, batch_size, label_model='label.txt')
    validate_loader = validate_loader_double

    epochs = 300 
    batch_size = 4
    vcug_dir = '..'
    train_loader = load_data_vcug_2branch_631_all(vcug_dir=vcug_dir, batch_size=batch_size)

    # train_loader, validate_loader, test_loader = load_data_vcug_2branch_631(vcug_dir=vcug_dir, batch_size=batch_size)


    # model_name = "googlenet_2branch"
    # model_name = "vgg16_2branch"
    # model_name = "resnet101_2branch"
    # model_name = "mobildenet_v2_2branch"
    # model_name = 'densenet161_2branch'
    # model_name = "shufflenet_v2_x1_2branch"

    # model_name = "efficientnet_b0_2branch"
    model_name = "RegNetY_400MF_2branch"
    # model_name = "efficientnetv2_l_2branch"
    print(model_name)

    net, optimizer, scheduler, model_name = get_model_2branch_631(model_name=model_name)

    net.to(device)
    loss_function = nn.CrossEntropyLoss()

    
    best_acc = 0.0
    best_epoch = 0
    
    num_batches = len(train_loader)
    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        for step, data in enumerate(train_loader):
            images, labels_l, labels_r = data
            optimizer.zero_grad()
            if net.__module__ == 'models.googlenet':
                logits_l, aux_logits2_l, aux_logits1_l, logits_r, aux_logits2_r, aux_logits1_r = net(images.to(device))
                loss0_l = loss_function(logits_l, labels_l.to(device))
                loss1_l = loss_function(aux_logits1_l, labels_l.to(device))
                loss2_l = loss_function(aux_logits2_l, labels_l.to(device))
                loss_l = loss0_l + loss1_l * 0.3 + loss2_l * 0.3

                loss0_r = loss_function(logits_r, labels_r.to(device))
                loss1_r = loss_function(aux_logits1_r, labels_r.to(device))
                loss2_r = loss_function(aux_logits2_r, labels_r.to(device))
                loss_r = loss0_r + loss1_r * 0.3 + loss2_r * 0.3
                loss = loss_l + loss_r

                outputs_l = logits_l
                outputs_r = logits_r
            else:
                outputs_l, outputs_r = net(images.to(device))
                loss = loss_function(outputs_l, labels_l.to(device))+loss_function(outputs_r, labels_r.to(device))

            loss.backward()
            optimizer.step()


            running_loss += loss.item()
            if ((step + 1) % (num_batches // 2) == 0 or step == num_batches - 1):
                predict_y_l = torch.max(outputs_l, dim=1)[1]
                predict_y_r = torch.max(outputs_r, dim=1)[1]
                acc = (torch.eq(predict_y_l, labels_l.to(device)).sum().item() + torch.eq(predict_y_r, labels_r.to(device)).sum().item())
                train_accurate = acc / labels_l.__len__()*2
                print("Epoch [{}/{}], Loss: {:.3f}, Train Accuracy: {:.3f}".format(epoch + 1, epochs, loss, train_accurate))

        if scheduler != None:
            scheduler.step()

        net.eval()
        acc = 0.0
        num = 0.0
        with torch.no_grad():
            for val_data in validate_loader:
                images, labels_l, labels_r, _  = val_data
                outputs_l, outputs_r = net(images.to(device))
                predict_y_l = torch.max(outputs_l, dim=1)[1]
                predict_y_r = torch.max(outputs_r, dim=1)[1]
                acc += (torch.eq(predict_y_l, labels_l.to(device)).sum().item() + torch.eq(predict_y_r, labels_r.to(device)).sum().item())
                num += labels_l.__len__()*2

        val_accurate = acc / num
        print("val_accurate:", val_accurate)


        if val_accurate > best_acc:
            best_acc = val_accurate
            best_epoch = epoch
            save_path = '..'
            torch.save(net.state_dict(), save_path) 
        save_path = '..'
        torch.save(net.state_dict(), save_path) 

    print(f'{model_name}: best_acc: {best_acc:.3f}, best_epoch: {best_epoch}/{epochs},  last_acc: {val_accurate:.3f}')

    return val_accurate
    



if __name__ == '__main__':
    acc = train()
    print("acc", acc)
