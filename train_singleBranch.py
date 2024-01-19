import torch
import torch.nn as nn

from models.getmodels import get_model
from datasets import load_data_vcug_631_all, load_data_vcug_out


def train():
    epochs = 300 
    batch_size = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 1
    vcug_dir_single = '..'
    vcug_dir_double = '..'
    validate_loader_single, validate_loader_double = load_data_vcug_out(vcug_dir_single, vcug_dir_double, batch_size, label_model='label.txt')
    validate_loader = validate_loader_single
    vcug_dir = '..'
    train_loader = load_data_vcug_631_all(vcug_dir=vcug_dir, batch_size=batch_size)

    # train_loader, validate_loader, test_loader = load_data_vcug_631(vcug_dir=vcug_dir, batch_size=batch_size)


    val_num = validate_loader.dataset.__len__()

    # model_name = "densenet161"  
    # model_name = "vgg16"
    # model_name = "resnet101"
    # model_name = "mobildenet_v2"
    # model_name = "shufflenet_v2_x1_0"
    # model_name = "efficientnet_b0"
    model_name = "RegNetY_400MF"
    # model_name = "efficientnetv2_l"
    # model_name = "googlenet"
    print(model_name)

    net, optimizer, scheduler, model_name = get_model(model_name=model_name)

    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    
    best_acc = 0.0
    best_epoch = 0
    
    num_batches = len(train_loader)
    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        for step, data in enumerate(train_loader):
            images, labels = data
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

            running_loss += loss.item()
            if ((step + 1) % (num_batches // 2) == 0 or step == num_batches - 1):
                predict_y = torch.max(outputs, dim=1)[1]
                acc = torch.eq(predict_y, labels.to(device)).sum().item()
                train_accurate = acc / labels.__len__()
                print("Epoch [{}/{}], Loss: {:.3f}, Train Accuracy: {:.3f}".format(epoch + 1, epochs, loss, train_accurate))

        if scheduler != None:
            scheduler.step()

        net.eval()
        acc = 0.0 
        with torch.no_grad():
            for val_data in validate_loader:
                val_images, val_label, _  = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_label.to(device)).sum().item()

        val_accurate = acc / val_num
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

