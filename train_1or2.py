import torch
import torch.nn as nn

from models.getmodels import get_model

from datasets import load_data_vcug_1or2_631

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epochs = 100
    batch_size = 8
    vcug_dir = '..'
    train_loader, validate_loader, test_loader = load_data_vcug_1or2_631(vcug_dir=vcug_dir, batch_size=batch_size)
    val_num = validate_loader.dataset.__len__()

    model_name = 'resnet101'
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
            if 'googlenet' in net.__module__:
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

        if scheduler is not None:
            scheduler.step()

        net.eval()
        acc = 0.0
        acc_0 = 0.0
        acc_1 = 0.0
        num_0 = 0.0
        num_1 = 0.0

        with torch.no_grad():
            for val_data in validate_loader:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                num_0 += (val_labels == 0).sum().item()
                num_1 += (val_labels == 1).sum().item()
                acc_0 += torch.eq(predict_y, val_labels.to(device))[val_labels == 0].sum().item()
                acc_1 += torch.eq(predict_y, val_labels.to(device))[val_labels == 1].sum().item()

        val_accurate = acc / val_num
        val_acc_0 = acc_0 / num_0
        val_acc_1 = acc_1 / num_1
        print("val_accurate:", val_accurate)
        print("val_acc_0", val_acc_0)
        print("val_acc_1", val_acc_1)

        if val_accurate > best_acc:
            best_acc = val_accurate
            best_epoch = epoch
            save_path = '..'.format(model_name)
            torch.save(net.state_dict(), save_path)
        save_path = '..'.format(model_name)
        torch.save(net.state_dict(), save_path)

    print(f'{model_name}: best_acc: {best_acc:.3f}, best_epoch: {best_epoch}/{epochs},  last_acc: {val_accurate:.3f}')

    return val_accurate
    


if __name__ == '__main__':
    acc = train()

