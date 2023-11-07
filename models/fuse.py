
from turtle import forward
import torch
from torch import device, nn

from models.resnet import resnet101, resnet34
from models.getmodels import get_model_trained

class fc_vote_results(nn.Module):
    def __init__(self,num_voter,num_classes=11):
        super(fc_vote_results, self).__init__()
        self.hidden = nn.Linear(num_voter, 32) # 隐藏层
        self.act = nn.ReLU()
        self.output = nn.Linear(32, num_classes)  # 输出层

    def forward(self, x):
        a = self.act(self.hidden(x))
        return self.output(a)

class multi_fuse_conv1d(nn.Module):
    def __init__(self, model_names, num_classes, k):
        super(multi_fuse_conv1d, self).__init__()
        # ---------初始化模型------------------------------------------
        self.models = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for model_name in model_names:
            model = get_model_trained(model_name=model_name, num_classes=num_classes, k=k)
            model.to(device)
            model.eval()
            self.models.append(model)
            print("{}初始完成".format(model_name))
    
        # -------定义卷积核用于进行加权-------------------------------------
        # 结合博客https://blog.csdn.net/sunny_xsc1994/article/details/82969867中的图看，in_channel就是d；out_channel就是卷积后图像的d
        self.conv = nn.Conv1d(in_channels=len(model_names), out_channels=1, kernel_size=1)

    def forward(self, x):
        # 对不同模型产生的进行拼接，然后进行一维卷积，得到结果
        outputs = None # 最后拼接的输出
        for model in self.models:
            output = model(x)
            output = output.unsqueeze(1)
            if outputs == None:
                outputs = output
            else:
                outputs = torch.cat((outputs,output),1)
        outputs = self.conv(outputs)
        outputs = outputs.squeeze(1)
        return outputs

class multi_fuse_param(nn.Module):
    def __init__(self, model_names, num_classes, k):
        super(multi_fuse_param, self).__init__()
        # ---------初始化模型------------------------------------------
        self.models = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for model_name in model_names:
            model = get_model_trained(model_name=model_name, num_classes=num_classes, k=k)
            model.to(device)
            model.eval()
            self.models.append(model)
            print("{}初始完成".format(model_name))
    
        # -------定义卷积核用于进行加权-------------------------------------
        # 结合博客https://blog.csdn.net/sunny_xsc1994/article/details/82969867中的图看，in_channel就是d；out_channel就是卷积后图像的d
        # self.conv = nn.Conv1d(in_channels=len(model_names), out_channels=1, kernel_size=1)
        self.a = nn.Parameter(torch.Tensor(len(model_names)))
        self.a.data.normal_(mean=1/len(model_names), std=0.00001)


    def forward(self, x):
        # 对不同模型产生的进行拼接，然后进行一维卷积，得到结果
        result = 0 # 最后拼接的输出
        for i in range(len(self.models)):
            output = self.models[i](x)
            result += output*self.a[i]
        return result



class multi_fuse_param_cls(nn.Module):
    def __init__(self, model_names, num_classes, k):
        super(multi_fuse_param_cls, self).__init__()
        # ---------初始化模型------------------------------------------
        self.models = []
        self.num_classes = num_classes
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for model_name in model_names:
            model = get_model_trained(model_name=model_name, num_classes=num_classes, k=k)
            model.to(device)
            model.eval()
            self.models.append(model)
            print("{}初始完成".format(model_name))
    
        # -------定义卷积核用于进行加权-------------------------------------
        # 结合博客https://blog.csdn.net/sunny_xsc1994/article/details/82969867中的图看，in_channel就是d；out_channel就是卷积后图像的d
        # self.conv = nn.Conv1d(in_channels=len(model_names), out_channels=1, kernel_size=1)
        self.a = nn.Parameter(torch.Tensor(len(model_names),num_classes))
        for i in range(num_classes):
            self.a[:,i].data.normal_(mean=1/len(model_names), std=0.00001)

    def forward(self, x):
        # 对不同模型产生的进行拼接，然后进行一维卷积，得到结果
        
        temp_output = self.models[0](x)
        result = torch.zeros_like(temp_output)
        for i in range(len(self.models)):
            output = self.models[i](x)
            # 对每个类别分配一个系数，进行更加精细的分类
            for j in range(self.num_classes):
                result[:,j] += output[:,j]*self.a[i][j]
        return result




class multi_fuse_pic(nn.Module):
    def __init__(self, model_names, num_classes, k):
        super(multi_fuse_pic, self).__init__()
        # ---------初始化模型------------------------------------------
        self.models = []
        self.num_classes = num_classes
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for model_name in model_names:
            model = get_model_trained(model_name=model_name, num_classes=num_classes, k=k)
            model.to(device)
            model.eval()
            self.models.append(model)
            print("{}初始完成".format(model_name))
    
        # -------利用resnet来从图像获取权重-------------------------------------
        self.resnet = resnet34()
        model_weight_path = "/home/zelongtan/code/VCUG/pretrained/resnet34-pre.pth"
        self.resnet.load_state_dict(torch.load(model_weight_path, map_location=device))

        in_channel = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_channel, len(model_names))


    def forward(self, x):
        # 对不同模型产生的进行拼接，然后进行一维卷积，得到结果
        w = self.resnet(x)
        temp_output = self.models[0](x)
        batch, num_classes = temp_output.shape
        result = torch.zeros_like(temp_output)

        for i in range(len(self.models)):
            output = self.models[i](x)
            for s in range(batch):
                result[s] += output[s]*w[s][i]

        return result




class multi_fuse_pic_cls(nn.Module):
    def __init__(self, model_names, num_classes, k):
        super(multi_fuse_pic_cls, self).__init__()
        # ---------初始化模型------------------------------------------
        self.models = []
        self.num_classes = num_classes
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for model_name in model_names:
            model = get_model_trained(model_name=model_name, num_classes=num_classes, k=k)
            model.to(device)
            model.eval()
            self.models.append(model)
            print("{}初始完成".format(model_name))
    
        # -------利用resnet来从图像获取权重-------------------------------------
        self.resnet = resnet34()
        model_weight_path = "/home/zelongtan/code/VCUG/pretrained/resnet34-pre.pth"
        self.resnet.load_state_dict(torch.load(model_weight_path, map_location=device))

        in_channel = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_channel, len(model_names)*num_classes)


    def forward(self, x):
        # 获取一些基本参数，并初始化result
        temp_output = self.models[0](x)
        batch, num_classes = temp_output.shape
        result = torch.zeros_like(temp_output)
        
        # 得到参数并改变形状
        w = self.resnet(x)
        w = w.reshape(batch, len(self.models), self.num_classes)

        for i in range(len(self.models)):
            output = self.models[i](x)
            for s in range(batch):
                for j in range(self.num_classes):
                    result[s][j] += output[s][j]*w[s][i][j]

        return result

class turkey(nn.Module):
    def __init__(self, num_classes, k, fea_num):
        super(turkey, self).__init__()
        # ---------初始化模型------------------------------------------
        self.num_classes = num_classes
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model_dense = get_model_trained(model_name="densenet161", num_classes=num_classes, k=k)
        self.model_google = get_model_trained(model_name="googlenet", num_classes=num_classes, k=k)
        self.model_mobile = get_model_trained(model_name="mobildenet_v2", num_classes=num_classes, k=k)

        self.model_dense.to(device)
        self.model_dense.eval()
        self.model_google.to(device)
        self.model_google.eval()
        self.model_mobile.to(device)
        self.model_mobile.eval()

        # self.fea_index = self.get_fea_index(fea_num=fea_num, k=k)

        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(fea_num, num_classes)
        # self.fc = nn.Linear(4512, num_classes)

    def forward(self, x):
        fea_dense = self.model_dense.get_feature(x)
        fea_google = self.model_google.get_feature(x)
        fea_mobile = self.model_mobile.get_feature(x)
        fea = torch.cat((fea_dense, fea_google, fea_mobile),1)
        # print(fea.shape)
        # fea = fea[:,self.fea_index]
        fea = self.dropout(fea)
        fea = self.fc(fea)
        return fea

    def get_fea_index(self, fea_num, k):
        fea = {
                32:
                [
                    [0, 1, 3, 4, 5, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 24, 25, 27, 28, 30, 32, 35, 37, 41, 42, 43, 44, 48, 49, 54],
                    [0, 1, 2, 3, 4, 6, 7, 9, 10, 11, 12, 13, 15, 19, 21, 22, 24, 25, 26, 28, 31, 32, 34, 36, 37, 42, 44, 46, 48, 49, 50, 51],
                    [4, 5, 6, 7, 8, 9, 11, 12, 17, 18, 20, 21, 22, 25, 30, 32, 33, 34, 36, 37, 41, 42, 43, 44, 46, 47, 48, 49, 50, 53, 56, 57],
                    [0, 1, 2, 6, 9, 12, 13, 15, 16, 17, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 35, 36, 38, 39, 41, 42, 43, 44, 46],
                    [0, 1, 2, 4, 5, 6, 7, 8, 9, 11, 13, 14, 16, 17, 18, 19, 20, 21, 22, 24, 25, 26, 28, 30, 32, 36, 37, 38, 39, 42, 44, 47]],
                64:
                [
                    [0, 1, 3, 4, 5, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 24, 25, 27, 28, 30, 32, 35, 37, 41, 42, 43, 44, 48, 49, 54, 56, 58, 61, 62, 63, 64, 66, 67, 68, 70, 73, 74, 75, 76, 79, 80, 81, 82, 84, 85, 86, 88, 89, 91, 93, 97, 104, 106, 107, 108, 109, 110],
                    [0, 1, 2, 3, 4, 6, 7, 9, 10, 11, 12, 13, 15, 19, 21, 22, 24, 25, 26, 28, 31, 32, 34, 36, 37, 42, 44, 46, 48, 49, 50, 51, 53, 55, 57, 58, 59, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 75, 77, 78, 79, 80, 83, 84, 85, 88, 89, 90, 91, 93, 102, 103, 104],
                    [4, 5, 6, 7, 8, 9, 11, 12, 17, 18, 20, 21, 22, 25, 30, 32, 33, 34, 36, 37, 41, 42, 43, 44, 46, 47, 48, 49, 50, 53, 56, 57, 61, 62, 63, 64, 66, 68, 70, 71, 72, 73, 74, 77, 78, 79, 80, 81, 82, 84, 85, 89, 90, 91, 94, 95, 96, 97, 99, 101, 102, 103, 104, 106],
                    [0, 1, 2, 6, 9, 12, 13, 15, 16, 17, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 35, 36, 38, 39, 41, 42, 43, 44, 46, 49, 50, 52, 53, 55, 56, 60, 61, 62, 64, 66, 69, 72, 74, 75, 76, 78, 80, 81, 84, 89, 90, 91, 92, 93, 94, 95, 97, 100, 103, 104, 106],
                    [0, 1, 2, 4, 5, 6, 7, 8, 9, 11, 13, 14, 16, 17, 18, 19, 20, 21, 22, 24, 25, 26, 28, 30, 32, 36, 37, 38, 39, 42, 44, 47, 50, 52, 54, 56, 57, 58, 60, 62, 67, 68, 69, 70, 71, 72, 73, 75, 76, 77, 78, 79, 81, 82, 83, 84, 85, 86, 91, 94, 96, 99, 100, 101]
                ],
                128:
                [
                    [0, 1, 3, 4, 5, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 24, 25, 27, 28, 30, 32, 35, 37, 41, 42, 43, 44, 48, 49, 54, 56, 58, 61, 62, 63, 64, 66, 67, 68, 70, 73, 74, 75, 76, 79, 80, 81, 82, 84, 85, 86, 88, 89, 91, 93, 97, 104, 106, 107, 108, 109, 110, 111, 113, 115, 117, 120, 121, 125, 126, 127, 129, 132, 133, 135, 136, 137, 138, 140, 143, 146, 147, 148, 154, 156, 157, 158, 160, 161, 164, 166, 167, 168, 171, 174, 175, 176, 177, 182, 183, 184, 187, 190, 192, 193, 194, 197, 198, 199, 202, 203, 206, 207, 208, 210, 211, 217, 218, 220, 222, 223, 225, 226, 229, 231, 232],
                    [0, 1, 2, 3, 4, 6, 7, 9, 10, 11, 12, 13, 15, 19, 21, 22, 24, 25, 26, 28, 31, 32, 34, 36, 37, 42, 44, 46, 48, 49, 50, 51, 53, 55, 57, 58, 59, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 75, 77, 78, 79, 80, 83, 84, 85, 88, 89, 90, 91, 93, 102, 103, 104, 106, 107, 108, 111, 113, 114, 115, 116, 119, 121, 128, 129, 130, 132, 134, 135, 136, 138, 139, 140, 141, 142, 143, 147, 148, 150, 154, 155, 156, 157, 158, 160, 162, 163, 165, 166, 168, 171, 172, 174, 175, 176, 177, 178, 179, 181, 182, 186, 188, 190, 191, 192, 193, 194, 197, 198, 199, 200, 203, 206, 207, 208, 209, 210],
                    [4, 5, 6, 7, 8, 9, 11, 12, 17, 18, 20, 21, 22, 25, 30, 32, 33, 34, 36, 37, 41, 42, 43, 44, 46, 47, 48, 49, 50, 53, 56, 57, 61, 62, 63, 64, 66, 68, 70, 71, 72, 73, 74, 77, 78, 79, 80, 81, 82, 84, 85, 89, 90, 91, 94, 95, 96, 97, 99, 101, 102, 103, 104, 106, 108, 111, 114, 115, 116, 117, 120, 121, 122, 123, 124, 125, 127, 130, 132, 133, 134, 135, 137, 139, 140, 141, 142, 143, 146, 147, 148, 150, 153, 154, 156, 157, 158, 159, 160, 162, 163, 164, 165, 166, 167, 168, 169, 171, 172, 175, 178, 179, 181, 183, 184, 186, 187, 188, 190, 191, 193, 194, 195, 197, 198, 199, 200, 202],
                    [0, 1, 2, 6, 9, 12, 13, 15, 16, 17, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 35, 36, 38, 39, 41, 42, 43, 44, 46, 49, 50, 52, 53, 55, 56, 60, 61, 62, 64, 66, 69, 72, 74, 75, 76, 78, 80, 81, 84, 89, 90, 91, 92, 93, 94, 95, 97, 100, 103, 104, 106, 108, 113, 115, 119, 120, 124, 126, 130, 134, 135, 137, 138, 141, 142, 143, 144, 145, 148, 150, 153, 154, 155, 157, 158, 159, 162, 164, 165, 166, 167, 169, 172, 173, 175, 176, 177, 178, 181, 183, 184, 186, 187, 190, 192, 197, 198, 199, 200, 203, 204, 205, 206, 207, 209, 210, 211, 213, 214, 217, 218, 219, 222, 223, 224],
                    [0, 1, 2, 4, 5, 6, 7, 8, 9, 11, 13, 14, 16, 17, 18, 19, 20, 21, 22, 24, 25, 26, 28, 30, 32, 36, 37, 38, 39, 42, 44, 47, 50, 52, 54, 56, 57, 58, 60, 62, 67, 68, 69, 70, 71, 72, 73, 75, 76, 77, 78, 79, 81, 82, 83, 84, 85, 86, 91, 94, 96, 99, 100, 101, 103, 104, 106, 107, 108, 109, 114, 115, 116, 120, 121, 124, 127, 128, 129, 130, 133, 134, 135, 137, 138, 141, 142, 143, 147, 148, 150, 151, 153, 154, 155, 157, 158, 159, 160, 161, 162, 169, 170, 172, 174, 175, 177, 179, 182, 183, 184, 186, 190, 191, 192, 193, 194, 195, 198, 200, 205, 207, 211, 213, 214, 218, 220, 225]
                ]
                }
        


        return fea[fea_num][k]


def get_turkey_trained(k):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = turkey(num_classes=11, k=k, fea_num=4512)
    model_weight_path = "/home/zelongtan/code/VCUG/result09/result0930/turkey_{}.pth".format(k)
    net.load_state_dict(torch.load(model_weight_path, map_location=device))
    return net













