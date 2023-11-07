from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision
import torch
from torch import nn
import torch.nn.functional as F

def get_FasterRcnn(num_classes=2, checkpoint=None, size=224):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.transform.min_size = (size,)
    model.transform.max_size = size
    if checkpoint!=None:
        checkpoint = torch.load(checkpoint, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        model.load_state_dict(checkpoint['model'])
    return model