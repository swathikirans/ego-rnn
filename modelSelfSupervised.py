import torch
import resnetMod
import torch.nn as nn
from torch.nn import functional as F
from MyConvLSTMCell import *
from MotionSegmentationModule import *

# TODO: handle this thing --> non è possibile avere il DEVICE passato così a cazzo di cane
debug = False
if debug:
    DEVICE = "cpu"
    n_workers = 0
else:
    DEVICE = "cuda"
    n_workers  = 4

class SelfSupervisedAttentionModel(nn.Module):
    def __init__(self, num_classes=61, mem_size=512, cam=True):
        super(SelfSupervisedAttentionModel, self).__init__()
        self.num_classes = num_classes
        self.resNet = resnetMod.resnet34(True, True)
        self.mem_size = mem_size
        self.weight_softmax = self.resNet.fc.weight
        self.lstm_cell = MyConvLSTMCell(512, mem_size)
        self.avgpool = nn.AvgPool2d(7)
        self.dropout = nn.Dropout(0.7)
        self.fc = nn.Linear(mem_size, self.num_classes)
        self.classifier = nn.Sequential(self.dropout, self.fc)
        self.cam = cam
        # Motion Segmentation Module
        self.ms_module = MotionSegmentationModule(512)

    def forward(self, inputVariable):
        state = (torch.zeros((inputVariable.size(1), self.mem_size, 7, 7)).to(DEVICE),
                 torch.zeros((inputVariable.size(1), self.mem_size, 7, 7)).to(DEVICE))
        feats_ms = []

        for t in range(inputVariable.size(0)):
            logit, feature_conv, feature_convNBN = self.resNet(inputVariable[t])
            bz, nc, h, w = feature_conv.size()
            #print(feature_conv.size())
            # bz : ??
            # nc : num channels (?)
            # h  : height
            # w  : width

            feature_conv1 = feature_conv.view(bz, nc, h * w)
            probs, idxs = logit.sort(1, True)

            # MS self-supervised task
            feats_ms.append(self.ms_module(feature_conv))

            if self.cam:
                # Attention layer
                class_idx = idxs[:, 0]
                cam = torch.bmm(self.weight_softmax[class_idx].unsqueeze(1), feature_conv1)
                attentionMAP = F.softmax(cam.squeeze(1), dim=1)
                attentionMAP = attentionMAP.view(attentionMAP.size(0), 1, 7, 7)
                attentionFeat = feature_convNBN * attentionMAP.expand_as(feature_conv)
                state = self.lstm_cell(attentionFeat, state)
            else:
                # Without attention layer
                state = self.lstm_cell(feature_conv1, state)

        feats1 = self.avgpool(state[1]).view(state[1].size(0), -1)
        feats = self.classifier(feats1)

        feats_ms = torch.stack(feats_ms, 0).permute(1, 0, 2)
        #feats_ms = feats_ms.view(bz, inputVariable.size(0), 49)

        return feats, feats1, feats_ms
