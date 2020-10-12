import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import glob
import random


def gen_split(root_dir, splits, stack_size):
    Dataset = []
    Labels = []
    NumFrames = []

    # root_dir  =  #drive/.../GTEA61/processed_frames/
    for split in splits:  # root_dir/SX
        dir1 = os.path.join(root_dir, split)
        print(dir1)
        class_id = 0
        for target in sorted(os.listdir(dir1)):  # root_dir/SX/target/
            dir2 = os.path.join(dir1, target)
            print(dir2)
            insts = sorted(os.listdir(dir2))  #  root_dir/SX/target/Y
            if insts != []:
                for inst in insts:
                    inst_dir = os.path.join(dir2, inst)
                    numFrames = len(glob.glob1(os.path.join(inst_dir, 'rgb'), '*[0-9].png'))
                    if numFrames >= stack_size:
                        Dataset.append(inst_dir)
                    Labels.append(class_id)
                    NumFrames.append(numFrames)

            class_id += 1
    return Dataset, Labels, NumFrames


class makeDataset(Dataset):
    def __init__(self, root_dir, splits,
                 spatial_transform=None,
                 stack_size=5,
                 seqLen=20,
                 train=True,
                 fmt='.png'):
        self.images, self.labels, self.numFrames = gen_split(root_dir, splits, stack_size)
        self.spatial_transform = spatial_transform
        self.train = train
        self.seqLen = seqLen
        self.fmt = fmt

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        vid_name = self.images[idx]
        label = self.labels[idx]
        numFrame = self.numFrames[idx]
        inpSeq = []
        self.spatial_transform.randomize_parameters()

        for i in np.linspace(1, numFrame, self.seqLen, endpoint=False):
            fl_name = vid_name + '/rgb/' + 'rgb' + str(int(np.floor(i))).zfill(4) + self.fmt
            img = Image.open(fl_name)
            inpSeq.append(self.spatial_transform(img.convert('RGB')))
        inpSeq = torch.stack(inpSeq, 0)
        return inpSeq, label