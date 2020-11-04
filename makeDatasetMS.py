import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import glob
import random


def gen_split(root_dir, splits, stack_size):
    DatasetF = []
    DatasetMS = []
    Labels = []
    NumFramesF = []
    NumFramesMS = []

    # root_dir  =  #drive/.../GTEA61/processed_frames/
    for split in splits:  # root_dir/SX
        dir1 = os.path.join(root_dir, split)
        # print(dir1)
        class_id = 0
        for target in sorted(os.listdir(dir1)):  # root_dir/SX/target/
            dir2 = os.path.join(dir1, target)
            # print(dir2)
            insts = sorted(os.listdir(dir2))  # root_dir/SX/target/Y
            if insts:
                for inst in insts:
                    inst_dir = os.path.join(dir2, inst)
                    numFramesF = len(glob.glob1(os.path.join(inst_dir, 'rgb'), '*[0-9].png'))
                    numFramesMS = len(glob.glob1(os.path.join(inst_dir, 'mmaps'), '*[0-9].png'))
                    if numFramesF >= stack_size and numFramesMS >= stack_size:
                        DatasetF.append(inst_dir)
                        DatasetMS.append(inst_dir.replace('rgb', 'mmaps'))  # TODO: check if it is the only change
                        Labels.append(class_id)
                        NumFramesF.append(numFramesF)
                        NumFramesMS.append(numFramesMS)
            class_id += 1
    return DatasetF, DatasetMS, Labels, NumFramesF, NumFramesMS


class makeDataset(Dataset):
    def __init__(self, root_dir, splits,
                 spatial_transform=None,
                 stack_size=5,
                 seqLen=20,
                 train=True,
                 fmt='.png',
                 verbose=False):
        self.imagesF, self.imagesMS, \
            self.labels, self.numFramesF, \
            self.numFramesMS = gen_split(root_dir, splits, stack_size)
        self.spatial_transform = spatial_transform
        self.train = train
        self.seqLen = seqLen
        self.fmt = fmt
        self.verbose = verbose

    def __len__(self):
        return len(self.imagesF)

    def __getitem__(self, idx):
        vid_nameF = self.imagesF[idx]
        vid_nameMS = self.imagesMS[idx]
        label = self.labels[idx]
        numFrame = self.numFramesF[idx]
        inpSeqF = []
        inpSeqMS = []
        self.spatial_transform.randomize_parameters()
        if self.verbose:
            print(vid_nameF, idx)
        # TODO: nota questo loop prende le immagini facendo un linspace sulla durata totale, ha senso?
        for i in np.linspace(1, numFrame, self.seqLen, endpoint=False):
            fl_name = vid_nameF + 'rgb' + str(int(np.floor(i))).zfill(4) + self.fmt
            img = Image.open(fl_name)
            inpSeqF.append(self.spatial_transform(img.convert('RGB')))

            # TODO: controllare che il path contenga anche la cartella mmaps
            fl_name = vid_nameMS + '/map' + str(int(np.floor(i))).zfill(4) + self.fmt
            img = Image.open(fl_name)
            inpSeqMS.append(self.spatial_transform(img))

        inpSeqF = torch.stack(inpSeqF, 0)
        inpSeqMS = torch.stack(inpSeqMS, 0)

        return inpSeqF, inpSeqMS, label
