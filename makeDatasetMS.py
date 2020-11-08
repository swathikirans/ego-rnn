import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
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
                    if not inst.startswith('.'):
                        inst_dir = os.path.join(dir2, inst)
                        numFramesF = len(glob.glob1(os.path.join(inst_dir, 'rgb'), '*[0-9].png'))
                        numFramesMS = len(glob.glob1(os.path.join(inst_dir, 'mmaps'), '*[0-9].png'))
                        if numFramesF >= stack_size and numFramesMS >= stack_size:
                            DatasetF.append(inst_dir)
                            DatasetMS.append(inst_dir.replace('rgb', 'mmaps'))  # TODO: check if it is the only change -> per ora è inutile
                            Labels.append(class_id)
                            NumFramesF.append(numFramesF)
                            NumFramesMS.append(numFramesMS)
            class_id += 1
    return DatasetF, DatasetMS, Labels, NumFramesF, NumFramesMS


class makeDataset(Dataset): # TODO: refactor makeDatasetMS ??
    def __init__(self, root_dir, splits,
                 spatial_transform=None,
                 transform_rgb=None,
                 transform_MS=None,
                 stack_size=5,
                 seqLen=20,
                 train=True,
                 fmt='.png',
                 verbose=False,
                 regression=False):
        self.imagesF, self.imagesMS, \
            self.labels, self.numFramesF, \
            self.numFramesMS = gen_split(root_dir, splits, stack_size)
        self.spatial_transform = spatial_transform
        self.transform_rgb = transform_rgb
        self.transform_MS = None # TODO:
        self.train = train
        self.seqLen = seqLen
        self.fmt = fmt
        self.verbose = verbose
        self.regression = regression

    def __len__(self):
        return len(self.imagesF)

    def __getitem__(self, idx):
        vid_nameF = self.imagesF[idx]
        vid_nameMS = self.imagesMS[idx]
        label = self.labels[idx]
        numFrame = self.numFramesF[idx]
        numFrameMS = self.numFramesMS[idx]
        inpSeqF = []
        inpSeqMS = []
        self.spatial_transform.randomize_parameters()
        if self.verbose:
            print(vid_nameF, idx)
        # TODO: nota questo loop prende le immagini facendo un linspace sulla durata totale, ha senso?
        for i in np.linspace(1, numFrame, self.seqLen, endpoint=False):
            fl_name = vid_nameF + '/rgb/rgb' + str(int(np.floor(i))).zfill(4) + self.fmt
            img = Image.open(fl_name)
            inpSeqF.append(
                self.transform_rgb(                             # ToTensor and normalize
                    self.spatial_transform(img.convert('RGB'))  # Data augmentation
                )
            )

            # TODO: controllare che il path contenga anche la cartella mmaps -> non c'è bisogno di ottenerlo da gen_split
            # Check if the mmaps exist and if not continue forward
            fl_name = vid_nameMS + '/mmaps/map' + str(int(np.floor(i))).zfill(4) + self.fmt
            if not os.path.exists(fl_name): # if not exist loop until a new frame is available
                # print(fl_name, " does not exists")
                # print("SEARCHING a new frame")
                curr_idx = int(i)
                found = False
                for j in range(curr_idx, numFrameMS):
                    fl_name = vid_nameMS + '/mmaps/map' + str(int(np.floor(j))).zfill(4) + self.fmt
                    if os.path.exists(fl_name):
                        found = True
                        break
                    #else:
                    #    print(fl_name, " does not exists")
                if not found:
                    for j in reversed(range(0, curr_idx)):
                        fl_name = vid_nameMS + '/mmaps/map' + str(int(np.floor(j))).zfill(4) + self.fmt
                        if os.path.exists(fl_name):
                            break
                        #else:
                        #    print(fl_name, " does not exists")
            img = Image.open(fl_name)
            if self.regression:         # regression task
                # convert the image using grey scale
                img = img.convert('L')
            else:                       # classification task
                # convert the image into a binary 0-1 scale
                img = img.convert('1')
            img = self.spatial_transform(img)           # Data Augmentation
            inpSeqMS.append(
                transforms.ToTensor()(
                    transforms.Resize((7, 7))(img)
                )
            ) # Resize 7x7 and ToTensor 1x7x7

        inpSeqF = torch.stack(inpSeqF, 0)
        inpSeqMS = torch.stack(inpSeqMS, 0)

        return inpSeqF, inpSeqMS, label
