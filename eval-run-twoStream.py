from __future__ import print_function, division
from flow_resnet import *
from objectAttentionModelConvLSTM import *
from spatial_transforms import (Compose, ToTensor, CenterCrop, Scale, Normalize)
from torch.autograd import Variable
from torch.utils.data.sampler import WeightedRandomSampler
from makeDatasetTwoStream import *
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import argparse


def main_run(dataset, flowModel_state_dict, RGBModel_state_dict, dataset_dir, stackSize, seqLen, memSize, numSeg):

    if dataset == 'gtea61':
        num_classes = 61
    elif dataset == 'gtea71':
      num_classes = 71
    elif dataset == 'gtea_gaze':
        num_classes = 44
    elif dataset == 'egtea':
        num_classes = 106

    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]

    normalize = Normalize(mean=mean, std=std)

    flow_wt = 0.5
    testBatchSize = 1
    sequence = True
    spatial_transform = Compose([Scale(256), CenterCrop(224), ToTensor(), normalize])

    vid_seq_test = makeDataset(dataset_dir, spatial_transform=spatial_transform, sequence=sequence, numSeg=numSeg,
                               stackSize=stackSize, fmt='.jpg', phase='Test', seqLen=seqLen)

    test_loader = torch.utils.data.DataLoader(vid_seq_test, batch_size=testBatchSize,
                            shuffle=False, num_workers=2, pin_memory=True)

    modelFlow = flow_resnet34(False, channels=2*stackSize, num_classes=num_classes)
    modelFlow.load_state_dict(torch.load(flowModel_state_dict))
    modelRGB = attentionModel(num_classes=num_classes, mem_size=memSize)
    modelRGB.load_state_dict(torch.load(RGBModel_state_dict))


    for params in modelFlow.parameters():
        params.requires_grad = False

    for params in modelRGB.parameters():
        params.requires_grad = False

    modelFlow.train(False)
    modelRGB.train(False)
    modelFlow.cuda()
    modelRGB.cuda()
    test_samples = vid_seq_test.__len__()
    print('Number of samples = {}'.format(test_samples))
    print('Evaluating...')
    numCorrTwoStream = 0

    true_labels = []
    predicted_labels = []

    for j, (inputFlow, inputFrame, targets) in enumerate(test_loader):
        inputVariableFlow = Variable(inputFlow[0].cuda(), volatile=True)
        inputVariableFrame = Variable(inputFrame.permute(1, 0, 2, 3, 4).cuda(), volatile=True)
        output_labelFlow, _ = modelFlow(inputVariableFlow)
        output_labelFrame, _ = modelRGB(inputVariableFrame)
        output_label_meanFlow = torch.mean(output_labelFlow.data, 0, True)
        output_label_meanTwoStream = (flow_wt * output_label_meanFlow) + ((1-flow_wt) * output_labelFrame.data)
        _, predictedTwoStream = torch.max(output_label_meanTwoStream, 1)
        numCorrTwoStream += (predictedTwoStream == targets[0]).sum()
        true_labels.append(targets)
        predicted_labels.append(predictedTwoStream)
    test_accuracyTwoStream = (numCorrTwoStream / test_samples) * 100
    print('Test Accuracy = {}'.format(test_accuracyTwoStream))

    cnf_matrix = confusion_matrix(true_labels, predicted_labels).astype(float)
    cnf_matrix_normalized = cnf_matrix / cnf_matrix.sum(axis=1)[:, np.newaxis]

    ticks = np.linspace(0, 60, num=61)
    plt.imshow(cnf_matrix_normalized, interpolation='none', cmap='binary')
    plt.colorbar()
    plt.xticks(ticks, fontsize=6)
    plt.yticks(ticks, fontsize=6)
    plt.grid(True)
    plt.clim(0, 1)
    plt.savefig(dataset + '-twoStream.jpg', bbox_inches='tight')
    plt.show()

def __main__():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='gtea61', help='Dataset')
    parser.add_argument('--datasetDir', type=str, default='./dataset/gtea_warped_flow_61/split2/train',
                        help='Dataset directory')
    parser.add_argument('--flowModelStateDict', type=str, default='./models/gtea61/best_model_state_dict_flow_split2.pth',
                        help='Flow Model path')
    parser.add_argument('--RGBModelStateDict', type=str, default='./models/gtea61/best_model_state_dict_rgb_split2.pth',
                        help='RGB Model path')
    parser.add_argument('--seqLen', type=int, default=25, help='Length of sequence')
    parser.add_argument('--stackSize', type=int, default=5, help='Number of optical flow images in input')
    parser.add_argument('--memSize', type=int, default=512, help='ConvLSTM hidden state size')
    parser.add_argument('--numSegs', type=int, default=10, help='Number of flow segments')

    args = parser.parse_args()

    dataset = args.dataset
    flowModel_state_dict = args.flowModelStateDict
    RGBModel_state_dict = args.RGBModelStateDict
    dataset_dir = args.datasetDir
    seqLen = args.seqLen
    stackSize = args.stackSize
    memSize = args.memSize
    numSeg = args.numSegs

    main_run(dataset, flowModel_state_dict, RGBModel_state_dict, dataset_dir, stackSize, seqLen, memSize, numSeg)

__main__()
