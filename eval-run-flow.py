from __future__ import print_function, division
from flow_resnet import *
from spatial_transforms import (Compose, ToTensor, CenterCrop, Scale, Normalize)
from torch.autograd import Variable
from torch.utils.data.sampler import WeightedRandomSampler
from makeDatasetFlow import *
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import argparse
import sys

def main_run(dataset, model_state_dict, dataset_dir, stackSize, numSeg):

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

    spatial_transform = Compose([Scale(256), CenterCrop(224), ToTensor(), normalize])

    vid_seq_test = makeDataset(dataset_dir, spatial_transform=spatial_transform, sequence=True,
                               numSeg=numSeg, stackSize=stackSize, fmt='.jpg', phase='Test')

    test_loader = torch.utils.data.DataLoader(vid_seq_test, batch_size=1,
                            shuffle=False, num_workers=2, pin_memory=True)

    model = flow_resnet34(False, channels=2*stackSize, num_classes=num_classes)
    model.load_state_dict(torch.load(model_state_dict))
    for params in model.parameters():
        params.requires_grad = False

    model.train(False)
    model.cuda()
    test_samples = vid_seq_test.__len__()
    print('Number of samples = {}'.format(test_samples))
    print('Evaluating...')
    numCorr = 0
    true_labels = []
    predicted_labels = []

    for j, (inputs, targets) in enumerate(test_loader):
        inputVariable = Variable(inputs[0].cuda(), volatile=True)
        output_label, _ = model(inputVariable)
        output_label_mean = torch.mean(output_label.data, 0, True)
        _, predicted = torch.max(output_label_mean, 1)
        numCorr += (predicted == targets[0]).sum()
        true_labels.append(targets)
        predicted_labels.append(predicted)
    test_accuracy = (numCorr / test_samples) * 100
    print('Test Accuracy  = {}%'.format(test_accuracy))

    cnf_matrix = confusion_matrix(true_labels, predicted_labels).astype(float)
    cnf_matrix_normalized = cnf_matrix / cnf_matrix.sum(axis=1)[:, np.newaxis]

    ticks = np.linspace(0, 60, num=61)
    plt.imshow(cnf_matrix_normalized, interpolation='none', cmap='binary')
    plt.colorbar()
    plt.xticks(ticks, fontsize=6)
    plt.yticks(ticks, fontsize=6)
    plt.grid(True)
    plt.clim(0, 1)
    plt.savefig(dataset + '-flow.jpg', bbox_inches='tight')
    plt.show()

def __main__():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='gtea61', help='Dataset')
    parser.add_argument('--datasetDir', type=str, default='./dataset/gtea_warped_flow_61/split2/test',
                        help='Dataset directory')
    parser.add_argument('--modelStateDict', type=str,
                        default='./models/gtea61/best_model_state_dict_flow_split2.pth',
                        help='Model path')
    parser.add_argument('--stackSize', type=int, default=5, help='Number of optical flow images in input')
    parser.add_argument('--numSegs', type=int, default=5, help='Number of stacked optical flows')

    args = parser.parse_args()

    dataset = args.dataset
    model_state_dict = args.modelStateDict
    dataset_dir = args.datasetDir
    stackSize = args.stackSize
    numSegs = args.numSegs

    main_run(dataset, model_state_dict, dataset_dir, stackSize, numSegs)

__main__()