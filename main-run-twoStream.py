from __future__ import print_function, division
from spatial_transforms import (Compose, ToTensor, CenterCrop, Scale, Normalize, MultiScaleCornerCrop,
                                RandomHorizontalFlip)
from tensorboardX import SummaryWriter
import torch.nn as nn
from twoStreamModel import *
from torch.autograd import Variable
from torch.utils.data.sampler import WeightedRandomSampler
from makeDatasetTwoStream import *
import argparse

import sys


def main_run(dataset, flowModel, rgbModel, stackSize, seqLen, memSize, trainDatasetDir, valDatasetDir, outDir,
             trainBatchSize, valBatchSize, lr1, numEpochs, decay_step, decay_factor):


    if dataset == 'gtea61':
        num_classes = 61
    elif dataset == 'gtea71':
        num_classes = 71
    elif dataset == 'gtea_gaze':
        num_classes = 44
    elif dataset == 'egtea':
        num_classes = 106
    else:
        print('Dataset not found')
        sys.exit()

    model_folder = os.path.join('./', outDir, dataset, 'twoStream')  # Dir for saving models and log files
    # Create the dir
    if os.path.exists(model_folder):
        print('Dir {} exists!'.format(model_folder))
        sys.exit()
    os.makedirs(model_folder)

    # Log files
    writer = SummaryWriter(model_folder)
    train_log_loss = open((model_folder + '/train_log_loss.txt'), 'w')
    train_log_acc = open((model_folder + '/train_log_acc.txt'), 'w')
    val_log_loss = open((model_folder + '/val_log_loss.txt'), 'w')
    val_log_acc = open((model_folder + '/val_log_acc.txt'), 'w')


    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    normalize = Normalize(mean=mean, std=std)

    spatial_transform = Compose([Scale(256), RandomHorizontalFlip(), MultiScaleCornerCrop([1, 0.875, 0.75, 0.65625], 224),
                                 ToTensor(), normalize])

    vid_seq_train = makeDataset(trainDatasetDir,spatial_transform=spatial_transform,
                               sequence=False, numSeg=1, stackSize=stackSize, fmt='.jpg', seqLen=seqLen)

    train_loader = torch.utils.data.DataLoader(vid_seq_train, batch_size=trainBatchSize,
                            shuffle=True, num_workers=4, pin_memory=True)

    if valDatasetDir is not None:

        vid_seq_val = makeDataset(valDatasetDir,
                                   spatial_transform=Compose([Scale(256), CenterCrop(224), ToTensor(), normalize]),
                                   sequence=False, numSeg=1, stackSize=stackSize, fmt='.jpg', phase='Test',
                                   seqLen=seqLen)

        val_loader = torch.utils.data.DataLoader(vid_seq_val, batch_size=valBatchSize,
                                shuffle=False, num_workers=2, pin_memory=True)
        valSamples = vid_seq_val.__len__()

    model = twoStreamAttentionModel(flowModel=flowModel, frameModel=rgbModel, stackSize=stackSize, memSize=memSize,
                                    num_classes=num_classes)

    for params in model.parameters():
        params.requires_grad = False

    model.train(False)
    train_params = []

    for params in model.classifier.parameters():
        params.requires_grad = True
        train_params += [params]

    for params in model.frameModel.lstm_cell.parameters():
        train_params += [params]
        params.requires_grad = True

    for params in model.frameModel.resNet.layer4[0].conv1.parameters():
        params.requires_grad = True
        train_params += [params]

    for params in model.frameModel.resNet.layer4[0].conv2.parameters():
        params.requires_grad = True
        train_params += [params]

    for params in model.frameModel.resNet.layer4[1].conv1.parameters():
        params.requires_grad = True
        train_params += [params]

    for params in model.frameModel.resNet.layer4[1].conv2.parameters():
        params.requires_grad = True
        train_params += [params]

    for params in model.frameModel.resNet.layer4[2].conv1.parameters():
        params.requires_grad = True
        train_params += [params]
    #
    for params in model.frameModel.resNet.layer4[2].conv2.parameters():
        params.requires_grad = True
        train_params += [params]
    #
    for params in model.frameModel.resNet.fc.parameters():
        params.requires_grad = True
        train_params += [params]

    base_params = []
    for params in model.flowModel.layer4.parameters():
        base_params += [params]
        params.requires_grad = True

    model.cuda()

    trainSamples = vid_seq_train.__len__()
    min_accuracy = 0

    loss_fn = nn.CrossEntropyLoss()
    optimizer_fn = torch.optim.SGD([
        {'params': train_params},
        {'params': base_params, 'lr': 1e-4},
    ], lr=lr1, momentum=0.9, weight_decay=5e-4)

    optim_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_fn, step_size=decay_step, gamma=decay_factor)
    train_iter = 0

    for epoch in range(numEpochs):
        optim_scheduler.step()
        epoch_loss = 0
        numCorrTrain = 0
        iterPerEpoch = 0
        model.classifier.train(True)
        model.flowModel.layer4.train(True)
        for j, (inputFlow, inputFrame, targets) in enumerate(train_loader):
            train_iter += 1
            iterPerEpoch += 1
            optimizer_fn.zero_grad()
            inputVariableFlow = Variable(inputFlow.cuda())
            inputVariableFrame = Variable(inputFrame.permute(1, 0, 2, 3, 4).cuda())
            labelVariable = Variable(targets.cuda())
            output_label = model(inputVariableFlow, inputVariableFrame)
            loss = loss_fn(F.log_softmax(output_label, dim=1), labelVariable)
            loss.backward()
            optimizer_fn.step()
            _, predicted = torch.max(output_label.data, 1)
            numCorrTrain += (predicted == targets.cuda()).sum()
            epoch_loss += loss.data[0]
        avg_loss = epoch_loss / iterPerEpoch
        trainAccuracy = (numCorrTrain / trainSamples) * 100
        print('Average training loss after {} epoch = {} '.format(epoch + 1, avg_loss))
        print('Training accuracy after {} epoch = {}% '.format(epoch + 1, trainAccuracy))
        writer.add_scalar('train/epoch_loss', avg_loss, epoch + 1)
        writer.add_scalar('train/accuracy', trainAccuracy, epoch + 1)
        train_log_loss.write('Training loss after {} epoch = {}\n'.format(epoch + 1, avg_loss))
        train_log_acc.write('Training accuracy after {} epoch = {}\n'.format(epoch + 1, trainAccuracy))
        if valDatasetDir is not None:
            if (epoch + 1) % 1 == 0:
                model.train(False)
                val_loss_epoch = 0
                val_iter = 0
                numCorr = 0
                for j, (inputFlow, inputFrame, targets) in enumerate(val_loader):
                    val_iter += 1
                    inputVariableFlow = Variable(inputFlow.cuda())
                    inputVariableFrame = Variable(inputFrame.permute(1, 0, 2, 3, 4).cuda())
                    labelVariable = Variable(targets.cuda())
                    output_label = model(inputVariableFlow, inputVariableFrame)
                    loss = loss_fn(F.log_softmax(output_label, dim=1), labelVariable)
                    val_loss_epoch += loss.data[0]
                    _, predicted = torch.max(output_label.data, 1)
                    numCorr += (predicted == labelVariable.data).sum()
                val_accuracy = (numCorr / valSamples) * 100
                avg_val_loss = val_loss_epoch / val_iter
                print('Val Loss after {} epochs, loss = {}'.format(epoch + 1, avg_val_loss))
                print('Val Accuracy after {} epochs = {}%'.format(epoch + 1, val_accuracy))
                writer.add_scalar('val/epoch_loss', avg_val_loss, epoch + 1)
                writer.add_scalar('val/accuracy', val_accuracy, epoch + 1)
                val_log_loss.write('Val Loss after {} epochs = {}\n'.format(epoch + 1, avg_val_loss))
                val_log_acc.write('Val Accuracy after {} epochs = {}%\n'.format(epoch + 1, val_accuracy))
                if val_accuracy > min_accuracy:
                    save_path_model = (model_folder + '/model_twoStream_state_dict.pth')
                    torch.save(model.state_dict(), save_path_model)
                    min_accuracy = val_accuracy
        else:
            if (epoch + 1) % 10 == 0:
                save_path_model = (model_folder + '/model_twoStream_state_dict_epoch' + str(epoch + 1) + '.pth')
                torch.save(model.state_dict(), save_path_model)
    train_log_loss.close()
    train_log_acc.close()
    val_log_acc.close()
    val_log_loss.close()
    writer.export_scalars_to_json(model_folder + "/all_scalars.json")
    writer.close()


def __main__():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='gtea61', help='Dataset')
    parser.add_argument('--flowModel', type=str, default='./experiments/gtea61/flow/best_model_state_dict.pth',
                        help='Flow model')
    parser.add_argument('--rgbModel', type=str, default='./experiments/gtea61/rgb/best_model_state_dict.pth',
                        help='RGB model')
    parser.add_argument('--trainDatasetDir', type=str, default='./dataset/gtea_warped_flow_61/split2/train',
                        help='Train set directory')
    parser.add_argument('--valDatasetDir', type=str, default=None,
                        help='Validation set directory')
    parser.add_argument('--outDir', type=str, default='experiments', help='Directory to save results')
    parser.add_argument('--seqLen', type=int, default=25, help='Length of sequence')
    parser.add_argument('--stackSize', type=int, default=5, help='Number of opticl flow images in input')
    parser.add_argument('--trainBatchSize', type=int, default=32, help='Training batch size')
    parser.add_argument('--valBatchSize', type=int, default=32, help='Validation batch size')
    parser.add_argument('--numEpochs', type=int, default=250, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate')
    parser.add_argument('--stepSize', type=float, default=1, help='Learning rate decay step')
    parser.add_argument('--decayRate', type=float, default=0.99, help='Learning rate decay rate')
    parser.add_argument('--memSize', type=int, default=512, help='ConvLSTM hidden state size')

    args = parser.parse_args()

    dataset = args.dataset
    flowModel = args.flowModel
    rgbModel = args.rgbModel
    trainDatasetDir = args.trainDatasetDir
    valDatasetDir = args.valDatasetDir
    outDir = args.outDir
    stackSize = args.stackSize
    seqLen = args.seqLen
    trainBatchSize = args.trainBatchSize
    valBatchSize = args.valBatchSize
    numEpochs = args.numEpochs
    lr1 = args.lr
    decay_step = args.stepSize
    decay_factor = args.decayRate
    memSize = args.memSize

    main_run(dataset, flowModel, rgbModel, stackSize, seqLen, memSize, trainDatasetDir, valDatasetDir, outDir,
             trainBatchSize, valBatchSize, lr1, numEpochs, decay_step, decay_factor)

__main__()
