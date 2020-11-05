from __future__ import print_function, division
from modelSelfSupervised import *
from spatial_transforms import (Compose, ToTensor, CenterCrop, Scale, Normalize, MultiScaleCornerCrop,
                                RandomHorizontalFlip)
from torchvision.transforms import Resize
from tensorboardX import SummaryWriter
from makeDatasetMS import *
import argparse
import sys

DEVICE = "cpu"

def main_run(dataset, stage, train_data_dir, val_data_dir, stage1_dict, out_dir, seqLen, trainBatchSize,
             valBatchSize, numEpochs, lr1, decay_factor, decay_step, memSize, CAM=True):
    # GTEA 61
    num_classes = 61

    # Train/Validation/Test split
    train_splits = ["S1", "S3", "S4"]
    val_splits = ["S2"]


    model_folder = os.path.join('./', out_dir, dataset, 'rgb', 'stage'+str(stage))  # Dir for saving models and log files
    # Create the dir
    if os.path.exists(model_folder):
        print('Directory {} exists!'.format(model_folder))
        sys.exit()
    os.makedirs(model_folder)

    # Log files
    writer = SummaryWriter(model_folder)
    train_log_loss = open((model_folder + '/train_log_loss.txt'), 'w')
    train_log_acc = open((model_folder + '/train_log_acc.txt'), 'w')
    val_log_loss = open((model_folder + '/val_log_loss.txt'), 'w')
    val_log_acc = open((model_folder + '/val_log_acc.txt'), 'w')

    # Data loader
    normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    spatial_transform = Compose([Scale(256),
                                 RandomHorizontalFlip(),
                                 MultiScaleCornerCrop([1, 0.875, 0.75, 0.65625], 224),
                                 # ToTensor(),
                                 # normalize
                                 ])
    transform_rgb = Compose([ToTensor(), normalize])
    transform_MS  = Compose([Resize((7, 7)), ToTensor()])

    vid_seq_train = makeDataset(train_data_dir, splits=train_splits,
                                spatial_transform=spatial_transform,
                                transform_rgb=transform_rgb,
                                transform_MS=transform_MS,
                                seqLen=seqLen, fmt='.png')
    n_workers = 0 #4
    train_loader = torch.utils.data.DataLoader(vid_seq_train, batch_size=trainBatchSize,
                                               shuffle=True, num_workers=n_workers, pin_memory=True)

    vid_seq_val = makeDataset(train_data_dir, splits=val_splits,
                              spatial_transform=Compose([Scale(256), CenterCrop(224), ToTensor(), normalize]),
                              seqLen=seqLen, fmt='.png', verbose=False)

    val_loader = torch.utils.data.DataLoader(vid_seq_val, batch_size=valBatchSize,
                                             shuffle=False, num_workers=n_workers, pin_memory=True)
    valInstances = vid_seq_val.__len__()

    '''
    if val_data_dir is not None:

        vid_seq_val = makeDataset(val_data_dir,
                                  spatial_transform=Compose([Scale(256), CenterCrop(224), ToTensor(), normalize]),
                                  seqLen=seqLen, fmt='.jpg')

        val_loader = torch.utils.data.DataLoader(vid_seq_val, batch_size=valBatchSize,
                                                 shuffle=False, num_workers=2, pin_memory=True)
        valInstances = vid_seq_val.__len__()
    '''
    trainInstances = vid_seq_train.__len__()

    train_params = []

    model = SelfSupervisedAttentionModel(num_classes=num_classes, mem_size=memSize)

    model.train(False)
    for params in model.parameters():
        params.requires_grad = False
    # Add all the parameters of the model
    # ResNet feature extractor
    for params in model.resNet.layer4[0].conv1.parameters():
        params.requires_grad = True
        train_params += [params]

    for params in model.resNet.layer4[0].conv2.parameters():
        params.requires_grad = True
        train_params += [params]

    for params in model.resNet.layer4[1].conv1.parameters():
        params.requires_grad = True
        train_params += [params]

    for params in model.resNet.layer4[1].conv2.parameters():
        params.requires_grad = True
        train_params += [params]

    for params in model.resNet.layer4[2].conv1.parameters():
        params.requires_grad = True
        train_params += [params]
    #
    for params in model.resNet.layer4[2].conv2.parameters():
        params.requires_grad = True
        train_params += [params]
    #
    for params in model.resNet.fc.parameters():
        params.requires_grad = True
        train_params += [params]

    model.resNet.layer4[0].conv1.train(True)
    model.resNet.layer4[0].conv2.train(True)
    model.resNet.layer4[1].conv1.train(True)
    model.resNet.layer4[1].conv2.train(True)
    model.resNet.layer4[2].conv1.train(True)
    model.resNet.layer4[2].conv2.train(True)
    model.resNet.fc.train(True)

    # ConvLSTM parameters
    for params in model.lstm_cell.parameters():
        params.requires_grad = True
        train_params += [params]

    for params in model.classifier.parameters():
        params.requires_grad = True
        train_params += [params]

    model.lstm_cell.train(True)

    model.classifier.train(True)
    model.ms_module.train(True)
    model.to(DEVICE)

    loss_fn = nn.CrossEntropyLoss()
    loss_ms_fn = nn.CrossEntropyLoss()  # TODO: check paper Planamente

    optimizer_fn = torch.optim.Adam(train_params, lr=lr1, weight_decay=4e-5, eps=1e-4)

    optim_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_fn, milestones=decay_step,
                                                           gamma=decay_factor)

    train_iter = 0
    min_accuracy = 0

    for epoch in range(numEpochs):
        epoch_loss = 0
        numCorrTrain = 0
        trainSamples = 0
        iterPerEpoch = 0
        writer.add_scalar('lr', optimizer_fn.param_groups[0]['lr'], epoch + 1)

        model.train(True)
        # model.lstm_cell.train(True)
        # model.classifier.train(True)
        # model.resNet.layer4[0].conv1.train(True)
        # model.resNet.layer4[0].conv2.train(True)
        # model.resNet.layer4[1].conv1.train(True)
        # model.resNet.layer4[1].conv2.train(True)
        # model.resNet.layer4[2].conv1.train(True)
        # model.resNet.layer4[2].conv2.train(True)
        # model.resNet.fc.train(True)
        for i, (inputsRGB, inputsMS, targets) in enumerate(train_loader):
            # Inputs:
            #   - inputsRGB : the rgb frame input
            # Labels :
            #   - inputsMS  : the motion task label
            #   - targets   : output

            train_iter += 1
            iterPerEpoch += 1
            optimizer_fn.zero_grad()
            inputVariable = inputsRGB.permute(1, 0, 2, 3, 4).to(DEVICE)
            labelVariable = targets.to(DEVICE)
            msVariable = inputsMS.to(DEVICE)
            trainSamples += inputsRGB.size(0)
            output_label, _, output_ms = model(inputVariable)
            loss_c = loss_fn(output_label, labelVariable)
            # print(loss_c)
            loss_ms = loss_ms_fn(torch.reshape(output_ms, (seqLen * 7 * 7, 2, output_ms.size(0))),
                                 torch.reshape(msVariable, (seqLen * 7 * 7, msVariable.size(0))).long())
            # print(loss_ms)
            loss = loss_c + loss_ms
            # print(loss)
            # loss = loss_fn(output_label, labelVariable) + loss_ms_fn(output_ms, inputsMS) # TODO (forse): invertire 0 e 1 dim per inputsMS # output1 = F.softmax(torch.reshape(output_ms, (32, 7, 2, 7*7))[0, 0, :, :], dim=0)
            loss.backward()
            optimizer_fn.step()
            _, predicted = torch.max(output_label.data, 1)
            numCorrTrain += (predicted == targets.to(DEVICE)).sum()
            epoch_loss += loss.data.item()
        avg_loss = epoch_loss/iterPerEpoch
        trainAccuracy = (numCorrTrain.data.item() / trainSamples) * 100

        train_log_loss.write('Training loss after {} epoch = {}\n'.format(epoch + 1, avg_loss))
        train_log_acc.write('Training accuracy after {} epoch = {}\n'.format(epoch + 1, trainAccuracy))
        print('Train: Epoch = {} | Loss = {} | Accuracy = {}'.format(epoch+1, avg_loss, trainAccuracy))
        writer.add_scalar('train/epoch_loss', avg_loss, epoch+1)
        writer.add_scalar('train/accuracy', trainAccuracy, epoch+1)

        # VALIDATION PHASE
        #if val_data_dir is not None:
        if (epoch+1) % 1 == 0:
            model.train(False)
            val_loss_epoch = 0
            val_iter = 0
            val_samples = 0
            numCorr = 0
            for j, (inputs, inputsMS, targets) in enumerate(val_loader):
                val_iter += 1
                val_samples += inputsRGB.size(0)
                inputVariable = inputsRGB.permute(1, 0, 2, 3, 4).to(DEVICE) # la permutazione è a solo scopo di computazione
                labelVariable = targets.to(DEVICE)
                msVariable = inputsMS.to(DEVICE)
                output_label, _, output_ms = model(inputVariable)
                loss_c = loss_fn(output_label, labelVariable)
                loss_ms = loss_ms_fn(torch.reshape(output_ms, (seqLen * 7 * 7, 2, output_ms.size(0))),
                                 torch.reshape(msVariable, (seqLen * 7 * 7, msVariable.size(0))).long())
                val_loss = loss_c + loss_ms
                # val_loss = loss_fn(output_label, labelVariable) # TODO: add ms Loss
                val_loss_epoch += val_loss.data.item()
                _, predicted = torch.max(output_label.data, 1)
                numCorr += (predicted == targets.to(DEVICE)).sum()
            val_accuracy = (numCorr.data.item() / val_samples) * 100
            avg_val_loss = val_loss_epoch / val_iter
            print('Valid: Epoch = {} | Loss {} | Accuracy = {}'.format(epoch + 1, avg_val_loss, val_accuracy))
            writer.add_scalar('val/epoch_loss', avg_val_loss, epoch + 1)
            writer.add_scalar('val/accuracy', val_accuracy, epoch + 1)
            val_log_loss.write('Val Loss after {} epochs = {}\n'.format(epoch + 1, avg_val_loss))
            val_log_acc.write('Val Accuracy after {} epochs = {}%\n'.format(epoch + 1, val_accuracy))
            if val_accuracy > min_accuracy:
                save_path_model = (model_folder + '/model_rgb_state_dict.pth')
                torch.save(model.state_dict(), save_path_model)
                min_accuracy = val_accuracy
            '''else:
                if (epoch+1) % 10 == 0:
                    save_path_model = (model_folder + '/model_rgb_state_dict_epoch' + str(epoch+1) + '.pth')
                    torch.save(model.state_dict(), save_path_model)
                '''
        optim_scheduler.step()


    train_log_loss.close()
    train_log_acc.close()
    val_log_acc.close()
    val_log_loss.close()
    writer.export_scalars_to_json(model_folder + "/all_scalars.json")
    writer.close()


def __main__():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='gtea61', help='Dataset')
    parser.add_argument('--stage', type=int, default=1, help='Training stage')
    parser.add_argument('--trainDatasetDir', type=str, default='./dataset/gtea_warped_flow_61/split2/train',
                        help='Train set directory')
    parser.add_argument('--valDatasetDir', type=str, default=None,
                        help='Val set directory')
    parser.add_argument('--outDir', type=str, default='experiments', help='Directory to save results')
    parser.add_argument('--stage1Dict', type=str, default='./experiments/gtea61/rgb/stage1/best_model_state_dict.pth',
                        help='Stage 1 model path')
    parser.add_argument('--seqLen', type=int, default=25, help='Length of sequence')
    parser.add_argument('--trainBatchSize', type=int, default=32, help='Training batch size')
    parser.add_argument('--valBatchSize', type=int, default=64, help='Validation batch size')
    parser.add_argument('--numEpochs', type=int, default=300, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--stepSize', type=float, default=[25, 75, 150], nargs="+", help='Learning rate decay step')
    parser.add_argument('--decayRate', type=float, default=0.1, help='Learning rate decay rate')
    parser.add_argument('--memSize', type=int, default=512, help='ConvLSTM hidden state size')
    parser.add_argument('--CAM', type=str, default='y', help='C Attention Maps')

    args = parser.parse_args()

    dataset = args.dataset
    stage = args.stage
    trainDatasetDir = args.trainDatasetDir
    valDatasetDir = args.valDatasetDir
    outDir = args.outDir
    stage1Dict = args.stage1Dict
    seqLen = args.seqLen
    trainBatchSize = args.trainBatchSize
    valBatchSize = args.valBatchSize
    numEpochs = args.numEpochs
    lr1 = args.lr
    stepSize = args.stepSize
    decayRate = args.decayRate
    memSize = args.memSize
    CAM = args.CAM == 'y'

    main_run(dataset, stage, trainDatasetDir, valDatasetDir, stage1Dict, outDir, seqLen, trainBatchSize,
             valBatchSize, numEpochs, lr1, decayRate, stepSize, memSize, CAM)

__main__()
