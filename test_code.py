import numpy as np
from util.util import read_data
from util.get_sn import get_sn
from CPM_Nets import CPMNet_Works
import util.classfiy as classfiy
from sklearn.metrics import accuracy_score
import os
import warnings
import torch
warnings.filterwarnings("ignore")
device = torch.device('cuda:2')
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--lsd-dim', type=int, default=150,
                        help='dimensionality of the latent space data [default: 150]')
    parser.add_argument('--epochs-train', type=int, default=30, metavar='N',
                        help='number of epochs to train [default: 30]')
    parser.add_argument('--epochs-test', type=int, default=30, metavar='N',
                        help='number of epochs to test [default: 30]')
    parser.add_argument('--lamb', type=float, default=1,
                        help='trade off parameter [default: 1]')
    parser.add_argument('--missing-rate', type=float, default=0,
                        help='view missing rate [default: 0]')
    args = parser.parse_args()

    # read data
    trainData, testData, view_num = read_data('./data/PIE_face_10.mat', 0.8, 1)
    outdim_size = [trainData.data[str(i)].shape[1] for i in range(view_num)]
    # set layer size
    layer_size = [[300, outdim_size[i]] for i in range(view_num)]
    # set parameter
    epoch = [args.epochs_train, args.epochs_test]
    learning_rate = [0.01, 0.01]
    # Randomly generated missing matrix
    Sn = get_sn(view_num, trainData.num_examples + testData.num_examples, args.missing_rate)
    Sn_train = Sn[np.arange(trainData.num_examples)]
    Sn_test = Sn[np.arange(testData.num_examples) + trainData.num_examples]




    
    Sn = torch.LongTensor(Sn).cuda()
    Sn_train = torch.LongTensor(Sn_train).cuda()
    Sn_test = torch.LongTensor(Sn_test).cuda()

    # Model building
    model = CPMNet_Works(device, view_num, trainData.num_examples, testData.num_examples, layer_size, args.lsd_dim, learning_rate,
                    args.lamb).cuda()

        
    # train
    gt1 = trainData.labels.reshape(trainData.num_examples)
    gt1 = gt1.reshape([gt1.shape[0],1])
    gt1 = torch.LongTensor(gt1)
    class_num = (torch.max(gt1) - torch.min(gt1) + 1).cpu()
    batch_size = torch.tensor(gt1.shape[0])
    label_onehot = (torch.zeros(batch_size,class_num).scatter_(1,gt1 - 1,1)) # gt1 begin from 1 so we need to set the minimum of it to 0
    H_train = model.train(trainData.data, Sn_train, label_onehot, gt1, epoch[0])
    
    # test
    gt2 = testData.labels.reshape(testData.num_examples)
    gt2 = gt2.reshape([gt2.shape[0],1])
    gt2 = torch.LongTensor(gt2)
    H_test = model.test(testData.data, Sn_test, epoch[1])


    label_pre = classfiy.ave(H_train, H_test, label_onehot.cuda(), testData.num_examples)
    print('Accuracy on the test set is {:.4f}'.format(accuracy_score(testData.labels, label_pre)))