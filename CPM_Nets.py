import util.classfiy as classfiy
import torch
import numpy as np
from numpy.random import shuffle
from util.util import xavier_init
from util.CPM import CPMNets
from torch.autograd import Variable
import torch.nn as nn


class CPMNet_Works(nn.Module):
    """build model
    """
    def __init__(self, device, view_num, trainLen, testLen, layer_size, lsd_dim=128, learning_rate=[0.001, 0.001], lamb=1):
        """
        :param learning_rate:learning rate of network and h
        :param view_num:view number
        :param layer_size:node of each net
        :param lsd_dim:latent space dimensionality
        :param trainLen:training dataset samples
        :param testLen:testing dataset samples
        """
        super(CPMNet_Works, self).__init__()
        # initialize parameter
        self.view_num = view_num
        self.layer_size = layer_size
        self.lsd_dim = lsd_dim
        self.trainLen = trainLen
        self.testLen = testLen
        self.lamb = lamb
        self.learning_rate = learning_rate
        self.device = device
        # initialize latent space data
        self.h_train = self.H_init('train')
        self.h_test = self.H_init('test')
        self.h = torch.cat([self.h_train, self.h_test], axis=0).cuda()
        # initialize nets for different views
        self.net, self.train_net_op = self.bulid_model()
        
    def H_init(self, a):
        if a == 'train':
            h = Variable(xavier_init(self.trainLen, self.lsd_dim), requires_grad = True)
        elif a == 'test':
            h = Variable(xavier_init(self.testLen, self.lsd_dim), requires_grad = True)
        return h

    def reconstruction_loss(self,h,x,sn):#输入为train或test的h，目标x，转化为字典的sn
        loss = 0
        x_pred = self.calculate(h.cuda())
        for num in range(self.view_num):
            loss = loss + (torch.pow((x_pred[str(num)].cpu() - x[str(num)].cpu())
                       , 2.0) * sn[str(num)].cpu()
            ).sum()
        return loss

    def classification_loss(self,label_onehot, gt, h_temp):#标签、经过训练过的h
        h_temp = h_temp.float()
        h_temp = h_temp.cuda()
        F_h_h = torch.mm(h_temp, (h_temp.T))
        F_hn_hn = torch.eye(F_h_h.shape[0],F_h_h.shape[1])
        F_h_h = F_h_h - F_h_h * (F_hn_hn.cuda())
        label_num = label_onehot.sum(0, keepdim=True)  # should sub 1.Avoid numerical errors; the number of samples of per label
        label_onehot = label_onehot.float()
        F_h_h_sum = torch.mm(F_h_h, label_onehot)
        F_h_h_mean = F_h_h_sum / label_num
        gt1 = torch.max(F_h_h_mean, axis=1)[1]  # gt begin from 1
        gt_ = gt1.type(torch.IntTensor) + 1
        F_h_h_mean_max = torch.max(F_h_h_mean, axis=1, keepdim=False)[0]
        gt_ = gt_.cuda()
        gt_ = gt_.reshape([gt_.shape[0],1])
        theta = torch.ne(gt, gt_).type(torch.FloatTensor)
        F_h_hn_mean_ = F_h_h_mean * label_onehot
        F_h_hn_mean = F_h_hn_mean_.sum(axis=1)
        F_h_h_mean_max = F_h_h_mean_max.reshape([F_h_h_mean_max.shape[0],1])
        F_h_hn_mean = F_h_hn_mean.reshape([F_h_hn_mean.shape[0],1])
        theta = theta.cuda()
        return (torch.nn.functional.relu(theta + F_h_h_mean_max - F_h_hn_mean)).sum()

    def train(self, data, sn, label_onehot, gt, epoch, step=[5, 5]):
        global Reconstruction_LOSS
        index = np.array([x for x in range(self.trainLen)])
        shuffle(index)
        gt = gt.cuda()
        label_onehot = label_onehot.cuda()
        sn1 = dict()
        data1 = dict()
        for v_num in range(self.view_num):
            data1[str(v_num)] = torch.from_numpy(data[str(v_num)]).cuda() 
        for i in range(self.view_num):
            sn1[str(i)] = sn[:, i].reshape(self.trainLen, 1).cuda()
        train_hn_op = torch.optim.Adam([self.h_train], self.learning_rate[1])
        for iter in range(epoch):
            for i in range(step[0]):
                Reconstruction_LOSS = self.reconstruction_loss(self.h_train,data1,sn1).float()
                for v_num in range(self.view_num):
                    self.train_net_op[v_num].zero_grad()
                    Reconstruction_LOSS.backward(retain_graph=True)
                    self.train_net_op[v_num].step()
            for i in range(step[1]):
                loss1 = self.reconstruction_loss(self.h_train,data1,sn1).float().cuda() 
                loss2 = self.lamb * self.classification_loss(label_onehot, gt, self.h_train).float().cuda()
                train_hn_op.zero_grad()
                loss1.backward()
                loss2.backward()
                train_hn_op.step()
            Classification_LOSS = self.classification_loss(label_onehot,gt,self.h_train)
            Reconstruction_LOSS = self.reconstruction_loss(self.h_train,data1,sn1)
            output = "Epoch : {:.0f}  ===> Reconstruction Loss = {:.4f}, Classification Loss = {:.4f} " \
                .format((iter + 1), Reconstruction_LOSS, Classification_LOSS)
            print(output)
        return (self.h_train)

    def bulid_model(self):
        # initialize network
        net = dict()
        train_net_op = []
        for v_num in range(self.view_num):
            net[str(v_num)] = CPMNets(self.view_num, self.trainLen, self.testLen, self.layer_size, v_num,
            self.lsd_dim, self.learning_rate, self.lamb).cuda()
            train_net_op.append(torch.optim.Adam([{"params":net[str(v_num)].parameters()}], self.learning_rate[0]))
        return net,train_net_op

    def calculate(self,h):
        h_views = dict()
        for v_num in range(self.view_num):
            h_views[str(v_num)] = self.net[str(v_num)](h.cuda())
        return h_views

    def test(self, data, sn, epoch):
        sn1 = dict()
        data1 = dict()
        for v_num in range(self.view_num):
            data1[str(v_num)] = torch.from_numpy(data[str(v_num)]).cuda() 
        for i in range(self.view_num):
            sn1[str(i)] = sn[:, i].reshape(self.testLen, 1).cuda()
        adj_hn_op = torch.optim.Adam([self.h_test], self.learning_rate[0])
        for iter in range(epoch):
            # update the h
            for i in range(5):
                Reconstruction_LOSS = self.reconstruction_loss(self.h_test, data1, sn1).float()
                adj_hn_op.zero_grad()
                Reconstruction_LOSS.backward()
                adj_hn_op.step()
            Reconstruction_LOSS = self.reconstruction_loss(self.h_test, data1, sn1).float()
            output = "Epoch : {:.0f}  ===> Reconstruction Loss = {:.4f}" \
                .format((iter + 1), Reconstruction_LOSS)
            print(output)
        return self.h_test

    

