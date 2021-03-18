# neuralnet.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/29/2019
# Modified by Mahir Morshed for the spring 2021 semester

"""
This is the main entry point for MP3. You should only modify code
within this file and neuralnet_part2.py -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class1 = 0
class2 = 19

class NeuralNet(nn.Module):
    def __init__(self, lrate, loss_fn, in_size, out_size):
        """
        Initializes the layers of your neural network.

        @param lrate: learning rate for the model
        @param loss_fn: A loss function defined as follows:
            @param yhat - an (N,out_size) Tensor
            @param y - an (N,) Tensor
            @return l(x,y) an () Tensor that is the mean loss
        @param in_size: input dimension
        @param out_size: output dimension

        For Part 1 the network should have the following architecture (in terms of hidden units):

        in_size -> 32 ->  out_size
        
        We recommend setting lrate to 0.01 for part 1.

        """
        super(NeuralNet, self).__init__()
        self.lrate = lrate
        self.loss_fn = loss_fn
        self.in_size = in_size
        self.out_size = out_size
        
        print('GELU SiLU .06')
        self.model = nn.Sequential(nn.Linear(in_size, 32), nn.SiLU(), nn.Linear(32, out_size))

    def set_parameters(self, params):
        """ Sets the parameters of your network.

        @param params: a list of tensors containing all parameters of the network
        """

        return
    
    def get_parameters(self):
        """ Gets the parameters of your network.

        @return params: a list of tensors containing all parameters of the network
        """
        return self.model.parameters()

    def forward(self, x):
        """Performs a forward pass through your neural net (evaluates f(x)).

        @param x: an (N, in_size) Tensor
        @return y: an (N, out_size) Tensor of output from the network
        """
        return self.model( (x - x.mean()) / x.std() )

    def step(self, x,y):
        """
        Performs one gradient step through a batch of data x with labels y.

        @param x: an (N, in_size) Tensor
        @param y: an (N,) Tensor
        @return L: total empirical risk (mean of losses) at this timestep as a float
        """
        ## optim.atoms l2 regularization
        optimSGD = optim.SGD(self.get_parameters(), self.lrate, weight_decay=0.06)
        lossData = self.loss_fn(self.forward(x), y)
        optimSGD.zero_grad()
        lossData.backward()
        optimSGD.step()

        return lossData.item()


def fit(train_set,train_labels,dev_set,n_iter,batch_size=100):
    """ Make NeuralNet object 'net' and use net.step() to train a neural net
    and net(x) to evaluate the neural net.

    @param train_set: an (N, in_size) Tensor
    @param train_labels: an (N,) Tensor
    @param dev_set: an (M,) Tensor
    @param n_iter: an int, the number of iterations of training
    @param batch_size: size of each batch to train on. (default 100)

    This method _must_ work for arbitrary M and N.

    @return losses: array of total loss at the beginning and after each iteration.
            Ensure that len(losses) == n_iter.
    @return yhats: an (M,) NumPy array of binary labels for dev_set
    @return net: a NeuralNet object
    """
    # play with lrate
    # learning rate: 0.01, loss function: CrossEntropyLoss, in size: train_set[0] size, out size: 2
    neuralNet = NeuralNet(0.01, nn.CrossEntropyLoss(), len(train_set[0]), 2)

    # training phase
    train_set = (train_set - train_set.mean()) / train_set.std()
    lossData = [0] * int(len(train_labels) / batch_size)
    for i in range(n_iter):
        batchBegin = 0
        for curBatch in range(int(len(train_labels) / batch_size) - 1):
            label = train_labels[batchBegin: batchBegin + batch_size]
            image = train_set[batchBegin: batchBegin + batch_size]
            lossData[curBatch] = neuralNet.step(image, label) # heavylifting is here
            batchBegin += batch_size
    

    # classification
    dev_set = (dev_set - dev_set.mean()) / dev_set.std()
    results = neuralNet(dev_set).detach().numpy()
    predictions = np.array([np.argmax(results[i]) for i in range(len(neuralNet(dev_set)))])

    return lossData, predictions, neuralNet
