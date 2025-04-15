import torch.nn as nn
import torch


class GLU(nn.Module):
    def __init__(self, input_num):
        super(GLU, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(input_num, input_num)

    def forward(self, x):
        lin = self.linear(x.permute(0, 2, 3, 1))
        lin = lin.permute(0, 3, 1, 2)
        sig = self.sigmoid(x)
        res = lin * sig
        return res


class ContextGating(nn.Module):
    def __init__(self, input_num):
        super(ContextGating, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(input_num, input_num)

    def forward(self, x):
        lin = self.linear(x.permute(0, 2, 3, 1))
        lin = lin.permute(0, 3, 1, 2)
        sig = self.sigmoid(lin)
        res = x * sig
        return res


class CNN(nn.Module):

    def __init__(self, n_in_channel, activation="Relu", conv_dropout=0,
                 kernel_size=[3, 3, 3], padding=[1, 1, 1], stride=[1, 1, 1], nb_filters=[64, 64, 64], pooling=[(1, 4), (1, 4), (1, 4)],pool_type="MaxPool"):
        super(CNN, self).__init__()
        self.nb_filters = nb_filters
        cnn = nn.Sequential()

        def conv(i, batchNormalization=False, dropout=None, activ="relu"):
            nIn = n_in_channel if i == 0 else nb_filters[i - 1]
            nOut = nb_filters[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, kernel_size[i], stride[i], padding[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut, eps=0.001, momentum=0.99))
            if activ.lower() == "leakyrelu":
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2))
            elif activ.lower() == "relu":
                cnn.add_module('relu{0}'.format(i), nn.ReLU())
            elif activ.lower() == "glu":
                cnn.add_module('glu{0}'.format(i), GLU(nOut))
            elif activ.lower() == "cg":
                cnn.add_module('cg{0}'.format(i), ContextGating(nOut))
            if dropout is not None:
                cnn.add_module('dropout{0}'.format(i),
                               nn.Dropout(dropout))

        batch_norm = True
        # 128x862x64
        for i in range(len(nb_filters)):

            conv(i, batch_norm, conv_dropout, activ=activation)
            if pooling[i] is not None:
                if pool_type=="MaxPool":

                    cnn.add_module('pooling{0}'.format(i), nn.MaxPool2d(pooling[i]))  # bs x tframe x mels
                elif pool_type=="AvgPool":
                    cnn.add_module('pooling{0}'.format(i), nn.AvgPool2d(pooling[i]))  # bs x tframe x mels
            self.cnn = cnn



    def forward(self, x):
        # input size : (batch_size, n_channels, n_frames, n_freq)
        # conv features
        x = self.cnn(x)
        return x

if __name__ == '__main__':


    cnn_kwargs = {"n_in_channel": 1,
                   "activation": "glu",
                   "conv_dropout": 0.5,
                   "kernel_size": 3 * [3], "padding": 3 * [1], "stride": 3 * [1], "nb_filters": [64, 128, 256],
                   "pooling": list(3 * ((2, 4),))}


    cnn_kwargs = {"n_in_channel": 1,
                   "activation": "glu",
                   "conv_dropout": 0.5,
                   "kernel_size": 6 * [3], "padding": 6 * [1], "stride": 6 * [1], "nb_filters": [64,64,128,128,256,256],
                   "pooling": list(3 * ((1, 2),(2,2)))}



    cnn_kwargs = {"n_in_channel": 1,
                   "activation": "glu",
                   "conv_dropout": 0.5,
                   "kernel_size": 3 * [3], "padding": 3 * [1], "stride": 3 * [1], "nb_filters": [64, 128, 256],
                   "pooling": list(3 * ((2, 4),))}




    cnn_kwargs = {"n_in_channel": 1,
                   "activation": "relu",
                   "conv_dropout":None,
                   "kernel_size": [5,5,3], "padding":   [2,2,1], "stride": 3 * [1], "nb_filters": [160,160,160],
                   "pooling":  [(1, 4), (1, 4), (1, 4)]}

    net=CNN(**cnn_kwargs)



    print(net)
    x=torch.randn(11,1,640,64)
    y=net(x)

    print(y.shape)
    # print(y[1].shape)




