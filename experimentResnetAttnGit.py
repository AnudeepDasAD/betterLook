import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchvision.transforms.functional as TF
from torch.autograd import Variable
import os
import pickle

from torch.utils.data import TensorDataset
from transformer_test import TransformerModel, PositionalEncoding, generate_square_subsequent_mask
from foolbox import PyTorchModel, accuracy, samples
from foolbox.attacks import LinfPGD
from torchvision import models


def check_robust_accuracy(success, epsilons):
    robust_accuracy = 1 - torch.mean(success, axis=-1, dtype=torch.float32)
    for eps, acc in zip(epsilons, robust_accuracy):
        print(f"  Linf norm ≤ {eps:<6}: {acc.item() * 100:4.1f} %")


def test_accuracy(net, dl):
    total_correct = 0.
    j = 1
    for x,t in dl:
        y = net(x.to(device))
        print(f't: {t}')
        if isinstance(y, int):
            return total_correct / (j * len(t))
        blah = torch.argmax(y.cpu(), dim=1)
        print(f'res: {blah}')
        total_correct += torch.sum(blah==t).cpu().item()
        print(f'Batch {j}: Accuracy so far: {total_correct / (j * len(t))}')
        j+=1
    return total_correct/len(dl.dataset) # ORIGINALLY DL.DATASET WHEN USING ACTUAL DL


def attack_accuracy(net, dl, ds, use_ds_for_t = False): # ORIGINALLY DS WAS NOT INCLUDED
    total_correct = 0.
    j = 1
    for x,t in dl:
        y = net(x.to(device))
        if use_ds_for_t:
            t = torch.tensor([ds.classes[val] for val in t])
        print(f't: {t}')
        if isinstance(y, int):
            return total_correct / (j * len(t))
        blah = torch.argmax(y.cpu(), dim=1)
        blah = torch.tensor([ds.classes[val] for val in blah])
        print(f'res: {blah}')
        total_correct += torch.sum(blah==t).cpu().item()
        print(f'Batch {j}: Accuracy so far: {total_correct / (j * len(t))}')
        j+=1
    return total_correct / (j * len(t)) # ORIGINALLY DL.DATASET WHEN USING ACTUAL DL


def draw(x):
    # '''Displays a flattened MNIST digit'''
    # with torch.no_grad():
    #     plt.imshow(x, cmap='gray');
    #     plt.axis('off');
     with torch.no_grad():
        plt.imshow(x.cpu().numpy().reshape((img_size,img_size)), cmap='gray')
        plt.axis('off')


def fgsm(net, x, t, optimizer=None, eps=0.01, targ=False):
    '''
        x_adv = FGSM(net, x, t, eps=0.01, targ=False)
        
        Performs the Fast Gradient Sign Method, perturbing each input by
        eps (in infinity norm) in an attempt to have it misclassified.
        
        Inputs:
          net    PyTorch Module object
          x      (D,I) tensor containing a batch of D inputs
          t      tensor of D corresponding class indices
          eps    the maximum infinity-norm perturbation from the input
          targ   Boolean, indicating if the FGSM is targetted
                   - if targ is False, then t is considered to be the true
                     class of the input, and FGSM will work to increase the cost
                     for that target
                   - if targ is True, then t is considered to be the target
                     class for the perturbation, and FGSM will work to decrease the
                     cost of the output for that target class
        
        Output:
          x_adv  tensor of a batch of adversarial inputs, the same size as x
    '''

    # You probably want to create a copy of x so you can work with it.
    x_adv = None
    if optimizer is None:
        x_adv = x.clone().to('cuda')
    
    # Forward pass
    y = net(x)
    if isinstance(y, int) and y == -1:
        return -1, -1, -1
    
    # loss
    loss = net.loss_fcn(y,t)
    if optimizer:
        optimizer.zero_grad()
    loss.backward()
    if optimizer:
        optimizer.step()
    else:
        # The gradients should be populated now because we did backward()
        # If targ == True, then we do gradient descent
        multiplier = 1
        if targ:
            multiplier = -1
        
        x_adv = x + multiplier*eps*torch.sign(x.grad)
    
    return y, loss, x_adv


import torch.nn as nn
from torchvision import models

class ConvLstm(nn.Module):
    def __init__(self, input_dim=784, hidden_size=64, lstm_layers=2, bidirectional=True, n_class=100, timesteps=4):
        super(ConvLstm, self).__init__()
        self.conv_model = My_conv(input_dim)
        # self.Lstm = Lstm(latent_dim, hidden_size, lstm_layers, bidirectional)
        # self.output_layer = nn.Sequential(
        #     nn.Linear(2 * hidden_size if bidirectional==True else hidden_size, n_class),
        #     nn.Softmax(dim=-1)
        # )
        # self.lstm = nn.LSTM(input_dim, hidden_size, lstm_layers, batch_first=True)
        ntokens = hidden_size  # transformer output
        self.emsize = 784  # embedding dimension
        d_hid = 200  # dimension of the feedforward network model in nn.TransformerEncoder
        nlayers = 2  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        nhead = 2  # number of heads in nn.MultiheadAttention
        dropout = 0.2  # dropout probability
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        bptt = 8
        self.transformer_model = TransformerModel(ntokens, self.emsize, nhead, d_hid, nlayers, dropout).to(device)

        self.fc1 = nn.Linear(hidden_size, n_class)
        self.timesteps=6
        self.loss_fcn = torch.nn.CrossEntropyLoss()
        self.losses = []
        self.to('cuda')
        # redacted for github

    def forward(self, x):
        batch_size = x.shape[0]
        new_xs = []
        # redacted for github
        return x
    
    
    def learn(self, dl, optimizer=None, epochs=10):
        '''
        net.learn(dl, optimizer=None, epochs=10)
        Train the network on the dataset represented by the DataLoader dl.
        The default optimizer is Adam().
        The targets for the dataset are assumed to be class indices.
        '''
        
        if optimizer is None:
            print('Need to specify an optimizer and loss function')
            return
        full_accs = []
        for epoch in tqdm(range(epochs)):
            total_loss = 0.
            count = 0.
            for x, t in dl:
              x = x.to('cuda') # for use with a GPU
              t = t.to('cuda')
#               y = self(x)
              # print(y.shape)
              # print(t.shape)
#               loss = self.loss_fcn(y, t)

              x.requires_grad=True
    
              y, loss, x_advs = fgsm(self,x,t, optimizer=optimizer, eps=0.3)
              if isinstance(y, int) and isinstance(loss, int) and isinstance(x_advs, int) \
                and y == loss == x_advs:
                continue

#               optimizer.zero_grad()
#               loss.backward(retain_graph=True)
#               optimizer.step()
              total_loss += loss.detach().cpu().numpy()
              count += 1.
        
            # Just do it once for the epoch to get an idea
            # new_res = self(x_advs)
            # new_res_labels = torch.argmax(new_res, dim=1) 
            # acc = torch.sum(torch.tensor([1 for i, p in enumerate(new_res_labels) if p==t[i]]))/len(new_res_labels)
            # print(f'Last x_advs accuracy in epoch number {epoch} is: {acc}')
            # full_accs.append(acc)
            self.losses.append(total_loss/len(dl))
#             if epoch % 100 == 0:
            print(f'Epoch: {epoch}, loss: {total_loss/count}')
            torch.save(model, f'trained_models/imgnette_attn_detrots/resnet_glance_imgnette_learned_attn_trial1_epoch{epoch}.pt')

            # acc = accuracy(model, (x_test,y_test))
            # print(f'Accuracy = {acc*100.:0.2f}%')
        plt.figure(figsize=(4,4))
        plt.plot(self.losses); plt.yscale('log')
        plt.plot(full_accs); plt.yscale('log')


class My_conv(nn.Module):
    def __init__(self, latent_dim):
        super(My_conv, self).__init__()
        self.conv_model = prepareModel()
        self.conv_model.eval()
        # self.conv_model = torch.load('mnist_CNN_aug16.pt').to('cuda')
        print(self.conv_model)
        # ====== freezing all of the layers ======
        for param in self.conv_model.parameters():
            param.requires_grad = False
        # ====== changing the last FC layer to an output with the size we need. this layer is un freezed ======
        self.conv_model.fc = nn.Linear(self.conv_model.fc.in_features, latent_dim)
        # self.conv_model.classifier[6] = nn.Linear(self.conv_model.classifier[6].in_features, latent_dim)

        # self.conv_model.lyrs[2] = nn.Linear(self.conv_model.lyrs[2].in_features, latent_dim)
        # self.conv_model.lyrs[3] = nn.ReLU()

        # self.conv_model.fc2 = nn.Linear(self.conv_model.fc2.in_features, latent_dim)
        # self.final_layer = nn.Sequential(nn.ReLU())
        # print(self.conv_model)

    def forward(self, x):
        # print(type(x))
        return self.conv_model(x)


class Lstm(nn.Module):
    def __init__(self, latent_dim, hidden_size, lstm_layers, bidirectional):
        super(Lstm, self).__init__()
        self.Lstm = nn.LSTM(latent_dim, hidden_size=hidden_size, num_layers=lstm_layers, batch_first=True, bidirectional=bidirectional)
        self.hidden_state = None

    def reset_hidden_state(self):
        self.hidden_state = None

    def forward(self,x):
        output, self.hidden_state = self.Lstm(x, self.hidden_state)
        return output


def prepareModel():
    weights=models.ResNet50_Weights.IMAGENET1K_V2
    base_model = models.resnet50(weights=weights)
    return base_model

def prepareDatasets():
    ds = torch.load('imagenette_train_preproc_dataset0.pt')
    dl = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=True)
    ds_test = torch.load('imagenette_preproc_valset.pt')
    dl_test = torch.utils.data.DataLoader(ds_test, batch_size=8, shuffle=True)

    return dl, dl_test, ds_test


def getAttackDatasets(model_name, attack_name, custom = False):
    # suffixes = [80,160,240,320,400,480,560,625]
    suffixes = [80,160,240,320,400,480,491]
    if custom:
        suffixes = suffixes[:-1]
    sets = []
    for suffix in suffixes:
        # Loading gives us a list of size 2. The first list contains the sample, second is the label
        sets.append(list(zip(*torch.load(f'x_advs_{model_name}_imagenette_fool_{attack_name}_eps16255_{suffix}.pt'))))
        # print(len(sets[-1][0]))
        # break
    return sets

# transform = T.ToPILImage()


if __name__ == '__main__':
    # print(os.listdir('..'))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dl, dl_test, ds_test = prepareDatasets()

    # model = ConvLstm()
    # model.learn(dl, optimizer=torch.optim.Adam(model.parameters(), lr=0.0001),epochs=10)
    # torch.save(model, 'trained_models/imgnette_attn_detrots/resnet_glance_imgnette_learned_attn_final_trial1.pt')

    model = torch.load('trained_models\imgnette_attn_detrots\\resnet_glance_imgnette_learned_attn_trial1_epoch8.pt')
    # acc = test_accuracy(model, dl_test)

    # model = prepareModel().eval()
    model_name = 'resnetglancedetattn'
    attack_name = 'pgd'
    attack_batchsets = getAttackDatasets(model_name, attack_name, custom=True)
    for i, batchset in enumerate(attack_batchsets):
        acc = attack_accuracy(model.cuda(), batchset, ds_test, use_ds_for_t=True)
        print(f'acc for batch {i}: {acc}')

