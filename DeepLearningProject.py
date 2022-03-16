import os
# os.system('pip3 install torch==1.10.1+cu102 torchvision==0.11.2+cu102 torchaudio===0.10.1+cu102 -f https://download.pytorch.org/whl/cu102/torch_stable.html')
import pickle

import matplotlib.pyplot as plt
import torch

import DataGenerator
import DataVisualization
import DeepSet
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm, trange
from torch import optim
import numpy as np
import tensorflow_addons as tfa

import DeepSetExtended
import FocalLoss
from sklearn import metrics
#from tqdm.notebook import tqdm_notebook as tqdm
import ridnik_asymetric_loss as ASL
#    _____ _       _           _                                          _
#   / ____| |     | |         | |                                        | |
#  | |  __| | ___ | |__   __ _| |    _ __   __ _ _ __ __ _ _ __ ___   ___| |_ ___ _ __ ___
#  | | |_ | |/ _ \| '_ \ / _` | |   | '_ \ / _` | '__/ _` | '_ ` _ \ / _ \ __/ _ \ '__/ __|
#  | |__| | | (_) | |_) | (_| | |   | |_) | (_| | | | (_| | | | | | |  __/ ||  __/ |  \__ \
#   \_____|_|\___/|_.__/ \__,_|_|   | .__/ \__,_|_|  \__,_|_| |_| |_|\___|\__\___|_|  |___/
#                                   | |
#                                   |_|
SEED = 7
LEARNING_RATE = 0.001
BATCH_SIZE = 16
IMG_CHANNELS = 1
CLASSES = 2
EPOCHS = 100
DEBUG = True
TRACKS_INPUT_DIM = 3  # theta, phi, track_val
TRACKS_DEEPSET_STRUCTURE = [256, 960, 600, 256, 100, 50]
TRACKS_FC_NETWORK_STRUCTURE = [128, 32, 8, CLASSES]
NUM_OF_PKL_FILES = -1
NETWORK_OUT_DIR = 'NetworkParameters/'
LOAD_BEST_LOSS_MODEL_FOR_EVAL = True
IGNORE_TRAINING = False
IGNORE_EVAL = False
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

#   _______        _       _
#  |__   __|      (_)     (_)
#     | |_ __ __ _ _ _ __  _ _ __   __ _
#     | | '__/ _` | | '_ \| | '_ \ / _` |
#     | | | | (_| | | | | | | | | | (_| |
#     |_|_|  \__,_|_|_| |_|_|_| |_|\__, |
#                                   __/ |
#                                  |___/

def NormalizeBatch(BatchData):
    """x = BatchData[:,:,0]
    x = x - x.mean()
    x = x / x.var()
    BatchData[:, :, 0] = x
    x = BatchData[:, :, 1]
    x = x - x.mean()
    x = x / x.var()
    BatchData[:, :, 1] = x
    x = BatchData[:,:,2]
    x = x - x.mean()
    x = x / x.var()
    BatchData[:, :, 2] = x"""
    return BatchData


def adjust_loss_function(output, y, cur_gamma, gamma_neg=1, gain=0.05, epsilon=0.05):
    output = torch.softmax(output, dim=-1)
    output = output.cpu()
    bg_array = output[:, 0].detach().numpy()
    sig_array = output[:, 1].detach().numpy()
    """for i in range(len(y)):
        pt_pos[i] = """
    votes_bg = np.where(bg_array > 0.5)
    votes_sig = np.where(sig_array > 0.5)

    bg_confidence = bg_array[votes_bg].mean()
    sig_confidence = sig_array[votes_sig].mean()
    delta_conf = bg_confidence - sig_confidence
    next_gamma = cur_gamma
    if not np.isnan(delta_conf):
        print('Delta P: ' + str(delta_conf) + ' BG P: ' + str(bg_confidence) + ' SIG P ' + str(sig_confidence))
        if True:
            if abs(delta_conf) > epsilon:
                next_gamma = cur_gamma + gain*delta_conf
    loss = ASL.ASLSingleLabel(gamma_pos=1.0, gamma_neg=next_gamma)
    return loss, next_gamma


def train_deep_set(net_name='default_name'):
# if not IGNORE_TRAINING:
    min_loss = 10000
    loss_vs_epoch = []
    validation_history = []
    train_history = []
    min_validation_loss = (-1, 1000.)
    for epoch in range(EPOCHS):

        train_batch_sampler = DataGenerator.BatchSampler(np.array(DataGenerator.training_ds.event_size), BATCH_SIZE)
        data_loader = DataLoader(DataGenerator.training_ds, batch_sampler=train_batch_sampler)
        net.train()
        batch_losses = []
        gamma = 1.0
        global loss_func
        for x, y, cells, cells_img in tqdm(data_loader):
            DataVisualization.plot_single_sample(tracks=x[0],cell=cells_img[0], label=np.array([1]), metadata= {'event_num':0, 'sample_num':0}, dot_size=40)
            x = NormalizeBatch(x.float())
            x = x.float()
            cells = cells.float().cuda()
            x = x.cuda()
            y = y.float().cuda()
            optimizer.zero_grad()
            output = net(x, cells)
            loss = loss_func(output, y.long())
            batch_losses.append(loss.item())
            loss.backward()
            optimizer.step()
            #loss_func, gamma = adjust_loss_function(output, cur_gamma=gamma)

        loss_vs_epoch.append([epoch, np.mean(batch_losses)])
        print(loss_vs_epoch[-1])
        print('Gamma: ' + str(gamma))
        valid_batch_sampler = DataGenerator.BatchSampler(np.array(DataGenerator.test_ds.event_size), BATCH_SIZE)
        valid_data_loader = DataLoader(DataGenerator.test_ds, batch_sampler=valid_batch_sampler)
        net.eval()
        valid_losses = []
        total_predictions = []
        total_y = []
        total_pt = []
        with torch.no_grad():
            for x, y, cells, cells_img in tqdm(valid_data_loader):
                x = NormalizeBatch(x.float())
                cells = cells.float().cuda()
                x = x.float().cuda()
                y = y.float().cuda()
                output = net(x, cells)
                loss = loss_func(output, y.long())
                valid_losses.append(loss.item())
                output = torch.softmax(output, dim=1)
                if DataGenerator.SIMPLE_LABEL:
                    total_predictions += output[:, 1].cpu()
                    total_y += y.cpu()
                    y_true = total_y
                else:
                    total_predictions += torch.argmax(output.cpu(), dim=1).float()
                    total_y += torch.argmax(y.cpu(), dim=1).float()
                    y_true = np.asarray(np.greater_equal(np.array(total_y), 0.5), int)

                #total_pt += pt.cpu()
        y = np.asarray(np.greater_equal(np.array(total_predictions), 0.5), int)
        cm = metrics.confusion_matrix(y_true, y)
        print('--------------------')
        print(cm[0][:]/(cm[0].sum()))
        print(cm[1][:]/(cm[1].sum()))
        print('--------------------')
        valid_loss = np.mean(valid_losses)
        train_history.append(loss_vs_epoch[-1][1])
        validation_history.append(valid_loss)
        print('validation_loss' + str(valid_loss))
        if valid_loss < min_validation_loss[1]:
            min_validation_loss = (epoch, loss_vs_epoch[-1][1])
            print('new min loss:', min_validation_loss)
            torch.save(net, NETWORK_OUT_DIR + net_name + 'BL' + '.pt')

    torch.save(net, NETWORK_OUT_DIR + net_name + str(EPOCHS) + ' epocs' + '.pt')
    return loss_vs_epoch, validation_history


if IGNORE_TRAINING:
    validation_history = []

#  ______          _             _   _
# |  ____|        | |           | | (_)
# | |____   ____ _| |_   _  __ _| |_ _  ___  _ __
# |  __\ \ / / _` | | | | |/ _` | __| |/ _ \| '_ \
# | |___\ V / (_| | | |_| | (_| | |_| | (_) | | | |
# |______\_/ \__,_|_|\__,_|\__,_|\__|_|\___/|_| |_|

def evaluate_model(simulation_title='', net_name='default_name', validation_history = [], loss_vs_epoch = []):
#if not IGNORE_EVAL:
    if LOAD_BEST_LOSS_MODEL_FOR_EVAL:
        net = torch.load(NETWORK_OUT_DIR + net_name + 'BL' + '.pt')
    net.eval()
    with torch.no_grad():
        train_batch_sampler = DataGenerator.BatchSampler(np.array(DataGenerator.test_ds.event_size), BATCH_SIZE)
        data_loader = DataLoader(DataGenerator.test_ds, batch_sampler=train_batch_sampler)
        total_predictions = []
        total_y = []
        for x, y, cells, cells_img in tqdm(data_loader):
            x = NormalizeBatch(x.float())
            x = x.float().cuda()
            cells = cells.float().cuda()
            output = torch.softmax(net(x, cells), dim=1)
            if DataGenerator.SIMPLE_LABEL:
                total_predictions += output[:, 1].cpu()
                total_y += y.cpu()
            else:
                total_predictions += torch.argmax(output.cpu(), dim=1).float()
                total_y += torch.argmax(y.cpu(), dim=1).float()
    print(f'Number of parameters - {sum(p.numel() for p in net.parameters())}')
    mat = DataVisualization.plot_simulation_summary(total_predictions, total_y, loss_vs_epoch, validation_history,
                                              simulation_title)
    return mat


#   _   _      _                      _
#  | \ | |    | |                    | |
#  |  \| | ___| |___      _____  _ __| | __
#  | . ` |/ _ \ __\ \ /\ / / _ \| '__| |/ /
#  | |\  |  __/ |_ \ V  V / (_) | |  |   <
#  |_| \_|\___|\__| \_/\_/ \___/|_|  |_|\_\
#
#

# simulation/regression
pos_gamma = [1]
#neg_gamma = [7.3369]
neg_gamma = [1]
loss_alpha = [0.01]
learning_rates = [0.0001]

# LoadDatasets
DataGenerator.training_ds, DataGenerator.test_ds = DataGenerator.generate_data_structures(NUM_OF_PKL_FILES)


"""
with open('training_ds.pkl', 'rb') as fi:
    DataGenerator.training_ds = pickle.load(fi)
with open('test_ds.pkl', 'rb') as fi2:
    DataGenerator.test_ds = pickle.load(fi2)


with open('training_ds_balanced.pkl', 'rb') as fi:
    DataGenerator.training_ds = pickle.load(fi)
with open('test_ds_balanced.pkl', 'rb') as fi2:
    DataGenerator.test_ds = pickle.load(fi2)
"""
"""n_sig = 0
n_bg = 0

for x, y, _ in DataGenerator.test_ds:
    if y == 0:
        continue
        _ = plt.hist(x[:, 0], bins='auto')
        plt.title("Histogram eta bg")
        plt.show()
        _ = plt.hist(x[:,1],bins='auto')
        plt.title("Histogram phi bg")
        plt.show()
        _ = plt.hist(x[:, 2], bins='auto')
        plt.title("Histogram tracks bg")
        plt.show()
    if y == 1:
        _ = plt.hist(x[:, 1], bins='auto')
        plt.title("Histogram eta sig")
        plt.show()
        _ = plt.hist(x[:,1],bins='auto')
        plt.title("Histogram phi sig")
        plt.show()
        _ = plt.hist(x[:, 2], bins='auto')
        plt.title("Histogram tracks sig")
        plt.show()

"""


for lr in learning_rates:
    for alpha in pos_gamma:
        for gamma in neg_gamma:
            #net = DeepSet.DeepSet(TRACKS_INPUT_DIM, TRACKS_DEEPSET_STRUCTURE, TRACKS_FC_NETWORK_STRUCTURE)
            net = DeepSetExtended.DeepSetExtended(3, 4, [512, 128, 32], [32, 32, 32], [64, 256, 64, 16, 2])
            #net = torch.load('NetworkParameters/lr_1e-05_al_1_gma_1new_loss100 epocs.pt')
            #loss_func = FocalLoss.FocalLoss(gamma=2, alpha=0.2, logits=True)
            #loss_func = ASL.AsymmetricLossOptimized(gamma_neg=gamma, gamma_pos=alpha, eps=0.5)
            #loss_func = ASL.ASLSingleLabel(gamma_pos=alpha, gamma_neg=gam. ma)
            weight_neg = 20.0 / 560.0
            weights = torch.FloatTensor([weight_neg, 1]).cuda()
            loss_func = nn.CrossEntropyLoss()
            optimizer = optim.Adadelta(net.parameters(), lr=lr)
            #optimizer = optim.Adam(net.parameters(), lr=lr)
            net = net.cuda()
            name = 'lr_'+str(lr)+'adadelta_with_cells_unbalanced'
            losses, validation_history = train_deep_set(net_name=name)
            cm = evaluate_model(simulation_title=name, net_name=name, loss_vs_epoch=losses, validation_history=validation_history)
