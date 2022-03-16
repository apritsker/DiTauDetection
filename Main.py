import matplotlib.colors
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
import numpy as np

import DitauCNN
import DitauDataGenerator
import DitauDataLoader
from DitauCNN import CellCNN
from DitauDeepSet import DeepSet
from DitauDeepSetExtended import DeepSetExtended
from DitauCombinedNet import DitauCombinedNet
import DataVisualization
import torchvision.models as models
import FocalLoss
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pickle
import random
from sklearn import metrics
from matplotlib import colors
import TripleDeepset
import DiTauCombined

from VAE_models import *

import DiTauVAE

MODE = 'TrainEval'
# MODE = 'Compare'
# MODE = 'sample'
# MODE = 'GenerateDataset'
# MODE = 'TrainEvalVAE'
#MODE = 'Eval'
# MODE = 'AnomalyDetection'


data_dir = './../data/'
samples_data_dir = data_dir + 'output/'
signals_data_dir = data_dir + 'new_output/'
tar_out = data_dir + 'DitauFineRes.tar.gz'

current_config_name = 'TracksOnly'
CELL_IMAGE_SIZE = (64, 64, 3)
SIMPLE_LABEL = True

if __name__ == '__main__':

    if MODE == 'GenerateDataset':
        ## PREPARING DATASET ##
        signal_csv = data_dir + 'signal.csv'
        background_csv = data_dir + 'background.csv'
        DitauDataGenerator.GenerateDitauData(signal_csv, background_csv, samples_data_dir, tar_out,
                                             image_size=CELL_IMAGE_SIZE, generate_tar=False, retain_extracted=True,
                                             equal_num_background_to_samples=False)
        exit()

    #    _____ _       _           _                                        _
    #   / ____| |     | |         | |                                      | |
    #  | |  __| | ___ | |__   __ _| |    _ __   __ _ _ __ __ _ _ __ ___   ___| |_ ___ _ __ ___
    #  | | |_ | |/ _ \| '_ \ / _` | |   | '_ \ / _` | '__/ _` | '_ ` _ \ / _ \ __/ _ \ '__/ __|
    #  | |__| | | (_) | |_) | (_| | |   | |_) | (_| | | | (_| | | | | | |  __/ ||  __/ |  \__ \
    #   \_____|_|\___/|_.__/ \__,_|_|   | .__/ \__,_|_|  \__,_|_| |_| |_|\___|\__\___|_|  |___/
    #                                   | |
    #
    SEED = 7
    LEARNING_RATE = 0.00002
    BATCH_SIZE = 32
    IMG_CHANNELS = 1
    CLASSES = 2
    EPOCHS = 70
    DEBUG = True
    TRACKS_INPUT_DIM = 3  # theta, phi, track_val
    EM_INPUT_DIM = 3
    HAD_INPUT_DIM = 3
    TRACKS_DEEPSET_STRUCTURE = [256, 64, 16]  # [256, 256, 30]
    TRACKS_MH_ATTENTION_STRUCTURE = [16, 16, 16, 16, 16, 16]
    CELLS_INPUT_DIM = 3
    CELLS_DEEPSET_STRUCTURE = TRACKS_DEEPSET_STRUCTURE
    CELLS_MH_ATTENTION_STRUCTURE = TRACKS_MH_ATTENTION_STRUCTURE
    EXTENDED_DS_FC_NN_STRUCT = [32, 256, 64, 16, CLASSES]
    TRIPLE_DS_FC_NN_STRUCT = [48, 512, 128, 64, 32, 8, CLASSES]
    TRACKS_FC_NETWORK_STRUCTURE = [128, 32, 8, CLASSES]

    NETWORK_OUT_DIR = 'NetworkParameters/'
    LOAD_BEST_LOSS_MODEL_FOR_EVAL = True
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    SIMPLE_LABEL = False


    def split_dataset(dataset, validation_split=0.2, batch_size=BATCH_SIZE):
        shuffle_dataset = True
        random_seed = 42

        # Creating data indices for training and validation splits:
        dataset_size = len(dataset)
        indices = list(range(dataset_size - 1))
        split = int(np.floor(validation_split * dataset_size))
        if shuffle_dataset:
            np.random.seed(random_seed)
            np.random.shuffle(indices)
        train_indices, valid_indices = np.array(indices[split:]), np.array(indices[:split])

        tracks_same_len = np.array([dataset.max_tracks_padding for i in range(len(dataset.track_sizes - 1))])
        train_batch_sampler = DitauDataLoader.SplitBatchSampler(tracks_same_len, batch_size, train_indices)
        valid_batch_sampler = DitauDataLoader.SplitBatchSampler(tracks_same_len, batch_size, valid_indices)

        train_data_loader = DataLoader(dataset, batch_sampler=train_batch_sampler)
        valid_data_loader = DataLoader(dataset, batch_sampler=valid_batch_sampler)

        return train_data_loader, valid_data_loader


    dataset = DitauDataLoader.DiTauDataset(samples_data_dir)

    if (MODE == 'sample'):
        idxs = [1, 299, 784, 3056, 11000, 14869, 12324]
        for idx in idxs:
            t, cells, x, y, m = dataset[idx]
            plt.imshow(x[:, :, 0],
                       cmap="nipy_spectral",
                       vmin=0.0,
                       vmax=1.0
                       )
            plt.title('EM for ' + str(idx))
            plt.colorbar()
            plt.show()
            plt.imshow(x[:, :, 1],
                       cmap="nipy_spectral",
                       vmin=0.0,
                       vmax=1.0
                       )
            plt.title('Had for ' + str(idx))
            plt.colorbar()
            plt.show()
            plt.imshow(x[:, :, 2],
                       cmap="nipy_spectral",
                       vmin=0.0,
                       vmax=1.0
                       )
            plt.title('Tracks for ' + str(idx))
            plt.colorbar()
            plt.show()
        exit()

    history_outfile_name = NETWORK_OUT_DIR + current_config_name + ' loss hisory' + '.pkl'

    #    __      __     ______
    #    \ \    / /\   |  ____|
    #     \ \  / /  \  | |__
    #      \ \/ / /\ \ |  __|
    #       \  / ____ \| |____
    #        \/_/    \_\______|

    if MODE == 'AnomalyDetection':
        epocs = 10
        lr = 0.002
        sim_name = 'dip_VAE_Anomaly'
        # net = dip_vae.DIPVAE(in_channels=3, latent_dim=128, lambda_diag=0.05, lambda_offdiag=0.1)
        # net = betatc_vae.BetaTCVAE(3, 128)
        # net = mssim_vae.MSSIMVAE(in_channels=3, latent_dim=1024)
        net = VanillaVAE(in_channels=3, latent_dim=512)
        optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)

        # optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        train_loader, _ = split_dataset(dataset, validation_split=0.1, batch_size=32)
        # DiTauVAE.train_vae_classic(net, optimizer, epocs, train_loader, 0.5, sim_name)
        test_ds = DitauDataLoader.DiTauDataset(signals_data_dir)
        train_loader, _ = split_dataset(dataset, validation_split=0.0, batch_size=1)
        test_loader, _ = split_dataset(test_ds, validation_split=0.0, batch_size=1)
        vae = vae = torch.load(sim_name + '_BL.pt')
        general_th, reconst_th, _, _ = DiTauVAE.evaluate_losses(vae, test_loader=train_loader, kl_weight=0.000025)
        # DiTauVAE.evaluate_vae_anomaly(vae,5.353584289550781, 5.353584289550781, test_loader, 0.000025)
        DiTauVAE.evaluate_vae_anomaly(vae, general_th, reconst_th, test_loader, 0.000025)

    if MODE == 'TrainEvalVAE':
        epocs = 10
        learning_rate = 0.0005
        sim_name = 'Vanilla_VAE'
        net = VanillaVAE(3, 128)
        optimizer = torch.optim.Adagrad(net.parameters(), lr=learning_rate)
        train_loader, valid_data_loader = split_dataset(dataset)
        # DiTauVAE.train_vae_classic(net, optimizer, epocs, train_loader, 0.000025)
        # vae = torch.load(sim_name+'_BL.pt')
        # DiTauVAE.train_svm(vae, train_loader, sim_name)
        # svm = torch.load(sim_name + '_SVM.pt')
        # DiTauVAE.evaluate_vae_classic(vae, svm, valid_data_loader)
        #################################
        """"sim_name = 'ConditionalVAE'
        learning_rate = 0.0005
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
        net = cvae.ConditionalVAE(in_channels=3, num_classes=2, latent_dim=10)
        DiTauVAE.train_vae_classic(net, optimizer, epocs, train_loader, 1.0, sim_name, conditional=True)
        vae = torch.load(sim_name + '_BL.pt')
        DiTauVAE.train_svm(vae, train_loader, sim_name, conditional=True)
        svm = torch.load(sim_name + '_SVM.pt')
        DiTauVAE.evaluate_vae_classic(vae, svm, valid_data_loader, conditional=True)"""
        ##################################
        """sim_name = 'BetaTCVAE'
        net = betatc_vae.BetaTCVAE(3, 10)
        DiTauVAE.train_vae_classic(net, optimizer, epocs, train_loader, 0.0005, sim_name)
        vae = torch.load(sim_name + '_BL.pt')
        DiTauVAE.train_svm(vae, train_loader, sim_name)
        svm = torch.load(sim_name + '_SVM.pt')
        DiTauVAE.evaluate_vae_classic(vae, svm, valid_data_loader)
        ##################################
        sim_name = 'CategoricalVAE'
        net = cat_vae.CategoricalVAE(3, 10)
        epocs = 20
        DiTauVAE.train_vae_classic(net, optimizer, epocs, train_loader, 0.0005, sim_name)
        vae = torch.load(sim_name + '_BL.pt')
        DiTauVAE.train_svm(vae, train_loader, sim_name)
        svm = torch.load(sim_name + '_SVM.pt')
        DiTauVAE.evaluate_vae_classic(vae, svm, valid_data_loader)"""
        #################################
        """sim_name = 'DFCVAE'
        net = dfcvae.DFCVAE(3, 10)
        DiTauVAE.train_vae_classic(net, optimizer, epocs, train_loader, 0.0005, sim_name)
        vae = torch.load(sim_name + '_BL.pt')
        DiTauVAE.train_svm(vae, train_loader, sim_name)
        svm = torch.load(sim_name + '_SVM.pt')
        DiTauVAE.evaluate_vae_classic(vae, svm, valid_data_loader)"""
        #################################
        """sim_name = 'DIPVAE'
        net = dip_vae.DIPVAE(in_channels=3, latent_dim=128, lambda_diag=0.05, lambda_offdiag=0.1)
        epocs = 20
        optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
        DiTauVAE.train_vae_classic(net, optimizer, epocs, train_loader, 1.0, sim_name)
        vae = torch.load(sim_name + '_BL.pt')
        DiTauVAE.train_svm(vae, train_loader, sim_name)
        svm = torch.load(sim_name + '_SVM.pt')
        DiTauVAE.evaluate_vae_classic(vae, svm, valid_data_loader)"""
        #################################
        """sim_name = 'FactorVAE'
        net = fvae.FactorVAE(in_channels=3, latent_dim=128)
        epocs = 20
        optimizer = torch.optim.Adagrad(net.parameters(), lr=0.0002)
        DiTauVAE.train_vae_classic(net, optimizer, epocs, train_loader, 1.0, sim_name)
        vae = torch.load(sim_name + '_BL.pt')
        DiTauVAE.train_svm(vae, train_loader, sim_name)
        svm = torch.load(sim_name + '_SVM.pt')
        DiTauVAE.evaluate_vae_classic(vae, svm, valid_data_loader)"""
        #################################
        """sim_name = 'GammaVAE' # this VAE does not converge well 
        net = gamma_vae.GammaVAE(in_channels=3, latent_dim=128)
        epocs = 10
        optimizer = torch.optim.Adam(net.parameters(), lr=0.002)
        DiTauVAE.train_vae_classic(net, optimizer, epocs, train_loader, 1.0, sim_name)
        vae = torch.load(sim_name + '_BL.pt')
        DiTauVAE.train_svm(vae, train_loader, sim_name)
        svm = torch.load(sim_name + '_SVM.pt')
        DiTauVAE.evaluate_vae_classic(vae, svm, train_loader)"""
        ###################################
        sim_name = 'MSSIMVAE'  # this VAE does not converge well
        net = mssim_vae.MSSIMVAE(in_channels=3, latent_dim=1024)
        epocs = 10
        optimizer = torch.optim.Adam(net.parameters(), lr=0.002)
        DiTauVAE.train_vae_classic(net, optimizer, epocs, train_loader, 0.00025, sim_name)
        vae = torch.load(sim_name + '_BL.pt')
        DiTauVAE.train_svm(vae, train_loader, sim_name)
        svm = torch.load(sim_name + '_SVM.pt')
        DiTauVAE.evaluate_vae_classic(vae, svm, train_loader)

    #   _______        _       _
    #  |__   __|      (_)     (_)
    #     | |_ __ __ _ _ _ __  _ _ __   __ _
    #     | | '__/ _` | | '_ \| | '_ \ / _` |
    #     | | | | (_| | | | | | | | | | (_| |
    #     |_|_|  \__,_|_|_| |_|_|_| |_|\__, |
    #                                   __/ |
    #                                  |___/

    if MODE == 'Train' or MODE == 'TrainEval':
        net = TripleDeepset.TripleDeepSet(TRACKS_INPUT_DIM, EM_INPUT_DIM, HAD_INPUT_DIM, TRACKS_DEEPSET_STRUCTURE,
                                          TRACKS_MH_ATTENTION_STRUCTURE, TRIPLE_DS_FC_NN_STRUCT)

        deepset = TripleDeepset.DeepSet(TRACKS_INPUT_DIM, TRACKS_DEEPSET_STRUCTURE, TRACKS_MH_ATTENTION_STRUCTURE)
        cnn = DitauCNN.CellCNN(0, (3, 64, 64), classify=False)
        #net = DiTauCombined.CombinedNet(deepset=deepset, image_cnn=cnn, classifier_layers=(1552, 512, 128, 32, 2))
        loss_vs_epoch = []
        loss_func = FocalLoss.FocalLoss(gamma=4.0, alpha=0.1)
        # loss_func = nn.CrossEntropyLoss()
        net = torch.load(NETWORK_OUT_DIR + current_config_name + '.pt')
        optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
        net = net.cuda()
        train_loader, valid_data_loader = split_dataset(dataset)
        min_loss = 10000
        train_history = []
        validation_history = []
        min_validation_loss = (-1, 1000.)
        net.train()

        for epoch in range(EPOCHS):
            # net.train()
            batch_losses = []
            print("EPOCH " + str(epoch))
            for tracks, em, had, cell_image, label, _ in tqdm(train_loader):
                x = tracks.float().cuda()
                #em = em.float().cuda()
                #had = had.float().cuda()
                em = torch.zeros((x.shape[0], 189, 3))
                had = torch.zeros((x.shape[0], 189, 3))
                y = label.float().cuda()
                optimizer.zero_grad()
                output = net(x, em.float().cuda(), had.float().cuda())
                #output = net(x, cell_image.float().cuda())
                output = torch.softmax(output, dim=-1)
                loss = loss_func(output, y)
                batch_losses.append(loss.item())
                loss.backward()
                optimizer.step()
            loss_vs_epoch.append([epoch, np.mean(batch_losses)])
            print(loss_vs_epoch[-1])
            if loss_vs_epoch[-1][1] < min_validation_loss[1]:
                min_validation_loss = (epoch, loss_vs_epoch[-1][1])
                print('new min loss:', min_validation_loss)
                torch.save(net, NETWORK_OUT_DIR + current_config_name + '_intermediate' + '.pt')

            net.eval()
            valid_losses = []
            total_predictions = []
            total_y = []
            network_outputs_0 = []
            network_outputs_1 = []
            with torch.no_grad():
                for tracks, em, had, cell_image, label, _ in tqdm(valid_data_loader):
                    #em = em.float().cuda()
                    #had = had.float().cuda()
                    x = tracks.float().cuda()
                    em = torch.zeros((x.shape[0], 189, 3))
                    had = torch.zeros((x.shape[0], 189, 3))

                    y = label.float().cuda()
                    output = net(x, em.float().cuda(), had.float().cuda())
                    #output = net(x, cell_image.float().cuda())
                    output = torch.softmax(output, dim=1)
                    loss = loss_func(output, y)
                    valid_losses.append(loss.item())
                    if SIMPLE_LABEL:
                        total_predictions += output[:, 1].cpu()
                        total_y += y.cpu()
                        y_true = total_y
                    else:
                        total_predictions += torch.argmax(output.cpu(), dim=1).float()
                        total_y += torch.argmax(y.cpu(), dim=1).float()
                        y_true = np.asarray(np.greater_equal(np.array(total_y), 0.5), int)
                    network_outputs_1 += output[:, 1].cpu()
                    network_outputs_0 += output[:, 0].cpu()

                    # total_pt += pt.cpu()
            y = np.asarray(np.greater_equal(np.array(total_predictions), 0.5), int)
            np_net_out1 = np.array(network_outputs_1)
            np_net_out0 = np.array(network_outputs_0)
            if epoch % 10 == 0:
                DataVisualization.decision_histfit(np_net_out0, np_net_out1)
            cm = metrics.confusion_matrix(y_true, y)
            print('--------------------')
            print(cm[0][:] / (cm[0].sum()))
            print(cm[1][:] / (cm[1].sum()))
            print('--------------------')
            valid_loss = np.mean(valid_losses)
            train_history.append(loss_vs_epoch[-1][1])
            validation_history.append(valid_loss)
            print('validation_loss' + str(valid_loss))

        torch.save(net, NETWORK_OUT_DIR + current_config_name + '.pt')

        with open(history_outfile_name, 'wb') as fo:
            pickle.dump(loss_vs_epoch, fo)
            pickle.dump(validation_history, fo)

    #  ______          _             _   _
    # |  ____|        | |           | | (_)
    # | |____   ____ _| |_   _  __ _| |_ _  ___  _ __
    # |  __\ \ / / _` | | | | |/ _` | __| |/ _ \| '_ \
    # | |___\ V / (_| | | |_| | (_| | |_| | (_) | | | |
    # |______\_/ \__,_|_|\__,_|\__,_|\__|_|\___/|_| |_|

    if MODE == 'Eval' or MODE == 'TrainEval':
        _, valid_loader = split_dataset(dataset, batch_size=4)
        net = TripleDeepset.TripleDeepSet(TRACKS_INPUT_DIM, EM_INPUT_DIM, HAD_INPUT_DIM, TRACKS_DEEPSET_STRUCTURE,
                                          TRACKS_MH_ATTENTION_STRUCTURE, TRIPLE_DS_FC_NN_STRUCT)
        if LOAD_BEST_LOSS_MODEL_FOR_EVAL:
            #net = torch.load(NETWORK_OUT_DIR + 'triple_deepset_deeper_focal_loss' + '.pt')
            net = torch.load(NETWORK_OUT_DIR + current_config_name + '.pt')
        with open(history_outfile_name, 'rb') as fo:
            loss_vs_epoch = pickle.load(fo)
            validation_history = pickle.load(fo)
        net.eval()
        with torch.no_grad():
            total_predictions = []
            total_y = []
            for tracks, em, had, cell_image, label, _ in tqdm(valid_loader):
                #em = em.float().cuda()
                #had = had.float().cuda()
                x = tracks.float().cuda()
                em = torch.zeros((x.shape[0], 189, 3))
                had = torch.zeros((x.shape[0], 189, 3))
                y = label.float().cuda()
                output = net(x, em.float().cuda(), had.float().cuda())
                output = torch.softmax(output, dim=1)
                #total_predictions += torch.argmax(output.cpu(), dim=1).float()
                cur_decision = output[:,1].cpu().float()
                total_predictions += cur_decision
                total_y += torch.argmax(y.cpu(), dim=1).float()
                y_true = np.asarray(np.greater_equal(np.array(total_y), 0.5), int)
        DataVisualization.plot_simulation_summary(total_predictions, total_y, loss_vs_epoch, validation_history,
                                                  title='Single DeepSet - Tracks only')

        torch.save(total_predictions, 'total_pred_'+current_config_name+'.pt')
        torch.save(total_y, 'total_y_'+current_config_name+'.pt')
        exit()
        net = DiTauCombined.CombinedNet(deepset=deepset, image_cnn=cnn, classifier_layers=(1552, 512, 128, 32, 2))
        net.eval()
        net.cuda()

        total_predictions = []
        total_y = []
        net = torch.load(NETWORK_OUT_DIR + 'new_bg_deepset_tracks_only.pt')
        # net = DeepSetExtended(TRACKS_INPUT_DIM, CELLS_INPUT_DIM,TRACKS_DEEPSET_STRUCTURE, TRACKS_MH_ATTENTION_STRUCTURE,
        #                      EXTENDED_DS_FC_NN_STRUCT)
        net.eval()
        net.cuda()

        for tracks, _, _, _, label, _ in tqdm(valid_loader):
            # em = em.float().cpu()
            # had = had.float().cpu()
            x = tracks.float().cuda()
            y = label.float().cuda()
            cell = torch.zeros((x.shape[0], 1, 3)).float().cuda()
            output = net(x, cell)
            output = torch.softmax(output, dim=1)
            if SIMPLE_LABEL:
                total_predictions += output[:, 1].cpu()
                total_y += y.cpu()
                y_true = total_y
            else:
                cur_decision = output[:, 1].cpu().float()
                total_predictions += cur_decision
                total_y += torch.argmax(y.cpu(), dim=1).float()
                y_true = np.asarray(np.greater_equal(np.array(total_y), 0.5), int)
        total_predictions1 = torch.load('total_pred1.pt')
        total_y1 = torch.load('total_y1.pt')
        DataVisualization.plot_rocs_comparison([(total_predictions, total_y, 'DeepSet  Tracks only', 'dashed', 'c'),
                                                (total_predictions1, total_y1, 'DeepSet Tracks+Cells', 'dashed', 'g')])
        print(f'Number of parameters - {sum(p.numel() for p in net.parameters())}')
        DataVisualization.plot_simulation_summary(total_predictions, total_y, loss_vs_epoch, validation_history,
                                                  title='Single DeepSet')


    def eval(net, input_type, eval_dataloader):
        net.eval()
        with torch.no_grad():
            total_predictions = []
            total_y = []
            for tracks, em, had, cell_image, label, _ in tqdm(valid_loader):
                em = em.float().cuda()
                had = had.float().cuda()
                x = tracks.float().cuda()
                y = label.float().cuda()
                output = net(x, em, had)
                output = torch.softmax(output, dim=1)
                if SIMPLE_LABEL:
                    total_predictions += output[:, 1].cpu()
                    total_y += y.cpu()
                    y_true = total_y
                else:
                    total_predictions += torch.argmax(output.cpu(), dim=1).float()
                    total_y += torch.argmax(y.cpu(), dim=1).float()
                    y_true = np.asarray(np.greater_equal(np.array(total_y), 0.5), int)
        return total_predictions, total_y


    if MODE == 'Compare':
        # user should configure these four lists
        configs = ['', 'Deepset_TracksOnly_2_lr1e-4']
        datadirs = ['data/fineRes/', 'data/fineRes/']
        names = ['Simple CNN (Cells only)', 'Simple Deepset (Tracks only)']
        styles = ['dashed', 'dashdot']
        colors = ['c', 'g']
        inputs = ['img', 'tracks']

        roc_tuples = []
        for i in range(len(configs)):
            net = torch.load(NETWORK_OUT_DIR + configs[i] + '.pt')

            samples_data_dir = datadirs[i] + 'out/'
            dataset = DitauDataLoader.DiTauDataset(samples_data_dir)
            _, valid_loader = split_dataset(dataset)
            preds, trues = eval(net, inputs[i], valid_loader)
            roc_tuples.append((trues, preds, names[i], styles[i], colors[i]))

        DataVisualization.plot_rocs_comparison([(total_predictions, total_y, 'DeepSet  Tracks only', 'c'),
                                                (total_predictions1, total_y1, 'DeepSet Tracks+Cells', 'g')])

