import torch
from tqdm import tqdm
import numpy as np
from sklearn.svm import SVC
from sklearn import metrics


def train_vae_classic(net, optimizer, epocs, training_loader, kl_weight, sim_name='Vanilla_VAE', conditional=False):
    best_loss = 1e10
    net.cuda()
    net.train()
    for epoch in range(epocs):
        i=0
        for _, _, _, cell_image, label, _ in tqdm(training_loader):
            optimizer.zero_grad()
            x = cell_image.float().cuda()
            if conditional:
                y = net(x, label.float().cuda())
            else:
                y = net(x)
            loss = net.loss_function(*y, M_N=kl_weight)

            loss['loss'].backward()
            optimizer.step()
            if loss['loss'] < best_loss:
                torch.save(net, sim_name+'_BL.pt')
            if i % 20 ==0:
                print('\nEpoch: ' + str(epoch) + '\nLoss: ' + str(loss['loss']) + '\nReconstruction loss: ' +str(loss['Reconstruction_Loss']) + '\nKLD loss: ' + str(loss['KLD']))
            i +=1
    torch.save(net, sim_name+'_Latest.pt')

    print('Done training VAE..')


def train_svm(net, training_loader,sim_name, labeled_size=3000, image_size=64, conditional=False):
    class_0 = 0
    class_1 = 0
    svm_img_lst = np.zeros((labeled_size*2, 3, image_size, image_size))
    if conditional:
        svm_lbl_lst0 = np.zeros((2*labeled_size, 2))
    svm_lbl_lst = np.zeros((2*labeled_size,))
    print('getting data to train the SVM')
    idx = 0
    flg = False
    net.eval()
    for _, _, _, cell_image, label, _ in training_loader:
        for i in range(label.shape[0]):
            if class_0 < labeled_size and np.argmax(label[i]) == 0:
                svm_img_lst[idx] = cell_image[i].detach().numpy()
                svm_lbl_lst[idx] = 0
                class_0 += 1
                idx += 1
                flg = True
            elif class_1 < labeled_size and np.argmax(label[i]) == 1:
                svm_img_lst[idx] = cell_image[i].detach().numpy()
                svm_lbl_lst[idx] = 1
                class_1 += 1
                idx += 1
                flg = True
            if flg and conditional:
                svm_lbl_lst0[idx-1] = label[i].detach().numpy()
                flg = False

    print('Done building data to train the SVM..')
    print('training SVM..')
    svm = SVC(kernel='rbf', gamma='scale')
    net.cpu()
    x = torch.from_numpy(svm_img_lst).float().cpu()
    if conditional:
        y = torch.from_numpy(svm_lbl_lst0).float().cpu()
        encoded = net.encode(x, y)
    else:
        encoded = net.encode(x)
    z = net.reparameterize(*encoded)
    svm.fit(z.detach().numpy(), svm_lbl_lst)
    torch.save(svm, sim_name + '_SVM.pt')


def evaluate_vae_classic(vae, svm, test_loader, conditional=False):
    print('Evaluating the model..')
    vae.eval()
    total_y = []
    total_predictions = []
    for _, _, _, cell_image, label, _ in tqdm(test_loader):
        vae.cpu()
        x = cell_image.float().cpu()
        if conditional:
            encoded = vae.encode(x, label.float().cpu())
        else:
            encoded = vae.encode(x)
        z = vae.reparameterize(*encoded)
        current_predict = svm.predict(z.detach().numpy())
        total_predictions += current_predict.tolist()
        total_y += torch.argmax(label, dim=1).float()
    y_true = np.asarray(np.greater_equal(np.array(total_y), 0.5), int)
    cm = metrics.confusion_matrix(y_true, total_predictions)
    print('--------------------')
    print(cm[0][:] / (cm[0].sum()))
    print(cm[1][:] / (cm[1].sum()))
    print('--------------------')
    return y_true, total_predictions


def evaluate_losses(vae, test_loader, kl_weight):
    print('Evaluating the model losses..')
    general_losses = []
    reconst_losses = []
    vae.eval()
    vae.cpu()
    for _, _, _, cell_image, label, _ in tqdm(test_loader):
        vae.cpu()
        x = cell_image.float().cpu()
        y = vae(x)
        loss = vae.loss_function(*y, M_N=kl_weight)
        general_losses.append(float(loss['loss'].detach().numpy()))
        reconst_losses.append(float(loss['Reconstruction_Loss'].detach().numpy()))
    general_losses.sort()
    reconst_losses.sort()
    th_gen_loss = general_losses[int(len(general_losses)*0.99)]
    th_reconst_loss = reconst_losses[int(len(reconst_losses) * 0.99)]
    print('General loss stats: ' + str(general_losses[0]) + '\t\t' + str(general_losses[-1]) + '\t\t' + str(th_gen_loss))
    print('Reconst loss stats: ' + str(reconst_losses[0]) + '\t\t' + str(reconst_losses[-1]) + '\t\t' + str(th_reconst_loss))
    return (th_reconst_loss, th_gen_loss, reconst_losses[0], general_losses[0])


def evaluate_vae_anomaly(vae,gen_th, reconst_th, test_loader, kl_weight):
    print('Evaluating the model..')
    vae.eval()
    total_y = []
    total_general_predictions = []
    tot_reconst_pred = []
    for _, _, _, cell_image, label, _ in tqdm(test_loader):
        vae.cpu()
        x = cell_image.float().cpu()
        z = vae(x)
        loss = vae.loss_function(*z, M_N=kl_weight)
        total_general_predictions.append(loss['loss'].detach().numpy() >= gen_th)
        tot_reconst_pred.append(loss['Reconstruction_Loss'].detach().numpy())
        total_y += torch.argmax(label, dim=1).float()
    y_true = np.asarray(np.greater_equal(np.array(total_y), 0.5), int)
    cm = metrics.confusion_matrix(y_true, total_general_predictions)
    print('--------------------')
    print(cm[0][:] / (cm[0].sum()))
    print(cm[1][:] / (cm[1].sum()))
    print('--------------------')
    return y_true, total_general_predictions, tot_reconst_pred
