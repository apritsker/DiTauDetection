import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.mlab as mlab
import itertools
import numpy as np
from sklearn import metrics
from DataGenerator import SIMPLE_LABEL
from scipy.stats import norm

def plot_confusion_matrix(cm, ax, classes=['BkGd', 'signal'],
                          title='Test Confusion matrix',
                          cmap='Set3'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    ax.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, round(cm[i, j], 4),
                 horizontalalignment="center",
                 color="black" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def plot_history(train_loss, validation_loss, title='Convergence Graph', labels =('Training Loss','Validation Loss'),
                      x_axis='Epochs', y_axis='Loss'):
    plt.figure()
    epochs_range = range(0, len(train_loss))
    plt.plot(epochs_range, train_loss, label=labels[0])
    plt.plot(epochs_range, validation_loss, label=labels[1])
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.legend()
    plt.title(title)
    plt.show()


def plot_roc(total_y, total_predictions, ax, title='Test AUC', x_label='False Positive Rate',
             y_label='True Positive Rate', linestyle='dashdot', name='Classifier', plot_noskill=True):
    fpr, tpr, thresholds = metrics.roc_curve(np.array(total_y), np.array(total_predictions))
    ax.plot([0, 1], [0, 1], 'm', linestyle='--', label='No Skill')
    ax.plot(fpr, tpr, 'c', linestyle=linestyle, label=name)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend()
    return fpr, tpr, thresholds


def subplot_loss(loss_vs_epoch, validation_history, ax, title='Convergence Graph',
                 labels=('Training Loss', 'Validation Loss'), x_axis='Epochs', y_axis='Loss'):
    loss_vs_epoch_train = [loss_vs_epoch[i][1] for i in range(0, len(loss_vs_epoch))]
    loss_vs_epoch_validation = [validation_history[i] for i in range(0, len(loss_vs_epoch))]
    ax.plot(loss_vs_epoch_train, 'c', linestyle='dashdot', label=labels[0])
    ax.plot(loss_vs_epoch_validation, 'm', linestyle='--', label=labels[1])
    ax.set_title(title)
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    ax.legend()


def plot_simulation_summary(total_predictions, total_y, loss_vs_epoch, validation_history, title):
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    subplot_loss(loss_vs_epoch, validation_history, ax1)
    if SIMPLE_LABEL:
        y_true = total_y
    else:
        y_true = np.asarray(np.greater_equal(np.array(total_y), 0.5), int)
    y = np.asarray(np.greater_equal(np.array(total_predictions), 0.5), int)
    plot_roc(y_true, y, ax2)
    cm = metrics.confusion_matrix(y_true, y)
    plot_confusion_matrix(cm, ax3, title=title)
    f.tight_layout(pad=3.0)
    plt.show()
    return cm


def plot_single_sample(tracks, cell, label, metadata, dot_size):
    #img_size = (cell.shape[1], cell.shape[2])
    img_size = (cell.shape[0], cell.shape[1])
    fig, ax = plt.subplots()   
    
    plt.imshow(cell[:,:,0], cmap='Greens')
    plt.imshow(cell[:,:,1], cmap='Reds', alpha=0.5)
    plt.imshow(cell[:, :, 2], cmap='Blues', alpha=0.5)
    rect = patches.Rectangle((-0.5, -0.5), img_size[1], img_size[0], linewidth=1, edgecolor='black', facecolor='none')
    ax.add_patch(rect)

    #label_str = 'Background' if np.argwhere(label > 0).item() == 0 else 'Signal'
    label_str = ''
    plt.title("Event #" + str(metadata['event_num']) + ", Sample #" + str(metadata['sample_num']) + ", Label: " + label_str)
    #plt.scatter((tracks[:, 0] * 0.5 + 0.5) * cell.shape[1], (tracks[:, 1] * 0.5 + 0.5) * cell.shape[2], tracks[:, 2] * dot_size)
    plt.axis('off')
    plt.show()   


def plot_rocs_comparison(datalist, title='ROC Comparison'):
    """datalist: list where each element is a tuple of:
    ( vector of y_true, vector of y_predicted, name, style, color )
    """
    f, ax = plt.subplots()
    # ax.plot([0, 1], [0, 1], 'm', linestyle='dotted', label='No Skill')
    for y_true, y_pred, name, style, color in datalist:
        fpr, tpr, thresholds = metrics.roc_curve(np.array(y_true), np.array(y_pred))
        # ax.plot(fpr, tpr, color, linestyle=style, label=name)
        ind = np.where(fpr > 0.0001)
        ax.plot(tpr[ind], 1 / fpr[ind], color, linestyle=style, label=name)

    ax.set_title(title)
    ax.set_ylabel('Background Mistag Rate (1 / FPR)')
    # ax.set_xlabel('FPR')
    ax.set_xlabel('Signal Sensitivity (TPR)')
    plt.yscale('log')
    ax.legend()
    plt.show()


def plot_nn_score(y_true, y_pred):
    f, ax = plt.subplots()
    y_t = np.array(y_true)
    y_p = np.array(y_pred)
    ind_true_signal = np.where(y_t > 0.5)
    ind_true_background = np.where(y_p < 0.5)
    n, bins, patches = plt.hist(y_p[ind_true_signal], 50, density=True, facecolor='orange', label='True Signal')
    n, bins, patches = plt.hist(y_p[ind_true_background], 50, density=True, facecolor='blue', alpha=0.5, label='True Background')
    plt.title('Prediction distributions')
    plt.xlabel('Score')
    plt.ylabel('Prevalence')
    plt.legend()
    plt.show()

def decision_histfit(decision_0, decision_1, plot_type = 'norm'):
    mu0, sigma0 = norm.fit(decision_0)
    mu1, sigma1 = norm.fit(decision_1)
    x = np.arange(0.0, 1.0, 0.01)
    if plot_type == 'norm':
        norm0 = norm.pdf(x,mu0,sigma0)
        norm1 = norm.pdf(x,mu1,sigma1)
        prefix = 'normal fit for '
    elif plot_type == 'gamma':
        norm0 = norm.gamma.pdf(x, mu0, sigma0)
        norm1 = norm.gamma.pdf(x, mu1, sigma1)
        prefix = 'gamma fit for '
    else:
        norm0 = norm.beta.pdf(x, mu0, sigma0)
        norm1 = norm.beta.pdf(x, mu1, sigma1)
        prefix = 'beta fit for '
    max_val = max(max(norm0), max(norm1))
    norm0 = norm0/float(max_val)
    norm1 = norm1/float(max_val)
    plt.plot(x, norm0, label=prefix+"background votes")
    plt.plot(x, norm1, label=prefix + "signal votes")
    plt.title('Neural network outputs distributions')
    plt.grid(True)
    plt.legend(loc='best')
    plt.show()


def sample_visualization(x, idx):
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