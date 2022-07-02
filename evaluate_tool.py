"""
evaluate_tool.py
===============================================
This is the module for evaluate the model result
"""

from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,accuracy_score
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt 
import numpy as np
import random


def get_confusionmatrix(y_pred:np.array, y_true:np.array, classes_list:list, title:str, 
                       save_fig:bool = False, save_path:str = ""):
    """Generate confusion matrix for original classification.

    Args:
        y_pred (np.array): predicted label.
        y_true (np.array): true label.
        classes_list (list): list of classes name for corresponding prediction.
        title (str): graph title.
        save_fig (bool, optional): save the figure or not. Defaults to False.
        save_path (str, optional): path to save the graph. Defaults to "".
    """
    plt.rcParams["figure.figsize"] = (20,20)
    cm = confusion_matrix( np.argmax(y_true,axis=1), np.argmax(y_pred,axis=1))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=classes_list)
    disp.plot(xticks_rotation='vertical')
    plt.title(title)
    
    if save_fig:
        plt.savefig(save_path)

def derive_mucosal(y_pred, y_true):
    """Derive mucosal label.

    Args:
        y_pred (np.array): predicted label.
        y_true (np.array): ture label.
    """
    y_pred_mucosal = [0 if np.argmax(y)<5 else 1 for y in y_pred]
    y_true_mucosal = [0 if np.argmax(y)<5 else 1 for y in y_true]
    return y_pred_mucosal, y_true_mucosal   
    
def get_derived_mucosal(y_pred: np.array, y_true: np.array, title: str, 
                       save_fig: bool = False, save_path: bool = ""):
    """Generate confusion matrix for mucous / Non-Mucous prediction

    Args:
        y_pred (np.array): predicted label.
        y_true (np.array): true label.
        title (str): graph title.
        save_fig (bool, optional): save the figure or not. Defaults to False.
        save_path (bool, optional): path to save the graph. Defaults to "".
    """
    y_pred_mucosal = [0 if np.argmax(y)<5 else 1 for y in y_pred]
    y_train_mucosal = [0 if np.argmax(y)<5 else 1 for y in y_true]
    cm = confusion_matrix(y_train_mucosal, y_pred_mucosal)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=['Mucous','Non-Mucous'])
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=['Mucous','Non-Mucous'])
    disp.plot(xticks_rotation='vertical')
    plt.title(title)
    
    if save_fig:
        plt.savefig(save_path)

def get_classification_report(y_pred: np.array, y_true: np.array):
    """Generate classification report.

    Args:
        y_pred (np.array): predicted label.
        y_true (np.array): true label.

    Returns:
        str or dict: report representation.
    """
    return classification_report(y_pred, y_true)

def plot_instance(seen_series, actual_next, pred_next, axis, color_map):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))
    true_signal = np.concatenate([seen_series, actual_next]).T
    pred_signal = np.concatenate([seen_series, pred_next]).T

    for i in range(len(axis)):
        if i < 3:
            ax1.plot(true_signal[i], f'{color_map[i]}-', label=f'[{axis[i]}]true')
            ax1.plot(pred_signal[i], f'{color_map[i]}--', label=f'[{axis[i]}]pred')
            ax1.set_title('Acc')
        else:
            ax2.plot(true_signal[i], f'{color_map[i - 3]}-', label=f'[{axis[i]}]true')
            ax2.plot(pred_signal[i], f'{color_map[i - 3]}--', label=f'[{axis[i]}]pred')
            ax2.set_title('Gyro')
    ax1.legend(loc ="lower left");
    ax2.legend(loc="lower left")
    plt.legend()
    plt.show()
    
def visualize_reconstruction(model, testX:np.array, y_pred: np.array, testy:np.array):
    """Visualize the reconstructed signal.

    Args:
        testX (np.array): training input data.
        y_pred (np.array): reconstruced data.
        testy (np.array): true signal.
    """
    axis = ['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz']
    color_map = ['r', 'g', 'b']
    indices = random.choices(list(range(len(testX))), k = 10)
    y_pred = model.predict(testX)

    for idx in indices:
        plot_instance(testX[idx], testy[idx], y_pred[idx], axis, color_map)