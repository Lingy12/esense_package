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
    
def visualize_reconstruction(trainX:np.array, y_pred: np.array, target:np.array):
    """Visualize the reconstructed signal.

    Args:
        trainX (np.array): training input data.
        y_pred (np.array): reconstruced data.
        target (np.array): true signal.
    """
    for j in range(50):
        data_id = j
        index_pred = [i for i in range(len(trainX[data_id]),len(trainX[data_id])+len(y_pred[data_id]))]
        plt.plot(trainX[data_id,:,3],'-.b',label="input data (x_pre-touching)")
        plt.plot(index_pred,target[data_id],'-.g',label="following data (x_touching)")
        plt.plot(index_pred,y_pred[data_id],'-.r',label="predicted data (x_prediction)")
        plt.legend()
        plt.show()