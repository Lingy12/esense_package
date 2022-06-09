"""
evaluate_tool.py
===============================================
This is the module for evaluate the model result
"""

from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,accuracy_score
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt 


def get_confusionmatrix(y_pred, y_true, classes_list, title, 
                       save_fig = False, save_path = ""):
    if user == -1:
        user = ""
    cm = confusion_matrix( np.argmax(y_true,axis=1), np.argmax(y_pred,axis=1))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=classes_list)
    disp.plot(xticks_rotation='vertical')
    plt.title(title)
    
    if save_fig:
        plt.savefig(save_path)

def get_derived_mucosal(y_pred, y_true, title, 
                       save_fig = False, save_path = ""):
    y_pred_mucosal = [0 if np.argmax(y)<5 else 1 for y in y_pred]
    y_train_mucosal = [0 if np.argmax(y)<5 else 1 for y in y_true]
    cm = confusion_matrix(y_train_mucosal, y_pred_mucosal)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=['Mucous','Non-Mucous'])
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=['Mucous','Non-Mucous'])
    disp.plot(xticks_rotation='vertical')
    plt.title(title)
    
    if save_fig:
        plt.savefig(save_path)

def get_classification_report(y_pred, y_true):
    return classification_report(y_pred, y_true)
    
def visualize_reconstruction(trainX, y_pred, target):
    for j in range(50):
        data_id = j
        index_pred = [i for i in range(len(trainX[data_id]),len(trainX[data_id])+len(y_pred[data_id]))]
        plt.plot(trainX[data_id,:,3],'-.b',label="input data (x_pre-touching)")
        plt.plot(index_pred,target[data_id],'-.g',label="following data (x_touching)")
        plt.plot(index_pred,y_pred[data_id],'-.r',label="predicted data (x_prediction)")
        plt.legend()
        plt.show()