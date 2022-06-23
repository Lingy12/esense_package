"""
models.py
=================================
This is the module for models usage and training pipeline
"""
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import BatchNormalization, Activation, Conv1DTranspose,concatenate, add
from tensorflow.keras.regularizers import L1, L2, L1L2
from .modules import Conv1DBlock, UNet
        
class ClassificationModel(tf.keras.Model):
    def __init__(self, filters_num:int, kernel_size:int, input_shape:int, output_size:int, 
                                        feature_num: int = 100, dropout_rate:int = 0.5, pool_size:int=2, 
                                        deploy_regularization:bool=False, regularize_ratio = 0):
        super().__init__()
        self.layer = Conv1DBlock(filters_num, kernel_size, input_shape, output_size, 
                                 feature_num, dropout_rate, pool_size, regularize_ratio, 'ConvLayer')
    
    def call(self, inputs):
        return self.layer(inputs)
    
class UNetModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.layer = UNet()
    
    def call(self, inputs):
        return self.layer(inputs)