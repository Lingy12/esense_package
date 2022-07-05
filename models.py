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
                                        regularize_ratio:float = 0):
        """Model for classification

        Args:
            filters_num (int): Filter number for conv layers.
            kernel_size (int): Kernel size for conv layers.
            input_shape (int): Input shape of the data.
            output_size (int): Output size of model.
            feature_num (int, optional): Number of features to be extracted. Defaults to 100.
            dropout_rate (int, optional): Dropout rate for model. Defaults to 0.5.
            pool_size (int, optional): Pooling size for pooling layers. Defaults to 2.
            regularize_ratio (float, optional): Lambda value for model. Defaults to 0.
        """
        super().__init__()
        self.layer = Conv1DBlock(filters_num, kernel_size, input_shape, output_size, 
                                 feature_num, dropout_rate, pool_size, regularize_ratio, 'ConvLayer')
    
    def call(self, inputs):
        return self.layer(inputs)

class ForcastingCNNModel(tf.keras.Model):
        def __init__(self, filters_num:int, kernel_size:int, input_shape:int, forcasting_length:int, 
                                        feature_num: int = 100, dropout_rate:int = 0.5, pool_size:int=2, 
                                        regularize_ratio:float = 0):
            super().__init__()
            self.layer = Conv1DBlock(filters_num, kernel_size, input_shape, forcasting_length * 6, 
                                     feature_num, dropout_rate, pool_size, regularize_ratio, True, 'ConvLayer')
            self.reshape_layer = tf.keras.layers.Reshape((-1, 6))

        def call(self, inputs):
            x = self.layer(inputs)
            
            return self.reshape_layer(x)
    
class UNetSegmentationModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.layer = UNet()
    
    def call(self, inputs):
        return self.layer(inputs)

class UNetForcastingModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.layer = UNet(for_segamentation=False)
    
    def call(self, inputs):
        return self.layer(inputs)