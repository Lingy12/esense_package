"""
models.py
=================================
This is the module for models usage and training pipeline
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D

class Model:
    def __init__(self, model_name, log=False):
        self.model = None
        self.model_name = model_name
        self.log = log
    
    def fit(self, x, y, **kwargs):
        assert self.model != None
        self.model.fit(x, y, **kwargs)
    
    def evaluate(self, x, y, **kwargs):
        assert self.model != None
        return self.model.evaluate(x, y, **kwargs)
    
    def compile(self, **kwargs):
        assert self.model != None
        self.model.compile(**kwargs)
    
    def print_model(self):
        assert self.model != None
        self.model.summary()
    
    def init_1d_cnn_model(self, filters_num, kernel_size, input_shape, output_size, 
                                        feature_num = 100, dropout_rate = 0.5, pool_size=2):
        model = Sequential()
        model.add(Conv1D(filters=filters_num, kernel_size=kernel_size, activation='relu', 
                        input_shape=input_shape))
        model.add(Conv1D(filters=filters_num, kernel_size=kernel_size, activation='relu'))
        model.add(Dropout(dropout_rate))
        model.add(MaxPooling1D(pool_size=pool_size))
        model.add(Flatten())
        model.add(Dense(feature_num, activation='relu'))
        model.add(Dense(output_size, activation='softmax'))
        self.model = model
    
    # Add more model
