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

class Model:
    """Represent a model for training
    """
    def __init__(self, model_name:str, log: bool=False):
        """Initialize a model with a name.

        Args:
            model_name (str): name of the model.
            log (bool, optional): log or not. Defaults to False.
        """
        # TODO: log not supported
        self.model = None
        self.model_name = model_name
        self.log = log
        self.logging_path = f'/my_checkpoint/{model_name}'
        self.logging_dir = os.path.dirname(self.logging_path)
    
    def init_1d_cnn_model(self, filters_num:int, kernel_size:int, input_shape:int, output_size:int, 
                                        feature_num: int = 100, dropout_rate:int = 0.5, pool_size:int=2):
        """Initialize the model with 1d cnn structure.

        Args:
            filters_num (int): number of filters for each layer.
            kernel_size (int): kernel_size for each layer.
            input_shape (int): shape of the input data.
            output_size (int): size of output unit.
            feature_num (int, optional): number of feature extracted from 1d cnn (second last layer). Defaults to 100.
            dropout_rate (int, optional): dropout rate. Defaults to 0.5.
            pool_size (int, optional): size of the pooling layer. Defaults to 2.
        """
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
        
    def fit(self, x, y, **kwargs):
        """Fit the model using tensorflow fit function.
        """
        assert self.model != None
        if self.log == True:
            cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.logging_dir,
                                                 save_weights_only=True,
                                                 verbose=1)
            kwargs['callbacks'].append(cp_callback)
        
        self.model.fit(x, y, **kwargs)
    
    def evaluate(self, x, y, **kwargs):
        """Evaluate the model using tensorflow evaluate function.
        """
        assert self.model != None
        return self.model.evaluate(x, y, **kwargs)
    
    def compile(self, **kwargs):
        """Compime the initialized model.
        """
        assert self.model != None
        self.model.compile(**kwargs)
    
    def print_model(self):
        """Print out the model structure.
        """
        assert self.model != None
        self.model.summary()
    
    # Add more model
