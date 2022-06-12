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
from tensorflow.keras.layers import BatchNormalization, Activation


class Model:
    """Represent a model for training
    """
    def __init__(self, model_name:str, log: bool=False):
        """Initialize a model with a name.

        Args:
            model_name (str): name of the model.
            log (bool, optional): log or not. Defaults to False.
        """
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
    
    def conv1d_block(input_tensor, n_filters, kernel_size = 3, batchnorm = True):
        """Function to add 2 convolutional layers with the parameters passed to it"""
        # first layer
        x = Conv1D(filters = n_filters, kernel_size = kernel_size,\
                kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
        if batchnorm:
            x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        # second layer
        x = Conv1D(filters = n_filters, kernel_size = kernel_size,\
                kernel_initializer = 'he_normal', padding = 'same')(x)
        if batchnorm:
            x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        return x
    
    def get_unet(input_imu, output_unit = -1, for_segamentation = True, n_filters = 16, dropout = 0.1, batchnorm = True):
        """Function to define the UNET Model"""
        # Contracting Path
        c1 = conv1d_block(input_imu, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
        p1 = MaxPooling1D(2)(c1)
        p1 = Dropout(dropout)(p1)
        
        c2 = conv1d_block(p1, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
        p2 = MaxPooling1D(2)(c2)
        p2 = Dropout(dropout)(p2)
        
        c3 = conv1d_block(p2, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
        p3 = MaxPooling1D(2)(c3)
        p3 = Dropout(dropout)(p3)
        
        c4 = conv1d_block(p3, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
        p4 = MaxPooling1D(2)(c4)
        p4 = Dropout(dropout)(p4)
        
        c5 = conv1d_block(p4, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm)
        
        # Expansive Path
        u6 = Conv1DTranspose(n_filters * 8, 3, strides = 2, padding = 'same')(c5)
        u6 = concatenate([u6, c4])
        u6 = Dropout(dropout)(u6)
        c6 = conv1d_block(u6, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
        
        u7 = Conv1DTranspose(n_filters * 4, 3, strides = 2, padding = 'same')(c6)
        u7 = concatenate([u7, c3])
        u7 = Dropout(dropout)(u7)
        c7 = conv1d_block(u7, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
        
        u8 = Conv1DTranspose(n_filters * 2, 3, strides = 2, padding = 'same')(c7)
        u8 = concatenate([u8, c2])
        u8 = Dropout(dropout)(u8)
        c8 = conv1d_block(u8, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
        
        u9 = Conv1DTranspose(n_filters * 1, 3, strides = 2, padding = 'same')(c8)
        u9 = concatenate([u9, c1])
        u9 = Dropout(dropout)(u9)
        c9 = conv1d_block(u9, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
        
        outputs = Conv1D(1, 1, activation='sigmoid')(c9)
        
        if not for_segamentation:
            outputs = Dense(output_unit)
        model = Model(inputs=[input_imu], outputs=[outputs])
        
    #     model = Model(inputs=[input_imu], outputs=[c5])
        
        return model
    
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
