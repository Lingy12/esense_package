import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import BatchNormalization, Activation, Conv1DTranspose,concatenate, add
from tensorflow.keras.regularizers import L1, L2, L1L2

class Conv1DBlock(tf.Module):
    def __init__(self, filters_num:int, kernel_size:int, input_shape:int, output_size:int, 
                                    feature_num: int = 100, dropout_rate:int = 0.5, pool_size:int=2, 
                                    regularize_ratio: float = 0, for_forcasting=False, name=None):
        """Conv1D block for classification

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
        super().__init__(name=name)
        with self.name_scope:
            self.layer1 = Conv1D(filters=filters_num, kernel_size=kernel_size, activation='relu', 
                            input_shape=input_shape, kernel_regularizer=L2(regularize_ratio))
            self.layer2 = Conv1D(filters=filters_num, kernel_size=kernel_size, activation='relu', kernel_regularizer=L2(regularize_ratio))
            self.dropout = Dropout(dropout_rate)
            self.pool_layer = MaxPooling1D(pool_size)
            self.flatten = Flatten()
            self.dense1 = Dense(feature_num, activation='relu')
            self.dense2 = Dense(output_size, activation='softmax')
            self.for_forcasting = for_forcasting
            self.out = Dense(output_size, name=name)
    
    @tf.Module.with_name_scope
    def __call__(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.dropout(x)
        x = self.pool_layer(x)
        x = self.flatten(x)
        x = self.dense1(x)
        if self.for_forcasting:
            x = self.out(x)
        else:
            x = self.dense2(x)
        return x

class Conv1DBlockWithoutOut(tf.Module):
    def __init__(self, filters_num:int, kernel_size:int, input_shape:int,  
                                    feature_num: int = 100, dropout_rate:int = 0.5, pool_size:int=2, 
                                    regularize_ratio: float = 0, name=None):
        """Conv1D block for classification

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
        super().__init__(name=name)
        with self.name_scope:
            self.layer1 = Conv1D(filters=filters_num, kernel_size=kernel_size, activation='relu', 
                            input_shape=input_shape, kernel_regularizer=L2(regularize_ratio))
            self.layer2 = Conv1D(filters=filters_num, kernel_size=kernel_size, activation='relu', kernel_regularizer=L2(regularize_ratio))
            self.dropout = Dropout(dropout_rate)
            self.pool_layer = MaxPooling1D(pool_size)
            self.flatten = Flatten()
            self.dense1 = Dense(feature_num, activation='relu')
    
    @tf.Module.with_name_scope
    def __call__(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.dropout(x)
        x = self.pool_layer(x)
        x = self.flatten(x)
        x = self.dense1(x)
        return x

class UNetConv1DBlock(tf.Module):
    def __init__(self, n_filters, kernel_size = 3, batchnorm = True, name=None):
        """Conv block for Unet

        Args:
            n_filters (_type_): Number of filters.
            kernel_size (int, optional): Kernel size for conv layers. Defaults to 3.
            batchnorm (bool, optional): Using batch normlaization or not. Defaults to True.
            name (_type_, optional): module name. Defaults to None.
        """
        super().__init__(name=name)
        with self.name_scope:
            self.conv_layer1 = Conv1D(filters = n_filters, kernel_size = kernel_size,\
                kernel_initializer = 'he_normal', padding = 'same')
            self.batchnorm = batchnorm
            self.batchnorm_layer1 = BatchNormalization()
            self.activation1 = Activation('relu')
            self.conv_layer2 = Conv1D(filters = n_filters, kernel_size = kernel_size,\
                kernel_initializer = 'he_normal', padding = 'same')
            self.batchnorm_layer2 = BatchNormalization()
            self.activation2 = Activation('relu')
    @tf.Module.with_name_scope
    def __call__(self, x):
        x = self.conv_layer1(x)
        
        if self.batchnorm:
            x = self.batchnorm_layer1(x)
        x = self.activation1(x)
        
        x = self.conv_layer2(x)
        
        if self.batchnorm:
            x = self.batchnorm_layer2(x)
        x = self.activation2(x)
        return x
        
class UNet(tf.Module):
    def __init__(self, out_unit=-1, for_segamentation = True, n_filters = 16, dropout = 0.1, batchnorm = True, name=None):
        """Unet module.

        Args:
            output_unit (int, optional): Unit for output layer. Defaults to -1.
            for_segamentation (bool, optional):The model is for segmentation or not. Defaults to True.
            n_filters (int, optional): Number of filters in conv block. Defaults to 16.
            dropout (float, optional): Dropout rate for model. Defaults to 0.1.
            batchnorm (bool, optional): Use batch normalization or not. Defaults to True.
            name (_type_, optional): Module name. Defaults to None.
        """
        super().__init__(name=name)
        with self.name_scope:
            self.downsample1 = UNetConv1DBlock(n_filters * 1, kernel_size=3, batchnorm=batchnorm)
            self.pool1 = MaxPooling1D(2)
            self.drop1 = Dropout(dropout)
            
            self.downsample2 = UNetConv1DBlock(n_filters * 2, kernel_size=3, batchnorm=batchnorm)
            self.pool2 = MaxPooling1D(2)
            self.drop2 = Dropout(dropout)
            
            self.downsample3 = UNetConv1DBlock(n_filters * 4, kernel_size=3, batchnorm=batchnorm)
            self.pool3 = MaxPooling1D(2)
            self.drop3 = Dropout(dropout)
            
            self.downsample4 = UNetConv1DBlock(n_filters * 8, kernel_size=3, batchnorm=batchnorm)
            self.pool4 = MaxPooling1D(2)
            self.drop4 = Dropout(dropout)     
            
            self.downsample5 = UNetConv1DBlock(n_filters * 16, kernel_size=3, batchnorm=batchnorm)
            
            self.upsampling1 = Conv1DTranspose(n_filters * 8, 3, strides = 2, padding = 'same')
            self.drop5 = Dropout(dropout)
            self.downsample6 = UNetConv1DBlock(n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
            
            self.upsampling2 = Conv1DTranspose(n_filters * 4, 3, strides = 2, padding = 'same')
            self.drop6 = Dropout(dropout)
            self.downsample7 = UNetConv1DBlock(n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
            
            self.upsampling3 = Conv1DTranspose(n_filters * 2, 3, strides = 2, padding = 'same')
            self.drop7 = Dropout(dropout)
            self.downsample8 = UNetConv1DBlock(n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
            
            self.upsampling4 = Conv1DTranspose(n_filters * 1, 3, strides = 2, padding = 'same')
            self.drop8 = Dropout(dropout)
            self.downsample9 = UNetConv1DBlock(n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
            
            self.out_layer = Conv1D(1, 1, activation='sigmoid', name=name)
            
            self.out_layer_forcasting = Conv1D(6, 1, activation='linear', name=name)
            self.for_seg = for_segamentation
            

    @tf.Module.with_name_scope
    def __call__(self, x):
        c1 = self.downsample1(x)
        p1 = self.pool1(c1)
        p1 = self.drop1(p1)
        
        c2 = self.downsample2(p1)
        p2 = self.pool2(c2)
        p2 = self.drop2(p2)
        
        c3 = self.downsample3(p2)
        p3 = self.pool3(c3)
        p3 = self.drop3(p3)
        
        c4 = self.downsample4(p3)
        p4 = self.pool4(c4)
        p4 = self.drop4(p4)
        
        c5 = self.downsample5(p4)
        
        u6 = self.upsampling1(c5)
        u6 = concatenate([u6, c4])
        u6 = self.drop5(u6)
        c6 = self.downsample6(u6)
        
        u7 = self.upsampling2(c6)
        u7 = concatenate([u7, c3])
        u7 = self.drop6(u7)
        c7 = self.downsample7(u7)
        
        u8 = self.upsampling3(c7)
        u8 = concatenate([u8, c2])
        u8 = self.drop7(u8)
        c8 = self.downsample8(u8)
        
        u9 = self.upsampling4(c8)
        u9 = concatenate([u9, c1])
        u9 = self.drop8(u9)
        c9 = self.downsample9(u9)
        
        if self.for_seg:
            outputs = self.out_layer(c9)
        else:
            outputs = self.out_layer_forcasting(c9)
        
        return outputs   