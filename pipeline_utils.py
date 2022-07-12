import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from esense_package.data_tool import Dataset, DataGenerator
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from esense_package.visualize import plot_instances
from esense_package.train_helper import TrainHelper
from esense_package.models import ClassificationModel, UNetForcastingModel
from tensorflow.keras.utils import to_categorical
from esense_package.evaluate_tool import get_confusionmatrix, get_derived_mucosal, get_classification_report, derive_mucosal
from esense_package.data_tool import mucous_activity_label_list, non_mucous_activity_label_list
from esense_package.loss import activity_musocal_loss, activity_musocal_loss2, activity_musocal_loss_brian, activity_musocal_loss_yunlong


def create_data_classification(**kwargs):
    touching_label_threshold = kwargs['touching_label_threshold']
    pre_touching_label_threshold = kwargs['pre_touching_label_threshold']
    skip_length = kwargs['skip_length']
    data_length = kwargs['data_length']
    step_size = kwargs['step_size']
    window_num = kwargs['window_num']
    data_following_length = kwargs['data_following_length']
    label_pattern = kwargs['label_pattern']
    imu_file_name = kwargs['imu_file_name']
    label_file_name = kwargs['label_file_name']
    source = kwargs['source']
    session = kwargs['session']
    test_ratio = kwargs['test_ratio']
    
    dataset = Dataset(imu_file_name = imu_file_name, label_file_name = label_file_name)
    dataset.filter_source_session(source=source, session=session)
    dataset.deploy_train_test_split(test_ratio)
    train_df = dataset.get_train_df()
    test_df = dataset.get_test_df()
    train_dg = DataGenerator(train_df, touching_label_threshold=touching_label_threshold, pre_touching_label_threshold=pre_touching_label_threshold, skip_length=skip_length)
    test_dg = DataGenerator(test_df, touching_label_threshold=touching_label_threshold, pre_touching_label_threshold=pre_touching_label_threshold, skip_length=skip_length)

    train_dg.reset()
    test_dg.reset()

    train_dg.generate_data(data_length = data_length, step_size = step_size, window_num = window_num, 
                            data_following_length = data_following_length, label_pattern=label_pattern)
    test_dg.generate_data(data_length = data_length, step_size = step_size, window_num = window_num, 
                            data_following_length = data_following_length, label_pattern=label_pattern)

    trainX, trainy, trainy_bi = train_dg.get_list_for_classification()
    testX, testy, testy_bi = test_dg.get_list_for_classification()

    print(np.histogram(trainy, bins=kwargs['num_class']))
    print(np.histogram(testy, bins=kwargs['num_class']))
    testX, testy = np.array(testX), to_categorical(np.array(testy))
    trainX, trainy= np.array(trainX), to_categorical(np.array(trainy))

    print(f'Training instance: {len(trainX)}')
    print(f'Test instance: {len(testX)}')
    
    return trainX, trainy, testX, testy

def create_data_forcasting(**kwargs):
    touching_label_threshold = kwargs['touching_label_threshold']
    pre_touching_label_threshold = kwargs['pre_touching_label_threshold']
    skip_length = kwargs['skip_length']
    data_length = kwargs['data_length']
    step_size = kwargs['step_size']
    window_num = kwargs['window_num']
    data_following_length = kwargs['data_following_length']
    label_pattern = kwargs['label_pattern']
    imu_file_name = kwargs['imu_file_name']
    label_file_name = kwargs['label_file_name']
    source = kwargs['source']
    session = kwargs['session']
    test_ratio = kwargs['test_ratio']
    
    dataset = Dataset(imu_file_name = imu_file_name, label_file_name = label_file_name)
    dataset.filter_source_session(source=source, session=session)
    dataset.deploy_train_test_split(test_ratio)
    train_df = dataset.get_train_df()
    test_df = dataset.get_test_df()
    train_dg = DataGenerator(train_df, touching_label_threshold=touching_label_threshold, pre_touching_label_threshold=pre_touching_label_threshold, skip_length=skip_length)
    test_dg = DataGenerator(test_df, touching_label_threshold=touching_label_threshold, pre_touching_label_threshold=pre_touching_label_threshold, skip_length=skip_length)

    train_dg.reset()
    test_dg.reset()

    train_dg.generate_data(data_length = data_length, step_size = step_size, window_num = window_num, 
                            data_following_length = data_following_length, label_pattern=label_pattern)
    test_dg.generate_data(data_length = data_length, step_size = step_size, window_num = window_num, 
                            data_following_length = data_following_length, label_pattern=label_pattern)

    trainX, trainy = train_dg.get_list_for_forcasting()
    testX, testy = test_dg.get_list_for_forcasting()

    testX, testy = np.array(testX), np.array(testy)
    trainX, trainy= np.array(trainX), np.array(trainy)

    print(f'Training instance: {len(trainX)}')
    print(f'Test instance: {len(testX)}')
    
    return trainX, trainy, testX, testy

def create_data_forcasting_and_segmentation(**kwargs):
    touching_label_threshold = kwargs['touching_label_threshold']
    pre_touching_label_threshold = kwargs['pre_touching_label_threshold']
    skip_length = kwargs['skip_length']
    data_length = kwargs['data_length']
    step_size = kwargs['step_size']
    window_num = kwargs['window_num']
    data_following_length = kwargs['data_following_length']
    label_pattern = kwargs['label_pattern']
    imu_file_name = kwargs['imu_file_name']
    label_file_name = kwargs['label_file_name']
    source = kwargs['source']
    session = kwargs['session']
    test_ratio = kwargs['test_ratio']
    
    dataset = Dataset(imu_file_name = imu_file_name, label_file_name = label_file_name)
    dataset.filter_source_session(source=source, session=session)
    dataset.deploy_train_test_split(test_ratio)
    train_df = dataset.get_train_df()
    test_df = dataset.get_test_df()
    train_dg = DataGenerator(train_df, touching_label_threshold=touching_label_threshold, pre_touching_label_threshold=pre_touching_label_threshold, skip_length=skip_length)
    test_dg = DataGenerator(test_df, touching_label_threshold=touching_label_threshold, pre_touching_label_threshold=pre_touching_label_threshold, skip_length=skip_length)

    train_dg.reset()
    test_dg.reset()

    train_dg.generate_data(data_length = data_length, step_size = step_size, window_num = window_num, 
                            data_following_length = data_following_length, label_pattern=label_pattern)
    test_dg.generate_data(data_length = data_length, step_size = step_size, window_num = window_num, 
                            data_following_length = data_following_length, label_pattern=label_pattern)

    trainX, train_forcasting, train_seg = train_dg.get_list_for_forcasting_and_segmentation()
    testX, test_forcasting, test_seg = test_dg.get_list_for_forcasting_and_segmentation()

    testX, test_forcasting, test_seg = np.array(testX), np.array(test_forcasting), np.array(test_seg)
    trainX, train_forcasting, train_seg= np.array(trainX), np.array(train_forcasting), np.array(train_seg)

    print(f'Training instance: {len(trainX)}')
    print(f'Test instance: {len(testX)}')
    
    return trainX, train_forcasting, train_seg, testX, test_forcasting, test_seg

def create_data(**kwargs):
    touching_label_threshold = kwargs['touching_label_threshold']
    pre_touching_label_threshold = kwargs['pre_touching_label_threshold']
    skip_length = kwargs['skip_length']
    data_length = kwargs['data_length']
    step_size = kwargs['step_size']
    window_num = kwargs['window_num']
    data_following_length = kwargs['data_following_length']
    label_pattern = kwargs['label_pattern']
    imu_file_name = kwargs['imu_file_name']
    label_file_name = kwargs['label_file_name']
    source = kwargs['source']
    session = kwargs['session']
    test_ratio = kwargs['test_ratio']
    
    dataset = Dataset(imu_file_name = imu_file_name, label_file_name = label_file_name)
    dataset.filter_source_session(source=source, session=session)
    dataset.deploy_train_test_split(test_ratio)
    train_df = dataset.get_train_df()
    test_df = dataset.get_test_df()
    train_dg = DataGenerator(train_df, touching_label_threshold=touching_label_threshold, pre_touching_label_threshold=pre_touching_label_threshold, skip_length=skip_length)
    test_dg = DataGenerator(test_df, touching_label_threshold=touching_label_threshold, pre_touching_label_threshold=pre_touching_label_threshold, skip_length=skip_length)

    train_dg.reset()
    test_dg.reset()

    train_dg.generate_data(data_length = data_length, step_size = step_size, window_num = window_num, 
                            data_following_length = data_following_length, label_pattern=label_pattern)
    test_dg.generate_data(data_length = data_length, step_size = step_size, window_num = window_num, 
                            data_following_length = data_following_length, label_pattern=label_pattern)

    trainX, train_class, train_forcasting, train_seg = train_dg.get_list_for_esense_task()
    testX, test_class, test_forcasting, test_seg = test_dg.get_list_for_esense_task()
    
    print(np.histogram(train_class, bins=kwargs['num_class']))
    print(np.histogram(test_class, bins=kwargs['num_class']))
    
    testX, test_class, test_forcast, test_seg = np.array(testX), to_categorical(np.array(test_class)), np.array(test_forcasting), np.array(test_seg)
    trainX, train_class, train_forcast, train_seg= np.array(trainX), to_categorical(np.array(train_class)), np.array(train_forcasting), np.array(train_seg)

    print(f'Training instance: {len(trainX)}')
    print(f'Test instance: {len(testX)}')
    
    val_size = int(len(trainX) / 10)

    train_ds = tf.data.Dataset.from_tensor_slices((trainX[:-val_size], {'class_out': train_class[:-val_size], 'forcast_out':train_forcast[:-val_size], 'seg_out':train_seg[:-val_size]}))
    val_ds = tf.data.Dataset.from_tensor_slices((trainX[-val_size:], {'class_out': train_class[-val_size:], 'forcast_out':train_forcast[-val_size:], 'seg_out':train_seg[-val_size:]}))
    test_ds = tf.data.Dataset.from_tensor_slices((testX, {'class_out': test_class, 'forcast_out':test_forcast, 'seg_out':test_seg}))

    return train_ds, val_ds, test_ds
    
def train_and_evaludate_classification_model(data, **kwargs):
    trainX, trainy, testX, testy = data
    verbose, epochs, batch_size = kwargs['verbose'], kwargs['epochs'], kwargs['batch_size']
    model_name = kwargs['model_name']
    loss = kwargs['loss']
    optimizer = kwargs['optimizer']
    metrics = kwargs['metrics']
    label_list = kwargs['label_list']
    criteria = kwargs['criteria']
    
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    
    model = ClassificationModel(64, 3, (n_timesteps,n_features), n_outputs, 
                                    feature_num = 100, regularize_ratio=0.01)
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    helper = TrainHelper(model, model_name, log=True)
    hist = helper.train_model(trainX, trainy, criteria=criteria, epochs=epochs, batch_size=batch_size, 
            verbose=verbose, validation_data=(testX, testy))
    model = helper.get_best_model()

    print('Testing:')
    _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=1)
    # print('Testing:')
    # _, train_acc = model.evaluate(trainX, trainy, batch_size=batch_size, verbose=0)
    y_pred = model.predict(testX)

    get_confusionmatrix(y_pred, testy, label_list, 'CM', save_fig=True, save_path=f'plots/CM_{model_name}')
    
    return hist
    
def train_forcasting_model(data, **kwargs):
    trainX, trainy, testX, testy = data
    verbose, epochs, batch_size = kwargs['verbose'], kwargs['epochs'], kwargs['batch_size']
    model_name = kwargs['model_name']
    loss = kwargs['loss']
    optimizer = kwargs['optimizer']
    metrics = kwargs['metrics']
    label_list = kwargs['label_list']
    criteria = kwargs['criteria']
    
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    
    model = UNetForcastingModel()
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    helper = TrainHelper(model, model_name, log=True)
    helper.train_model(trainX, trainy, criteria=criteria, epochs=epochs, batch_size=batch_size, 
            verbose=verbose, validation_data=(testX, testy))
    model = helper.get_best_model()

    _, test_loss = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
    _, train_loss = model.evaluate(trainX, trainy, batch_size=batch_size, verbose=0)
    y_pred = model.predict(testX)

    print(f'Training {criteria}: {train_loss}')
    print(f'Testing {criteria}: {test_loss}')
    return model