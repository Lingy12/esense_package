from scipy.signal import medfilt, butter, lfilter, filtfilt, freqz
import numpy as np

def butter_filter(data, cutoff, fs, freq_cutoff=None, sampling_rate=None, order=3, filter_type='lowpass'):
    normal_cutoff = 2*cutoff/fs
    b, a = butter(order, normal_cutoff, btype=filter_type, analog=False)
    y = filtfilt(b, a, data) # run filter backwards and forwards, since Butterworth filter assumes signals starting from zero; ref: https://dsp.stackexchange.com/a/28198
    return y

def filter_remove_noise_and_gravity(data):
    test_row_butter_filtered = butter_filter(data,15,100)
    test_row_butter_filtered_body = butter_filter(test_row_butter_filtered,0.3,100,filter_type='highpass')
    return test_row_butter_filtered_body

def filter_remove_noise(data):
    test_row_butter_filtered = butter_filter(data,15,100)
    return test_row_butter_filtered

def normalization_minmax(data_list):
    min_score = min(data_list)
    max_score = max(data_list)
    range_score = max_score-min_score
    return [(data-min_score)/range_score for data in data_list]

def normalization_standard(data_list):
    mean_score = np.mean(data_list)
    std_score = np.std(data_list)
    return [(data-mean_score)/std_score for data in data_list]